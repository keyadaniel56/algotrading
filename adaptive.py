"""
Deriv Adaptive AI Trading Bot  v10.0
=====================================
Fixes from live log analysis of v9.0:

FIX 1 — streak:0×85 on 60-item deque (phantom streak)
  _streak_count is now recomputed by scanning the actual deque tail on every
  push, so it can never exceed len(history). The counter was previously
  incrementing unboundedly as old digits dropped off silently.

FIX 2 — Streak chi2_strength=99.0 inflated DIGITDIFF scores
  Streak signals now use the real chi2 of the current distribution instead of
  an arbitrary 99.0. A streak is only emitted when p < DIGIT_CHI2_PVALUE
  confirms the distribution is genuinely non-uniform.

FIX 3 — DIGITDIFF dominated all 4 quota slots with barrier=0
  (a) DIGITDIFF barrier logic corrected: barrier is now the RAREST digit
      (low freq → win_prob ≈ 1 - rare_freq ≈ 90%+), not the dominant one
      (dominant freq → win_prob = 1 - 0.85 = 15% — a losing trade).
  (b) QuotaManager anti-repeat: the same DIGITDIFF barrier cannot fill
      consecutive picks in the same wildcard slot, forcing barrier variety.

FIX 4 — Duration always 5t
  With corrected DIGITDIFF logic the win_prob for a rare-barrier DIFF trade
  is now 90%+ → dur_add=0, state_adj at most +1 → duration 1-2t typically.

FIX 5 — Scoring sigmoid recentred
  chi2 sigmoid now centred at 12 (was 15) so moderate chi2 signals
  (p ≈ 0.10) get meaningful weight rather than being near-zero.

Install:
  pip install websocket-client numpy scikit-learn scipy python-dotenv

Setup (.env):
  DERIV_APP_ID=your_app_id
  DERIV_API_TOKEN=your_token
  BASE_STAKE=1.0
  MAX_DAILY_LOSS=20.0
  TAKE_PROFIT=50.0
"""

import os, json, time, threading, logging
import numpy as np
from scipy import stats as sp_stats
from datetime import date, datetime
from collections import deque
from enum import Enum, auto
from typing import Optional
from pathlib import Path

import websocket
from sklearn.cluster import KMeans
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

APP_ID         = os.getenv("DERIV_APP_ID", "1089")
API_TOKEN      = os.getenv("DERIV_API_TOKEN", "")
BASE_STAKE     = float(os.getenv("BASE_STAKE", "1.0"))
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "20.0"))
TAKE_PROFIT    = float(os.getenv("TAKE_PROFIT",    "2.0"))
STATE_FILE     = os.getenv("STATE_FILE", "bot_state.json")

DERIV_WS = f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}"

ALL_SYMBOLS = [
    "R_10", "R_25", "R_50", "R_75", "R_100",
    "1HZ10V", "1HZ25V", "1HZ50V",
]

# ── HMM ──────────────────────────────────────────────────────────────────────
N_STATES = 3

MIN_SAMPLES_FOR_TRAINING  = 60
VARIANCE_STABILITY_EPS    = 0.15
KL_DRIFT_THRESHOLD        = 0.15
STATE_DOMINANCE_THRESHOLD = 0.70

# Digit analysis
DIGIT_CHI2_PVALUE        = 0.15
DIFF_FREQ_THRESHOLD      = 0.28
STREAK_THRESHOLD         = 5
DIGIT_HISTORY_MAX        = 60
MIN_DEFICIT_FOR_OVER_UNDER = 0.03

# ── Trade Quota Table ─────────────────────────────────────────────────────────
# Each entry: (contract_type, barrier, quota_count)
# The bot will execute exactly this many trades of each type per cycle,
# subject to signal availability. Priority: top scorers fill slots first.
QUOTA_TABLE: list[tuple[str, int, int]] = [
    ("DIGITOVER",  0, 4),   # OVER 0  — 4 trades
    ("DIGITUNDER", 9, 4),   # UNDER 9 — 4 trades
    ("DIGITDIFF",  -1, 4),  # DIGIT DIFFERS (any dominant digit) — 4 trades
    ("DIGITOVER",  1, 2),   # OVER 1  — 2 trades
    ("DIGITUNDER", 8, 2),   # UNDER 8 — 2 trades
]
# DIGITDIFF barrier=-1 is a sentinel meaning "pick the best DIGITDIFF signal"
TOTAL_QUOTA = sum(q for _, _, q in QUOTA_TABLE)

# Recovery trade type and barriers (forced after any loss)
RECOVERY_OVER_BARRIER  = 5   # OVER  5
RECOVERY_UNDER_BARRIER = 4   # UNDER 4

# ── Wait / re-analyse ─────────────────────────────────────────────────────────
MIN_WAIT_DIGITS        = 15  # digits collected before re-analyse in WAITING
DIGIT_CHANGE_THRESHOLD = 0.10

# ── Recovery thresholds ───────────────────────────────────────────────────────
RECOVERY_L1_LOSSES = 1   # 1 loss  → mid-barrier next trade
RECOVERY_L2_LOSSES = 3   # 3 losses → also switch symbol
RECOVERY_L3_LOSSES = 5   # 5 losses → hard cooldown

# ── Symbol scoring ────────────────────────────────────────────────────────────
SCORE_WINDOW = 50
SCORE_DECAY  = 0.95

# ── Signal scoring weights ────────────────────────────────────────────────────
SIGNAL_W_DOMINANCE = 0.35
SIGNAL_W_CHI2      = 0.30
SIGNAL_W_STREAK    = 0.20
SIGNAL_W_SYMBOL    = 0.15
SIGNAL_MIN_SCORE   = 0.38   # slightly lower to ensure quota slots can fill

# ── Duration ─────────────────────────────────────────────────────────────────
DURATION_MIN = 1
DURATION_MAX = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("AdaptiveBot")


# ─────────────────────────────────────────────────────────────────────────────
# Duration calculator
# ─────────────────────────────────────────────────────────────────────────────

def compute_duration(contract_type: str, barrier: int,
                     hmm_state: str, digit_freqs: np.ndarray,
                     volatility: float) -> int:
    """
    Tick duration in [1, 5]. Starts at 1 and adds ticks only for weak edges.
      win_prob ≥ 0.90 → 1t,  ≥ 0.80 → 2t,  ≥ 0.70 → 3t,
      ≥ 0.60 → 4t,  < 0.60 → 5t
    High volatility trims 1 tick. Neutral HMM adds 1 tick.
    """
    if contract_type == "DIGITOVER":
        wp = float(digit_freqs[barrier + 1:].sum()) if barrier < 9 else 0.0
    elif contract_type == "DIGITUNDER":
        wp = float(digit_freqs[:barrier].sum()) if barrier > 0 else 0.0
    else:
        b  = max(0, barrier)   # DIGITDIFF; use freq of the digit to avoid
        wp = float(1.0 - digit_freqs[b])

    if   wp >= 0.90: da = 0
    elif wp >= 0.80: da = 1
    elif wp >= 0.70: da = 2
    elif wp >= 0.60: da = 3
    else:            da = 4

    vol_adj   = -1 if volatility > 2e-4 else 0
    state_adj =  1 if hmm_state == "neutral" else 0

    return int(np.clip(1 + da + vol_adj + state_adj, DURATION_MIN, DURATION_MAX))


# ─────────────────────────────────────────────────────────────────────────────
# Phase
# ─────────────────────────────────────────────────────────────────────────────

class Phase(Enum):
    WARMUP   = auto()
    ANALYSE  = auto()
    TRADING  = auto()
    WAITING  = auto()   # used ONLY when no signal passes min score
    RECOVERY = auto()   # L3 cooldown only
    STOPPED  = auto()   # TP or SL reached


# ─────────────────────────────────────────────────────────────────────────────
# Quota Manager  (NEW in v9)
# ─────────────────────────────────────────────────────────────────────────────

class QuotaManager:
    """
    Tracks how many trades of each (type, barrier) slot remain in the
    current cycle. When all slots are empty the cycle resets.

    DIGITDIFF slots use barrier=-1 as a wildcard — any DIGITDIFF signal
    fills that slot regardless of which digit is targeted.

    Anti-repeat rule: within a single DIGITDIFF wildcard slot run, the same
    barrier cannot be selected twice in a row. This prevents one dominant
    digit (e.g. barrier=0 with a massive streak) from filling all 4 DIFF
    slots with identical trades.
    """

    def __init__(self):
        self._reset()

    def _reset(self):
        self._remaining: list[int] = [q for _, _, q in QUOTA_TABLE]
        # Track last-used barrier per quota slot to prevent repeats
        self._last_barrier: dict[int, int] = {}
        log.info(f"[QUOTA] Cycle reset — {TOTAL_QUOTA} trades: "
                 + " | ".join(f"{ctype}{'_'+str(b) if b>=0 else ''}×{q}"
                               for ctype, b, q in QUOTA_TABLE))

    @property
    def cycle_complete(self) -> bool:
        return all(r == 0 for r in self._remaining)

    def remaining_summary(self) -> str:
        parts = []
        for i, (ctype, barrier, _) in enumerate(QUOTA_TABLE):
            if self._remaining[i] > 0:
                label = f"{ctype}{'_'+str(barrier) if barrier>=0 else ''}"
                parts.append(f"{label}:{self._remaining[i]}")
        return " | ".join(parts) if parts else "ALL DONE"

    def best_slot_for(self, candidates: list[dict]) -> Optional[dict]:
        """
        Pick the highest-scoring candidate that fits a non-exhausted quota slot.

        For DIGITDIFF wildcard slots: skip a barrier that was already used
        the previous trade on that slot, encouraging barrier variety.
        """
        for cand in sorted(candidates, key=lambda x: -x["score"]):
            ctype   = cand["type"]
            barrier = cand["barrier"]

            for i, (qt, qb, _) in enumerate(QUOTA_TABLE):
                if self._remaining[i] == 0:
                    continue

                if ctype == "DIGITDIFF" and qt == "DIGITDIFF":
                    # Anti-repeat: skip if this barrier was used last pick
                    if self._last_barrier.get(i) == barrier:
                        continue
                    return {**cand, "_quota_slot": i}

                if ctype == qt and barrier == qb:
                    return {**cand, "_quota_slot": i}

        return None

    def consume(self, slot_index: int, barrier: int = -1):
        if 0 <= slot_index < len(self._remaining):
            self._remaining[slot_index] = max(0, self._remaining[slot_index] - 1)
            self._last_barrier[slot_index] = barrier
            if self.cycle_complete:
                log.info("[QUOTA] All slots consumed — resetting cycle")
                self._reset()

    def reset_now(self):
        self._reset()


# ─────────────────────────────────────────────────────────────────────────────
# Data Readiness
# ─────────────────────────────────────────────────────────────────────────────

class DataReadiness:
    def __init__(self):
        self._ret_history: deque = deque(maxlen=200)
        self._last_train_ret_mean: Optional[float] = None
        self._last_train_ret_std:  Optional[float] = None
        self._last_digit_chi2:     Optional[float] = None
        self._var_window: deque = deque(maxlen=20)

    def push_return(self, ret: float):
        if np.isfinite(ret):
            self._ret_history.append(ret)
            if len(self._ret_history) >= 10:
                self._var_window.append(np.std(list(self._ret_history)[-20:]))

    def recent_volatility(self) -> float:
        if len(self._ret_history) < 5:
            return 1e-4
        return float(np.std(list(self._ret_history)[-30:]))

    def is_ready_to_train(self) -> tuple[bool, str]:
        n = len(self._ret_history)
        if n < MIN_SAMPLES_FOR_TRAINING:
            return False, f"only {n}/{MIN_SAMPLES_FOR_TRAINING} samples"
        if len(self._var_window) < 10:
            return False, "variance window not yet filled"
        vs = list(self._var_window)
        rc = np.std(vs) / (np.mean(vs) + 1e-10)
        if rc > VARIANCE_STABILITY_EPS:
            return False, f"variance unstable ({rc:.3f})"
        return True, f"ready (n={n}, stability={rc:.4f})"

    def record_train_snapshot(self, returns: np.ndarray):
        if len(returns) > 0:
            self._last_train_ret_mean = float(np.mean(returns))
            self._last_train_ret_std  = float(np.std(returns))

    def should_retrain(self) -> tuple[bool, str]:
        if self._last_train_ret_mean is None:
            return False, "no snapshot"
        if len(self._ret_history) < 20:
            return False, "insufficient data"
        recent  = list(self._ret_history)[-40:]
        cm, cs  = float(np.mean(recent)), float(np.std(recent)) + 1e-10
        rs      = self._last_train_ret_std + 1e-10
        kl      = np.log(rs/cs) + (cs**2 + (cm-self._last_train_ret_mean)**2)/(2*rs**2) - 0.5
        kl      = float(np.clip(kl, 0, None))
        if kl > KL_DRIFT_THRESHOLD:
            return True, f"drift KL={kl:.4f}"
        return False, f"stable KL={kl:.4f}"

    def record_signal_snapshot(self, chi2: float):
        self._last_digit_chi2 = chi2

    def should_reanalyse(self, chi2: float) -> tuple[bool, str]:
        if self._last_digit_chi2 is None:
            return True, "no snapshot"
        ch = abs(chi2 - self._last_digit_chi2) / (self._last_digit_chi2 + 1e-6)
        if ch > DIGIT_CHANGE_THRESHOLD:
            return True, f"digit shift {ch:.3f}"
        return False, f"stable ({ch:.3f})"


# ─────────────────────────────────────────────────────────────────────────────
# Digit Analyser
# ─────────────────────────────────────────────────────────────────────────────

class DigitAnalyser:
    def __init__(self):
        self.history: deque = deque(maxlen=DIGIT_HISTORY_MAX)
        self._streak_digit: Optional[int] = None
        self._streak_count: int = 0

    def _recount_streak(self) -> int:
        """
        Recount consecutive identical digits at the tail of history.
        This prevents the counter exceeding the deque window when old
        digits drop off — the root cause of 'streak:0x85' on a 60-item deque.
        """
        if not self.history:
            return 0
        h   = list(self.history)
        ref = h[-1]
        cnt = 0
        for d in reversed(h):
            if d == ref:
                cnt += 1
            else:
                break
        return cnt

    def push(self, digit: int):
        self.history.append(digit)
        # Update streak digit tracker
        if digit != self._streak_digit:
            self._streak_digit = digit
        # Always recompute from the actual deque tail — never blindly increment
        self._streak_count = self._recount_streak()

    def chi2_stat(self) -> float:
        if len(self.history) < 10:
            return 0.0
        counts = np.bincount(list(self.history), minlength=10).astype(float)
        return float(np.sum((counts - len(self.history)/10.0)**2 / (len(self.history)/10.0)))

    def freqs(self) -> np.ndarray:
        if len(self.history) < 1:
            return np.full(10, 0.1)
        return np.bincount(list(self.history), minlength=10).astype(float) / len(self.history)

    def all_biases(self) -> list[dict]:
        """
        Returns all statistically supported trade candidates.
        Covers DIGITOVER (barriers 0-8), DIGITUNDER (barriers 1-9),
        and DIGITDIFF (any dominant digit).
        """
        candidates: list[dict] = []
        seen: set = set()

        def add(ctype, barrier, chi2s, sk, tag):
            key = (ctype, barrier)
            if key not in seen:
                seen.add(key)
                candidates.append({"type": ctype, "barrier": barrier,
                                   "chi2_strength": chi2s, "streak": sk,
                                   "reason_tag": tag})

        sk, sd = self._streak_count, self._streak_digit

        if len(self.history) < 10:
            # Not enough data for chi2 — only emit streak signal if very strong
            if sk >= STREAK_THRESHOLD and sd is not None:
                # Use a modest strength since we can't compute chi2 yet
                candidates.append({"type": "DIGITDIFF", "barrier": sd,
                                   "chi2_strength": 20.0, "streak": sk,
                                   "reason_tag": f"streak:{sd}×{sk}"})
            return candidates

        digits = list(self.history)
        n      = len(digits)
        counts = np.bincount(digits, minlength=10).astype(float)
        fq     = counts / n
        chi2, p = sp_stats.chisquare(counts, np.full(10, n / 10.0))

        # 1. Streak → DIGITDIFF  (uses real chi2, not inflated 99.0)
        #    A streak is only a valid signal if the streaking digit actually
        #    appears at above-random frequency in the window AND chi2 confirms
        #    the distribution is non-uniform. This prevents a digit that
        #    appeared 60× in a row from a stale deque masking everything else.
        if sk >= STREAK_THRESHOLD and sd is not None and p < DIGIT_CHI2_PVALUE:
            add("DIGITDIFF", sd, chi2, sk, f"streak:{sd}×{sk}")

        # 2. DIGITDIFF — barrier should be the RAREST digit, not the dominant one.
        #    DIGITDIFF wins when last digit ≠ barrier.
        #    If barrier is the rarest digit (low freq), then most ticks ≠ barrier → high win prob.
        #    If barrier is the dominant digit (high freq), most ticks = barrier → low win prob (bad).
        #
        #    We emit DIGITDIFF for any digit whose frequency is BELOW the rare threshold,
        #    making it a strong "avoid this digit" signal.
        #    We also emit DIGITDIFF for streak-identified digits (already handled above).
        for d in range(10):
            if fq[d] <= (0.10 - MIN_DEFICIT_FOR_OVER_UNDER) and p < DIGIT_CHI2_PVALUE:
                # Rare digit d → DIFF barrier=d wins ~(1-fq[d]) ≈ 93%+ of the time
                strength = chi2 * (1.0 - fq[d] / 0.10)
                add("DIGITDIFF", d, strength, sk if sd == d else 0,
                    f"diff_rare:{d}(f={fq[d]:.2%})")

        if p < DIGIT_CHI2_PVALUE:
            # 3. DIGITOVER — digit b rare → last digit likely > b
            for b in range(9):
                if fq[b] <= (0.10 - MIN_DEFICIT_FOR_OVER_UNDER):
                    strength = chi2 * (1.0 - fq[b] / 0.10)
                    add("DIGITOVER", b, strength, 0,
                        f"over{b}(f={fq[b]:.2%})")

            # 4. DIGITUNDER — digit b rare → last digit likely < b
            for b in range(1, 10):
                if fq[b] <= (0.10 - MIN_DEFICIT_FOR_OVER_UNDER):
                    strength = chi2 * (1.0 - fq[b] / 0.10)
                    add("DIGITUNDER", b, strength, 0,
                        f"under{b}(f={fq[b]:.2%})")

        return candidates

    def mid_bias(self) -> Optional[str]:
        if len(self.history) < 10:
            return None
        lf = sum(1 for d in self.history if d <= 4) / len(self.history)
        return "low" if lf >= 0.60 else ("high" if lf <= 0.40 else None)

    def streak(self) -> int:
        return self._streak_count

    def stats(self) -> str:
        if len(self.history) < 5:
            return "n/a"
        counts = np.bincount(list(self.history), minlength=10)
        parts  = [f"{d}:{counts[d]}" for d in range(10) if counts[d] > 0]
        return f"[{' '.join(parts)}] chi2={self.chi2_stat():.1f}"


# ─────────────────────────────────────────────────────────────────────────────
# Signal Scorer
# ─────────────────────────────────────────────────────────────────────────────

def score_candidate(dominance: float, chi2_strength: float,
                    streak: int, symbol_score: float,
                    contract_type: str, hmm_state: str) -> float:
    """
    Composite score in [0, 1].

    chi2_norm uses a sigmoid centred at chi2=12 (df=9, p≈0.21 threshold).
    streak_norm is capped at 1.0 (streak >= STREAK_THRESHOLD = full credit).

    Direction alignment:
      DIGITOVER  + bull  → +0.06 bonus
      DIGITOVER  + bear  → penalty scaling with dominance
      DIGITUNDER + bear  → +0.06 bonus
      DIGITUNDER + bull  → penalty scaling with dominance
      DIGITDIFF          → no direction modifier (direction-agnostic)

    IMPORTANT: DIGITDIFF does NOT get a bonus for being direction-agnostic —
    it competes on the same footing as OVER/UNDER. This prevents the scorer
    from always preferring DIGITDIFF and starving the quota slots.
    """
    chi2_norm   = 1.0 / (1.0 + np.exp(-0.20 * (chi2_strength - 12.0)))
    streak_norm = min(streak / max(STREAK_THRESHOLD, 1), 1.0)

    raw = (SIGNAL_W_DOMINANCE * dominance
           + SIGNAL_W_CHI2    * chi2_norm
           + SIGNAL_W_STREAK  * streak_norm
           + SIGNAL_W_SYMBOL  * symbol_score)

    if   contract_type == "DIGITOVER":
        if   hmm_state == "bull": raw += 0.06
        elif hmm_state == "bear": raw -= (0.04 + 0.12 * dominance)
    elif contract_type == "DIGITUNDER":
        if   hmm_state == "bear": raw += 0.06
        elif hmm_state == "bull": raw -= (0.04 + 0.12 * dominance)
    # DIGITDIFF: no modifier

    return float(np.clip(raw, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Signal Generator
# ─────────────────────────────────────────────────────────────────────────────

class SignalGenerator:
    def __init__(self):
        self._path_buf: deque = deque(maxlen=20)

    def record_state(self, state: str):
        self._path_buf.append(state)

    def _dominant_state(self) -> tuple[Optional[str], float]:
        if len(self._path_buf) < 5:
            return None, 0.0
        path = list(self._path_buf)
        for label in ["bear", "neutral", "bull"]:
            frac = path.count(label) / len(path)
            if frac >= STATE_DOMINANCE_THRESHOLD:
                return label, frac
        return None, 0.0

    def reset(self):
        self._path_buf.clear()

    def evaluate_all(self, state: str, confidence: float,
                     digit_biases: list[dict],
                     symbol: str, symbol_score: float) -> list[dict]:
        dominant, dominance = self._dominant_state()
        if dominant is None:
            return []
        results = []
        for b in digit_biases:
            sc = score_candidate(dominance, b["chi2_strength"], b["streak"],
                                 symbol_score, b["type"], dominant)
            if sc < SIGNAL_MIN_SCORE:
                continue
            results.append({
                "type":      b["type"],
                "barrier":   b["barrier"],
                "symbol":    symbol,
                "hmm_state": dominant,
                "dominance": dominance,
                "score":     sc,
                "chi2s":     b["chi2_strength"],
                "reason": (f"{symbol} HMM={dominant}({dominance:.0%}) "
                           f"conf={confidence:.0%} {b['reason_tag']} "
                           f"→ {b['type']} {b['barrier']} score={sc:.3f}"),
            })
        results.sort(key=lambda x: -x["score"])
        return results

    def recovery_signal(self, symbol: str, symbol_score: float) -> Optional[dict]:
        """Returns an OVER 5 or UNDER 4 signal based on current HMM state."""
        dominant, dominance = self._dominant_state()
        state = dominant or "neutral"
        # Choose OVER 5 for bull/neutral, UNDER 4 for bear
        if state in ("bull", "neutral"):
            ctype, barrier = "DIGITOVER", RECOVERY_OVER_BARRIER
        else:
            ctype, barrier = "DIGITUNDER", RECOVERY_UNDER_BARRIER
        sc = score_candidate(dominance, 0, 0, symbol_score, ctype, state)
        return {
            "type": ctype, "barrier": barrier,
            "symbol": symbol, "hmm_state": state,
            "dominance": dominance, "score": sc, "chi2s": 0.0,
            "reason": f"[RECOVERY] {symbol} {state} → {ctype} {barrier}",
            "_is_recovery": True,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Symbol Model
# ─────────────────────────────────────────────────────────────────────────────

class SymbolModel:
    def __init__(self, symbol: str):
        self.symbol       = symbol
        self.ticks: deque = deque(maxlen=600)
        self.trained      = False
        self._n_trains    = 0
        self._means = self._vars = self._trans = self._start = None
        self.state_labels = ["bear", "neutral", "bull"]
        self._digit_analyser = DigitAnalyser()
        self._signal_gen     = SignalGenerator()
        self._readiness      = DataReadiness()

    @staticmethod
    def _last_digit(price: float) -> int:
        return int(f"{price:.4f}".replace(".", "")[-1])

    def _features(self, ticks: list) -> np.ndarray:
        arr    = np.array(ticks, dtype=np.float64)
        rets   = np.diff(arr) / (arr[:-1] + 1e-10)
        digits = np.array([self._last_digit(t) for t in ticks[1:]], dtype=np.float64)
        roll5  = np.convolve(digits, np.ones(5)/5, mode="same") / 9.0
        feat   = np.column_stack([rets, digits/9.0, np.abs(rets), roll5])
        return feat[np.isfinite(feat).all(axis=1)]

    def train(self, force: bool = False) -> bool:
        feat = self._features(list(self.ticks))
        if len(feat) < MIN_SAMPLES_FOR_TRAINING:
            return False
        self._n_trains += 1
        try:
            km     = KMeans(n_clusters=N_STATES, n_init=10,
                            random_state=42 + self._n_trains)
            labels = km.fit_predict(feat)
        except Exception as e:
            log.warning(f"[{self.symbol}] KMeans failed: {e}")
            return False

        order = np.argsort([feat[labels==k, 0].mean() for k in range(N_STATES)])
        remap = np.empty(N_STATES, dtype=int)
        for si, rk in enumerate(order):
            remap[rk] = si
        sl = remap[labels]

        means = np.zeros((N_STATES, feat.shape[1]))
        varrs = np.ones((N_STATES, feat.shape[1])) * 1e-4
        for s in range(N_STATES):
            m = feat[sl == s]
            if len(m) < 2:
                return False
            means[s] = m.mean(axis=0)
            varrs[s] = np.clip(m.var(axis=0), 1e-6, None)

        tc = np.ones((N_STATES, N_STATES))
        for t in range(len(sl)-1):
            tc[sl[t], sl[t+1]] += 1
        tm = tc / tc.sum(axis=1, keepdims=True)
        sp = np.bincount(sl, minlength=N_STATES).astype(float) + 1
        sp /= sp.sum()

        self._means  = means
        self._vars   = varrs
        self._trans  = np.log(tm + 1e-300)
        self._start  = np.log(sp + 1e-300)
        self.trained = True
        self._readiness.record_train_snapshot(feat[:, 0])

        log.info(f"[{self.symbol}] HMM #{self._n_trains} | n={len(feat)} | "
                 f"ret={means[:,0].round(7)} | sizes={[(sl==s).sum() for s in range(N_STATES)]}")
        return True

    def online_update(self, profit: float):
        if not self.trained or self._means is None:
            return
        lr = 0.02
        if profit >= 0:
            feat = self._features(list(self.ticks)[-15:])
            if len(feat) > 0:
                self._means += lr * (feat.mean(axis=0) - self._means)
        else:
            self._vars = np.clip(self._vars * (1 + lr), 1e-6, None)

    def _log_emit(self, obs: np.ndarray) -> np.ndarray:
        lp = np.zeros(N_STATES)
        for s in range(N_STATES):
            d = obs - self._means[s]
            lp[s] = -0.5 * np.sum(d**2/self._vars[s] + np.log(2*np.pi*self._vars[s]))
        return lp

    def analyse(self) -> tuple[Optional[str], float, list]:
        if not self.trained:
            return None, 0.0, []
        feat = self._features(list(self.ticks)[-200:])
        if len(feat) < 5:
            return None, 0.0, []
        try:
            T   = len(feat)
            vit = np.full((T, N_STATES), -np.inf)
            bp  = np.zeros((T, N_STATES), dtype=int)
            vit[0] = self._start + self._log_emit(feat[0])
            for t in range(1, T):
                em = self._log_emit(feat[t])
                for s in range(N_STATES):
                    tp      = vit[t-1] + self._trans[:, s]
                    bp[t,s] = int(np.argmax(tp))
                    vit[t,s]= tp[bp[t,s]] + em[s]
            path     = np.zeros(T, dtype=int)
            path[-1] = int(np.argmax(vit[-1]))
            for t in range(T-2, -1, -1):
                path[t] = bp[t+1, path[t+1]]
            sc   = vit[-1]; sc -= sc.max()
            exps = np.exp(sc)
            conf = float(exps[path[-1]] / exps.sum())
            return (self.state_labels[path[-1]], conf,
                    [self.state_labels[p] for p in path[-20:]])
        except Exception as e:
            log.debug(f"[{self.symbol}] viterbi error: {e}")
            return None, 0.0, []

    def push(self, price: float):
        self.ticks.append(price)
        self._digit_analyser.push(self._last_digit(price))
        if len(self.ticks) >= 2:
            tl  = list(self.ticks)
            ret = (tl[-1] - tl[-2]) / (tl[-2] + 1e-10)
            self._readiness.push_return(ret)

    def get_all_signals(self, symbol_score: float) -> list[dict]:
        state, conf, path = self.analyse()
        if state is None:
            return []
        for s in path:
            self._signal_gen.record_state(s)
        sigs = self._signal_gen.evaluate_all(
            state, conf,
            self._digit_analyser.all_biases(),
            self.symbol, symbol_score,
        )
        fq  = self._digit_analyser.freqs()
        vol = self._readiness.recent_volatility()
        for sig in sigs:
            sig["_digit_freqs"] = fq
            sig["_volatility"]  = vol
        return sigs

    def get_recovery_signal(self, symbol_score: float) -> Optional[dict]:
        """Returns a forced OVER 5 / UNDER 4 recovery signal."""
        state, conf, path = self.analyse()
        if path:
            for s in path:
                self._signal_gen.record_state(s)
        sig = self._signal_gen.recovery_signal(self.symbol, symbol_score)
        if sig:
            fq  = self._digit_analyser.freqs()
            vol = self._readiness.recent_volatility()
            sig["_digit_freqs"] = fq
            sig["_volatility"]  = vol
        return sig

    def readiness_status(self) -> str:
        r, r1 = self._readiness.is_ready_to_train()
        d, d1 = self._readiness.should_retrain()
        return f"ready={r}({r1}) drift={d}({d1})"

    def digit_stats(self) -> str:
        return self._digit_analyser.stats()


# ─────────────────────────────────────────────────────────────────────────────
# Symbol Scorer
# ─────────────────────────────────────────────────────────────────────────────

class SymbolScorer:
    def __init__(self):
        self._history = {s: deque(maxlen=SCORE_WINDOW) for s in ALL_SYMBOLS}
        self._scores  = {s: 0.5 for s in ALL_SYMBOLS}

    def record(self, symbol: str, profit: float):
        self._history[symbol].append((profit, time.time()))
        h  = list(self._history[symbol])
        ww = wt = 0.0
        for i, (p, _) in enumerate(h):
            w   = SCORE_DECAY ** (len(h) - 1 - i)
            wt += w
            if p >= 0: ww += w
        self._scores[symbol] = ww / max(wt, 1e-9)

    def best_symbol(self, ready: list) -> Optional[str]:
        return max(ready, key=lambda s: self._scores.get(s, 0.5)) if ready else None

    def score(self, s: str) -> float:
        return self._scores.get(s, 0.5)

    def ranking(self) -> list:
        return sorted(self._scores.items(), key=lambda x: -x[1])

    def to_dict(self) -> dict: return dict(self._scores)
    def from_dict(self, d: dict): self._scores.update(d)


# ─────────────────────────────────────────────────────────────────────────────
# Recovery Engine
# ─────────────────────────────────────────────────────────────────────────────

class RecoveryEngine:
    def __init__(self):
        self.consecutive_losses = 0
        self.recovery_level     = 0
        self._in_cooldown       = False
        self._cooldown_model: Optional[SymbolModel] = None

    @property
    def in_recovery(self) -> bool:
        return self.recovery_level >= RECOVERY_L1_LOSSES

    def record_loss(self):
        self.consecutive_losses += 1
        prev = self.recovery_level
        if self.consecutive_losses >= RECOVERY_L3_LOSSES:
            self.recovery_level = 3
        elif self.consecutive_losses >= RECOVERY_L2_LOSSES:
            self.recovery_level = 2
        elif self.consecutive_losses >= RECOVERY_L1_LOSSES:
            self.recovery_level = 1
        if self.recovery_level > prev:
            log.warning(f"[RECOVERY] Level → {self.recovery_level} "
                        f"({self.consecutive_losses} consecutive losses)")

    def record_win(self):
        self.consecutive_losses = 0
        if self.recovery_level > 0:
            self.recovery_level = max(0, self.recovery_level - 1)
            log.info(f"[RECOVERY] Win — level down to {self.recovery_level}")

    def get_stake_multiplier(self) -> float:
        return 0.5 if self.recovery_level >= 1 else 1.0

    def needs_symbol_switch(self) -> bool:
        return self.recovery_level >= 2

    def needs_hard_cooldown(self) -> bool:
        return self.recovery_level >= 3

    def start_cooldown(self, model: SymbolModel):
        self._in_cooldown    = True
        self._cooldown_model = model
        log.warning(f"[RECOVERY L3] Cooldown — waiting for {model.symbol} to stabilise")

    def check_cooldown_done(self) -> bool:
        if not self._in_cooldown:
            return True
        if self._cooldown_model is None:
            self._in_cooldown = False
            return True
        ready, reason = self._cooldown_model._readiness.is_ready_to_train()
        if ready:
            log.info(f"[RECOVERY] Cooldown done — {reason}")
            self._cooldown_model.train(force=True)
            self._in_cooldown    = False
            self._cooldown_model = None
            return True
        return False

    def status(self) -> str:
        return (f"L{self.recovery_level} losses={self.consecutive_losses} "
                f"cooldown={self._in_cooldown}")


# ─────────────────────────────────────────────────────────────────────────────
# Risk Manager
# ─────────────────────────────────────────────────────────────────────────────

class RiskManager:
    def __init__(self):
        self.base_stake     = BASE_STAKE
        self.max_daily_loss = MAX_DAILY_LOSS
        self.take_profit    = TAKE_PROFIT
        self.daily_pnl      = 0.0
        self.trade_date     = date.today()
        self.total_trades   = 0
        self.total_wins     = 0

    def _day_reset(self):
        if date.today() != self.trade_date:
            log.info(f"New day | prev PnL={self.daily_pnl:+.2f} | "
                     f"W/L={self.total_wins}/{self.total_trades-self.total_wins}")
            self.daily_pnl  = 0.0
            self.trade_date = date.today()

    def is_halted(self) -> tuple[bool, str]:
        self._day_reset()
        if self.daily_pnl <= -abs(self.max_daily_loss):
            return True, f"STOP LOSS hit (pnl={self.daily_pnl:+.2f} ≤ -{self.max_daily_loss})"
        if self.daily_pnl >= abs(self.take_profit):
            return True, f"TAKE PROFIT hit (pnl={self.daily_pnl:+.2f} ≥ +{self.take_profit})"
        return False, ""

    def stake(self, mult: float = 1.0) -> float:
        """Capped at 40% of remaining daily loss budget."""
        remaining = abs(self.max_daily_loss) - abs(min(self.daily_pnl, 0))
        cap       = remaining * 0.40
        raw       = self.base_stake * mult
        return round(max(min(raw, cap), 0.35), 2)

    def record(self, profit: float):
        self._day_reset()
        self.daily_pnl    += profit
        self.total_trades += 1
        if profit >= 0:
            self.total_wins += 1

    def win_rate(self) -> float:
        return self.total_wins / max(1, self.total_trades)

    def summary(self) -> str:
        return (f"trades={self.total_trades} wr={self.win_rate():.0%} "
                f"pnl={self.daily_pnl:+.2f} "
                f"[SL=-{self.max_daily_loss} TP=+{self.take_profit}]")

    def to_dict(self) -> dict:
        return {"daily_pnl": self.daily_pnl, "total_trades": self.total_trades,
                "total_wins": self.total_wins, "trade_date": str(self.trade_date)}

    def from_dict(self, d: dict):
        self.daily_pnl    = d.get("daily_pnl", 0.0)
        self.total_trades = d.get("total_trades", 0)
        self.total_wins   = d.get("total_wins", 0)
        td = date.fromisoformat(d.get("trade_date", str(date.today())))
        self.daily_pnl  = 0.0 if td != date.today() else self.daily_pnl
        self.trade_date = date.today()


# ─────────────────────────────────────────────────────────────────────────────
# State persistence
# ─────────────────────────────────────────────────────────────────────────────

class StatePersistence:
    def __init__(self):
        self.path = Path(STATE_FILE)

    def save(self, risk: RiskManager, scorer: SymbolScorer, log_: list):
        try:
            self.path.write_text(json.dumps({
                "saved_at":  datetime.now().isoformat(),
                "risk":      risk.to_dict(),
                "scores":    scorer.to_dict(),
                "trade_log": log_[-500:],
            }, indent=2))
        except Exception as e:
            log.warning(f"State save failed: {e}")

    def load(self) -> Optional[dict]:
        if not self.path.exists():
            return None
        try:
            return json.loads(self.path.read_text())
        except Exception as e:
            log.warning(f"State load failed: {e}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Main Bot
# ─────────────────────────────────────────────────────────────────────────────

class DerivAdaptiveBot:
    def __init__(self):
        self.ws: Optional[websocket.WebSocketApp] = None
        self._req_id = 1
        self._lock   = threading.Lock()

        self.models   = {s: SymbolModel(s) for s in ALL_SYMBOLS}
        self.scorer   = SymbolScorer()
        self.recovery = RecoveryEngine()
        self.risk     = RiskManager()
        self.persist  = StatePersistence()
        self.quota    = QuotaManager()

        self.phase          = Phase.WARMUP
        self.active_symbol  = ALL_SYMBOLS[0]
        self.current_signal: Optional[dict] = None
        self.open_cid: Optional[str] = None
        self.trade_log: list = []

        # Flags
        self._analysis_pending      = False
        self._pending_recovery_trade = False  # True = next trade must be recovery
        self._wait_digits_collected = 0
        self._last_readiness_log: dict = {}

        data = self.persist.load()
        if data:
            self.risk.from_dict(data.get("risk", {}))
            self.scorer.from_dict(data.get("scores", {}))
            self.trade_log = data.get("trade_log", [])
            log.info(f"State loaded | {self.risk.summary()}")

    # ── WebSocket ─────────────────────────────────────────────────────────────

    def start(self):
        self.ws = websocket.WebSocketApp(
            DERIV_WS,
            on_open    = self._on_open,
            on_message = self._on_message,
            on_error   = self._on_error,
            on_close   = self._on_close,
        )
        log.info("Connecting…")
        self.ws.run_forever(ping_interval=30, ping_timeout=10, reconnect=5)

    def stop(self):
        self.persist.save(self.risk, self.scorer, self.trade_log)
        if self.ws:
            self.ws.close()
        log.info(f"Stopped | {self.risk.summary()}")

    def _send(self, p: dict):
        with self._lock:
            p["req_id"] = self._req_id
            self._req_id += 1
        if self.ws:
            try:
                self.ws.send(json.dumps(p))
            except Exception as e:
                log.warning(f"Send error: {e}")

    def _on_open(self, ws):
        log.info("WebSocket open")
        if API_TOKEN:
            self._send({"authorize": API_TOKEN})
        else:
            log.warning("No token — DEMO mode")
            self._subscribe_all()

    def _on_message(self, ws, raw: str):
        try:
            msg = json.loads(raw)
            t   = msg.get("msg_type")
            if   t == "authorize":              self._on_auth(msg)
            elif t == "tick":                   self._on_tick(msg)
            elif t == "buy":                    self._on_buy(msg)
            elif t == "proposal_open_contract": self._on_poc(msg)
            elif t == "error":
                log.warning(f"API: {msg.get('error',{}).get('message', msg)}")
        except Exception as e:
            log.error(f"Handler error: {e}", exc_info=False)

    def _on_error(self, ws, e): log.error(f"WS error: {e}")
    def _on_close(self, ws, c, r): log.info(f"WS closed {c}: {r}")

    def _on_auth(self, msg: dict):
        if msg.get("error"):
            log.error(f"Auth failed: {msg['error']['message']}")
            self.stop()
            return
        a = msg["authorize"]
        log.info(f"Auth OK | {a['loginid']} | bal={a['balance']} {a['currency']}")
        self._subscribe_all()

    def _subscribe_all(self):
        for sym in ALL_SYMBOLS:
            self._send({"ticks": sym, "subscribe": 1})
        log.info(f"Subscribed to {len(ALL_SYMBOLS)} symbols")

    # ── Tick handler ──────────────────────────────────────────────────────────

    def _on_tick(self, msg: dict):
        if self.phase == Phase.STOPPED:
            return

        td    = msg.get("tick", {})
        sym   = td.get("symbol", "")
        price = float(td.get("quote", 0))
        if sym not in self.models or price <= 0:
            return

        model = self.models[sym]
        model.push(price)

        # Training (all symbols always)
        if not model.trained:
            ready, reason = model._readiness.is_ready_to_train()
            if reason != self._last_readiness_log.get(sym):
                self._last_readiness_log[sym] = reason
                log.info(f"[{sym}] Readiness: {reason}")
            if ready and model.train():
                log.info(f"[{sym}] Model trained")
        else:
            should, reason = model._readiness.should_retrain()
            if should:
                log.info(f"[{sym}] Retraining — {reason}")
                model.train()

        # Dirty flag for analyse phases
        if self.phase in (Phase.WARMUP, Phase.ANALYSE):
            self._analysis_pending = True

        # Wait counter
        if self.phase == Phase.WAITING and sym == self.active_symbol:
            self._wait_digits_collected += 1

        # Recovery cooldown
        if self.phase == Phase.RECOVERY and sym == self.active_symbol:
            if self.recovery.check_cooldown_done():
                self._enter_analyse()
            return

        self._drive_cycle(sym)

    # ── Cycle state machine ───────────────────────────────────────────────────

    def _drive_cycle(self, sym: str):
        if self.phase == Phase.STOPPED or self.phase == Phase.TRADING:
            return

        if self.phase == Phase.WARMUP:
            if any(m.trained for m in self.models.values()):
                log.info("[WARMUP] Model ready — entering ANALYSE")
                self._enter_analyse()
            return

        if self.phase == Phase.WAITING:
            if sym != self.active_symbol:
                return
            if self._wait_digits_collected < MIN_WAIT_DIGITS:
                return
            am   = self.models[self.active_symbol]
            chi2 = am._digit_analyser.chi2_stat()
            ok, reason = am._readiness.should_reanalyse(chi2)
            if ok:
                log.info(f"[WAITING] Re-analyse after {self._wait_digits_collected} digits — {reason}")
                for m in self.models.values():
                    if m.trained:
                        dr, _ = m._readiness.should_retrain()
                        if dr: m.train()
                self._enter_analyse()
            return

        if self.phase == Phase.ANALYSE:
            if self._analysis_pending:
                self._analysis_pending = False
                self._run_global_analysis()
            return

    def _enter_analyse(self):
        self._wait_digits_collected = 0
        self._analysis_pending      = True
        self.phase                  = Phase.ANALYSE
        log.info(f"Phase → ANALYSE | quota=[{self.quota.remaining_summary()}] "
                 f"recovery={self.recovery.status()}")

    def _switch_active_symbol(self, new_sym: str):
        if new_sym != self.active_symbol:
            log.info(f"Symbol switch: {self.active_symbol} → {new_sym}")
            self.active_symbol          = new_sym
            self._wait_digits_collected = 0
            self.models[new_sym]._signal_gen.reset()

    # ── Global analysis — quota-aware ─────────────────────────────────────────

    def _run_global_analysis(self):
        halted, reason = self.risk.is_halted()
        if halted:
            log.warning(f"[RISK] {reason}")
            self._stop_trading(reason)
            return

        trained = [s for s in ALL_SYMBOLS if self.models[s].trained]
        if not trained:
            return

        # ── Recovery trade (forced) ───────────────────────────────────────────
        if self._pending_recovery_trade:
            self._pending_recovery_trade = False
            # Get recovery signal from best-scoring trained symbol
            best_sym = self.scorer.best_symbol(trained) or trained[0]
            model    = self.models[best_sym]
            sig      = model.get_recovery_signal(self.scorer.score(best_sym))
            if sig:
                fq  = sig.pop("_digit_freqs", np.full(10, 0.1))
                vol = sig.pop("_volatility",  1e-4)
                sig["duration"] = compute_duration(
                    sig["type"], sig["barrier"], sig["hmm_state"], fq, vol)
                self._switch_active_symbol(best_sym)
                self.current_signal = sig
                log.info(f"╔ RECOVERY TRADE | {sig['reason']} | dur={sig['duration']}t")
                self.phase = Phase.TRADING
                self._place_trade()
                return

        # ── Normal quota-based selection ──────────────────────────────────────
        all_candidates: list[dict] = []
        for sym in trained:
            cands = self.models[sym].get_all_signals(self.scorer.score(sym))
            all_candidates.extend(cands)

        if not all_candidates:
            log.info("[ANALYSE] No signals — entering WAITING")
            am   = self.models[self.active_symbol]
            am._readiness.record_signal_snapshot(am._digit_analyser.chi2_stat())
            self._wait_digits_collected = 0
            self.phase = Phase.WAITING
            return

        # Find the best candidate that fits a quota slot
        best = self.quota.best_slot_for(all_candidates)

        if best is None:
            # All quota slots satisfied — reset and try again immediately
            log.info("[QUOTA] No matching slot — resetting cycle now")
            self.quota.reset_now()
            best = self.quota.best_slot_for(all_candidates)

        if best is None:
            # Still nothing (edge case: no signal fits any type)
            log.info("[ANALYSE] No quota match — waiting for digit shift")
            self.phase = Phase.WAITING
            return

        slot_idx = best.pop("_quota_slot")

        # Log top candidates
        top_n = min(5, len(all_candidates))
        log.info(f"[ANALYSE] {len(all_candidates)} candidates | quota=[{self.quota.remaining_summary()}]")
        for i, c in enumerate(sorted(all_candidates, key=lambda x:-x["score"])[:top_n]):
            log.info(f"  #{i+1} {c['reason']}")

        # Compute duration
        fq  = best.pop("_digit_freqs", np.full(10, 0.1))
        vol = best.pop("_volatility",  1e-4)
        best["duration"] = compute_duration(
            best["type"], best["barrier"], best["hmm_state"], fq, vol)

        # Snapshot for re-analyse
        chosen_model = self.models[best["symbol"]]
        chosen_model._readiness.record_signal_snapshot(
            chosen_model._digit_analyser.chi2_stat())

        self._switch_active_symbol(best["symbol"])
        self.current_signal = best
        self.quota.consume(slot_idx, barrier=best.get("barrier", -1))

        log.info(f"╔ SIGNAL | {best['reason']} | dur={best['duration']}t "
                 f"| quota=[{self.quota.remaining_summary()}]")
        self.phase = Phase.TRADING
        self._place_trade()

    def _stop_trading(self, reason: str):
        """Called when TP or SL is hit — disconnect cleanly."""
        log.warning(f"╔══ TRADING STOPPED ══╗")
        log.warning(f"  Reason : {reason}")
        log.warning(f"  Final  : {self.risk.summary()}")
        log.warning(f"╚═════════════════════╝")
        self.phase = Phase.STOPPED
        self.persist.save(self.risk, self.scorer, self.trade_log)
        # Disconnect in a thread to avoid deadlock inside WS callback
        threading.Thread(target=self.stop, daemon=True).start()

    # ── Trade execution ───────────────────────────────────────────────────────

    def _place_trade(self):
        sig      = self.current_signal
        stake    = self.risk.stake(self.recovery.get_stake_multiplier())
        ctype    = sig["type"]
        duration = sig.get("duration", 1)
        is_rec   = sig.get("_is_recovery", False)

        label = "[REC] " if is_rec else ""
        log.info(f"├── {label}TRADE | {ctype} {sig['symbol']} "
                 f"barrier={sig.get('barrier','?')} | "
                 f"stake=${stake} dur={duration}t "
                 f"score={sig.get('score',0):.3f}")

        if not API_TOKEN:
            fake_cid = f"DEMO-{id(self)}-{int(time.time()*1000)%99999}"
            self.open_cid = fake_cid
            import random
            if ctype == "DIGITOVER":
                wr = 0.91 if sig.get("barrier", 0) <= 1 else 0.81
            elif ctype == "DIGITUNDER":
                wr = 0.91 if sig.get("barrier", 9) >= 8 else 0.81
            else:
                wr = 0.85
            sim_profit = round(stake * 0.87, 2) if random.random() < wr else -stake
            sim_status = "won" if sim_profit >= 0 else "lost"
            log.info(f"└── [DEMO] #{fake_cid} → {sim_status} ({sim_profit:+.2f})")
            self._on_poc({"proposal_open_contract": {
                "contract_id": fake_cid,
                "status":      sim_status,
                "is_sold":     True,
                "profit":      sim_profit,
            }})
            return

        payload = {
            "buy": 1, "price": stake,
            "parameters": {
                "amount": stake, "basis": "stake",
                "contract_type": ctype, "currency": "USD",
                "duration": duration, "duration_unit": "t",
                "symbol": sig["symbol"],
            },
        }
        if sig.get("barrier") is not None:
            payload["parameters"]["barrier"] = str(sig["barrier"])
        self._send(payload)

    def _on_buy(self, msg: dict):
        if msg.get("error"):
            log.error(f"Buy error: {msg['error']['message']}")
            self._enter_analyse()
            return
        b = msg.get("buy", {})
        self.open_cid = b.get("contract_id")
        log.info(f"└── #{self.open_cid} payout={b.get('payout')} cost={b.get('buy_price')}")
        self._send({"proposal_open_contract": 1,
                    "contract_id": self.open_cid, "subscribe": 1})

    def _on_poc(self, msg: dict):
        poc          = msg.get("proposal_open_contract", {})
        incoming_cid = str(poc.get("contract_id", ""))
        if self.open_cid is None or incoming_cid != str(self.open_cid):
            return
        status   = poc.get("status", "")
        is_final = poc.get("is_sold", False) or status in ("won", "lost")
        if not is_final:
            return
        profit = float(poc.get("profit", 0))
        log.info(f"    #{self.open_cid} → {status.upper()} {profit:+.2f}")
        self.open_cid = None
        self._after_settlement(profit)

    def _after_settlement(self, profit: float):
        sig = self.current_signal
        sym = sig["symbol"] if sig else self.active_symbol

        self.risk.record(profit)
        self.scorer.record(sym, profit)
        self.models[sym].online_update(profit)

        if profit >= 0:
            self.recovery.record_win()
        else:
            self.recovery.record_loss()

        self.trade_log.append({
            "ts":       datetime.now().isoformat(),
            "symbol":   sym,
            "type":     sig["type"]             if sig else "?",
            "barrier":  sig.get("barrier")      if sig else None,
            "dur":      sig.get("duration")     if sig else None,
            "score":    round(sig.get("score",0),4) if sig else 0,
            "recovery": sig.get("_is_recovery", False) if sig else False,
            "profit":   profit,
            "cons_losses": self.recovery.consecutive_losses,
            "daily_pnl":   self.risk.daily_pnl,
        })
        if len(self.trade_log) % 5 == 0:
            self.persist.save(self.risk, self.scorer, self.trade_log)

        log.info(f"    {self.risk.summary()} | "
                 f"sym={self.scorer.score(sym):.0%} | "
                 f"rec={self.recovery.status()}")

        # ── Check TP / SL after every trade ──────────────────────────────────
        halted, reason = self.risk.is_halted()
        if halted:
            self._stop_trading(reason)
            return

        # ── L3: hard cooldown ─────────────────────────────────────────────────
        if self.recovery.needs_hard_cooldown():
            self.phase = Phase.RECOVERY
            alts = [s for s in ALL_SYMBOLS if self.models[s].trained and s != sym]
            new_sym = self.scorer.best_symbol(alts) or self.active_symbol
            self._switch_active_symbol(new_sym)
            self.models[new_sym].train(force=True)
            self.recovery.start_cooldown(self.models[new_sym])
            self.scorer._scores[sym] = 0.3
            return

        # ── L2: symbol switch ─────────────────────────────────────────────────
        if profit < 0 and self.recovery.needs_symbol_switch():
            alts = [s for s in ALL_SYMBOLS if self.models[s].trained and s != sym]
            ns   = self.scorer.best_symbol(alts)
            if ns:
                self._switch_active_symbol(ns)
                log.info(f"[L2] Switched to {ns}")

        # ── On any loss: flag next trade as recovery (OVER 5 / UNDER 4) ──────
        if profit < 0:
            self._pending_recovery_trade = True
            log.info(f"[RECOVERY] Next trade → OVER {RECOVERY_OVER_BARRIER} "
                     f"/ UNDER {RECOVERY_UNDER_BARRIER}")

        # ── Continue trading immediately — no waiting ─────────────────────────
        self._enter_analyse()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def profit_str(p: float) -> str:
    return f"{p:+.2f}"


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║    Deriv Adaptive AI Trading Bot  v10.0                     ║
║    Streak fix · Correct DIFF logic · Balanced quota         ║
╚══════════════════════════════════════════════════════════════╝""")
    print(f"  Symbols    : {', '.join(ALL_SYMBOLS)}")
    print(f"  Base stake : ${BASE_STAKE:.2f}")
    print(f"  Stop loss  : -${MAX_DAILY_LOSS:.2f}   Take profit: +${TAKE_PROFIT:.2f}")
    print(f"  Stake cap  : 40% of remaining daily budget per trade")
    print()
    print("  Trade Quota (per cycle — resets when all filled):")
    for ctype, barrier, quota in QUOTA_TABLE:
        label = f"{ctype}" + (f" barrier={barrier}" if barrier >= 0 else " (any dominant digit)")
        print(f"    {label:<30s} × {quota}")
    print(f"    {'TOTAL':<30s} × {TOTAL_QUOTA}")
    print()
    print("  Recovery (on any loss):")
    print(f"    Next trade → DIGITOVER {RECOVERY_OVER_BARRIER} or DIGITUNDER {RECOVERY_UNDER_BARRIER}")
    print(f"    L2 ({RECOVERY_L2_LOSSES} losses) → also switch symbol")
    print(f"    L3 ({RECOVERY_L3_LOSSES} losses) → data-driven cooldown")
    print()
    print("  Duration: 1-5 ticks based on win probability")
    print(f"  Auth token : {'SET ✓' if API_TOKEN else 'NOT SET — demo mode'}")
    print()

    bot = DerivAdaptiveBot()
    try:
        bot.start()
    except KeyboardInterrupt:
        log.info("Interrupted by user")
        bot.stop()


if __name__ == "__main__":
    main()