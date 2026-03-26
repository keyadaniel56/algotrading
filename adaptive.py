"""
Deriv Adaptive AI Trading Bot  v8.0
=====================================
Changes over v7.0 — from live log analysis:

FIX 1 — Duration too long (was outputting 7t+)
  Duration base reset to 1 tick. Max capped at 5.
  Strong edges (win_prob ≥ 0.90) stay at 1-2 ticks.
  Only weak edges (win_prob < 0.60) reach 5 ticks.
  Preferred range is 1-3 ticks as requested.

FIX 2 — Loss → immediate mid-barrier switch
  RECOVERY_L1_LOSSES = 1  (was 2).
  On the very first loss the bot enters "mid" mode.
  Next trade after ANY loss becomes OVER 5 or UNDER 4,
  not after the second loss as before.
  L2 (symbol switch) at 3 losses, L3 (cooldown) at 5.

FIX 3 — Stake can exceed MAX_DAILY_LOSS
  stake() now caps at 40% of remaining daily loss budget.
  With MAX_DAILY_LOSS=$20 and BASE_STAKE=$30, effective
  stake is $8 on the first trade, shrinking as losses mount.
  The bot can no longer be halted by a single trade.

FIX 4 — Streak tag showed wrong format in logs
  reason_tag for diff_dom path now clearly labelled.
  Streak guard confirmed: _streak_count starts at 0 and
  is set to 1 on first unique digit push — correct.

Install:
  pip install websocket-client numpy scikit-learn scipy python-dotenv

Setup (.env):
  DERIV_APP_ID=your_app_id
  DERIV_API_TOKEN=your_token
  BASE_STAKE=1.0
  MAX_DAILY_LOSS=20.0
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
STATE_FILE     = os.getenv("STATE_FILE", "bot_state.json")

DERIV_WS = f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}"

ALL_SYMBOLS = [
    "R_10", "R_25", "R_50", "R_75", "R_100",
    "1HZ10V", "1HZ25V", "1HZ50V",
]

# ── HMM ──────────────────────────────────────────────────────────────────────
N_STATES = 3

# Readiness thresholds (data-quality based, not tick counts)
MIN_SAMPLES_FOR_TRAINING  = 60
VARIANCE_STABILITY_EPS    = 0.15
KL_DRIFT_THRESHOLD        = 0.15
STATE_DOMINANCE_THRESHOLD = 0.70

# Digit analysis
DIGIT_CHI2_PVALUE   = 0.15   # p < 0.15 = distribution is non-uniform
DIFF_FREQ_THRESHOLD = 0.28   # digit freq >= 28% → strong DIFFERS target
RARE_FREQ_THRESHOLD = 0.06   # digit freq <= 6%  → valid OVER/UNDER edge
STREAK_THRESHOLD    = 5      # consecutive same digit = strong DIFFERS signal
DIGIT_HISTORY_MAX   = 60     # rolling window cap (raised from 50 for better stats)

# Minimum expected frequency deviation to consider a barrier valid
# (how much rarer than 10% a digit must be to justify an OVER/UNDER trade)
# 0.03 means freq ≤ 7% qualifies — deliberately loose so borderline edges
# still get generated and scored, then the composite score filters them.
MIN_DEFICIT_FOR_OVER_UNDER = 0.03   # freq ≤ 10% - 3% = 7% threshold

# Cycle
TRADES_PER_CYCLE       = 2
DIGIT_CHANGE_THRESHOLD = 0.10
MIN_WAIT_DIGITS        = 10   # minimum new digits before re-analyse is considered

# Recovery thresholds
# L1=1 → ANY loss immediately triggers mid-barrier mode (OVER 5 / UNDER 4)
# L2=3 → 3 consecutive losses → also switch symbol
# L3=5 → 5 consecutive losses → hard cooldown
RECOVERY_L1_LOSSES = 1
RECOVERY_L2_LOSSES = 3
RECOVERY_L3_LOSSES = 5

# Symbol scoring
SCORE_WINDOW = 50
SCORE_DECAY  = 0.95

# ── Signal scoring weights ────────────────────────────────────────────────────
SIGNAL_W_DOMINANCE = 0.35
SIGNAL_W_CHI2      = 0.30
SIGNAL_W_STREAK    = 0.20
SIGNAL_W_SYMBOL    = 0.15

# Minimum composite score to place a trade
SIGNAL_MIN_SCORE = 0.42

# ── Duration: prefer 1-3 ticks, only extend when edge is genuinely weak ──────
# Base is 1 tick — the bot should win fast when edge is strong.
# compute_duration() adds ticks only when win probability is low.
DURATION_BASE = {
    "DIGITOVER":  1,
    "DIGITUNDER": 1,
    "DIGITDIFF":  1,
}
DURATION_MIN = 1
DURATION_MAX = 5   # hard cap — never more than 5 ticks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("AdaptiveBot")


# ─────────────────────────────────────────────────────────────────────────────
# Duration calculator  (replaces hardcoded CONTRACT_DURATION_MAP)
# ─────────────────────────────────────────────────────────────────────────────

def compute_duration(contract_type: str, barrier: int,
                     hmm_state: str, digit_freqs: np.ndarray,
                     volatility: float) -> int:
    """
    Returns tick duration in [DURATION_MIN, DURATION_MAX] = [1, 5].

    Philosophy: start at 1 tick (fastest) and add ticks only when the
    win probability is low enough that the edge needs more time to play out.

      OVER  barrier b: win if last digit > b  → win_prob = sum(freq[b+1..9])
      UNDER barrier b: win if last digit < b  → win_prob = sum(freq[0..b-1])
      DIFF  barrier b: win if last digit ≠ b  → win_prob = 1 - freq[b]

    win_prob mapping to duration additions:
      ≥ 0.90 → +0  (very high edge, 1 tick is enough)
      ≥ 0.80 → +1  (good edge, 2 ticks)
      ≥ 0.70 → +2  (moderate edge, 3 ticks)
      ≥ 0.60 → +3  (weaker edge, 4 ticks)
      <  0.60 → +4  (low edge, 5 ticks to let it develop)

    Volatility:
      High vol (digits shift fast) → subtract 1 tick (edge resolves faster)
      Low  vol (digits sticky)     → no adjustment needed

    HMM state:
      neutral → +1 tick (less conviction about direction)
      bull/bear → no adjustment
    """
    base = DURATION_BASE.get(contract_type, 1)

    # Estimate win probability from live digit frequencies
    if contract_type == "DIGITOVER":
        win_prob = float(digit_freqs[barrier + 1:].sum()) if barrier < 9 else 0.0
    elif contract_type == "DIGITUNDER":
        win_prob = float(digit_freqs[:barrier].sum()) if barrier > 0 else 0.0
    else:  # DIGITDIFF
        win_prob = float(1.0 - digit_freqs[barrier])

    # Map win_prob → extra ticks
    if   win_prob >= 0.90: dur_add = 0
    elif win_prob >= 0.80: dur_add = 1
    elif win_prob >= 0.70: dur_add = 2
    elif win_prob >= 0.60: dur_add = 3
    else:                  dur_add = 4

    # High volatility: digits resolve faster → trim 1 tick
    vol_adj = -1 if volatility > 2e-4 else 0

    # Neutral HMM: less directional conviction → add 1 tick
    state_adj = 1 if hmm_state == "neutral" else 0

    duration = base + dur_add + vol_adj + state_adj
    return int(np.clip(duration, DURATION_MIN, DURATION_MAX))


# ─────────────────────────────────────────────────────────────────────────────
# Phase
# ─────────────────────────────────────────────────────────────────────────────

class Phase(Enum):
    WARMUP   = auto()
    ANALYSE  = auto()
    TRADING  = auto()
    WAITING  = auto()
    RECOVERY = auto()


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
        """Returns std of recent returns — used by duration calculator."""
        if len(self._ret_history) < 5:
            return 1e-4
        return float(np.std(list(self._ret_history)[-30:]))

    def is_ready_to_train(self) -> tuple[bool, str]:
        n = len(self._ret_history)
        if n < MIN_SAMPLES_FOR_TRAINING:
            return False, f"only {n}/{MIN_SAMPLES_FOR_TRAINING} samples"
        if len(self._var_window) < 10:
            return False, "variance window not yet filled"
        var_samples = list(self._var_window)
        var_std  = np.std(var_samples)
        var_mean = np.mean(var_samples) + 1e-10
        rc = var_std / var_mean
        if rc > VARIANCE_STABILITY_EPS:
            return False, f"variance unstable ({rc:.3f} > {VARIANCE_STABILITY_EPS})"
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
        recent   = list(self._ret_history)[-40:]
        cur_mean = float(np.mean(recent))
        cur_std  = float(np.std(recent)) + 1e-10
        ref_std  = self._last_train_ret_std + 1e-10
        kl = (np.log(ref_std / cur_std)
              + (cur_std**2 + (cur_mean - self._last_train_ret_mean)**2)
              / (2 * ref_std**2) - 0.5)
        kl = float(np.clip(kl, 0, None))
        if kl > KL_DRIFT_THRESHOLD:
            return True, f"drift KL={kl:.4f}"
        return False, f"stable KL={kl:.4f}"

    def record_signal_snapshot(self, digit_chi2: float):
        self._last_digit_chi2 = digit_chi2

    def should_reanalyse(self, current_chi2: float) -> tuple[bool, str]:
        if self._last_digit_chi2 is None:
            return True, "no previous snapshot"
        change = abs(current_chi2 - self._last_digit_chi2) / (self._last_digit_chi2 + 1e-6)
        if change > DIGIT_CHANGE_THRESHOLD:
            return True, f"digit shift {change:.3f} > {DIGIT_CHANGE_THRESHOLD}"
        return False, f"digit stable (change={change:.3f})"


# ─────────────────────────────────────────────────────────────────────────────
# Digit Analyser  — complete overhaul of all_biases()
# ─────────────────────────────────────────────────────────────────────────────

class DigitAnalyser:
    def __init__(self):
        self.history: deque = deque(maxlen=DIGIT_HISTORY_MAX)
        self._streak_digit: Optional[int] = None
        self._streak_count: int = 0

    def push(self, digit: int):
        self.history.append(digit)
        if digit == self._streak_digit:
            self._streak_count += 1
        else:
            self._streak_digit = digit
            self._streak_count = 1

    def chi2_stat(self) -> float:
        if len(self.history) < 10:
            return 0.0
        counts   = np.bincount(list(self.history), minlength=10).astype(float)
        expected = np.full(10, len(self.history) / 10.0)
        return float(np.sum((counts - expected) ** 2 / expected))

    def freqs(self) -> np.ndarray:
        """Returns frequency array of digits 0-9."""
        if len(self.history) < 1:
            return np.full(10, 0.1)
        counts = np.bincount(list(self.history), minlength=10).astype(float)
        return counts / len(self.history)

    def all_biases(self) -> list[dict]:
        """
        Returns EVERY statistically supported trade candidate across ALL
        barrier values and ALL three contract types.

        DIGITDIFF barrier=d  — digit d appears significantly MORE than expected.
                               Win condition: last digit ≠ d.
                               Sources: streak, or freq[d] >= DIFF_FREQ_THRESHOLD.

        DIGITOVER barrier=b  — last digit will be strictly > b.
                               Edge exists when digit b is RARE (freq[b] < threshold),
                               meaning the market under-produces that digit, so most
                               ticks settle above it.
                               We scan every b from 0 to 8.

        DIGITUNDER barrier=b — last digit will be strictly < b.
                               Edge exists when digit b is RARE (freq[b] < threshold),
                               meaning most ticks settle below it.
                               We scan every b from 1 to 9.

        Each candidate carries chi2_strength and streak so the scorer can rank them.
        """
        candidates: list[dict] = []
        seen: set = set()  # (type, barrier) dedup

        def add(ctype: str, barrier: int, chi2s: float, sk: int, tag: str):
            key = (ctype, barrier)
            if key in seen:
                return
            seen.add(key)
            candidates.append({
                "type": ctype, "barrier": barrier,
                "chi2_strength": chi2s, "streak": sk,
                "reason_tag": tag,
            })

        streak  = self._streak_count
        strk_d  = self._streak_digit

        # ── 1. Streak → DIGITDIFF (unconditional, highest weight) ────────────
        if streak >= STREAK_THRESHOLD and strk_d is not None:
            add("DIGITDIFF", strk_d, 99.0, streak,
                f"streak:{strk_d}×{streak}")

        if len(self.history) < 10:
            return candidates

        digits = list(self.history)
        n      = len(digits)
        counts = np.bincount(digits, minlength=10).astype(float)
        fq     = counts / n
        expected = np.full(10, n / 10.0)
        chi2, p  = sp_stats.chisquare(counts, expected)

        # ── 2. DIGITDIFF — scan ALL digits for dominance ──────────────────────
        #    Any digit appearing >= DIFF_FREQ_THRESHOLD is a DIFFERS target.
        #    We don't cap at one: if digits 3 and 7 are both at 30%, both fire.
        for d in range(10):
            if fq[d] >= DIFF_FREQ_THRESHOLD:
                sk = streak if strk_d == d else 0
                add("DIGITDIFF", d, chi2, sk,
                    f"diff_dom:{d}({fq[d]:.0%})")

        # ── 3. DIGITOVER — scan barriers 0 through 8 ─────────────────────────
        #    OVER barrier b wins when last digit > b.
        #    Edge: digit b is RARE, so the market rarely lands on b,
        #    meaning most ticks go above it.
        #    We require p < DIGIT_CHI2_PVALUE to confirm non-uniformity,
        #    AND freq[b] ≤ 10% - MIN_DEFICIT (i.e. meaningfully under-represented).
        if p < DIGIT_CHI2_PVALUE:
            for b in range(9):  # barriers 0..8 (OVER 9 doesn't exist)
                if fq[b] <= (0.10 - MIN_DEFICIT_FOR_OVER_UNDER):
                    # How strong is this edge? Proportional to how rare digit b is.
                    strength = chi2 * (1.0 - fq[b] / 0.10)
                    add("DIGITOVER", b, strength, 0,
                        f"over{b}(freq[{b}]={fq[b]:.2%})")

        # ── 4. DIGITUNDER — scan barriers 1 through 9 ────────────────────────
        #    UNDER barrier b wins when last digit < b.
        #    Edge: digit b is RARE, so most ticks land below it.
        if p < DIGIT_CHI2_PVALUE:
            for b in range(1, 10):  # barriers 1..9 (UNDER 0 doesn't exist)
                if fq[b] <= (0.10 - MIN_DEFICIT_FOR_OVER_UNDER):
                    strength = chi2 * (1.0 - fq[b] / 0.10)
                    add("DIGITUNDER", b, strength, 0,
                        f"under{b}(freq[{b}]={fq[b]:.2%})")

        return candidates

    def mid_bias(self) -> Optional[str]:
        if len(self.history) < 10:
            return None
        digits   = list(self.history)
        n        = len(digits)
        low_freq = sum(1 for d in digits if d <= 4) / n
        if low_freq >= 0.60:
            return "low"
        if low_freq <= 0.40:
            return "high"
        return None

    def streak(self) -> int:
        return self._streak_count

    def stats(self) -> str:
        if len(self.history) < 5:
            return "n/a"
        counts = np.bincount(list(self.history), minlength=10)
        parts  = [f"{d}:{counts[d]}" for d in range(10) if counts[d] > 0]
        chi2   = self.chi2_stat()
        return f"[{' '.join(parts)}] chi2={chi2:.1f}"


# ─────────────────────────────────────────────────────────────────────────────
# Signal Scorer
# ─────────────────────────────────────────────────────────────────────────────

def score_candidate(dominance: float, chi2_strength: float,
                    streak: int, symbol_score: float,
                    contract_type: str, hmm_state: str) -> float:
    """
    Composite score in [0, 1].

    The HMM state is now a SOFT modifier, not a hard gate:
      - Aligned (OVER+bull, UNDER+bear):   +0.06 bonus
      - Neutral state (any type):          +0.00
      - DIGITDIFF (direction-agnostic):    +0.00
      - Mildly opposed (OVER+neutral bear, UNDER+neutral bull): -0.04
      - Strongly opposed (high-dominance bear + OVER, or bull + UNDER): penalty
        scales with dominance so a 95%-dominant bear truly blocks an OVER signal

    This means a very strong digit edge (high chi2, high streak) can still
    overcome a mildly opposing HMM state, but a strongly dominant opposing
    state will suppress the signal below SIGNAL_MIN_SCORE.
    """
    chi2_norm   = 1.0 / (1.0 + np.exp(-0.15 * (chi2_strength - 15.0)))
    streak_norm = min(streak / max(STREAK_THRESHOLD, 1), 1.0)

    raw = (SIGNAL_W_DOMINANCE * dominance
           + SIGNAL_W_CHI2    * chi2_norm
           + SIGNAL_W_STREAK  * streak_norm
           + SIGNAL_W_SYMBOL  * symbol_score)

    # Direction alignment modifier
    if contract_type == "DIGITDIFF":
        pass  # direction-agnostic — no modifier
    elif contract_type == "DIGITOVER":
        if hmm_state == "bull":
            raw += 0.06
        elif hmm_state == "bear":
            # Penalty scales with how dominant the opposing state is
            raw -= 0.04 + 0.12 * dominance
    elif contract_type == "DIGITUNDER":
        if hmm_state == "bear":
            raw += 0.06
        elif hmm_state == "bull":
            raw -= 0.04 + 0.12 * dominance

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
                     symbol: str,
                     symbol_score: float,
                     recovery_mode: bool = False,
                     mid_bias: Optional[str] = None) -> list[dict]:
        """
        Returns ALL viable signals for this symbol, scored and sorted.
        No hard direction gates — only soft score penalties for misalignment.
        """
        dominant, dominance = self._dominant_state()
        if dominant is None:
            return []

        effective_state = dominant

        if recovery_mode:
            sig = self._recovery_signal(
                effective_state, dominance, mid_bias, symbol, symbol_score
            )
            return [sig] if sig else []

        results = []
        for bias_info in digit_biases:
            ctype   = bias_info["type"]
            barrier = bias_info["barrier"]
            chi2s   = bias_info["chi2_strength"]
            streak  = bias_info["streak"]
            tag     = bias_info["reason_tag"]

            sc = score_candidate(dominance, chi2s, streak,
                                 symbol_score, ctype, effective_state)
            if sc < SIGNAL_MIN_SCORE:
                continue

            results.append({
                "type":      ctype,
                "barrier":   barrier,
                "symbol":    symbol,
                "hmm_state": effective_state,
                "dominance": dominance,
                "score":     sc,
                "chi2s":     chi2s,
                "reason": (
                    f"{symbol} HMM={effective_state}({dominance:.0%}) "
                    f"conf={confidence:.0%} {tag} "
                    f"→ {ctype} barrier={barrier}  score={sc:.3f}"
                ),
            })

        results.sort(key=lambda x: -x["score"])
        return results

    def _recovery_signal(self, state: str, dominance: float,
                          mid_bias: Optional[str],
                          symbol: str, symbol_score: float) -> Optional[dict]:
        sc = score_candidate(dominance, 0, 0, symbol_score,
                             "DIGITOVER" if state in ("bull", "neutral") else "DIGITUNDER",
                             state)
        base = {"symbol": symbol, "hmm_state": state,
                "dominance": dominance, "score": sc, "chi2s": 0.0}

        if state == "bull":
            if mid_bias in ("high", None):
                return {**base, "type": "DIGITOVER", "barrier": 5,
                        "reason": f"[REC] {symbol} bull({dominance:.0%}) mid={mid_bias} → OVER 5"}
        elif state == "bear":
            if mid_bias in ("low", None):
                return {**base, "type": "DIGITUNDER", "barrier": 4,
                        "reason": f"[REC] {symbol} bear({dominance:.0%}) mid={mid_bias} → UNDER 4"}
        elif state == "neutral":
            if mid_bias == "high":
                return {**base, "type": "DIGITOVER",  "barrier": 5,
                        "reason": f"[REC] {symbol} neutral mid=high → OVER 5"}
            if mid_bias == "low":
                return {**base, "type": "DIGITUNDER", "barrier": 4,
                        "reason": f"[REC] {symbol} neutral mid=low → UNDER 4"}
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Symbol Model
# ─────────────────────────────────────────────────────────────────────────────

class SymbolModel:
    def __init__(self, symbol: str):
        self.symbol       = symbol
        self.ticks: deque = deque(maxlen=600)
        self.trained      = False
        self._n_trains    = 0

        self._means: Optional[np.ndarray] = None
        self._vars:  Optional[np.ndarray] = None
        self._trans: Optional[np.ndarray] = None
        self._start: Optional[np.ndarray] = None
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
        roll5  = np.convolve(digits, np.ones(5) / 5, mode="same") / 9.0
        feat   = np.column_stack([rets, digits / 9.0, np.abs(rets), roll5])
        return feat[np.isfinite(feat).all(axis=1)]

    def train(self, force: bool = False) -> bool:
        ticks = list(self.ticks)
        feat  = self._features(ticks)
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

        means_ret = [feat[labels == k, 0].mean() for k in range(N_STATES)]
        order     = np.argsort(means_ret)
        remap     = np.empty(N_STATES, dtype=int)
        for si, rk in enumerate(order):
            remap[rk] = si
        sorted_labels = remap[labels]

        means = np.zeros((N_STATES, feat.shape[1]))
        varrs = np.ones((N_STATES, feat.shape[1])) * 1e-4
        for s in range(N_STATES):
            m = feat[sorted_labels == s]
            if len(m) < 2:
                return False
            means[s] = m.mean(axis=0)
            varrs[s] = np.clip(m.var(axis=0), 1e-6, None)

        tc = np.ones((N_STATES, N_STATES))
        for t in range(len(sorted_labels) - 1):
            tc[sorted_labels[t], sorted_labels[t + 1]] += 1
        transmat = tc / tc.sum(axis=1, keepdims=True)

        sp = np.bincount(sorted_labels, minlength=N_STATES).astype(float) + 1
        sp /= sp.sum()

        self._means  = means
        self._vars   = varrs
        self._trans  = np.log(transmat + 1e-300)
        self._start  = np.log(sp + 1e-300)
        self.trained = True

        self._readiness.record_train_snapshot(feat[:, 0])
        log.info(f"[{self.symbol}] HMM #{self._n_trains} trained | n={len(feat)} | "
                 f"mean_ret={means[:,0].round(7)} | "
                 f"sizes={[(sorted_labels==s).sum() for s in range(N_STATES)]}")
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
            d     = obs - self._means[s]
            lp[s] = -0.5 * np.sum(d**2 / self._vars[s]
                                  + np.log(2 * np.pi * self._vars[s]))
        return lp

    def analyse(self) -> tuple[Optional[str], float, list]:
        if not self.trained:
            return None, 0.0, []
        ticks = list(self.ticks)
        feat  = self._features(ticks[-200:])
        if len(feat) < 5:
            return None, 0.0, []
        try:
            T      = len(feat)
            vit    = np.full((T, N_STATES), -np.inf)
            bp     = np.zeros((T, N_STATES), dtype=int)
            vit[0] = self._start + self._log_emit(feat[0])
            for t in range(1, T):
                em = self._log_emit(feat[t])
                for s in range(N_STATES):
                    tp       = vit[t-1] + self._trans[:, s]
                    bp[t, s] = int(np.argmax(tp))
                    vit[t,s] = tp[bp[t,s]] + em[s]
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

    def get_all_signals(self, symbol_score: float,
                        recovery_mode: bool = False) -> list[dict]:
        """
        Runs Viterbi, updates path buffer, returns ALL scored candidates.
        Also attaches digit_freqs and volatility to each signal so the
        duration calculator has what it needs.
        """
        state, conf, path = self.analyse()
        if state is None:
            return []
        for s in path:
            self._signal_gen.record_state(s)

        biases = self._digit_analyser.all_biases()
        sigs   = self._signal_gen.evaluate_all(
            state, conf, biases, self.symbol,
            symbol_score  = symbol_score,
            recovery_mode = recovery_mode,
            mid_bias      = self._digit_analyser.mid_bias(),
        )

        # Attach live digit frequencies and volatility for duration computation
        fq  = self._digit_analyser.freqs()
        vol = self._readiness.recent_volatility()
        for sig in sigs:
            sig["_digit_freqs"] = fq
            sig["_volatility"]  = vol

        return sigs

    def readiness_status(self) -> str:
        ready, r1 = self._readiness.is_ready_to_train()
        drift, r2 = self._readiness.should_retrain()
        return f"ready={ready}({r1}) drift={drift}({r2})"

    def digit_stats(self) -> str:
        return self._digit_analyser.stats()


# ─────────────────────────────────────────────────────────────────────────────
# Symbol Scorer
# ─────────────────────────────────────────────────────────────────────────────

class SymbolScorer:
    def __init__(self):
        self._history: dict = {s: deque(maxlen=SCORE_WINDOW) for s in ALL_SYMBOLS}
        self._scores:  dict = {s: 0.5 for s in ALL_SYMBOLS}

    def record(self, symbol: str, profit: float):
        self._history[symbol].append((profit, time.time()))
        self._recompute(symbol)

    def _recompute(self, symbol: str):
        history = list(self._history[symbol])
        if not history:
            self._scores[symbol] = 0.5
            return
        ww, wt = 0.0, 0.0
        for i, (p, _) in enumerate(history):
            w   = SCORE_DECAY ** (len(history) - 1 - i)
            wt += w
            if p >= 0:
                ww += w
        self._scores[symbol] = ww / max(wt, 1e-9)

    def best_symbol(self, ready: list) -> Optional[str]:
        return max(ready, key=lambda s: self._scores.get(s, 0.5)) if ready else None

    def score(self, symbol: str) -> float:
        return self._scores.get(symbol, 0.5)

    def ranking(self) -> list:
        return sorted(self._scores.items(), key=lambda x: -x[1])

    def to_dict(self) -> dict:
        return dict(self._scores)

    def from_dict(self, d: dict):
        self._scores.update(d)


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
    def contract_mode(self) -> str:
        return "mid" if self.recovery_level >= 1 else "normal"

    def record_loss(self):
        self.consecutive_losses += 1
        self._update_level()

    def record_win(self):
        self.consecutive_losses = 0
        if self.recovery_level > 0:
            log.info("[RECOVERY] Win — stepping down level")
            self.recovery_level = max(0, self.recovery_level - 1)

    def _update_level(self):
        prev = self.recovery_level
        if self.consecutive_losses >= RECOVERY_L3_LOSSES:
            self.recovery_level = 3
        elif self.consecutive_losses >= RECOVERY_L2_LOSSES:
            self.recovery_level = 2
        elif self.consecutive_losses >= RECOVERY_L1_LOSSES:
            self.recovery_level = 1
        if self.recovery_level > prev:
            log.warning(f"[RECOVERY] Level → {self.recovery_level} "
                        f"({self.consecutive_losses} losses)")

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

    def get_stake_multiplier(self) -> float:
        return 0.5 if self.recovery_level >= 1 else 1.0

    def needs_symbol_switch(self) -> bool:
        return self.recovery_level >= 2

    def needs_hard_cooldown(self) -> bool:
        return self.recovery_level >= 3

    def status(self) -> str:
        return (f"level={self.recovery_level} mode={self.contract_mode} "
                f"losses={self.consecutive_losses} cooldown={self._in_cooldown}")


# ─────────────────────────────────────────────────────────────────────────────
# Risk Manager
# ─────────────────────────────────────────────────────────────────────────────

class RiskManager:
    def __init__(self):
        self.base_stake     = BASE_STAKE
        self.max_daily_loss = MAX_DAILY_LOSS
        self.daily_pnl      = 0.0
        self.trade_date     = date.today()
        self.total_trades   = 0
        self.total_wins     = 0

    def _day_reset(self):
        if date.today() != self.trade_date:
            log.info(f"New day | prev PnL={self.daily_pnl:+.2f} | "
                     f"W/L={self.total_wins}/{self.total_trades - self.total_wins}")
            self.daily_pnl  = 0.0
            self.trade_date = date.today()

    def is_halted(self) -> tuple[bool, str]:
        self._day_reset()
        if self.daily_pnl <= -abs(self.max_daily_loss):
            return True, f"daily loss limit ({self.daily_pnl:.2f})"
        return False, ""

    def stake(self, mult: float = 1.0) -> float:
        """
        Returns stake capped at 40% of the remaining daily loss budget.
        This prevents a single trade from wiping out the daily limit.
        E.g. MAX_DAILY_LOSS=$20, daily_pnl=-$5 → remaining=$15 → cap=$6.
        """
        remaining_budget = abs(self.max_daily_loss) - abs(min(self.daily_pnl, 0))
        budget_cap       = remaining_budget * 0.40
        raw              = self.base_stake * mult
        return round(max(min(raw, budget_cap), 0.35), 2)

    def record(self, profit: float):
        self._day_reset()
        self.daily_pnl    += profit
        self.total_trades += 1
        if profit >= 0:
            self.total_wins += 1

    def win_rate(self) -> float:
        return self.total_wins / max(1, self.total_trades)

    def summary(self) -> str:
        return (f"trades={self.total_trades} | "
                f"wr={self.win_rate():.0%} | pnl={self.daily_pnl:+.2f}")

    def to_dict(self) -> dict:
        return {"daily_pnl": self.daily_pnl, "total_trades": self.total_trades,
                "total_wins": self.total_wins, "trade_date": str(self.trade_date)}

    def from_dict(self, d: dict):
        self.daily_pnl    = d.get("daily_pnl", 0.0)
        self.total_trades = d.get("total_trades", 0)
        self.total_wins   = d.get("total_wins", 0)
        td = date.fromisoformat(d.get("trade_date", str(date.today())))
        if td != date.today():
            self.daily_pnl  = 0.0
            self.trade_date = date.today()
        else:
            self.trade_date = td


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

        self.phase             = Phase.WARMUP
        self.active_symbol     = ALL_SYMBOLS[0]
        self.current_signal: Optional[dict] = None
        self.trades_this_cycle = 0
        self.open_cid: Optional[str] = None
        self.trade_log: list = []

        # ── BUG-5 FIX: dirty flag prevents analysis storm ─────────────────────
        # Set to True each time a new tick arrives for any trained symbol.
        # Consumed (set False) once _run_global_analysis() fires.
        self._analysis_pending = False

        # ── BUG-6 FIX: counter tied to active_symbol; reset on switch ─────────
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
        td    = msg.get("tick", {})
        sym   = td.get("symbol", "")
        price = float(td.get("quote", 0))
        if sym not in self.models or price <= 0:
            return

        model = self.models[sym]
        model.push(price)

        # Training decisions (all symbols, all the time)
        if not model.trained:
            ready, reason = model._readiness.is_ready_to_train()
            if reason != self._last_readiness_log.get(sym):
                self._last_readiness_log[sym] = reason
                log.info(f"[{sym}] Readiness: {reason}")
            if ready:
                if model.train():
                    log.info(f"[{sym}] Initial model trained")
        else:
            should, reason = model._readiness.should_retrain()
            if should:
                log.info(f"[{sym}] Retraining — {reason}")
                model.train()

        # ── BUG-5 FIX: set dirty flag, don't call analysis directly ──────────
        # Only set the flag during phases where analysis is needed.
        if self.phase in (Phase.WARMUP, Phase.ANALYSE):
            self._analysis_pending = True

        # WAITING: count digits only for active symbol
        if self.phase == Phase.WAITING and sym == self.active_symbol:
            self._wait_digits_collected += 1

        # RECOVERY: check on active symbol's ticks only
        if self.phase == Phase.RECOVERY and sym == self.active_symbol:
            if self.recovery.check_cooldown_done():
                self._enter_analyse()
            return

        # Dispatch cycle logic once per tick per phase
        self._drive_cycle(sym)

    # ── Cycle state machine ───────────────────────────────────────────────────

    def _drive_cycle(self, sym: str):

        if self.phase == Phase.WARMUP:
            ready_count = sum(1 for m in self.models.values() if m.trained)
            if ready_count > 0:
                log.info(f"[WARMUP] {ready_count} symbol(s) ready — entering ANALYSE")
                self._enter_analyse()
            elif self._analysis_pending:
                if len(self.models[sym].ticks) % 30 == 0:
                    log.info(f"[WARMUP] {sym}: {self.models[sym].readiness_status()}")
            return

        if self.phase == Phase.WAITING:
            if sym != self.active_symbol:
                return
            if self._wait_digits_collected < MIN_WAIT_DIGITS:
                return
            active_model = self.models[self.active_symbol]
            chi2 = active_model._digit_analyser.chi2_stat()
            should, reason = active_model._readiness.should_reanalyse(chi2)
            if should:
                log.info(f"[WAITING] Re-analyse after {self._wait_digits_collected} "
                         f"digits — {reason}")
                for m in self.models.values():
                    if m.trained:
                        drift, _ = m._readiness.should_retrain()
                        if drift:
                            m.train()
                self._enter_analyse()
            return

        # ── ANALYSE: consume dirty flag — run exactly once per dirty batch ────
        if self.phase == Phase.ANALYSE:
            if self._analysis_pending:
                self._analysis_pending = False
                self._run_global_analysis()
            return

    def _enter_analyse(self):
        self.current_signal         = None
        self.trades_this_cycle      = 0
        self._wait_digits_collected = 0   # BUG-6 FIX: always reset here
        self._analysis_pending      = True
        self.phase                  = Phase.ANALYSE
        log.info(f"Phase → ANALYSE | recovery={self.recovery.status()}")

    def _switch_active_symbol(self, new_sym: str):
        """Central symbol-switch with BUG-6 counter reset."""
        if new_sym != self.active_symbol:
            log.info(f"Symbol switch: {self.active_symbol} → {new_sym}")
            self.active_symbol          = new_sym
            self._wait_digits_collected = 0   # BUG-6 FIX
            self.models[new_sym]._signal_gen.reset()

    def _run_global_analysis(self):
        """
        Scans ALL trained symbols for ALL viable signal types and picks the
        highest-scoring (symbol, contract_type, barrier) combination.
        """
        halted, reason = self.risk.is_halted()
        if halted:
            log.info(f"[RISK] Halted: {reason}")
            self.phase = Phase.WAITING
            return

        in_recov = self.recovery.contract_mode == "mid"
        all_candidates: list[dict] = []

        trained = [s for s in ALL_SYMBOLS if self.models[s].trained]
        if not trained:
            return

        for sym in trained:
            cands = self.models[sym].get_all_signals(
                symbol_score  = self.scorer.score(sym),
                recovery_mode = in_recov,
            )
            all_candidates.extend(cands)

        if not all_candidates:
            log.debug("[ANALYSE] No signals found — waiting for next tick")
            return

        all_candidates.sort(key=lambda x: -x["score"])

        # Log top candidates for full transparency
        top_n = min(6, len(all_candidates))
        log.info(f"[ANALYSE] {len(all_candidates)} candidates across "
                 f"{len(trained)} symbols — top {top_n}:")
        for i, c in enumerate(all_candidates[:top_n]):
            log.info(f"  #{i+1} {c['reason']}")

        best = all_candidates[0]

        # Compute data-driven duration (BUG-1 FIX)
        fq  = best.pop("_digit_freqs", np.full(10, 0.1))
        vol = best.pop("_volatility",  1e-4)
        best["duration"] = compute_duration(
            best["type"], best["barrier"], best["hmm_state"], fq, vol
        )

        # Snapshot digit distribution for re-analyse detection
        chosen_model = self.models[best["symbol"]]
        chosen_model._readiness.record_signal_snapshot(
            chosen_model._digit_analyser.chi2_stat()
        )

        self._switch_active_symbol(best["symbol"])
        self.current_signal = best
        log.info(f"╔ SIGNAL | {best['reason']} | dur={best['duration']}t")
        self.phase = Phase.TRADING
        self._place_trade(1)

    # ── Trade execution ───────────────────────────────────────────────────────

    def _place_trade(self, trade_num: int):
        sig      = self.current_signal
        stake    = self.risk.stake(self.recovery.get_stake_multiplier())
        ctype    = sig["type"]
        duration = sig.get("duration", DURATION_BASE.get(ctype, 5))

        log.info(f"├── TRADE {trade_num}/{TRADES_PER_CYCLE} | "
                 f"{ctype} {sig['symbol']} barrier={sig.get('barrier','n/a')} | "
                 f"stake=${stake} dur={duration}t HMM={sig.get('hmm_state','?')} "
                 f"score={sig.get('score', 0):.3f}")

        if not API_TOKEN:
            fake_cid = f"DEMO-{id(self)}-{self.trades_this_cycle}"
            self.open_cid           = fake_cid
            self.trades_this_cycle += 1
            import random
            # Realistic win rates by type and barrier proximity
            if ctype == "DIGITOVER":
                wr = 0.90 if sig.get("barrier", 0) <= 1 else 0.80
            elif ctype == "DIGITUNDER":
                wr = 0.90 if sig.get("barrier", 9) >= 8 else 0.80
            else:  # DIGITDIFF
                wr = 0.85
            sim_profit = round(stake * 0.87, 2) if random.random() < wr else -stake
            sim_status = "won" if sim_profit >= 0 else "lost"
            log.info(f"└── [DEMO] #{fake_cid} → {sim_status} ({profit_str(sim_profit)})")
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
        self.trades_this_cycle += 1
        log.info(f"└── #{self.open_cid} payout={b.get('payout')} cost={b.get('buy_price')}")
        self._send({"proposal_open_contract": 1,
                    "contract_id": self.open_cid, "subscribe": 1})

    def _on_poc(self, msg: dict):
        poc          = msg.get("proposal_open_contract", {})
        incoming_cid = str(poc.get("contract_id", ""))
        if self.open_cid is None:
            return
        if incoming_cid != str(self.open_cid):
            return
        status   = poc.get("status", "")
        is_final = poc.get("is_sold", False) or status in ("won", "lost")
        if not is_final:
            return
        profit = float(poc.get("profit", 0))
        log.info(f"    #{self.open_cid} → {status.upper()} {profit_str(profit)}")
        self.open_cid = None
        self._after_settlement(profit)

    def _after_settlement(self, profit: float):
        sym = self.current_signal["symbol"] if self.current_signal else self.active_symbol
        self.risk.record(profit)
        self.scorer.record(sym, profit)
        self.models[sym].online_update(profit)

        if profit >= 0:
            self.recovery.record_win()
        else:
            self.recovery.record_loss()

        self.trade_log.append({
            "ts":      datetime.now().isoformat(),
            "symbol":  sym,
            "type":    self.current_signal["type"]    if self.current_signal else "?",
            "barrier": self.current_signal.get("barrier") if self.current_signal else None,
            "dur":     self.current_signal.get("duration") if self.current_signal else None,
            "score":   round(self.current_signal.get("score", 0), 4) if self.current_signal else 0,
            "profit":  profit,
            "streak":  self.recovery.consecutive_losses,
        })
        if len(self.trade_log) % 10 == 0:
            self.persist.save(self.risk, self.scorer, self.trade_log)

        log.info(f"    {self.risk.summary()} | "
                 f"sym_score={self.scorer.score(sym):.0%} | "
                 f"recovery={self.recovery.status()}")

        halted, reason = self.risk.is_halted()
        if halted:
            log.warning(f"Halted: {reason}")
            self.phase = Phase.WAITING
            return

        if self.recovery.needs_hard_cooldown():
            self.phase = Phase.RECOVERY
            alts = [s for s in ALL_SYMBOLS if self.models[s].trained and s != sym]
            new_sym = self.scorer.best_symbol(alts) or self.active_symbol
            self._switch_active_symbol(new_sym)
            new_model = self.models[new_sym]
            new_model.train(force=True)
            self.recovery.start_cooldown(new_model)
            self.scorer._scores[sym] = 0.3
            return

        # ── On any loss: immediately force mid-barrier on the very next trade ──
        # The re-scan in _run_global_analysis will pick up recovery_mode=True
        # because recovery.contract_mode is now "mid" (L1=1 loss triggers it).
        # This means the bot switches to OVER 5 / UNDER 4 on the NEXT trade
        # without waiting for a cycle boundary.
        if profit < 0 and self.recovery.needs_symbol_switch():
            # L2+: also switch symbol
            alts = [s for s in ALL_SYMBOLS if self.models[s].trained and s != sym]
            if alts:
                new_sym = self.scorer.best_symbol(alts)
                if new_sym:
                    self._switch_active_symbol(new_sym)
                    log.info(f"[RECOVERY L{self.recovery.recovery_level}] "
                             f"Symbol switch → {new_sym}")

        if self.trades_this_cycle < TRADES_PER_CYCLE:
            # Re-scan all symbols — recovery_mode flag will already be True if
            # we just took a loss (L1=1), so next signal will be mid-barrier.
            log.info(f"├── Re-scanning before trade "
                     f"{self.trades_this_cycle + 1}/{TRADES_PER_CYCLE}"
                     + (" [MID-BARRIER mode]" if self.recovery.contract_mode == "mid" else "") + "…")
            self._analysis_pending = True
            self.phase = Phase.ANALYSE
            self._run_global_analysis()
        else:
            # Cycle done → baseline snapshot and enter WAITING
            active_model = self.models[self.active_symbol]
            end_chi2     = active_model._digit_analyser.chi2_stat()
            active_model._readiness.record_signal_snapshot(end_chi2)
            self._wait_digits_collected = 0
            log.info(
                f"╚ CYCLE DONE | {TRADES_PER_CYCLE}/{TRADES_PER_CYCLE} trades | "
                f"baseline chi2={end_chi2:.1f} | waiting for digit shift…"
            )
            self.phase = Phase.WAITING


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def profit_str(p: float) -> str:
    return f"{p:+.2f}"


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║    Deriv Adaptive AI Trading Bot  v8.0                      ║
║    1-3 tick preference · Instant loss recovery · Safe stake ║
╚══════════════════════════════════════════════════════════════╝""")
    print(f"  Symbols          : {', '.join(ALL_SYMBOLS)}")
    print(f"  Base stake        : ${BASE_STAKE:.2f}")
    print(f"  Max daily loss    : ${MAX_DAILY_LOSS:.2f}")
    print(f"  Stake cap         : 40% of remaining daily budget per trade")
    print()
    print("  Duration (ticks):")
    print(f"    win_prob ≥ 90%  → 1t   (strong edge, exit fast)")
    print(f"    win_prob ≥ 80%  → 2t")
    print(f"    win_prob ≥ 70%  → 3t   (target range)")
    print(f"    win_prob ≥ 60%  → 4t")
    print(f"    win_prob <  60% → 5t   (max, weak edge)")
    print()
    print("  Recovery (immediate):")
    print(f"    L1 = {RECOVERY_L1_LOSSES} loss  → mid-barrier NOW (OVER 5 / UNDER 4) on next trade")
    print(f"    L2 = {RECOVERY_L2_LOSSES} losses → switch symbol + mid-barrier")
    print(f"    L3 = {RECOVERY_L3_LOSSES} losses → data-driven cooldown on new symbol")
    print()
    print("  Signal coverage:")
    print(f"    DIGITOVER  barriers : 0–8   DIGITUNDER barriers : 1–9")
    print(f"    DIGITDIFF  barriers : 0–9   HMM gate : SOFT penalty")
    print(f"    score = {SIGNAL_W_DOMINANCE:.0%}×HMM + {SIGNAL_W_CHI2:.0%}×chi2 + "
          f"{SIGNAL_W_STREAK:.0%}×streak + {SIGNAL_W_SYMBOL:.0%}×sym_perf  "
          f"min={SIGNAL_MIN_SCORE}")
    print(f"  Auth token : {'SET ✓' if API_TOKEN else 'NOT SET — demo mode'}")
    print()

    bot = DerivAdaptiveBot()
    try:
        bot.start()
    except KeyboardInterrupt:
        log.info("Interrupted")
        bot.stop()


if __name__ == "__main__":
    main()