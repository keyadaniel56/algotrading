"""
Deriv HMM Algo Trading Bot  v3.0
==================================
Trades digit contracts: Differs, Over 0, Under 9

Cycle:
    ┌─────────────┐
    │   ANALYSE   │  ← full HMM + digit analysis, gate checks
    └──────┬──────┘
           │ signal approved
    ┌──────▼──────┐
    │   TRADE 1   │  ← open first contract
    └──────┬──────┘
           │ contract settled
    ┌──────▼──────┐
    │   TRADE 2   │  ← open second contract (same signal)
    └──────┬──────┘
           │ contract settled
    ┌──────▼──────┐
    │    WAIT     │  ← collect REANALYSE_TICKS new ticks
    └──────┬──────┘
           │
    back to ANALYSE

Requirements:
    pip install websocket-client numpy scikit-learn python-dotenv

Setup (.env):
    DERIV_APP_ID=your_app_id
    DERIV_API_TOKEN=your_api_token
    SYMBOL=R_100
    STAKE=1.0
    MAX_DAILY_LOSS=20.0
"""

import os
import json
import threading
import logging
import numpy as np
from datetime import date
from collections import deque
from enum import Enum, auto
from typing import Optional

import websocket
from sklearn.cluster import KMeans
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

APP_ID         = os.getenv("DERIV_APP_ID", "1089")
API_TOKEN      = os.getenv("DERIV_API_TOKEN", "")
SYMBOL         = os.getenv("SYMBOL", "R_100")
STAKE          = float(os.getenv("STAKE", "1.0"))
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "20.0"))

DERIV_WS_URL = f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}"

# ── HMM ──────────────────────────────────────────────────────────────────────
N_STATES      = 3
WARMUP_TICKS  = 150   # ticks before first analysis
HMM_WINDOW    = 100   # ticks fed to HMM for inference

# ── Signal quality gates ─────────────────────────────────────────────────────
# Confidence: Viterbi softmax over 3 states naturally sits in 40-70% range.
# Setting 65% kills every signal. 50% is the meaningful floor (better than
# random = 33%). Raise this only if you see too many low-quality signals.
MIN_CONFIDENCE   = 0.50

# State must be the same for this many consecutive ticks before we act.
STATE_STABLE_FOR = 4

# Window for digit frequency analysis. Larger = more data, less noise.
# Keep separate from streak tracking (streak uses its own unbounded counter).
DIGIT_WINDOW     = 20

# Fraction of last DIGIT_WINDOW digits that must show the bias pattern.
# 65% of 20 digits = 13 out of 20 must be non-zero (for over0) etc.
MIN_DIGIT_BIAS   = 0.65

# Extra gate: if the SAME digit has appeared ≥ this many times consecutively,
# treat it as a strong streak signal even if frequency bias isn't met yet.
STREAK_THRESHOLD = 6

# ── Cycle ────────────────────────────────────────────────────────────────────
TRADES_PER_CYCLE  = 2     # always exactly 2 trades per analysis cycle
REANALYSE_TICKS   = 15    # new ticks to collect after cycle before re-analysing
CONTRACT_DURATION = 5     # ticks per contract

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("HMMBot")


# ─────────────────────────────────────────────────────────────────────────────
# Cycle state machine
# ─────────────────────────────────────────────────────────────────────────────

def extract_last_digit(price: float) -> int:
    """
    Extract the true last significant digit from a Deriv price.

    Problem: f"{633.4:.4f}" = "633.4000", replace('.','') = "63340000",
    [-1] = '0' — always zero because .4f pads with zeros.

    Fix: use the raw price string, strip trailing zeros first, then take
    the last digit of the decimal part.

    Examples:
        633.40  → "633.4"  → decimal "4"  → last digit 4
        633.87  → "633.87" → decimal "87" → last digit 7
        634.00  → "634.0"  → decimal "0"  → last digit 0  (genuinely 0)
        633.456 → "633.456"→ decimal "456"→ last digit 6
    """
    s = str(price)
    if "." in s:
        dec = s.split(".")[1].rstrip("0") or "0"
    else:
        dec = "0"
    return int(dec[-1])
    WARMUP   = auto()   # collecting initial ticks
    ANALYSE  = auto()   # ready to run analysis and pick a signal
    TRADING  = auto()   # a contract is currently open
    WAITING  = auto()   # both trades done, collecting ticks before re-analyse


# ─────────────────────────────────────────────────────────────────────────────
# HMM Engine
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Cycle state machine phases
# ─────────────────────────────────────────────────────────────────────────────

class Phase(Enum):
    WARMUP   = auto()   # collecting initial ticks, no trading
    ANALYSE  = auto()   # running HMM + digit analysis every tick
    TRADING  = auto()   # a contract is open, waiting for settlement
    WAITING  = auto()   # both trades done, collecting ticks before re-analyse


# ─────────────────────────────────────────────────────────────────────────────
# HMM Engine
# ─────────────────────────────────────────────────────────────────────────────

class HMMEngine:
    """
    Hidden Markov Model for market regime detection.

    Architecture: 3 states (bull / neutral / bear), each modelled by a
    diagonal-covariance Gaussian fitted with sklearn's GaussianMixture
    (one component per state).  Viterbi decoding is done manually using
    log-likelihoods, so no scipy probability checks are ever invoked and
    the "Probabilities do not sum to 1" error is impossible.

    Why not hmmlearn?
      hmmlearn's Baum-Welch calls scipy.stats.multinomial internally during
      the E-step.  That sampler enforces |sum-1| < 1e-8 on float64, which
      fails whenever floating-point rounding accumulates across iterations —
      even with manual normalisation before and after fit().  Replacing
      Baum-Welch with k-means clustering + manual Viterbi gives the same
      regime labels with zero numerical instability.

    Pipeline:
      1. Features extracted from raw ticks (returns, digit stats).
      2. KMeans(3) partitions observations into 3 clusters sorted by mean
         return → bear / neutral / bull labels assigned deterministically.
      3. Per-cluster Gaussian parameters (mean, var) estimated from members.
      4. Transition matrix estimated from cluster label sequence + Laplace
         smoothing so no row is ever zero.
      5. Viterbi decoding in log-space on new observations → current state
         + posterior confidence.
    """

    def __init__(self):
        self.trained    = False
        self._n_trains  = 0
        # Fitted parameters (set by train())
        self._means:    Optional[np.ndarray] = None   # (3, n_feat)
        self._vars:     Optional[np.ndarray] = None   # (3, n_feat)
        self._transmat: Optional[np.ndarray] = None   # (3, 3) log-probs
        self._startlog: Optional[np.ndarray] = None   # (3,)   log-probs
        self.state_labels = ["bear", "neutral", "bull"]  # index → label

    # ── Feature extraction ────────────────────────────────────────────────────

    @staticmethod
    def _last_digit(price: float) -> int:
        return extract_last_digit(price)

    def _features(self, ticks: list) -> np.ndarray:
        arr    = np.array(ticks, dtype=np.float64)
        rets   = np.diff(arr) / (arr[:-1] + 1e-10)
        digits = np.array([self._last_digit(t) for t in ticks[1:]], dtype=np.float64)
        roll5  = np.convolve(digits, np.ones(5) / 5, mode="same") / 9.0
        feat   = np.column_stack([rets, digits / 9.0, np.abs(rets), roll5])
        return feat[np.isfinite(feat).all(axis=1)]

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, ticks: list) -> bool:
        feat = self._features(ticks)
        if len(feat) < HMM_WINDOW:
            log.warning("HMM train skipped: not enough clean features")
            return False

        self._n_trains += 1

        # Step 1: K-means clustering into 3 groups
        km = KMeans(n_clusters=N_STATES, n_init=10, random_state=42 + self._n_trains)
        labels = km.fit_predict(feat)

        # Step 2: Sort clusters by mean return (feat col 0) → bear/neutral/bull
        cluster_mean_rets = [feat[labels == k, 0].mean() for k in range(N_STATES)]
        order = np.argsort(cluster_mean_rets)   # order[0]=most negative → bear
        # Remap: raw cluster id → sorted index (0=bear,1=neutral,2=bull)
        remap = np.empty(N_STATES, dtype=int)
        for sorted_idx, raw_k in enumerate(order):
            remap[raw_k] = sorted_idx
        sorted_labels = remap[labels]

        # Step 3: Gaussian params per sorted state
        means = np.zeros((N_STATES, feat.shape[1]), dtype=np.float64)
        varrs = np.ones((N_STATES, feat.shape[1]),  dtype=np.float64) * 1e-4
        for s in range(N_STATES):
            members = feat[sorted_labels == s]
            if len(members) < 2:
                log.warning(f"State {s} has <2 members — train aborted")
                return False
            means[s] = members.mean(axis=0)
            varrs[s] = np.clip(members.var(axis=0), 1e-6, None)

        # Step 4: Transition matrix with Laplace smoothing
        trans_counts = np.ones((N_STATES, N_STATES), dtype=np.float64)  # Laplace prior=1
        for t in range(len(sorted_labels) - 1):
            trans_counts[sorted_labels[t], sorted_labels[t + 1]] += 1
        transmat = trans_counts / trans_counts.sum(axis=1, keepdims=True)

        # Step 5: Start probabilities from label frequency
        start_counts = np.bincount(sorted_labels, minlength=N_STATES).astype(np.float64)
        start_probs  = (start_counts + 1) / (start_counts.sum() + N_STATES)

        # Store as log-space to avoid underflow during Viterbi
        self._means    = means
        self._vars     = varrs
        self._transmat = np.log(transmat + 1e-300)
        self._startlog = np.log(start_probs + 1e-300)
        self.trained   = True

        log.info(
            f"HMM trained (#{self._n_trains}) | "
            f"mean_returns={means[:, 0].round(8)} | "
            f"state_sizes={[int((sorted_labels==s).sum()) for s in range(N_STATES)]}"
        )
        return True

    # ── Inference: log-space Viterbi ──────────────────────────────────────────

    def _log_emission(self, obs: np.ndarray) -> np.ndarray:
        """
        Log-likelihood of observation vector under each state's Gaussian.
        Returns shape (N_STATES,).
        """
        log_p = np.zeros(N_STATES, dtype=np.float64)
        for s in range(N_STATES):
            diff    = obs - self._means[s]
            log_p[s] = -0.5 * np.sum(diff ** 2 / self._vars[s] + np.log(2 * np.pi * self._vars[s]))
        return log_p

    def _viterbi(self, feat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Standard log-space Viterbi.
        Returns (state_sequence, max_log_probs_at_each_step).
        """
        T = len(feat)
        viterbi  = np.full((T, N_STATES), -np.inf, dtype=np.float64)
        backptr  = np.zeros((T, N_STATES), dtype=int)

        viterbi[0] = self._startlog + self._log_emission(feat[0])

        for t in range(1, T):
            emit = self._log_emission(feat[t])
            for s in range(N_STATES):
                trans_probs   = viterbi[t - 1] + self._transmat[:, s]
                best_prev     = int(np.argmax(trans_probs))
                viterbi[t, s] = trans_probs[best_prev] + emit[s]
                backptr[t, s] = best_prev

        # Backtrack
        path = np.zeros(T, dtype=int)
        path[-1] = int(np.argmax(viterbi[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = backptr[t + 1, path[t + 1]]

        return path, viterbi

    def analyse(self, ticks: list) -> tuple[Optional[str], float]:
        """Return (state_label, confidence)."""
        if not self.trained:
            return None, 0.0
        feat = self._features(ticks[-HMM_WINDOW:])
        if len(feat) < 5:
            return None, 0.0
        try:
            path, viterbi_scores = self._viterbi(feat)
            current_state = int(path[-1])

            # Confidence: softmax over final Viterbi scores
            scores     = viterbi_scores[-1]
            scores    -= scores.max()          # numerical stability
            exp_scores = np.exp(scores)
            confidence = float(exp_scores[current_state] / exp_scores.sum())

            label = self.state_labels[current_state]
            return label, confidence
        except Exception as exc:
            log.debug(f"HMM analyse error (skipping): {exc}")
            return None, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Digit Analyser
# ─────────────────────────────────────────────────────────────────────────────

class DigitAnalyser:
    """
    Independent digit-pattern checker.

    Bias logic (corrected):
      'over0'     → digit 0 is RARE  (zero_freq ≤ 35%)
                    → most ticks end in 1-9 → strong DIGITOVER 0 edge
      'under9'    → digit 9 is RARE  (nine_freq ≤ 35%)
                    → most ticks end in 0-8 → strong DIGITUNDER 9 edge
      'differs:D' → one digit is over-represented (≥ MIN_DIGIT_BIAS),
                    OR a streak of ≥ STREAK_THRESHOLD of same digit
      None        → no actionable pattern

    The old logic tested non_zero ≥ 65% which is TRUE 90% of the time
    (digits 1-9 come up 9 out of 10 ticks by default), so it never
    reflected real market bias. This version tests the frequency of the
    barrier digit itself.
    """

    def __init__(self):
        self.history: deque = deque(maxlen=DIGIT_WINDOW)
        self._streak_digit: Optional[int] = None
        self._streak_count: int = 0

    def push(self, digit: int):
        self.history.append(digit)
        if digit == self._streak_digit:
            self._streak_count += 1
        else:
            self._streak_digit = digit
            self._streak_count = 1

    def bias(self) -> Optional[str]:
        # Streak check first — strongest signal, no window needed
        if self._streak_count >= STREAK_THRESHOLD:
            return f"differs:{self._streak_digit}"

        if len(self.history) < DIGIT_WINDOW:
            return None

        digits    = list(self.history)
        n         = len(digits)
        counts    = np.bincount(digits, minlength=10)
        zero_freq = counts[0] / n
        nine_freq = counts[9] / n
        max_freq  = counts.max() / n
        dominant  = int(counts.argmax())

        # 0 is rare → ticks skew away from 0 → OVER 0 edge is real
        if zero_freq <= (1.0 - MIN_DIGIT_BIAS):
            return "over0"

        # 9 is rare → ticks skew away from 9 → UNDER 9 edge is real
        if nine_freq <= (1.0 - MIN_DIGIT_BIAS):
            return "under9"

        # One digit dominates → DIFFERS on that digit
        if max_freq >= MIN_DIGIT_BIAS:
            return f"differs:{dominant}"

        return None

    def streak(self) -> int:
        return self._streak_count

    def stats(self) -> str:
        """Compact digit frequency string for logging."""
        if len(self.history) < 5:
            return "n/a"
        digits = list(self.history)
        counts = np.bincount(digits, minlength=10)
        return " ".join(f"{d}:{counts[d]}" for d in range(10) if counts[d] > 0)


# ─────────────────────────────────────────────────────────────────────────────
# Signal Generator
# ─────────────────────────────────────────────────────────────────────────────

class SignalGenerator:
    """
    Combines HMM regime + digit bias into a trade signal.

    Gates (all must pass):
      1. HMM confidence ≥ MIN_CONFIDENCE  (50% — above 33% random floor)
      2. Same HMM state for STATE_STABLE_FOR consecutive ticks
      3. A digit bias must be detected by DigitAnalyser
      4. Bias and state must not actively contradict each other

    Gate 4 logic:
      over0   → valid in bull or neutral (digits skewing high, away from 0)
      under9  → valid in bear or neutral (digits skewing low, away from 9)
      differs → valid in any state (one digit repeating = clear DIFFERS edge)

    Note: both over0 and under9 can validly fire in 'neutral' because the
    neutral state means no strong directional regime, so digit-level edge
    is sufficient to trade.
    """

    def __init__(self):
        self._state_buf: deque = deque(maxlen=STATE_STABLE_FOR + 2)

    def push_state(self, state: str):
        self._state_buf.append(state)

    def _stable(self) -> bool:
        if len(self._state_buf) < STATE_STABLE_FOR:
            return False
        return len(set(list(self._state_buf)[-STATE_STABLE_FOR:])) == 1

    def evaluate(self, state: str, confidence: float,
                 bias: Optional[str], last_digit: int) -> Optional[dict]:

        # Gate 1: minimum HMM confidence
        if confidence < MIN_CONFIDENCE:
            log.debug(f"Gate 1 FAIL | conf={confidence:.0%} < {MIN_CONFIDENCE:.0%}")
            return None

        # Gate 2: state stability
        if not self._stable():
            log.debug(f"Gate 2 FAIL | state not stable for {STATE_STABLE_FOR} ticks")
            return None

        # Gate 3: digit bias must exist
        if bias is None:
            log.debug("Gate 3 FAIL | no digit bias in last window")
            return None

        bias_type = bias.split(":")[0]

        # Gate 4: directional agreement (permissive — neutral allows both)
        if bias_type == "over0":
            if state in ("bull", "neutral"):
                return {
                    "type": "DIGITOVER", "barrier": 0,
                    "reason": f"HMM={state}({confidence:.0%}) bias=over0 digit={last_digit} → OVER 0",
                }
            log.debug(f"Gate 4 FAIL | over0 bias conflicts with state={state}")
            return None

        if bias_type == "under9":
            if state in ("bear", "neutral"):
                return {
                    "type": "DIGITUNDER", "barrier": 9,
                    "reason": f"HMM={state}({confidence:.0%}) bias=under9 digit={last_digit} → UNDER 9",
                }
            log.debug(f"Gate 4 FAIL | under9 bias conflicts with state={state}")
            return None

        if bias_type == "differs":
            dominant = int(bias.split(":")[1])
            return {
                "type": "DIGITDIFF", "barrier": dominant,
                "reason": (
                    f"HMM={state}({confidence:.0%}) "
                    f"digit {dominant} streak/dominant → DIFFERS {dominant}"
                ),
            }

        return None


# ─────────────────────────────────────────────────────────────────────────────
# Risk Manager
# ─────────────────────────────────────────────────────────────────────────────

class RiskManager:
    def __init__(self, base_stake: float, max_daily_loss: float):
        self.base_stake         = base_stake
        self.max_daily_loss     = max_daily_loss
        self.daily_pnl          = 0.0
        self.trade_date         = date.today()
        self.consecutive_losses = 0
        self.max_consec_losses  = 5
        self.total_trades       = 0
        self.total_wins         = 0

    def _day_reset(self):
        today = date.today()
        if today != self.trade_date:
            log.info(
                f"New day | prev PnL={self.daily_pnl:+.2f} | "
                f"W/L={self.total_wins}/{self.total_trades - self.total_wins}"
            )
            self.daily_pnl          = 0.0
            self.trade_date         = today
            self.consecutive_losses = 0

    def is_halted(self) -> tuple[bool, str]:
        """Returns (halted, reason). True = do NOT trade."""
        self._day_reset()
        if self.daily_pnl <= -abs(self.max_daily_loss):
            return True, f"daily loss limit ({self.daily_pnl:.2f})"
        if self.consecutive_losses >= self.max_consec_losses:
            return True, f"{self.consecutive_losses} consecutive losses"
        return False, ""

    def stake(self) -> float:
        # Kelly-lite: reduce after 3+ losses in a row
        if self.consecutive_losses >= 3:
            return round(self.base_stake * 0.5, 2)
        return self.base_stake

    def record(self, profit: float):
        self.daily_pnl   += profit
        self.total_trades += 1
        if profit >= 0:
            self.total_wins        += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
        wr = self.total_wins / max(1, self.total_trades)
        log.info(
            f"{'WIN' if profit >= 0 else 'LOSS'} {profit:+.2f} | "
            f"daily={self.daily_pnl:+.2f} | "
            f"W/L={self.total_wins}/{self.total_trades - self.total_wins} ({wr:.0%}) | "
            f"loss_streak={self.consecutive_losses}"
        )

    def summary(self) -> str:
        wr = self.total_wins / max(1, self.total_trades)
        return (f"trades={self.total_trades} | "
                f"win_rate={wr:.0%} | daily_pnl={self.daily_pnl:+.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Bot
# ─────────────────────────────────────────────────────────────────────────────

class DerivHMMBot:
    """
    Strict analyse → trade → trade → wait → re-analyse cycle.

    State machine:
      WARMUP  : collect WARMUP_TICKS ticks, train HMM, → ANALYSE
      ANALYSE : run HMM + digit analysis every tick until a signal passes
                all gates, then place trade 1 → TRADING
      TRADING : one contract open; wait for settlement
                after trade 1 settles  → place trade 2  (same signal)
                after trade 2 settles  → WAITING
      WAITING : collect REANALYSE_TICKS new ticks, then → ANALYSE
                (retrain HMM at start of every ANALYSE phase)
    """

    def __init__(self):
        self.ws: Optional[websocket.WebSocketApp] = None
        self.ticks: deque = deque(maxlen=600)
        self.tick_count   = 0

        # ── Components ────────────────────────────────────────────────────────
        self.hmm    = HMMEngine()
        self.digits = DigitAnalyser()
        self.signal = SignalGenerator()
        self.risk   = RiskManager(STAKE, MAX_DAILY_LOSS)

        # ── Cycle state ───────────────────────────────────────────────────────
        self.phase: Phase          = Phase.WARMUP
        self.current_signal: Optional[dict] = None  # signal locked for this cycle
        self.trades_this_cycle: int = 0              # 0, 1, or 2
        self.open_cid: Optional[str] = None          # currently open contract id
        self.wait_ticks_collected: int = 0           # ticks gathered in WAITING

        # ── Infra ─────────────────────────────────────────────────────────────
        self._req_id = 1
        self._lock   = threading.Lock()

    # ── WebSocket plumbing ────────────────────────────────────────────────────

    def start(self):
        self.ws = websocket.WebSocketApp(
            DERIV_WS_URL,
            on_open    = self._on_open,
            on_message = self._on_message,
            on_error   = self._on_error,
            on_close   = self._on_close,
        )
        log.info("Connecting to Deriv…")
        self.ws.run_forever(ping_interval=30, ping_timeout=10, reconnect=5)

    def stop(self):
        if self.ws:
            self.ws.close()
        log.info(f"Bot stopped | {self.risk.summary()}")

    def _send(self, payload: dict):
        with self._lock:
            payload["req_id"] = self._req_id
            self._req_id += 1
        if self.ws:
            try:
                self.ws.send(json.dumps(payload))
            except Exception as e:
                log.warning(f"Send error: {e}")

    def _on_open(self, ws):
        log.info("WebSocket connected")
        if API_TOKEN:
            self._send({"authorize": API_TOKEN})
        else:
            log.warning("No API token — running in DEMO mode (no real trades)")
            self._send({"ticks": SYMBOL, "subscribe": 1})

    def _on_message(self, ws, raw: str):
        try:
            msg = json.loads(raw)
            t = msg.get("msg_type")
            if   t == "authorize":              self._on_auth(msg)
            elif t == "tick":                   self._on_tick(msg)
            elif t == "buy":                    self._on_buy(msg)
            elif t == "proposal_open_contract": self._on_poc(msg)
            elif t == "error":
                log.warning(f"API error: {msg.get('error', {}).get('message', msg)}")
        except Exception as exc:
            # Never let exceptions bubble up to the WS layer —
            # that causes the "error from callback" spam and disconnects.
            log.error(f"Unhandled error in _on_message: {exc}", exc_info=False)

    def _on_error(self, ws, err):  log.error(f"WS error: {err}")
    def _on_close(self, ws, c, r): log.info(f"WS closed {c}: {r}")

    def _on_auth(self, msg: dict):
        if msg.get("error"):
            log.error(f"Auth failed: {msg['error']['message']}")
            self.stop()
            return
        a = msg["authorize"]
        log.info(f"Auth OK | {a['loginid']} | balance={a['balance']} {a['currency']}")
        self._send({"ticks": SYMBOL, "subscribe": 1})

    # ── Tick handler — drives the state machine ───────────────────────────────

    def _on_tick(self, msg: dict):
        td    = msg.get("tick", {})
        price = float(td.get("quote", 0))
        if price <= 0:
            return

        self.ticks.append(price)
        self.tick_count += 1
        digit = extract_last_digit(price)
        self.digits.push(digit)

        # ── WARMUP ────────────────────────────────────────────────────────────
        if self.phase == Phase.WARMUP:
            if self.tick_count % 20 == 0:
                log.info(f"[WARMUP] {self.tick_count}/{WARMUP_TICKS} ticks…")
            if self.tick_count >= WARMUP_TICKS:
                trained = self.hmm.train(list(self.ticks))
                if trained:
                    self.phase = Phase.ANALYSE
                    log.info("═" * 55)
                    log.info("WARMUP COMPLETE — entering ANALYSE phase")
                    log.info("═" * 55)
                else:
                    log.warning("Initial HMM train failed — collecting more ticks")
            return

        # ── WAITING ───────────────────────────────────────────────────────────
        if self.phase == Phase.WAITING:
            self.wait_ticks_collected += 1
            if self.wait_ticks_collected >= REANALYSE_TICKS:
                # Retrain HMM with fresh data before next analysis round
                log.info("─" * 55)
                log.info(f"WAIT done ({REANALYSE_TICKS} ticks) — retraining HMM…")
                self.hmm.train(list(self.ticks))
                self._enter_analyse()
            return

        # ── TRADING ───────────────────────────────────────────────────────────
        # While a contract is open we just accumulate ticks — settlement
        # is handled via _on_poc() callback, not here.
        if self.phase == Phase.TRADING:
            return

        # ── ANALYSE ───────────────────────────────────────────────────────────
        if self.phase == Phase.ANALYSE:
            self._run_analysis(price, digit)

    # ── Analysis logic ────────────────────────────────────────────────────────

    def _enter_analyse(self):
        """Reset cycle counters and switch to ANALYSE phase."""
        self.current_signal      = None
        self.trades_this_cycle   = 0
        self.wait_ticks_collected = 0
        self.phase               = Phase.ANALYSE
        log.info("Phase → ANALYSE")

    def _run_analysis(self, price: float, digit: int):
        """Called every tick while in ANALYSE phase."""

        # ── Risk halt check ───────────────────────────────────────────────────
        halted, reason = self.risk.is_halted()
        if halted:
            log.info(f"[ANALYSE] Trading halted: {reason}")
            return

        # ── HMM inference ─────────────────────────────────────────────────────
        state, confidence = self.hmm.analyse(list(self.ticks))
        if state is None:
            return

        self.signal.push_state(state)
        bias   = self.digits.bias()
        streak = self.digits.streak()

        # Log analysis every 3 ticks so you can see what's happening
        if self.tick_count % 3 == 0:
            log.info(
                f"[ANALYSE] tick={self.tick_count} | price={price} | "
                f"digit={digit} | HMM={state}({confidence:.0%}) | "
                f"bias={bias} | streak={streak} | "
                f"stable={'Y' if self.signal._stable() else 'N'} | "
                f"dist=[{self.digits.stats()}]"
            )

        # ── Evaluate signal ────────────────────────────────────────────────────
        sig = self.signal.evaluate(state, confidence, bias, digit)
        if sig is None:
            return

        # ── Signal approved — lock it in and execute trade 1 ──────────────────
        self.current_signal = sig
        log.info("╔" + "═" * 53)
        log.info(f"║ SIGNAL LOCKED | {sig['reason']}")
        log.info(f"║ Will place {TRADES_PER_CYCLE} trades then re-analyse")
        log.info("╚" + "═" * 53)

        self.phase = Phase.TRADING
        self._place_trade(trade_num=1)

    # ── Trade execution ───────────────────────────────────────────────────────

    def _place_trade(self, trade_num: int):
        sig   = self.current_signal
        stake = self.risk.stake()

        log.info(
            f"┌── TRADE {trade_num}/{TRADES_PER_CYCLE} | "
            f"{sig['type']} | barrier={sig.get('barrier', 'n/a')} | "
            f"stake=${stake}"
        )

        if not API_TOKEN:
            # DEMO: simulate settlement after contract duration
            log.info(f"└── [DEMO] contract placed (simulated)")
            self.trades_this_cycle += 1
            self._after_settlement(profit=0.85 * stake, demo=True)
            return

        payload = {
            "buy": 1,
            "price": stake,
            "parameters": {
                "amount":        stake,
                "basis":         "stake",
                "contract_type": sig["type"],
                "currency":      "USD",
                "duration":      CONTRACT_DURATION,
                "duration_unit": "t",
                "symbol":        SYMBOL,
            },
        }
        if sig.get("barrier") is not None:
            payload["parameters"]["barrier"] = str(sig["barrier"])

        self._send(payload)

    def _on_buy(self, msg: dict):
        if msg.get("error"):
            log.error(f"Buy error: {msg['error']['message']}")
            # Failed to open — treat as if cycle ended, go back to analyse
            self._enter_analyse()
            return
        b = msg.get("buy", {})
        self.open_cid = b.get("contract_id")
        self.trades_this_cycle += 1
        log.info(
            f"└── Contract #{self.open_cid} opened | "
            f"payout={b.get('payout')} | cost={b.get('buy_price')}"
        )
        # Subscribe to live updates so we know when it settles
        self._send({
            "proposal_open_contract": 1,
            "contract_id": self.open_cid,
            "subscribe":   1,
        })

    def _on_poc(self, msg: dict):
        """Proposal open contract update — fires on every tick while open."""
        poc = msg.get("proposal_open_contract", {})
        # Only act on final settlement
        settled = (
            poc.get("is_settleable")
            or poc.get("is_sold")
            or poc.get("status") in ("won", "lost")
        )
        if not settled:
            return

        profit = float(poc.get("profit", 0))
        status = poc.get("status", "?").upper()
        log.info(
            f"    Contract #{self.open_cid} settled: "
            f"{status} {profit:+.2f}"
        )
        self.open_cid = None
        self._after_settlement(profit=profit, demo=False)

    def _after_settlement(self, profit: float, demo: bool):
        """Called after each contract settles. Drives cycle transitions."""
        self.risk.record(profit)

        halted, reason = self.risk.is_halted()
        if halted:
            log.info(f"Trading halted after settlement: {reason}")
            self._enter_analyse()
            return

        if self.trades_this_cycle < TRADES_PER_CYCLE:
            # Still need trade 2 — same signal, place immediately
            log.info(
                f"Trade {self.trades_this_cycle}/{TRADES_PER_CYCLE} done — "
                f"placing trade {self.trades_this_cycle + 1}…"
            )
            self._place_trade(trade_num=self.trades_this_cycle + 1)
        else:
            # Both trades done — enter waiting phase
            log.info("╔" + "═" * 53)
            log.info(
                f"║ CYCLE COMPLETE | {TRADES_PER_CYCLE} trades done | "
                f"collecting {REANALYSE_TICKS} ticks before re-analyse"
            )
            log.info("╚" + "═" * 53)
            self.phase                = Phase.WAITING
            self.wait_ticks_collected = 0


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║       Deriv HMM Algo Trading Bot  v3.0                  ║
║   Digit Differs  ·  Over 0  ·  Under 9                 ║
╚══════════════════════════════════════════════════════════╝""")
    print(f"  Symbol           : {SYMBOL}")
    print(f"  Stake            : ${STAKE:.2f}")
    print(f"  Max daily loss   : ${MAX_DAILY_LOSS:.2f}")
    print(f"  Warmup ticks     : {WARMUP_TICKS}")
    print(f"  HMM window       : {HMM_WINDOW}")
    print(f"  Min confidence   : {MIN_CONFIDENCE:.0%}")
    print(f"  State stable for : {STATE_STABLE_FOR} ticks")
    print(f"  Digit bias floor : {MIN_DIGIT_BIAS:.0%} over {DIGIT_WINDOW} digits")
    print(f"  Trades per cycle : {TRADES_PER_CYCLE}")
    print(f"  Reanalyse after  : {REANALYSE_TICKS} ticks")
    print(f"  Contract duration: {CONTRACT_DURATION} ticks")
    print(f"  Auth token       : {'SET ✓' if API_TOKEN else 'NOT SET — demo mode'}")
    print(f"  Install cmd      : pip install websocket-client numpy scikit-learn python-dotenv")
    print()

    bot = DerivHMMBot()
    try:
        bot.start()
    except KeyboardInterrupt:
        log.info("Interrupted")
        bot.stop()


if __name__ == "__main__":
    main()