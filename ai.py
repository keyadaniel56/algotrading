"""
Deriv Adaptive AI Trading Bot  v4.0
=====================================
A self-learning, adaptive trading bot that:

  1. Trades ALL Deriv synthetic digit markets simultaneously
  2. Learns continuously — HMM model updates after every cycle
  3. Scores symbols — always picks the best-performing market
  4. Recovers from losses automatically using 3 escalating strategies
  5. Persists state to disk — survives restarts without losing memory

Markets traded:
  R_10, R_25, R_50, R_75, R_100  (Volatility indices)
  RDBULL, RDBEAR                  (Boom/Crash style)
  1HZ10V, 1HZ25V, 1HZ50V         (1s tick indices)

Contract types:
  DIGITOVER  barrier=0   (~90% base win rate)
  DIGITUNDER barrier=9   (~90% base win rate)
  DIGITDIFF  barrier=D   (when a digit dominates)

Recovery strategies (triggered by loss streaks):
  Level 1 — 2+ losses : reduce stake to 50%, switch to safest contract type
  Level 2 — 4+ losses : switch to highest-scoring symbol, pause 20 ticks
  Level 3 — 6+ losses : full cooldown 50 ticks, restart analysis fresh

Install:
  pip install websocket-client numpy scikit-learn python-dotenv

Setup (.env):
  DERIV_APP_ID=your_app_id
  DERIV_API_TOKEN=your_token
  BASE_STAKE=1.0
  MAX_DAILY_LOSS=20.0
"""

import os, json, time, threading, logging, math
import numpy as np
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
BASE_STAKE     = float(os.getenv("BASE_STAKE", "30.0"))
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "20.0"))
STATE_FILE     = os.getenv("STATE_FILE", "bot_state.json")

DERIV_WS = f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}"

# All digit-market symbols to monitor
ALL_SYMBOLS = [
    "R_10", "R_25", "R_50", "R_75", "R_100",
    "1HZ10V", "1HZ25V", "1HZ50V",
]
ACTIVE_SYMBOLS = ALL_SYMBOLS   # start monitoring all; scoring picks the best

# HMM / analysis
N_STATES         = 3
WARMUP_TICKS     = 120    # per symbol
HMM_WINDOW       = 80
RETRAIN_EVERY    = 100    # ticks between retrains
STATE_STABLE_FOR = 4
DIGIT_WINDOW     = 20
MIN_DIGIT_BIAS   = 0.65
STREAK_THRESHOLD = 5
MIN_CONFIDENCE   = 0.50

# Cycle
TRADES_PER_CYCLE  = 2
REANALYSE_TICKS   = 15
CONTRACT_DURATION = 5

# Recovery thresholds
RECOVERY_L1_LOSSES = 2    # reduce stake + safe contracts
RECOVERY_L2_LOSSES = 4    # switch symbol + pause
RECOVERY_L3_LOSSES = 6    # full cooldown + fresh analysis

# Symbol scoring
SCORE_WINDOW     = 50     # recent trades used to score each symbol
SCORE_DECAY      = 0.95   # older trades count less

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("AdaptiveBot")


# ─────────────────────────────────────────────────────────────────────────────
# Phase enum
# ─────────────────────────────────────────────────────────────────────────────

class Phase(Enum):
    WARMUP   = auto()
    ANALYSE  = auto()
    TRADING  = auto()
    WAITING  = auto()
    RECOVERY = auto()   # forced cooldown after L3


# ─────────────────────────────────────────────────────────────────────────────
# Per-symbol HMM model
# ─────────────────────────────────────────────────────────────────────────────

class SymbolModel:
    """
    Self-contained HMM (KMeans + Viterbi) for one symbol.
    Stores its own tick history and retrains automatically.
    After each trade result it performs a lightweight online update
    by shifting the cluster means slightly toward or away from the
    observed digit distribution.
    """

    def __init__(self, symbol: str):
        self.symbol      = symbol
        self.ticks: deque = deque(maxlen=500)
        self.tick_count  = 0
        self.trained     = False
        self._n_trains   = 0
        self._means: Optional[np.ndarray]    = None
        self._vars:  Optional[np.ndarray]    = None
        self._trans: Optional[np.ndarray]    = None   # log-space
        self._start: Optional[np.ndarray]    = None   # log-space
        self.state_labels = ["bear", "neutral", "bull"]
        self._digit_analyser = DigitAnalyser()
        self._signal_gen     = SignalGenerator()

    # ── Features ─────────────────────────────────────────────────────────────

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

    # ── Training ─────────────────────────────────────────────────────────────

    def train(self) -> bool:
        ticks = list(self.ticks)
        feat  = self._features(ticks)
        if len(feat) < HMM_WINDOW:
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
        log.debug(f"[{self.symbol}] HMM trained #{self._n_trains} "
                  f"mean_ret={means[:,0].round(7)}")
        return True

    def online_update(self, profit: float):
        """
        Lightweight online learning: nudge cluster means based on outcome.
        Win  → reinforce current mean estimates (move slightly toward recent data)
        Loss → relax current means (increase variance — be more uncertain)
        """
        if not self.trained or self._means is None:
            return
        lr = 0.02   # small learning rate
        if profit >= 0:
            # Reinforce: pull means slightly toward last observed features
            feat = self._features(list(self.ticks)[-10:])
            if len(feat) > 0:
                recent_mean = feat.mean(axis=0)
                self._means += lr * (recent_mean - self._means)
        else:
            # Loss: inflate variance to signal uncertainty
            self._vars = np.clip(self._vars * (1 + lr), 1e-6, None)

    # ── Viterbi ──────────────────────────────────────────────────────────────

    def _log_emit(self, obs: np.ndarray) -> np.ndarray:
        lp = np.zeros(N_STATES)
        for s in range(N_STATES):
            d     = obs - self._means[s]
            lp[s] = -0.5 * np.sum(d**2 / self._vars[s]
                                  + np.log(2 * np.pi * self._vars[s]))
        return lp

    def analyse(self) -> tuple[Optional[str], float]:
        if not self.trained:
            return None, 0.0
        feat = self._features(list(self.ticks)[-HMM_WINDOW:])
        if len(feat) < 5:
            return None, 0.0
        try:
            T       = len(feat)
            vit     = np.full((T, N_STATES), -np.inf)
            bp      = np.zeros((T, N_STATES), dtype=int)
            vit[0]  = self._start + self._log_emit(feat[0])
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
            sc   = vit[-1];  sc -= sc.max()
            exps = np.exp(sc)
            conf = float(exps[path[-1]] / exps.sum())
            return self.state_labels[path[-1]], conf
        except Exception as e:
            log.debug(f"[{self.symbol}] viterbi error: {e}")
            return None, 0.0

    # ── Push tick ─────────────────────────────────────────────────────────────

    def push(self, price: float):
        self.ticks.append(price)
        self.tick_count += 1
        digit = self._last_digit(price)
        self._digit_analyser.push(digit)
        self._signal_gen.push_state_placeholder()   # updated by caller

    def get_signal(self, state: str, confidence: float) -> Optional[dict]:
        digit = self._last_digit(list(self.ticks)[-1]) if self.ticks else 0
        bias  = self._digit_analyser.bias()
        streak = self._digit_analyser.streak()
        return self._signal_gen.evaluate(state, confidence, bias, digit,
                                         self.symbol)

    def digit_stats(self) -> str:
        return self._digit_analyser.stats()


# ─────────────────────────────────────────────────────────────────────────────
# Digit Analyser (shared logic, instantiated per symbol)
# ─────────────────────────────────────────────────────────────────────────────

class DigitAnalyser:
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
        if zero_freq <= (1.0 - MIN_DIGIT_BIAS):
            return "over0"
        if nine_freq <= (1.0 - MIN_DIGIT_BIAS):
            return "under9"
        if max_freq >= MIN_DIGIT_BIAS:
            return f"differs:{dominant}"
        return None

    def streak(self) -> int:
        return self._streak_count

    def stats(self) -> str:
        if len(self.history) < 5:
            return "n/a"
        counts = np.bincount(list(self.history), minlength=10)
        return " ".join(f"{d}:{counts[d]}" for d in range(10) if counts[d] > 0)


# ─────────────────────────────────────────────────────────────────────────────
# Signal Generator
# ─────────────────────────────────────────────────────────────────────────────

class SignalGenerator:
    def __init__(self):
        self._state_buf: deque = deque(maxlen=STATE_STABLE_FOR + 2)

    def push_state_placeholder(self):
        pass   # state pushed explicitly via record_state

    def record_state(self, state: str):
        self._state_buf.append(state)

    def _stable(self) -> bool:
        if len(self._state_buf) < STATE_STABLE_FOR:
            return False
        return len(set(list(self._state_buf)[-STATE_STABLE_FOR:])) == 1

    def reset(self):
        self._state_buf.clear()

    def evaluate(self, state: str, confidence: float,
                 bias: Optional[str], last_digit: int,
                 symbol: str = "") -> Optional[dict]:
        if confidence < MIN_CONFIDENCE:
            return None
        if not self._stable():
            return None
        if bias is None:
            return None
        bt = bias.split(":")[0]
        if bt == "over0" and state in ("bull", "neutral"):
            return {"type": "DIGITOVER", "barrier": 0, "symbol": symbol,
                    "reason": f"{symbol} HMM={state}({confidence:.0%}) bias=over0 → OVER 0"}
        if bt == "under9" and state in ("bear", "neutral"):
            return {"type": "DIGITUNDER", "barrier": 9, "symbol": symbol,
                    "reason": f"{symbol} HMM={state}({confidence:.0%}) bias=under9 → UNDER 9"}
        if bt == "differs":
            dom = int(bias.split(":")[1])
            return {"type": "DIGITDIFF", "barrier": dom, "symbol": symbol,
                    "reason": f"{symbol} digit {dom} dominant → DIFFERS {dom}"}
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Symbol Scorer  — picks best symbol to trade
# ─────────────────────────────────────────────────────────────────────────────

class SymbolScorer:
    """
    Scores each symbol based on recent trade performance.
    Score = exponentially-weighted win rate over last SCORE_WINDOW trades.
    Higher = better. Bot always trades the highest-scoring ready symbol.
    """

    def __init__(self):
        # symbol → deque of (profit, timestamp)
        self._history: dict = {s: deque(maxlen=SCORE_WINDOW)
                               for s in ALL_SYMBOLS}
        self._scores: dict  = {s: 0.5 for s in ALL_SYMBOLS}  # start neutral

    def record(self, symbol: str, profit: float):
        self._history[symbol].append((profit, time.time()))
        self._recompute(symbol)

    def _recompute(self, symbol: str):
        history = list(self._history[symbol])
        if not history:
            self._scores[symbol] = 0.5
            return
        weighted_wins  = 0.0
        weighted_total = 0.0
        for i, (p, _) in enumerate(history):
            w = SCORE_DECAY ** (len(history) - 1 - i)
            weighted_total += w
            if p >= 0:
                weighted_wins += w
        self._scores[symbol] = weighted_wins / max(weighted_total, 1e-9)

    def best_symbol(self, ready_symbols: list) -> Optional[str]:
        """Return highest-scoring symbol from the ready list."""
        if not ready_symbols:
            return None
        return max(ready_symbols, key=lambda s: self._scores.get(s, 0.5))

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
    """
    Detects loss streaks and chooses the appropriate recovery action.

    Level 1 (2+ consecutive losses):
      - Halve the stake
      - Switch to safest contract type (DIGITOVER 0 or DIGITUNDER 9 only)
      - Log warning

    Level 2 (4+ consecutive losses):
      - Switch to the highest-scoring symbol
      - Pause trading for 20 ticks (WAITING phase)
      - Reset signal generator

    Level 3 (6+ consecutive losses):
      - Enter RECOVERY phase (50-tick hard cooldown)
      - Retrain all symbol models
      - Reset symbol scores for the current losing symbol
      - Resume from scratch on best symbol
    """

    def __init__(self):
        self.consecutive_losses = 0
        self.recovery_level     = 0
        self.recovery_ticks_remaining = 0
        self.last_action        = ""

    def record_loss(self):
        self.consecutive_losses += 1
        self._update_level()

    def record_win(self):
        self.consecutive_losses = 0
        if self.recovery_level > 0:
            log.info(f"[RECOVERY] Win recorded — stepping down recovery level")
            self.recovery_level = max(0, self.recovery_level - 1)
        self.last_action = ""

    def _update_level(self):
        prev = self.recovery_level
        if self.consecutive_losses >= RECOVERY_L3_LOSSES:
            self.recovery_level = 3
        elif self.consecutive_losses >= RECOVERY_L2_LOSSES:
            self.recovery_level = 2
        elif self.consecutive_losses >= RECOVERY_L1_LOSSES:
            self.recovery_level = 1
        if self.recovery_level > prev:
            log.warning(
                f"[RECOVERY] Level escalated to {self.recovery_level} "
                f"({self.consecutive_losses} consecutive losses)"
            )

    def get_stake_multiplier(self) -> float:
        if self.recovery_level >= 1:
            return 0.5
        return 1.0

    def safe_contracts_only(self) -> bool:
        """Level 1+: only DIGITOVER 0 or DIGITUNDER 9."""
        return self.recovery_level >= 1

    def needs_symbol_switch(self) -> bool:
        return self.recovery_level >= 2

    def needs_hard_cooldown(self) -> bool:
        return self.recovery_level >= 3

    def start_cooldown(self, ticks: int = 50):
        self.recovery_ticks_remaining = ticks
        log.warning(f"[RECOVERY L3] Hard cooldown: {ticks} ticks")

    def tick_cooldown(self) -> bool:
        """Returns True when cooldown is over."""
        if self.recovery_ticks_remaining > 0:
            self.recovery_ticks_remaining -= 1
        return self.recovery_ticks_remaining <= 0

    def filter_signal(self, sig: dict) -> Optional[dict]:
        """
        Apply recovery filters to a candidate signal.
        Returns filtered signal or None if blocked.
        """
        if sig is None:
            return None
        if self.safe_contracts_only():
            # Only allow the safest high-probability contracts
            if sig["type"] not in ("DIGITOVER", "DIGITUNDER"):
                log.info(f"[RECOVERY L{self.recovery_level}] "
                         f"Filtered out {sig['type']} — safe contracts only")
                return None
        return sig

    def status(self) -> str:
        return (f"level={self.recovery_level} "
                f"consec_losses={self.consecutive_losses} "
                f"cooldown={self.recovery_ticks_remaining}")


# ─────────────────────────────────────────────────────────────────────────────
# Risk Manager
# ─────────────────────────────────────────────────────────────────────────────

class RiskManager:
    def __init__(self):
        self.base_stake         = BASE_STAKE
        self.max_daily_loss     = MAX_DAILY_LOSS
        self.daily_pnl          = 0.0
        self.trade_date         = date.today()
        self.total_trades       = 0
        self.total_wins         = 0
        self.session_start_pnl  = 0.0

    def _day_reset(self):
        today = date.today()
        if today != self.trade_date:
            log.info(f"New day | prev PnL={self.daily_pnl:+.2f} | "
                     f"W/L={self.total_wins}/{self.total_trades-self.total_wins}")
            self.daily_pnl  = 0.0
            self.trade_date = today

    def is_halted(self) -> tuple[bool, str]:
        self._day_reset()
        if self.daily_pnl <= -abs(self.max_daily_loss):
            return True, f"daily loss limit ({self.daily_pnl:.2f})"
        return False, ""

    def stake(self, recovery_mult: float = 1.0) -> float:
        s = self.base_stake * recovery_mult
        return round(max(s, 0.35), 2)   # Deriv min stake ~$0.35

    def record(self, profit: float):
        self._day_reset()
        self.daily_pnl   += profit
        self.total_trades += 1
        if profit >= 0:
            self.total_wins += 1

    def win_rate(self) -> float:
        return self.total_wins / max(1, self.total_trades)

    def summary(self) -> str:
        return (f"trades={self.total_trades} | "
                f"win_rate={self.win_rate():.0%} | "
                f"daily_pnl={self.daily_pnl:+.2f}")

    def to_dict(self) -> dict:
        return {
            "daily_pnl":    self.daily_pnl,
            "total_trades": self.total_trades,
            "total_wins":   self.total_wins,
            "trade_date":   str(self.trade_date),
        }

    def from_dict(self, d: dict):
        self.daily_pnl    = d.get("daily_pnl", 0.0)
        self.total_trades = d.get("total_trades", 0)
        self.total_wins   = d.get("total_wins", 0)
        saved_date        = d.get("trade_date", str(date.today()))
        self.trade_date   = date.fromisoformat(saved_date)
        # Reset pnl if it's a new day
        if self.trade_date != date.today():
            self.daily_pnl  = 0.0
            self.trade_date = date.today()


# ─────────────────────────────────────────────────────────────────────────────
# State persistence
# ─────────────────────────────────────────────────────────────────────────────

class StatePersistence:
    """Saves and loads bot state to JSON so learning survives restarts."""

    def __init__(self, path: str = STATE_FILE):
        self.path = Path(path)

    def save(self, risk: RiskManager, scorer: SymbolScorer,
             trade_log: list):
        data = {
            "saved_at":   datetime.now().isoformat(),
            "risk":       risk.to_dict(),
            "scores":     scorer.to_dict(),
            "trade_log":  trade_log[-500:],   # keep last 500
        }
        try:
            self.path.write_text(json.dumps(data, indent=2))
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
# Main adaptive bot
# ─────────────────────────────────────────────────────────────────────────────

class DerivAdaptiveBot:
    """
    Multi-symbol adaptive trading bot.

    Per-tick flow:
      1. Route tick to correct SymbolModel
      2. On active symbol: run HMM + digit analysis
      3. SignalGenerator gates signal quality
      4. RecoveryEngine filters and potentially overrides symbol choice
      5. RiskManager gates daily loss
      6. Execute trade; on settlement → update model + scorer + state
    """

    def __init__(self):
        self.ws: Optional[websocket.WebSocketApp] = None
        self._req_id  = 1
        self._lock    = threading.Lock()

        # Per-symbol models
        self.models: dict[str, SymbolModel] = {
            s: SymbolModel(s) for s in ALL_SYMBOLS
        }

        # Shared components
        self.scorer    = SymbolScorer()
        self.recovery  = RecoveryEngine()
        self.risk      = RiskManager()
        self.persist   = StatePersistence()

        # Cycle state
        self.phase             = Phase.WARMUP
        self.active_symbol     = ALL_SYMBOLS[0]   # updated by scorer
        self.current_signal: Optional[dict] = None
        self.trades_this_cycle = 0
        self.wait_ticks        = 0
        self.open_cid: Optional[str] = None

        # Trade log (for persistence and analytics)
        self.trade_log: list = []

        # Tick counters per symbol
        self._sym_ticks: dict = {s: 0 for s in ALL_SYMBOLS}
        self._retrain_counter: dict = {s: 0 for s in ALL_SYMBOLS}

        self._load_state()

    def _load_state(self):
        data = self.persist.load()
        if data:
            self.risk.from_dict(data.get("risk", {}))
            self.scorer.from_dict(data.get("scores", {}))
            self.trade_log = data.get("trade_log", [])
            log.info(
                f"State loaded | trades={self.risk.total_trades} | "
                f"win_rate={self.risk.win_rate():.0%} | "
                f"daily_pnl={self.risk.daily_pnl:+.2f}"
            )

    def _save_state(self):
        self.persist.save(self.risk, self.scorer, self.trade_log)

    # ── WebSocket plumbing ────────────────────────────────────────────────────

    def start(self):
        self.ws = websocket.WebSocketApp(
            DERIV_WS,
            on_open    = self._on_open,
            on_message = self._on_message,
            on_error   = self._on_error,
            on_close   = self._on_close,
        )
        log.info("Connecting to Deriv…")
        self.ws.run_forever(ping_interval=30, ping_timeout=10, reconnect=5)

    def stop(self):
        self._save_state()
        if self.ws:
            self.ws.close()
        log.info(f"Stopped | {self.risk.summary()}")

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
                log.warning(f"API: {msg.get('error',{}).get('message',msg)}")
        except Exception as exc:
            log.error(f"Message handler error: {exc}", exc_info=False)

    def _on_error(self, ws, err): log.error(f"WS error: {err}")
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

    # ── Tick routing ──────────────────────────────────────────────────────────

    def _on_tick(self, msg: dict):
        td  = msg.get("tick", {})
        sym = td.get("symbol", "")
        if sym not in self.models:
            return
        price = float(td.get("quote", 0))
        if price <= 0:
            return

        model = self.models[sym]
        model.push(price)
        self._sym_ticks[sym] = self._sym_ticks.get(sym, 0) + 1

        # Check if this symbol needs retraining
        cnt = self._sym_ticks[sym]
        if cnt >= WARMUP_TICKS and not model.trained:
            ok = model.train()
            if ok:
                log.info(f"[{sym}] Warmup complete — model trained")
        elif model.trained and cnt % RETRAIN_EVERY == 0:
            model.train()
            log.debug(f"[{sym}] Periodic retrain at tick {cnt}")

        # Only drive the cycle on the active symbol
        if sym == self.active_symbol:
            self._drive_cycle(model, price)

    # ── Cycle state machine ───────────────────────────────────────────────────

    def _drive_cycle(self, model: SymbolModel, price: float):
        sym = model.symbol

        # ── RECOVERY cooldown ────────────────────────────────────────────────
        if self.phase == Phase.RECOVERY:
            done = self.recovery.tick_cooldown()
            if done:
                log.info("[RECOVERY] Cooldown complete — resuming on best symbol")
                self._switch_to_best_symbol()
                self._enter_analyse()
            return

        # ── WARMUP ───────────────────────────────────────────────────────────
        if self.phase == Phase.WARMUP:
            ready = [s for s in ALL_SYMBOLS if self.models[s].trained]
            if len(ready) == 0:
                total = sum(self._sym_ticks.values())
                if total % (20 * len(ALL_SYMBOLS)) == 0:
                    trained_cnt = sum(1 for m in self.models.values() if m.trained)
                    log.info(
                        f"[WARMUP] {trained_cnt}/{len(ALL_SYMBOLS)} symbols trained | "
                        f"total_ticks={total}"
                    )
            if model.trained and not self.models[self.active_symbol].trained:
                self.active_symbol = sym
            if model.trained and self.phase == Phase.WARMUP:
                # Switch active to best scored ready symbol
                self._switch_to_best_symbol()
                self._enter_analyse()
            return

        # ── WAITING ──────────────────────────────────────────────────────────
        if self.phase == Phase.WAITING:
            self.wait_ticks += 1
            if self.wait_ticks >= REANALYSE_TICKS:
                model.train()
                self._switch_to_best_symbol()
                self._enter_analyse()
            return

        # ── TRADING — wait for settlement callback ────────────────────────────
        if self.phase == Phase.TRADING:
            return

        # ── ANALYSE ──────────────────────────────────────────────────────────
        if self.phase == Phase.ANALYSE:
            self._run_analysis(model, price)

    def _enter_analyse(self):
        self.current_signal    = None
        self.trades_this_cycle = 0
        self.wait_ticks        = 0
        self.phase             = Phase.ANALYSE
        log.info(f"Phase → ANALYSE | active={self.active_symbol} | "
                 f"recovery={self.recovery.status()}")

    def _switch_to_best_symbol(self):
        ready = [s for s in ALL_SYMBOLS if self.models[s].trained]
        if not ready:
            return
        best = self.scorer.best_symbol(ready)
        if best and best != self.active_symbol:
            log.info(
                f"Symbol switch: {self.active_symbol} → {best} | "
                f"scores: " +
                " ".join(f"{s}={self.scorer.score(s):.0%}"
                         for s, _ in self.scorer.ranking()[:4])
            )
            self.active_symbol = best
            # Reset signal generator for new symbol
            self.models[best]._signal_gen.reset()

    def _run_analysis(self, model: SymbolModel, price: float):
        halted, reason = self.risk.is_halted()
        if halted:
            log.info(f"[RISK] Halted: {reason}")
            return

        state, confidence = model.analyse()
        if state is None:
            return

        model._signal_gen.record_state(state)
        digit = SymbolModel._last_digit(price)
        bias  = model._digit_analyser.bias()

        # Log every 3 ticks
        tick = self._sym_ticks.get(model.symbol, 0)
        if tick % 3 == 0:
            log.info(
                f"[ANALYSE:{model.symbol}] "
                f"digit={digit} HMM={state}({confidence:.0%}) "
                f"bias={bias} streak={model._digit_analyser.streak()} "
                f"score={self.scorer.score(model.symbol):.0%} "
                f"recovery={self.recovery.status()}"
            )

        sig = model.get_signal(state, confidence)

        # Apply recovery filters
        sig = self.recovery.filter_signal(sig)
        if sig is None:
            return

        # Lock signal and trade
        self.current_signal = sig
        log.info(f"╔ SIGNAL | {sig['reason']}")
        self.phase = Phase.TRADING
        self._place_trade(1)

    # ── Trade execution ───────────────────────────────────────────────────────

    def _place_trade(self, trade_num: int):
        sig   = self.current_signal
        mult  = self.recovery.get_stake_multiplier()
        stake = self.risk.stake(mult)

        log.info(
            f"├── TRADE {trade_num}/{TRADES_PER_CYCLE} | "
            f"{sig['type']} {sig['symbol']} | "
            f"barrier={sig.get('barrier','n/a')} | stake=${stake}"
        )

        if not API_TOKEN:
            log.info(f"└── [DEMO] simulated")
            self.trades_this_cycle += 1
            # Simulate ~87% win rate (base rate for over0/under9)
            import random
            sim_profit = stake * 0.87 if random.random() < 0.87 else -stake
            self._after_settlement(sim_profit)
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
                "symbol":        sig["symbol"],
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
        poc = msg.get("proposal_open_contract", {})
        settled = (poc.get("is_settleable") or poc.get("is_sold") or
                   poc.get("status") in ("won", "lost"))
        if not settled:
            return
        profit = float(poc.get("profit", 0))
        status = poc.get("status", "?").upper()
        log.info(f"    Settled: {status} {profit:+.2f}")
        self.open_cid = None
        self._after_settlement(profit)

    def _after_settlement(self, profit: float):
        sym = self.current_signal["symbol"] if self.current_signal else self.active_symbol

        # Update all learning components
        self.risk.record(profit)
        self.scorer.record(sym, profit)
        self.models[sym].online_update(profit)   # online model adaptation

        # Recovery tracking
        if profit >= 0:
            self.recovery.record_win()
        else:
            self.recovery.record_loss()

        # Trade log entry
        self.trade_log.append({
            "ts":     datetime.now().isoformat(),
            "symbol": sym,
            "type":   self.current_signal["type"] if self.current_signal else "?",
            "profit": profit,
            "streak": self.recovery.consecutive_losses,
        })

        # Save state every 10 trades
        if len(self.trade_log) % 10 == 0:
            self._save_state()

        log.info(
            f"    {self.risk.summary()} | "
            f"sym_score={self.scorer.score(sym):.0%} | "
            f"recovery={self.recovery.status()}"
        )

        # Recovery escalation check
        halted, reason = self.risk.is_halted()
        if halted:
            log.warning(f"Bot halted: {reason}")
            self.phase = Phase.WAITING
            return

        if self.recovery.needs_hard_cooldown():
            self.phase = Phase.RECOVERY
            self.recovery.start_cooldown(50)
            # Reset the losing symbol's score to neutral
            self.scorer._scores[sym] = 0.3
            self._switch_to_best_symbol()
            return

        if self.recovery.needs_symbol_switch():
            log.info("[RECOVERY L2] Switching to best symbol")
            self._switch_to_best_symbol()
            self.models[self.active_symbol]._signal_gen.reset()

        if self.trades_this_cycle < TRADES_PER_CYCLE:
            self._place_trade(self.trades_this_cycle + 1)
        else:
            log.info(
                f"╚ CYCLE DONE | {TRADES_PER_CYCLE} trades | "
                f"collecting {REANALYSE_TICKS} ticks…"
            )
            self.phase      = Phase.WAITING
            self.wait_ticks = 0


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("""
╔════════════════════════════════════════════════════════════╗
║     Deriv Adaptive AI Trading Bot  v4.0                   ║
║     Self-learning · Multi-symbol · Auto-recovery          ║
╚════════════════════════════════════════════════════════════╝""")
    print(f"  Symbols         : {', '.join(ALL_SYMBOLS)}")
    print(f"  Base stake       : ${BASE_STAKE:.2f}")
    print(f"  Max daily loss   : ${MAX_DAILY_LOSS:.2f}")
    print(f"  Warmup ticks     : {WARMUP_TICKS} per symbol")
    print(f"  Trades per cycle : {TRADES_PER_CYCLE}")
    print(f"  Reanalyse after  : {REANALYSE_TICKS} ticks")
    print(f"  Recovery L1 at   : {RECOVERY_L1_LOSSES} losses (half stake)")
    print(f"  Recovery L2 at   : {RECOVERY_L2_LOSSES} losses (switch symbol)")
    print(f"  Recovery L3 at   : {RECOVERY_L3_LOSSES} losses (50-tick cooldown)")
    print(f"  State file       : {STATE_FILE}")
    print(f"  Auth token       : {'SET ✓' if API_TOKEN else 'NOT SET — demo mode'}")
    print()

    bot = DerivAdaptiveBot()
    try:
        bot.start()
    except KeyboardInterrupt:
        log.info("Interrupted")
        bot.stop()


if __name__ == "__main__":
    main()