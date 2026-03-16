# deriv_digit_ai_bot_v2.py
# Strategy : Over 4 (digits 5-9) and Under 5 (digits 0-4) ONLY
# Martingale: DISABLED – fixed stake always
# Recovery  : DISABLED – every trade does fresh market analysis
# Requirements: pip install websocket-client numpy

import json
import math
import random
import signal
import threading
import time
from collections import deque
from typing import Optional, Tuple, List

import numpy as np
import websocket

# ────────────────────────────────────────────────
#               CONFIGURABLE PARAMETERS
# ────────────────────────────────────────────────
APP_ID        = "1089"
AUTH_TOKEN    = "WFabi7aeCbFjgvp"
SYMBOL        = "R_100"
DURATION      = 1
DURATION_UNIT = "t"
CURRENCY      = "USD"

INITIAL_STAKE = 1.0          # fixed stake – never changes

MAX_CONSECUTIVE_LOSSES = 30
STOP_LOSS              = -100.0
TAKE_PROFIT            = 100.0

SEQ_LEN              = 20
WINDOW_SIZE          = 150
HIDDEN_SIZE          = 48
FEATURE_SIZE         = 28
LEARNING_RATE        = 0.006
LR_DECAY             = 0.9999
TRAIN_EPOCHS_INITIAL = 80
TRAIN_EPOCHS_UPDATE  = 10
UPDATE_MODEL_EVERY   = 8

# ── Entry thresholds ──────────────────────────────
# Over 4  : wins on digits 5,6,7,8,9  → base probability = 50%
# Under 5 : wins on digits 0,1,2,3,4  → base probability = 50%
# Because both sides always sum to 1.0, a low threshold fires randomly.
# We require a HARD minimum model edge + pattern confirmation before entry.
PROB_THRESHOLD_OVER4  = 0.63   # p(digits 5-9) must exceed this (hard minimum)
PROB_THRESHOLD_UNDER5 = 0.63   # p(digits 0-4) must exceed this (hard minimum)

# Pattern confirmation is REQUIRED – model edge alone is not enough.
# At least one pattern must agree with the trade direction before entry.
REQUIRE_PATTERN_CONFIRMATION = True

# Minimum pattern confidence required when confirming a direction
MIN_PATTERN_CONFIDENCE = 0.55

# How many of the last N ticks must agree with the direction (trend filter)
TREND_LOOKBACK         = 10    # look at last 10 ticks
TREND_MIN_RATIO_OVER4  = 0.60  # ≥60% of last 10 digits must be 5-9 to confirm Over 4
TREND_MIN_RATIO_UNDER5 = 0.60  # ≥60% of last 10 digits must be 0-4 to confirm Under 5

# Minimum gap between trades (ticks) to prevent overtrading
MIN_TICKS_BETWEEN_TRADES = 3

WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"

# ────────────────────────────────────────────────
#                 GLOBAL STATE
# ────────────────────────────────────────────────
last_digits        = deque(maxlen=WINDOW_SIZE)
total_profit       = 0.0
consecutive_losses = 0
global_step        = 0
current_lr         = LEARNING_RATE

W1xh = W1hh = b1h = None
W2xh = W2hh = b2h = None
Why  = by   = None

pattern_cache     = {}
trade_in_progress = False
ticks_since_trade = 0

ws      = None
running = True
lock    = threading.Lock()


# ────────────────────────────────────────────────
#     LOGGING
# ────────────────────────────────────────────────

def think(msg): print(f"  🧠 {msg}")
def info(msg):  print(f"  ℹ️  {msg}")
def warn(msg):  print(f"  ⚠️  {msg}")
def tlog(msg):  print(f"  💰 {msg}")
def pat(msg):   print(f"  🔍 {msg}")

def section(title):
    bar = "─" * 80
    print(f"\n{bar}\n  {title}\n{bar}")


# ────────────────────────────────────────────────
#     FEATURE ENGINEERING  (28 features)
# ────────────────────────────────────────────────

def build_features(digit: int, history: list) -> np.ndarray:
    vec = np.zeros(FEATURE_SIZE, dtype=np.float32)
    vec[digit] = 1.0                                    # 0-9  one-hot
    vec[10] = digit / 9.0                               # 10   normalised value
    vec[11] = 1.0 if digit % 2 == 0 else 0.0           # 11   even flag
    vec[12] = 1.0 if digit <= 4 else 0.0               # 12   low half
    vec[13] = 1.0 if digit >= 5 else 0.0               # 13   high half

    n = len(history)

    # 14  streak length
    streak = 0
    for d in reversed(history):
        if d == digit: streak += 1
        else: break
    vec[14] = min(streak, 10) / 10.0

    # 15  ticks since last same digit
    last_same = n
    for i, d in enumerate(reversed(history)):
        if d == digit: last_same = i; break
    vec[15] = min(last_same, 10) / 10.0

    # 16  hi/lo alternation signal
    if n >= 2:
        a, b = history[-1], history[-2]
        vec[16] = 1.0 if ((a >= 5) != (b >= 5)) else 0.0

    # 17  delta from previous digit
    if n >= 1:
        vec[17] = (digit - history[-1]) / 9.0

    # 18-19  rolling mean / std (last 10)
    recent10 = history[-10:] if n >= 10 else history
    if recent10:
        vec[18] = np.mean(recent10) / 9.0
        vec[19] = np.std(recent10)  / 4.5

    # 20-22  short/long frequency + imbalance
    if n >= 20:
        sf = (history[-20:]).count(digit) / 20.0
        lh = history[-50:] if n >= 50 else history
        lf = lh.count(digit) / len(lh)
        vec[20] = sf; vec[21] = lf; vec[22] = sf - lf

    # 23-24  autocorrelation lag 1 and 2
    if n >= 4:
        arr = np.array(history[-20:] if n >= 20 else history, dtype=float)
        if len(arr) > 2:
            c1 = np.corrcoef(arr[:-1], arr[1:])[0, 1]
            vec[23] = np.sign(c1) if not np.isnan(c1) else 0.0
        if len(arr) > 3:
            c2 = np.corrcoef(arr[:-2], arr[2:])[0, 1]
            vec[24] = np.sign(c2) if not np.isnan(c2) else 0.0

    # 25-26  positional cycle (sin/cos)
    pos = len(history) % 10
    vec[25] = math.sin(2 * math.pi * pos / 10)
    vec[26] = math.cos(2 * math.pi * pos / 10)

    # 27  5-tick momentum
    if n >= 5:
        slope = np.polyfit(range(5), history[-5:], 1)[0]
        vec[27] = float(np.tanh(slope))

    return vec


def build_sequence_matrix(history: list) -> np.ndarray:
    n   = len(history)
    seq = []
    start = max(0, n - SEQ_LEN)
    for i in range(start, n):
        seq.append(build_features(history[i], history[:i]))
    while len(seq) < SEQ_LEN:
        seq.insert(0, np.zeros(FEATURE_SIZE, dtype=np.float32))
    return np.array(seq, dtype=np.float32)


# ────────────────────────────────────────────────
#     PATTERN DETECTORS
# ────────────────────────────────────────────────

def detect_patterns(history: list) -> dict:
    p = {}
    n = len(history)
    if n < 10:
        return p

    # Streak
    streak_len = 1
    for i in range(n - 2, max(n - 8, -1), -1):
        if history[i] == history[-1]: streak_len += 1
        else: break
    if streak_len >= 2:
        p["streak"] = {
            "digit": history[-1], "length": streak_len,
            "confidence": min(streak_len / 5, 1.0),
            "signal": f"digit {history[-1]} repeating ({streak_len}x) – break expected"
        }

    # Hi/lo alternation
    window = history[-8:]
    alt_count = sum(1 for i in range(1, len(window)) if (window[i] >= 5) != (window[i-1] >= 5))
    alt_ratio = alt_count / (len(window) - 1)
    if alt_ratio >= 0.75:
        p["alternation"] = {
            "confidence": alt_ratio,
            "next_expected": "LOW (0-4)" if history[-1] >= 5 else "HIGH (5-9)",
            "signal": f"hi/lo alternation {alt_ratio:.0%} – next likely {'LOW' if history[-1]>=5 else 'HIGH'}"
        }

    # Even/odd alternation
    eo_count = sum(1 for i in range(1, len(window)) if (window[i] % 2) != (window[i-1] % 2))
    eo_ratio = eo_count / (len(window) - 1)
    if eo_ratio >= 0.75:
        p["even_odd_alternation"] = {
            "confidence": eo_ratio,
            "next_expected": "ODD" if history[-1] % 2 == 0 else "EVEN",
            "signal": f"even/odd alternation {eo_ratio:.0%}"
        }

    # Rising / falling run
    run_up = run_dn = 1
    for i in range(n-2, max(n-6, -1), -1):
        if history[i] < history[i+1]: run_up += 1
        else: break
    for i in range(n-2, max(n-6, -1), -1):
        if history[i] > history[i+1]: run_dn += 1
        else: break
    if run_up >= 3:
        p["rising_run"] = {"length": run_up, "confidence": min(run_up/5,1.0),
                           "signal": f"{run_up}-tick rising run – reversal likely"}
    if run_dn >= 3:
        p["falling_run"] = {"length": run_dn, "confidence": min(run_dn/5,1.0),
                            "signal": f"{run_dn}-tick falling run – reversal likely"}

    # Frequency bias (last 30)
    recent30 = history[-30:] if n >= 30 else history
    hf = sum(1 for d in recent30 if d >= 5) / len(recent30)
    lf = 1.0 - hf
    if hf >= 0.60:
        p["freq_bias_high"] = {"frequency": hf, "confidence": (hf-0.5)*2,
                               "signal": f"digits 5-9 = {hf:.0%} of last 30 (hot)"}
    elif lf >= 0.60:
        p["freq_bias_low"]  = {"frequency": lf, "confidence": (lf-0.5)*2,
                               "signal": f"digits 0-4 = {lf:.0%} of last 30 (hot)"}

    # Cold / hot digit (last 50)
    if n >= 50:
        cnt = np.bincount(history[-50:], minlength=10) / 50.0
        cd  = int(np.argmin(cnt)); hd = int(np.argmax(cnt))
        if 0.10 - cnt[cd] >= 0.05:
            p["cold_digit"] = {"digit": cd, "freq": cnt[cd],
                               "confidence": min((0.10-cnt[cd])/0.10, 1.0),
                               "signal": f"digit {cd} only {cnt[cd]:.0%} (cold – statistically due)"}
        if cnt[hd] - 0.10 >= 0.07:
            p["hot_digit"]  = {"digit": hd, "freq": cnt[hd],
                               "confidence": min((cnt[hd]-0.10)/0.15, 1.0),
                               "signal": f"digit {hd} = {cnt[hd]:.0%} (hot – likely to cool)"}

    # Mean reversion (last 20)
    if n >= 20:
        m = float(np.mean(history[-20:]))
        dev = m - 4.5
        if abs(dev) >= 1.5:
            p["mean_reversion"] = {"mean": m, "deviation": dev,
                                   "confidence": min(abs(dev)/3.0, 1.0),
                                   "signal": f"20-tick mean={m:.2f} (expect 4.5) – pressure {'DOWN' if dev>0 else 'UP'}"}

    # Cycle detector (period 3-6)
    if n >= 20:
        for period in range(3, 7):
            blk  = history[-period:]
            prev = history[-period*2:-period]
            if len(prev) == period:
                matches = sum(a == b for a, b in zip(blk, prev))
                cc = matches / period
                if cc >= 0.60:
                    p[f"cycle_{period}"] = {"period": period, "confidence": cc,
                                            "signal": f"{period}-tick repeating cycle ({cc:.0%} match)"}
                    break

    return p


def pattern_confidence_summary(patterns: dict) -> Tuple[float, str]:
    if not patterns:
        return 0.0, "no patterns"
    lines = [f"{d.get('signal', k)} [{d.get('confidence',0):.0%}]"
             for k, d in patterns.items()]
    mc = max(d.get("confidence", 0) for d in patterns.values())
    return mc, " | ".join(lines)


# ────────────────────────────────────────────────
#     TWO-LAYER RNN
# ────────────────────────────────────────────────

def init_model():
    global W1xh, W1hh, b1h, W2xh, W2hh, b2h, Why, by
    np.random.seed(42)
    def X(r, c): return np.random.randn(r, c) * np.sqrt(2.0/(r+c))
    W1xh = X(HIDDEN_SIZE, FEATURE_SIZE); W1hh = X(HIDDEN_SIZE, HIDDEN_SIZE); b1h = np.zeros((HIDDEN_SIZE,1))
    W2xh = X(HIDDEN_SIZE, HIDDEN_SIZE);  W2hh = X(HIDDEN_SIZE, HIDDEN_SIZE); b2h = np.zeros((HIDDEN_SIZE,1))
    Why  = X(10, HIDDEN_SIZE);           by   = np.zeros((10, 1))
    think(f"2-layer RNN ready: {FEATURE_SIZE} features → {HIDDEN_SIZE}×2 hidden → 10 output")


def forward(seq):
    h1 = np.zeros((HIDDEN_SIZE,1)); h2 = np.zeros((HIDDEN_SIZE,1))
    h1s=[]; h2s=[]; outs=[]
    for t in range(SEQ_LEN):
        x  = seq[t].reshape(-1,1)
        h1 = np.tanh(W1xh @ x  + W1hh @ h1 + b1h)
        h2 = np.tanh(W2xh @ h1 + W2hh @ h2 + b2h)
        h1s.append(h1.copy()); h2s.append(h2.copy())
        outs.append((Why @ h2 + by).copy())
    return h1s, h2s, outs


def predict_probs_from_history(history: list) -> Optional[np.ndarray]:
    if W1xh is None: return None
    _, _, outs = forward(build_sequence_matrix(history))
    lg = outs[-1]; e = np.exp(lg - np.max(lg)); return (e/e.sum()).flatten()


def train_on_sequence(history: list, epochs: int):
    global W1xh, W1hh, b1h, W2xh, W2hh, b2h, Why, by, global_step, current_lr
    if len(history) < SEQ_LEN + 1: return

    pairs = [(build_sequence_matrix(history[:i]), history[i])
             for i in range(SEQ_LEN, len(history))]

    def clip(g, lim=1.0):
        n = np.linalg.norm(g); return g*(lim/n) if n > lim else g

    for ep in range(epochs):
        tl = 0.0; random.shuffle(pairs)
        for seq, target in pairs:
            h1s, h2s, outs = forward(seq)
            lg = outs[-1]; e = np.exp(lg-np.max(lg)); pr = e/e.sum()
            tl += float(-np.log(pr[target]+1e-12))
            dl = pr.copy(); dl[target] -= 1.0
            dWhy = dl @ h2s[-1].T; dby = dl.reshape(-1,1); dh2 = Why.T @ dl
            dz2  = dh2 * (1 - h2s[-1]**2)
            dW2x = dz2 @ h1s[-1].T
            dW2h = dz2 @ (h2s[-2].T if len(h2s)>1 else np.zeros((1,HIDDEN_SIZE)))
            db2  = dz2; dh1 = W2xh.T @ dz2; dz1 = dh1*(1-h1s[-1]**2)
            dW1x = dz1 @ seq[-1].reshape(1,-1)
            dW1h = dz1 @ (h1s[-2].T if len(h1s)>1 else np.zeros((1,HIDDEN_SIZE)))
            db1  = dz1
            lr = current_lr
            Why  -= lr*clip(dWhy);  by   -= lr*clip(dby)
            W2xh -= lr*clip(dW2x);  W2hh -= lr*clip(dW2h); b2h -= lr*clip(db2)
            W1xh -= lr*clip(dW1x);  W1hh -= lr*clip(dW1h); b1h -= lr*clip(db1)
            current_lr = max(current_lr*LR_DECAY, 1e-5); global_step += 1
        if ep % 20 == 0:
            think(f"[epoch {ep:2d}] loss={tl/max(len(pairs),1):.4f}  lr={current_lr:.6f}")


# ────────────────────────────────────────────────
#     ANALYSIS DISPLAY
# ────────────────────────────────────────────────

def show_analysis(history, probs, patterns):
    section("MARKET ANALYSIS")
    think(f"Last digit: {history[-1]}   |   Buffer: {len(history)}/{WINDOW_SIZE}")

    cnt   = np.bincount(history[-50:] if len(history)>=50 else history, minlength=10)
    total = cnt.sum()
    hot   = set(np.argsort(cnt)[-3:].tolist())
    cold  = set(np.argsort(cnt)[:3].tolist())
    think("Digit frequency (last 50):")
    for i in range(10):
        bar = "█" * int(cnt[i]/total*40)
        dev = cnt[i]/total - 0.10
        tag = f"  ◄ HOT  +{dev:.0%}" if i in hot else (f"  ◄ COLD {dev:.0%}" if i in cold else "")
        print(f"       {i} : {bar:<40} {cnt[i]:3d} ({cnt[i]/total*100:4.1f}%){tag}")

    if patterns:
        pat("Detected patterns:")
        for name, data in patterns.items():
            c = data.get("confidence",0); bar = "●"*int(c*10)
            print(f"       {bar:<10} {c:.0%}  {data.get('signal', name)}")
    else:
        pat("No significant patterns this tick")

    if probs is not None:
        print()
        think("Model per-digit probabilities:")
        for i in range(10):
            bar = "▓" * int(probs[i]*100)
            print(f"       digit {i} : {bar:<20} {probs[i]:.3f}")
        print()
        p_over4  = float(np.sum(probs[5:]))
        p_under5 = float(np.sum(probs[:5]))
        think(f"  Over  4 (digits 5-9): {p_over4:.1%}   threshold > {PROB_THRESHOLD_OVER4:.0%}")
        think(f"  Under 5 (digits 0-4): {p_under5:.1%}   threshold > {PROB_THRESHOLD_UNDER5:.0%}")


# pattern_trade_hint removed – replaced by directional pattern gates
# (pattern_agrees_over4 / pattern_agrees_under5)


# ────────────────────────────────────────────────
#     TREND FILTER
# ────────────────────────────────────────────────

def trend_confirms_over4(history: list) -> Tuple[bool, float]:
    """
    Returns (confirmed, ratio).
    Checks that the recent TREND_LOOKBACK ticks lean toward high digits (5-9),
    i.e. the market is actually moving in the Over 4 direction right now.
    """
    recent = history[-TREND_LOOKBACK:]
    ratio  = sum(1 for d in recent if d >= 5) / len(recent)
    return ratio >= TREND_MIN_RATIO_OVER4, ratio

def trend_confirms_under5(history: list) -> Tuple[bool, float]:
    """
    Returns (confirmed, ratio).
    Checks that recent ticks lean toward low digits (0-4).
    """
    recent = history[-TREND_LOOKBACK:]
    ratio  = sum(1 for d in recent if d <= 4) / len(recent)
    return ratio >= TREND_MIN_RATIO_UNDER5, ratio


# ────────────────────────────────────────────────
#     PATTERN DIRECTION CHECK
# ────────────────────────────────────────────────

def pattern_agrees_over4(patterns: dict) -> Tuple[bool, float, str]:
    """
    Returns (agrees, best_confidence, reason).
    At least one strong pattern must point toward HIGH digits for Over 4.
    """
    best_c = 0.0; reason = ""
    # Alternation predicts HIGH next
    if "alternation" in patterns:
        p = patterns["alternation"]
        if "HIGH" in p.get("next_expected", "") and p["confidence"] >= MIN_PATTERN_CONFIDENCE:
            if p["confidence"] > best_c:
                best_c = p["confidence"]; reason = f"alternation→HIGH [{p['confidence']:.0%}]"
    # Low-frequency bias → expect high digits next
    if "freq_bias_low" in patterns:
        p = patterns["freq_bias_low"]
        if p["confidence"] >= MIN_PATTERN_CONFIDENCE and p["confidence"] > best_c:
            best_c = p["confidence"]; reason = f"lo-freq-bias→HIGH [{p['confidence']:.0%}]"
    # Mean below 4.5 → upward pressure
    if "mean_reversion" in patterns:
        p = patterns["mean_reversion"]
        if p["deviation"] < 0 and p["confidence"] >= MIN_PATTERN_CONFIDENCE and p["confidence"] > best_c:
            best_c = p["confidence"]; reason = f"mean-rev↑ [{p['confidence']:.0%}]"
    # Falling run → reversal to high digits
    if "falling_run" in patterns:
        p = patterns["falling_run"]
        if p["confidence"] >= MIN_PATTERN_CONFIDENCE and p["confidence"] > best_c:
            best_c = p["confidence"]; reason = f"falling-run reversal [{p['confidence']:.0%}]"

    return best_c >= MIN_PATTERN_CONFIDENCE, best_c, reason


def pattern_agrees_under5(patterns: dict) -> Tuple[bool, float, str]:
    """
    Returns (agrees, best_confidence, reason).
    At least one strong pattern must point toward LOW digits for Under 5.
    """
    best_c = 0.0; reason = ""
    # Alternation predicts LOW next
    if "alternation" in patterns:
        p = patterns["alternation"]
        if "LOW" in p.get("next_expected", "") and p["confidence"] >= MIN_PATTERN_CONFIDENCE:
            if p["confidence"] > best_c:
                best_c = p["confidence"]; reason = f"alternation→LOW [{p['confidence']:.0%}]"
    # High-frequency bias → expect low digits next (reversion)
    if "freq_bias_high" in patterns:
        p = patterns["freq_bias_high"]
        if p["confidence"] >= MIN_PATTERN_CONFIDENCE and p["confidence"] > best_c:
            best_c = p["confidence"]; reason = f"hi-freq-bias→LOW [{p['confidence']:.0%}]"
    # Mean above 4.5 → downward pressure
    if "mean_reversion" in patterns:
        p = patterns["mean_reversion"]
        if p["deviation"] > 0 and p["confidence"] >= MIN_PATTERN_CONFIDENCE and p["confidence"] > best_c:
            best_c = p["confidence"]; reason = f"mean-rev↓ [{p['confidence']:.0%}]"
    # Rising run → reversal to low digits
    if "rising_run" in patterns:
        p = patterns["rising_run"]
        if p["confidence"] >= MIN_PATTERN_CONFIDENCE and p["confidence"] > best_c:
            best_c = p["confidence"]; reason = f"rising-run reversal [{p['confidence']:.0%}]"

    return best_c >= MIN_PATTERN_CONFIDENCE, best_c, reason


# ────────────────────────────────────────────────
#     TRADE DECISION  – strict 3-gate filter
# ────────────────────────────────────────────────

def trade_decision(probs, patterns, history):
    """
    THREE gates must ALL pass before a trade is placed.

    Gate 1 – Model edge   : p(winning side) > hard threshold (0.63)
    Gate 2 – Trend filter : last 10 ticks must lean in the trade direction (≥60%)
    Gate 3 – Pattern check: at least one pattern must explicitly agree with direction

    This prevents the bot from entering on near-50/50 model outputs and from
    trading against the current market direction.

    Over  4 → DIGITOVER  barrier "4"  wins on digits 5-9
    Under 5 → DIGITUNDER barrier "5"  wins on digits 0-4
    """
    think("─── 3-gate entry filter ─────────────────────────────────────────")

    p_over4  = float(np.sum(probs[5:]))   # model: p(digits 5-9)
    p_under5 = float(np.sum(probs[:5]))   # model: p(digits 0-4)

    think(f"  Gate 1 – Model edge:")
    think(f"    Over  4 p={p_over4:.4f}  need>{PROB_THRESHOLD_OVER4}  "
          f"{'✅' if p_over4 > PROB_THRESHOLD_OVER4 else '❌'}")
    think(f"    Under 5 p={p_under5:.4f}  need>{PROB_THRESHOLD_UNDER5}  "
          f"{'✅' if p_under5 > PROB_THRESHOLD_UNDER5 else '❌'}")

    # ── Gate 1: model edge ────────────────────────────────────────────────────
    o4_gate1 = p_over4  > PROB_THRESHOLD_OVER4
    u5_gate1 = p_under5 > PROB_THRESHOLD_UNDER5

    # ── Gate 2: trend filter ──────────────────────────────────────────────────
    trend_o4_ok,  tr_o4  = trend_confirms_over4(history)
    trend_u5_ok,  tr_u5  = trend_confirms_under5(history)

    think(f"  Gate 2 – Trend (last {TREND_LOOKBACK} ticks):")
    think(f"    Over  4 high-digit ratio={tr_o4:.0%}  need≥{TREND_MIN_RATIO_OVER4:.0%}  "
          f"{'✅' if trend_o4_ok else '❌'}")
    think(f"    Under 5 low-digit  ratio={tr_u5:.0%}  need≥{TREND_MIN_RATIO_UNDER5:.0%}  "
          f"{'✅' if trend_u5_ok else '❌'}")

    # ── Gate 3: pattern confirmation ──────────────────────────────────────────
    pat_o4_ok, pat_o4_conf, pat_o4_reason = pattern_agrees_over4(patterns)
    pat_u5_ok, pat_u5_conf, pat_u5_reason = pattern_agrees_under5(patterns)

    if REQUIRE_PATTERN_CONFIRMATION:
        think(f"  Gate 3 – Pattern confirmation (need confidence≥{MIN_PATTERN_CONFIDENCE:.0%}):")
        think(f"    Over  4: {'✅ ' + pat_o4_reason if pat_o4_ok else '❌ no confirming pattern'}")
        think(f"    Under 5: {'✅ ' + pat_u5_reason if pat_u5_ok else '❌ no confirming pattern'}")
    else:
        # Pattern gate disabled – treat as always passing
        pat_o4_ok = pat_u5_ok = True
        think(f"  Gate 3 – Pattern confirmation: DISABLED")

    # ── All 3 gates ───────────────────────────────────────────────────────────
    over4_clear  = o4_gate1 and trend_o4_ok and pat_o4_ok
    under5_clear = u5_gate1 and trend_u5_ok and pat_u5_ok

    think(f"  ─── Result: Over4={'ALL CLEAR ✅' if over4_clear else 'BLOCKED ❌'}  "
          f"Under5={'ALL CLEAR ✅' if under5_clear else 'BLOCKED ❌'}")

    if not over4_clear and not under5_clear:
        think("→ No trade – at least one gate failed on both sides")
        return None, None, None

    # If both somehow clear, pick the one with stronger model confidence
    if over4_clear and under5_clear:
        if p_over4 >= p_under5:
            under5_clear = False
            think(f"  Both sides clear – choosing Over 4 (model edge {p_over4:.1%} > {p_under5:.1%})")
        else:
            over4_clear = False
            think(f"  Both sides clear – choosing Under 5 (model edge {p_under5:.1%} > {p_over4:.1%})")

    if over4_clear:
        reason = f"Over 4 | model={p_over4:.1%} trend={tr_o4:.0%} pattern={pat_o4_reason}"
        think(f"→ ENTER Over 4  [{reason}]")
        return "DIGITOVER", "4", reason

    reason = f"Under 5 | model={p_under5:.1%} trend={tr_u5:.0%} pattern={pat_u5_reason}"
    think(f"→ ENTER Under 5  [{reason}]")
    return "DIGITUNDER", "5", reason


# ────────────────────────────────────────────────
#     GATE + ROUTER
# ────────────────────────────────────────────────

def get_trade_decision():
    global ticks_since_trade, pattern_cache
    if trade_in_progress:
        think("Gate locked – trade open"); return None, None, None
    ticks_since_trade += 1
    if ticks_since_trade < MIN_TICKS_BETWEEN_TRADES:
        think(f"Cooldown {ticks_since_trade}/{MIN_TICKS_BETWEEN_TRADES}"); return None, None, None

    history = list(last_digits)
    min_buf = max(WINDOW_SIZE//2, SEQ_LEN + TREND_LOOKBACK + 5)
    if len(history) < min_buf:
        think(f"Buffer {len(history)}/{min_buf}"); return None, None, None

    probs    = predict_probs_from_history(history)
    patterns = detect_patterns(history)
    pattern_cache = patterns

    if probs is None:
        think("Model not ready"); return None, None, None

    show_analysis(history, probs, patterns)
    return trade_decision(probs, patterns, history)


# ────────────────────────────────────────────────
#     DERIV API HELPERS
# ────────────────────────────────────────────────

def authorize():       ws.send(json.dumps({"authorize": AUTH_TOKEN}))
def subscribe_ticks(): ws.send(json.dumps({"ticks": SYMBOL, "subscribe": 1}))

def send_proposal(ct, barrier):
    ws.send(json.dumps({"proposal": 1, "amount": INITIAL_STAKE, "basis": "stake",
        "contract_type": ct, "currency": CURRENCY, "duration": DURATION,
        "duration_unit": DURATION_UNIT, "symbol": SYMBOL, "barrier": barrier}))

def buy_contract(pid, ask):
    ws.send(json.dumps({"buy": str(pid), "price": ask}))

def subscribe_contract(cid):
    ws.send(json.dumps({"proposal_open_contract": 1, "contract_id": cid, "subscribe": 1}))


# ────────────────────────────────────────────────
#     WEBSOCKET CALLBACKS
# ────────────────────────────────────────────────

def on_message(ws_obj, message):
    global total_profit, consecutive_losses
    global trade_in_progress, ticks_since_trade

    try:    data = json.loads(message)
    except: warn("Invalid JSON"); return

    if "error" in data:
        warn(f"API Error: {data['error'].get('message', data['error'])}")
        if data["error"].get("code") in ("ContractCreationFailure", "InvalidBarrier", "BarrierValidationError"):
            with lock:
                trade_in_progress = False
                think("Gate released after API error")
        return

    mt = data.get("msg_type")

    # ── Tick received ──────────────────────────────────────────────────────────
    if mt == "tick":
        quote = data["tick"]["quote"]
        digit = int(math.floor(quote * 100)) % 10
        with lock:
            last_digits.append(digit)
            think(f"Tick → {quote:.5f}  digit={digit}  "
                  f"[{'🔒' if trade_in_progress else '🔓'}  buf={len(last_digits)}]")

            # Initial model training once buffer is full
            if len(last_digits) == WINDOW_SIZE and W1xh is None:
                section("INITIALISING 2-LAYER RNN")
                init_model()
                think(f"Training on {len(last_digits)} ticks…")
                train_on_sequence(list(last_digits), TRAIN_EPOCHS_INITIAL)
                info("Model ready ✓")
            # Incremental model update
            elif (W1xh is not None
                  and len(last_digits) >= SEQ_LEN + 5
                  and len(last_digits) % UPDATE_MODEL_EVERY == 0):
                think("Incremental update…")
                train_on_sequence(list(last_digits)[-60:], TRAIN_EPOCHS_UPDATE)
                think("Updated ✓")

            ct, barrier, name = get_trade_decision()
            if ct:
                trade_in_progress = True
                ticks_since_trade = 0
                execute_trade(ct, barrier, name)

    # ── Proposal received ─────────────────────────────────────────────────────
    elif mt == "proposal":
        p = data["proposal"]
        think(f"Proposal → ask={p['ask_price']:.4f}  payout={p['payout']:.4f}")
        buy_contract(p["id"], p["ask_price"])

    # ── Contract purchased ────────────────────────────────────────────────────
    elif mt == "buy":
        cid = data["buy"]["contract_id"]
        tlog(f"Contract opened → ID:{cid}  stake={INITIAL_STAKE:.2f}  (fixed – no martingale)")
        subscribe_contract(cid)

    # ── Contract settled ──────────────────────────────────────────────────────
    elif mt == "proposal_open_contract":
        poc = data["proposal_open_contract"]
        if poc.get("is_sold", 0) == 1:
            profit = poc.get("profit", 0.0)
            total_profit += profit
            section("TRADE RESULT")
            tlog(f"{'✅ WIN' if profit > 0 else '❌ LOSS'}  profit={profit:+.2f}   "
                 f"session={total_profit:+.2f}   stake always={INITIAL_STAKE:.2f} (fixed)")
            with lock:
                trade_in_progress = False
                ticks_since_trade = 0
                think("Gate unlocked – next tick will re-analyse market")
                if profit > 0:
                    consecutive_losses = 0
                    think("Win recorded – stake unchanged (fixed)")
                else:
                    consecutive_losses += 1
                    think(f"Loss recorded – stake unchanged (fixed) | streak={consecutive_losses}")
                    think("No martingale, no recovery mode – waiting for next clean signal")
            check_risk_limits()

    # ── Authorisation ─────────────────────────────────────────────────────────
    elif mt == "authorize":
        info("Authorised ✓")
        subscribe_ticks()


def on_error(ws_obj, err):
    warn(f"WS error: {err}")
    with lock:
        global trade_in_progress
        trade_in_progress = False

def on_close(ws_obj, c, m): global running; info("WS closed"); running = False
def on_open(ws_obj):        info("Connected"); authorize()


# ────────────────────────────────────────────────
#     TRADE EXECUTION & RISK
# ────────────────────────────────────────────────

def execute_trade(ct, barrier, name):
    section("OPENING TRADE")
    tlog(f"Signal : {name}")
    tlog(f"Type   : {ct}  barrier={barrier}  stake={INITIAL_STAKE:.2f} (fixed)")
    send_proposal(ct, barrier)

def check_risk_limits():
    global running
    if total_profit <= STOP_LOSS:
        warn(f"Stop loss hit → session P&L={total_profit:.2f}"); running = False
    elif total_profit >= TAKE_PROFIT:
        info(f"Take profit hit → session P&L={total_profit:.2f}"); running = False
    elif consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
        warn(f"Max consecutive losses ({consecutive_losses}) reached"); running = False


# ────────────────────────────────────────────────
#     SHUTDOWN
# ────────────────────────────────────────────────

def signal_handler(sig, frame):
    global running
    info("Shutting down…")
    running = False
    if ws: ws.close()

signal.signal(signal.SIGINT,  signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ────────────────────────────────────────────────
#     MAIN
# ────────────────────────────────────────────────

def main():
    global ws
    section("DERIV DIGIT AI BOT v2 – Over 4 / Under 5 Edition")
    info(f"Symbol={SYMBOL}  stake={INITIAL_STAKE} (FIXED – no martingale)  SL={STOP_LOSS}  TP={TAKE_PROFIT}")
    info(f"Model: 2-layer RNN | {FEATURE_SIZE} engineered features | seq_len={SEQ_LEN} | hidden={HIDDEN_SIZE}x2")
    info(f"Trades: Over 4 (digits 5-9, threshold>{PROB_THRESHOLD_OVER4})  |  "
         f"Under 5 (digits 0-4, threshold>{PROB_THRESHOLD_UNDER5})")
    info(f"Martingale: DISABLED  |  Recovery mode: DISABLED")
    info(f"Every trade re-analyses market – no stale signals")
    think("Pattern detectors: streak | hi/lo alternation | even/odd alternation |")
    think("                   rising/falling run | freq bias | cold/hot digit |")
    think("                   mean reversion | cycle detector (period 3-6)")
    think("Connecting…\n")

    websocket.enableTrace(False)
    ws = websocket.WebSocketApp(WS_URL, on_open=on_open, on_message=on_message,
                                on_error=on_error, on_close=on_close)
    t = threading.Thread(target=ws.run_forever, daemon=True)
    t.start()

    try:
        while running:
            time.sleep(0.5)
    finally:
        if ws: ws.close()
        section("BOT STOPPED")
        info(f"Final P&L: {total_profit:+.2f}   Loss streak: {consecutive_losses}   "
             f"Model steps: {global_step}")


if __name__ == "__main__":
    random.seed(time.time())
    np.random.seed(int(time.time() * 1000) % 2**32)
    main()