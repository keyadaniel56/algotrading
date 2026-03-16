# deriv_digit_ai_bot_numpy.py
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
AUTH_TOKEN    = ""
SYMBOL        = "R_100"
DURATION      = 1
DURATION_UNIT = "t"
CURRENCY      = "USD"

INITIAL_STAKE          = 1.0
MARTINGALE             = True    # normal mode martingale
MARTINGALE_MULTIPLIER  = 2.0

# Recovery always uses its own martingale regardless of MARTINGALE flag
RECOVERY_MARTINGALE_MULTIPLIER = 2.0
MAX_CONSECUTIVE_LOSSES = 30
STOP_LOSS              = -100.0
TAKE_PROFIT            = 100.0

SEQ_LEN              = 20
WINDOW_SIZE          = 150
HIDDEN_SIZE          = 48
# FIX 2: expanded from 28 → 32 to include 4 pattern features
FEATURE_SIZE         = 32
LEARNING_RATE        = 0.006
LR_DECAY             = 0.9999
TRAIN_EPOCHS_INITIAL = 80
# FIX 4: increased from 10 → 20 epochs per update
TRAIN_EPOCHS_UPDATE  = 20
# FIX 4: update twice as often (was 8)
UPDATE_MODEL_EVERY   = 4

PROB_THRESHOLD_OVER   = 0.80   # FIX 1: lowered from 0.87 — patterns now gate, not boost
# NOTE: Under 1 is fully disabled – it only wins on digit 0 and is not used in this bot
PATTERN_BOOST         = 0.03

RECOVERY_OVER_BARRIER    = "3"   # Over 3  = digits 4-9 win
RECOVERY_UNDER_BARRIER   = "6"   # Under 6 = digits 0-5 win
# FIX 3: raised thresholds from 0.55 — patterns now boost before check
RECOVERY_OVER_THRESHOLD  = 0.62
RECOVERY_UNDER_THRESHOLD = 0.62
RECOVERY_TRADES_COUNT    = 3

WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"

# ────────────────────────────────────────────────
#                 GLOBAL STATE
# ────────────────────────────────────────────────
last_digits        = deque(maxlen=WINDOW_SIZE)
total_profit       = 0.0
current_stake      = INITIAL_STAKE
recovery_stake     = INITIAL_STAKE   # tracks martingale stake inside recovery
consecutive_losses = 0
global_step        = 0
current_lr         = LEARNING_RATE

W1xh = W1hh = b1h = None
W2xh = W2hh = b2h = None
Why  = by   = None

pattern_cache        = {}
recovery_mode        = False
recovery_trades_left = 0
trade_in_progress    = False
ticks_since_trade    = 0

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
def pat(msg):   print(f"  🔍 {pat}")

def section(title):
    bar = "─" * 80
    print(f"\n{bar}\n  {title}\n{bar}")


# ────────────────────────────────────────────────
#     FEATURE ENGINEERING  (32 features)
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

    # ── FIX 2: 4 new pattern features baked into training data ──────────────

    # 28  hi/lo alternation strength over last 8 ticks
    if n >= 8:
        window = history[-8:]
        alt_count = sum(1 for i in range(1, len(window))
                        if (window[i] >= 5) != (window[i-1] >= 5))
        vec[28] = alt_count / (len(window) - 1)

    # 29  cold-digit pressure: how far the coldest digit is below expected 10%
    if n >= 50:
        cnt = np.bincount(history[-50:], minlength=10) / 50.0
        min_freq = cnt.min()
        vec[29] = max(0.0, (0.10 - min_freq) / 0.10)

    # 30  rising run length (normalised to 5)
    run_up = 0
    for i in range(n - 1, max(n - 6, 0), -1):
        if history[i] > history[i - 1]:
            run_up += 1
        else:
            break
    vec[30] = min(run_up, 5) / 5.0

    # 31  falling run length (normalised to 5)
    run_dn = 0
    for i in range(n - 1, max(n - 6, 0), -1):
        if history[i] < history[i - 1]:
            run_dn += 1
        else:
            break
    vec[31] = min(run_dn, 5) / 5.0

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

def show_analysis(history, probs, patterns, mode_label):
    section(f"MARKET ANALYSIS  [{mode_label} mode]")
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
        print("\n  🔍 Detected patterns:")
        for name, data in patterns.items():
            c = data.get("confidence",0); bar = "●"*int(c*10)
            print(f"       {bar:<10} {c:.0%}  {data.get('signal', name)}")
    else:
        print("\n  🔍 No significant patterns this tick")

    if probs is not None:
        print()
        think("Model per-digit probabilities:")
        for i in range(10):
            bar = "▓" * int(probs[i]*100)
            print(f"       digit {i} : {bar:<20} {probs[i]:.3f}")
        print()
        p_u5 = float(np.sum(probs[:5]))
        think(f"  p(digit 0) alone: {probs[0]:.1%}  ← Under 1 uses THIS (wins ONLY on digit 0)")
        think(f"  Over 1  (2-9)   : {float(np.sum(probs[2:])):.1%}")
        think(f"  Under 6 (0-5)   : {p_u5:.1%}   Over 3 (4-9): {float(np.sum(probs[4:])):.1%}")


# ────────────────────────────────────────────────
#     PATTERN → TRADE HINT  (kept for display/debug)
# ────────────────────────────────────────────────

def pattern_trade_hint(patterns):
    u1 = o1 = 0.0; reasons = []
    if "alternation" in patterns:
        c = patterns["alternation"]["confidence"]
        exp = patterns["alternation"]["next_expected"]
        if "LOW" in exp:  u1 += c*0.04; reasons.append(f"alternation→LOW +{c*4:.1f}%U")
        else:             o1 += c*0.04; reasons.append(f"alternation→HIGH +{c*4:.1f}%O")
    if "freq_bias_high" in patterns:
        c = patterns["freq_bias_high"]["confidence"]
        u1 += c*0.03; reasons.append(f"hi-bias revert +{c*3:.1f}%U")
    if "freq_bias_low" in patterns:
        c = patterns["freq_bias_low"]["confidence"]
        o1 += c*0.03; reasons.append(f"lo-bias revert +{c*3:.1f}%O")
    if "mean_reversion" in patterns:
        c = patterns["mean_reversion"]["confidence"]
        dev = patterns["mean_reversion"]["deviation"]
        if dev > 0: u1 += c*0.03; reasons.append(f"mean-rev↓ +{c*3:.1f}%U")
        else:       o1 += c*0.03; reasons.append(f"mean-rev↑ +{c*3:.1f}%O")
    return u1, o1, " | ".join(reasons) if reasons else "no pattern bonus"


# ────────────────────────────────────────────────
#     FIX 1: NORMAL MODE – patterns as GATE not boost
# ────────────────────────────────────────────────

def normal_decision(probs, patterns):
    """
    Normal mode places ONLY Over 1 (digits 2-9).
    Under 1 is NOT placed in normal mode.

    FIX 1: Patterns now act as a gate/veto rather than a tiny additive boost.
      - Strong contradicting patterns VETO the trade regardless of model prob.
      - When model prob is moderate (0.72–0.80), a confirming pattern is REQUIRED.
      - Model prob > 0.80 alone is sufficient (lowered from 0.87).
    """
    p_o1_raw = float(np.sum(probs[2:]))

    think("Normal mode (Over 1 only – Under 1 is disabled):")

    # ── VETO CHECK: contradicting patterns block the trade ──────────────────
    if "alternation" in patterns:
        exp = patterns["alternation"]["next_expected"]
        confidence = patterns["alternation"]["confidence"]
        if "LOW" in exp and confidence > 0.75:
            think(f"→ VETO: alternation ({confidence:.0%}) says LOW next – blocking Over 1")
            return None, None, None

    if "freq_bias_low" in patterns:
        c = patterns["freq_bias_low"]["confidence"]
        if c > 0.70:
            think(f"→ VETO: low-digit frequency bias ({c:.0%}) – blocking Over 1")
            return None, None, None

    if "falling_run" in patterns:
        c = patterns["falling_run"]["confidence"]
        if c >= 0.80:
            think(f"→ VETO: strong falling run ({c:.0%}) – blocking Over 1")
            return None, None, None

    # ── CONFIRM LOGIC ────────────────────────────────────────────────────────
    confirming = False
    confirm_reason = ""

    if p_o1_raw > PROB_THRESHOLD_OVER:
        # Model alone is strong enough above threshold
        confirming = True
        confirm_reason = f"model p={p_o1_raw:.1%} exceeds threshold {PROB_THRESHOLD_OVER}"
    elif p_o1_raw > 0.72:
        # Moderate model signal — require at least one confirming pattern
        if "alternation" in patterns:
            exp = patterns["alternation"]["next_expected"]
            if "HIGH" in exp:
                confirming = True
                confirm_reason = f"model p={p_o1_raw:.1%} + alternation→HIGH"
        if not confirming and "freq_bias_high" in patterns:
            c = patterns["freq_bias_high"]["confidence"]
            if c > 0.60:
                confirming = True
                confirm_reason = f"model p={p_o1_raw:.1%} + hi-freq bias ({c:.0%})"
        if not confirming and "cold_digit" in patterns:
            # Cold low digit suggests reversion toward higher digits
            cd = patterns["cold_digit"]["digit"]
            if cd <= 3:
                confirming = True
                confirm_reason = f"model p={p_o1_raw:.1%} + cold low digit {cd}"
        if not confirming and "rising_run" in patterns:
            c = patterns["rising_run"]["confidence"]
            if c >= 0.60:
                confirming = True
                confirm_reason = f"model p={p_o1_raw:.1%} + rising run ({c:.0%})"

    if not confirming:
        think(f"  Over 1 (digits 2-9): raw={p_o1_raw:.4f}  need>{PROB_THRESHOLD_OVER} or pattern confirm  ❌ no signal")
        think("→ No edge – skipping this tick")
        return None, None, None

    think(f"  Over 1 confirmed: {confirm_reason}")
    think(f"→ Over 1 signal (raw p={p_o1_raw:.1%})")
    return "DIGITOVER", "1", f"Over 1 (p={p_o1_raw:.1%})"


# ────────────────────────────────────────────────
#     FIX 3: RECOVERY MODE – patterns boost BEFORE threshold check
# ────────────────────────────────────────────────

def recovery_decision(probs, patterns):
    """
    Recovery uses ONLY Over 3 (digits 4-9) or Under 6 (digits 0-5).
    Re-analyses the market before each recovery trade.

    FIX 3: Pattern adjustments now applied BEFORE threshold check,
           thresholds raised to 0.62 to compensate, and pattern
           contradictions can veto a direction entirely.
    """
    think("Recovery – re-analysing market for Over 3 or Under 6:")

    p_over3  = float(np.sum(probs[4:]))
    p_under6 = float(np.sum(probs[:6]))

    # ── FIX 3: Apply pattern adjustments BEFORE threshold check ─────────────
    if "alternation" in patterns:
        c   = patterns["alternation"]["confidence"]
        exp = patterns["alternation"]["next_expected"]
        if "HIGH" in exp:
            boost = c * 0.12
            p_over3  += boost
            think(f"  Alternation→HIGH boosts Over 3 by +{boost:.1%}")
        elif "LOW" in exp:
            boost = c * 0.12
            p_under6 += boost
            think(f"  Alternation→LOW boosts Under 6 by +{boost:.1%}")

    if "freq_bias_high" in patterns:
        c = patterns["freq_bias_high"]["confidence"]
        boost = c * 0.08
        p_over3 += boost
        think(f"  Hi-freq bias boosts Over 3 by +{boost:.1%}")
    if "freq_bias_low" in patterns:
        c = patterns["freq_bias_low"]["confidence"]
        boost = c * 0.08
        p_under6 += boost
        think(f"  Lo-freq bias boosts Under 6 by +{boost:.1%}")

    if "cold_digit" in patterns:
        cd = patterns["cold_digit"]["digit"]
        c  = patterns["cold_digit"]["confidence"]
        if cd <= 3:
            boost = c * 0.05
            p_over3 += boost
            think(f"  Cold low digit {cd} boosts Over 3 by +{boost:.1%}")
        elif cd >= 6:
            boost = c * 0.05
            p_under6 += boost
            think(f"  Cold high digit {cd} boosts Under 6 by +{boost:.1%}")

    if "rising_run" in patterns:
        c = patterns["rising_run"]["confidence"]
        p_over3 += c * 0.06
        think(f"  Rising run boosts Over 3 by +{c*0.06:.1%}")
    if "falling_run" in patterns:
        c = patterns["falling_run"]["confidence"]
        p_under6 += c * 0.06
        think(f"  Falling run boosts Under 6 by +{c*0.06:.1%}")

    # ── VETO: strong contradicting pattern zeroes out a direction ────────────
    if "alternation" in patterns:
        exp = patterns["alternation"]["next_expected"]
        c   = patterns["alternation"]["confidence"]
        if "LOW" in exp and c > 0.85 and p_over3 < 0.65:
            think(f"  Pattern VETO: alternation→LOW, nullifying Over 3")
            p_over3 = 0.0
        elif "HIGH" in exp and c > 0.85 and p_under6 < 0.65:
            think(f"  Pattern VETO: alternation→HIGH, nullifying Under 6")
            p_under6 = 0.0

    think(f"  Over  3 (digits 4-9): p={p_over3:.4f}  need>{RECOVERY_OVER_THRESHOLD}  "
          f"{'✅ SIGNAL' if p_over3 > RECOVERY_OVER_THRESHOLD else '❌ no signal'}")
    think(f"  Under 6 (digits 0-5): p={p_under6:.4f}  need>{RECOVERY_UNDER_THRESHOLD}  "
          f"{'✅ SIGNAL' if p_under6 > RECOVERY_UNDER_THRESHOLD else '❌ no signal'}")

    # Pick the stronger signal
    if p_over3 > RECOVERY_OVER_THRESHOLD or p_under6 > RECOVERY_UNDER_THRESHOLD:
        if p_over3 >= p_under6 and p_over3 > RECOVERY_OVER_THRESHOLD:
            think(f"→ Over 3 chosen (stronger signal p={p_over3:.1%})")
            return "DIGITOVER", "3", f"Over 3 recovery (p={p_over3:.1%})"
        elif p_under6 > RECOVERY_UNDER_THRESHOLD:
            think(f"→ Under 6 chosen (stronger signal p={p_under6:.1%})")
            return "DIGITUNDER", "6", f"Under 6 recovery (p={p_under6:.1%})"

    think("→ Neither Over 3 nor Under 6 is strong enough – skipping this tick")
    think("   Waiting for a clearer recovery signal before placing trade")
    return None, None, None


# ────────────────────────────────────────────────
#     GATE + ROUTER
# ────────────────────────────────────────────────

def get_trade_decision():
    global ticks_since_trade, pattern_cache
    if trade_in_progress:
        think("Gate locked – trade open"); return None,None,None
    ticks_since_trade += 1
    if ticks_since_trade < 2:
        think(f"Cooldown {ticks_since_trade}/2"); return None,None,None

    history = list(last_digits)
    min_buf = max(WINDOW_SIZE//2, SEQ_LEN+5)
    if len(history) < min_buf:
        think(f"Buffer {len(history)}/{min_buf}"); return None,None,None

    probs    = predict_probs_from_history(history)
    patterns = detect_patterns(history)
    pattern_cache = patterns

    if probs is None:
        think("Model not ready"); return None,None,None

    show_analysis(history, probs, patterns, "RECOVERY" if recovery_mode else "Normal")
    return recovery_decision(probs, patterns) if recovery_mode else normal_decision(probs, patterns)


# ────────────────────────────────────────────────
#     DERIV API HELPERS
# ────────────────────────────────────────────────

def authorize():    ws.send(json.dumps({"authorize": AUTH_TOKEN}))
def subscribe_ticks(): ws.send(json.dumps({"ticks": SYMBOL, "subscribe": 1}))

def send_proposal(ct, barrier):
    ws.send(json.dumps({"proposal":1,"amount":current_stake,"basis":"stake",
        "contract_type":ct,"currency":CURRENCY,"duration":DURATION,
        "duration_unit":DURATION_UNIT,"symbol":SYMBOL,"barrier":barrier}))

def buy_contract(pid, ask):
    ws.send(json.dumps({"buy": str(pid), "price": ask}))

def subscribe_contract(cid):
    ws.send(json.dumps({"proposal_open_contract":1,"contract_id":cid,"subscribe":1}))


# ────────────────────────────────────────────────
#     WEBSOCKET CALLBACKS
# ────────────────────────────────────────────────

def on_message(ws_obj, message):
    global total_profit, current_stake, recovery_stake, consecutive_losses
    global recovery_mode, recovery_trades_left
    global trade_in_progress, ticks_since_trade

    try:    data = json.loads(message)
    except: warn("Invalid JSON"); return

    if "error" in data:
        warn(f"API Error: {data['error'].get('message', data['error'])}")
        if data["error"].get("code") in ("ContractCreationFailure","InvalidBarrier","BarrierValidationError"):
            with lock:
                trade_in_progress = False; think("Gate released")
        return

    mt = data.get("msg_type")

    if mt == "tick":
        quote = data["tick"]["quote"]
        digit = int(math.floor(quote * 100)) % 10
        with lock:
            last_digits.append(digit)
            think(f"Tick → {quote:.5f}  digit={digit}  "
                  f"[{'🔒' if trade_in_progress else '🔓'}  buf={len(last_digits)}]")

            if len(last_digits) == WINDOW_SIZE and W1xh is None:
                section("INITIALISING 2-LAYER RNN")
                init_model()
                think(f"Training on {len(last_digits)} ticks…")
                train_on_sequence(list(last_digits), TRAIN_EPOCHS_INITIAL)
                info("Model ready ✓")
            elif (W1xh is not None and len(last_digits) >= SEQ_LEN+5
                  and len(last_digits) % UPDATE_MODEL_EVERY == 0):
                think("Incremental update…")
                # FIX 4: use more history (100 ticks) and more epochs (20)
                train_on_sequence(list(last_digits)[-100:], TRAIN_EPOCHS_UPDATE)
                think("Updated ✓")

            ct, barrier, name = get_trade_decision()
            if ct:
                trade_in_progress = True; ticks_since_trade = 0
                execute_trade(ct, barrier, name)

    elif mt == "proposal":
        p = data["proposal"]
        think(f"Proposal → ask={p['ask_price']:.4f}  payout={p['payout']:.4f}")
        buy_contract(p["id"], p["ask_price"])

    elif mt == "buy":
        cid = data["buy"]["contract_id"]
        tlog(f"Contract opened → ID:{cid}  stake={current_stake:.2f}")
        subscribe_contract(cid)

    elif mt == "proposal_open_contract":
        poc = data["proposal_open_contract"]
        if poc.get("is_sold",0) == 1:
            profit = poc.get("profit", 0.0); total_profit += profit
            section("TRADE RESULT")
            tlog(f"{'✅' if profit>0 else '❌'}  profit={profit:+.2f}   session={total_profit:+.2f}")
            with lock:
                trade_in_progress = False; ticks_since_trade = 0
                think("Gate unlocked")
                if profit > 0:
                    consecutive_losses = 0
                    if recovery_mode:
                        recovery_trades_left -= 1
                        think(f"Recovery WIN ✅ – {recovery_trades_left} recovery trades remaining")
                        think(f"Recovery stake was {current_stake:.2f} → profit covers previous losses")
                        if recovery_trades_left <= 0:
                            recovery_mode  = False
                            recovery_stake = INITIAL_STAKE
                            current_stake  = INITIAL_STAKE
                            think("Recovery complete – stake reset to initial, back to NORMAL mode")
                        else:
                            recovery_stake = INITIAL_STAKE
                            current_stake  = INITIAL_STAKE
                            think(f"Recovery trade won – recovery stake reset to {INITIAL_STAKE:.2f} for next recovery trade")
                    else:
                        current_stake = INITIAL_STAKE
                        think("Win – stake reset to initial")
                else:
                    consecutive_losses += 1
                    if recovery_mode:
                        recovery_stake *= RECOVERY_MARTINGALE_MULTIPLIER
                        current_stake   = recovery_stake
                        recovery_trades_left -= 1
                        think(f"Recovery LOSS ❌ – martingale: next recovery stake = {current_stake:.2f}")
                        think(f"  (doubling to recover previous loss of {recovery_stake/RECOVERY_MARTINGALE_MULTIPLIER:.2f})")
                        if recovery_trades_left <= 0:
                            recovery_mode  = False
                            recovery_stake = INITIAL_STAKE
                            current_stake  = INITIAL_STAKE
                            think("Recovery trades exhausted – stake reset, back to normal mode")
                    else:
                        if MARTINGALE:
                            current_stake *= MARTINGALE_MULTIPLIER
                            think(f"Normal martingale → stake={current_stake:.2f}")
                        recovery_mode        = True
                        recovery_trades_left = RECOVERY_TRADES_COUNT
                        recovery_stake       = INITIAL_STAKE
                        current_stake        = recovery_stake
                        think(f"⚠️  RECOVERY MODE ({RECOVERY_TRADES_COUNT} trades) – will use Over 3 or Under 6")
                        think(f"   Recovery martingale multiplier: x{RECOVERY_MARTINGALE_MULTIPLIER}")
                        think(f"   Stakes will be: "
                              + " → ".join(f"{INITIAL_STAKE * (RECOVERY_MARTINGALE_MULTIPLIER**i):.2f}"
                                           for i in range(RECOVERY_TRADES_COUNT)))
            check_risk_limits()

    elif mt == "authorize":
        info("Authorised ✓"); subscribe_ticks()


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
    tlog(f"Signal: {name}  |  {ct} barrier={barrier}  stake={current_stake:.2f}")
    send_proposal(ct, barrier)

def check_risk_limits():
    global running
    if total_profit <= STOP_LOSS:   warn(f"Stop loss {total_profit:.2f}"); running=False
    elif total_profit >= TAKE_PROFIT: info(f"Take profit {total_profit:.2f}"); running=False
    elif consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
        warn(f"Max losses ({consecutive_losses})"); running=False


# ────────────────────────────────────────────────
#     SHUTDOWN
# ────────────────────────────────────────────────

def signal_handler(sig, frame):
    global running; info("Shutting down…"); running=False
    if ws: ws.close()

signal.signal(signal.SIGINT,  signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ────────────────────────────────────────────────
#     MAIN
# ────────────────────────────────────────────────

def main():
    global ws
    section("DERIV DIGIT AI BOT  –  Pattern-Learning Edition v2")
    info(f"Symbol={SYMBOL}  stake={INITIAL_STAKE}  SL={STOP_LOSS}  TP={TAKE_PROFIT}")
    info(f"Model: 2-layer RNN | {FEATURE_SIZE} engineered features | seq_len={SEQ_LEN} | hidden={HIDDEN_SIZE}x2")
    info(f"Normal:   Over 1 only  threshold>{PROB_THRESHOLD_OVER}  (Under 1 disabled)")
    info(f"Recovery: Over3>{RECOVERY_OVER_THRESHOLD}  Under6>{RECOVERY_UNDER_THRESHOLD}  (patterns boost before threshold check)")
    think("FIX 1: Patterns are now a GATE/VETO, not a tiny additive boost")
    think("FIX 2: 4 pattern features baked into RNN training (FEATURE_SIZE=32)")
    think("FIX 3: Recovery patterns boost p BEFORE threshold check (raised to 0.62)")
    think("FIX 4: Model updates every 4 ticks on 100 ticks history, 20 epochs")
    think("Pattern detectors: streak | hi/lo alternation | even/odd alternation |")
    think("                   rising/falling run | freq bias | cold/hot digit |")
    think("                   mean reversion | cycle detector (period 3-6)")
    think("Connecting…\n")

    websocket.enableTrace(False)
    ws = websocket.WebSocketApp(WS_URL, on_open=on_open, on_message=on_message,
                                on_error=on_error, on_close=on_close)
    t = threading.Thread(target=ws.run_forever, daemon=True); t.start()

    try:
        while running: time.sleep(0.5)
    finally:
        if ws: ws.close()
        section("BOT STOPPED")
        info(f"P&L: {total_profit:+.2f}   Loss streak: {consecutive_losses}   Steps: {global_step}")


if __name__ == "__main__":
    random.seed(time.time())
    np.random.seed(int(time.time()*1000) % 2**32)
    main()