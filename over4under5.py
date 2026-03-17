# deriv_digit_ai_bot_numpy_under4_over5.py
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
# CONFIGURABLE PARAMETERS
# ────────────────────────────────────────────────
APP_ID = "1089"
AUTH_TOKEN = "WFabi7aeCbFjgvp"
SYMBOL = "R_100"
DURATION = 1
DURATION_UNIT = "t"
CURRENCY = "USD"
INITIAL_STAKE = 0.35
MARTINGALE = True
MARTINGALE_MULTIPLIER = 2.0
# Recovery always uses its own martingale
RECOVERY_MARTINGALE_MULTIPLIER = 2.0
MAX_CONSECUTIVE_LOSSES = 30
STOP_LOSS = -10.0
TAKE_PROFIT = 0.70
SEQ_LEN = 20
WINDOW_SIZE = 150
HIDDEN_SIZE = 48
FEATURE_SIZE = 32
LEARNING_RATE = 0.006
LR_DECAY = 0.9999
TRAIN_EPOCHS_INITIAL = 80
TRAIN_EPOCHS_UPDATE = 20
UPDATE_MODEL_EVERY = 4
PROB_THRESHOLD_OVER5 = 0.82     # Higher threshold for ~40% contract
PROB_THRESHOLD_UNDER4 = 0.82
PATTERN_BOOST = 0.03
# ── New barriers ─────────────────────────────────
NORMAL_CONTRACT = "DIGITOVER"   # Only Over 5 in normal mode
NORMAL_BARRIER  = "5"
RECOVERY_OVER_BARRIER  = "5"    # Over 5 = digits 6-9 win
RECOVERY_UNDER_BARRIER = "4"    # Under 4 = digits 0-3 win
RECOVERY_OVER_THRESHOLD  = 0.64
RECOVERY_UNDER_THRESHOLD = 0.64
RECOVERY_TRADES_COUNT = 3

WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"

# ────────────────────────────────────────────────
# GLOBAL STATE (mostly unchanged)
# ────────────────────────────────────────────────
last_digits = deque(maxlen=WINDOW_SIZE)
total_profit = 0.0
current_stake = INITIAL_STAKE
recovery_stake = INITIAL_STAKE
consecutive_losses = 0
global_step = 0
current_lr = LEARNING_RATE
W1xh = W1hh = b1h = None
W2xh = W2hh = b2h = None
Why = by = None
pattern_cache = {}
recovery_mode = False
recovery_trades_left = 0
trade_in_progress = False
ticks_since_trade = 0
ws = None
running = True
lock = threading.Lock()

# Logging functions unchanged...
def think(msg): print(f" 🧠 {msg}")
def info(msg): print(f" ℹ️ {msg}")
def warn(msg): print(f" ⚠️ {msg}")
def tlog(msg): print(f" 💰 {msg}")
def section(title):
    bar = "─" * 80
    print(f"\n{bar}\n {title}\n{bar}")

# ────────────────────────────────────────────────
# FEATURE ENGINEERING (unchanged – still 32 features)
# ────────────────────────────────────────────────
# ... (keep your build_features and build_sequence_matrix exactly as before)

# ────────────────────────────────────────────────
# PATTERN DETECTORS (minor tweak for low/high split)
# ────────────────────────────────────────────────
def detect_patterns(history: list) -> dict:
    p = {}
    n = len(history)
    if n < 10:
        return p

    # ... keep most as is ...

    # Hi/lo alternation (already good for 0-3 vs 6-9)
    window = history[-8:]
    alt_count = sum(1 for i in range(1, len(window))
                    if (window[i] >= 6) != (window[i-1] >= 6))   # changed to 6
    alt_ratio = alt_count / (len(window) - 1)
    if alt_ratio >= 0.75:
        p["alternation"] = {
            "confidence": alt_ratio,
            "next_expected": "LOW (0-3)" if history[-1] >= 6 else "HIGH (6-9)",
            "signal": f"hi/lo alternation {alt_ratio:.0%} – next likely {'LOW' if history[-1]>=6 else 'HIGH'}"
        }

    # Frequency bias (last 30) – adjusted split
    recent30 = history[-30:] if n >= 30 else history
    hf = sum(1 for d in recent30 if d >= 6) / len(recent30)   # high = 6-9
    lf = 1.0 - hf
    if hf >= 0.60:
        p["freq_bias_high"] = {"frequency": hf, "confidence": (hf-0.5)*2,
                               "signal": f"digits 6-9 = {hf:.0%} of last 30 (hot)"}
    elif lf >= 0.60:
        p["freq_bias_low"] = {"frequency": lf, "confidence": (lf-0.5)*2,
                              "signal": f"digits 0-3 = {lf:.0%} of last 30 (hot)"}

    # ... keep cold/hot, mean reversion, runs, cycles as is ...

    return p

# pattern_confidence_summary unchanged

# ────────────────────────────────────────────────
# MODEL (forward, predict, train) unchanged
# ... keep init_model, forward, predict_probs_from_history, train_on_sequence as is ...

# ────────────────────────────────────────────────
# ANALYSIS DISPLAY – updated for new barriers
# ────────────────────────────────────────────────
def show_analysis(history, probs, patterns, mode_label):
    section(f"MARKET ANALYSIS [{mode_label} mode]")
    think(f"Last digit: {history[-1]} | Buffer: {len(history)}/{WINDOW_SIZE}")
    cnt = np.bincount(history[-50:] if len(history)>=50 else history, minlength=10)
    total = cnt.sum()
    think("Digit frequency (last 50):")
    for i in range(10):
        bar = "█" * int(cnt[i]/total*40)
        dev = cnt[i]/total - 0.10
        tag = ""
        if i <= 3: tag = f" ◄ LOW {dev:+.0%}" if dev > 0 else f" ◄ COLD LOW {dev:+.0%}"
        if i >= 6: tag = f" ◄ HIGH {dev:+.0%}" if dev > 0 else f" ◄ COLD HIGH {dev:+.0%}"
        print(f" {i} : {bar:<40} {cnt[i]:3d} ({cnt[i]/total*100:4.1f}%){tag}")

    if patterns:
        print("\n 🔍 Detected patterns:")
        for name, data in patterns.items():
            c = data.get("confidence",0); bar = "●"*int(c*10)
            print(f" {bar:<10} {c:.0%} {data.get('signal', name)}")

    if probs is not None:
        print()
        think("Model probabilities:")
        for i in range(10):
            bar = "▓" * int(probs[i]*100)
            print(f" digit {i} : {bar:<20} {probs[i]:.3f}")
        print()
        p_low  = float(np.sum(probs[:4]))   # 0-3
        p_high = float(np.sum(probs[6:]))   # 6-9
        think(f" Under 4 (0-3)  : {p_low:.1%}")
        think(f" Over  5 (6-9)  : {p_high:.1%}")
        think(f" Middle (4-5)   : {probs[4]+probs[5]:.1%} – avoided")

# ────────────────────────────────────────────────
# NORMAL MODE – only Over 5, patterns as gate/veto
# ────────────────────────────────────────────────
def normal_decision(probs, patterns):
    think("Normal mode: Over 5 only (digits 6-9)")
    p_over5_raw = float(np.sum(probs[6:]))

    # VETO: strong low-side patterns block Over 5
    veto = False
    if "alternation" in patterns:
        exp = patterns["alternation"]["next_expected"]
        c = patterns["alternation"]["confidence"]
        if "LOW" in exp and c > 0.78:
            think(f"VETO: alternation ({c:.0%}) expects LOW (0-3) → block Over 5")
            veto = True
    if "freq_bias_low" in patterns:
        c = patterns["freq_bias_low"]["confidence"]
        if c > 0.72:
            think(f"VETO: low bias ({c:.0%}) → block Over 5")
            veto = True
    if "cold_digit" in patterns and patterns["cold_digit"]["digit"] <= 3:
        c = patterns["cold_digit"]["confidence"]
        if c > 0.70:
            think(f"VETO: very cold low digit → expect low → block Over 5")
            veto = True

    if veto:
        think("→ Vetoed – skipping")
        return None, None, None

    # Confirmation
    confirming = False
    reason = ""
    if p_over5_raw > PROB_THRESHOLD_OVER5:
        confirming = True
        reason = f"model p={p_over5_raw:.1%} > {PROB_THRESHOLD_OVER5}"
    elif p_over5_raw > 0.74:
        # Need confirming pattern
        if "alternation" in patterns and "HIGH" in patterns["alternation"]["next_expected"]:
            confirming = True
            reason = f"model {p_over5_raw:.1%} + alternation HIGH"
        elif "freq_bias_high" in patterns and patterns["freq_bias_high"]["confidence"] > 0.62:
            confirming = True
            reason = f"model {p_over5_raw:.1%} + high bias"
        elif "rising_run" in patterns:
            confirming = True
            reason = f"model {p_over5_raw:.1%} + rising run"

    if not confirming:
        think(f"Over 5 raw p={p_over5_raw:.4f} – no strong signal or confirmation ❌")
        return None, None, None

    think(f"Over 5 confirmed: {reason}")
    return "DIGITOVER", "5", f"Over 5 (p={p_over5_raw:.1%})"

# ────────────────────────────────────────────────
# RECOVERY MODE – patterns boost BEFORE threshold, then pick best
# ────────────────────────────────────────────────
def recovery_decision(probs, patterns):
    think("Recovery mode: Under 4 or Over 5")
    p_over5   = float(np.sum(probs[6:]))
    p_under4  = float(np.sum(probs[:4]))

    # Apply pattern boosts BEFORE check
    if "alternation" in patterns:
        c = patterns["alternation"]["confidence"]
        exp = patterns["alternation"]["next_expected"]
        boost = c * 0.12
        if "HIGH" in exp:
            p_over5 += boost
            think(f"Alternation HIGH → +{boost:.1%} to Over 5")
        elif "LOW" in exp:
            p_under4 += boost
            think(f"Alternation LOW → +{boost:.1%} to Under 4")

    if "freq_bias_high" in patterns:
        boost = patterns["freq_bias_high"]["confidence"] * 0.09
        p_over5 += boost
        think(f"High bias → +{boost:.1%} to Over 5")
    if "freq_bias_low" in patterns:
        boost = patterns["freq_bias_low"]["confidence"] * 0.09
        p_under4 += boost
        think(f"Low bias → +{boost:.1%} to Under 4")

    if "cold_digit" in patterns:
        cd = patterns["cold_digit"]["digit"]
        c = patterns["cold_digit"]["confidence"]
        boost = c * 0.06
        if cd <= 3:
            p_over5 += boost
            think(f"Cold low {cd} → +{boost:.1%} to Over 5 (reversion)")
        elif cd >= 6:
            p_under4 += boost
            think(f"Cold high {cd} → +{boost:.1%} to Under 4 (reversion)")

    if "rising_run" in patterns:
        p_over5 += patterns["rising_run"]["confidence"] * 0.07
    if "falling_run" in patterns:
        p_under4 += patterns["falling_run"]["confidence"] * 0.07

    # Veto strong contradictions
    if "alternation" in patterns:
        exp = patterns["alternation"]["next_expected"]
        c = patterns["alternation"]["confidence"]
        if "LOW" in exp and c > 0.85 and p_over5 < 0.68:
            think("VETO: strong LOW alternation → nullify Over 5")
            p_over5 = 0.0
        elif "HIGH" in exp and c > 0.85 and p_under4 < 0.68:
            think("VETO: strong HIGH alternation → nullify Under 4")
            p_under4 = 0.0

    think(f"Over 5  p={p_over5:.4f}  need>{RECOVERY_OVER_THRESHOLD}  {'✅' if p_over5 > RECOVERY_OVER_THRESHOLD else '❌'}")
    think(f"Under 4 p={p_under4:.4f} need>{RECOVERY_UNDER_THRESHOLD} {'✅' if p_under4 > RECOVERY_UNDER_THRESHOLD else '❌'}")

    if p_over5 > RECOVERY_OVER_THRESHOLD or p_under4 > RECOVERY_UNDER_THRESHOLD:
        if p_over5 >= p_under4 and p_over5 > RECOVERY_OVER_THRESHOLD:
            think(f"→ Over 5 chosen (p={p_over5:.1%})")
            return "DIGITOVER", "5", f"Over 5 recovery (p={p_over5:.1%})"
        else:
            think(f"→ Under 4 chosen (p={p_under4:.1%})")
            return "DIGITUNDER", "4", f"Under 4 recovery (p={p_under4:.1%})"

    think("→ Neither strong enough – skip")
    return None, None, None

# ────────────────────────────────────────────────
# get_trade_decision, on_message, execute_trade, etc. – minor updates
# ────────────────────────────────────────────────
def get_trade_decision():
    # ... same logic ...
    if recovery_mode:
        return recovery_decision(probs, patterns)
    else:
        return normal_decision(probs, patterns)

# In on_message → proposal_open_contract handling:
    # ... keep profit/loss logic, but update think messages if desired ...
    # Example: think(f"{'✅ WIN' if profit>0 else '❌ LOSS'} on {'Over 5' if barrier=='5' else 'Under 4'}")

# send_proposal, buy_contract unchanged

# In main():
def main():
    section("DERIV DIGIT AI BOT – Under 4 / Over 5 Edition")
    info(f"Symbol={SYMBOL} | Normal: Over 5 only | Recovery: Under 4 or Over 5")
    info(f"Thresholds: Normal >{PROB_THRESHOLD_OVER5} | Recovery >{RECOVERY_OVER_THRESHOLD}/{RECOVERY_UNDER_THRESHOLD}")
    # ... rest same ...

if __name__ == "__main__":
    random.seed(time.time())
    np.random.seed(int(time.time()*1000) % 2**32)
    main()