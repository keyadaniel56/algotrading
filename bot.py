# deriv_digit_ai_bot_numpy.py
# Requirements: pip install websocket-client numpy

import json
import math
import random
import signal
import sys
import threading
import time
from collections import deque
from typing import Optional, Tuple

import numpy as np
import websocket

# ────────────────────────────────────────────────
#               CONFIGURABLE PARAMETERS
# ────────────────────────────────────────────────
APP_ID = "1089"
AUTH_TOKEN = "WFabi7aeCbFjgvp"      # ← CHANGE THIS TO YOUR REAL TOKEN
SYMBOL = "R_100"
DURATION = 1
DURATION_UNIT = "t"
CURRENCY = "USD"

INITIAL_STAKE = 1.0
MARTINGALE = False
MARTINGALE_MULTIPLIER = 2.0
MAX_CONSECUTIVE_LOSSES = 3
STOP_LOSS = -100.0
TAKE_PROFIT = 100.0

WINDOW_SIZE = 100
HIDDEN_SIZE = 24
LEARNING_RATE = 0.008
TRAIN_EPOCHS_INITIAL = 60
TRAIN_EPOCHS_UPDATE = 6
UPDATE_MODEL_EVERY = 10

# Balanced thresholds – harder to enter Under 1, easier for Over 1
PROB_THRESHOLD_UNDER = 0.28
PROB_THRESHOLD_OVER  = 0.72

# Recovery re-analysis thresholds (slightly more aggressive)
RECOVERY_PROB_THRESHOLD_UNDER = 0.26
RECOVERY_PROB_THRESHOLD_OVER  = 0.68

# Recovery barriers to cycle through when re-analysis is inconclusive
RECOVERY_BARRIERS = ["3", "3", "6"]     # 2× Over 3, 1× Under 6
RECOVERY_TRADES_COUNT = 2

WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"

# ────────────────────────────────────────────────
#                 GLOBAL STATE
# ────────────────────────────────────────────────
last_digits = deque(maxlen=WINDOW_SIZE)
total_profit = 0.0
current_stake = INITIAL_STAKE
consecutive_losses = 0

# Model parameters (NumPy arrays)
Wxh = None
Whh = None
Why = None
bh  = None
by  = None

# Recovery logic
recovery_mode          = False
recovery_trades_left   = 0
recovery_barrier_index = 0

# ── ONE-TRADE-AT-A-TIME gate ──────────────────────
trade_in_progress = False   # True while waiting for open contract to close
ticks_since_trade  = 0      # cooldown counter between trades

ws      = None
running = True
lock    = threading.Lock()


# ────────────────────────────────────────────────
#     LOGGING HELPERS
# ────────────────────────────────────────────────

def think(msg: str):
    """Print a 'thinking' line with consistent prefix."""
    print(f"  🧠 {msg}")

def info(msg: str):
    print(f"  ℹ️  {msg}")

def warn(msg: str):
    print(f"  ⚠️  {msg}")

def trade_log(msg: str):
    print(f"  💰 {msg}")

def section(title: str):
    bar = "─" * 80
    print(f"\n{bar}\n  {title}\n{bar}")


# ────────────────────────────────────────────────
#     Simple GRU-like recurrent net (NumPy only)
# ────────────────────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def init_model():
    global Wxh, Whh, Why, bh, by
    np.random.seed(42)
    scale_x = np.sqrt(2.0 / (10 + HIDDEN_SIZE))
    scale_h = np.sqrt(2.0 / (HIDDEN_SIZE + HIDDEN_SIZE))
    Wxh = np.random.randn(HIDDEN_SIZE, 10) * scale_x
    Whh = np.random.randn(HIDDEN_SIZE, HIDDEN_SIZE) * scale_h
    Why = np.random.randn(10, HIDDEN_SIZE) * scale_h
    bh  = np.zeros((HIDDEN_SIZE, 1))
    by  = np.zeros((10, 1))


def one_hot(digit: int):
    vec = np.zeros((10, 1))
    vec[digit] = 1.0
    return vec


def predict_next_probabilities(last_digit: int) -> Optional[np.ndarray]:
    if Wxh is None:
        return None
    x = one_hot(last_digit)
    h = np.zeros((HIDDEN_SIZE, 1))
    z = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    h = z * 0.6 + h * 0.4
    logits = np.dot(Why, h) + by
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    return probs.flatten()


def train_on_sequence(seq: list, epochs: int):
    global Wxh, Whh, Why, bh, by

    if len(seq) < 2:
        return

    h_prev_global = np.zeros((HIDDEN_SIZE, 1))

    for ep in range(epochs):
        total_loss = 0.0
        h_prev_global = np.zeros((HIDDEN_SIZE, 1))

        for i in range(len(seq) - 1):
            x      = one_hot(seq[i])
            target = seq[i + 1]

            z      = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h_prev_global) + bh)
            h      = z * 0.6 + h_prev_global * 0.4
            logits = np.dot(Why, h) + by
            exp_l  = np.exp(logits - np.max(logits))
            probs  = exp_l / np.sum(exp_l)

            loss        = -np.log(probs[target] + 1e-12)
            total_loss += loss

            d_logits       = probs.copy()
            d_logits[target] -= 1.0
            d_Why          = np.dot(d_logits, h.T)
            d_by           = d_logits
            d_h            = np.dot(Why.T, d_logits)

            Why -= LEARNING_RATE * d_Why
            by  -= LEARNING_RATE * d_by.reshape(-1, 1)
            Wxh -= LEARNING_RATE * np.dot(d_h, x.T) * 0.3
            Whh -= LEARNING_RATE * np.dot(d_h, h_prev_global.T) * 0.3
            bh  -= LEARNING_RATE * d_h * 0.3

            h_prev_global = h

        if ep % 20 == 0:
            think(f"[epoch {ep:2d}] training loss: {total_loss / (len(seq)-1):.4f}")


# ────────────────────────────────────────────────
#     ANALYSIS DISPLAY
# ────────────────────────────────────────────────

def show_analysis(last_digit: int, probs: np.ndarray = None, mode_label: str = "Normal"):
    section(f"MARKET ANALYSIS  [{mode_label} mode]")
    think(f"Last digit observed : {last_digit}")
    think(f"Window size         : {len(last_digits)} ticks")

    if len(last_digits) > 0:
        counts = np.bincount(list(last_digits), minlength=10)
        total  = len(last_digits)
        think("Digit frequency over window:")
        for i in range(10):
            bar = "█" * int(counts[i] / total * 40)
            print(f"       {i} : {bar:<40} {counts[i]:3d}  ({counts[i]/total*100:5.1f}%)")

        # Hot / cold digits
        hot  = np.argsort(counts)[-3:][::-1]
        cold = np.argsort(counts)[:3]
        think(f"Hot  digits (most frequent) : {list(hot)}")
        think(f"Cold digits (least frequent): {list(cold)}")

    if probs is not None:
        p_under1 = probs[0] + probs[1]
        p_over1  = 1.0 - p_under1
        p_under6 = float(np.sum(probs[:6]))
        p_over3  = float(np.sum(probs[4:]))

        think("Model probability predictions:")
        print(f"       Under 1  (digit 0–1) : {p_under1:6.1%}")
        print(f"       Over  1  (digit 2–9) : {p_over1:6.1%}")
        print(f"       Under 6  (digit 0–5) : {p_under6:6.1%}")
        print(f"       Over  3  (digit 4–9) : {p_over3:6.1%}")
        print()

        think("Per-digit probability breakdown:")
        for i in range(10):
            bar = "▓" * int(probs[i] * 100)
            print(f"       digit {i} : {bar:<20} {probs[i]:.3f}")

    print()


# ────────────────────────────────────────────────
#     DECISION LOGIC
# ────────────────────────────────────────────────

def analyze_market_for_trade(in_recovery: bool = False) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Core analysis routine.
    Returns (contract_type, barrier, display_name) or (None, None, None).
    Uses slightly more aggressive thresholds in recovery mode.
    """
    if len(last_digits) < WINDOW_SIZE:
        think(f"Not enough data yet ({len(last_digits)}/{WINDOW_SIZE} ticks) – skipping analysis")
        return None, None, None

    last  = last_digits[-1]
    probs = predict_next_probabilities(last)
    if probs is None:
        think("Model not ready yet – skipping analysis")
        return None, None, None

    mode_label = "RECOVERY" if in_recovery else "Normal"
    show_analysis(last, probs, mode_label)

    p_under1 = float(probs[0] + probs[1])
    p_over1  = 1.0 - p_under1
    p_under6 = float(np.sum(probs[:6]))
    p_over3  = float(np.sum(probs[4:]))

    thresh_under = RECOVERY_PROB_THRESHOLD_UNDER if in_recovery else PROB_THRESHOLD_UNDER
    thresh_over  = RECOVERY_PROB_THRESHOLD_OVER  if in_recovery else PROB_THRESHOLD_OVER

    # Small boost toward Over 1 when model is uncertain
    p_over1_adjusted = p_over1 + 0.06 if p_over1 > 0.50 else p_over1

    think(f"Evaluating signals (thresholds: under={thresh_under}, over={thresh_over}):")

    # ── Under 1 signal ──────────────────────────────────────
    think(f"  p_under1={p_under1:.4f} vs threshold {thresh_under} → {'✅ SIGNAL' if p_under1 > thresh_under else '❌ no signal'}")
    if p_under1 > thresh_under:
        think(f"Under 1 signal confirmed! p={p_under1:.1%}")
        return "DIGITUNDER", "1", f"Under 1  (p={p_under1:.1%})"

    # ── Over 1 signal ───────────────────────────────────────
    think(f"  p_over1_adj={p_over1_adjusted:.4f} vs threshold {thresh_over} → {'✅ SIGNAL' if p_over1_adjusted > thresh_over else '❌ no signal'}")
    if p_over1_adjusted > thresh_over:
        think(f"Over 1 signal confirmed! adjusted p={p_over1_adjusted:.1%}")
        return "DIGITOVER", "1", f"Over 1   (adj p={p_over1_adjusted:.1%})"

    # ── Recovery-only: try Over 3 / Under 6 ─────────────────
    if in_recovery:
        think(f"  Recovery fallback → checking Over 3 (p={p_over3:.4f}) and Under 6 (p={p_under6:.4f})")
        if p_over3 >= 0.60:
            think(f"Recovery Over 3 signal! p={p_over3:.1%}")
            return "DIGITOVER", "3", f"Over 3   (recovery, p={p_over3:.1%})"
        if p_under6 >= 0.60:
            think(f"Recovery Under 6 signal! p={p_under6:.1%}")
            return "DIGITUNDER", "6", f"Under 6  (recovery, p={p_under6:.1%})"

        think("Recovery re-analysis found no strong signal – skipping trade this tick")

    think("No trade signal found this tick – waiting for a clearer opportunity")
    return None, None, None


def get_trade_decision() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Gate function: respects trade_in_progress and routes to analysis.
    """
    global trade_in_progress, ticks_since_trade

    # ── One-trade-at-a-time guard ────────────────────────────
    if trade_in_progress:
        think("A trade is currently open – waiting for it to close before opening another")
        return None, None, None

    # ── Minimum cooldown between trades (1 tick) ─────────────
    ticks_since_trade += 1
    if ticks_since_trade < 2:
        think(f"Cooldown: {ticks_since_trade}/2 ticks since last trade – holding off")
        return None, None, None

    return analyze_market_for_trade(in_recovery=recovery_mode)


# ────────────────────────────────────────────────
#     DERIV API FUNCTIONS
# ────────────────────────────────────────────────

def authorize():
    ws.send(json.dumps({"authorize": AUTH_TOKEN}))


def subscribe_ticks():
    ws.send(json.dumps({"ticks": SYMBOL, "subscribe": 1}))


def send_proposal(contract_type: str, barrier: str):
    payload = {
        "proposal": 1,
        "amount": current_stake,
        "basis": "stake",
        "contract_type": contract_type,
        "currency": CURRENCY,
        "duration": DURATION,
        "duration_unit": DURATION_UNIT,
        "symbol": SYMBOL,
        "barrier": barrier
    }
    ws.send(json.dumps(payload))


def buy_contract(proposal_id: int, ask_price: float):
    ws.send(json.dumps({"buy": str(proposal_id), "price": ask_price}))


def subscribe_contract(contract_id: int):
    ws.send(json.dumps({
        "proposal_open_contract": 1,
        "contract_id": contract_id,
        "subscribe": 1
    }))


# ────────────────────────────────────────────────
#     WEBSOCKET CALLBACKS
# ────────────────────────────────────────────────

def on_message(ws_obj, message):
    global total_profit, current_stake, consecutive_losses
    global recovery_mode, recovery_trades_left
    global trade_in_progress, ticks_since_trade

    try:
        data = json.loads(message)
    except Exception:
        warn("Invalid JSON received – ignoring")
        return

    if "error" in data:
        warn(f"API Error: {data['error'].get('message', data['error'])}")
        # If proposal fails while trade is in-progress flag, unblock
        if data["error"].get("code") in ("ContractCreationFailure", "InvalidBarrier"):
            with lock:
                trade_in_progress = False
        return

    msg_type = data.get("msg_type")

    # ── Tick received ────────────────────────────────────────
    if msg_type == "tick":
        quote  = data["tick"]["quote"]
        digit  = int(math.floor(quote * 100)) % 10

        with lock:
            last_digits.append(digit)
            think(f"Tick received → quote={quote:.5f}  last digit={digit}  "
                  f"[buffer {len(last_digits)}/{WINDOW_SIZE}]")

            # ── Initial model training ───────────────────────
            if len(last_digits) == WINDOW_SIZE and Wxh is None:
                section("INITIALIZING MODEL")
                think("Buffer full – training NumPy sequence model for the first time…")
                init_model()
                train_on_sequence(list(last_digits), TRAIN_EPOCHS_INITIAL)
                info("Model ready ✓\n")

            # ── Periodic model update ────────────────────────
            elif (Wxh is not None and
                  len(last_digits) > WINDOW_SIZE and
                  (len(last_digits) - WINDOW_SIZE) % UPDATE_MODEL_EVERY == 0):
                think("Periodic model update triggered…")
                train_on_sequence(list(last_digits)[-WINDOW_SIZE:], TRAIN_EPOCHS_UPDATE)
                think("Model weights updated ✓\n")

            # ── Trade decision ───────────────────────────────
            contract_type, barrier, display_name = get_trade_decision()
            if contract_type:
                trade_in_progress = True   # lock gate BEFORE sending proposal
                ticks_since_trade = 0
                execute_trade(contract_type, barrier, display_name)

    # ── Proposal received ────────────────────────────────────
    elif msg_type == "proposal":
        p = data["proposal"]
        think(f"Proposal received → ask_price={p['ask_price']:.4f}  payout={p['payout']:.4f}")
        think(f"Accepting proposal and buying contract…")
        buy_contract(p["id"], p["ask_price"])

    # ── Buy confirmed ────────────────────────────────────────
    elif msg_type == "buy":
        cid = data["buy"]["contract_id"]
        trade_log(f"Contract opened → ID: {cid}  stake={current_stake:.2f}")
        think("Monitoring contract until it settles…")
        subscribe_contract(cid)

    # ── Contract settled ─────────────────────────────────────
    elif msg_type == "proposal_open_contract":
        poc = data["proposal_open_contract"]

        if poc.get("is_sold", 0) == 1:
            profit = poc.get("profit", 0.0)
            total_profit += profit

            section("TRADE RESULT")
            if profit > 0:
                trade_log(f"WIN  ✅  profit={profit:+.2f}   session total={total_profit:+.2f}")
            else:
                trade_log(f"LOSS ❌  profit={profit:+.2f}   session total={total_profit:+.2f}")

            with lock:
                trade_in_progress = False   # ── UNBLOCK the gate ──
                ticks_since_trade = 0
                think("Trade closed – gate unlocked, ready for next opportunity")

                if profit > 0:
                    consecutive_losses = 0
                    current_stake = INITIAL_STAKE

                    if recovery_mode:
                        recovery_mode        = False
                        recovery_trades_left = 0
                        think("Recovery trade won – exiting recovery mode ✓")
                    else:
                        think("Win recorded – stake reset to initial")

                else:
                    consecutive_losses += 1
                    think(f"Loss recorded – consecutive losses: {consecutive_losses}")

                    if MARTINGALE:
                        current_stake *= MARTINGALE_MULTIPLIER
                        think(f"Martingale active – next stake: {current_stake:.2f}")

                    if not recovery_mode:
                        recovery_mode        = True
                        recovery_trades_left = RECOVERY_TRADES_COUNT
                        think(f"Entering RECOVERY MODE – will re-analyse market "
                              f"before each of the next {RECOVERY_TRADES_COUNT} trades")
                    else:
                        recovery_trades_left -= 1
                        think(f"Still in recovery – {recovery_trades_left} recovery trades remaining")
                        if recovery_trades_left <= 0:
                            recovery_mode = False
                            think("Recovery trades exhausted – returning to normal mode")

            check_risk_limits()

    # ── Authorize ────────────────────────────────────────────
    elif msg_type == "authorize":
        info("Authorised successfully")
        think("Subscribing to tick stream…")
        subscribe_ticks()


def on_error(ws_obj, err):
    global trade_in_progress
    warn(f"WebSocket error: {err}")
    # Safety: release gate on websocket errors so bot doesn't freeze
    with lock:
        trade_in_progress = False


def on_close(ws_obj, code, msg):
    global running
    info("WebSocket connection closed")
    running = False


def on_open(ws_obj):
    info("WebSocket connected")
    think("Sending authorisation token…")
    authorize()


# ────────────────────────────────────────────────
#     TRADE EXECUTION
# ────────────────────────────────────────────────

def execute_trade(contract_type: str, barrier: str, display_name: str):
    section("OPENING TRADE")
    trade_log(f"Signal : {display_name}")
    trade_log(f"Type   : {contract_type}  barrier={barrier}  stake={current_stake:.2f}")
    think("Requesting price proposal from Deriv…")
    send_proposal(contract_type, barrier)


# ────────────────────────────────────────────────
#     RISK MANAGEMENT
# ────────────────────────────────────────────────

def check_risk_limits():
    global running
    if total_profit <= STOP_LOSS:
        warn(f"STOP LOSS triggered at {total_profit:.2f} – shutting down bot")
        running = False
    elif total_profit >= TAKE_PROFIT:
        info(f"TAKE PROFIT reached at {total_profit:.2f} – shutting down bot")
        running = False
    elif consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
        warn(f"Max consecutive losses ({consecutive_losses}) reached – shutting down bot")
        running = False


# ────────────────────────────────────────────────
#     SHUTDOWN HANDLING
# ────────────────────────────────────────────────

def signal_handler(sig, frame):
    global running
    print("\n")
    info("Interrupt received – shutting down gracefully…")
    running = False
    if ws:
        ws.close()


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ────────────────────────────────────────────────
#                      MAIN
# ────────────────────────────────────────────────

def main():
    global ws

    section("DERIV DIGIT AI BOT  –  NumPy Edition")
    info(f"Symbol       : {SYMBOL}")
    info(f"Duration     : {DURATION}{DURATION_UNIT}")
    info(f"Initial stake: {INITIAL_STAKE}")
    info(f"Stop loss    : {STOP_LOSS}")
    info(f"Take profit  : {TAKE_PROFIT}")
    info(f"Martingale   : {'ON  x' + str(MARTINGALE_MULTIPLIER) if MARTINGALE else 'OFF'}")
    info(f"Window size  : {WINDOW_SIZE} ticks")
    think("Connecting to Deriv WebSocket API…\n")

    websocket.enableTrace(False)

    ws = websocket.WebSocketApp(
        WS_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    wst = threading.Thread(target=ws.run_forever, daemon=True)
    wst.start()

    try:
        while running:
            time.sleep(0.5)
    finally:
        if ws:
            ws.close()
        section("BOT STOPPED")
        info(f"Final session P&L : {total_profit:+.2f}")
        info(f"Consecutive losses: {consecutive_losses}")


if __name__ == "__main__":
    random.seed(time.time())
    np.random.seed(int(time.time() * 1000) % 2**32)
    main()