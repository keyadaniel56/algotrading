# deriv_digit_ai_bot.py
# Requirements:
# pip install websocket-client torch numpy

import json
import math
import random
import signal
import sys
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import websocket

# ────────────────────────────────────────────────
#               CONFIGURABLE PARAMETERS
# ────────────────────────────────────────────────
APP_ID = "1089"                     # Your Deriv app ID
AUTH_TOKEN = "WFabi7aeCbFjgvp"      # Your API token (keep secret!)
SYMBOL = "R_100"                    # Volatility 100 Index
DURATION = 1                        # ticks
DURATION_UNIT = "t"
CURRENCY = "USD"
INITIAL_STAKE = 1.0
MARTINGALE = False
MARTINGALE_MULTIPLIER = 2.0
MAX_CONSECUTIVE_LOSSES = 3
STOP_LOSS = -100.0
TAKE_PROFIT = 100.0

# AI / Strategy parameters
WINDOW_SIZE = 100
EMBEDDING_DIM = 8
HIDDEN_SIZE = 32
LEARNING_RATE = 0.005
TRAIN_EPOCHS_INITIAL = 80
TRAIN_EPOCHS_UPDATE = 8
UPDATE_MODEL_EVERY = 12             # new ticks after initial training
PROB_THRESHOLD_UNDER = 0.18         # enter UNDER if predicted P(digit<2) > this
PROB_THRESHOLD_OVER = 0.82          # enter OVER if predicted P(digit>1) > this

WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"

# ────────────────────────────────────────────────
#                 GLOBAL STATE
# ────────────────────────────────────────────────
last_digits: deque[int] = deque(maxlen=WINDOW_SIZE)
total_profit = 0.0
current_stake = INITIAL_STAKE
consecutive_losses = 0
win_low = win_high = loss_low = loss_high = 0

model: Optional[nn.Module] = None
optimizer: Optional[optim.Optimizer] = None
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ws: Optional[websocket.WebSocketApp] = None
running = True

lock = threading.Lock()

# ────────────────────────────────────────────────
#                    LSTM MODEL
# ────────────────────────────────────────────────
class DigitLSTM(nn.Module):
    def __init__(self, vocab_size=10, embed_dim=8, hidden_size=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len) → long tensor of digit indices
        embeds = self.embedding(x)                     # → (batch, seq_len, embed)
        lstm_out, _ = self.lstm(embeds)                # → (batch, seq_len, hidden)
        out = self.fc(lstm_out[:, -1, :])              # last hidden → logits
        return out

def init_model():
    global model, optimizer
    model = DigitLSTM(vocab_size=10, embed_dim=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_on_sequence(seq: List[int], epochs: int):
    if len(seq) < 2:
        return
    model.train()
    data = torch.tensor(seq[:-1], dtype=torch.long, device=device).unsqueeze(0)   # (1, len-1)
    targets = torch.tensor(seq[1:],  dtype=torch.long, device=device).unsqueeze(0)  # (1, len-1)

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(data)                        # (1, vocab)
        loss = criterion(logits, targets[:, -1])    # predict next after whole prefix
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

def predict_next_probabilities(last_digit: int) -> Optional[np.ndarray]:
    if model is None:
        return None
    model.eval()
    with torch.no_grad():
        inp = torch.tensor([[last_digit]], dtype=torch.long, device=device)
        logits = model(inp)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    return probs

# ────────────────────────────────────────────────
#                 DERIV API MESSAGES
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
#                   WEBSOCKET CALLBACKS
# ────────────────────────────────────────────────
def on_message(ws, message):
    global total_profit, current_stake, consecutive_losses
    global win_low, win_high, loss_low, loss_high

    try:
        data = json.loads(message)
    except:
        print("Invalid JSON received")
        return

    msg_type = data.get("msg_type")

    if "error" in data:
        print(f"API Error: {data['error']}")
        return

    if msg_type == "tick":
        quote = data["tick"]["quote"]
        last_digit = int(math.floor(quote * 100)) % 10

        with lock:
            last_digits.append(last_digit)

            # Train / update model
            if len(last_digits) == WINDOW_SIZE and model is None:
                print("Initializing LSTM model...")
                init_model()
                train_on_sequence(list(last_digits), TRAIN_EPOCHS_INITIAL)
                print("Model initialized")
            elif len(last_digits) > WINDOW_SIZE and (len(last_digits) - WINDOW_SIZE) % UPDATE_MODEL_EVERY == 0:
                print("Updating model...")
                train_on_sequence(list(last_digits), TRAIN_EPOCHS_UPDATE)
                print("Model updated")

            decision = get_trade_decision()
            if decision:
                execute_trade(decision)

    elif msg_type == "proposal":
        prop = data["proposal"]
        print(f"Proposal → ask: {prop['ask_price']:.4f} | payout: {prop['payout']:.4f}")
        buy_contract(prop["id"], prop["ask_price"])

    elif msg_type == "buy":
        contract_id = data["buy"]["contract_id"]
        print(f"Contract bought → ID: {contract_id}")
        subscribe_contract(contract_id)

    elif msg_type == "proposal_open_contract":
        poc = data["proposal_open_contract"]
        if poc.get("is_sold", 0) == 1:
            profit = poc.get("profit", 0.0)
            total_profit += profit
            print(f"Trade closed → Profit: {profit:.2f} | Total: {total_profit:.2f}")

            won = profit > 0
            if won:
                consecutive_losses = 0
                current_stake = INITIAL_STAKE
            else:
                consecutive_losses += 1
                if MARTINGALE:
                    current_stake *= MARTINGALE_MULTIPLIER

            # Update stats (for logging / future adaptive bias if you want)
            # ...

            check_risk_limits()

    elif msg_type == "authorize":
        print("Authorized successfully")
        subscribe_ticks()

def on_error(ws, error):
    print(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed")
    global running
    running = False

def on_open(ws):
    print("WebSocket connected")
    authorize()

# ────────────────────────────────────────────────
#                   STRATEGY LOGIC
# ────────────────────────────────────────────────
def get_trade_decision() -> Optional[str]:
    if len(last_digits) < WINDOW_SIZE or model is None:
        return None

    last = last_digits[-1]
    probs = predict_next_probabilities(last)
    if probs is None:
        return None

    p_under = probs[0] + probs[1]          # digit 0 or 1
    p_over  = sum(probs[2:])               # 2–9

    print(f"AI probs → Under(0-1): {p_under:.4f} | Over(2-9): {p_over:.4f}")

    if p_under > PROB_THRESHOLD_UNDER:
        return "DIGITUNDER"
    if p_over > PROB_THRESHOLD_OVER:
        return "DIGITOVER"

    return None

def execute_trade(contract_type: str):
    barrier = "1"   # DigitOver 1 → barrier=1  (payout on 2-9)
                    # DigitUnder 1 → barrier=1 (payout on 0-1)
    print(f"→ Sending proposal for {contract_type}")
    send_proposal(contract_type, barrier)

# ────────────────────────────────────────────────
#                 RISK MANAGEMENT
# ────────────────────────────────────────────────
def check_risk_limits():
    global running
    if total_profit <= STOP_LOSS:
        print(f"STOP LOSS hit → {total_profit:.2f}")
        running = False
    elif total_profit >= TAKE_PROFIT:
        print(f"TAKE PROFIT hit → {total_profit:.2f}")
        running = False
    elif consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
        print(f"Max consecutive losses ({consecutive_losses}) reached")
        running = False

# ────────────────────────────────────────────────
#                 GRACEFUL SHUTDOWN
# ────────────────────────────────────────────────
def signal_handler(sig, frame):
    print("\nShutting down...")
    global running
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
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        ws.close()
        print("Bot stopped.")

if __name__ == "__main__":
    random.seed(time.time())
    np.random.seed(int(time.time()))
    torch.manual_seed(int(time.time()))
    main()