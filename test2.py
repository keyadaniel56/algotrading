import websocket
import json
import numpy as np
from math import floor

# Simple Markov Chain for last digits prediction
class MarkovChain:
    def __init__(self, states=10):
        self.states = states
        self.transition = np.zeros((states, states))
        self.counts = np.zeros((states, states))
        for i in range(states):
            self.transition[i] = np.ones(states) / states

    def update(self, from_digit, to_digit):
        self.counts[from_digit][to_digit] += 1
        total = np.sum(self.counts[from_digit])
        if total > 0:
            self.transition[from_digit] = self.counts[from_digit] / total

    def batch_update(self, digits):
        for i in range(1, len(digits)):
            self.counts[digits[i-1]][digits[i]] += 1
        for from_digit in range(self.states):
            total = np.sum(self.counts[from_digit])
            if total > 0:
                self.transition[from_digit] = self.counts[from_digit] / total
            else:
                self.transition[from_digit] = np.ones(self.states) / self.states

    def predict_prob_over(self, current_digit, threshold):
        probs = self.transition[current_digit]
        return np.sum(probs[threshold+1:])  # P > threshold

    def predict_prob_under(self, current_digit, threshold):
        probs = self.transition[current_digit]
        return np.sum(probs[:threshold])  # P < threshold

# Deriv Bot Class
class DerivBot:
    def __init__(self, token, app_id, symbol='1HZ75V', duration=1, stake=0.35, win_stake=0.35, expected_profit=0.70, stop_loss=10, max_losses=6):
        self.ws = None
        self.token = token
        self.app_id = app_id
        self.symbol = symbol
        self.duration = duration
        self.stake = stake
        self.current_stake = round(stake, 2)
        self.win_stake = round(win_stake, 2)
        self.loss = 0
        self.count_loss = 0
        self.total_profit = 0
        self.total_loss = 0.0
        self.expected_profit = expected_profit
        self.stop_loss = stop_loss
        self.max_losses = max_losses
        self.payout_percent = 39.0
        self.martingale_split = 2.0
        self.last_digit = None
        self.recent_digits = []
        self.contract_id = None
        self.active_contract = False
        self.is_running = True  # NEW: Master kill switch
        self.ticks_since_buy = 0
        self.markov = MarkovChain()
        self.history_loaded = False
        self.req_id = 1

    def connect(self):
        ws_url = f"wss://ws.binaryws.com/websockets/v3?app_id={self.app_id}"
        self.ws = websocket.WebSocketApp(ws_url,
                                         on_open=self.on_open,
                                         on_message=self.on_message,
                                         on_error=self.on_error,
                                         on_close=self.on_close)
        self.ws.run_forever()

    def send(self, data):
        if self.ws and self.ws.sock and self.ws.sock.connected:
            self.ws.send(json.dumps(data))
            self.req_id += 1

    def on_open(self, ws):
        print("Connection opened")
        self.authorize()

    def authorize(self):
        self.send({
            "authorize": self.token,
            "req_id": self.req_id
        })

    def load_history(self):
        self.send({
            "ticks_history": self.symbol,
            "end": "latest",
            "count": 1000,
            "req_id": self.req_id,
            "style": "ticks"
        })

    def subscribe_ticks(self):
        self.send({
            "ticks": self.symbol,
            "subscribe": 1,
            "req_id": self.req_id
        })

    def buy_contract(self):
        # NEW: Check if bot should stop before buying
        if not self.is_running:
            return

        if self.last_digit is None:
            return

        if len(self.recent_digits) < 2:
            return

        tick1 = self.recent_digits[-1]
        tick2 = self.recent_digits[-2]

        if not (tick1 >= 5 and tick1 != 8 and tick2 <= 5):
            return

        prob_under7 = self.markov.predict_prob_under(self.last_digit, 7)
        prob_over2 = self.markov.predict_prob_over(self.last_digit, 2)

        if prob_under7 > prob_over2:
            if prob_under7 < 0.6: return
            contract_type = "DIGITUNDER"
            prediction = 7
        else:
            if prob_over2 < 0.6: return
            contract_type = "DIGITOVER"
            prediction = 2

        params = {
            "buy": 1,
            "price": self.current_stake,
            "parameters": {
                "amount": self.current_stake,
                "basis": "stake",
                "contract_type": contract_type,
                "currency": "USD",
                "duration": self.duration,
                "duration_unit": "t",
                "symbol": self.symbol,
                "barrier": str(prediction)
            },
            "req_id": self.req_id
        }
        print(f"Executing Trade: {contract_type} {prediction} | Stake: {self.current_stake}")
        self.send(params)

    def on_message(self, ws, message):
        data = json.loads(message)
        
        if 'error' in data:
            print(f"API Error: {data['error']}")
            return
        
        msg_type = data.get('msg_type')

        if msg_type == 'authorize':
            print("Authorized")
            self.load_history()

        elif msg_type == 'history':
            self.process_history(data)

        elif msg_type == 'tick':
            self.process_tick(data)

        elif msg_type == 'proposal_open_contract':
            self.process_contract(data)

        elif msg_type == 'buy':
            self.contract_id = data.get('buy', {}).get('contract_id')
            if self.contract_id:
                self.active_contract = True
                self.ticks_since_buy = 0
                self.subscribe_contract()

    def process_history(self, data):
        if not self.history_loaded:
            ticks = data.get('history', {}).get('prices', [])
            last_digits = [floor((price * 100) % 10) for price in ticks]
            self.markov.batch_update(last_digits)
            self.recent_digits = last_digits[-10:]
            self.history_loaded = True
            print("Markov chain initialized. Starting subscription...")
            self.subscribe_ticks()

    def process_tick(self, data):
        # NEW: Ignore ticks if targets met
        if not self.is_running:
            return

        quote = data.get('tick', {}).get('quote')
        if quote:
            self.last_digit = floor((quote * 100) % 10)

            if hasattr(self, 'prev_digit') and self.prev_digit is not None:
                self.markov.update(self.prev_digit, self.last_digit)
            self.prev_digit = self.last_digit

            self.recent_digits.append(self.last_digit)
            if len(self.recent_digits) > 10:
                self.recent_digits = self.recent_digits[-10:]

            if self.active_contract:
                self.ticks_since_buy += 1
                if self.ticks_since_buy > 15: # Safety reset
                    self.active_contract = False
            else:
                self.buy_contract()

    def subscribe_contract(self):
        self.send({
            "proposal_open_contract": 1,
            "contract_id": self.contract_id,
            "subscribe": 1,
            "req_id": self.req_id
        })

    def process_contract(self, data):
        contract = data.get('proposal_open_contract', {})
        if not contract.get('is_sold'):
            return

        profit = contract.get('profit', 0)
        self.total_profit += profit
        
        print(f"Contract Closed. Profit: {profit:.2f} | Total: {self.total_profit:.2f}")

        if profit > 0:
            print("Outcome: WIN")
            self.loss = 0
            self.total_loss = 0
            self.current_stake = self.win_stake
        else:
            print("Outcome: LOSS")
            self.loss += 1
            self.total_loss += abs(profit)
            # Simple Martingale logic
            required = self.total_loss * (1 + (100 / self.payout_percent))
            self.current_stake = round(required / self.martingale_split, 2)

        if self.current_stake < 0.35:
            self.current_stake = 0.35

        self.active_contract = False
        self.contract_id = None

        # NEW: Final Check for targets
        if self.total_profit >= self.expected_profit:
            print("--- TARGET REACHED ---")
            self.is_running = False
            self.ws.close()
        elif self.total_profit <= -self.stop_loss:
            print("--- STOP LOSS REACHED ---")
            self.is_running = False
            self.ws.close()

    def on_error(self, ws, error):
        print(f"Error: {error}")

    def on_close(self, ws, close_status_code, close_reason):
        print("Bot Stopped. Connection closed.")

# Usage
token = "y5XlAyZZDrPz764" # Ensure this is your real token
app_id = "1089"

bot = DerivBot(token, app_id)
bot.connect()