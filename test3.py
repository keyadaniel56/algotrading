import websocket
import json
import numpy as np
from math import floor

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

    def predict_prob_over(self, current_digit, threshold):
        probs = self.transition[current_digit]
        return np.sum(probs[threshold+1:])

    def predict_prob_under(self, current_digit, threshold):
        probs = self.transition[current_digit]
        return np.sum(probs[:threshold])

class DerivBot:
    def __init__(self, token, app_id, symbol='1HZ75V', duration=1, stake=0.35):
        self.ws = None
        self.token = token
        self.app_id = app_id
        self.symbol = symbol
        self.duration = duration
        
        # Financial Settings
        self.initial_stake = stake
        self.current_stake = stake
        self.win_stake = stake
        self.total_profit = 0
        self.total_loss_to_recover = 0.0
        
        # Limits
        self.expected_profit = stake
        self.stop_loss = 50.0
        
        # Martingale Logic
        self.payout_percent = 39.0
        self.martingale_split = 2  # Split recovery into 2 trades
        self.recovery_count = 0    # Tracks which split step we are on
        
        # State Management
        self.active_contract = False
        self.contract_id = None
        self.last_digit = None
        self.recent_digits = []
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
        data['req_id'] = self.req_id
        self.ws.send(json.dumps(data))
        self.req_id += 1

    def on_open(self, ws):
        print("Connected to Deriv. Authorizing...")
        self.send({"authorize": self.token})

    def buy_contract(self):
        # CRITICAL: Do not open a trade if one is already active
        if self.active_contract:
            return

        if self.last_digit is None or len(self.recent_digits) < 2:
            return

        # Simple Analysis Condition
        tick1 = self.recent_digits[-1]
        tick2 = self.recent_digits[-2]
        if not (tick1 >= 5 and tick1 != 8 and tick2 <= 5):
            return

        # Probability Logic
        prob_under7 = self.markov.predict_prob_under(self.last_digit, 7)
        prob_over2 = self.markov.predict_prob_over(self.last_digit, 2)

        if prob_under7 > prob_over2:
            contract_type, prediction = "DIGITUNDER", 7
        else:
            contract_type, prediction = "DIGITOVER", 2

        print(f"Opening Trade: {contract_type} {prediction} with stake {self.current_stake}")
        
        self.send({
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
            }
        })
        self.active_contract = True # Lock trades immediately

    def process_contract(self, data):
        contract = data.get('proposal_open_contract', {})
        if contract.get('is_sold'):
            profit = contract.get('profit', 0)
            self.total_profit += profit
            
            if profit > 0:
                print(f"WON: ${profit:.2f}. Total Profit: ${self.total_profit:.2f}")
                # If we were in recovery mode, decrement recovery count
                if self.recovery_count > 0:
                    self.recovery_count -= 1
                    self.total_loss_to_recover = max(0, self.total_loss_to_recover - (profit * 1.0))
                
                # If recovery is finished, go back to initial stake
                if self.recovery_count == 0:
                    self.current_stake = self.win_stake
                    self.total_loss_to_recover = 0
            else:
                print(f"LOST: ${abs(profit):.2f}")
                self.total_loss_to_recover += abs(self.current_stake)
                self.recovery_count = self.martingale_split # Set to 2
                
                # Calculate Split Martingale: (Loss / Payout%) / Split_Factor
                # For 0.70 loss: (0.70 / 0.39) / 2 = 1.79 / 2 = ~0.89 + original stake
                # To get 1.33: We need to recover the stake and profit
                needed_to_recover = self.total_loss_to_recover * (100 / self.payout_percent)
                self.current_stake = round(needed_to_recover / self.martingale_split, 2)
                
                if self.current_stake < 0.35: self.current_stake = 0.35
                print(f"New Recovery Stake (Split Martingale): {self.current_stake}")

            # Unlock for next trade
            self.active_contract = False
            self.contract_id = None

    def on_message(self, ws, message):
        data = json.loads(message)
        msg_type = data.get('msg_type')

        if msg_type == 'authorize':
            self.send({"ticks_history": self.symbol, "end": "latest", "count": 100, "style": "ticks"})
        
        elif msg_type == 'history':
            prices = data.get('history', {}).get('prices', [])
            digits = [floor((p * 100) % 10) for p in prices]
            self.markov.batch_update(digits)
            self.send({"ticks": self.symbol, "subscribe": 1})
            print("History loaded. Waiting for ticks...")

        elif msg_type == 'tick':
            quote = data.get('tick', {}).get('quote')
            self.last_digit = floor((quote * 100) % 10)
            self.recent_digits.append(self.last_digit)
            if len(self.recent_digits) > 10: self.recent_digits.pop(0)
            
            # Update Markov
            if len(self.recent_digits) > 1:
                self.markov.update(self.recent_digits[-2], self.recent_digits[-1])
            
            # Check for trade if no active contract
            if not self.active_contract:
                self.buy_contract()

        elif msg_type == 'buy':
            if 'error' in data:
                print(f"Buy Error: {data['error']['message']}")
                self.active_contract = False # Unlock on error
            else:
                self.contract_id = data['buy']['contract_id']
                self.send({"proposal_open_contract": 1, "contract_id": self.contract_id, "subscribe": 1})

        elif msg_type == 'proposal_open_contract':
            self.process_contract(data)

    def on_error(self, ws, error): print(f"Error: {error}")
    def on_close(self, ws, a, b): print("Connection Closed")

# Run
token = "y5XlAyZZDrPz764" # Ensure this is your actual token
bot = DerivBot(token, "1089")
bot.connect()