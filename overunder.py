import websocket
import json
import numpy as np
from math import floor
from flask import Flask, render_template_string
import threading

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
        self.symbol = symbol  # Volatility 75 Index
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
        self.ticks_since_buy = 0
        self.markov = MarkovChain()
        self.history_loaded = False
        self.req_id = 1

        # For dashboard
        self.dashboard_app = Flask(__name__)
        self.setup_dashboard()
        self.dashboard_thread = threading.Thread(target=self.run_dashboard)
        self.dashboard_thread.daemon = True
        self.dashboard_thread.start()

    def setup_dashboard(self):
        @self.dashboard_app.route('/')
        def index():
            html = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Deriv Bot Dashboard</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1 { color: #333; }
                    .stat { margin: 10px 0; font-size: 18px; }
                    .positive { color: green; }
                    .negative { color: red; }
                </style>
            </head>
            <body>
                <h1>Deriv Trading Bot Dashboard</h1>
                <div class="stat">Total Profit: <span class="{{ 'positive' if total_profit > 0 else 'negative' }}">${{ total_profit }}</span></div>
                <div class="stat">Total Loss: <span class="negative">${{ total_loss }}</span></div>
                <div class="stat">Current Stake: ${{ current_stake }}</div>
                <div class="stat">Consecutive Losses: {{ loss }}</div>
                <div class="stat">Count Loss: {{ count_loss }}</div>
                <div class="stat">Active Contract: {{ 'Yes' if active_contract else 'No' }}</div>
                <div class="stat">Last Digit: {{ last_digit if last_digit is not None else 'N/A' }}</div>
                <div class="stat">Recent Digits: {{ recent_digits }}</div>
                <p>Refresh the page to update stats.</p>
            </body>
            </html>
            """
            return render_template_string(html, 
                                          total_profit=self.total_profit,
                                          total_loss=self.total_loss,
                                          current_stake=self.current_stake,
                                          loss=self.loss,
                                          count_loss=self.count_loss,
                                          active_contract=self.active_contract,
                                          last_digit=self.last_digit,
                                          recent_digits=self.recent_digits)

    def run_dashboard(self):
        self.dashboard_app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

    def connect(self):
        ws_url = f"wss://ws.binaryws.com/websockets/v3?app_id={self.app_id}"
        self.ws = websocket.WebSocketApp(ws_url,
                                         on_open=self.on_open,
                                         on_message=self.on_message,
                                         on_error=self.on_error,
                                         on_close=self.on_close)
        self.ws.run_forever()

    def send(self, data):
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
        if self.last_digit is None:
            return

        if len(self.recent_digits) < 2:
            print("Skipping trade: Not enough recent digits for analysis")
            return

        tick1 = self.recent_digits[-1]
        tick2 = self.recent_digits[-2]

        if not (tick1 >= 5 and tick1 != 8 and tick2 <= 5):
            print("Skipping trade: Analysis condition not met")
            return

        # Calculate probabilities for both options
        prob_under7 = self.markov.predict_prob_under(self.last_digit, 7)
        prob_over2 = self.markov.predict_prob_over(self.last_digit, 2)

        print(f"Predicted prob under 7: {prob_under7:.2f}")
        print(f"Predicted prob over 2: {prob_over2:.2f}")

        # Choose the option with the higher probability
        if prob_under7 > prob_over2:
            if prob_under7 < 0.6:
                print("Skipping trade: Probability too low for under 7")
                return
            contract_type = "DIGITUNDER"
            prediction = 7
            print(f"Choosing under 7 with prob {prob_under7:.2f}")
        else:
            if prob_over2 < 0.6:
                print("Skipping trade: Probability too low for over 2")
                return
            contract_type = "DIGITOVER"
            prediction = 2
            print(f"Choosing over 2 with prob {prob_over2:.2f}")

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
        self.send(params)

    def on_message(self, ws, message):
        data = json.loads(message)
        
        if 'error' in data:
            print(f"API Error: {data['error']}")
            if self.active_contract and 'contract_id' in data.get('echo_req', {}):
                if data['echo_req']['contract_id'] == self.contract_id:
                    # Reset on error for this contract
                    self.active_contract = False
                    self.contract_id = None
                    self.ticks_since_buy = 0
        
        msg_type = data.get('msg_type')

        if msg_type == 'authorize':
            print("Authorized")
            self.load_history()

        elif msg_type == 'history':
            self.process_history(data)

        elif msg_type == 'tick':
            self.process_tick(data)

        elif msg_type == 'proposal_open_contract':
            print(f"Received contract update: {data}")
            self.process_contract(data)

        elif msg_type == 'buy':
            self.contract_id = data.get('buy', {}).get('contract_id')
            if self.contract_id:
                print(f"Contract bought: {self.contract_id}")
                self.active_contract = True
                self.ticks_since_buy = 0
                self.subscribe_contract()
            else:
                if 'error' in data:
                    print(f"Buy error: {data['error']}")

        else:
            print(f"Unknown message type: {msg_type}, data: {data}")

    def process_history(self, data):
        if not self.history_loaded:
            ticks = data.get('history', {}).get('prices', [])
            last_digits = [floor((price * 100) % 10) for price in ticks]
            self.markov.batch_update(last_digits)
            self.recent_digits = last_digits[-10:]
            self.history_loaded = True
            print("Markov chain initialized with history")
            self.subscribe_ticks()

    def process_tick(self, data):
        quote = data.get('tick', {}).get('quote')
        if quote:
            self.last_digit = floor((quote * 100) % 10)
            print(f"Last digit: {self.last_digit}")

            if hasattr(self, 'prev_digit') and self.prev_digit is not None:
                self.markov.update(self.prev_digit, self.last_digit)
            self.prev_digit = self.last_digit

            self.recent_digits.append(self.last_digit)
            if len(self.recent_digits) > 10:
                self.recent_digits = self.recent_digits[-10:]

            if self.active_contract:
                self.ticks_since_buy += 1
                if self.ticks_since_buy > 10:  # Timeout after 10 ticks without resolution
                    print(f"Contract {self.contract_id} timeout after {self.ticks_since_buy} ticks, resetting")
                    self.active_contract = False
                    self.contract_id = None
                    self.ticks_since_buy = 0
                    print("Treating timeout as loss")
                    self.loss += 1
                    self.total_loss += self.current_stake
                    self.total_profit -= self.current_stake
                    if -self.total_profit >= self.stop_loss:
                        print("Stop loss reached due to timeout")
                        self.ws.close()
            else:
                if self.loss >= self.max_losses:
                    self.loss = 0
                    self.current_stake = self.win_stake

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
        is_sold = contract.get('is_sold', 0)
        if is_sold:
            profit = contract.get('profit', 0)
            self.total_profit += profit
            self.total_loss -= profit
            if self.total_loss < 0:
                self.total_loss = 0

            if profit > 0:
                print("Win")
                self.loss = 0
                self.count_loss = 0
                self.current_stake = self.win_stake
            else:
                print("Loss")
                self.loss += 1

            if self.total_loss > 0:
                self.count_loss += 1
                if self.count_loss == 1:
                    required = self.total_loss * 100 / self.payout_percent
                    self.current_stake = round(required / self.martingale_split, 2)
            else:
                self.count_loss = 0
                self.current_stake = self.win_stake

            if self.current_stake < 0.35:
                self.current_stake = 0.35

            self.active_contract = False
            self.contract_id = None
            self.ticks_since_buy = 0

            if self.total_profit >= self.expected_profit:
                print("Expected profit reached")
                self.ws.close()
            elif -self.total_profit >= self.stop_loss:
                print("Stop loss reached")
                self.ws.close()

    def on_error(self, ws, error):
        print(f"Error: {error}")

    def on_close(self, ws, close_status_code, close_reason):
        print("Connection closed")

# Usage
# Replace with your Deriv token and app_id
token = "y5XlAyZZDrPz764"
app_id = "1089"

bot = DerivBot(token, app_id)
bot.connect()