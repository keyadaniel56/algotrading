# improved_fixed_deriv_hmm_trading_bot_with_ui_and_charts.py
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import websocket
import json
import time
from datetime import datetime, timedelta
import threading
import queue
import warnings
import http.server
import socketserver
import io
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
warnings.filterwarnings('ignore')
import os
class FixedDigitHMMPredictor:
    def __init__(self, n_components=4): # Increased components for better modeling
        self.n_components = n_components
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.sequence_history = []
        self.min_training_samples = 100 # Increased minimum samples for better training
        self.last_training_time = None
       
    def create_model(self):
        """Create a new HMM model with proper initialization"""
        self.model = hmm.GaussianHMM(
            n_components=self.n_components,
            covariance_type="diag",
            n_iter=1000, # Increased iterations
            random_state=42,
            tol=1e-5, # Tighter tolerance
            init_params="stmc"
        )
       
    def prepare_features(self, digit_sequence):
        """Convert digit sequence to robust features for HMM - added more features"""
        if len(digit_sequence) < 5:
            features = [[d, 0.5, 0.5, 0.5, 0.0] for d in digit_sequence]
            return np.array(features)
           
        digits = np.array(digit_sequence).reshape(-1, 1)
        features = []
       
        for i in range(len(digits)):
            current_digit = digits[i][0]
           
            if i < 3:
                window_size = i + 1
            else:
                window_size = min(7, i + 1) # Larger window
               
            start_idx = max(0, i - window_size + 1)
            window = digits[start_idx:i+1, 0]
           
            diff = current_digit - digits[i-1][0] if i > 0 else 0 # Added diff feature
           
            feature = [
                current_digit,
                np.mean(window),
                np.std(window) if len(window) > 1 else 0.5,
                np.mean([1 if x % 2 == 0 else 0 for x in window]),
                diff # New feature: difference from previous
            ]
            features.append(feature)
       
        features_array = np.array(features)
       
        if len(features_array) > 1:
            noise = np.random.normal(0, 0.001, features_array.shape)
            features_array += noise
           
        return features_array
   
    def safe_fit(self, digit_sequence):
        """Safely train HMM model with error handling"""
        if len(digit_sequence) < self.min_training_samples:
            print(f"⚠️ Insufficient data for training: {len(digit_sequence)}/{self.min_training_samples}")
            return False
           
        try:
            unique_digits = len(set(digit_sequence))
            if unique_digits < 4: # Increased minimum unique digits
                print(f"⚠️ Low diversity: {unique_digits} unique digits, skipping training")
                return False
               
            features = self.prepare_features(digit_sequence)
            self.sequence_history = digit_sequence.copy()
           
            features_scaled = self.scaler.fit_transform(features)
           
            self.create_model()
            self.model.fit(features_scaled)
            self.is_fitted = True
            self.last_training_time = datetime.now()
           
            score = self.model.score(features_scaled)
            print(f"✓ Model trained (score: {score:.2f}, diversity: {unique_digits} digits)")
            return True
               
        except Exception as e:
            print(f"❌ Model training failed: {e}")
            self.is_fitted = False
            return False
   
    def predict_next_digit_proba(self, digit_sequence):
        """Predict probability distribution for next digit with safety checks"""
        if not self.is_fitted or len(digit_sequence) < 20: # Increased minimum sequence length
            return None
           
        try:
            current_features = self.prepare_features(digit_sequence)
            current_features_scaled = self.scaler.transform(current_features)
           
            hidden_states = self.model.predict(current_features_scaled)
            current_state = hidden_states[-1]
           
            state_indices = np.where(hidden_states == current_state)[0]
            if len(state_indices) < 3: # Increased minimum instances
                return None
               
            next_digits = []
            for idx in state_indices:
                if idx + 1 < len(digit_sequence):
                    next_digits.append(digit_sequence[idx + 1])
           
            if len(next_digits) < 3:
                return None
               
            digit_counts = {i: 0.01 for i in range(10)}
            for digit in next_digits:
                digit_counts[digit] += 1
               
            total = sum(digit_counts.values())
            probabilities = {digit: count/total for digit, count in digit_counts.items()}
           
            return probabilities
           
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            return None
   
    def predict_low_high_proba(self, digit_sequence):
        """Predict probability of next digit being in low (0,1,2) or high (7,8,9)"""
        digit_proba = self.predict_next_digit_proba(digit_sequence)
       
        if digit_proba is None:
            return {
                'low': 0.3,
                'high': 0.3,
                'confidence': 0.0,
                'reliable': False
            }
           
        low_prob = sum(digit_proba[digit] for digit in [0, 1, 2])
        high_prob = sum(digit_proba[digit] for digit in [7, 8, 9])
       
        confidence = abs(low_prob - high_prob)
       
        reliable = confidence > 0.05 # Fixed: removed bounding on prob, only confidence threshold
       
        return {
            'low': low_prob,
            'high': high_prob,
            'confidence': confidence,
            'reliable': reliable
        }
class DerivWebSocketClient:
    def __init__(self, token, demo=True):
        self.token = token
        self.demo = demo
        self.ws_url = "wss://ws.derivws.com/websockets/v3?app_id=1089" if demo else "wss://ws.derivws.com/websockets/v3?app_id=1090"
        self.ws = None
        self.connected = False
        self.response_queue = queue.Queue()
        self.last_ticks = []
        self.req_id = 1
        self.contract_subscriptions = {}
        self.last_epoch = None # Added for incremental history
        self.balance = 0.0
       
    def connect(self):
        """Connect to Deriv WebSocket"""
        try:
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
           
            self.ws_thread = threading.Thread(target=self.ws.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
           
            timeout = 10
            start_time = time.time()
            while not self.connected and time.time() - start_time < timeout:
                time.sleep(0.1)
               
            return self.connected
           
        except Exception as e:
            print(f"WebSocket connection error: {e}")
            return False
   
    def on_open(self, ws):
        print("✅ WebSocket connected")
        self.connected = True
       
    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            self.response_queue.put(data)
           
            if 'tick' in data:
                self.last_ticks.append(data['tick'])
                if len(self.last_ticks) > 1000:
                    self.last_ticks = self.last_ticks[-1000:]
           
            if 'proposal_open_contract' in data:
                contract_data = data['proposal_open_contract']
                contract_id = contract_data.get('contract_id')
                if contract_id:
                    self.contract_subscriptions[contract_id] = contract_data
           
            if 'balance' in data:
                self.balance = data['balance']['balance']
                   
        except Exception as e:
            print(f"Error processing message: {e}")
       
    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")
       
    def on_close(self, ws, close_status_code, close_msg):
        print("❌ WebSocket disconnected")
        self.connected = False
   
    def send_request(self, request, wait_for_response=True, timeout=10):
        """Send request and optionally wait for response"""
        if not self.connected:
            print("WebSocket not connected")
            return None
           
        request['req_id'] = self.req_id
        self.req_id += 1
       
        self.ws.send(json.dumps(request))
       
        if not wait_for_response:
            return None
          
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = self.response_queue.get(timeout=0.1)
                if response.get('req_id') == request['req_id']:
                    return response
            except queue.Empty:
                continue
               
        print("Timeout waiting for response")
        return None
   
    def authorize(self):
        """Authorize with token"""
        auth_request = {
            "authorize": self.token
        }
        response = self.send_request(auth_request)
        if response and 'error' not in response:
            self.balance = response.get('authorize', {}).get('balance', 0.0)
            print("✅ Authorized successfully")
            return True
        # else:
        # print(f"Authorization failed: {response.get('error', 'Unknown error')}")
        # return False
   
    def get_ticks_history(self, symbol="R_100", count=200): # Increased initial count
        """Get ticks history via WebSocket - improved to fetch incremental data"""
        print(f"Requesting ticks for {symbol}...")
       
        history_request = {
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "end": "latest"
        }
       
        if self.last_epoch is not None:
            history_request["start"] = self.last_epoch + 1
            print(f"Fetching incremental from epoch {history_request['start']}")
        else:
            history_request["count"] = count
            print(f"Fetching initial {count} ticks")
       
        response = self.send_request(history_request, timeout=15)
       
        if response and 'history' in response:
            if 'times' in response['history'] and response['history']['times']:
                self.last_epoch = max(response['history']['times'])
            print(f"✓ Received ticks history with {len(response['history'].get('prices', []))} prices")
            return response
        else:
            print(f"❌ No history in response: {response}")
            return None
   
    def subscribe_ticks(self, symbol="R_100"):
        """Subscribe to tick stream"""
        subscribe_request = {
            "ticks": symbol,
            "subscribe": 1
        }
        return self.send_request(subscribe_request)
   
    def subscribe_contract(self, contract_id):
        """Subscribe to contract updates"""
        subscribe_request = {
            "proposal_open_contract": 1,
            "contract_id": contract_id,
            "subscribe": 1
        }
        return self.send_request(subscribe_request)
class DerivAPIClient:
    def __init__(self, token, demo=True):
        self.token = token
        self.demo = demo
        self.ws_client = DerivWebSocketClient(token, demo)
       
    def connect(self):
        """Connect to Deriv via WebSocket"""
        if self.ws_client.connect():
            if self.ws_client.authorize():
                # Subscribe to balance updates
                self.ws_client.send_request({"balance": 1, "subscribe": 1}, wait_for_response=False)
                print("✅ Fully connected and authorized")
                return True
        print("❌ Failed to connect to Deriv API")
        return False
   
    def get_ticks_history(self, symbol="R_100", count=200):
        """Get historical tick data"""
        return self.ws_client.get_ticks_history(symbol, count)
   
    def get_recent_ticks(self, count=10):
        """Get recently received ticks"""
        return self.ws_client.last_ticks[-count:] if self.ws_client.last_ticks else []
   
    def proposal(self, contract_type, symbol="R_100", amount=1, barrier=None):
        """Get trade proposal"""
        proposal_request = {
            "proposal": 1,
            "amount": amount,
            "basis": "stake",
            "contract_type": contract_type,
            "currency": "USD",
            "duration": 1,
            "duration_unit": "t",
            "symbol": symbol
        }
        if barrier is not None:
            proposal_request["barrier"] = barrier
        return self.ws_client.send_request(proposal_request)
   
    def buy_contract(self, proposal_id, price):
        """Buy contract"""
        buy_request = {
            "buy": proposal_id,
            "price": price
        }
        return self.ws_client.send_request(buy_request)
   
    def subscribe_to_contract(self, contract_id):
        """Subscribe to contract updates"""
        return self.ws_client.subscribe_contract(contract_id)
   
    def get_contract_update(self, contract_id, timeout=30):
        """Get contract update with timeout"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if contract_id in self.ws_client.contract_subscriptions:
                contract_data = self.ws_client.contract_subscriptions[contract_id]
                if contract_data.get('is_expired') or contract_data.get('is_sold'):
                    return contract_data
            time.sleep(0.5)
        return None
   
    def place_trade(self, contract_type, symbol="R_100", amount=1, barrier=None):
        """Place a real trade on Deriv"""
        try:
            print(f"Getting proposal for {contract_type} on {symbol}...")
           
            proposal_data = self.proposal(contract_type, symbol, amount, barrier)
            if not proposal_data or 'error' in proposal_data:
                print(f"Failed to get proposal: {proposal_data.get('error', 'Unknown error')}")
                return None
           
            proposal_id = proposal_data['proposal']['id']
            print(f"Proposal received: {proposal_id}")
           
            buy_data = self.buy_contract(proposal_id, amount)
            if not buy_data or 'error' in buy_data:
                print(f"Failed to buy contract: {buy_data.get('error', 'Unknown error')}")
                return None
           
            print(f"✓ Trade placed successfully - Contract ID: {buy_data['buy']['contract_id']}")
            return buy_data
               
        except Exception as e:
            print(f"Error placing trade: {e}")
            return None
class FixedDerivTradingBot:
    def __init__(self, token=None, demo=True):
        if not token:
            raise ValueError("API token is required for real trading")
           
        self.api_client = DerivAPIClient(token, demo)
        self.demo = demo
        self.hmm_predictor = FixedDigitHMMPredictor()
        self.digit_history = []
        self.trade_history = []
        self.min_confidence = 0.15 # Increased minimum confidence
        self.symbol = "R_100"
       
        # Martingale parameters
        self.martingale_multiplier = 2.0
        self.max_martingale_steps = 8
        self.current_martingale_step = 0
        self.base_amount = 0.35
        self.max_amount = 100.0
       
        # Risk management
        self.risk_per_trade = 0.01 # 1% of balance
        self.max_risk_per_trade = 0.05 # 5% max for martingale steps
        self.min_balance = 10.0
       
        # Trading state
        self.active_trade = None
        self.last_trade_result = None
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.last_retrain_time = None
        self.pending_retrain = False
       
        self.prediction_history = []
       
        # UI control
        self.running = False
        self.trading_thread = None
        self.logs = [] # To store log messages
        self.lock = threading.Lock()
       
        # Take profit and stop loss (session-based)
        self.take_profit = 100.0 # Example: stop if profit >= 100
        self.stop_loss = -50.0 # Example: stop if profit <= -50
        self.current_profit = 0.0
       
        # Load past trades from CSV
        self.csv_file = 'trades.csv'
        try:
            df = pd.read_csv(self.csv_file)
            self.trade_history = df.to_dict(orient='records')
            self.log(f"Loaded {len(self.trade_history)} past trades from CSV")
        except FileNotFoundError:
            self.trade_history = []
            self.log("No existing trades CSV found, starting fresh")
       
    def log(self, message):
        """Add log message"""
        with self.lock:
            timestamp = datetime.now().strftime('%H:%M:%S')
            log_entry = f"[{timestamp}] {message}"
            print(log_entry) # Still print to console
            self.logs.append(log_entry)
            if len(self.logs) > 100: # Keep last 100 logs
                self.logs = self.logs[-100:]
   
    def save_trades_to_csv(self):
        """Save trade history to CSV"""
        columns = ['timestamp', 'contract_type', 'amount', 'profit', 'final_result']
        df = pd.DataFrame([{k: t.get(k, None) for k in columns} for t in self.trade_history])
        df.to_csv(self.csv_file, index=False)
        self.log("Trades saved to CSV")
   
    def extract_digits_from_ticks(self, ticks_data):
        """Extract last digits from real tick prices with better parsing"""
        digits = []
       
        if not ticks_data:
            return digits
           
        self.log(f"🔍 Raw data type: {type(ticks_data)}, keys: {list(ticks_data.keys()) if isinstance(ticks_data, dict) else 'not dict'}")
       
        if isinstance(ticks_data, dict):
            if 'history' in ticks_data and 'prices' in ticks_data['history']:
                prices = ticks_data['history']['prices']
                self.log(f"📊 Found {len(prices)} prices in history")
                for i, price in enumerate(prices):
                    if i < 3:
                        self.log(f" Price {i}: {price} (type: {type(price)})")
                    digit = self._extract_digit_from_price(price)
                    if digit is not None:
                        digits.append(digit)
            else:
                self.log("❌ No 'history' or 'prices' found in response")
        else:
            self.log(f"❌ Unexpected data format: {type(ticks_data)}")
       
        self.log(f"🎯 Extracted {len(digits)} digits: {digits[:10]}{'...' if len(digits) > 10 else ''}")
        return digits
   
    def _extract_digit_from_price(self, price):
        """Better digit extraction from various price formats"""
        try:
            if price is None:
                return None
               
            price_str = str(price).strip()
           
            cleaned = ''.join(c for c in price_str if c.isdigit() or c == '.')
           
            if not cleaned:
                return None
               
            if '.' in cleaned:
                parts = cleaned.split('.')
                if len(parts) == 2:
                    decimal_part = parts[1]
                    if decimal_part:
                        last_digit = int(decimal_part[-1])
                        return last_digit
                integer_part = parts[0]
                if integer_part:
                    return int(integer_part[-1])
            else:
                if cleaned:
                    return int(cleaned[-1])
                   
        except (ValueError, IndexError, TypeError) as e:
            self.log(f"⚠️ Error extracting digit from price {price}: {e}")
            return None
       
        return None
   
    def get_real_ticks_data(self, count=200):
        """Get real ticks data from Deriv WebSocket"""
        self.log(f"📡 Fetching real market data for {self.symbol}...")
       
        ticks_data = self.api_client.get_ticks_history(self.symbol, count)
       
        if not ticks_data:
            self.log("❌ No ticks data received")
            return None
           
        if not isinstance(ticks_data, dict):
            self.log(f"❌ Unexpected data format: {type(ticks_data)}")
            return None
           
        has_prices = (
            ticks_data.get('history') and
            isinstance(ticks_data['history'], dict) and
            ticks_data['history'].get('prices') and
            len(ticks_data['history']['prices']) > 0
        )
       
        if has_prices:
            digits = self.extract_digits_from_ticks(ticks_data)
            if digits:
                unique_digits = len(set(digits))
                self.log(f"✓ Received {len(digits)} data points ({unique_digits} unique digits)")
                return ticks_data
            else:
                self.log("❌ No digits could be extracted from prices")
        else:
            self.log("❌ No price data in response")
       
        return None
   
    def get_live_ticks(self, count=50):
        """Try to get live ticks as fallback"""
        self.log("🔄 Trying to get live ticks...")
        live_ticks = self.api_client.get_recent_ticks(count)
        if live_ticks:
            digits = self.extract_digits_from_ticks({"history": {"prices": [t['quote'] for t in live_ticks]}}) # Adapted for live ticks
            if digits:
                self.log(f"✓ Using {len(digits)} live ticks")
                return {"live_ticks": live_ticks}
        return None
   
    def validate_data_quality(self, digits):
        """Improved validation with better diagnostics"""
        if not digits:
            return False, "No digits extracted"
           
        valid_digits = all(0 <= d <= 9 for d in digits)
        if not valid_digits:
            invalid_digits = [d for d in digits if not (0 <= d <= 9)]
            return False, f"Invalid digit values: {invalid_digits[:5]}"
           
        unique_digits = len(set(digits))
        if unique_digits < 3 and len(digits) >= 20:
            return False, f"Low digit diversity: {unique_digits} unique digits"
        elif unique_digits < 5 and len(digits) >= 50: # Stricter for larger datasets
            return False, f"Low digit diversity: {unique_digits} unique digits"
           
        even_count = sum(1 for d in digits if d % 2 == 0)
        even_ratio = even_count / len(digits)
       
        if even_ratio > 0.9 or even_ratio < 0.1:
            return False, f"Extreme even/odd ratio: {even_ratio:.1%}"
           
        return True, f"Valid: {len(digits)} digits, {unique_digits} unique, {even_ratio:.1%} even"
   
    def update_model(self, new_digits):
        """Update HMM model with new digits"""
        if len(new_digits) > 0:
            self.digit_history.extend(new_digits)
           
            if len(self.digit_history) > 1000: # Increased max history
                self.digit_history = self.digit_history[-1000:]
           
            is_valid, message = self.validate_data_quality(self.digit_history)
           
            if not self.hmm_predictor.is_fitted and is_valid and len(self.digit_history) >= self.hmm_predictor.min_training_samples:
                self.log(f"🔄 Initial model training... ({message})")
                success = self.hmm_predictor.safe_fit(self.digit_history)
                if success:
                    self.log(f"✓ Model trained with {len(self.digit_history)} digits")
                    self.last_retrain_time = datetime.now()
                else:
                    self.log("⚠️ Model training failed")
            else:
                self.log(f"📊 Data collection: {message}")
   
    def check_retrain_conditions(self):
        """Check if retraining is needed based on win/loss patterns - added more conditions"""
        if self.last_trade_result == 'win':
            self.consecutive_wins += 1
            if self.consecutive_wins >= 4:
                self.log(f"🔄 4 consecutive wins reached ({self.consecutive_wins})")
                self.consecutive_wins = 0
                return True
        elif self.last_trade_result == 'loss':
            self.consecutive_wins = 0
       
        # Additional condition: retrain every 20 trades regardless
        if len(self.trade_history) % 20 == 0 and len(self.trade_history) > 0:
            self.log(f"🔄 Periodic retrain after {len(self.trade_history)} trades")
            return True
       
        return False
   
    def perform_retraining(self):
        """Perform model retraining with current data"""
        if len(self.digit_history) < self.hmm_predictor.min_training_samples:
            self.log(f"⚠️ Not enough data for retraining: {len(self.digit_history)}/{self.hmm_predictor.min_training_samples}")
            return False
           
        is_valid, message = self.validate_data_quality(self.digit_history)
        if not is_valid:
            self.log(f"⚠️ Data quality insufficient for retraining: {message}")
            return False
           
        self.log(f"🔄 Retraining model with {len(self.digit_history)} digits...")
        success = self.hmm_predictor.safe_fit(self.digit_history)
        if success:
            self.log(f"✓ Model retrained successfully")
            self.last_retrain_time = datetime.now()
            self.pending_retrain = False
            return True
        else:
            self.log("⚠️ Model retraining failed")
            return False
   
    def trading_decision(self):
        """Make trading decision with improved reliability checks"""
        if len(self.digit_history) < 50: # Increased min data
            return None, None, f"Insufficient data ({len(self.digit_history)}/50 digits)"
       
        if self.pending_retrain:
            self.log("🔄 Pending retrain detected, retraining before decision...")
            if self.perform_retraining():
                self.log("✓ Retraining completed")
            else:
                self.log("⚠️ Retraining failed, proceeding with current model")
       
        prediction = self.hmm_predictor.predict_low_high_proba(self.digit_history[-50:]) # Longer recent sequence
       
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'prediction': prediction,
            'data_points': len(self.digit_history)
        })
       
        if not prediction['reliable']:
            return None, None, f"Unreliable prediction (conf: {prediction['confidence']:.3f})"
       
        confidence = prediction['confidence']
       
        if confidence < self.min_confidence:
            return None, None, f"Low confidence: {confidence:.3f}"
       
        if prediction['low'] > prediction['high']:
            return 'DIGITUNDER', '3', f"LOW (p={prediction['low']:.3f}, conf={confidence:.3f})"
        else:
            return 'DIGITOVER', '6', f"HIGH (p={prediction['high']:.3f}, conf={confidence:.3f})"
   
    def analyze_prediction_patterns(self):
        """Analyze prediction patterns to detect model issues"""
        if len(self.prediction_history) < 5:
            return "Insufficient prediction history"
           
        recent_predictions = self.prediction_history[-20:] # Longer analysis window
        low_predictions = sum(1 for p in recent_predictions
                             if p['prediction']['low'] > p['prediction']['high'])
        low_ratio = low_predictions / len(recent_predictions)
       
        if low_ratio > 0.85 or low_ratio < 0.15: # Tighter bounds
            return f"Biased predictions ({low_ratio:.1%} LOW) - consider retraining"
        else:
            return f"Normal patterns ({low_ratio:.1%} LOW)"
    def calculate_trade_amount(self):
        """Calculate trade amount using safe martingale strategy with risk management"""
        risk_amount = self.api_client.ws_client.balance * self.risk_per_trade
        max_risk_amount = self.api_client.ws_client.balance * self.max_risk_per_trade
       
        if self.last_trade_result == 'win':
            self.current_martingale_step = 0
            self.consecutive_losses = 0
            amount = min(self.base_amount, risk_amount)
            self.log(f"✅ Last trade won, resetting martingale. Amount: ${amount}")
        elif self.last_trade_result == 'loss':
            self.consecutive_losses += 1
            if self.current_martingale_step < self.max_martingale_steps:
                self.current_martingale_step += 1
                calculated = self.base_amount * (self.martingale_multiplier ** self.current_martingale_step)
                amount = min(calculated, max_risk_amount, self.max_amount)
                self.log(f"🔁 Martingale step {self.current_martingale_step}, amount: ${amount:.2f}")
            else:
                self.log("🛑 Max martingale steps reached, pausing and resetting")
                self.current_martingale_step = 0
                self.consecutive_losses = 0
                amount = 0 # Pause trading on this cycle
        else:
            amount = min(self.base_amount, risk_amount)
           
        return amount
    def determine_trade_outcome(self, contract_data, expected_direction):
        """Determine if trade was win or loss based on actual contract result"""
        if not contract_data:
            return 'unknown'
           
        buy_price = float(contract_data.get('buy_price', 0))
        sell_price = float(contract_data.get('sell_price', 0))
        profit = float(contract_data.get('profit', 0))
       
        self.log(f"📊 Contract result - Buy: ${buy_price}, Sell: ${sell_price}, Profit: ${profit}")
       
        if profit > 0:
            return 'win'
        elif profit < 0:
            return 'loss'
        else:
            return 'unknown'
    def place_trade(self, contract_type, barrier, symbol="R_100", amount=None):
        """Place a real trade using Deriv API"""
        if self.active_trade:
            self.log("⏳ Trade already active, waiting for completion...")
            return None
           
        if amount is None:
            amount = self.calculate_trade_amount()
       
        if amount == 0:
            self.log("🛑 Pausing trade due to max martingale reached")
            return None
           
        try:
            self.log(f"💸 Placing {contract_type} trade on {symbol} with barrier {barrier} - Amount: ${amount}")
           
            result = self.api_client.place_trade(contract_type, symbol, amount, barrier)
           
            if result:
                contract_id = result['buy']['contract_id']
               
                self.active_trade = {
                    'timestamp': datetime.now(),
                    'contract_type': contract_type,
                    'amount': amount,
                    'symbol': symbol,
                    'contract_id': contract_id,
                    'status': 'open',
                    'expected_direction': contract_type
                }
               
                trade_record = {
                    'timestamp': datetime.now().isoformat(),
                    'contract_type': contract_type,
                    'amount': amount,
                    'symbol': symbol,
                    'status': 'placed',
                    'result': result,
                    'martingale_step': self.current_martingale_step,
                    'contract_id': contract_id,
                    'pending_retrain': self.pending_retrain,
                    'profit': 0.0
                }
                self.trade_history.append(trade_record)
               
                self.monitor_trade(contract_id, contract_type)
               
            return result
               
        except Exception as e:
            self.log(f"Error placing trade: {e}")
            return None
    def monitor_trade(self, contract_id, expected_direction):
        """Monitor an active trade using REAL Deriv contract data"""
        def trade_monitor():
            self.log(f"🔍 Starting REAL trade monitoring for contract {contract_id}")
           
            self.api_client.subscribe_to_contract(contract_id)
           
            contract_data = self.api_client.get_contract_update(contract_id, timeout=30)
           
            if contract_data:
                actual_outcome = self.determine_trade_outcome(contract_data, expected_direction)
               
                if actual_outcome == 'win':
                    self.log("🎉 Trade WON based on actual contract result!")
                    self.last_trade_result = 'win'
                elif actual_outcome == 'loss':
                    self.log("💥 Trade LOST based on actual contract result!")
                    self.last_trade_result = 'loss'
                else:
                    self.log("❓ Trade outcome unknown")
                    self.last_trade_result = 'unknown'
               
                if self.trade_history:
                    last_trade = self.trade_history[-1]
                    last_trade['final_result'] = self.last_trade_result
                    last_trade['close_time'] = datetime.now().isoformat()
                    last_trade['contract_data'] = contract_data
                   
                    if 'profit' in contract_data:
                        profit = float(contract_data['profit'])
                        last_trade['profit'] = profit
                        self.current_profit += profit
               
                if actual_outcome == 'loss':
                    self.log("🔄 Loss detected - will retrain before next trade")
                    self.pending_retrain = True
                elif self.check_retrain_conditions():
                    self.log("🔄 Retrain condition met - will retrain before next trade")
                    self.pending_retrain = True
                   
            else:
                self.log("⏰ Contract monitoring timeout - assuming loss for safety")
                self.last_trade_result = 'loss'
                if self.trade_history:
                    last_trade = self.trade_history[-1]
                    last_trade['final_result'] = 'loss'
                    last_trade['close_time'] = datetime.now().isoformat()
                    last_trade['profit'] = 0.0
                    self.current_profit += 0.0 # or assume loss amount, but for simplicity 0
               
                self.log("🔄 Timeout assumed as loss - will retrain before next trade")
                self.pending_retrain = True
           
            self.active_trade = None
           
            self.log(f"📊 Status - Martingale: {self.current_martingale_step}, Losses: {self.consecutive_losses}, Wins: {self.consecutive_wins}, Retrain pending: {self.pending_retrain}")
            self.log(f"💰 Current profit: ${self.current_profit:.2f}")
           
            # Save to CSV after each trade
            self.save_trades_to_csv()
       
        monitor_thread = threading.Thread(target=trade_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()
    def run_trading_cycle(self):
        """Execute one trading cycle with improved safety checks"""
        self.log(f"Trading cycle...")
       
        if self.active_trade:
            self.log("⏳ Active trade in progress, skipping new trade...")
            return
       
        ticks_data = self.get_real_ticks_data()
       
        if not ticks_data:
            ticks_data = self.get_live_ticks() # Fallback to live
       
        if not ticks_data:
            self.log("❌ No data available - skipping cycle")
            return
       
        new_digits = self.extract_digits_from_ticks(ticks_data)
       
        if not new_digits:
            self.log("❌ No digits extracted - skipping cycle")
            return
       
        self.log(f"📊 Processing {len(new_digits)} real digits")
       
        self.update_model(new_digits)
       
        if len(self.prediction_history) >= 5:
            pattern_analysis = self.analyze_prediction_patterns()
            self.log(f"📈 Pattern analysis: {pattern_analysis}")
            if "Biased" in pattern_analysis:
                self.pending_retrain = True # Trigger retrain on bias detection
       
        contract_type, barrier, reason = self.trading_decision()
        self.log(f"🤖 Decision: {contract_type} with barrier {barrier} - Reason: {reason}")
       
        if contract_type:
            result = self.place_trade(contract_type, barrier, self.symbol)
            if result:
                self.log("✅ Trade placed successfully")
            else:
                self.log("❌ Trade failed")
        else:
            self.log("⏭️ No trade placed - waiting for better opportunity")
       
        if len(self.digit_history) >= 10:
            recent = self.digit_history[-10:]
            recent_even = sum(1 for d in recent if d % 2 == 0)
            recent_odd = len(recent) - recent_even
            self.log(f"📊 Recent even/odd: {recent_even}/{recent_odd} ({recent_even/len(recent):.1%} even)")
       
        if self.last_retrain_time:
            time_since_retrain = datetime.now() - self.last_retrain_time
            self.log(f"⏱️ Last retrain: {time_since_retrain.seconds//60}m {time_since_retrain.seconds%60}s ago")
        self.log(f"📈 Win streak: {self.consecutive_wins}")
        self.log(f"🔄 Retrain pending: {self.pending_retrain}")
    def trading_loop(self):
        """Main trading loop in thread"""
        cycle_count = 0
        max_cycles = 100000
        while self.running and cycle_count < max_cycles:
            # Check min balance
            if self.api_client.ws_client.balance <= self.min_balance:
                self.log(f"🛑 Low balance: ${self.api_client.ws_client.balance:.2f} <= ${self.min_balance:.2f}, stopping trading")
                self.running = False
                break
           
            # Check TP/SL
            if self.current_profit >= self.take_profit:
                self.log(f"🎯 Take profit reached: ${self.current_profit:.2f} >= ${self.take_profit:.2f}")
                self.running = False
                break
            if self.current_profit <= self.stop_loss:
                self.log(f"🛑 Stop loss hit: ${self.current_profit:.2f} <= ${self.stop_loss:.2f}")
                self.running = False
                break
           
            cycle_count += 1
            self.log(f"🔁 Cycle {cycle_count}/{max_cycles}")
            self.log("-" * 30)
            self.run_trading_cycle()
            for trade in self.trade_history:
                if trade.get('final_result') and trade.get('status') == 'placed':
                    if 'processed' not in trade:
                        # Update stats here if needed
                        trade['processed'] = True
                        trade['status'] = 'completed'
            wait_time = 10 if self.active_trade else 5
            self.log(f"⏰ Waiting {wait_time} seconds...")
            time.sleep(wait_time)
       
        self.log("Trading loop stopped")
    def start_trading(self):
        """Start the trading thread"""
        if not self.running:
            self.running = True
            self.current_profit = 0.0 # Reset for new session
            self.trading_thread = threading.Thread(target=self.trading_loop)
            self.trading_thread.daemon = True
            self.trading_thread.start()
            self.log("Trading started")
            return "Trading started"
        return "Trading already running"
    def stop_trading(self):
        """Stop the trading thread"""
        if self.running:
            self.running = False
            if self.trading_thread:
                self.trading_thread.join(timeout=10)
            self.log("Trading stopped")
            return "Trading stopped"
        return "Trading not running"
    def get_status(self):
        """Get current status"""
        with self.lock:
            status = {
                "running": self.running,
                "consecutive_wins": self.consecutive_wins,
                "consecutive_losses": self.consecutive_losses,
                "total_trades": len(self.trade_history),
                "model_fitted": self.hmm_predictor.is_fitted,
                "digits_collected": len(self.digit_history),
                "current_profit": self.current_profit,
                "balance": self.api_client.ws_client.balance,
                "logs": self.logs[-20:] # Last 20 logs
            }
            return json.dumps(status)
    def retrain_model(self):
        """Force retrain"""
        self.pending_retrain = True
        self.perform_retraining()
        return "Retraining initiated"
    def get_chart_data(self):
        """Get data for charts"""
        with self.lock:
            # Profit over time
            profits = []
            cumulative_profit = 0.0
            timestamps = []
            for trade in self.trade_history:
                if 'profit' in trade:
                    cumulative_profit += trade['profit']
                    profits.append(cumulative_profit)
                    timestamps.append(trade['timestamp'])
           
            # Recent digits (last 50)
            recent_digits = self.digit_history[-50:]
           
            # Digit distribution
            digit_counts = [0] * 10
            for d in self.digit_history:
                digit_counts[d] += 1
           
            # Even/odd ratio
            even_count = sum(1 for d in self.digit_history if d % 2 == 0)
            odd_count = len(self.digit_history) - even_count
           
            # Prediction confidence
            conf_timestamps = [p['timestamp'].isoformat() for p in self.prediction_history[-20:]]
            confidences = [p['prediction']['confidence'] for p in self.prediction_history[-20:]]
           
            # Trades for table (all trades, including past)
            trades = [{
                'timestamp': t.get('timestamp', ''),
                'contract_type': t.get('contract_type', ''),
                'amount': t.get('amount', 0.0),
                'profit': t.get('profit', 0.0),
                'final_result': t.get('final_result', 'pending')
            } for t in self.trade_history]
           
            data = {
                "profit_timestamps": timestamps[-20:], # Last 20 trades
                "cumulative_profits": profits[-20:],
                "recent_digits": recent_digits,
                "digit_counts": digit_counts,
                "even_odd": [even_count, odd_count],
                "conf_timestamps": conf_timestamps,
                "confidences": confidences,
                "trades": trades # All trades
            }
            return json.dumps(data)
# HTML content with charts
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deriv Trading Bot | Advanced Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --primary-black: #0a0a0a;
            --secondary-black: #1a1a1a;
            --tertiary-black: #2a2a2a;
            --accent-orange: #ff4500;
            --accent-green: #32cd32;
            --accent-white: #ffffff;
            --accent-gray: #404040;
            --accent-blue: #1e90ff;
            --grid-color: #333333;
            --shadow-color: rgba(255, 69, 0, 0.1);
        }
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: var(--primary-black);
            color: var(--accent-white);
            min-height: 100vh;
            overflow-x: hidden;
            position: relative;
        }
        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background:
                radial-gradient(circle at 20% 50%, rgba(255, 69, 0, 0.05) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(50, 205, 50, 0.03) 0%, transparent 50%),
                linear-gradient(45deg, var(--primary-black) 0%, var(--secondary-black) 100%);
            z-index: -1;
        }
        .dashboard-container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
            display: grid;
            grid-template-columns: 1fr;
            gap: 24px;
        }
        /* Header */
        .dashboard-header {
            background: linear-gradient(135deg, var(--secondary-black), var(--tertiary-black));
            border: 1px solid var(--accent-gray);
            border-radius: 16px;
            padding: 24px 32px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            display: flex;
            justify-content: space-between;
            align-items: center;
            backdrop-filter: blur(10px);
            position: relative;
            overflow: hidden;
        }
        .dashboard-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, var(--accent-orange), var(--accent-green));
        }
        .header-left h1 {
            font-size: 28px;
            font-weight: 700;
            background: linear-gradient(90deg, var(--accent-white), var(--accent-orange));
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            letter-spacing: -0.5px;
            margin-bottom: 8px;
        }
        .header-left .subtitle {
            font-size: 14px;
            color: #888;
            font-weight: 400;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .header-right {
            display: flex;
            gap: 20px;
            align-items: center;
        }
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: var(--accent-green);
            box-shadow: 0 0 20px var(--accent-green);
            animation: pulse 2s infinite;
        }
        .status-dot.offline {
            background: var(--accent-orange);
            box-shadow: 0 0 20px var(--accent-orange);
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        /* Controls */
        .controls-panel {
            background: var(--secondary-black);
            border: 1px solid var(--accent-gray);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
        }
        .controls-panel h2 {
            font-size: 16px;
            font-weight: 600;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 24px;
            position: relative;
            display: inline-block;
        }
        .controls-panel h2::after {
            content: '';
            position: absolute;
            bottom: -8px;
            left: 0;
            width: 40px;
            height: 2px;
            background: var(--accent-orange);
        }
        .controls {
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
        }
        .btn {
            padding: 14px 32px;
            font-size: 14px;
            font-weight: 600;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        .btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
            transition: 0.5s;
        }
        .btn:hover::before {
            left: 100%;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px var(--shadow-color);
        }
        .btn:active {
            transform: translateY(0);
        }
        .btn-start {
            background: linear-gradient(135deg, var(--accent-green), #228b22);
            color: var(--primary-black);
            box-shadow: 0 4px 15px rgba(50, 205, 50, 0.3);
        }
        .btn-stop {
            background: linear-gradient(135deg, var(--accent-orange), #cc3700);
            color: var(--accent-white);
            box-shadow: 0 4px 15px rgba(255, 69, 0, 0.3);
        }
        .btn-retrain {
            background: linear-gradient(135deg, var(--accent-blue), #0066cc);
            color: var(--accent-white);
            box-shadow: 0 4px 15px rgba(30, 144, 255, 0.3);
        }
        /* Cards Grid */
        .cards-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 24px;
        }
        .card {
            background: var(--secondary-black);
            border: 1px solid var(--accent-gray);
            border-radius: 16px;
            padding: 24px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        .card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--accent-orange), transparent);
        }
        .card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
            border-color: var(--accent-orange);
        }
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .card-title {
            font-size: 14px;
            font-weight: 600;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .card-value {
            font-size: 24px;
            font-weight: 700;
            color: var(--accent-white);
        }
        /* Status Card */
        #status {
            background: var(--tertiary-black);
            padding: 16px;
            border-radius: 12px;
            border: 1px solid var(--grid-color);
            min-height: 200px;
            max-height: 200px;
            overflow-y: auto;
        }
        #status-text {
            font-family: 'Roboto Mono', 'Consolas', monospace;
            font-size: 13px;
            line-height: 1.6;
            color: #aaa;
        }
        /* Logs Card */
        #logs {
            height: 300px;
            overflow-y: auto;
            padding-right: 8px;
        }
        #log-content {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .log-entry {
            padding: 12px 16px;
            background: var(--tertiary-black);
            border-radius: 8px;
            border-left: 3px solid var(--accent-orange);
            font-size: 13px;
            font-family: 'Roboto Mono', 'Consolas', monospace;
            animation: slideIn 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            transition: all 0.2s;
        }
        .log-entry:hover {
            background: rgba(255, 69, 0, 0.1);
            transform: translateX(4px);
        }
        .log-entry.success {
            border-left-color: var(--accent-green);
        }
        .log-entry.error {
            border-left-color: var(--accent-orange);
        }
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        /* Trades Card */
        #trades {
            max-height: 400px;
            overflow-y: auto;
        }
        table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
        }
        thead {
            position: sticky;
            top: 0;
            background: var(--secondary-black);
            z-index: 10;
        }
        th {
            padding: 16px;
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #888;
            border-bottom: 2px solid var(--grid-color);
            text-align: left;
        }
        td {
            padding: 16px;
            border-bottom: 1px solid var(--grid-color);
            font-size: 14px;
            color: #ccc;
        }
        tr:hover td {
            background: rgba(255, 69, 0, 0.05);
            color: var(--accent-white);
        }
        .profit-positive {
            color: var(--accent-green);
            font-weight: 600;
        }
        .profit-negative {
            color: var(--accent-orange);
            font-weight: 600;
        }
        /* Charts Grid */
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 24px;
        }
        .chart-container {
            background: var(--secondary-black);
            border: 1px solid var(--accent-gray);
            border-radius: 16px;
            padding: 24px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        .chart-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--accent-green), transparent);
        }
        .chart-container:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        }
        .chart-container h2 {
            font-size: 14px;
            font-weight: 600;
            color: #888;
            margin-bottom: 20px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: var(--tertiary-black);
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb {
            background: var(--accent-gray);
            border-radius: 4px;
            transition: all 0.3s;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: var(--accent-orange);
        }
        /* Stats Bar */
        .stats-bar {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        .stat-card {
            background: var(--secondary-black);
            border: 1px solid var(--accent-gray);
            border-radius: 12px;
            padding: 20px;
            display: flex;
            align-items: center;
            gap: 16px;
            transition: all 0.3s;
        }
        .stat-card:hover {
            border-color: var(--accent-orange);
            transform: translateY(-2px);
        }
        .stat-icon {
            width: 48px;
            height: 48px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            background: rgba(255, 69, 0, 0.1);
            color: var(--accent-orange);
        }
        .stat-content {
            flex: 1;
        }
        .stat-value {
            font-size: 24px;
            font-weight: 700;
            color: var(--accent-white);
            margin-bottom: 4px;
        }
        .stat-label {
            font-size: 12px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        /* Responsive */
        @media (max-width: 1200px) {
            .charts-grid {
                grid-template-columns: 1fr;
            }
           
            .chart-container {
                min-height: 400px;
            }
        }
        @media (max-width: 768px) {
            .dashboard-container {
                padding: 16px;
                gap: 16px;
            }
           
            .dashboard-header {
                flex-direction: column;
                text-align: center;
                gap: 20px;
            }
           
            .controls {
                justify-content: center;
            }
           
            .btn {
                flex: 1;
                min-width: 120px;
            }
           
            .cards-grid {
                grid-template-columns: 1fr;
            }
           
            .charts-grid {
                grid-template-columns: 1fr;
            }
        }
        /* Glowing effect for important elements */
        .glow {
            animation: glow 2s ease-in-out infinite alternate;
        }
        @keyframes glow {
            from {
                box-shadow: 0 0 20px rgba(255, 69, 0, 0.3);
            }
            to {
                box-shadow: 0 0 30px rgba(255, 69, 0, 0.6);
            }
        }
        /* Performance metrics */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px;
            margin-top: 20px;
        }
        .metric {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 12px;
            text-align: center;
        }
        .metric-value {
            font-size: 18px;
            font-weight: 600;
            color: var(--accent-white);
            margin-bottom: 4px;
        }
        .metric-label {
            font-size: 11px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
    </style>
</head>
<body>
    <div class="dashboard-container">
        <!-- Header -->
        <header class="dashboard-header">
            <div class="header-left">
                <h1>ALGOCDK </h1>
                <div class="subtitle">Advanced Algorithmic Trading System For Low/High</div>
            </div>
            <div class="header-right">
                <div class="status-indicator">
                    <div class="status-dot" id="live-status-dot"></div>
                    <span id="live-status-text">CONNECTING...</span>
                </div>
                <div class="status-indicator">
                    <div class="status-dot"></div>
                    <span>API: ACTIVE</span>
                </div>
            </div>
        </header>
        <!-- Stats Bar -->
        <div class="stats-bar">
            <div class="stat-card">
                <div class="stat-icon">₿</div>
                <div class="stat-content">
                    <div class="stat-value" id="total-profit">$0.00</div>
                    <div class="stat-label">Total Profit</div>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">↗</div>
                <div class="stat-content">
                    <div class="stat-value" id="win-rate">0%</div>
                    <div class="stat-label">Win Rate</div>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">⏱</div>
                <div class="stat-content">
                    <div class="stat-value" id="active-time">00:00</div>
                    <div class="stat-label">Active Time</div>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">📈</div>
                <div class="stat-content">
                    <div class="stat-value" id="trade-count">0</div>
                    <div class="stat-label">Total Trades</div>
                </div>
            </div>
        </div>
        <!-- Controls Panel -->
        <section class="controls-panel">
            <h2>System Controls</h2>
            <div class="controls">
                <button class="btn btn-start" onclick="control('start')">
                    START TRADING
                </button>
                <button class="btn btn-stop" onclick="control('stop')">
                    STOP TRADING
                </button>
                <button class="btn btn-retrain" onclick="control('retrain')">
                    RETRAIN MODEL
                </button>
            </div>
           
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-value" id="model-accuracy">--%</div>
                    <div class="metric-label">Model Accuracy</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="confidence-level">--%</div>
                    <div class="metric-label">Confidence</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="risk-level">MEDIUM</div>
                    <div class="metric-label">Risk Level</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="sharpe-ratio">--</div>
                    <div class="metric-label">Sharpe Ratio</div>
                </div>
            </div>
        </section>
        <!-- Cards Grid -->
        <div class="cards-grid">
            <!-- Status Card -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">System Status</div>
                    <div class="card-value" id="system-status">IDLE</div>
                </div>
                <div id="status">
                    <pre id="status-text">Initializing system...</pre>
                </div>
            </div>
            <!-- Logs Card -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">Live Activity Log</div>
                    <div class="card-value" id="log-count">0</div>
                </div>
                <div id="logs">
                    <div id="log-content"></div>
                </div>
            </div>
            <!-- Trades Card -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">Recent Trades</div>
                    <div class="card-value" id="recent-trades">0</div>
                </div>
                <div id="trades">
                    <table>
                        <thead>
                            <tr>
                                <th>Time</th>
                                <th>Type</th>
                                <th>Amount</th>
                                <th>Profit</th>
                                <th>Result</th>
                            </tr>
                        </thead>
                        <tbody id="trades-body">
                            <tr><td colspan="5" style="text-align: center; padding: 40px; color: #666;">Awaiting first trade...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        <!-- Charts Grid -->
        <div class="charts-grid">
            <div class="chart-container">
                <h2>Profit & Loss Timeline</h2>
                <canvas id="profitChart"></canvas>
            </div>
            <div class="chart-container">
                <h2>Market Pattern Analysis</h2>
                <canvas id="digitsChart"></canvas>
            </div>
            <div class="chart-container">
                <h2>Digit Frequency Distribution</h2>
                <canvas id="digitDistChart"></canvas>
            </div>
            <div class="chart-container">
                <h2>Even/Odd Pattern Ratio</h2>
                <canvas id="evenOddChart"></canvas>
            </div>
            <div class="chart-container">
                <h2>AI Model Confidence</h2>
                <canvas id="confidenceChart"></canvas>
            </div>
        </div>
    </div>
    <script>
        let profitChart, digitsChart, digitDistChart, evenOddChart, confidenceChart;
        let startTime = Date.now();
        function initCharts() {
            const gridColor = '#333333';
            const textColor = '#888888';
           
            Chart.defaults.color = textColor;
            Chart.defaults.borderColor = gridColor;
            Chart.defaults.font.family = "'Inter', -apple-system, sans-serif";
            // Professional chart options
            const chartOptions = {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        labels: {
                            color: textColor,
                            font: {
                                size: 12,
                                weight: '500'
                            }
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rg(26, 26, 26, 0.9)',
                        titleColor: '#ffffff',
                        bodyColor: '#cccccc',
                        borderColor: gridColor,
                        borderWidth: 1,
                        cornerRadius: 8,
                        displayColors: false
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: gridColor,
                            drawBorder: false
                        },
                        ticks: {
                            color: textColor,
                            font: {
                                size: 11
                            }
                        }
                    },
                    y: {
                        grid: {
                            color: gridColor,
                            drawBorder: false
                        },
                        ticks: {
                            color: textColor,
                            font: {
                                size: 11
                            }
                        }
                    }
                }
            };
            // Profit Chart with gradient
            const profitCtx = document.getElementById('profitChart').getContext('2d');
            const profitGradient = profitCtx.createLinearGradient(0, 0, 0, 400);
            profitGradient.addColorStop(0, 'rgba(50, 205, 50, 0.3)');
            profitGradient.addColorStop(1, 'rgba(50, 205, 50, 0)');
            profitChart = new Chart(profitCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Cumulative Profit ($)',
                        data: [],
                        borderColor: '#32cd32',
                        backgroundColor: profitGradient,
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4,
                        pointBackgroundColor: '#32cd32',
                        pointBorderColor: '#0a0a0a',
                        pointBorderWidth: 2,
                        pointRadius: 4,
                        pointHoverRadius: 6
                    }]
                },
                options: {
                    ...chartOptions,
                    scales: {
                        ...chartOptions.scales,
                        y: {
                            ...chartOptions.scales.y,
                            beginAtZero: true,
                            ticks: {
                                ...chartOptions.scales.y.ticks,
                                callback: function(value) {
                                    return '$' + value.toFixed(2);
                                }
                            }
                        }
                    }
                }
            });
            // Digits Chart
            digitsChart = new Chart(document.getElementById('digitsChart'), {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Digit Value',
                        data: [],
                        borderColor: '#ff4500',
                        backgroundColor: 'rgba(255, 69, 0, 0.1)',
                        borderWidth: 2,
                        tension: 0.4,
                        pointStyle: 'circle',
                        pointBackgroundColor: '#ff4500',
                        pointBorderColor: '#0a0a0a',
                        pointBorderWidth: 2
                    }]
                },
                options: {
                    ...chartOptions,
                    scales: {
                        ...chartOptions.scales,
                        y: {
                            ...chartOptions.scales.y,
                            min: 0,
                            max: 9,
                            ticks: {
                                stepSize: 1
                            }
                        }
                    }
                }
            });
            // Digit Distribution Chart
            digitDistChart = new Chart(document.getElementById('digitDistChart'), {
                type: 'bar',
                data: {
                    labels: ['0','1','2','3','4','5','6','7','8','9'],
                    datasets: [{
                        label: 'Frequency',
                        data: [],
                        backgroundColor: Array(10).fill().map((_, i) =>
                            i % 2 === 0 ? '#32cd32' : '#ff4500'
                        ),
                        borderColor: '#0a0a0a',
                        borderWidth: 1,
                        borderRadius: 4
                    }]
                },
                options: chartOptions
            });
            // Even/Odd Chart
            evenOddChart = new Chart(document.getElementById('evenOddChart'), {
                type: 'doughnut',
                data: {
                    labels: ['Even', 'Odd'],
                    datasets: [{
                        data: [],
                        backgroundColor: ['#32cd32', '#ff4500'],
                        borderColor: '#0a0a0a',
                        borderWidth: 2,
                        hoverOffset: 15
                    }]
                },
                options: {
                    ...chartOptions,
                    cutout: '70%',
                    plugins: {
                        ...chartOptions.plugins,
                        legend: {
                            ...chartOptions.plugins.legend,
                            position: 'bottom',
                            labels: {
                                ...chartOptions.plugins.legend.labels,
                                padding: 20
                            }
                        }
                    }
                }
            });
            // Confidence Chart
            confidenceChart = new Chart(document.getElementById('confidenceChart'), {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Confidence Level',
                        data: [],
                        borderColor: '#1e90ff',
                        backgroundColor: 'rgba(30, 144, 255, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    ...chartOptions,
                    scales: {
                        ...chartOptions.scales,
                        y: {
                            ...chartOptions.scales.y,
                            min: 0,
                            max: 1,
                            ticks: {
                                ...chartOptions.scales.y.ticks,
                                callback: function(value) {
                                    return (value * 100).toFixed(0) + '%';
                                }
                            }
                        }
                    }
                }
            });
        }
        function updateCharts(data) {
            // Update status indicators
            const statusDot = document.getElementById('live-status-dot');
            const statusText = document.getElementById('live-status-text');
           
            if (data.is_running) {
                statusDot.style.background = '#32cd32';
                statusDot.style.boxShadow = '0 0 20px #32cd32';
                statusText.textContent = 'LIVE TRADING';
                document.getElementById('system-status').textContent = 'ACTIVE';
                document.getElementById('system-status').style.color = '#32cd32';
            } else {
                statusDot.style.background = '#ff4500';
                statusDot.style.boxShadow = '0 0 20px #ff4500';
                statusText.textContent = 'STOPPED';
                document.getElementById('system-status').textContent = 'IDLE';
                document.getElementById('system-status').style.color = '#ff4500';
            }
            // Update stats bar
            if (data.cumulative_profits && data.cumulative_profits.length > 0) {
                const totalProfit = data.cumulative_profits[data.cumulative_profits.length - 1];
                document.getElementById('total-profit').textContent = '$' + totalProfit.toFixed(2);
                document.getElementById('total-profit').style.color = totalProfit >= 0 ? '#32cd32' : '#ff4500';
            }
            // Update trade count
            if (data.trades) {
                const tradeCount = data.trades.length;
                document.getElementById('trade-count').textContent = tradeCount;
                document.getElementById('recent-trades').textContent = Math.min(tradeCount, 10);
               
                // Calculate win rate
                if (tradeCount > 0) {
                    const wins = data.trades.filter(t => t.final_result === 'win').length;
                    const winRate = ((wins / tradeCount) * 100).toFixed(1);
                    document.getElementById('win-rate').textContent = winRate + '%';
                    document.getElementById('win-rate').style.color = winRate >= 50 ? '#32cd32' : '#ff4500';
                }
            }
            // Update active time
            const activeTime = Math.floor((Date.now() - startTime) / 1000);
            const hours = Math.floor(activeTime / 3600);
            const minutes = Math.floor((activeTime % 3600) / 60);
            document.getElementById('active-time').textContent =
                hours.toString().padStart(2, '0') + ':' + minutes.toString().padStart(2, '0');
            // Update log count
            const logCount = document.getElementById('log-content').children.length;
            document.getElementById('log-count').textContent = logCount;
            // Profit chart
            if (data.profit_timestamps && data.cumulative_profits) {
                profitChart.data.labels = data.profit_timestamps.map(t => new Date(t).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}));
                profitChart.data.datasets[0].data = data.cumulative_profits;
                profitChart.update('none');
            }
            // Recent digits
            if (data.recent_digits) {
                digitsChart.data.labels = Array.from({length: data.recent_digits.length}, (_, i) => 'T' + (i + 1));
                digitsChart.data.datasets[0].data = data.recent_digits;
                digitsChart.update('none');
            }
            // Digit distribution
            if (data.digit_counts) {
                digitDistChart.data.datasets[0].data = data.digit_counts;
                digitDistChart.update('none');
            }
            // Even/Odd
            if (data.even_odd) {
                evenOddChart.data.datasets[0].data = data.even_odd;
                evenOddChart.update('none');
            }
            // Confidence
            if (data.conf_timestamps && data.confidences) {
                confidenceChart.data.labels = data.conf_timestamps.map(t => new Date(t).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}));
                confidenceChart.data.datasets[0].data = data.confidences;
                confidenceChart.update('none');
               
                // Update confidence metric
                if (data.confidences.length > 0) {
                    const lastConfidence = data.confidences[data.confidences.length - 1];
                    document.getElementById('confidence-level').textContent = (lastConfidence * 100).toFixed(0) + '%';
                }
            }
            // Trades table
            if (data.trades && data.trades.length > 0) {
                const tradesBody = document.getElementById('trades-body');
                tradesBody.innerHTML = '';
               
                // Show only last 10 trades
                const recentTrades = data.trades.slice(-10);
               
                recentTrades.forEach(trade => {
                    const tr = document.createElement('tr');
                    const profitClass = trade.profit >= 0 ? 'profit-positive' : 'profit-negative';
                    const resultClass = trade.final_result === 'win' ? 'profit-positive' : 'profit-negative';
                   
                    tr.innerHTML = `
                        <td>${new Date(trade.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'})}</td>
                        <td>${trade.contract_type}</td>
                        <td>$${trade.amount.toFixed(2)}</td>
                        <td class="${profitClass}">${trade.profit >= 0 ? '+' : ''}$${trade.profit.toFixed(2)}</td>
                        <td class="${resultClass}">${trade.final_result.toUpperCase()}</td>
                    `;
                    tradesBody.appendChild(tr);
                });
            }
        }
        function control(action) {
            const btn = event.target;
            const originalText = btn.innerHTML;
           
            // Show loading state
            btn.innerHTML = '<span style="display:inline-block;animation:spin 1s linear infinite;">⟳</span> PROCESSING';
            btn.style.opacity = '0.8';
            btn.disabled = true;
            // Add spin animation
            const style = document.createElement('style');
            style.textContent = '@keyframes spin { 100% { transform: rotate(360deg); } }';
            document.head.appendChild(style);
            fetch(`/control?action=${action}`)
                .then(response => response.text())
                .then(data => {
                    // Show success feedback
                    btn.innerHTML = '✓ DONE';
                    btn.style.opacity = '1';
                   
                    setTimeout(() => {
                        btn.innerHTML = originalText;
                        btn.disabled = false;
                    }, 1000);
                   
                    // Add log entry
                    addLog(`Command "${action}" executed successfully`, 'success');
                   
                    // Update immediately
                    updateStatus();
                    updateChartData();
                })
                .catch(error => {
                    console.error('Error:', error);
                    btn.innerHTML = '✗ ERROR';
                   
                    setTimeout(() => {
                        btn.innerHTML = originalText;
                        btn.disabled = false;
                    }, 1000);
                   
                    addLog(`Error executing "${action}": ${error.message}`, 'error');
                });
        }
        function addLog(message, type = 'info') {
            const logContent = document.getElementById('log-content');
            const div = document.createElement('div');
            div.className = `log-entry ${type}`;
            div.textContent = `[${new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'})}] ${message}`;
           
            logContent.appendChild(div);
           
            // Keep only last 20 logs
            while (logContent.children.length > 20) {
                logContent.removeChild(logContent.firstChild);
            }
           
            logContent.scrollTop = logContent.scrollHeight;
        }
        function updateStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    const formattedStatus = JSON.stringify(data, null, 2)
                        .replace(/"([^"]+)":/g, '<span style="color:#ff4500">$1</span>:')
                        .replace(/: "([^"]+)"/g, ': <span style="color:#32cd32">"$1"</span>')
                        .replace(/: (\d+)/g, ': <span style="color:#1e90ff">$1</span>')
                        .replace(/: (true|false)/g, ': <span style="color:#ff4500">$1</span>');
                   
                    document.getElementById('status-text').innerHTML = formattedStatus;
                })
                .catch(error => {
                    console.error('Error:', error);
                    addLog(`Status update failed: ${error.message}`, 'error');
                });
        }
        function updateChartData() {
            fetch('/chart_data')
                .then(response => response.json())
                .then(data => {
                    updateCharts(data);
                })
                .catch(error => {
                    console.error('Error:', error);
                    addLog(`Chart data update failed: ${error.message}`, 'error');
                });
        }
        // Poll status and charts every 2 seconds
        setInterval(() => {
            updateStatus();
            updateChartData();
        }, 2000);
        // Initialize everything
        initCharts();
        updateStatus();
        updateChartData();
        // Add initial logs
        setTimeout(() => {
            addLog('System initialized successfully', 'success');
            addLog('Dashboard ready for trading operations', 'info');
            addLog('Connecting to Deriv API...', 'info');
        }, 1000);
    </script>
</body>
</html>
"""
class BotRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, bot=None, **kwargs):
        self.bot = bot
        super().__init__(*args, directory=None, **kwargs)
    def do_GET(self):
        parsed_path = urlparse(self.path)
        if parsed_path.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode())
        elif parsed_path.path == '/status':
            status = self.bot.get_status()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(status.encode())
        elif parsed_path.path == '/chart_data':
            chart_data = self.bot.get_chart_data()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(chart_data.encode())
        elif parsed_path.path == '/control':
            params = parse_qs(parsed_path.query)
            action = params.get('action', [None])[0]
            if action == 'start':
                response = self.bot.start_trading()
            elif action == 'stop':
                response = self.bot.stop_trading()
            elif action == 'retrain':
                response = self.bot.retrain_model()
            else:
                response = "Invalid action"
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(response.encode())
        else:
            self.send_error(404)
def run_server(bot, port=8000):
    handler = lambda *args, **kwargs: BotRequestHandler(*args, bot=bot, **kwargs)
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Serving UI at http://localhost:{port}")
        httpd.serve_forever()
def main():
    """Main function with UI integration"""
    print("🚀 IMPROVED Deriv HMM Trading Bot with Web UI and Charts")
    print("=" * 60)
    print("Access the UI at http://localhost:8000")
    print("=" * 60)
    load_dotenv()
    # DERIV_TOKEN = 'y5XlAyZZDrPz764'
    # print(f"Using token: {DERIV_TOKEN[:8]}...")
    DERIV_TOKEN = os.getenv('DERIV_TOKEN')
    bot = FixedDerivTradingBot(token=DERIV_TOKEN, demo=True)
    if not bot.api_client.connect():
        print("❌ Could not connect to Deriv API. Exiting.")
        return
    bot.api_client.ws_client.subscribe_ticks("R_100") # Added subscription for live ticks
    # Start server in thread
    server_thread = threading.Thread(target=run_server, args=(bot,))
    server_thread.daemon = True
    server_thread.start()
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n⏹️ Server stopped by user")
if __name__ == "__main__":
    main()