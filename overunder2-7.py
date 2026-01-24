import streamlit as st
import websocket
import json
import numpy as np
from math import floor
import threading
import queue
import time
import pandas as pd
from datetime import datetime
from scipy import stats
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# Enhanced Pattern Recognition System
# ──────────────────────────────────────────────
class PatternRecognizer:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.price_buffer = []
        self.digit_buffer = []
        self.patterns = {}
        
    def add_data(self, price, digit):
        self.price_buffer.append(price)
        self.digit_buffer.append(digit)
        
        if len(self.price_buffer) > self.window_size:
            self.price_buffer.pop(0)
            self.digit_buffer.pop(0)
    
    def detect_micro_patterns(self):
        if len(self.price_buffer) < 20:
            return None
            
        patterns = {}
        
        # 1. Detect V-shaped reversals
        if len(self.price_buffer) >= 10:
            recent = self.price_buffer[-10:]
            if recent[0] > recent[5] < recent[9]:  # V-shape
                patterns['v_shape'] = True
                
        # 2. Detect momentum shifts
        if len(self.digit_buffer) >= 15:
            digits = np.array(self.digit_buffer[-15:])
            momentum = np.diff(digits)
            patterns['momentum_positive'] = np.sum(momentum > 0) > 10
            patterns['momentum_negative'] = np.sum(momentum < 0) > 10
            
        # 3. Detect digit clusters
        digit_counts = np.bincount(self.digit_buffer[-20:], minlength=10)
        patterns['digit_clusters'] = np.where(digit_counts > 3)[0].tolist()
        
        # 4. Detect volatility regimes
        if len(self.price_buffer) >= 30:
            prices = np.array(self.price_buffer[-30:])
            volatility = np.std(prices[-10:]) / np.std(prices[-30:-10])
            patterns['volatility_spike'] = volatility > 1.5
            patterns['volatility_drop'] = volatility < 0.7

        # 5. Detect serial correlation (hidden pattern in digits)
        if len(self.digit_buffer) >= 20:
            autocorr = stats.pearsonr(self.digit_buffer[-20:-1], self.digit_buffer[-19:])[0]
            patterns['high_autocorr'] = abs(autocorr) > 0.3  # Indicates persistent pattern

        # 6. Entropy for predictability (low entropy = hidden repeatable patterns)
        if len(self.digit_buffer) >= 20:
            counts = np.bincount(self.digit_buffer[-20:], minlength=10)
            probs = counts / np.sum(counts)
            probs = probs[probs > 0]  # Avoid log(0)
            entropy = stats.entropy(probs)
            patterns['low_entropy'] = entropy < 2.0  # Low entropy suggests predictable hidden patterns
            
        return patterns

# ──────────────────────────────────────────────
# Advanced Markov Chain with Memory and Seasonality
# ──────────────────────────────────────────────
class EnhancedMarkovChain:
    def __init__(self, memory_length=4):  # Increased memory to 4 for better hidden pattern capture
        self.memory_length = memory_length
        # Dynamic state space based on memory
        self.state_space_size = 10 ** memory_length
        self.transitions = np.ones((self.state_space_size, 10)) * 0.1  # Laplace smoothing
        self.state_visits = np.zeros(self.state_space_size)
        self.time_patterns = {}  # Track patterns by time of day
        
    def state_to_index(self, digits):
        """Convert recent digits to state index"""
        idx = 0
        for i, d in enumerate(digits):
            idx += d * (10 ** i)
        return idx
    
    def update(self, new_digit, recent_digits):
        if len(recent_digits) >= self.memory_length:
            state_digits = recent_digits[-(self.memory_length):]
            state_idx = self.state_to_index(state_digits)
            
            self.state_visits[state_idx] += 1
            self.transitions[state_idx][new_digit] += 1
            
            # Normalize
            total = np.sum(self.transitions[state_idx])
            if total > 0:
                self.transitions[state_idx] /= total
                
        # Update time patterns
        hour = datetime.now().hour
        if hour not in self.time_patterns:
            self.time_patterns[hour] = {'count': 0, 'digits': []}
        self.time_patterns[hour]['digits'].append(new_digit)
        if len(self.time_patterns[hour]['digits']) > 100:
            self.time_patterns[hour]['digits'].pop(0)
        self.time_patterns[hour]['count'] += 1
    
    def predict_probabilities(self, recent_digits):
        if len(recent_digits) < self.memory_length:
            return np.ones(10) * 0.1
            
        state_idx = self.state_to_index(recent_digits[-(self.memory_length):])
        return self.transitions[state_idx]
    
    def get_time_based_bias(self):
        """Get digit probabilities based on time of day patterns"""
        hour = datetime.now().hour
        if hour in self.time_patterns and len(self.time_patterns[hour]['digits']) > 20:
            digits = self.time_patterns[hour]['digits']
            counts = np.bincount(digits, minlength=10)
            return counts / np.sum(counts)
        return None

# ──────────────────────────────────────────────
# Micro-Tick Trading Strategy
# ──────────────────────────────────────────────
class MicroTickStrategy:
    def __init__(self):
        self.last_trade_time = 0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.trade_patterns = []
        self.win_rate_threshold = 0.6
        
    def analyze_tick_sequence(self, recent_ticks, recent_digits):
        """Analyze micro-patterns in tick data"""
        if len(recent_ticks) < 10:
            return {'signal': 'wait', 'confidence': 0}
            
        signals = {}
        
        # 1. Tick velocity (speed of price movement)
        tick_changes = np.diff(recent_ticks[-10:])
        velocity = np.mean(np.abs(tick_changes))
        signals['high_velocity'] = velocity > 0.0005
        
        # 2. Tick acceleration
        if len(tick_changes) >= 5:
            acceleration = np.diff(tick_changes[-5:])
            signals['accelerating_up'] = np.mean(acceleration) > 0.0001
            signals['accelerating_down'] = np.mean(acceleration) < -0.0001
            
        # 3. Digit mean reversion
        digits = np.array(recent_digits[-20:])
        mean_digit = np.mean(digits)
        current_digit = digits[-1]
        signals['far_from_mean'] = abs(current_digit - mean_digit) > 3
        
        # 4. Consecutive direction
        if len(recent_ticks) >= 5:
            directions = np.sign(np.diff(recent_ticks[-5:]))
            signals['consistent_up'] = np.all(directions > 0)
            signals['consistent_down'] = np.all(directions < 0)
            
        return signals
    
    def should_trade(self, signals, markov_probs, recent_performance, micro_patterns=None):
        """Decision logic for micro-tick trading"""
        
        confidence_score = 0
        
        # Base confidence from Markov
        if markov_probs is not None:
            max_prob = np.max(markov_probs)
            if max_prob > 0.25:  # Strong probability
                confidence_score += 0.3
        
        # Add signals
        if signals.get('high_velocity') and signals.get('far_from_mean'):
            confidence_score += 0.2
            
        if signals.get('consistent_up') or signals.get('consistent_down'):
            confidence_score += 0.15
            
        # Performance adjustment
        win_rate = recent_performance.get('win_rate', 0.5)
        if win_rate > self.win_rate_threshold:
            confidence_score += 0.1
        elif win_rate < 0.4:
            confidence_score -= 0.1

        # Incorporate micro patterns for hidden digit patterns
        if micro_patterns:
            if micro_patterns.get('low_entropy'):
                confidence_score += 0.15  # High confidence in predictable patterns
            if micro_patterns.get('high_autocorr'):
                confidence_score += 0.1  # Persistent patterns detected
            if micro_patterns.get('volatility_spike'):
                confidence_score += 0.05
            if micro_patterns.get('momentum_positive') or micro_patterns.get('momentum_negative'):
                confidence_score += 0.1
            
        # Time since last trade
        time_since = time.time() - self.last_trade_time
        if time_since < 2:  # Minimum 2 seconds between trades
            return False, 0
            
        return confidence_score > 0.6, confidence_score  # Increased threshold for higher certainty

# ──────────────────────────────────────────────
# Enhanced DerivBot with Micro-Tick Trading - FIXED VERSION
# ──────────────────────────────────────────────
class EnhancedDerivBot:
    def __init__(self, token, app_id, symbol='1HZ75V', recovery_symbol='R_10', 
                 duration=1, stake=0.35, log_queue=None, 
                 take_profit=100.0, stop_loss=-50.0):
        self.ws = None
        self.token = token
        self.app_id = app_id
        self.symbol = symbol
        self.recovery_symbol = recovery_symbol
        self.symbols = list(set([symbol, recovery_symbol]))
        self.duration = duration
        
        # Financial Settings
        self.initial_stake = stake
        self.current_stake = stake
        self.win_stake = stake
        self.total_profit = 0.0
        self.total_loss_to_recover = 0.0
        self.profit_history = []
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        
        # Enhanced tracking
        self.recent_performance = {'win_rate': 0.5, 'last_5': []}
        self.tick_data = {sym: [] for sym in self.symbols}
        self.micro_patterns = {sym: [] for sym in self.symbols}
        
        # Limits
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        
        # Martingale Logic
        self.payout_percent = 39.0
        self.martingale_split = 2
        self.recovery_count = 0
        
        # Enhanced AI Components
        self.pattern_recognizer = {sym: PatternRecognizer() for sym in self.symbols}
        self.markov = {sym: EnhancedMarkovChain(memory_length=4) for sym in self.symbols}  # Increased memory
        self.strategy = MicroTickStrategy()
        
        # Anomaly detection
        self.anomaly_detector = {sym: IsolationForest(contamination=0.1) for sym in self.symbols}
        self.anomaly_data = {sym: [] for sym in self.symbols}
        
        # State Management
        self.active_contract = False
        self.contract_id = None
        self.last_digit = {sym: None for sym in self.symbols}
        self.last_price = {sym: None for sym in self.symbols}
        self.recent_digits = {sym: [] for sym in self.symbols}
        self.history_loaded = {sym: False for sym in self.symbols}
        self.req_id = 1
        
        # Logging & UI
        self.log_queue = log_queue
        self.running = False
        self.paused = False
        
        # Micro-tick settings
        self.min_tick_gap = 1  # Minimum seconds between ticks to trade
        self.last_trade_tick = {sym: 0 for sym in self.symbols}
        self.tick_analysis_window = 20

    def log(self, message):
        timestamp = time.strftime("%H:%M:%S.%f")[:-3]
        full_msg = f"[{timestamp}] {message}"
        if self.log_queue:
            self.log_queue.put(full_msg)
        print(full_msg)

    def connect(self):
        ws_url = f"wss://ws.binaryws.com/websockets/v3?app_id={self.app_id}"
        self.ws = websocket.WebSocketApp(ws_url,
                                         on_open=self.on_open,
                                         on_message=self.on_message,
                                         on_error=self.on_error,
                                         on_close=self.on_close)
        self.running = True
        self.log("🔌 Connecting to Deriv API with enhanced pattern recognition...")
        self.ws.run_forever()

    def send(self, data):
        data['req_id'] = self.req_id
        self.ws.send(json.dumps(data))
        self.req_id += 1

    def on_open(self, ws):
        self.log("✅ Connected. Authorizing...")
        self.send({"authorize": self.token})

    def analyze_micro_patterns(self, symbol):
        """Analyze recent ticks for hidden patterns"""
        if len(self.tick_data[symbol]) < 10:
            return None
            
        recent_ticks = self.tick_data[symbol][-self.tick_analysis_window:]
        recent_digits = self.recent_digits[symbol][-self.tick_analysis_window:]
        
        analysis = {}
        
        # 1. Detect micro-trends
        if len(recent_ticks) >= 5:
            prices = np.array(recent_ticks)
            z = np.polyfit(range(len(prices)), prices, 1)
            analysis['micro_trend'] = z[0]  # Slope
            
        # 2. Detect mean reversion points
        if len(recent_ticks) >= 10:
            prices = np.array(recent_ticks[-10:])
            mean = np.mean(prices)
            std = np.std(prices)
            current = prices[-1]
            analysis['z_score'] = (current - mean) / std if std > 0 else 0
            
        # 3. Detect digit volatility
        if len(recent_digits) >= 15:
            digit_changes = np.diff(recent_digits[-15:])
            analysis['digit_volatility'] = np.std(digit_changes)
            
        # 4. Time-based analysis
        current_minute = datetime.now().minute
        analysis['minute_of_hour'] = current_minute
        
        return analysis

    def should_trade_micro(self, symbol, analysis, markov_probs):
        """Enhanced decision making for micro-tick trading with higher certainty"""
        
        if self.active_contract or self.paused:
            return False, None, None, 0
            
        if len(self.recent_digits[symbol]) < 5:
            return False, None, None, 0
            
        # Check minimum time between trades
        current_time = time.time()
        if current_time - self.last_trade_tick[symbol] < self.min_tick_gap:
            return False, None, None, 0
        
        # Get strategy signals
        recent_ticks = self.tick_data[symbol][-10:] if len(self.tick_data[symbol]) >= 10 else []
        recent_digits = self.recent_digits[symbol][-10:] if len(self.recent_digits[symbol]) >= 10 else []
        
        signals = self.strategy.analyze_tick_sequence(recent_ticks, recent_digits)

        # Get micro patterns for hidden digit insights
        micro_patterns = self.pattern_recognizer[symbol].detect_micro_patterns()
        
        # Combine Markov probabilities with time bias
        time_bias = self.markov[symbol].get_time_based_bias()
        if time_bias is not None and markov_probs is not None:
            # Blend probabilities (70% Markov, 30% time bias)
            blended_probs = 0.7 * markov_probs + 0.3 * time_bias
        else:
            blended_probs = markov_probs
            
        # Get decision from strategy with higher threshold
        should_trade, confidence = self.strategy.should_trade(
            signals, 
            blended_probs, 
            self.recent_performance,
            micro_patterns
        )
        
        if not should_trade:
            return False, None, None, confidence
            
        # Determine contract type based on strongest signal with higher certainty
        if blended_probs is not None:
            under_prob = np.sum(blended_probs[:8])  # Digits 0-7
            over_prob = np.sum(blended_probs[3:])   # Digits 3-9
            
            if under_prob > over_prob and under_prob > 0.75:  # Increased threshold for certainty
                return True, "DIGITUNDER", 7, confidence
            elif over_prob > under_prob and over_prob > 0.75:  # Increased threshold for certainty
                return True, "DIGITOVER", 2, confidence
                
        # Fallback to pattern-based signals
        if analysis and analysis.get('z_score', 0) < -1.5:  # Oversold
            return True, "DIGITOVER", 2, confidence
        elif analysis and analysis.get('z_score', 0) > 1.5:  # Overbought
            return True, "DIGITUNDER", 7, confidence
            
        return False, None, None, confidence

    def buy_micro_contract(self, symbol):
        """Execute micro-tick trade"""
        
        # Get current state for analysis
        recent_digits = self.recent_digits[symbol]
        if len(recent_digits) < 4:  # Adjusted for increased memory
            return
            
        # Get Markov predictions
        markov_probs = self.markov[symbol].predict_probabilities(recent_digits)
        
        # Analyze micro-patterns
        analysis = self.analyze_micro_patterns(symbol)
        
        # Check if we should trade
        should_trade, contract_type, barrier, confidence = self.should_trade_micro(
            symbol, analysis, markov_probs
        )
        
        if not should_trade:
            return
            
        # Choose symbol based on recovery mode
        trade_symbol = self.recovery_symbol if self.recovery_count > 0 else symbol
        
        self.log(f"🔍 MICRO-TRADE Signal | {contract_type} {barrier} on {trade_symbol}")
        self.log(f"   Confidence: {confidence:.2%} | Stake: ${self.current_stake:.2f}")
        
        if analysis:
            self.log(f"   Analysis: Trend={analysis.get('micro_trend',0):.6f}, Z={analysis.get('z_score',0):.2f}")
        
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
                "symbol": trade_symbol,
                "barrier": str(barrier)
            }
        })
        
        self.active_contract = True
        self.last_trade_tick[trade_symbol] = time.time()

    def process_contract(self, data):
        """Process contract results with enhanced analytics"""
        contract = data.get('proposal_open_contract', {})
        if contract.get('is_sold'):
            profit = contract.get('profit', 0)
            self.total_profit += profit
            self.profit_history.append(self.total_profit)
            self.trade_count += 1
            
            # Update performance tracking
            self.recent_performance['last_5'].append(1 if profit > 0 else 0)
            if len(self.recent_performance['last_5']) > 10:
                self.recent_performance['last_5'].pop(0)
            self.recent_performance['win_rate'] = np.mean(self.recent_performance['last_5'])
            
            if profit > 0:
                self.win_count += 1
                self.strategy.consecutive_wins += 1
                self.strategy.consecutive_losses = 0
                self.log(f"✅ WIN +${profit:.2f} | Total: ${self.total_profit:.2f} | WR: {self.recent_performance['win_rate']:.1%}")
                
                if self.recovery_count > 0:
                    self.recovery_count -= 1
                    self.total_loss_to_recover = max(0, self.total_loss_to_recover - profit)
                
                if self.recovery_count == 0:
                    self.current_stake = self.win_stake
                    self.total_loss_to_recover = 0
            else:
                self.loss_count += 1
                self.strategy.consecutive_losses += 1
                self.strategy.consecutive_wins = 0
                self.log(f"❌ LOSS -${abs(profit):.2f}")
                
                self.total_loss_to_recover += abs(self.current_stake)
                self.recovery_count = self.martingale_split
                
                needed = self.total_loss_to_recover * (100 / self.payout_percent)
                self.current_stake = round(needed / self.martingale_split, 2)
                if self.current_stake < 0.35:
                    self.current_stake = 0.35
                self.log(f"   Recovery stake → ${self.current_stake:.2f}")

            # Update strategy patterns
            self.strategy.trade_patterns.append({
                'time': time.time(),
                'profit': profit,
                'stake': self.current_stake
            })
            if len(self.strategy.trade_patterns) > 50:
                self.strategy.trade_patterns.pop(0)

            # Check limits
            if self.total_profit >= self.take_profit:
                self.log(f"🎯 TAKE-PROFIT HIT at ${self.total_profit:.2f}! Pausing trading.")
                self.paused = True
            if self.total_profit <= self.stop_loss:
                self.log(f"🛑 STOP-LOSS TRIGGERED at ${self.total_profit:.2f}! Pausing trading.")
                self.paused = True

            self.active_contract = False
            self.contract_id = None

    def on_message(self, ws, message):
        data = json.loads(message)
        msg_type = data.get('msg_type')
        echo_req = data.get('echo_req', {})

        if msg_type == 'authorize':
            for sym in self.symbols:
                self.send({"ticks_history": sym, "end": "latest", "count": 500, "style": "ticks"})  # More history
        
        elif msg_type == 'history':
            sym = echo_req.get('ticks_history')
            if sym in self.symbols:
                prices = data.get('history', {}).get('prices', [])
                digits = [floor((p * 100) % 10) for p in prices]
                
                # Train Markov chain with historical data
                if len(digits) >= 4:  # Adjusted for memory_length=4
                    for i in range(3, len(digits)):
                        recent_digits = digits[max(0, i-4):i]
                        self.markov[sym].update(digits[i], recent_digits)
                    
                    # Store tick data
                    self.tick_data[sym] = prices[-100:]  # Keep last 100 ticks
                    self.recent_digits[sym] = digits[-100:]
                
                self.history_loaded[sym] = True
                self.log(f"📊 Loaded {len(prices)} historical ticks for {sym}")
            
            # Subscribe once all histories are loaded
            if all(self.history_loaded.values()):
                for sym in self.symbols:
                    self.send({"ticks": sym, "subscribe": 1})
                self.log("✅ All histories loaded. Monitoring ticks for micro-patterns...")

        elif msg_type == 'tick':
            sym = data['tick'].get('symbol')
            if sym in self.symbols:
                quote = data['tick'].get('quote')
                digit = floor((quote * 100) % 10)
                
                # Update tick data
                self.tick_data[sym].append(quote)
                if len(self.tick_data[sym]) > 200:
                    self.tick_data[sym].pop(0)
                
                # Update digit data
                self.last_digit[sym] = digit
                self.last_price[sym] = quote
                self.recent_digits[sym].append(digit)
                if len(self.recent_digits[sym]) > 200:
                    self.recent_digits[sym].pop(0)
                
                # Update anomaly data
                self.anomaly_data[sym].append(digit)
                if len(self.anomaly_data[sym]) > 500:
                    self.anomaly_data[sym].pop(0)
                
                # Periodically fit anomaly detector
                if len(self.anomaly_data[sym]) >= 100 and len(self.anomaly_data[sym]) % 50 == 0:
                    anomaly_features = np.array(self.anomaly_data[sym]).reshape(-1, 1)
                    self.anomaly_detector[sym].fit(anomaly_features)
                
                # Check for anomaly before updating Markov or trading
                if len(self.anomaly_data[sym]) >= 100:
                    current_feature = np.array([[digit]])
                    pred = self.anomaly_detector[sym].predict(current_feature)
                    if pred == -1:
                        self.log(f"⚠️ Anomaly detected in digit {digit} for {sym}. Skipping potential trade.")
                        return  # Skip trading on anomalies to avoid uncertain patterns
                
                # Update Markov chain with memory
                if len(self.recent_digits[sym]) >= 4:
                    recent_digits = self.recent_digits[sym][-5:-1]  # Last 4 digits before current
                    self.markov[sym].update(digit, recent_digits)
                
                # Update pattern recognizer
                self.pattern_recognizer[sym].add_data(quote, digit)
                
                # Look for micro-trading opportunities
                if not self.active_contract and not self.paused:
                    self.buy_micro_contract(sym)

        elif msg_type == 'buy':
            if 'error' in data:
                self.log(f"❌ Buy failed: {data['error']['message']}")
                self.active_contract = False
            else:
                self.contract_id = data['buy']['contract_id']
                self.send({"proposal_open_contract": 1, "contract_id": self.contract_id, "subscribe": 1})

        elif msg_type == 'proposal_open_contract':
            self.process_contract(data)

    def on_error(self, ws, error): 
        self.log(f"⚠️ WebSocket Error: {error}")

    def on_close(self, ws, *args):  # FIXED: Accept variable arguments
        self.log("🔌 Connection closed.")
        self.running = False

    def stop(self):
        if self.ws:
            self.ws.close()
        self.running = False
        self.paused = False

# ──────────────────────────────────────────────
# Enhanced Streamlit UI with Pattern Visualization - FIXED VERSION
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Micro-Tick Trading Bot • Advanced Pattern Recognition",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    /* Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #ffffff;
    }
    
    /* Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        margin: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #00c6ff, #0072ff);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 114, 255, 0.4);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
    }
    
    /* Inputs */
    .stTextInput>div>div>input, 
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select {
        background: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 8px;
    }
    
    /* Charts */
    .stPlotlyChart {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 15px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 AI 2-7 Trading Bot")
st.markdown("### Advanced Pattern Recognition & High-Frequency Trading")

# Session state
if 'bot' not in st.session_state:
    st.session_state.bot = None
if 'log_queue' not in st.session_state:
    st.session_state.log_queue = queue.Queue()
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'patterns' not in st.session_state:
    st.session_state.patterns = {'detected': [], 'performance': {}}

# ─── SIDEBAR ─── Enhanced Controls
with st.sidebar:
    st.header("⚙️ Trading Configuration")
    
    with st.expander("🔐 API Settings", expanded=True):
        token = st.text_input("API Token", value="", type="password")
        app_id = st.text_input("App ID", value="1089")
    
    with st.expander("📊 Trading Parameters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.selectbox("Main Symbol", ["1HZ75V", "R_100", "R_10"], index=0)
            duration = st.number_input("Tick Duration", value=1, min_value=1, max_value=10)
        with col2:
            recovery_symbol = st.selectbox("Recovery Symbol", ["R_10", "1HZ75V", "R_100"], index=0)
            stake = st.number_input("Base Stake (USD)", value=0.35, min_value=0.35, step=0.05)
    
    with st.expander("🎯 Profit/Loss Limits", expanded=True):
        take_profit = st.number_input("Take Profit (USD)", value=100.0, step=5.0)
        stop_loss = st.number_input("Stop Loss (USD)", value=-50.0, step=5.0)
    
    with st.expander("🤖 AI Settings", expanded=True):
        ai_aggressiveness = st.slider("AI Aggressiveness", 1, 10, 7, 
                                       help="Higher = more frequent trades")
        pattern_sensitivity = st.slider("Pattern Sensitivity", 1, 10, 8,
                                         help="How sensitive to micro-patterns")
    
    st.markdown("---")
    
    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("🚀 Start AI Bot", type="primary", use_container_width=True):
            if st.session_state.bot is None or not st.session_state.bot.running:
                st.session_state.logs = []
                st.session_state.log_queue = queue.Queue()
                st.session_state.bot = EnhancedDerivBot(
                    token, app_id, symbol, recovery_symbol, duration, stake, 
                    st.session_state.log_queue, take_profit, stop_loss
                )
                threading.Thread(target=st.session_state.bot.connect, daemon=True).start()
                st.success("🤖 AI Bot started with micro-tick analysis!")
                st.rerun()
    
    with col_stop:
        if st.button("🛑 Stop Bot", type="secondary", use_container_width=True):
            if st.session_state.bot and st.session_state.bot.running:
                st.session_state.bot.stop()
                st.success("Bot stopped!")
                st.rerun()

# ─── MAIN DASHBOARD ───
tab_dashboard, tab_patterns, tab_logs, tab_analytics = st.tabs([
    "📊 Live Dashboard", 
    "🔍 Pattern Analysis", 
    "📜 Trading Logs", 
    "📈 Advanced Analytics"
])

with tab_dashboard:
    if st.session_state.bot:
        bot = st.session_state.bot
        
        # Top Metrics
        cols = st.columns(5)
        with cols[0]:
            st.metric("Total Profit", f"${bot.total_profit:.2f}")
        with cols[1]:
            st.metric("Win Rate", f"{bot.recent_performance['win_rate']:.1%}")
        with cols[2]:
            st.metric("Current Stake", f"${bot.current_stake:.2f}")
        with cols[3]:
            st.metric("Active Trades", f"{1 if bot.active_contract else 0}")
        with cols[4]:
            status = "🟢 Active" if bot.running else "🔴 Stopped"
            status += " ⏸️" if bot.paused else ""
            st.metric("Status", status)
        
        st.markdown("---")
        
        # Charts - FIXED use_container_width warnings
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Profit Evolution")
            if bot.profit_history:
                df_profit = pd.DataFrame({
                    "Profit": bot.profit_history,
                    "Trade": range(len(bot.profit_history))
                })
                # FIXED: Using width instead of use_container_width
                st.line_chart(df_profit.set_index("Trade")["Profit"], width='stretch')
            else:
                st.info("Waiting for trades...")
        
        with col2:
            st.subheader("🎰 Recent Digits Pattern")
            if bot.recent_digits.get(bot.symbol):
                digits = bot.recent_digits[bot.symbol][-30:]
                df_digits = pd.DataFrame({
                    "Digit": digits,
                    "Index": range(len(digits))
                })
                # FIXED: Using width instead of use_container_width
                st.area_chart(df_digits.set_index("Index")["Digit"], width='stretch')
                
                # Current digit info
                current_digit = bot.last_digit.get(bot.symbol)
                if current_digit is not None:
                    st.info(f"**Current Digit:** {current_digit}")
        
        # Real-time analysis
        st.subheader("🔍 Real-time Analysis")
        if bot.recent_digits.get(bot.symbol) and len(bot.recent_digits[bot.symbol]) >= 10:
            recent_digits = bot.recent_digits[bot.symbol][-10:]
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                # Markov predictions
                if len(recent_digits) >= 4:
                    probs = bot.markov[bot.symbol].predict_probabilities(recent_digits)
                    st.write("**Next Digit Probabilities:**")
                    for i, prob in enumerate(probs):
                        if prob > 0.15:
                            st.progress(float(prob), text=f"Digit {i}: {prob:.1%}")
            
            with col_b:
                # Pattern detection
                if bot.tick_data.get(bot.symbol):
                    analysis = bot.analyze_micro_patterns(bot.symbol)
                    if analysis:
                        st.write("**Micro-Patterns:**")
                        for key, value in analysis.items():
                            if isinstance(value, (int, float)):
                                st.write(f"{key}: {value:.4f}")
            
            with col_c:
                # Trading signals
                if bot.last_price.get(bot.symbol):
                    signals = bot.strategy.analyze_tick_sequence(
                        bot.tick_data[bot.symbol][-10:] if len(bot.tick_data[bot.symbol]) >= 10 else [],
                        recent_digits
                    )
                    st.write("**Trading Signals:**")
                    for key, value in signals.items():
                        if value:
                            st.write(f"✓ {key}")
    else:
        st.info("🚀 Start the AI bot to begin micro-tick trading and pattern analysis!")

with tab_patterns:
    st.subheader("🔍 Detected Patterns & Anomalies")
    
    if st.session_state.bot:
        bot = st.session_state.bot
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Markov Chain State Analysis**")
            if bot.recent_digits.get(bot.symbol) and len(bot.recent_digits[bot.symbol]) >= 4:
                recent = bot.recent_digits[bot.symbol][-4:]
                probs = bot.markov[bot.symbol].predict_probabilities(recent)
                
                df_probs = pd.DataFrame({
                    "Digit": range(10),
                    "Probability": probs
                }).sort_values("Probability", ascending=False)
                
                # FIXED: Using width parameter
                st.dataframe(df_probs, width='stretch')
        
        with col2:
            st.write("**⏰ Time-based Patterns**")
            if hasattr(bot.markov[bot.symbol], 'time_patterns'):
                time_patterns = bot.markov[bot.symbol].time_patterns
                if time_patterns:
                    hours = list(time_patterns.keys())
                    counts = [time_patterns[h]['count'] for h in hours]
                    
                    df_time = pd.DataFrame({
                        "Hour": hours,
                        "Trade Count": counts
                    }).sort_values("Hour")
                    
                    # FIXED: Using width parameter
                    st.bar_chart(df_time.set_index("Hour")["Trade Count"], width='stretch')
        
        # Pattern history
        st.write("**🔄 Recent Trading Patterns**")
        if bot.strategy.trade_patterns:
            patterns_df = pd.DataFrame(bot.strategy.trade_patterns)
            if not patterns_df.empty:
                # FIXED: Using width parameter
                st.dataframe(patterns_df, width='stretch')

        # Additional pattern visuals
        st.markdown("---")
        st.subheader("Additional Pattern Visuals")

        col3, col4 = st.columns(2)

        with col3:
            st.write("**📈 Digit Frequency Distribution**")
            if bot.recent_digits.get(bot.symbol):
                digits = bot.recent_digits[bot.symbol]
                if len(digits) > 0:
                    digit_counts = pd.Series(digits).value_counts().sort_index()
                    df_counts = pd.DataFrame({"Digit": digit_counts.index, "Count": digit_counts.values})
                    st.bar_chart(df_counts.set_index("Digit")["Count"], width='stretch')
                else:
                    st.info("No digits available yet.")

        with col4:
            st.write("**🔄 Digit Autocorrelation**")
            if bot.recent_digits.get(bot.symbol):
                digits = bot.recent_digits[bot.symbol]
                if len(digits) >= 20:
                    autocorrs = [pd.Series(digits).autocorr(lag=i) for i in range(1, 11)]
                    df_ac = pd.DataFrame({"Lag": range(1, 11), "Autocorrelation": autocorrs})
                    st.line_chart(df_ac.set_index("Lag")["Autocorrelation"], width='stretch')
                else:
                    st.info("Insufficient data for autocorrelation.")

        # More visuals: Entropy metric
        if bot.pattern_recognizer.get(bot.symbol):
            patterns = bot.pattern_recognizer[bot.symbol].detect_micro_patterns()
            if patterns and 'low_entropy' in patterns:
                entropy_status = "Low (Predictable)" if patterns['low_entropy'] else "High (Random)"
                st.metric("Current Entropy Level", entropy_status)

        # Digit sequence line chart
        st.write("**📉 Recent Digit Sequence**")
        if bot.recent_digits.get(bot.symbol):
            digits = bot.recent_digits[bot.symbol][-50:]
            if len(digits) > 0:
                df_seq = pd.DataFrame({"Index": range(len(digits)), "Digit": digits})
                st.line_chart(df_seq.set_index("Index")["Digit"], width='stretch')
            else:
                st.info("No digits available yet.")

with tab_logs:
    st.subheader("📜 Live Trading Logs")
    
    # Update logs from queue
    while True:
        try:
            msg = st.session_state.log_queue.get_nowait()
            st.session_state.logs.append(msg)
        except queue.Empty:
            break
    
    log_container = st.expander("Live Logs", expanded=True)
    with log_container:
        for log in st.session_state.logs[-50:]:
            st.write(log)
        
        # Controls
        col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Clear Logs", use_container_width=True):
            st.session_state.logs = []
            st.rerun()
    with col2:
        auto_refresh = st.checkbox("Auto-refresh (3s)", value=True)
    with col3:
        st.write(f"Total Logs: {len(st.session_state.logs)}")

with tab_analytics:
    st.subheader("📈 Advanced Trading Analytics")
    
    if st.session_state.bot:
        bot = st.session_state.bot
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            if bot.trade_count > 0:
                avg_win = bot.win_count / bot.trade_count * bot.initial_stake * 0.39 if bot.win_count > 0 else 0
                st.metric("Avg Win", f"${avg_win:.2f}")
        with col2:
            if bot.trade_count > 0:
                avg_loss = bot.loss_count / bot.trade_count * bot.initial_stake if bot.loss_count > 0 else 0
                profit_factor = (bot.win_count * avg_win) / (bot.loss_count * avg_loss) if bot.loss_count > 0 else float('inf')
                st.metric("Profit Factor", f"{profit_factor:.2f}")
        with col3:
            if bot.strategy.consecutive_wins > 0 or bot.strategy.consecutive_losses > 0:
                streak = f"W{bot.strategy.consecutive_wins}" if bot.strategy.consecutive_wins > 0 else f"L{bot.strategy.consecutive_losses}"
                st.metric("Current Streak", streak)
        
        # Detailed analysis - FIXED Arrow serialization error
        st.write("**📋 Trade Statistics**")
        if bot.trade_count > 0:
            # Create proper DataFrame with consistent data types
            win_rate_val = bot.win_count / bot.trade_count
            avg_profit_val = bot.total_profit / bot.trade_count
            
            stats_data = {
                "Total Trades": [bot.trade_count],
                "Wins": [bot.win_count],
                "Losses": [bot.loss_count],
                "Win Rate %": [f"{win_rate_val:.1%}"],
                "Total Profit": [f"${bot.total_profit:.2f}"],
                "Avg Profit/Trade": [f"${avg_profit_val:.2f}"],
                "Consecutive Wins": [max(bot.strategy.consecutive_wins, 0)],
                "Consecutive Losses": [max(bot.strategy.consecutive_losses, 0)]
            }
            
            stats_df = pd.DataFrame(stats_data)
            # FIXED: Using width parameter
            st.dataframe(stats_df.T, width='stretch')
        
        # Recovery analysis
        if bot.recovery_count > 0:
            st.warning(f"**Recovery Mode Active:** {bot.recovery_count} trades remaining")
            st.write(f"Loss to recover: ${bot.total_loss_to_recover:.2f}")
            st.write(f"Current stake: ${bot.current_stake:.2f}")

# Footer
st.markdown("---")
st.caption("""
    🤖 AI Micro-Tick Trading Bot v2.0 | 
    Built with advanced pattern recognition • 
    Trading involves significant risk • 
    Use at your own discretion
""")

# Auto-refresh logic
if 'tab_logs' in locals() and auto_refresh and st.session_state.bot and st.session_state.bot.running:
    time.sleep(3)
    st.rerun()