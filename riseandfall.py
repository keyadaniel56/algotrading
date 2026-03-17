# deriv_volatility_rise_fall_multi_tf_bot.py
# Requirements: pip install websocket-client numpy rich
import json
import time
import threading
import signal
from collections import defaultdict, deque
import numpy as np
import websocket
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.console import Console
from rich import box
from datetime import datetime
import os

# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────
APP_ID = "1089"
AUTH_TOKEN = "WFabi7aeCbFjgvp"          # ← CHANGE THIS (create at app.deriv.com)
SYMBOLS = ["R_10", "R_25", "R_50", "R_75", "R_100"]
SYMBOL_NAMES = {
    "R_10": "Volatility 10",
    "R_25": "Volatility 25", 
    "R_50": "Volatility 50",
    "R_75": "Volatility 75",
    "R_100": "Volatility 100"
}

INITIAL_STAKE = 0.35
MARTINGALE_MULTIPLIER = 1.8                  # Set 1.0 to disable martingale
MAX_CONCURRENT_TRADES = 3
STOP_LOSS = -15.0
TAKE_PROFIT = 5.0
MAX_LOSSES = 25

# Timeframes (seconds)
ANALYSIS_TFS = [60, 300, 900]                # 1m, 5m, 15m for trend confirmation
TF_NAMES = {60: "1m", 300: "5m", 900: "15m"}
ENTRY_DURATION = 30                          # Lower timeframe trade duration

# Trend detection settings (price action based)
TREND_STRENGTH_CANDLES = 5                    # Number of candles to confirm trend
HIGHER_TIMEFRAME_BIAS = True                  # Give more weight to higher timeframe
MIN_TREND_CANDLES = 3                          # Minimum candles needed for trend detection

WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"

# ────────────────────────────────────────────────
# GLOBAL STATE
# ────────────────────────────────────────────────
total_profit = 0.0
current_stake = INITIAL_STAKE
consecutive_losses = 0
open_trades = 0
running = True
ws = None
last_tick_time = time.time()
trade_check_counter = 0

# Per-symbol data
price_history = {sym: deque(maxlen=5000) for sym in SYMBOLS}           # raw ticks
candles = {sym: {tf: [] for tf in ANALYSIS_TFS} for sym in SYMBOLS}   # OHLC per TF
last_candle_time = {sym: {tf: 0 for tf in ANALYSIS_TFS} for sym in SYMBOLS}
last_trade_time = {sym: 0 for sym in SYMBOLS}
trend_cache = {sym: None for sym in SYMBOLS}
last_prices = {sym: 0.0 for sym in SYMBOLS}
price_change = {sym: 0.0 for sym in SYMBOLS}
trade_history = deque(maxlen=10)  # Last 10 trades

lock = threading.Lock()
console = Console()

# ────────────────────────────────────────────────
# PRICE ACTION TREND DETECTION
# ────────────────────────────────────────────────
def detect_price_action_trend(candles_list: list, lookback: int = 5) -> str | None:
    """Detect trend using price action"""
    if len(candles_list) < lookback + 1:
        return None
    
    recent_candles = candles_list[-lookback:]
    highs = [c['high'] for c in recent_candles]
    lows = [c['low'] for c in recent_candles]
    closes = [c['close'] for c in recent_candles]
    
    # Check for uptrend: higher highs AND higher lows
    higher_highs = all(highs[i] >= highs[i-1] for i in range(1, len(highs)))
    higher_lows = all(lows[i] >= lows[i-1] for i in range(1, len(lows)))
    
    # Check for downtrend: lower highs AND lower lows
    lower_highs = all(highs[i] <= highs[i-1] for i in range(1, len(highs)))
    lower_lows = all(lows[i] <= lows[i-1] for i in range(1, len(lows)))
    
    # Check candle bodies for momentum
    bullish_candles = sum(1 for c in recent_candles[-3:] if c['close'] > c['open'])
    bearish_candles = sum(1 for c in recent_candles[-3:] if c['close'] < c['open'])
    
    if (higher_highs or higher_lows) and bullish_candles >= 2:
        return "UP"
    
    if (lower_highs or lower_lows) and bearish_candles >= 2:
        return "DOWN"
    
    return None

def detect_trend_multi_tf(symbol: str) -> str | None:
    """Detect trend across multiple timeframes"""
    tf_trends = {}
    
    for tf in ANALYSIS_TFS:
        if len(candles[symbol][tf]) < MIN_TREND_CANDLES + 2:
            return None
        
        trend = detect_price_action_trend(candles[symbol][tf], TREND_STRENGTH_CANDLES)
        if trend:
            tf_trends[tf] = trend
    
    if len(tf_trends) < len(ANALYSIS_TFS):
        return None
    
    unique_trends = set(tf_trends.values())
    if len(unique_trends) == 1:
        return list(unique_trends)[0]
    
    if HIGHER_TIMEFRAME_BIAS:
        highest_tf = max(ANALYSIS_TFS)
        if highest_tf in tf_trends:
            return tf_trends[highest_tf]
    
    return None

# ────────────────────────────────────────────────
# CANDLE BUILDER
# ────────────────────────────────────────────────
def update_candles(symbol: str, price: float, timestamp: int):
    for tf in ANALYSIS_TFS:
        candle_start = (timestamp // tf) * tf
        
        if candle_start > last_candle_time[symbol][tf]:
            if candles[symbol][tf]:
                candles[symbol][tf][-1]['close'] = price
            
            candles[symbol][tf].append({
                'open': price, 'high': price, 'low': price, 'close': price,
                'time': candle_start
            })
            last_candle_time[symbol][tf] = candle_start
        else:
            if candles[symbol][tf]:
                c = candles[symbol][tf][-1]
                c['high'] = max(c['high'], price)
                c['low'] = min(c['low'], price)
                c['close'] = price

# ────────────────────────────────────────────────
# TRADE DECISION & EXECUTION
# ────────────────────────────────────────────────
def check_and_trade():
    global open_trades, current_stake
    
    with lock:
        if open_trades >= MAX_CONCURRENT_TRADES:
            return

        now = time.time()
        for symbol in SYMBOLS:
            if now - last_trade_time[symbol] < ENTRY_DURATION + 5:
                continue

            if len(price_history[symbol]) < 50:
                continue

            trend = detect_trend_multi_tf(symbol)
            if not trend:
                continue

            contract_type = "CALL" if trend == "UP" else "PUT"
            
            # Send proposal
            proposal_request = {
                "proposal": 1,
                "amount": current_stake,
                "basis": "stake",
                "contract_type": contract_type,
                "currency": "USD",
                "duration": ENTRY_DURATION,
                "duration_unit": "s",
                "symbol": symbol
            }
            ws.send(json.dumps(proposal_request))

            last_trade_time[symbol] = now
            open_trades += 1
            break

# ────────────────────────────────────────────────
# UI GENERATION
# ────────────────────────────────────────────────
def generate_ui():
    layout = Layout()
    
    # Split into header, main content, and footer
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=6)
    )
    
    # Split main into left (market data) and right (status)
    layout["main"].split_row(
        Layout(name="market_data", ratio=2),
        Layout(name="status", ratio=1)
    )
    
    # Header
    header_text = Text("Algocdk VOLATILITY PRICE ACTION TRADING BOT", style="bold cyan")
    header_text.append(f" | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="yellow")
    layout["header"].update(Panel(header_text, box=box.ROUNDED))
    
    # Market Data Table
    table = Table(title="Market Analysis", box=box.ROUNDED, header_style="bold magenta")
    table.add_column("Symbol", style="cyan", width=15)
    table.add_column("Price", justify="right", width=12)
    table.add_column("Change", justify="right", width=10)
    for tf in ANALYSIS_TFS:
        table.add_column(TF_NAMES[tf], justify="center", width=8)
    table.add_column("Action", justify="center", width=10)
    
    with lock:
        for symbol in SYMBOLS:
            # Price and change
            current_price = last_prices.get(symbol, 0)
            change = price_change.get(symbol, 0)
            change_str = f"{change:+.4f}"
            change_style = "green" if change > 0 else "red" if change < 0 else "white"
            
            # Trend for each timeframe
            tf_trends = []
            for tf in ANALYSIS_TFS:
                if len(candles[symbol][tf]) >= MIN_TREND_CANDLES:
                    trend = detect_price_action_trend(candles[symbol][tf], 3)
                    if trend == "UP":
                        tf_trends.append("▲ UP")
                    elif trend == "DOWN":
                        tf_trends.append("▼ DOWN")
                    else:
                        tf_trends.append("─")
                else:
                    tf_trends.append("⏳")
            
            # Overall trend and action
            overall_trend = detect_trend_multi_tf(symbol)
            if overall_trend == "UP":
                action = "📈 CALL"
                action_style = "green"
            elif overall_trend == "DOWN":
                action = "📉 PUT"
                action_style = "red"
            else:
                action = "⏸️ WAIT"
                action_style = "yellow"
            
            table.add_row(
                SYMBOL_NAMES.get(symbol, symbol),
                f"{current_price:.4f}",
                Text(change_str, style=change_style),
                *tf_trends,
                Text(action, style=action_style)
            )
    
    layout["market_data"].update(Panel(table, box=box.ROUNDED))
    
    # Status Panel
    status_table = Table(show_header=False, box=box.ROUNDED, padding=(0,1))
    status_table.add_column("Metric", style="cyan")
    status_table.add_column("Value", style="yellow")
    
    with lock:
        status_table.add_row("Total P&L", f"${total_profit:+.2f}")
        status_table.add_row("Current Stake", f"${current_stake:.2f}")
        status_table.add_row("Open Trades", f"{open_trades}/{MAX_CONCURRENT_TRADES}")
        status_table.add_row("Consecutive Losses", str(consecutive_losses))
        status_table.add_row("Win Rate", "0%" if open_trades == 0 else "0%")  # Calculate if needed
        status_table.add_row("Uptime", f"{int(time.time() - start_time)}s")
    
    layout["status"].update(Panel(status_table, title="Bot Status", box=box.ROUNDED))
    
    # Trade History Footer
    history_table = Table(title="Recent Trades", box=box.ROUNDED, header_style="bold blue")
    history_table.add_column("Time", width=8)
    history_table.add_column("Symbol", width=12)
    history_table.add_column("Type", width=6)
    history_table.add_column("Result", width=8)
    history_table.add_column("P&L", width=10)
    
    with lock:
        for trade in list(trade_history):
            result_style = "green" if trade['profit'] > 0 else "red"
            history_table.add_row(
                trade['time'],
                trade['symbol'],
                trade['type'],
                "WIN" if trade['profit'] > 0 else "LOSS",
                Text(f"${trade['profit']:+.2f}", style=result_style)
            )
    
    layout["footer"].update(Panel(history_table, box=box.ROUNDED))
    
    return layout

# ────────────────────────────────────────────────
# WEBSOCKET HANDLERS
# ────────────────────────────────────────────────
def on_open(ws_obj):
    ws_obj.send(json.dumps({"authorize": AUTH_TOKEN}))

def on_message(ws_obj, message):
    global total_profit, current_stake, open_trades, consecutive_losses, last_tick_time
    
    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        return

    if "error" in data:
        return

    mt = data.get("msg_type")

    if mt == "authorize":
        for sym in SYMBOLS:
            ws_obj.send(json.dumps({"ticks": sym, "subscribe": 1}))
            ws_obj.send(json.dumps({
                "ticks_history": sym,
                "count": 200,
                "end": "latest",
                "style": "ticks"
            }))

    elif mt == "tick":
        sym = data["tick"]["symbol"]
        price = float(data["tick"]["quote"])
        ts = int(data["tick"]["epoch"])
        last_tick_time = time.time()
        
        with lock:
            if sym in last_prices:
                price_change[sym] = price - last_prices[sym]
            last_prices[sym] = price
            price_history[sym].append(price)
            update_candles(sym, price, ts)

    elif mt == "history":
        sym = data["history"]["symbol"]
        prices = data["history"]["prices"]
        times = data["history"]["times"]
        
        with lock:
            for price, ts in zip(prices, times):
                price = float(price)
                ts = int(ts)
                price_history[sym].append(price)
                update_candles(sym, price, ts)

    elif mt == "proposal":
        proposal_id = data["proposal"]["id"]
        ask_price = data["proposal"]["ask_price"]
        ws_obj.send(json.dumps({"buy": proposal_id, "price": ask_price}))

    elif mt == "buy":
        contract_id = data["buy"]["contract_id"]
        contract_type = data["buy"]["contract_type"]
        symbol = data["buy"]["symbol"]
        buy_price = data["buy"]["buy_price"]

    elif mt == "proposal_open_contract":
        contract = data["proposal_open_contract"]
        
        if contract.get("is_sold") == 1:
            profit = float(contract.get("profit", 0))
            symbol = contract.get("symbol", "Unknown")
            contract_type = contract.get("contract_type", "Unknown")
            
            total_profit += profit
            open_trades = max(0, open_trades - 1)

            # Add to trade history
            trade_time = datetime.now().strftime("%H:%M:%S")
            trade_history.append({
                'time': trade_time,
                'symbol': SYMBOL_NAMES.get(symbol, symbol),
                'type': contract_type,
                'profit': profit
            })

            with lock:
                if profit > 0:
                    consecutive_losses = 0
                    current_stake = INITIAL_STAKE
                else:
                    consecutive_losses += 1
                    current_stake *= MARTINGALE_MULTIPLIER

            # Check risk limits
            if total_profit <= STOP_LOSS or total_profit >= TAKE_PROFIT or consecutive_losses >= MAX_LOSSES:
                global running
                running = False

def on_error(ws_obj, err): 
    pass

def on_close(ws_obj, *args): 
    global running
    running = False

# ────────────────────────────────────────────────
# MAIN LOOP
# ────────────────────────────────────────────────
def main():
    global ws, running, last_tick_time, total_profit, open_trades, consecutive_losses, current_stake, start_time
    
    start_time = time.time()
    
    # Clear screen
    os.system('cls' if os.name == 'nt' else 'clear')
    
    console.print("[bold cyan]Starting Deriv Volatility Trading Bot...[/bold cyan]")
    
    websocket.enableTrace(False)
    ws = websocket.WebSocketApp(
        WS_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    t = threading.Thread(target=ws.run_forever, daemon=True)
    t.start()

    # Live UI update
    with Live(generate_ui(), refresh_per_second=4, screen=True) as live:
        last_trade_check = time.time()
        
        try:
            while running:
                now = time.time()
                
                # Check for trades every 5 seconds
                if now - last_trade_check >= 5:
                    check_and_trade()
                    last_trade_check = now
                
                # Update UI
                live.update(generate_ui())
                time.sleep(0.25)
                
        except KeyboardInterrupt:
            running = False
    
    if ws:
        ws.close()
    
    console.print("\n[bold red]Bot Stopped[/bold red]")
    console.print(f"[yellow]Final P&L: ${total_profit:+.2f}[/yellow]")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s,f: globals().update({"running": False}))
    main()