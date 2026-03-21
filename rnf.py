# deriv_volatility_1tick_1min_signal_bot.py
# ────────────────────────────────────────────────
# 1-TICK MARTINGALE BOT – SIGNAL BASED ONLY ON 1-MINUTE TIMEFRAME
# Requirements: pip install websocket-client numpy rich
# ────────────────────────────────────────────────

import json
import time
import threading
import signal
from collections import deque
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
AUTH_TOKEN = "WFabi7aeCbFjgvp"  # ← CHANGE THIS (from app.deriv.com)

SYMBOLS = ["R_10", "R_25", "R_50", "R_75", "R_100"]
SYMBOL_NAMES = {
    "R_10": "Volatility 10", "R_25": "Volatility 25", "R_50": "Volatility 50",
    "R_75": "Volatility 75", "R_100": "Volatility 100"
}

INITIAL_STAKE = 0.35
MARTINGALE_MULTIPLIER = 1.8
MAX_CONCURRENT_TRADES = 1           # Only one trade/market at a time
TAKE_PROFIT = 5.0
MAX_LOSSES = 25

ENTRY_DURATION = 1
DURATION_UNIT = "t"                 # 1 tick

# Only 1-minute timeframe is used for signal
SIGNAL_TF = 60                      # 1 minute
TREND_STRENGTH_CANDLES = 5          # Lookback for trend detection

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
trade_history = deque(maxlen=10)
lock = threading.Lock()
console = Console()

price_history = {sym: deque(maxlen=5000) for sym in SYMBOLS}
candles_1m = {sym: [] for sym in SYMBOLS}           # Only keeping 1m candles
last_candle_time_1m = {sym: 0 for sym in SYMBOLS}
last_prices = {sym: 0.0 for sym in SYMBOLS}
price_change = {sym: 0.0 for sym in SYMBOLS}
last_trade_time = {sym: 0 for sym in SYMBOLS}

# ────────────────────────────────────────────────
# 1-MINUTE PRICE ACTION TREND DETECTION
# ────────────────────────────────────────────────
def detect_1m_trend(candles_list: list, lookback: int = TREND_STRENGTH_CANDLES) -> str | None:
    if len(candles_list) < lookback + 1:
        return None

    recent = candles_list[-lookback:]
    highs = [c['high'] for c in recent]
    lows  = [c['low']  for c in recent]

    higher_highs = all(highs[i] >= highs[i-1] for i in range(1, len(highs)))
    higher_lows  = all(lows[i]  >= lows[i-1]  for i in range(1, len(lows)))
    lower_highs  = all(highs[i] <= highs[i-1] for i in range(1, len(highs)))
    lower_lows   = all(lows[i]  <= lows[i-1]  for i in range(1, len(lows)))

    recent_3 = recent[-3:]
    bullish_count = sum(1 for c in recent_3 if c['close'] > c['open'])
    bearish_count = sum(1 for c in recent_3 if c['close'] < c['open'])

    if (higher_highs or higher_lows) and bullish_count >= 2:
        return "UP"
    if (lower_highs or lower_lows) and bearish_count >= 2:
        return "DOWN"
    return None

# ────────────────────────────────────────────────
# CANDLE BUILDER – ONLY 1-MINUTE
# ────────────────────────────────────────────────
def update_1m_candles(symbol: str, price: float, timestamp: int):
    candle_start = (timestamp // SIGNAL_TF) * SIGNAL_TF
    if candle_start > last_candle_time_1m[symbol]:
        if candles_1m[symbol]:
            candles_1m[symbol][-1]['close'] = price
        candles_1m[symbol].append({
            'open': price, 'high': price, 'low': price, 'close': price,
            'time': candle_start
        })
        last_candle_time_1m[symbol] = candle_start
    else:
        if candles_1m[symbol]:
            c = candles_1m[symbol][-1]
            c['high'] = max(c['high'], price)
            c['low']  = min(c['low'], price)
            c['close'] = price

# ────────────────────────────────────────────────
# TRADE DECISION – ONLY 1-MINUTE SIGNAL
# ────────────────────────────────────────────────
def check_and_trade():
    global open_trades, current_stake
    with lock:
        if open_trades >= MAX_CONCURRENT_TRADES:
            return
        now = time.time()
        for symbol in SYMBOLS:
            if now - last_trade_time[symbol] < 10:  # short cooldown
                continue
            if len(candles_1m[symbol]) < TREND_STRENGTH_CANDLES + 2:
                continue

            trend = detect_1m_trend(candles_1m[symbol])
            if not trend:
                continue

            contract_type = "CALL" if trend == "UP" else "PUT"

            proposal = {
                "proposal": 1,
                "amount": current_stake,
                "basis": "stake",
                "contract_type": contract_type,
                "currency": "USD",
                "duration": ENTRY_DURATION,
                "duration_unit": DURATION_UNIT,
                "symbol": symbol
            }
            ws.send(json.dumps(proposal))
            last_trade_time[symbol] = now
            open_trades += 1
            break  # Only one trade at a time

# ────────────────────────────────────────────────
# UI (shows 1m trend prominently)
# ────────────────────────────────────────────────
def generate_ui():
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=6)
    )
    layout["main"].split_row(
        Layout(name="market", ratio=2),
        Layout(name="status", ratio=1)
    )

    header = Text("1-TICK MARTINGALE BOT – 1 MINUTE SIGNAL ONLY", style="bold cyan")
    header.append(f" | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="yellow")
    layout["header"].update(Panel(header, box=box.ROUNDED))

    table = Table(title="Markets (Signal from 1m only)", box=box.ROUNDED, header_style="bold magenta")
    table.add_column("Symbol", style="cyan", width=15)
    table.add_column("Price", justify="right", width=12)
    table.add_column("Change", justify="right", width=10)
    table.add_column("1m Trend", justify="center", width=10)
    table.add_column("Action", justify="center", width=12)

    with lock:
        for sym in SYMBOLS:
            price = last_prices.get(sym, 0)
            chg = price_change.get(sym, 0)
            chg_str = f"{chg:+.4f}"
            chg_style = "green" if chg > 0 else "red" if chg < 0 else "white"

            trend_1m = "▲ UP" if detect_1m_trend(candles_1m[sym]) == "UP" else \
                       "▼ DOWN" if detect_1m_trend(candles_1m[sym]) == "DOWN" else "─"

            action = "📈 CALL" if trend_1m == "▲ UP" else \
                     "📉 PUT" if trend_1m == "▼ DOWN" else "⏸ WAIT"
            action_style = "green" if "CALL" in action else "red" if "PUT" in action else "yellow"

            table.add_row(
                SYMBOL_NAMES.get(sym, sym),
                f"{price:.4f}",
                Text(chg_str, style=chg_style),
                trend_1m,
                Text(action, style=action_style)
            )

    layout["market"].update(Panel(table, box=box.ROUNDED))

    status = Table(show_header=False, box=box.ROUNDED, padding=(0,1))
    status.add_column("Metric", style="cyan")
    status.add_column("Value", style="yellow")
    with lock:
        status.add_row("P&L", f"${total_profit:+.2f}")
        status.add_row("Stake", f"${current_stake:.2f}")
        status.add_row("Trades", f"{open_trades}/{MAX_CONCURRENT_TRADES}")
        status.add_row("Loss streak", str(consecutive_losses))
        status.add_row("Uptime", f"{int(time.time() - start_time)}s")

    layout["status"].update(Panel(status, title="Status", box=box.ROUNDED))

    hist = Table(title="Recent Trades", box=box.ROUNDED, header_style="bold blue")
    hist.add_column("Time", width=8)
    hist.add_column("Symbol", width=12)
    hist.add_column("Type", width=6)
    hist.add_column("Result", width=8)
    hist.add_column("P&L", width=10)
    with lock:
        for t in list(trade_history):
            style = "green" if t['profit'] > 0 else "red"
            hist.add_row(
                t['time'], t['symbol'], t['type'],
                "WIN" if t['profit'] > 0 else "LOSS",
                Text(f"${t['profit']:+.2f}", style=style)
            )
    layout["footer"].update(Panel(hist, box=box.ROUNDED))

    return layout

# ────────────────────────────────────────────────
# WEBSOCKET HANDLERS
# ────────────────────────────────────────────────
def on_open(ws_obj):
    ws_obj.send(json.dumps({"authorize": AUTH_TOKEN}))

def on_message(ws_obj, message):
    global total_profit, current_stake, open_trades, consecutive_losses
    try:
        data = json.loads(message)
    except:
        return
    if "error" in data:
        return

    mt = data.get("msg_type")

    if mt == "authorize":
        for sym in SYMBOLS:
            ws_obj.send(json.dumps({"ticks": sym, "subscribe": 1}))
            ws_obj.send(json.dumps({
                "ticks_history": sym, "count": 200, "end": "latest", "style": "ticks"
            }))

    elif mt == "tick":
        sym = data["tick"]["symbol"]
        price = float(data["tick"]["quote"])
        ts = int(data["tick"]["epoch"])
        with lock:
            if sym in last_prices:
                price_change[sym] = price - last_prices[sym]
            last_prices[sym] = price
            price_history[sym].append(price)
            update_1m_candles(sym, price, ts)

    elif mt == "history":
        sym = data["history"]["symbol"]
        for p, t in zip(data["history"]["prices"], data["history"]["times"]):
            update_1m_candles(sym, float(p), int(t))

    elif mt == "proposal":
        pid = data["proposal"]["id"]
        ask = data["proposal"]["ask_price"]
        ws_obj.send(json.dumps({"buy": pid, "price": ask}))

    elif mt == "proposal_open_contract":
        c = data["proposal_open_contract"]
        if c.get("is_sold") == 1:
            profit = float(c.get("profit", 0))
            sym = c.get("symbol", "Unknown")
            ctype = c.get("contract_type", "Unknown")

            total_profit += profit
            open_trades = max(0, open_trades - 1)

            trade_history.append({
                'time': datetime.now().strftime("%H:%M:%S"),
                'symbol': SYMBOL_NAMES.get(sym, sym),
                'type': ctype,
                'profit': profit
            })

            with lock:
                if profit > 0:
                    consecutive_losses = 0
                    current_stake = INITIAL_STAKE
                else:
                    consecutive_losses += 1
                    current_stake *= MARTINGALE_MULTIPLIER

                if total_profit >= TAKE_PROFIT or consecutive_losses >= MAX_LOSSES:
                    global running
                    running = False

def on_close(ws_obj, *args):
    global running
    running = False

# ────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────
def main():
    global ws, running, start_time
    start_time = time.time()

    os.system('cls' if os.name == 'nt' else 'clear')
    console.print("[bold cyan]Starting 1-TICK BOT – Signal from 1-minute candles only[/bold cyan]")

    websocket.enableTrace(False)
    ws = websocket.WebSocketApp(
        WS_URL,
        on_open=on_open,
        on_message=on_message,
        on_close=on_close
    )
    threading.Thread(target=ws.run_forever, daemon=True).start()

    with Live(generate_ui(), refresh_per_second=4, screen=True) as live:
        last_check = time.time()
        try:
            while running:
                now = time.time()
                if now - last_check >= 5:
                    check_and_trade()
                    last_check = now
                live.update(generate_ui())
                time.sleep(0.2)
        except KeyboardInterrupt:
            running = False
        if ws:
            ws.close()
        console.print("\n[bold red]Stopped[/bold red]")
        console.print(f"[yellow]Final P&L: ${total_profit:+.2f}[/yellow]")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s,f: globals().update(running=False))
    main()