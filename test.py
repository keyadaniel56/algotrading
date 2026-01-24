import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Your Deriv credentials (replace if needed)
API_TOKEN = 'y5XlAyZZDrPz764'  # From Deriv dashboard
APP_ID = 1089  # Your registered app_id
SYMBOL = 'R_100'  # Synthetic volatility index
WS_URL = f'wss://ws.derivws.com/websockets/v3?app_id={APP_ID}'
STAKE = 1  # USD per trade
DURATION = 5  # Minutes for contract
CONTRACT_TYPE_BUY = 'CALL'  # For buy signal (expect rise)
CONTRACT_TYPE_SELL = 'PUT'  # For sell signal (expect fall)
TIMEFRAME = '1min'  # 1-minute bars
GRANULARITY = 60  # Seconds for 1-min candles in historical fetch

async def deriv_api_call(ws, payload):
    await ws.send(json.dumps(payload))
    response = await ws.recv()
    return json.loads(response)

async def get_historical_data(ws, symbol, start_time, end_time):
    payload = {
        "ticks_history": symbol,
        "start": int(start_time.timestamp()),
        "end": "latest",
        "style": "candles",
        "granularity": GRANULARITY  # Required for candles
    }
    resp = await deriv_api_call(ws, payload)
    if 'candles' in resp:
        df = pd.DataFrame(resp['candles'])
        df['timestamp'] = pd.to_datetime(df['epoch'], unit='s')
        df.set_index('timestamp', inplace=True)
        return df[['open', 'high', 'low', 'close']]
    else:
        print("Error fetching history:", resp)
        return pd.DataFrame()

def compute_indicators(df):
    df['MA50'] = df['close'].rolling(50).mean()
    df['MA14'] = df['close'].rolling(14).mean()
    df['BB_mid'] = df['close'].rolling(20).mean()
    df['BB_std'] = df['close'].rolling(20).std()
    df['BB_upper'] = df['BB_mid'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_mid'] - 2 * df['BB_std']
    
    low_5 = df['low'].rolling(5).min()
    high_5 = df['high'].rolling(5).max()
    df['%K'] = (df['close'] - low_5) / (high_5 - low_5) * 100
    df['%D'] = df['%K'].rolling(3).mean()
    df['Slow%D'] = df['%D'].rolling(3).mean()
    
    df['MA10'] = df['close'].rolling(10).mean()
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

def generate_signals(df):
    df['buy_signal'] = np.where(
        (df['MA14'] > df['MA50']) &
        (df['close'] < df['BB_mid']) &
        (df['%K'].shift(1) <= df['%D'].shift(1)) &
        (df['%K'] > df['%D']) &
        (df['%D'].shift(1) < 20),
        1, 0
    )
    df['sell_signal'] = np.where(
        (df['close'] > df['BB_upper']) |
        (df['MA14'] < df['MA50']) |
        ((df['%K'].shift(1) >= df['%D'].shift(1)) &
         (df['%K'] < df['%D']) &
         (df['%D'].shift(1) > 80)) |
        ((df['MACD'].shift(1) >= df['MACD_signal'].shift(1)) &
         (df['MACD'] < df['MACD_signal'])),
        1, 0
    )
    return df

async def place_trade(ws, contract_type):
    payload = {
        "buy": "1",
        "price": STAKE,
        "subscribe": 1,
        "parameters": {
            "amount": STAKE,
            "basis": "stake",
            "contract_type": contract_type,
            "currency": "USD",
            "duration": DURATION,
            "duration_unit": "m",
            "symbol": SYMBOL
        }
    }
    resp = await deriv_api_call(ws, payload)
    print(f"Trade Placed ({contract_type}):", resp)

async def main():
    async with websockets.connect(WS_URL) as ws:
        # Authorize
        auth_payload = {"authorize": API_TOKEN}
        auth_resp = await deriv_api_call(ws, auth_payload)
        if 'error' in auth_resp:
            print("Auth Error:", auth_resp['error'])
            return
        print("Authorized successfully.")

        # Load historical bars (last 7 days to avoid excessive data; API caps ~5000 bars)
        start_time = datetime.now() - timedelta(days=7)
        end_time = datetime.now()
        bar_df = await get_historical_data(ws, SYMBOL, start_time, end_time)
        if bar_df.empty:
            print("No historical data; starting fresh.")
            bar_df = pd.DataFrame(columns=['open', 'high', 'low', 'close'])
            bar_df.index = pd.DatetimeIndex([])
        else:
            print(f"Loaded {len(bar_df)} historical bars.")
        bar_df = bar_df[~bar_df.index.duplicated(keep='last')]  # Handle any dupes
        bar_df = compute_indicators(bar_df)
        bar_df = generate_signals(bar_df)

        # Subscribe to ticks
        await deriv_api_call(ws, {"ticks": SYMBOL, "subscribe": 1})

        # Tick accumulation DF
        tick_df = pd.DataFrame({'quote': pd.Series(dtype='float64')})
        tick_df.index = pd.DatetimeIndex([], name='timestamp')
        last_tick_ts = None
        last_trade_time = None  # To avoid multiple trades per bar

        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            if 'tick' not in data:
                continue
            tick = data['tick']
            ts = pd.to_datetime(tick['epoch'], unit='s')
            new_tick = pd.DataFrame({'quote': [tick['quote']]}, index=[ts])
            print(f"Received tick: {ts} - {tick['quote']}")  # Log for debugging
            tick_df = pd.concat([tick_df, new_tick])

            if last_tick_ts is not None:
                last_minute = last_tick_ts.replace(second=0, microsecond=0)
                current_minute = ts.replace(second=0, microsecond=0)
                if current_minute > last_minute:
                    # New minute started; aggregate the previous complete bar
                    prev_bar_start = last_minute
                    prev_bar_end = current_minute
                    prev_ticks = tick_df[(tick_df.index >= prev_bar_start) & (tick_df.index < prev_bar_end)]
                    if not prev_ticks.empty:
                        new_bar = pd.DataFrame({
                            'open': [prev_ticks['quote'].iloc[0]],
                            'high': [prev_ticks['quote'].max()],
                            'low': [prev_ticks['quote'].min()],
                            'close': [prev_ticks['quote'].iloc[-1]]
                        }, index=[prev_bar_start])
                        bar_df = pd.concat([bar_df, new_bar])
                        bar_df = bar_df[~bar_df.index.duplicated(keep='last')]
                        bar_df = compute_indicators(bar_df)
                        bar_df = generate_signals(bar_df)

                        # Debugging prints
                        latest = bar_df.iloc[-1]
                        print(f"\nAdded new bar at {prev_bar_start}:")
                        print(new_bar)
                        print("Key indicators:")
                        print(latest[['MA14', 'MA50', 'close', 'BB_mid', '%K', '%D', 'MACD', 'MACD_signal']])
                        print(f"Previous %D: {bar_df['%D'].iloc[-2] if len(bar_df) > 1 else 'N/A'}")
                        print(f"Signals: Buy={latest['buy_signal']}, Sell={latest['sell_signal']}\n")

                        # Check signals and place trade (bi-directional, once per bar)
                        now = datetime.now()
                        if last_trade_time is None or (now - last_trade_time) > timedelta(minutes=1):
                            if latest['buy_signal'] == 1:
                                await place_trade(ws, CONTRACT_TYPE_BUY)
                                last_trade_time = now
                            elif latest['sell_signal'] == 1:
                                await place_trade(ws, CONTRACT_TYPE_SELL)
                                last_trade_time = now
                        else:
                            print("Skipped trade: Too soon after last trade.")

            last_tick_ts = ts

            # Trim old ticks to save memory (keep last 1000)
            tick_df = tick_df.tail(1000)

            # Sleep briefly to avoid CPU spin
            await asyncio.sleep(1)

asyncio.run(main())