import pandas as pd
import numpy as np

def initialize_dataset(json_data):
    df = pd.DataFrame(json_data)
    # Ensure columns are numeric and fill missing prices
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce').ffill()
    return df

def add_RSI(df, interval=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=interval).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=interval).mean()
    
    # Avoid division by zero
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Normalize to 0-1 range. 
    # Don't fill with 0.5 yet; let dropna handle the empty rows at the start.
    df['RSI'] = df['RSI'] / 100.0
    return df

def add_MA(df, interval=20):
    df[f'MA'] = df['Close'].rolling(window=interval).mean()
    df[f'MA_Dist'] = (df['Close'] - df[f'MA']) / df[f'MA']
    return df

def add_volatility(df):
    # Log High/Low is excellent. 
    # We add a small epsilon to avoid log(0) if high == low
    df['volatility'] = np.log(df['High'] / df['Low'].replace(0, 0.0001))
    return df

def add_rolling_volatility(df, interval=5):
    """
    Calculates the Standard Deviation of Log Returns.
    This tells the AI: 'Is the market getting crazier or calmer lately?'
    """
    # Standard deviation of price changes over the last X periods
    df[f'Vol_Std_{interval}'] = df['Log_Ret_Close'].rolling(window=interval).std()
    return df

def add_VWAP(df, interval=14):
    tp = (df['High'] + df['Low'] + df['Close']) / 3.0
    tp_vol = tp * df['Volume']
    df['VWAP'] = tp_vol.rolling(window=interval).sum() / df['Volume'].rolling(window=interval).sum()
    df['VWAP_Dist'] = (df['Close'] - df['VWAP']) / df['VWAP']
    return df

def add_Log_Return_Close(df):
    df['close_log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    return df

def add_Log_Return_Volume(df):
    # Replace 0 volume with 1 to avoid log(0)
    v = df['Volume'].replace(0, 1)
    df['volume_log_return'] = np.log(v / v.shift(1))
    return df

def drop_close(df):
    return df.drop(columns=['Close'], errors='ignore')

def drop_high(df):
    return df.drop(columns=['High'], errors='ignore')

def drop_low(df):
    return df.drop(columns=['Low'], errors='ignore')

def drop_open(df):
    return df.drop(columns=['Open'], errors='ignore')
    
def drop_volume(df):
    return df.drop(columns=['Volume'], errors='ignore')

def add_targets(df, horizon=5, buy_threshold=0.03, sell_threshold=-0.02):
    future_price = df['Close'].shift(-horizon)
    target_return = (future_price - df['Close']) / df['Close']
    
    df['Target'] = 1 # Hold
    df.loc[target_return >= buy_threshold, 'Target'] = 2 # Buy
    df.loc[target_return <= sell_threshold, 'Target'] = 0 # Sell
    
    # FINAL CLEANUP: This removes the 'warm-up' NaNs from RSI/MA 
    # AND the 'future' NaNs from the Target shift.
    return df.dropna().reset_index(drop=True)