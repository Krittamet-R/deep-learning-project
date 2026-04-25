import pandas as pd
import numpy as np

def initialize_dataset(json_data):
    df = pd.DataFrame(json_data)
    df['Close'] = df['Close'].ffill()
    df['High'] = df['High'].ffill()
    df['Low'] = df['Low'].ffill()
    df['Open'] = df['Open'].ffill()
    df['Volume'] = df['Volume'].ffill()
    return df

def add_RSI(df, interval=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=interval).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=interval).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50.0) / 100 # Normalized 0-1
    return df

def add_MA(df, interval=20):
    df[f'MA'] = df['Close'].rolling(window=interval).mean()
    return df

def add_MA_Dist(df, interval=20):
    df[f'MA_Dist_{interval}'] = (df['Close'] - df[f'MA_{interval}']) / df[f'MA_{interval}']
    return df

def add_volatility(df):
    """
    Calculates the Intraday/Intraweek Range.
    This tells the AI: 'How big was the battle today?'
    """
    # math.log(High/Low) equivalent for Pandas
    df['volatility'] = np.log(df['High'] / df['Low'])
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
    df['volume_log_return'] = np.log(df['Volume'].replace(0, 1) / df['Volume'].shift(1).replace(0, 1))
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
    # 1. Look ahead: What is the price 'horizon' days from now?
    future_price = df['Close'].shift(-horizon)
    
    # 2. Calculate the return
    target_return = (future_price - df['Close']) / df['Close']
    
    # 3. Apply the Triple Barrier (0=Sell, 1=Hold, 2=Buy)
    df['Target'] = 1 # Default to Hold
    df.loc[target_return >= buy_threshold, 'Target'] = 2
    df.loc[target_return <= sell_threshold, 'Target'] = 0
    
    # IMPORTANT: Remove the last 'horizon' rows because they have no future data
    return df.dropna()