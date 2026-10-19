import pandas as pd
import numpy as np

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Moving Averages, Volatility, Returns, RSI, and MACD.
    """
    df = df.copy()

    # Daily Returns & Volatility (20-day rolling std)
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility_20'] = df['Daily_Return'].rolling(window=20).std()

    # Simple Moving Averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # Exponential Moving Averages
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # MACD & Signal Line
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Relative Strength Index (RSI - 14 Days)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # Target Variable: Next Day Price Direction (1 = Up, 0 = Down)
    df['Target_Dir'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Target Variable: Next Day Close Price (for regression tasks)
    df['Target_Price'] = df['Close'].shift(-1)

    return df.dropna()