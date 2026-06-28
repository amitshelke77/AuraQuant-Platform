import pandas as pd
import numpy as np

def calculate_advanced_technicals(df_close, df_volume):
    """Calculates multi-dimensional technical arrays to prevent mean-reversion traps."""
    tech_data = {}
    
    for ticker in df_close.columns:
        prices = df_close[ticker].dropna()
        volumes = df_volume[ticker].dropna() if ticker in df_volume.columns else None
        
        if len(prices) < 20:
            continue
            
        # 1. RSI (Relative Strength Index)
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        
        # 2. MACD (Moving Average Convergence Divergence)
        ema12 = prices.ewm(span=12, adjust=False).mean()
        ema26 = prices.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - signal_line
        
        # 3. Structural Trend Alignment (Exponential Moving Averages)
        ema50 = prices.ewm(span=50, adjust=False).mean()
        ema200 = prices.ewm(span=200, adjust=False).mean()
        
        # 4. Volatility Filtering (True Range approximation)
        rolling_vol = prices.pct_change().rolling(20).std()
        
        latest_idx = prices.index[-1]
        tech_data[ticker] = {
            "RSI": float(rsi.loc[latest_idx]),
            "MACD_Hist": float(macd_hist.loc[latest_idx]),
            "Trend_50_200_Ratio": float(ema50.loc[latest_idx] / (ema200.loc[latest_idx] + 1e-9)),
            "Rolling_Volatility": float(rolling_vol.loc[latest_idx]),
            "Price_To_EMA50": float(prices.loc[latest_idx] / (ema50.loc[latest_idx] + 1e-9))
        }
        
    return tech_data
