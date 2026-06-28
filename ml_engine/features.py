import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def compile_ensemble_features(tickers, tech_data, fund_data, live_news_stream):
    """Combines all dimensions and dynamically appends sector-relative relative baselines."""
    master_records = []
    valid_tickers = []
    
    for ticker in tickers:
        if ticker not in tech_data or ticker not in fund_data:
            continue
            
        sentiment_score = live_news_stream.get(ticker, 0.0)
        
        record = {
            "RSI": tech_data[ticker]["RSI"],
            "MACD_Hist": tech_data[ticker]["MACD_Hist"],
            "Trend_Ratio": tech_data[ticker]["Trend_50_200_Ratio"],
            "Volatility": tech_data[ticker]["Rolling_Volatility"],
            "PE_Ratio": fund_data[ticker]["Trailing_PE"],
            "PEG_Ratio": fund_data[ticker]["PEG_Ratio"],
            "Revenue_Growth": fund_data[ticker]["Quarterly_Revenue_Growth"],
            "Sentiment": sentiment_score,
            "Sector": fund_data[ticker]["Sector"] # Track raw sector text
        }
        master_records.append(record)
        valid_tickers.append(ticker)
        
    if not master_records:
        return pd.DataFrame(), [], None
        
    df_features = pd.DataFrame(master_records, index=valid_tickers)
    
    # DYNAMIC SECTOR BASELINING: Group by sector and pull target median thresholds
    sector_medians = df_features.groupby("Sector")["PE_Ratio"].transform("median")
    
    # Calculate a relative multiplier score (e.g., 1.2 means 20% more expensive than its peers)
    df_features["Sector_Relative_PE"] = df_features["PE_Ratio"] / (sector_medians + 1e-9)
    
    # Drop raw sector strings before standard scaling for the Ridge Regressor matrix
    numeric_df = df_features.drop(columns=["Sector"])
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(numeric_df)
    
    return df_features, valid_tickers, scaled_matrix
