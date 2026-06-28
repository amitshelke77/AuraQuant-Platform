import yfinance as yf
import streamlit as st

@st.cache_data(ttl=86400)
def fetch_corporate_fundamentals(tickers):
    """
    Extracts fundamental valuation and balance sheet vectors along with 
    sector classifications for relative normalization.
    """
    fundamental_matrix = {}
    
    for ticker in tickers:
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            
            # Extract raw metrics with safe fallbacks
            fundamental_matrix[ticker] = {
                "Trailing_PE": float(info.get("trailingPE", 0.0)),
                "PEG_Ratio": float(info.get("pegRatio", 0.0)),
                "Profit_Margin": float(info.get("profitMargins", 0.0)),
                "Quarterly_Revenue_Growth": float(info.get("revenueGrowth", 0.0)),
                "Sector": info.get("sector", "Diversified") # Captures strict sector definitions
            }
        except Exception:
            fundamental_matrix[ticker] = {
                "Trailing_PE": 20.0, 
                "PEG_Ratio": 1.0, 
                "Profit_Margin": 0.10,
                "Quarterly_Revenue_Growth": 0.05, 
                "Sector": "Diversified"
            }
            
    return fundamental_matrix
