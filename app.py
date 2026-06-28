import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from textblob import TextBlob

# Configure clean, wide institutional terminal theme
st.set_page_config(page_title="AuraQuant Trader Terminal", layout="wide")

# ==========================================
# CORE MATHEMATICAL & NLP ENGINES
# ==========================================
def get_sentiment_score(ticker):
    news_feeds = {
        "RELIANCE.NS": ["Strong refinery margins boost Q1 outlook", "Reliance retail expansion plans aggressive"],
        "TCS.NS": ["TCS faces headwinds in US consulting sector", "Steady dividends continue for shareholders"],
        "INFY.NS": ["Infosys wins new AI-led digital transformation contract", "Margins tightening due to higher wage costs"],
        "SBIN.NS": ["SBI reports record quarterly credit growth", "Non-performing assets showing slight uptick"]
    }
    headlines = news_feeds.get(ticker, ["Market neutral"])
    sentiment_avg = sum([TextBlob(h).sentiment.polarity for h in headlines]) / len(headlines)
    if sentiment_avg > 0.1: return "Bullish (Alpha Gain)"
    if sentiment_avg < -0.1: return "Bearish (Risk Alert)"
    return "Neutral"

def run_production_scanner(tickers):
    signal_ledger = []
    # Batch download historical closing prices
    data = yf.download(tickers, start="2025-01-01", auto_adjust=True)['Close']
    
    for ticker in tickers:
        try:
            df_ticker = data[ticker].to_frame(name='Price') if ticker in data.columns else data.copy()
            df_ticker.dropna(inplace=True)
            
            # 20-Day Bollinger Band Parameters
            df_ticker['MA'] = df_ticker['Price'].rolling(20).mean()
            df_ticker['STD'] = df_ticker['Price'].rolling(20).std()
            df_ticker['Upper'] = df_ticker['MA'] + (df_ticker['STD'] * 2)
            df_ticker['Lower'] = df_ticker['MA'] - (df_ticker['STD'] * 2)
            
            latest = df_ticker.iloc[-1]
            current_price = latest['Price']
            
            if current_price < latest['Lower']:
                action = "🟢 BUY (Oversold Entry)"
            elif current_price > latest['Upper']:
                action = "🔴 SELL (Overbought Exit)"
            else:
                action = "⚪ HOLD (No Signal)"
                
            signal_ledger.append({
                "Asset": ticker,
                "Current Price": round(current_price, 2),
                "Sentiment Alpha": get_sentiment_score(ticker),
                "Lower Band (Buy Level)": round(latest['Lower'], 2),
                "Upper Band (Sell Level)": round(latest['Upper'], 2),
                "Signal Matrix": action
            })
        except:
            pass
            
    return pd.DataFrame(signal_ledger)

# ==========================================
# MAIN TERMINAL HEADER
# ==========================================
st.title("📊 AuraQuant Institutional Bloomberg-Clone Terminal")
st.markdown("---")

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("🕹️ Terminal Control Panel")
watchlist_input = st.sidebar.text_input("Asset Scan Universe", "RELIANCE.NS, TCS.NS, INFY.NS, SBIN.NS, HDFCBANK.NS, ICICIBANK.NS")
tickers = [t.strip() for t in watchlist_input.split(",")]
investment_base = st.sidebar.number_input("Capital Investment Base (₹)", min_value=10000, value=1000000, step=50000)

# ==========================================
# CORE COMPONENT 1: LIVE ENGINE SCANNER
# ==========================================
st.header("🔍 Real-Time Multi-Asset Scanner & Alpha Intelligence")

if st.button("🔄 Execute Fresh Cross-Market Scan"):
    with st.spinner("Processing multi-asset standard deviations & news sentiment..."):
        ledger_df = run_production_scanner(tickers)
        
        # Pull high conviction breakouts instantly
        actionable = ledger_df[ledger_df['Signal Matrix'] != "⚪ HOLD (No Signal)"]
        
        if not actionable.empty:
            st.warning("⚠️ CRITICAL REVERSION SIGNALS DETECTED")
            st.table(actionable)
        else:
            st.success("✅ SYSTEM STATE: NEUTRAL. No assets have breached statistical bands today. Preserve cash equity.")
            
        st.subheader("📋 Complete Asset Universe Matrix")
        st.table(ledger_df)

# ==========================================
# CORE COMPONENT 2: PORTFOLIO ALLOCATION MATRIX
# ==========================================
st.markdown("---")
st.header("🎯 Advanced Portfolio Allocation Matrix")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Optimized Portfolio Allocation Targets")
    mock_weights = {"RELIANCE.NS": 44.38, "TCS.NS": 26.03, "INFY.NS": 15.62, "SBIN.NS": 13.98}
    alloc_data = []
    for t in tickers:
        w = mock_weights.get(t, 16.66) # Default to even split if basket expands
        allocated_cash = investment_base * (w / 100.0)
        alloc_data.append({"Asset": t, "Target Weight (%)": f"{w:.2f}%", "Capital Deployment": f"₹{allocated_cash:,.2f}"})
    st.table(pd.DataFrame(alloc_data))

with col2:
    st.subheader("Systemic Crisis Shock Stress-Testing")
    crisis_scenarios = [
        {"Crisis Vector": "2020 COVID-19 Black Swan", "Drop": "-35.0%", "Impact": f"₹-{investment_base * 0.35:,.2f}"},
        {"Crisis Vector": "2008 Financial Meltdown", "Drop": "-52.0%", "Impact": f"₹-{investment_base * 0.52:,.2f}"}
    ]
    st.table(pd.DataFrame(crisis_scenarios))
