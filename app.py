import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from textblob import TextBlob

# Configure clean, wide institutional terminal theme
st.set_page_config(page_title="AuraQuant Trader Terminal", layout="wide")

# ==========================================
# CORE MATHEMATICAL & RISK ENGINES
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

def run_production_scanner(tickers, capital_base, max_risk_pct):
    signal_ledger = []
    
    # 1. Download raw ticker history cleanly
    raw_data = yf.download(tickers, start="2025-01-01", auto_adjust=True, progress=False)
    
    # 2. Force build an completely flat DataFrame from scratch for charting
    chart_clean_df = pd.DataFrame(index=raw_data.index)
    
    for ticker in tickers:
        try:
            # Safely locate the Close column regardless of yfinance formatting quirks
            if isinstance(raw_data.columns, pd.MultiIndex):
                if ('Close', ticker) in raw_data.columns:
                    series = raw_data[('Close', ticker)]
                elif (ticker, 'Close') in raw_data.columns:
                    series = raw_data[(ticker, 'Close')]
                else:
                    # Fallback to single level lookup if structure shifted
                    series = raw_data[ticker] if ticker in raw_data.columns else None
            else:
                series = raw_data[ticker] if ticker in raw_data.columns else None

            if series is None or series.empty:
                continue
                
            # Flatten to 1D series, drop nulls, force explicit floats
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
                
            series_cleaned = series.dropna().astype(float)
            
            # Map clean isolated data column directly into chart dataframe
            chart_clean_df[ticker] = series_cleaned
            
            # Generate Bollinger Band analysis array
            df_ticker = pd.DataFrame({'Price': series_cleaned})
            df_ticker['MA'] = df_ticker['Price'].rolling(20).mean()
            df_ticker['STD'] = df_ticker['Price'].rolling(20).std()
            df_ticker['Upper'] = df_ticker['MA'] + (df_ticker['STD'] * 2)
            df_ticker['Lower'] = df_ticker['MA'] - (df_ticker['STD'] * 2)
            
            latest = df_ticker.iloc[-1]
            current_price = latest['Price']
            lower_band = latest['Lower']
            upper_band = latest['Upper']
            
            # Risk Sizing Sizing Calculations
            stop_loss = lower_band * 0.98  
            risk_per_share = max(current_price - stop_loss, current_price * 0.02) 
            cash_at_risk = capital_base * (max_risk_pct / 100.0)
            calculated_shares = int(cash_at_risk // risk_per_share)
            
            if (calculated_shares * current_price) > capital_base:
                calculated_shares = int(capital_base // current_price)

            if current_price < lower_band:
                action = "🟢 BUY (Oversold Entry)"
                suggested_sizing = f"{calculated_shares:,} Shares"
            elif current_price > upper_band:
                action = "🔴 SELL (Overbought Exit)"
                suggested_sizing = "0 (Liquidate Position)"
            else:
                action = "⚪ HOLD (No Signal)"
                suggested_sizing = "0"
                
            signal_ledger.append({
                "Asset": ticker,
                "Current Price": round(current_price, 2),
                "Sentiment Alpha": get_sentiment_score(ticker),
                "Lower Band (Buy Level)": round(lower_band, 2),
                "Upper Band (Sell Level)": round(upper_band, 2),
                "Signal Matrix": action,
                "Recommended Sizing": suggested_sizing
            })
        except:
            pass
            
    return pd.DataFrame(signal_ledger), chart_clean_df

# ==========================================
# MAIN TERMINAL HEADER
# ==========================================
st.title("📊 AuraQuant Institutional Bloomberg-Clone Terminal")
st.markdown("---")

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("🕹️ Terminal Control Panel")
watchlist_input = st.sidebar.text_input("Asset Scan Universe", "RELIANCE.NS, TCS.NS, INFY.NS, SBIN.NS")
tickers = [t.strip() for t in watchlist_input.split(",")]

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ Risk Management Parameters")
investment_base = st.sidebar.number_input("Capital Investment Base (₹)", min_value=10000, value=1000000, step=50000)
risk_percentage = st.sidebar.slider("Max Account Risk Per Trade (%)", min_value=0.25, max_value=5.0, value=1.0, step=0.25)

total_risk_exposure = investment_base * (risk_percentage / 100.0)
st.sidebar.info(f"💼 Risk Threshold per Asset: ₹{total_risk_exposure:,.2f}")

# ==========================================
# CORE COMPONENT 1: LIVE ENGINE SCANNER
# ==========================================
st.header("🔍 Real-Time Multi-Asset Scanner & Alpha Intelligence")

if st.button("🔄 Execute Fresh Cross-Market Scan"):
    with st.spinner("Processing multi-asset risk bands & calculation models..."):
        ledger_df, historical_data = run_production_scanner(tickers, investment_base, risk_percentage)
        
        st.session_state['historical_data'] = historical_data
        
        actionable = ledger_df[ledger_df['Signal Matrix'] != "⚪ HOLD (No Signal)"]
        
        if not actionable.empty:
            st.warning("⚠️ CRITICAL REVERSION SIGNALS DETECTED - RISK ALLOCATION ACTIVE")
            st.table(actionable[["Asset", "Current Price", "Signal Matrix", "Recommended Sizing"]])
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
        w = mock_weights.get(t, 25.0) 
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

# ==========================================
# CORE COMPONENT 3: HISTORICAL INTERACTIVE VISUALIZATION
# ==========================================
if 'historical_data' in st.session_state and st.session_state['historical_data'] is not None:
    st.markdown("---")
    st.header("📈 Institutional Multi-Asset Price Intelligence Trends")
    
    selected_chart_asset = st.selectbox("Select Target Asset Timeline to Plot", tickers)
    
    chart_df = st.session_state['historical_data']
    
    if selected_chart_asset in chart_df.columns:
        # Isolate the exact selected column as a simple single-level clean series
        target_series = chart_df[selected_chart_asset].dropna()
        
        # Plot with native Streamlit engine
        st.line_chart(target_series)
    else:
        st.info("Execute a fresh cross-market scan above to generate visual historical charts.")
