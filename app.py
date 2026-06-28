import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from textblob import TextBlob
from datetime import datetime, timedelta

# Configure clean, wide institutional terminal theme
st.set_page_config(page_title="AuraQuant Trader Terminal", layout="wide")

# ==========================================
# INITIALIZE LIVE SESSION STATE LEDGERS
# ==========================================
if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = {}  # Format: { ticker: {"shares": X, "avg_price": Y} }
if 'trade_history' not in st.session_state:
    st.session_state['trade_history'] = []
if 'historical_data' not in st.session_state:
    st.session_state['historical_data'] = None
if 'current_prices' not in st.session_state:
    st.session_state['current_prices'] = {}

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
    chart_clean_df = pd.DataFrame()
    prices_map = {}
    
    for ticker in tickers:
        try:
            raw_ticker = yf.download(ticker, start="2025-01-01", progress=False)
            if raw_ticker.empty:
                continue
                
            if isinstance(raw_ticker.columns, pd.MultiIndex):
                raw_ticker.columns = raw_ticker.columns.get_level_values(-1)
            
            if 'Close' in raw_ticker.columns:
                series_cleaned = raw_ticker['Close'].dropna()
            else:
                series_cleaned = raw_ticker.iloc[:, 0].dropna()
            
            flat_prices = series_cleaned.values.flatten().astype(float)
            date_strings = series_cleaned.index.strftime('%Y-%m-%d')
            
            if chart_clean_df.empty:
                chart_clean_df = pd.DataFrame(index=date_strings)
                
            chart_clean_df[ticker] = pd.Series(flat_prices, index=date_strings)
            
            # Generate Bollinger Band metrics
            df_ticker = pd.DataFrame({'Price': flat_prices}, index=series_cleaned.index)
            df_ticker['MA'] = df_ticker['Price'].rolling(20).mean()
            df_ticker['STD'] = df_ticker['Price'].rolling(20).std()
            df_ticker['Upper'] = df_ticker['MA'] + (df_ticker['STD'] * 2)
            df_ticker['Lower'] = df_ticker['MA'] - (df_ticker['STD'] * 2)
            
            latest = df_ticker.iloc[-1]
            current_price = float(latest['Price'])
            lower_band = float(latest['Lower'])
            upper_band = float(latest['Upper'])
            
            prices_map[ticker] = current_price
            
            # Risk Sizing Math
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
            
    st.session_state['current_prices'] = prices_map
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

st.sidebar.markdown("---")
st.sidebar.subheader("🔮 Predictive Chart Tuner")
chart_lookback_days = st.sidebar.slider("Historical View Window (Days)", min_value=15, max_value=180, value=60, step=5)
future_horizon_days = st.sidebar.slider("Forecast Target Window (Days)", min_value=7, max_value=90, value=30, step=1)

# ==========================================
# CORE COMPONENT 1: LIVE ENGINE SCANNER
# ==========================================
st.header("🔍 Real-Time Multi-Asset Scanner & Alpha Intelligence")

if st.button("🔄 Execute Fresh Cross-Market Scan") or st.session_state['historical_data'] is not None:
    # Only pull fresh details if explicitly clicked or data hasn't been compiled yet
    if st.session_state['historical_data'] is None or st.sidebar.button("Force Re-download"):
        with st.spinner("Processing multi-asset risk bands & calculation models..."):
            ledger_df, historical_data = run_production_scanner(tickers, investment_base, risk_percentage)
            st.session_state['historical_data'] = historical_data
            st.session_state['ledger_df'] = ledger_df
    else:
        ledger_df = st.session_state['ledger_df']

    actionable = ledger_df[ledger_df['Signal Matrix'] != "⚪ HOLD (No Signal)"]
    if not actionable.empty:
        st.warning("⚠️ CRITICAL REVERSION SIGNALS DETECTED - RISK ALLOCATION ACTIVE")
        st.table(actionable[["Asset", "Current Price", "Signal Matrix", "Recommended Sizing"]])
        
    st.subheader("📋 Complete Asset Universe Matrix")
    st.table(ledger_df)

# ==========================================
# NEW FEATURE: SIMULATED ORDER EXECUTION DESK
# ==========================================
st.markdown("---")
st.header("⚡ Live Institutional Order Execution Panel")
order_col1, order_col2, order_col3 = st.columns(3)

with order_col1:
    trade_ticker = st.selectbox("Select Target Asset to Trade", tickers)
with order_col2:
    trade_action = st.radio("Order Direction", ["BUY", "SELL"], horizontal=True)
with order_col3:
    trade_shares = st.number_input("Order Quantity (Shares)", min_value=1, value=10, step=1)

if st.button("🚀 Transmit Order Payload"):
    current_market_price = st.session_state['current_prices'].get(trade_ticker, None)
    
    if current_market_price is None:
        st.error("Please execute a cross-market scan above to fetch current asset prices first.")
    else:
        # Business logic rules for trade processing
        if trade_action == "BUY":
            current_holding = st.session_state['portfolio'].get(trade_ticker, {"shares": 0, "avg_price": 0.0})
            total_shares = current_holding["shares"] + trade_shares
            total_cost = (current_holding["shares"] * current_holding["avg_price"]) + (trade_shares * current_market_price)
            avg_entry = total_cost / total_shares if total_shares > 0 else 0.0
            
            st.session_state['portfolio'][trade_ticker] = {"shares": total_shares, "avg_price": avg_entry}
            st.success(f"Execution Successful: Bought {trade_shares} of {trade_ticker} at ₹{current_market_price:,.2f}")
            
        elif trade_action == "SELL":
            current_holding = st.session_state['portfolio'].get(trade_ticker, {"shares": 0, "avg_price": 0.0})
            if current_holding["shares"] < trade_shares:
                st.error(f"Insufficient Inventory. You only hold {current_holding['shares']} shares of {trade_ticker}.")
            else:
                total_shares = current_holding["shares"] - trade_shares
                if total_shares == 0:
                    st.session_state['portfolio'].pop(trade_ticker, None)
                else:
                    st.session_state['portfolio'][trade_ticker]["shares"] = total_shares
                st.success(f"Execution Successful: Sold {trade_shares} of {trade_ticker} at ₹{current_market_price:,.2f}")

# ==========================================
# NEW FEATURE: PORTFOLIO LEDGER DASHBOARD
# ==========================================
if st.session_state['portfolio']:
    st.subheader("💼 Active Simulated Portfolio Positions")
    positions_summary = []
    
    for ticker, info in st.session_state['portfolio'].items():
        current_p = st.session_state['current_prices'].get(ticker, info["avg_price"])
        total_cost_basis = info["shares"] * info["avg_price"]
        current_value = info["shares"] * current_p
        floating_pnl = current_value - total_cost_basis
        pnl_pct = (floating_pnl / total_cost_basis) * 100 if total_cost_basis > 0 else 0.0
        
        positions_summary.append({
            "Asset": ticker,
            "Shares Owned": f"{info['shares']:,}",
            "Avg Entry Price": f"₹{info['avg_price']:,.2f}",
            "Current Price": f"₹{current_p:,.2f}",
            "Total Cost Basis": f"₹{total_cost_basis:,.2f}",
            "Current Value": f"₹{current_value:,.2f}",
            "Floating P&L (₹)": f"₹{floating_pnl:,.2f}",
            "Return (%)": f"{pnl_pct:+.2f}%"
        })
    st.table(pd.DataFrame(positions_summary))

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
# CORE COMPONENT 3: HISTORICAL & FORECAST VISUALIZATION
# ==========================================
if st.session_state['historical_data'] is not None:
    st.markdown("---")
    st.header("📈 Predictive Multi-Asset Price Horizon Trends")
    
    selected_chart_asset = st.selectbox("Select Target Asset Timeline to Plot", tickers)
    chart_df = st.session_state['historical_data']
    
    if selected_chart_asset in chart_df.columns:
        clean_history = chart_df[[selected_chart_asset]].dropna()
        clean_history = clean_history.tail(chart_lookback_days)
        
        historical_prices = clean_history[selected_chart_asset].values
        historical_dates = pd.to_datetime(clean_history.index)
        
        # Linear Regression Math
        lookback = len(historical_prices)
        x_train = np.arange(lookback)
        A = np.vstack([x_train, np.ones(lookback)]).T
        m, c = np.linalg.lstsq(A, historical_prices, rcond=None)[0]
        
        last_date = historical_dates[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, future_horizon_days + 1)]
        
        future_x = np.arange(lookback, lookback + future_horizon_days)
        future_predictions = m * future_x + c
        
        volatility_std = np.std(historical_prices) if len(historical_prices) > 1 else 1.0
        
        all_dates = list(historical_dates) + future_dates
        all_date_strings = [d.strftime('%Y-%m-%d') for d in all_dates]
        
        plot_df = pd.DataFrame(index=all_date_strings)
        plot_df['Historical Price'] = pd.Series(historical_prices, index=[d.strftime('%Y-%m-%d') for d in historical_dates])
        
        forecast_array = np.full(len(all_dates), np.nan)
        forecast_array[len(historical_prices) - 1] = historical_prices[-1]
        forecast_array[len(historical_prices):] = future_predictions
        plot_df['Expected Path Prediction'] = forecast_array
        
        upper_corridor = np.full(len(all_dates), np.nan)
        lower_corridor = np.full(len(all_dates), np.nan)
        upper_corridor[len(historical_prices) - 1] = historical_prices[-1]
        lower_corridor[len(historical_prices) - 1] = historical_prices[-1]
        
        for idx in range(future_horizon_days):
            time_factor = np.sqrt(idx + 1) * 0.35
            upper_corridor[len(historical_prices) + idx] = future_predictions[idx] + (volatility_std * time_factor)
            lower_corridor[len(historical_prices) + idx] = future_predictions[idx] - (volatility_std * time_factor)
            
        plot_df['Target Ceiling Boundary'] = upper_corridor
        plot_df['Target Floor Boundary'] = lower_corridor
        
        plot_ready_df = plot_df.reset_index().rename(columns={'index': 'Date'})
        st.line_chart(data=plot_ready_df, x='Date', y=['Historical Price', 'Expected Path Prediction', 'Target Ceiling Boundary', 'Target Floor Boundary'])
