import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from textblob import TextBlob
from datetime import datetime, timedelta
import plotly.graph_objects as go
import urllib.request
import xml.etree.ElementTree as ET
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Configure clean, wide institutional terminal theme
st.set_page_config(page_title="AuraQuant AI Terminal Core", layout="wide")

# ==========================================
# INITIALIZE LIVE SESSION STATE LEDGERS
# ==========================================
if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = {}
if 'historical_data' not in st.session_state:
    st.session_state['historical_data'] = None
if 'current_prices' not in st.session_state:
    st.session_state['current_prices'] = {}
if 'ledger_df' not in st.session_state:
    st.session_state['ledger_df'] = None

# ==========================================
# REAL-TIME LIVE RSS NEWS PARSER
# ==========================================
def fetch_live_sentiment(ticker):
    """Fetches real-time RSS news headlines from Yahoo Finance and calculates true sentiment polarity."""
    # Clean ticker name for standard financial RSS parsing formats
    clean_ticker = ticker.split(".")[0] if "." in ticker else ticker
    rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={clean_ticker}&region=US&lang=en-US"
    
    try:
        req = urllib.request.Request(rss_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            xml_data = response.read()
        
        root = ET.fromstring(xml_data)
        headlines = []
        for item in root.findall('.//item'):
            title = item.find('title')
            if title is not None and title.text:
                headlines.append(title.text)
        
        if not headlines:
            return "Neutral (No Active Feed Flow)", 0.0
            
        # Run text processing analytics across raw parsed strings
        total_polarity = 0.0
        for text in headlines[:5]:  # Process the top 5 freshest breaking alerts
            total_polarity += TextBlob(text).sentiment.polarity
            
        avg_polarity = total_polarity / min(len(headlines), 5)
        
        if avg_polarity > 0.05:
            return f"🟢 Bullish Alpha ({avg_polarity:+.2f})", avg_polarity
        elif avg_polarity < -0.05:
            return f"🔴 Bearish Risk ({avg_polarity:.2f})", avg_polarity
        return f"⚪ Neutral ({avg_polarity:+.2f})", avg_polarity
        
    except Exception as e:
        return "⚪ Neutral (Feed Timeout)", 0.0

# ==========================================
# ADVANCED MACHINE LEARNING ENGINE
# ==========================================
def train_predictive_ml_engine(historical_series, forward_horizon_days):
    """Trains a Ridge Regressor on historical technical features and forecasts out across a target window."""
    df = pd.DataFrame({'Price': historical_series.astype(float)})
    
    # Feature Engineering Layer
    df['Lag_1'] = df['Price'].shift(1)
    df['Lag_3'] = df['Price'].shift(3)
    df['Lag_5'] = df['Price'].shift(5)
    df['Rolling_Return'] = df['Price'].pct_change(5)
    df['Rolling_Volatility'] = df['Price'].rolling(10).std()
    
    df = df.dropna()
    if len(df) < 15:  # Fallback logic if the selected view window is too tiny
        return np.linspace(historical_series[-1], historical_series[-1] * 1.02, forward_horizon_days), np.std(historical_series)
        
    feature_cols = ['Lag_1', 'Lag_3', 'Lag_5', 'Rolling_Return', 'Rolling_Volatility']
    X = df[feature_cols].values
    y = df['Price'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train regularization ridge model
    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)
    
    # Iterative Walk-Forward Autoregressive Projection Core
    predictions = []
    last_known_features = df.iloc[-1].copy()
    current_price_pointer = float(last_known_features['Price'])
    
    # Collect tracking slices to maintain dynamic lag states
    recent_prices = list(df['Price'].values[-5:])
    recent_returns = list(df['Rolling_Return'].values[-5:])
    recent_vols = list(df['Rolling_Volatility'].values[-10:])
    
    for _ in range(forward_horizon_days):
        input_vector = np.array([[
            recent_prices[-1],  # Lag 1
            recent_prices[-3],  # Lag 3
            recent_prices[-5],  # Lag 5
            recent_returns[-1], # Rolling Return
            np.mean(recent_vols) # Expected Volatility Target
        ]])
        
        scaled_vector = scaler.transform(input_vector)
        next_pred = float(model.predict(scaled_vector)[0])
        predictions.append(next_pred)
        
        # Shift and update memory arrays recursively
        recent_prices.append(next_pred)
        recent_returns.append((next_pred - recent_prices[-2]) / recent_prices[-2] if recent_prices[-2] != 0 else 0)
        recent_vols.append(np.std(recent_prices[-5:]))
        
    residual_volatility = float(np.std(y - model.predict(X_scaled)))
    return np.array(predictions), residual_volatility

# ==========================================
# SYSTEM CORE CROSS-MARKET SCANNER
# ==========================================
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
            
            series_cleaned = raw_ticker['Close'].dropna() if 'Close' in raw_ticker.columns else raw_ticker.iloc[:, 0].dropna()
            flat_prices = series_cleaned.values.flatten().astype(float)
            date_strings = series_cleaned.index.strftime('%Y-%m-%d')
            
            if chart_clean_df.empty:
                chart_clean_df = pd.DataFrame(index=date_strings)
                
            chart_clean_df[ticker] = pd.Series(flat_prices, index=date_strings)
            
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
            
            # Integrated live news sentiment analytics pipeline
            sentiment_label, _ = fetch_live_sentiment(ticker)
            
            # Sizing calculation structures
            stop_loss = lower_band * 0.98  
            risk_per_share = max(current_price - stop_loss, current_price * 0.02) 
            cash_at_risk = capital_base * (max_risk_pct / 100.0)
            calculated_shares = int(cash_at_risk // risk_per_share)
            
            if (calculated_shares * current_price) > capital_base:
                calculated_shares = int(capital_base // current_price)

            if current_price < lower_band:
                action = "🟢 BUY (Oversold)"
                suggested_sizing = f"{calculated_shares:,} Shares"
            elif current_price > upper_band:
                action = "🔴 SELL (Overbought)"
                suggested_sizing = "0 (Liquidate)"
            else:
                action = "⚪ HOLD (Neutral)"
                suggested_sizing = "0"
                
            signal_ledger.append({
                "Asset Universe": ticker,
                "Live Value (₹)": round(current_price, 2),
                "Real-time News Sentiment": sentiment_label,
                "Support Band": round(lower_band, 2),
                "Resistance Band": round(upper_band, 2),
                "Algorithmic Action": action,
                "Risk Target Sizing": suggested_sizing
            })
        except Exception as e:
            pass
            
    st.session_state['current_prices'] = prices_map
    return pd.DataFrame(signal_ledger), chart_clean_df

# ==========================================
# TERMINAL USER INTERFACE VIEW LAYER
# ==========================================
st.title("🤖 AuraQuant AI Predictive Intelligence Terminal")
st.markdown("---")

st.sidebar.header("🕹️ Terminal Control Panel")
watchlist_input = st.sidebar.text_input("Asset Scan Universe", "RELIANCE.NS, TCS.NS, INFY.NS, SBIN.NS")
tickers = [t.strip() for t in watchlist_input.split(",")]

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ Risk Management Parameters")
investment_base = st.sidebar.number_input("Capital Investment Base (₹)", min_value=10000, value=1000000, step=50000)
risk_percentage = st.sidebar.slider("Max Account Risk Per Trade (%)", min_value=0.25, max_value=5.0, value=1.0, step=0.25)

st.sidebar.markdown("---")
st.sidebar.subheader("🔮 ML Autoregressive Tuner")
chart_lookback_days = st.sidebar.slider("Training Feature Window (Days)", min_value=15, max_value=180, value=60, step=5)
future_horizon_days = st.sidebar.slider("AI ML Forecast Horizon (Days)", min_value=7, max_value=90, value=30, step=1)

# ==========================================
# COMPONENT 1: CROSS MARKET ML ENGINE SCANNER
# ==========================================
st.header("🔍 Real-Time Multi-Asset Scanner & Machine Learning Intelligence")

if st.button("🔄 Execute Fresh Cross-Market Scan") or st.session_state['historical_data'] is not None:
    if st.session_state['historical_data'] is None or st.sidebar.button("Force Clear Engine Memory"):
        with st.spinner("Compiling structural feature matrix & connection loops..."):
            ledger_df, historical_data = run_production_scanner(tickers, investment_base, risk_percentage)
            st.session_state['historical_data'] = historical_data
            st.session_state['ledger_df'] = ledger_df
    else:
        ledger_df = st.session_state['ledger_df']

    st.subheader("📋 Core Intelligence Data Streams")
    st.table(ledger_df)

# ==========================================
# ORDER ROUTING DESK & LEDGER SIMULATION
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
        st.error("Please run the market scanner above to pull current equity valuations first.")
    else:
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
                st.error(f"Execution Aborted: Insufficient portfolio inventory for {trade_ticker}.")
            else:
                total_shares = current_holding["shares"] - trade_shares
                if total_shares == 0:
                    st.session_state['portfolio'].pop(trade_ticker, None)
                else:
                    st.session_state['portfolio'][trade_ticker]["shares"] = total_shares
                st.success(f"Execution Successful: Sold {trade_shares} of {trade_ticker} at ₹{current_market_price:,.2f}")

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
# ML PREDICTIVE PRICE INTELLIGENCE HORIZON
# ==========================================
if st.session_state['historical_data'] is not None:
    st.markdown("---")
    st.header("📈 ML Autoregressive Price Horizon Trends")
    
    selected_chart_asset = st.selectbox("Select Target Asset Timeline to Plot", tickers)
    chart_df = st.session_state['historical_data']
    
    if selected_chart_asset in chart_df.columns:
        clean_history = chart_df[[selected_chart_asset]].dropna()
        historical_prices = clean_history[selected_chart_asset].values[-chart_lookback_days:]
        historical_dates = pd.to_datetime(clean_history.index[-chart_lookback_days:])
        
        # Execute Scikit-Learn Predictive Routine
        with st.spinner("Re-training Ridge optimization matrices..."):
            future_predictions, model_error_std = train_predictive_ml_engine(
                clean_history[selected_chart_asset].values, 
                future_horizon_days
            )
        
        last_date = historical_dates[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, future_horizon_days + 1)]
        
        fig = go.Figure()
        
        # Trace 1: Ground Truth History
        fig.add_trace(go.Scatter(
            x=historical_dates, y=historical_prices,
            mode='lines', name='Historical Value',
            line=dict(color='#00bfff', width=2.5)
        ))
        
        # Trace 2: ML Auto-Regressive Projections
        pred_x = [historical_dates[-1]] + future_dates
        pred_y = [historical_prices[-1]] + list(future_predictions)
        fig.add_trace(go.Scatter(
            x=pred_x, y=pred_y,
            mode='lines', name='AI Machine Learning Forecast',
            line=dict(color='#ffaa00', width=2, dash='dash')
        ))
        
        # Trace 3 & 4: Volatility Error Boundaries
        upper_y = [historical_prices[-1]]
        lower_y = [historical_prices[-1]]
        for idx in range(future_horizon_days):
            time_factor = np.sqrt(idx + 1) * 0.45
            upper_y.append(future_predictions[idx] + (model_error_std * time_factor))
            lower_y.append(future_predictions[idx] - (model_error_std * time_factor))
            
        fig.add_trace(go.Scatter(
            x=pred_x, y=upper_y,
            mode='lines', name='Model High-Risk Ceiling',
            line=dict(color='#00ffcc', width=1, dash='dot')
        ))
        fig.add_trace(go.Scatter(
            x=pred_x, y=lower_y,
            mode='lines', name='Model Low-Risk Floor',
            line=dict(color='#ff3333', width=1, dash='dot')
        ))
        
        fig.update_layout(
            template='plotly_dark',
            margin=dict(l=40, r=40, t=20, b=40),
            height=500,
            hovermode='x unified',
            xaxis=dict(showgrid=True, gridcolor='#2b2b2b', tickformat='%b %d, %Y'),
            yaxis=dict(showgrid=True, gridcolor='#2b2b2b', title="Price Valuation (₹)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
