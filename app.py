import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import urllib.request
import xml.etree.ElementTree as ET
from textblob import TextBlob
from datetime import datetime, timedelta
import plotly.graph_objects as gr
from plotly.subplots import make_subplots

# Import AuraQuant Modular Structural Pipelines
from core.technicals import calculate_advanced_technicals
from core.fundamentals import fetch_corporate_fundamentals
from ml_engine.features import compile_ensemble_features
from ml_engine.model import generate_multi_factor_predictions

st.set_page_config(page_title="AuraQuant Multi-Factor Intelligence Terminal", layout="wide")

# ==========================================
# DYNAMIC NIFTY 500 UNIVERSE INGESTION
# ==========================================
@st.cache_data(ttl=86400)
def load_nifty_500_universe():
    """Scrapes the complete up-to-date Nifty 500 constituent ticker ledger from public indices."""
    try:
        url = "https://raw.githubusercontent.com/kprohith/nse-stock-analysis/master/ind_nifty500list.csv"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            df = pd.read_csv(response)
        return sorted([f"{sym.strip()}.NS" for sym in df['Symbol'].tolist() if pd.notna(sym)])
    except Exception:
        return ["RELIANCE.NS", "TCS.NS", "INFY.NS", "SBIN.NS", "GUJGASLTD.NS", "AUROPHARMA.NS"]

nifty_500_tickers = load_nifty_500_universe()

# ==========================================
# LIVE AUTOMATED SENTIMENT EXTRACTION ENGINE
# ==========================================
def compute_live_sentiment_vector(ticker):
    """Scrapes real-time financial headlines and computes numeric polarity scores [-1 to 1]."""
    clean_ticker = ticker.split(".")[0] if "." in ticker else ticker
    rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={clean_ticker}&region=US&lang=en-US"
    try:
        req = urllib.request.Request(rss_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=3) as response:
            root = ET.fromstring(response.read())
        
        polarities = []
        for item in root.findall('.//item')[:5]:
            title = item.find('title')
            if title is not None and title.text:
                polarities.append(TextBlob(title.text).sentiment.polarity)
        return float(np.mean(polarities)) if polarities else 0.0
    except Exception:
        return 0.0

# ==========================================
# TRACK INITIALIZED LEDGERS IN RAM STATE
# ==========================================
if 'master_records' not in st.session_state:
    st.session_state['master_records'] = None
if 'historical_raw_close' not in st.session_state:
    st.session_state['historical_raw_close'] = None
if 'historical_raw_volume' not in st.session_state:
    st.session_state['historical_raw_volume'] = None

# ==========================================
# STREAMLIT UI LAYER
# ==========================================
st.title("🤖 AuraQuant Multi-Factor Intelligence Terminal")
st.caption("Production Architecture: Sector-Relative Valuation + Volume Multipliers + Trend Structuring + Real-Time NLP Sentiment Ensemble Engine")
st.markdown("---")

st.sidebar.header("🕹️ Terminal Control Panel")
st.sidebar.success(f"📦 System Matrix: Loaded {len(nifty_500_tickers)} Nifty 500 Assets.")

tickers = st.sidebar.multiselect(
    "Asset Scan Universe Selection", 
    options=nifty_500_tickers, 
    default=["RELIANCE.NS", "TCS.NS", "INFY.NS", "SBIN.NS", "GUJGASLTD.NS", "AUROPHARMA.NS"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ Risk Parameters")
capital_base = st.sidebar.number_input("Capital Investment Base (₹)", min_value=10000, value=1000000, step=50000)
risk_pct = st.sidebar.slider("Max Account Risk Per Trade (%)", min_value=0.25, max_value=5.0, value=1.0, step=0.25)

if st.sidebar.button("🔄 Execute Multi-Factor ML Scan", use_container_width=True):
    if tickers:
        with st.spinner("Compiling global cross-market factor matrices & normalizing sector footprints..."):
            raw_data = yf.download(tickers, start=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'), progress=False)
            
            if not raw_data.empty:
                df_close = raw_data['Close'] if 'Close' in raw_data else raw_data
                df_volume = raw_data['Volume'] if 'Volume' in raw_data else pd.DataFrame()
                
                st.session_state['historical_raw_close'] = df_close
                st.session_state['historical_raw_volume'] = df_volume
                
                tech_data = calculate_advanced_technicals(df_close, df_volume)
                fund_data = fetch_corporate_fundamentals(tickers)
                live_sentiment = {ticker: compute_live_sentiment_vector(ticker) for ticker in tickers}
                
                volume_multipliers = {}
                for ticker in tickers:
                    if ticker in df_volume.columns:
                        series_vol = df_volume[ticker].dropna()
                        if len(series_vol) > 20:
                            volume_multipliers[ticker] = float(series_vol.iloc[-1] / series_vol.rolling(20).mean().iloc[-1])
                        else:
                            volume_multipliers[ticker] = 1.0
                    else:
                        volume_multipliers[ticker] = 1.0

                df_features, valid_tickers, scaled_matrix = compile_ensemble_features(
                    tickers, tech_data, fund_data, live_sentiment
                )
                
                summary_records = []
                for ticker in valid_tickers:
                    current_price = float(df_close[ticker].dropna().iloc[-1])
                    rsi_val = tech_data[ticker]["RSI"]
                    pe_val = fund_data[ticker]["Trailing_PE"]
                    trend_val = tech_data[ticker]["Trend_50_200_Ratio"]
                    sentiment_val = live_sentiment[ticker]
                    vol_mult = volume_multipliers[ticker]
                    
                    sector_pe_mult = df_features.loc[ticker, "Sector_Relative_PE"]
                    sector_name = df_features.loc[ticker, "Sector"]
                    
                    # Structural calculation tags
                    structural_trend = "Trading ABOVE EMA (Bullish)" if trend_val > 1.0 else "Trading BELOW EMA (Bearish)"
                    
                    if trend_val < 0.98 and sector_pe_mult > 1.25 and sentiment_val < -0.05:
                        action = "🔴 STRONG SELL / SHORT"
                    elif rsi_val < 35 and sector_pe_mult < 0.85 and sentiment_val < -0.05:
                        action = "⚪ MONITOR (Value Trap)"
                    elif rsi_val < 38 and sector_pe_mult < 0.90 and sentiment_val >= 0.0:
                        action = "🟢 ACCUMULATE"
                    elif trend_val > 1.02 and vol_mult > 1.4 and sentiment_val > 0.02 and sector_pe_mult < 1.15:
                        action = "🟢 STRONG BUY"
                    elif trend_val > 1.01 and rsi_val < 68:
                        action = "🟢 BUY"
                    else:
                        action = "⚪ HOLD (Neutral)"
                        
                    summary_records.append({
                        "Asset Ticker": ticker,
                        "Sector Cohort": sector_name,
                        "Market Price (₹)": round(current_price, 2),
                        "RSI (14D)": round(rsi_val, 1),
                        "P/E Ratio": round(pe_val, 1) if pe_val > 0 else "N/A",
                        "Sector P/E Mult": round(sector_pe_mult, 2),
                        "Vol Multiplier": round(vol_mult, 2),
                        "Structural Trend": structural_trend,
                        "AI Predictive Action": action
                    })
                    
                st.session_state['master_records'] = pd.DataFrame(summary_records)
            else:
                st.error("Data layer communication failure with yfinance.")
    else:
        st.warning("Please pick at least one asset symbol from the Nifty 500 selector sidebar panel to scan.")

# Display Data Grids with Multi-Dimensional Filtering Setup
if st.session_state['master_records'] is not None:
    st.header("🔍 Cross-Market Ensemble Scans (Multi-Factor Model Engine)")
    
    # ==========================================
    # DYNAMIC INTERACTIVE FILTER CONTROL BAR
    # ==========================================
    st.markdown("### 🎛️ Live Scan Output Filters")
    col1, col2 = st.columns(2)
    
    with col1:
        unique_signals = ["ALL"] + list(st.session_state['master_records']["AI Predictive Action"].unique())
        selected_signal_filter = st.selectbox("Filter Table by Predictive Action Signal:", options=unique_signals, index=0)
        
    with col2:
        unique_trends = ["ALL", "Trading ABOVE EMA (Bullish)", "Trading BELOW EMA (Bearish)"]
        selected_trend_filter = st.selectbox("Filter Table by Structural EMA Position:", options=unique_trends, index=0)
        
    # Apply user filter vectors to the dataframe state in RAM
    filtered_df = st.session_state['master_records'].copy()
    
    if selected_signal_filter != "ALL":
        filtered_df = filtered_df[filtered_df["AI Predictive Action"] == selected_signal_filter]
        
    if selected_trend_filter != "ALL":
        filtered_df = filtered_df[filtered_df["Structural Trend"] == selected_trend_filter]
        
    # Render final filtered output matrix
    st.markdown(f"**Showing {len(filtered_df)} matches out of active scanned metrics:**")
    st.table(filtered_df)
    
    st.markdown("---")
    
    # ==========================================
    # INTERACTIVE CHARTING INFRASTRUCTURE
    # ==========================================
    st.header("📊 Institutional Diagnostic Deep-Dive Chart")
    chart_target = st.selectbox("Select Target Asset to Inspect Technical & Volume Alignments:", options=tickers)
    
    if chart_target and st.session_state['historical_raw_close'] is not None:
        close_series = st.session_state['historical_raw_close'][chart_target].dropna()
        volume_series = st.session_state['historical_raw_volume'][chart_target].dropna() if chart_target in st.session_state['historical_raw_volume'].columns else None
        
        if not close_series.empty:
            ema50 = close_series.ewm(span=50, adjust=False).mean()
            ema200 = close_series.ewm(span=200, adjust=False).mean()
            vol_ma20 = volume_series.rolling(20).mean() if volume_series is not None else None
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.08, subplot_titles=(f'{chart_target} Price & Trend Struct', 'Volume Breakout Intensity Panel'),
                                row_width=[0.3, 0.7])
            
            fig.add_trace(gr.Scatter(x=close_series.index, y=close_series.values, name="Close Price", line=dict(color="#1f77b4", width=2)), row=1, col=1)
            fig.add_trace(gr.Scatter(x=ema50.index, y=ema50.values, name="50-Day EMA", line=dict(color="#2ca02c", width=1.5, dash='dash')), row=1, col=1)
            fig.add_trace(gr.Scatter(x=ema200.index, y=ema200.values, name="200-Day EMA", line=dict(color="#d62728", width=1.5, dash='dot')), row=1, col=1)
            
            if volume_series is not None:
                colors = ['#ff7f0e' if (v > 1.5 * m) else '#7f7f7f' for v, m in zip(volume_series.values, vol_ma20.values)]
                fig.add_trace(gr.Bar(x=volume_series.index, y=volume_series.values, name="Volume Traded", marker_color=colors), row=2, col=1)
                fig.add_trace(gr.Scatter(x=vol_ma20.index, y=vol_ma20.values, name="20-Day Vol MA", line=dict(color="#bcbd22", width=1.5)), row=2, col=1)
            
            fig.update_layout(height=600, template="plotly_dark", showlegend=True, margin=dict(l=40, r=40, t=40, b=40))
            fig.update_xaxes(title_text="Timeline Axis", row=2, col=1)
            fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
            fig.update_yaxes(title_text="Shares Vol", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("👋 Select assets and hit 'Execute Multi-Factor ML Scan' to activate the predictive pipelines.")
