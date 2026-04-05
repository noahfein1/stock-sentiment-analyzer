import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Stock Sentiment Analyzer", layout="wide", page_icon="📈")

TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "AMD"]

@st.cache_data
def load_data():
    posts = pd.read_csv("data/reddit_posts_scored.csv")
    prices = pd.read_csv("data/stock_prices.csv")
    daily = pd.read_csv("data/daily_sentiment.csv")
    merged = pd.read_csv("data/merged_features.csv")
    return posts, prices, daily, merged

@st.cache_resource
def train_model(merged):
    features = [
        "avg_sentiment", "sentiment_3d_avg", "post_count",
        "post_count_3d", "avg_score", "avg_upvote_ratio",
        "avg_comments", "daily_return"
    ]
    df = merged.dropna(subset=features + ["target"])
    if len(df) < 5:
        return None, None, features
    X = df[features]
    y = df["target"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    return rf, scaler, features

def get_prediction(ticker, merged, rf, scaler, features):
    df = merged[merged["ticker"] == ticker].copy()
    if len(df) == 0 or rf is None:
        return None, None
    latest = df.sort_values("date").iloc[-1]
    X = latest[features].values.reshape(1, -1)
    try:
        X_scaled = scaler.transform(X)
        prob = rf.predict_proba(X_scaled)[0]
        pred = rf.predict(X_scaled)[0]
        return pred, prob
    except:
        return None, None

# --- UI ---
st.title("📈 Stock Sentiment Analyzer")
st.markdown("*Analyzing Reddit sentiment vs. stock price movement*")

posts, prices, daily, merged = load_data()
rf, scaler, features = train_model(merged)

# Sidebar
st.sidebar.header("Settings")
selected_ticker = st.sidebar.selectbox("Select Ticker", TICKERS)
days_back = st.sidebar.slider("Days of price history", 30, 90, 60)

# Filter data
ticker_posts = posts[posts["ticker"] == selected_ticker]
ticker_prices = prices[prices["ticker"] == selected_ticker].copy()
ticker_prices["Date"] = pd.to_datetime(ticker_prices["Date"])
ticker_daily = daily[daily["ticker"] == selected_ticker].copy()
ticker_daily["date"] = pd.to_datetime(ticker_daily["date"]).dt.normalize()

cutoff = datetime.today() - timedelta(days=days_back)
ticker_prices = ticker_prices[ticker_prices["Date"] >= cutoff]

# --- Top metrics ---
col1, col2, col3, col4 = st.columns(4)

avg_sent = ticker_daily["avg_sentiment"].mean() if len(ticker_daily) > 0 else 0
total_posts = len(ticker_posts)
latest_price = ticker_prices["Close"].iloc[-1] if len(ticker_prices) > 0 else 0
price_change = ((ticker_prices["Close"].iloc[-1] - ticker_prices["Close"].iloc[-2]) /
                ticker_prices["Close"].iloc[-2] * 100) if len(ticker_prices) > 1 else 0

col1.metric("Avg Sentiment", f"{avg_sent:.3f}", delta="positive" if avg_sent > 0 else "negative")
col2.metric("Reddit Posts", total_posts)
col3.metric("Latest Close", f"${latest_price:.2f}", f"{price_change:+.2f}%")

pred, prob = get_prediction(selected_ticker, merged, rf, scaler, features)
if pred is not None:
    direction = "⬆️ UP" if pred == 1 else "⬇️ DOWN"
    confidence = max(prob) * 100
    col4.metric("Model Prediction", direction, f"{confidence:.0f}% confidence")
else:
    col4.metric("Model Prediction", "Insufficient data", "")

st.divider()

# --- Price Chart ---
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader(f"{selected_ticker} Price History")
    if len(ticker_prices) > 0:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ticker_prices["Date"], y=ticker_prices["Close"],
            mode="lines", name="Close Price",
            line=dict(color="#00b4d8", width=2)
        ))
        fig.update_layout(
            height=350, margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Date", yaxis_title="Price ($)",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Sentiment Over Time")
    if len(ticker_daily) > 0:
        fig2 = px.bar(
            ticker_daily.sort_values("date"),
            x="date", y="avg_sentiment",
            color="avg_sentiment",
            color_continuous_scale=["red", "gray", "green"],
            range_color=[-1, 1]
        )
        fig2.update_layout(
            height=350, margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False, coloraxis_showscale=False,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No sentiment data for this ticker yet.")

# --- Feature Importance ---
st.divider()
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("What Drives the Model")
    if rf is not None:
        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": rf.feature_importances_
        }).sort_values("Importance", ascending=True)
        fig3 = px.bar(importance_df, x="Importance", y="Feature", orientation="h",
                      color="Importance", color_continuous_scale="blues")
        fig3.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                           coloraxis_showscale=False,
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)

with col_b:
    st.subheader(f"Recent Reddit Posts — {selected_ticker}")
    if len(ticker_posts) > 0:
        display = ticker_posts[["title", "sentiment", "sentiment_label", "score"]].head(8)
        display.columns = ["Title", "Sentiment", "Label", "Upvotes"]
        st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.info("No Reddit posts collected for this ticker yet.")