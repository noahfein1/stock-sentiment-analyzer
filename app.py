import streamlit as st
import pandas as pd
import numpy as np
import praw
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
SUBREDDITS = ["wallstreetbets", "stocks", "investing"]

def get_reddit_client():
    try:
        return praw.Reddit(
            client_id=st.secrets["REDDIT_CLIENT_ID"],
            client_secret=st.secrets["REDDIT_CLIENT_SECRET"],
            user_agent=st.secrets["REDDIT_USER_AGENT"]
        )
    except:
        return None

@st.cache_data(ttl=3600)
def collect_reddit_posts(tickers, subreddits, limit=300):
    reddit = get_reddit_client()
    if reddit is None:
        return pd.DataFrame()
    posts = []
    for subreddit_name in subreddits:
        subreddit = reddit.subreddit(subreddit_name)
        for category in ["hot", "new", "top"]:
            try:
                feed = getattr(subreddit, category)(limit=limit)
                for post in feed:
                    title = post.title.upper()
                    for ticker in tickers:
                        if f" {ticker} " in f" {title} " or f"${ticker}" in title:
                            posts.append({
                                "ticker": ticker,
                                "title": post.title,
                                "score": post.score,
                                "upvote_ratio": post.upvote_ratio,
                                "num_comments": post.num_comments,
                                "created_utc": datetime.utcfromtimestamp(post.created_utc),
                                "subreddit": subreddit_name,
                            })
            except:
                continue
    if not posts:
        return pd.DataFrame()
    df = pd.DataFrame(posts).drop_duplicates(subset=["title", "ticker"])
    analyzer = SentimentIntensityAnalyzer()
    df["sentiment"] = df["title"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    df["sentiment_label"] = df["sentiment"].apply(
        lambda x: "positive" if x >= 0.05 else ("negative" if x <= -0.05 else "neutral")
    )
    return df

@st.cache_data(ttl=3600)
def collect_stock_prices(tickers, days=90):
    all_prices = []
    end = datetime.today()
    start = end - timedelta(days=days)
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            df["ticker"] = ticker
            df = df.reset_index()
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            all_prices.append(df[["Date", "ticker", "Open", "Close", "Volume"]])
        except:
            continue
    if not all_prices:
        return pd.DataFrame()
    return pd.concat(all_prices, ignore_index=True)

def aggregate_daily_sentiment(posts_df):
    if len(posts_df) == 0:
        return pd.DataFrame()
    posts_df["date"] = pd.to_datetime(posts_df["created_utc"]).dt.normalize()
    daily = posts_df.groupby(["ticker", "date"]).agg(
        avg_sentiment=("sentiment", "mean"),
        post_count=("title", "count"),
        avg_score=("score", "mean"),
        avg_upvote_ratio=("upvote_ratio", "mean"),
        avg_comments=("num_comments", "mean")
    ).reset_index()
    return daily

def merge_and_build_features(daily, prices):
    if len(daily) == 0 or len(prices) == 0:
        return pd.DataFrame()
    prices = prices.copy().sort_values(["ticker", "Date"])
    prices["daily_return"] = (prices["Close"] - prices["Open"]) / prices["Open"]
    prices["next_day_return"] = prices.groupby("ticker")["daily_return"].shift(-1)
    prices["target"] = (prices["next_day_return"] > 0).astype(int)
    merged = pd.merge(daily, prices, left_on=["ticker", "date"], right_on=["ticker", "Date"], how="inner")
    merged = merged.sort_values(["ticker", "date"])
    merged["sentiment_3d_avg"] = merged.groupby("ticker")["avg_sentiment"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    merged["post_count_3d"] = merged.groupby("ticker")["post_count"].transform(
        lambda x: x.rolling(3, min_periods=1).sum()
    )
    return merged

@st.cache_resource
def train_model(merged_json):
    merged = pd.read_json(merged_json)
    features = ["avg_sentiment", "sentiment_3d_avg", "post_count",
                "post_count_3d", "avg_score", "avg_upvote_ratio",
                "avg_comments", "daily_return"]
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
    try:
        X = latest[features].values.reshape(1, -1)
        X_scaled = scaler.transform(X)
        prob = rf.predict_proba(X_scaled)[0]
        pred = rf.predict(X_scaled)[0]
        return pred, prob
    except:
        return None, None

# --- UI ---
st.title("📈 Stock Sentiment Analyzer")
st.markdown("*Analyzing Reddit sentiment vs. stock price movement*")

with st.spinner("Collecting live Reddit posts and stock prices..."):
    posts = collect_reddit_posts(TICKERS, SUBREDDITS)
    prices = collect_stock_prices(TICKERS)
    daily = aggregate_daily_sentiment(posts)
    merged = merge_and_build_features(daily, prices)

if len(merged) > 0:
    rf, scaler, features = train_model(merged.to_json())
else:
    rf, scaler, features = None, None, []

# Sidebar
st.sidebar.header("Settings")
selected_ticker = st.sidebar.selectbox("Select Ticker", TICKERS)
days_back = st.sidebar.slider("Days of price history", 30, 90, 60)

# Filter data
ticker_posts = posts[posts["ticker"] == selected_ticker] if len(posts) > 0 else pd.DataFrame()
ticker_prices = prices[prices["ticker"] == selected_ticker].copy() if len(prices) > 0 else pd.DataFrame()
ticker_daily = daily[daily["ticker"] == selected_ticker].copy() if len(daily) > 0 else pd.DataFrame()

if len(ticker_prices) > 0:
    ticker_prices["Date"] = pd.to_datetime(ticker_prices["Date"])
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

pred, prob = get_prediction(selected_ticker, merged, rf, scaler, features) if len(merged) > 0 else (None, None)
if pred is not None:
    direction = "⬆️ UP" if pred == 1 else "⬇️ DOWN"
    confidence = max(prob) * 100
    col4.metric("Model Prediction", direction, f"{confidence:.0f}% confidence")
else:
    col4.metric("Model Prediction", "Insufficient data", "")

st.divider()

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