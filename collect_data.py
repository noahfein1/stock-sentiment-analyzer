import praw
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
from config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT, TICKERS, SUBREDDITS

# Setup Reddit client
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

def collect_reddit_posts(tickers, subreddits, limit=500):
    posts = []
    for subreddit_name in subreddits:
        subreddit = reddit.subreddit(subreddit_name)
        for category in ["hot", "new", "top"]:
            print(f"Scraping r/{subreddit_name} - {category}...")
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
                                "url": post.url
                            })
            except Exception as e:
                print(f"Error on {subreddit_name}/{category}: {e}")

    df = pd.DataFrame(posts).drop_duplicates(subset=["url", "ticker"])
    print(f"\nCollected {len(df)} posts across {len(tickers)} tickers")
    return df

def collect_stock_prices(tickers, days=90):
    all_prices = []
    end = datetime.today()
    start = end - timedelta(days=days)
    for ticker in tickers:
        print(f"Fetching price data for {ticker}...")
        df = yf.download(ticker, start=start, end=end, progress=False)
        df["ticker"] = ticker
        df = df.reset_index()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        all_prices.append(df[["Date", "ticker", "Open", "Close", "Volume"]])
    return pd.concat(all_prices, ignore_index=True)

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    # Collect Reddit posts
    reddit_df = collect_reddit_posts(TICKERS, SUBREDDITS)
    reddit_df.to_csv("data/reddit_posts.csv", index=False)
    print("Saved data/reddit_posts.csv")

    # Collect stock prices
    price_df = collect_stock_prices(TICKERS)
    price_df.to_csv("data/stock_prices.csv", index=False)
    print("Saved data/stock_prices.csv")

    print("\nDone! Check your /data folder.")