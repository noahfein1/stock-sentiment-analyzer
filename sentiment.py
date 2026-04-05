import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime

def score_sentiment(df):
    analyzer = SentimentIntensityAnalyzer()
    
    # Score each post title
    df["sentiment"] = df["title"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    df["sentiment_label"] = df["sentiment"].apply(
        lambda x: "positive" if x >= 0.05 else ("negative" if x <= -0.05 else "neutral")
    )
    return df

def aggregate_daily_sentiment(df):
    df["date"] = pd.to_datetime(df["created_utc"]).dt.date
    
    daily = df.groupby(["ticker", "date"]).agg(
        avg_sentiment=("sentiment", "mean"),
        post_count=("title", "count"),
        avg_score=("score", "mean"),
        avg_upvote_ratio=("upvote_ratio", "mean"),
        avg_comments=("num_comments", "mean")
    ).reset_index()
    
    return daily

if __name__ == "__main__":
    # Load reddit posts
    df = pd.read_csv("data/reddit_posts.csv")
    print(f"Loaded {len(df)} posts")

    # Score sentiment
    df = score_sentiment(df)
    df.to_csv("data/reddit_posts_scored.csv", index=False)
    print("Saved data/reddit_posts_scored.csv")

    # Aggregate by day
    daily = aggregate_daily_sentiment(df)
    daily.to_csv("data/daily_sentiment.csv", index=False)
    print("Saved data/daily_sentiment.csv")

    # Preview
    print("\nSample sentiment scores:")
    print(df[["ticker", "title", "sentiment", "sentiment_label"]].head(10).to_string())

    print("\nDaily sentiment summary:")
    print(daily.sort_values("date", ascending=False).head(10).to_string())