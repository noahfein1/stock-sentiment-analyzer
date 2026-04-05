import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

def load_and_merge():
    sentiment = pd.read_csv("data/daily_sentiment.csv")
    prices = pd.read_csv("data/stock_prices.csv")

    sentiment["date"] = pd.to_datetime(sentiment["date"])
    prices["Date"] = pd.to_datetime(prices["Date"])

    # Calculate daily return and next day direction
    prices = prices.sort_values(["ticker", "Date"])
    prices["daily_return"] = (prices["Close"] - prices["Open"]) / prices["Open"]
    prices["next_day_return"] = prices.groupby("ticker")["daily_return"].shift(-1)
    prices["target"] = (prices["next_day_return"] > 0).astype(int)  # 1 = up, 0 = down

    # Merge
    merged = pd.merge(
        sentiment,
        prices,
        left_on=["ticker", "date"],
        right_on=["ticker", "Date"],
        how="inner"
    )
    return merged

def build_features(df):
    # Rolling sentiment features
    df = df.sort_values(["ticker", "date"])
    df["sentiment_3d_avg"] = df.groupby("ticker")["avg_sentiment"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    df["post_count_3d"] = df.groupby("ticker")["post_count"].transform(
        lambda x: x.rolling(3, min_periods=1).sum()
    )

    features = [
        "avg_sentiment", "sentiment_3d_avg", "post_count",
        "post_count_3d", "avg_score", "avg_upvote_ratio",
        "avg_comments", "daily_return"
    ]

    df = df.dropna(subset=features + ["target"])
    return df, features

def train_models(df, features):
    X = df[features]
    y = df["target"]

    if len(X) < 10:
        print(f"Not enough data to train ({len(X)} rows). Collect more posts over more days.")
        return None, None, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)

    print("=== Logistic Regression ===")
    print(classification_report(y_test, lr_preds, zero_division=0))

    print("=== Random Forest ===")
    print(classification_report(y_test, rf_preds, zero_division=0))

    # Feature importance
    importance = pd.DataFrame({
        "feature": features,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)
    print("\nTop Features:")
    print(importance.to_string(index=False))

    return lr, rf, scaler

if __name__ == "__main__":
    print("Loading and merging data...")
    df = load_and_merge()
    print(f"Merged dataset: {len(df)} rows")

    print("\nBuilding features...")
    df, features = build_features(df)
    print(f"Training dataset: {len(df)} rows, {len(features)} features")

    print("\nTraining models...")
    lr, rf, scaler = train_models(df, features)

    if rf is not None:
        df.to_csv("data/merged_features.csv", index=False)
        print("\nSaved data/merged_features.csv")
        print("\nDone! Ready to build the Streamlit app.")