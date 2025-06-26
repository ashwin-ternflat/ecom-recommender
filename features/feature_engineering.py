import os
import pandas as pd

# Ensure output directory exists
os.makedirs("../data", exist_ok=True)

# Input file path
INPUT_CSV = '../data/sessionized_events.csv'

# Load and preprocess timestamp column
df = pd.read_csv(INPUT_CSV)
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
df['timestamp'] = df['timestamp'].dt.tz_localize(None)  # Make tz-naive

def engineer_features(df):
    # ---------------- User Features ----------------
    user_feats = df.groupby('user_id').agg({
        'timestamp': 'count',
        'session_id': pd.Series.nunique,
        'item_id': pd.Series.nunique
    })
    user_feats.columns = ['event_count', 'session_count', 'unique_items']

    # Handle user recency with consistent tz-naive timestamps
    latest_ts = df.groupby('user_id')['timestamp'].max()
    latest_ts = pd.to_datetime(latest_ts, utc=True).dt.tz_localize(None)
    now = pd.Timestamp.utcnow().replace(tzinfo=None)


    user_feats['user_recency'] = (now - latest_ts).dt.total_seconds() / 3600

    # ---------------- Item Features ----------------
    item_feats = df.groupby('item_id').agg({
        'user_id': 'count',
        'session_id': pd.Series.nunique
    })
    item_feats.columns = ['item_popularity', 'unique_sessions']

    return user_feats.reset_index(), item_feats.reset_index()

if __name__ == '__main__':
    user_df, item_df = engineer_features(df)
    user_df.to_csv('../data/user_features.csv', index=False)
    item_df.to_csv('../data/item_features.csv', index=False)
    print(" Saved user and item features")
