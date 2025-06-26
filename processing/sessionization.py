import os
os.makedirs("../data", exist_ok=True)



import pandas as pd
from datetime import datetime, timedelta

INPUT_CSV = '../data/user_events_batch.csv'
OUTPUT_CSV = '../data/sessionized_events.csv'
SESSION_TIMEOUT = timedelta(minutes=30)

def parse_timestamp(ts):
    return datetime.fromisoformat(ts)

def sessionize(df):
    df['timestamp'] = df['timestamp'].apply(parse_timestamp)
    df.sort_values(['user_id', 'timestamp'], inplace=True)
    session_ids, last_user, last_time, current_session = [], None, None, 0

    for _, row in df.iterrows():
        if row['user_id'] != last_user or (last_time and row['timestamp'] - last_time > SESSION_TIMEOUT):
            current_session += 1
        session_ids.append(current_session)
        last_user, last_time = row['user_id'], row['timestamp']
    df['session_id'] = session_ids
    return df

if __name__ == '__main__':
    df = pd.read_csv(INPUT_CSV)
    df = sessionize(df)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved sessionized logs to {OUTPUT_CSV}")
