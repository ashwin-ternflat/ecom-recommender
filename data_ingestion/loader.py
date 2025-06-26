import os
os.makedirs("../data", exist_ok=True)



import pandas as pd
from datetime import datetime
import random

def simulate_batch_events(n_rows=100):
    user_ids = [f"user_{i}" for i in range(1, 11)]
    item_ids = [f"item_{i}" for i in range(1, 21)]
    actions = ["click", "view", "add_to_cart", "purchase"]
    data = [{
        "user_id": random.choice(user_ids),
        "item_id": random.choice(item_ids),
        "action": random.choice(actions),
        "timestamp": datetime.utcnow().isoformat()
    } for _ in range(n_rows)]
    return pd.DataFrame(data)

if __name__ == '__main__':
    df = simulate_batch_events()
    df.to_csv("../data/user_events_batch.csv", index=False)
    print("Saved user event batch to data/user_events_batch.csv")
