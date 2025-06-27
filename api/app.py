from fastapi import FastAPI, HTTPException, Query
from typing import List
import pickle
import numpy as np

app = FastAPI(title="E-Commerce Recommender API")


# Load model + mappings
with open("models/als_model.pkl", "rb") as f:
    model, user_id_to_index, item_id_to_index = pickle.load(f)

# For reverse lookups
index_to_item_id = {idx: item for item, idx in item_id_to_index.items()}

# Extract user/item factors
user_factors = model.user_factors
item_factors = model.item_factors


def recommend_items_for_user(user_id: str, k: int = 10) -> List[str]:
    if user_id not in user_id_to_index:
        raise ValueError(f"User ID {user_id} not found")

    user_idx = user_id_to_index[user_id]
    user_vector = user_factors[user_idx]
    scores = user_vector @ item_factors.T

    top_indices = np.argsort(scores)[-k:][::-1]
    recommended_items = [index_to_item_id[i] for i in top_indices]

    return recommended_items



# API Endpoint

@app.get("/recommendations", response_model=List[str])
def get_recommendations(user_id: str = Query(...), k: int = Query(10)):
    """
    Get top-k product recommendations for a given user_id.
    """
    try:
        return recommend_items_for_user(user_id, k)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080) 
