import os
os.makedirs("../models", exist_ok=True)



import mlflow, pickle, pandas as pd, numpy as np
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares

df = pd.read_csv('../data/sessionized_events.csv')
df = df[df['action'].isin(['add_to_cart', 'purchase'])]

user_ids, item_ids = df['user_id'].unique(), df['item_id'].unique()
user_map = {u: i for i, u in enumerate(user_ids)}
item_map = {i: j for j, i in enumerate(item_ids)}

df['user_idx'] = df['user_id'].map(user_map)
df['item_idx'] = df['item_id'].map(item_map)

matrix = coo_matrix((np.ones(len(df)), (df['user_idx'], df['item_idx'])))
mlflow.set_experiment("ecommerce_recommender")

with mlflow.start_run():
    model = AlternatingLeastSquares(factors=20, regularization=0.1, iterations=20)
    model.fit(matrix)
    with open('../models/als_model.pkl', 'wb') as f:
        pickle.dump((model, user_map, item_map), f)
    mlflow.log_param("factors", 20)
    mlflow.log_param("regularization", 0.1)
    mlflow.log_param("iterations", 20)
    mlflow.log_artifact("../models/als_model.pkl")
    print("Logged model to MLflow.")
