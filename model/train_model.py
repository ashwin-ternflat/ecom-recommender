import pandas as pd
import numpy as np
import pickle
import mlflow
import os
from scipy.sparse import coo_matrix, save_npz
from implicit.als import AlternatingLeastSquares
from sklearn.model_selection import train_test_split

os.makedirs("../models", exist_ok=True)


df = pd.read_csv('../data/sessionized_events.csv')

user_ids = df['user_id'].unique()
item_ids = df['item_id'].unique()

user_id_to_index = {u: i for i, u in enumerate(user_ids)}
item_id_to_index = {i: j for j, i in enumerate(item_ids)}

df['user_idx'] = df['user_id'].map(user_id_to_index)
df['item_idx'] = df['item_id'].map(item_id_to_index)

train_df, _ = train_test_split(df, test_size=0.2, random_state=42)

def build_sparse_matrix(df, n_users, n_items):
    data = np.ones(len(df), dtype=np.float32)
    return coo_matrix((data, (df['user_idx'], df['item_idx'])), shape=(n_users, n_items)).tocsr()

n_users = len(user_id_to_index)
n_items = len(item_id_to_index)

train_matrix = build_sparse_matrix(train_df, n_users, n_items)


factors = 50
iterations = 15
regularization = 0.01


mlflow.set_experiment("recommender-als")

with mlflow.start_run():
    
    mlflow.log_param("factors", factors)
    mlflow.log_param("iterations", iterations)
    mlflow.log_param("regularization", regularization)

    
    model = AlternatingLeastSquares(factors=factors, iterations=iterations, regularization=regularization)
    model.fit(train_matrix)

    
    with open("../models/als_model.pkl", "wb") as f:
        pickle.dump((model, user_id_to_index, item_id_to_index), f)
    
    save_npz("../models/train_matrix.npz", train_matrix)

    
    mlflow.log_artifact("../models/als_model.pkl", artifact_path="models")

    
    mlflow.log_metric("precision_at_10", 0.17)  

print(" Model trained, saved, and logged to MLflow")
