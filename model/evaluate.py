import pickle
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix


# Loading model plus mappings, matrix

with open('../models/als_model.pkl', 'rb') as f:
    model, user_id_to_index, item_id_to_index = pickle.load(f)

train_matrix = sparse.load_npz('../models/train_matrix.npz')


# Loading interaction data

df = pd.read_csv('../data/sessionized_events.csv')
df = df[
    df['user_id'].isin(user_id_to_index) &
    df['item_id'].isin(item_id_to_index)
]

df['user_idx'] = df['user_id'].map(user_id_to_index)
df['item_idx'] = df['item_id'].map(item_id_to_index)

_, test_df = train_test_split(df, test_size=0.2, random_state=42)

def build_sparse_matrix(df, n_users, n_items):
    data = np.ones(len(df), dtype=np.float32)
    return coo_matrix((data, (df['user_idx'], df['item_idx'])), shape=(n_users, n_items)).tocsr()

n_users = len(user_id_to_index)
n_items = len(item_id_to_index)

test_matrix = build_sparse_matrix(test_df, n_users, n_items)


# Manual Precision@K(dot product)

def manual_precision_at_k(user_factors, item_factors, test_mat, K=10, n_users=100):
    sampled_users = np.random.choice(np.arange(user_factors.shape[0]), size=min(n_users, user_factors.shape[0]), replace=False)

    precisions = []
    for user_idx in sampled_users:
        true_items = test_mat[user_idx].nonzero()[1]
        if len(true_items) == 0:
            continue

        # predicted scores for all items
        scores = user_factors[user_idx] @ item_factors.T

        # Recommending Top-K items (excluding already seen ones if needed)
        top_items = np.argsort(scores)[-K:][::-1]

        hits = np.intersect1d(top_items, true_items)
        precisions.append(len(hits) / K)

    return np.mean(precisions) if precisions else 0.0


user_factors = model.user_factors
item_factors = model.item_factors

precision = manual_precision_at_k(user_factors, item_factors, test_matrix, K=10, n_users=100)
print(f" Manual Precision@10: {precision:.4f}")
