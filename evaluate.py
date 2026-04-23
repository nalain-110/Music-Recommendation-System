import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from data_generator import generate_all
from collaborative_filter import CollaborativeFilter
from feature_extractor import ContentBasedRecommender


def precision_at_k(recommended: list, relevant: list, k: int) -> float:
    top_k = recommended[:k]
    hits  = len(set(top_k) & set(relevant))
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: list, relevant: list, k: int) -> float:
    top_k = recommended[:k]
    hits  = len(set(top_k) & set(relevant))
    return hits / len(relevant) if relevant else 0.0


def evaluate_cf(interactions_df: pd.DataFrame, songs_df: pd.DataFrame,
                n_eval_users: int = 30, k: int = 10):
    print("\n" + "="*55)
    print("  Collaborative Filtering Evaluation")
    print("="*55)

    train_df, test_df = train_test_split(interactions_df, test_size=0.2, random_state=42)

    cf = CollaborativeFilter(n_factors=20, n_neighbors=15)
    cf.fit(train_df, songs_df)

    train_users = set(train_df["user_id"].unique())
    test_users  = set(test_df["user_id"].unique())
    eval_users  = list(train_users & test_users)[:n_eval_users]

    precisions, recalls, rmses = [], [], []

    for uid in eval_users:
        user_test = test_df[test_df["user_id"] == uid]
        relevant  = user_test[user_test["rating"] >= 4.0]["song_id"].tolist()
        if not relevant:
            continue

        recs = cf.recommend_svd(uid, n=k*2)
        rec_ids = [r["song_id"] for r in recs]

        prec = precision_at_k(rec_ids, relevant, k)
        rec  = recall_at_k(rec_ids, relevant, k)
        precisions.append(prec)
        recalls.append(rec)
        rated_test = user_test.to_dict(orient="records")
        for item in rated_test:
            sid = item["song_id"]
            actual = item["rating"]

            if uid in cf.user_index and sid in cf.song_index:
                ui = cf.user_index[uid]
                si = cf.song_index[sid]
                pred_vec = cf.item_latent @ cf.user_latent[ui]
                predicted = float(np.clip(pred_vec[si], 1, 5))
                rmses.append((actual - predicted) ** 2)

    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall    = np.mean(recalls) if recalls else 0
    rmse          = np.sqrt(np.mean(rmses)) if rmses else 0
    f1            = (2 * avg_precision * avg_recall / (avg_precision + avg_recall + 1e-9))

    print(f"  Evaluated on {len(precisions)} users  |  K = {k}")
    print(f"  Precision@{k}   : {avg_precision:.4f}")
    print(f"  Recall@{k}      : {avg_recall:.4f}")
    print(f"  F1@{k}          : {f1:.4f}")
    print(f"  RMSE           : {rmse:.4f}")
    return {"precision": avg_precision, "recall": avg_recall, "f1": f1, "rmse": rmse}


def evaluate_cbf(songs_df: pd.DataFrame, interactions_df: pd.DataFrame,
                 n_eval: int = 20, k: int = 10):
                     
    print("\n" + "="*55)
    print("  Content-Based Filtering Evaluation")
    print("="*55)

    cbf = ContentBasedRecommender()
    cbf.fit(songs_df)

    genre_hits = []
    artist_diversities = []

    song_ids = songs_df["song_id"].tolist()
    np.random.seed(42)
    sample_ids = np.random.choice(song_ids, size=min(n_eval, len(song_ids)), replace=False)

    for sid in sample_ids:
        seed_genre = songs_df[songs_df["song_id"]==sid]["genre"].values[0]
        recs = cbf.recommend(sid, n=k)

        genre_match = sum(1 for r in recs if r["genre"] == seed_genre) / len(recs) if recs else 0
        genre_hits.append(genre_match)

        artists = [r["artist"] for r in recs]
        diversity = len(set(artists)) / len(artists) if artists else 0
        artist_diversities.append(diversity)

    avg_genre   = np.mean(genre_hits)
    avg_diverse = np.mean(artist_diversities)

    print(f"  Evaluated on {n_eval} seed songs  |  K = {k}")
    print(f"  Genre Consistency : {avg_genre:.4f}   (↑ better)")
    print(f"  Artist Diversity  : {avg_diverse:.4f}   (↑ better)")
    return {"genre_consistency": avg_genre, "artist_diversity": avg_diverse}


def print_summary(cf_metrics, cbf_metrics):
    print("\n" + "="*55)
    print("  EVALUATION SUMMARY")
    print("="*55)
    print(f"  CF  Precision@10  : {cf_metrics['precision']:.4f}")
    print(f"  CF  Recall@10     : {cf_metrics['recall']:.4f}")
    print(f"  CF  F1@10         : {cf_metrics['f1']:.4f}")
    print(f"  CF  RMSE          : {cf_metrics['rmse']:.4f}")
    print(f"  CBF Genre Consist : {cbf_metrics['genre_consistency']:.4f}")
    print(f"  CBF Artist Divers : {cbf_metrics['artist_diversity']:.4f}")
    print("="*55)


if __name__ == "__main__":
    print(" Music Recommendation System — Model Evaluation")
    songs, users, interactions = generate_all()
    cf_metrics  = evaluate_cf(interactions, songs, n_eval_users=40, k=10)
    cbf_metrics = evaluate_cbf(songs, interactions, n_eval=30, k=10)
    print_summary(cf_metrics, cbf_metrics)
