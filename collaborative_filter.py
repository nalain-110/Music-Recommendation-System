import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import joblib
import os


class CollaborativeFilter:
    def __init__(self, n_factors: int = 20, n_neighbors: int = 15):
        self.n_factors = n_factors
        self.n_neighbors = n_neighbors
        self.svd = TruncatedSVD(n_components=n_factors, random_state=42)

        self.user_item_matrix = None 
        self.user_latent = None 
        self.item_latent = None      
        self.user_similarity = None    

        self.user_index = {}  
        self.song_index = {}   
        self.index_user = {}
        self.index_song = {}
        self.songs_df = None
        self.fitted = False

    def fit(self, interactions_df: pd.DataFrame, songs_df: pd.DataFrame):
        self.songs_df = songs_df.copy()

        user_ids = sorted(interactions_df["user_id"].unique())
        song_ids = sorted(interactions_df["song_id"].unique())

        for i, uid in enumerate(user_ids):
            self.user_index[uid] = i
            self.index_user[i] = uid
        for j, sid in enumerate(song_ids):
            self.song_index[sid] = j
            self.index_song[j] = sid

        n_users = len(user_ids)
        n_songs = len(song_ids)
        matrix = np.zeros((n_users, n_songs))
        for _, row in interactions_df.iterrows():
            ui = self.user_index.get(row["user_id"])
            si = self.song_index.get(row["song_id"])
            if ui is not None and si is not None:
                matrix[ui, si] = row["rating"]

        self.user_item_matrix = matrix

        self.user_latent = self.svd.fit_transform(matrix)      
        self.item_latent = self.svd.components_.T                 


        self.user_latent_norm = normalize(self.user_latent)

        self.user_similarity = cosine_similarity(self.user_latent_norm)

        self.fitted = True
        variance = self.svd.explained_variance_ratio_.sum()
        print(f" CF fitted: {n_users} users × {n_songs} songs, "
              f"SVD explains {variance:.1%} variance.")

    def _song_meta(self, song_id: int):
        rows = self.songs_df[self.songs_df["song_id"] == song_id]
        if rows.empty:
            return {"song_id": song_id, "title": "Unknown", "artist": "Unknown",
                    "genre": "Unknown", "popularity": 0}
        r = rows.iloc[0]
        return {
            "song_id":    int(song_id),
            "title":      r["title"],
            "artist":     r["artist"],
            "genre":      r["genre"],
            "popularity": int(r["popularity"]),
        }

    def recommend_svd(self, user_id: int, n: int = 10, exclude_ids: list = None):
        if not self.fitted or user_id not in self.user_index:
            return []

        ui = self.user_index[user_id]
        user_vec = self.user_latent[ui]   # (k,)

        # Predicted ratings = user_vec · item_latent.T
        pred_ratings = self.item_latent @ user_vec   # (n_songs,)

        exclude = set(exclude_ids or [])
        # Also exclude already-rated songs
        rated_cols = np.where(self.user_item_matrix[ui] > 0)[0]
        exclude |= {self.index_song[c] for c in rated_cols if c in self.index_song}

        song_scores = []
        for j, score in enumerate(pred_ratings):
            sid = self.index_song.get(j)
            if sid and sid not in exclude:
                song_scores.append((sid, float(score)))

        song_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for sid, score in song_scores[:n]:
            meta = self._song_meta(sid)
            meta["score"] = round(score, 4)
            meta["method"] = "SVD"
            results.append(meta)
        return results

    def recommend_knn(self, user_id: int, n: int = 10, exclude_ids: list = None):
        if not self.fitted or user_id not in self.user_index:
            return []

        ui = self.user_index[user_id]
        sim_row = self.user_similarity[ui]

        # Top-k similar users (exclude self)
        neighbor_indices = np.argsort(sim_row)[::-1]
        neighbors = [
            (idx, sim_row[idx])
            for idx in neighbor_indices
            if idx != ui
        ][:self.n_neighbors]

        # Weighted average of neighbor ratings
        song_scores = {}
        song_weights = {}
        for nb_idx, sim in neighbors:
            for j, rating in enumerate(self.user_item_matrix[nb_idx]):
                if rating > 0:
                    sid = self.index_song.get(j)
                    if sid:
                        song_scores[sid] = song_scores.get(sid, 0) + sim * rating
                        song_weights[sid] = song_weights.get(sid, 0) + abs(sim)

        normalized = {
            sid: song_scores[sid] / song_weights[sid]
            for sid in song_scores
            if song_weights[sid] > 0
        }

        exclude = set(exclude_ids or [])
        rated_cols = np.where(self.user_item_matrix[ui] > 0)[0]
        exclude |= {self.index_song[c] for c in rated_cols if c in self.index_song}

        sorted_songs = sorted(
            [(sid, sc) for sid, sc in normalized.items() if sid not in exclude],
            key=lambda x: x[1], reverse=True
        )

        results = []
        for sid, score in sorted_songs[:n]:
            meta = self._song_meta(sid)
            meta["score"] = round(score, 4)
            meta["method"] = "KNN"
            results.append(meta)
        return results

    def get_similar_users(self, user_id: int, n: int = 5):
        if user_id not in self.user_index:
            return []
        ui = self.user_index[user_id]
        sim_row = self.user_similarity[ui]
        top = np.argsort(sim_row)[::-1][1:n+1]
        return [
            {"user_id": self.index_user[i], "similarity": round(float(sim_row[i]), 4)}
            for i in top
        ]

    def save(self, path="models/cf_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"CF model saved → {path}")

    @staticmethod
    def load(path="models/cf_model.pkl"):
        return joblib.load(path)


if __name__ == "__main__":
    from data_generator import generate_all
    songs, users, interactions = generate_all()
    cf = CollaborativeFilter(n_factors=20, n_neighbors=15)
    cf.fit(interactions, songs)
    print("\nSVD recs for user 1:", cf.recommend_svd(1, n=5))
    print("\nKNN recs for user 1:", cf.recommend_knn(1, n=5))
    cf.save()
