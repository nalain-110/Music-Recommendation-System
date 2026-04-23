import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import joblib
import os

AUDIO_FEATURES = [
    "energy", "danceability", "valence", "tempo",
    "acousticness", "speechiness", "loudness",
    "instrumentalness", "liveness"
]

FEATURE_WEIGHTS = {
    "energy":           1.5,
    "danceability":     1.5,
    "valence":          1.2,
    "tempo":            1.0,
    "acousticness":     1.2,
    "speechiness":      0.8,
    "loudness":         0.8,
    "instrumentalness": 0.6,
    "liveness":         0.5,
}


class ContentBasedRecommender:
    def __init__(self):
        self.songs_df = None
        self.feature_matrix = None
        self.similarity_matrix = None
        self.scaler = MinMaxScaler()
        self.song_index = {}      # song_id → row index
        self.index_song = {}      # row index → song_id
        self.fitted = False

    def fit(self, songs_df: pd.DataFrame):
        self.songs_df = songs_df.reset_index(drop=True).copy()

        for i, sid in enumerate(self.songs_df["song_id"]):
            self.song_index[sid] = i
            self.index_song[i] = sid

        features = self.songs_df[AUDIO_FEATURES].copy()

        features["tempo"] = (features["tempo"] - 60) / (200 - 60)

        features["loudness"] = (features["loudness"] + 60) / 60

        scaled = self.scaler.fit_transform(features)

        weights = np.array([FEATURE_WEIGHTS[f] for f in AUDIO_FEATURES])
        self.feature_matrix = scaled * weights

        self.similarity_matrix = cosine_similarity(self.feature_matrix)
        self.fitted = True
        print(f" ContentBased fitted on {len(self.songs_df)} songs.")

    def recommend(self, song_id: int, n: int = 10, exclude_ids: list = None):
        if not self.fitted:
            raise RuntimeError("Fit the model first.")
        if song_id not in self.song_index:
            return []

        idx = self.song_index[song_id]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        exclude = set(exclude_ids or [])
        exclude.add(song_id)

        results = []
        for i, score in sim_scores:
            sid = self.index_song[i]
            if sid not in exclude:
                row = self.songs_df[self.songs_df["song_id"] == sid].iloc[0]
                results.append({
                    "song_id":      int(sid),
                    "title":        row["title"],
                    "artist":       row["artist"],
                    "genre":        row["genre"],
                    "score":        round(float(score), 4),
                    "energy":       round(float(row["energy"]), 2),
                    "danceability": round(float(row["danceability"]), 2),
                    "valence":      round(float(row["valence"]), 2),
                    "popularity":   int(row["popularity"]),
                })
            if len(results) >= n:
                break
        return results

    def recommend_for_profile(self, liked_song_ids: list, n: int = 10, exclude_ids: list = None):
        if not liked_song_ids:
            return []

        valid_ids = [sid for sid in liked_song_ids if sid in self.song_index]
        if not valid_ids:
            return []

        indices = [self.song_index[sid] for sid in valid_ids]
        profile_vec = np.mean(self.feature_matrix[indices], axis=0).reshape(1, -1)

        sim_scores = cosine_similarity(profile_vec, self.feature_matrix)[0]
        ranked = sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)

        exclude = set(exclude_ids or []) | set(liked_song_ids)
        results = []
        for i, score in ranked:
            sid = self.index_song[i]
            if sid not in exclude:
                row = self.songs_df[self.songs_df["song_id"] == sid].iloc[0]
                results.append({
                    "song_id":      int(sid),
                    "title":        row["title"],
                    "artist":       row["artist"],
                    "genre":        row["genre"],
                    "score":        round(float(score), 4),
                    "energy":       round(float(row["energy"]), 2),
                    "danceability": round(float(row["danceability"]), 2),
                    "valence":      round(float(row["valence"]), 2),
                    "popularity":   int(row["popularity"]),
                })
            if len(results) >= n:
                break
        return results

    def get_feature_vector(self, song_id: int):
        if song_id not in self.song_index:
            return {}
        row = self.songs_df[self.songs_df["song_id"] == song_id].iloc[0]
        return {f: round(float(row[f]), 4) for f in AUDIO_FEATURES}

    def save(self, path="models/cbf_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f" CBF model saved → {path}")

    @staticmethod
    def load(path="models/cbf_model.pkl"):
        return joblib.load(path)


def get_pca_data(songs_df: pd.DataFrame, n_components: int = 2):
    features = songs_df[AUDIO_FEATURES].copy()
    features["tempo"] = (features["tempo"] - 60) / 140
    features["loudness"] = (features["loudness"] + 60) / 60
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(scaled)
    result = songs_df[["song_id", "title", "artist", "genre"]].copy()
    result["x"] = coords[:, 0].round(4)
    result["y"] = coords[:, 1].round(4)
    return result.to_dict(orient="records")


if __name__ == "__main__":
    from data_generator import generate_all
    songs, _, _ = generate_all()
    cbf = ContentBasedRecommender()
    cbf.fit(songs)
    recs = cbf.recommend(1, n=5)
    print("\nTop 5 similar to 'Blinding Lights':")
    for r in recs:
        print(f"  {r['title']} — {r['artist']}  (score: {r['score']})")
    cbf.save()
