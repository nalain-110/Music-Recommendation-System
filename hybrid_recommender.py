import pandas as pd
import numpy as np
from collaborative_filter import CollaborativeFilter
from feature_extractor import ContentBasedRecommender


class HybridRecommender:
    def __init__(self, cf_weight: float = 0.6, cbf_weight: float = 0.4):
        self.cf_weight = cf_weight
        self.cbf_weight = cbf_weight
        self.cf: CollaborativeFilter = None
        self.cbf: ContentBasedRecommender = None
        self.songs_df: pd.DataFrame = None
        self.interactions_df: pd.DataFrame = None
        self.user_history: dict = {}   # user_id → list of song_ids
        self.user_ratings: dict = {}   # user_id → {song_id: rating}

    def fit(self, songs_df, users_df, interactions_df):
        self.songs_df = songs_df.copy()
        self.interactions_df = interactions_df.copy()

        for uid, grp in interactions_df.groupby("user_id"):
            self.user_history[uid] = grp["song_id"].tolist()
            self.user_ratings[uid] = dict(zip(grp["song_id"], grp["rating"]))

        self.cf = CollaborativeFilter(n_factors=20, n_neighbors=15)
        self.cf.fit(interactions_df, songs_df)

        self.cbf = ContentBasedRecommender()
        self.cbf.fit(songs_df)

        print(" HybridRecommender ready.")

    def _interaction_count(self, user_id: int) -> int:
        return len(self.user_history.get(user_id, []))

    def recommend(self, user_id: int, n: int = 10, strategy: str = "auto") -> list:
        history = self.user_history.get(user_id, [])
        n_history = len(history)

        if strategy == "auto":
            if n_history == 0:
                strategy = "cbf"
            elif n_history < 5:
                strategy = "knn_cbf"
            else:
                strategy = "hybrid"

        exclude = list(set(history))

        if strategy == "cbf":
            popular = self.songs_df.nlargest(5, "popularity")["song_id"].tolist()
            return self.cbf.recommend_for_profile(popular, n=n, exclude_ids=exclude)

        elif strategy == "svd":
            return self.cf.recommend_svd(user_id, n=n, exclude_ids=exclude)

        elif strategy == "knn":
            return self.cf.recommend_knn(user_id, n=n, exclude_ids=exclude)

        elif strategy == "knn_cbf":
            knn_recs  = self.cf.recommend_knn(user_id, n=n*2, exclude_ids=exclude)
            cbf_recs  = self.cbf.recommend_for_profile(history, n=n*2, exclude_ids=exclude)
            return self._merge(knn_recs, cbf_recs, w1=0.35, w2=0.65, n=n)

        else:
            svd_recs  = self.cf.recommend_svd(user_id, n=n*2, exclude_ids=exclude)
            cbf_recs  = self.cbf.recommend_for_profile(history, n=n*2, exclude_ids=exclude)
            cf_w  = self.cf_weight if n_history >= 20 else 0.5
            cbf_w = 1 - cf_w
            return self._merge(svd_recs, cbf_recs, w1=cf_w, w2=cbf_w, n=n)

    def _merge(self, list1: list, list2: list,
               w1: float, w2: float, n: int) -> list:
        scores = {}
        meta   = {}

        for item in list1:
            sid = item["song_id"]
            # Normalize score to [0,1] by assuming max=1
            scores[sid] = scores.get(sid, 0) + w1 * item["score"]
            meta[sid] = item

        for item in list2:
            sid = item["song_id"]
            scores[sid] = scores.get(sid, 0) + w2 * item["score"]
            if sid not in meta:
                meta[sid] = item

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Diversity: avoid same artist twice in a row
        final = []
        seen_artists = set()
        for sid, score in ranked:
            item = meta[sid].copy()
            item["score"] = round(score, 4)
            item["method"] = "Hybrid"
            artist = item.get("artist", "")
            if artist not in seen_artists or len(final) >= n - 2:
                final.append(item)
                seen_artists.add(artist)
            if len(final) >= n:
                break

        return final

    def explain(self, user_id: int, song_id: int) -> dict:
        history = self.user_history.get(user_id, [])
        if not history:
            return {"reason": "Popular track you might enjoy as a new listener."}

        liked = [sid for sid, r in self.user_ratings.get(user_id, {}).items() if r >= 4.0]
        if not liked:
            liked = history[-3:]

        best_sim = 0
        best_seed = None
        seed_title = ""
        for seed_id in liked:
            recs = self.cbf.recommend(seed_id, n=20)
            for r in recs:
                if r["song_id"] == song_id and r["score"] > best_sim:
                    best_sim = r["score"]
                    best_seed = seed_id
                    # look up title
                    row = self.songs_df[self.songs_df["song_id"] == seed_id]
                    if not row.empty:
                        seed_title = row.iloc[0]["title"]

        if best_seed:
            return {
                "reason": f"Because you liked '{seed_title}' — similar energy, mood, and style.",
                "similarity": round(best_sim, 3),
                "seed_song_id": best_seed,
            }
        return {"reason": "Matches the listening pattern of users with similar taste."}

    def record_feedback(self, user_id: int, song_id: int, rating: float):
        if user_id not in self.user_history:
            self.user_history[user_id] = []
            self.user_ratings[user_id] = {}
        if song_id not in self.user_history[user_id]:
            self.user_history[user_id].append(song_id)
        self.user_ratings[user_id][song_id] = rating

    def get_user_profile(self, user_id: int) -> dict:
        history = self.user_history.get(user_id, [])
        if not history:
            return {"history_count": 0, "top_genres": [], "avg_rating": None}

        rated = self.user_ratings.get(user_id, {})
        genres = []
        for sid in history:
            row = self.songs_df[self.songs_df["song_id"] == sid]
            if not row.empty:
                genres.append(row.iloc[0]["genre"])

        top_genres = pd.Series(genres).value_counts().head(3).index.tolist()
        avg_rating = np.mean(list(rated.values())) if rated else None

        return {
            "history_count": len(history),
            "top_genres":    top_genres,
            "avg_rating":    round(float(avg_rating), 2) if avg_rating else None,
            "liked_count":   sum(1 for r in rated.values() if r >= 4.0),
        }
