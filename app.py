from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import numpy as np
import os

# ── Absolute base directory (works no matter where you run from) ──
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")
UI_FILE    = os.path.join(STATIC_DIR, "index.html")

# ── Internal modules ──────────────────────────────────────────────
from data_generator import generate_all
from feature_extractor import ContentBasedRecommender, get_pca_data
from collaborative_filter import CollaborativeFilter
from hybrid_recommender import HybridRecommender

# ── Global state ──────────────────────────────────────────────────
DATA  = {}
MODEL = {}


# ══════════════════════════════════════════════════════════════════
# Lifespan — startup & shutdown
# ══════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🎵 Music Recommendation System starting up...")
    os.makedirs(DATA_DIR, exist_ok=True)

    songs_path = os.path.join(DATA_DIR, "songs.csv")
    if not os.path.exists(songs_path):
        songs, users, interactions = generate_all(DATA_DIR)
    else:
        songs        = pd.read_csv(os.path.join(DATA_DIR, "songs.csv"))
        users        = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
        interactions = pd.read_csv(os.path.join(DATA_DIR, "interactions.csv"))

    DATA["songs"]        = songs
    DATA["users"]        = users
    DATA["interactions"] = interactions

    hybrid = HybridRecommender(cf_weight=0.6, cbf_weight=0.4)
    hybrid.fit(songs, users, interactions)
    MODEL["hybrid"] = hybrid
    MODEL["cbf"]    = hybrid.cbf
    MODEL["cf"]     = hybrid.cf

    print("🚀 All models ready!\n")
    yield

    DATA.clear()
    MODEL.clear()
    print("👋 Shut down cleanly.")


# ══════════════════════════════════════════════════════════════════
# App
# ══════════════════════════════════════════════════════════════════
app = FastAPI(
    title="🎵 Music Recommendation System",
    description="ML-powered music recommender — Hybrid CF + CBF.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════
# Schemas
# ══════════════════════════════════════════════════════════════════
class FeedbackRequest(BaseModel):
    user_id: int
    song_id: int
    rating:  float = Field(..., ge=1.0, le=5.0)


# ══════════════════════════════════════════════════════════════════
# UI — absolute path, never fails due to CWD issues
# ══════════════════════════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse, tags=["UI"])
def serve_ui():
    if not os.path.exists(UI_FILE):
        return HTMLResponse(
            f"<h2 style='font-family:sans-serif;padding:40px'>"
            f"UI file not found at:<br><code>{UI_FILE}</code></h2>",
            status_code=500,
        )
    with open(UI_FILE, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


# ══════════════════════════════════════════════════════════════════
# Health
# ══════════════════════════════════════════════════════════════════
@app.get("/health", tags=["System"])
def health():
    return {
        "status":        "ok",
        "ui_file_found": os.path.exists(UI_FILE),
        "songs":         len(DATA.get("songs", [])),
        "users":         len(DATA.get("users", [])),
        "interactions":  len(DATA.get("interactions", [])),
        "models_loaded": list(MODEL.keys()),
    }


# ══════════════════════════════════════════════════════════════════
# Songs
# ══════════════════════════════════════════════════════════════════
@app.get("/songs", tags=["Songs"])
def get_songs(genre: Optional[str] = None,
              limit: int = Query(default=50, le=200),
              offset: int = 0):
    df = DATA["songs"].copy()
    if genre:
        df = df[df["genre"].str.lower().str.contains(genre.lower())]
    return df.iloc[offset: offset + limit].to_dict(orient="records")


@app.get("/songs/{song_id}", tags=["Songs"])
def get_song(song_id: int):
    row = DATA["songs"][DATA["songs"]["song_id"] == song_id]
    if row.empty:
        raise HTTPException(404, "Song not found")
    return row.iloc[0].to_dict()


@app.get("/songs/{song_id}/similar", tags=["Songs"])
def similar_songs(song_id: int, n: int = 10):
    recs = MODEL["cbf"].recommend(song_id, n=n)
    if not recs:
        raise HTTPException(404, "Song not found or no similar songs.")
    return {"song_id": song_id, "recommendations": recs}


@app.get("/genres", tags=["Songs"])
def get_genres():
    counts = DATA["songs"]["genre"].value_counts().reset_index()
    counts.columns = ["genre", "count"]
    return counts.to_dict(orient="records")


# ══════════════════════════════════════════════════════════════════
# Users
# ══════════════════════════════════════════════════════════════════
@app.get("/users", tags=["Users"])
def get_users(limit: int = 20, offset: int = 0):
    return DATA["users"].iloc[offset: offset + limit].to_dict(orient="records")


@app.get("/users/{user_id}", tags=["Users"])
def get_user(user_id: int):
    row = DATA["users"][DATA["users"]["user_id"] == user_id]
    if row.empty:
        raise HTTPException(404, "User not found")
    user = row.iloc[0].to_dict()
    user.update(MODEL["hybrid"].get_user_profile(user_id))
    return user


@app.get("/users/{user_id}/history", tags=["Users"])
def user_history(user_id: int):
    history_ids = MODEL["hybrid"].user_history.get(user_id, [])
    ratings     = MODEL["hybrid"].user_ratings.get(user_id, {})
    songs_df    = DATA["songs"]
    result = []
    for sid in history_ids:
        row = songs_df[songs_df["song_id"] == sid]
        if not row.empty:
            item = row.iloc[0].to_dict()
            item["user_rating"] = ratings.get(sid)
            result.append(item)
    return {"user_id": user_id, "history": result}


@app.get("/users/{user_id}/similar", tags=["Users"])
def similar_users(user_id: int, n: int = 5):
    return {"user_id": user_id,
            "similar_users": MODEL["cf"].get_similar_users(user_id, n=n)}


# ══════════════════════════════════════════════════════════════════
# Recommendations
# ══════════════════════════════════════════════════════════════════
@app.get("/recommend/{user_id}", tags=["Recommendations"])
def recommend(user_id: int,
              n:        int = Query(default=10, ge=1, le=50),
              strategy: str = Query(default="auto")):
    recs    = MODEL["hybrid"].recommend(user_id, n=n, strategy=strategy)
    profile = MODEL["hybrid"].get_user_profile(user_id)
    return {"user_id": user_id, "strategy": strategy,
            "profile": profile, "count": len(recs),
            "recommendations": recs}


@app.get("/recommend/{user_id}/explain/{song_id}", tags=["Recommendations"])
def explain_recommendation(user_id: int, song_id: int):
    exp = MODEL["hybrid"].explain(user_id, song_id)
    return {"user_id": user_id, "song_id": song_id, **exp}


@app.post("/feedback", tags=["Recommendations"])
def submit_feedback(req: FeedbackRequest):
    MODEL["hybrid"].record_feedback(req.user_id, req.song_id, req.rating)
    return {"status": "ok",
            "message": f"Feedback saved for user {req.user_id}, song {req.song_id}.",
            "updated_profile": MODEL["hybrid"].get_user_profile(req.user_id)}


# ══════════════════════════════════════════════════════════════════
# Analytics
# ══════════════════════════════════════════════════════════════════
@app.get("/analytics/overview", tags=["Analytics"])
def analytics_overview():
    songs        = DATA["songs"]
    interactions = DATA["interactions"]
    top_songs = (
        interactions.groupby("song_id")["rating"].mean()
        .nlargest(10).reset_index()
        .merge(songs[["song_id","title","artist","genre"]], on="song_id")
        .rename(columns={"rating": "avg_rating"})
        .to_dict(orient="records")
    )
    rating_dist = (
        interactions["rating"].apply(lambda x: round(x))
        .value_counts().sort_index().to_dict()
    )
    return {
        "total_songs":         len(songs),
        "total_users":         len(DATA["users"]),
        "total_interactions":  len(interactions),
        "avg_rating":          round(float(interactions["rating"].mean()), 2),
        "genre_distribution":  songs["genre"].value_counts().to_dict(),
        "top_songs":           top_songs,
        "rating_distribution": rating_dist,
    }


@app.get("/analytics/pca", tags=["Analytics"])
def pca_visualization():
    return {"points": get_pca_data(DATA["songs"])}


@app.get("/analytics/features/{song_id}", tags=["Analytics"])
def song_features(song_id: int):
    vec = MODEL["cbf"].get_feature_vector(song_id)
    if not vec:
        raise HTTPException(404, "Song not found")
    return {"song_id": song_id, "features": vec}


# ══════════════════════════════════════════════════════════════════
# Static files (CSS/JS assets)
# ══════════════════════════════════════════════════════════════════
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ══════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
