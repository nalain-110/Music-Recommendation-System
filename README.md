# 🎵 Music Recommendation System

A production-grade ML-powered music recommendation system using **Hybrid Collaborative + Content-Based Filtering**.

---

## 📁 Project Structure

```
music_recommender/
├── app.py                  ← FastAPI backend (main entry point)
├── data_generator.py       ← Synthetic dataset (50 songs, 200 users, 3000 interactions)
├── feature_extractor.py    ← Audio feature extraction + Content-Based Filtering
├── collaborative_filter.py ← SVD Matrix Factorization + User-Based KNN
├── hybrid_recommender.py   ← Hybrid engine combining CF + CBF
├── evaluate.py             ← Precision@K, Recall@K, RMSE evaluation
├── requirements.txt        ← Python dependencies
├── data/                   ← Auto-generated CSV datasets
│   ├── songs.csv
│   ├── users.csv
│   └── interactions.csv
├── models/                 ← Saved model files (auto-created)
└── static/
    └── index.html          ← Frontend UI
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
uvicorn app:app --reload --port 8000
```

### 3. Open in browser
```
http://localhost:8000
```

### 4. API Docs (Swagger)
```
http://localhost:8000/docs
```

---

## 🤖 ML Algorithms

| Algorithm | Type | Use Case |
|-----------|------|----------|
| SVD (Matrix Factorization) | Collaborative | Rich user history (20+ plays) |
| User-Based KNN | Collaborative | Sparse user history (5–20 plays) |
| Cosine Similarity | Content-Based | Cold-start / new users |
| Weighted Ensemble | Hybrid | Best overall performance |

---

## 🎛️ Recommendation Strategies

| Strategy | When Used | CF Weight | CBF Weight |
|----------|-----------|-----------|------------|
| `cbf` | Cold-start user (0 history) | 0% | 100% |
| `knn_cbf` | Sparse history (< 5 songs) | 35% | 65% |
| `hybrid` | Rich history (5+ songs) | 60% | 40% |
| `svd` | Manual override | 100% | 0% |
| `auto` | Default (system decides) | Dynamic | Dynamic |

---

## 🎵 Audio Features (12 dimensions)

| Feature | Description |
|---------|-------------|
| `energy` | Intensity and activity [0–1] |
| `danceability` | Suitability for dancing [0–1] |
| `valence` | Musical positivity/mood [0–1] |
| `tempo` | BPM (beats per minute) |
| `acousticness` | Acoustic vs electric [0–1] |
| `speechiness` | Spoken word presence [0–1] |
| `loudness` | Overall loudness (dB) |
| `instrumentalness` | Vocal vs instrumental [0–1] |
| `liveness` | Audience presence [0–1] |

---

## 📡 API Endpoints

### Recommendations
```
GET  /recommend/{user_id}?strategy=auto&n=10
GET  /recommend/{user_id}/explain/{song_id}
POST /feedback   {"user_id":1, "song_id":5, "rating":4.5}
```

### Songs
```
GET  /songs                  → All songs
GET  /songs?genre=Pop        → Filter by genre
GET  /songs/{song_id}        → Song details
GET  /songs/{song_id}/similar → Similar songs (CBF)
GET  /genres                 → Genre list with counts
```

### Users
```
GET  /users/{user_id}         → User profile
GET  /users/{user_id}/history → Listening history
GET  /users/{user_id}/similar → Similar users (KNN)
```

### Analytics
```
GET  /analytics/overview      → System stats
GET  /analytics/pca           → 2D PCA coordinates
GET  /analytics/features/{id} → Audio feature radar data
```

---

## 📊 Evaluation

```bash
python evaluate.py
```

Computes:
- **Precision@K** — fraction of top-K recs that are relevant
- **Recall@K** — fraction of relevant songs found in top-K
- **F1@K** — harmonic mean of P@K and R@K
- **RMSE** — rating prediction error
- **Genre Consistency** — CBF genre accuracy
- **Artist Diversity** — recommendation variety

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| API Framework | FastAPI + Uvicorn |
| ML | Scikit-learn (SVD, KNN, Cosine Sim) |
| Data | Pandas, NumPy |
| Audio | Librosa (integration-ready) |
| Frontend | Vanilla JS + CSS (no framework needed) |
| Model Persistence | Joblib |

---

## 🧪 Example Usage

```python
from hybrid_recommender import HybridRecommender
from data_generator import generate_all

songs, users, interactions = generate_all()

rec = HybridRecommender()
rec.fit(songs, users, interactions)

# Get 10 recommendations for user 1
recs = rec.recommend(user_id=1, n=10, strategy="auto")
for r in recs:
    print(f"{r['title']} — {r['artist']}  ({r['score']:.3f})")

# Submit feedback
rec.record_feedback(user_id=1, song_id=5, rating=4.5)

# Get user profile
profile = rec.get_user_profile(user_id=1)
print(profile)
```

---

## 👨‍💻 Author

Built as a BSAI Capstone Project — Music Recommendation System using ML.
