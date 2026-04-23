#  Music Recommendation System

> An intelligent music recommender built with Python & ML — suggests songs based on user listening history, audio features, and behavioral patterns using SVD Matrix Factorization, KNN, and Cosine Similarity in a Hybrid ensemble.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![ML](https://img.shields.io/badge/ML-Scikit--learn-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)
---

## Project Structure

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


## ML Algorithms

| Algorithm | Type | Use Case |
|-----------|------|----------|
| SVD (Matrix Factorization) | Collaborative | Rich user history (20+ plays) |
| User-Based KNN | Collaborative | Sparse user history (5–20 plays) |
| Cosine Similarity | Content-Based | Cold-start / new users |
| Weighted Ensemble | Hybrid | Best overall performance |

---

## Recommendation Strategies

| Strategy | When Used | CF Weight | CBF Weight |
|----------|-----------|-----------|------------|
| `cbf` | Cold-start user (0 history) | 0% | 100% |
| `knn_cbf` | Sparse history (< 5 songs) | 35% | 65% |
| `hybrid` | Rich history (5+ songs) | 60% | 40% |
| `svd` | Manual override | 100% | 0% |
| `auto` | Default (system decides) | Dynamic | Dynamic |

---

## Audio Features (12 dimensions)

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

### Songs
```
GET  /songs                  → All songs
GET  /songs?genre=Pop        → Filter by genre
GET  /songs/{song_id}        → Song details
GET  /songs/{song_id}/similar → Similar songs (CBF)
GET  /genres                 → Genre list with counts
```
---

## Author

Built as a BSAI Capstone Project — Music Recommendation System using ML.
