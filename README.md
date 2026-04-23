#  Music Recommendation System

> An intelligent music recommender built with Python & ML — suggests songs based on user listening history, audio features, and behavioral patterns using SVD Matrix Factorization, KNN, and Cosine Similarity in a Hybrid ensemble.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![ML](https://img.shields.io/badge/ML-Scikit--learn-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)


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
## 3. Open in browser

http://localhost:8000

### Songs
```
GET  /songs                  → All songs
GET  /songs?genre=Pop        → Filter by genre
GET  /songs/{song_id}        → Song details
GET  /songs/{song_id}/similar → Similar songs (CBF)
GET  /genres                 → Genre list with counts
```
