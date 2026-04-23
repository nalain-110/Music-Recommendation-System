"""
data_generator.py
Generates synthetic music dataset for the recommendation system.
Run once to create the data files.
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

# ── 50 Songs ──────────────────────────────────────────────
SONGS = [
    # (id, title, artist, genre, year)
    (1,  "Blinding Lights",      "The Weeknd",       "Pop",       2019),
    (2,  "Shape of You",         "Ed Sheeran",        "Pop",       2017),
    (3,  "Bohemian Rhapsody",    "Queen",             "Rock",      1975),
    (4,  "Lose Yourself",        "Eminem",            "Hip-Hop",   2002),
    (5,  "Uptown Funk",          "Bruno Mars",        "Funk/Pop",  2014),
    (6,  "Hotel California",     "Eagles",            "Rock",      1977),
    (7,  "Smells Like Teen Spirit","Nirvana",         "Rock",      1991),
    (8,  "Rolling in the Deep",  "Adele",             "Soul/Pop",  2010),
    (9,  "Sicko Mode",           "Travis Scott",      "Hip-Hop",   2018),
    (10, "Levitating",           "Dua Lipa",          "Pop",       2020),
    (11, "God's Plan",           "Drake",             "Hip-Hop",   2018),
    (12, "Someone Like You",     "Adele",             "Soul",      2011),
    (13, "Sweet Child O Mine",   "Guns N Roses",      "Rock",      1987),
    (14, "Sunflower",            "Post Malone",       "Hip-Hop",   2018),
    (15, "Watermelon Sugar",     "Harry Styles",      "Pop",       2019),
    (16, "Old Town Road",        "Lil Nas X",         "Country/Hip-Hop",2019),
    (17, "Bad Guy",              "Billie Eilish",     "Alt/Pop",   2019),
    (18, "Happier",              "Marshmello",        "EDM",       2018),
    (19, "Rockstar",             "Post Malone",       "Hip-Hop",   2017),
    (20, "HUMBLE.",              "Kendrick Lamar",    "Hip-Hop",   2017),
    (21, "Shallow",              "Lady Gaga",         "Pop/Country",2018),
    (22, "In My Feelings",       "Drake",             "Hip-Hop",   2018),
    (23, "SAD!",                 "XXXTENTACION",      "Hip-Hop",   2018),
    (24, "Thunder",              "Imagine Dragons",   "Pop/Rock",  2017),
    (25, "Stressed Out",         "21 Pilots",         "Alt/Pop",   2015),
    (26, "See You Again",        "Wiz Khalifa",       "Hip-Hop",   2015),
    (27, "Let Her Go",           "Passenger",         "Folk/Pop",  2012),
    (28, "Thinking Out Loud",    "Ed Sheeran",        "Pop",       2014),
    (29, "Stay With Me",         "Sam Smith",         "Soul/Pop",  2014),
    (30, "Counting Stars",       "OneRepublic",       "Pop/Rock",  2013),
    (31, "Demons",               "Imagine Dragons",   "Pop/Rock",  2012),
    (32, "Radioactive",          "Imagine Dragons",   "Rock",      2012),
    (33, "Photograph",           "Ed Sheeran",        "Pop",       2014),
    (34, "Love Yourself",        "Justin Bieber",     "Pop",       2015),
    (35, "Sorry",                "Justin Bieber",     "Pop",       2015),
    (36, "Perfect",              "Ed Sheeran",        "Pop",       2017),
    (37, "Dance Monkey",         "Tones and I",       "Indie Pop", 2019),
    (38, "Circles",              "Post Malone",       "Pop",       2019),
    (39, "Memories",             "Maroon 5",          "Pop",       2019),
    (40, "Senorita",             "Shawn Mendes",      "Pop",       2019),
    (41, "Truth Hurts",          "Lizzo",             "R&B/Pop",   2017),
    (42, "Juice",                "Lizzo",             "R&B/Pop",   2019),
    (43, "Savage",               "Megan Thee Stallion","Hip-Hop",  2020),
    (44, "Rockstar",             "DaBaby",            "Hip-Hop",   2020),
    (45, "Dynamite",             "BTS",               "K-Pop",     2020),
    (46, "Life Goes On",         "BTS",               "K-Pop",     2020),
    (47, "Butter",               "BTS",               "K-Pop",     2021),
    (48, "Peaches",              "Justin Bieber",     "R&B/Pop",   2021),
    (49, "drivers license",      "Olivia Rodrigo",    "Pop",       2021),
    (50, "good 4 u",             "Olivia Rodrigo",    "Pop/Rock",  2021),
]

GENRES = ["Pop", "Rock", "Hip-Hop", "Soul/Pop", "Funk/Pop", "EDM",
          "Alt/Pop", "R&B/Pop", "Indie Pop", "K-Pop", "Folk/Pop", "Country/Hip-Hop"]

def genre_to_features(genre):
    """Return base audio features per genre."""
    defaults = {
        "Pop":            dict(energy=0.65, danceability=0.70, valence=0.65, tempo=118, acousticness=0.25, speechiness=0.06),
        "Rock":           dict(energy=0.85, danceability=0.50, valence=0.45, tempo=130, acousticness=0.10, speechiness=0.05),
        "Hip-Hop":        dict(energy=0.75, danceability=0.80, valence=0.55, tempo=95,  acousticness=0.15, speechiness=0.25),
        "Soul/Pop":       dict(energy=0.60, danceability=0.65, valence=0.55, tempo=105, acousticness=0.35, speechiness=0.05),
        "Funk/Pop":       dict(energy=0.80, danceability=0.85, valence=0.80, tempo=115, acousticness=0.20, speechiness=0.08),
        "EDM":            dict(energy=0.90, danceability=0.85, valence=0.70, tempo=128, acousticness=0.05, speechiness=0.05),
        "Alt/Pop":        dict(energy=0.55, danceability=0.60, valence=0.40, tempo=110, acousticness=0.30, speechiness=0.10),
        "R&B/Pop":        dict(energy=0.65, danceability=0.78, valence=0.70, tempo=100, acousticness=0.25, speechiness=0.10),
        "Indie Pop":      dict(energy=0.60, danceability=0.72, valence=0.65, tempo=105, acousticness=0.30, speechiness=0.06),
        "K-Pop":          dict(energy=0.75, danceability=0.80, valence=0.72, tempo=120, acousticness=0.15, speechiness=0.08),
        "Folk/Pop":       dict(energy=0.40, danceability=0.50, valence=0.55, tempo=90,  acousticness=0.65, speechiness=0.04),
        "Country/Hip-Hop":dict(energy=0.70, danceability=0.75, valence=0.65, tempo=136, acousticness=0.20, speechiness=0.15),
    }
    # Find best matching key
    for key in defaults:
        if key in genre or genre in key:
            return defaults[key]
    return defaults["Pop"]

def generate_songs_df():
    rows = []
    for sid, title, artist, genre, year in SONGS:
        base = genre_to_features(genre)
        noise = lambda s=0.07: np.random.normal(0, s)
        rows.append({
            "song_id":      sid,
            "title":        title,
            "artist":       artist,
            "genre":        genre,
            "year":         year,
            "energy":       float(np.clip(base["energy"]       + noise(), 0, 1)),
            "danceability": float(np.clip(base["danceability"] + noise(), 0, 1)),
            "valence":      float(np.clip(base["valence"]      + noise(), 0, 1)),
            "tempo":        float(np.clip(base["tempo"]        + np.random.normal(0, 8), 60, 200)),
            "acousticness": float(np.clip(base["acousticness"] + noise(), 0, 1)),
            "speechiness":  float(np.clip(base["speechiness"]  + noise(0.04), 0, 1)),
            "loudness":     float(np.random.uniform(-12, -3)),
            "instrumentalness": float(np.clip(np.random.beta(1, 8), 0, 1)),
            "liveness":     float(np.clip(np.random.beta(2, 8), 0, 1)),
            "popularity":   int(np.clip(np.random.normal(75, 12), 40, 100)),
            "duration_ms":  int(np.random.normal(210000, 30000)),
        })
    return pd.DataFrame(rows)

def generate_users_df(n=200):
    rows = []
    taste_profiles = [
        {"fav_genre": "Pop",     "energy_pref": 0.65, "dance_pref": 0.70},
        {"fav_genre": "Rock",    "energy_pref": 0.85, "dance_pref": 0.50},
        {"fav_genre": "Hip-Hop", "energy_pref": 0.75, "dance_pref": 0.80},
        {"fav_genre": "Soul/Pop","energy_pref": 0.60, "dance_pref": 0.65},
        {"fav_genre": "EDM",     "energy_pref": 0.90, "dance_pref": 0.85},
    ]
    for i in range(1, n + 1):
        profile = taste_profiles[i % len(taste_profiles)]
        rows.append({
            "user_id":   i,
            "username":  f"user_{i:03d}",
            "fav_genre": profile["fav_genre"],
            "age_group": np.random.choice(["18-24","25-34","35-44","45+"],
                                          p=[0.35, 0.35, 0.20, 0.10]),
        })
    return pd.DataFrame(rows)

def generate_interactions(songs_df, users_df, n_interactions=3000):
    rows = []
    song_ids = songs_df["song_id"].tolist()
    user_ids = users_df["user_id"].tolist()
    genre_map = dict(zip(songs_df["song_id"], songs_df["genre"]))
    user_genre = dict(zip(users_df["user_id"], users_df["fav_genre"]))

    for _ in range(n_interactions):
        uid = np.random.choice(user_ids)
        sid = np.random.choice(song_ids)
        # Users rate songs in their genre higher
        base_rating = 3.0
        if user_genre.get(uid, "") in genre_map.get(sid, ""):
            base_rating = 4.2
        rating = float(np.clip(np.random.normal(base_rating, 0.8), 1, 5))
        rows.append({
            "user_id":    uid,
            "song_id":    sid,
            "rating":     round(rating, 1),
            "play_count": int(np.random.poisson(rating * 2)),
            "liked":      int(rating >= 4.0),
        })

    df = pd.DataFrame(rows).drop_duplicates(subset=["user_id", "song_id"])
    return df

def generate_all(output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    songs = generate_songs_df()
    users = generate_users_df()
    interactions = generate_interactions(songs, users)

    songs.to_csv(f"{output_dir}/songs.csv", index=False)
    users.to_csv(f"{output_dir}/users.csv", index=False)
    interactions.to_csv(f"{output_dir}/interactions.csv", index=False)

    print(f"✅ songs.csv          → {len(songs)} songs")
    print(f"✅ users.csv          → {len(users)} users")
    print(f"✅ interactions.csv   → {len(interactions)} interactions")
    return songs, users, interactions

if __name__ == "__main__":
    generate_all()
