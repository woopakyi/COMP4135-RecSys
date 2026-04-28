# COMP4135 RecSys – Movie Recommendation System

A Flask-based movie recommender web application built for COMP4135. It compares two recommendation algorithms (FM and SASRec) across two UI themes, with a full evaluation and feedback pipeline.

**Group Members:** Chow Tsz Hin (23228660), Woo Pak Yi (23229578), Cheung Yui Haang (21223270)

**Live Demo:** https://comp4135-recsys.up.railway.app

---

## Project Overview

The system recommends movies to users through five sections on the home page:

- **Movies based on your Genre Ratings** – genre-preference-driven suggestions
- **Recommended** – personalised algorithm-driven recommendations
- **Liked with Similar Items** – content-similar neighbours of liked movies
- **Liked** – movies the user has rated positively
- **Disliked** – movies the user has rated negatively

Users can choose between two recommendation algorithms and two UI themes at any time from the home or profile page.

---

## Algorithms

### Algorithm 1 – Factorization Machine (FM)
FM combines collaborative filtering (community ratings from `ratings.csv`) with content-based features (title, genre, year, overview from `movie_info.csv`). It handles the cold-start problem well, making it effective when users first arrive and set genre preferences.

Two variants were trained (Model A without TF-IDF, Model B with TF-IDF on overviews). Model A outperformed Model B and its weights (`fm_weights_A.pt`) are used in production.

| Model | Val RMSE | P@10 | R@10 | nDCG@10 | Composite |
|---|---|---|---|---|---|
| Model A (no TF-IDF) | 0.9049 | 0.6002 | 0.7134 | 0.8400 | 0.7741 |
| Model B (with TF-IDF) | 0.9886 | 0.5827 | 0.7031 | 0.8126 | 0.7512 |

### Algorithm 2 – SASRec (Self-Attention Sequential Recommendation)
SASRec is a Transformer-based sequential model that predicts the next movie a user would enjoy based on their watch history. It was benchmarked against GRU4Rec and outperformed it on the composite metric (HR@10 + nDCG@10). Weights are stored in `sasrec_weights.pt` / `best_sasrec.pt`.

| Model | Val Loss | HR@10 | nDCG@10 | Composite |
|---|---|---|---|---|
| SASRec | 0.6223 | 0.8412 | 0.3328 | 0.5108 |
| GRU4Rec | 0.5918 | 0.6722 | 0.4031 | 0.4973 |

---

## Pages

| Page | Description |
|---|---|
| **Home** | Genre/algorithm setup and movie recommendations |
| **Profile** | View/edit preferences, manage rated movies, feedback progress |
| **Feedback** | Submit participant or anonymous evaluations (4 feedback types) |
| **Evaluation** | Charts of aggregated feedback; admin-only rating details |
| **Login** | Email or Google OAuth sign-in |
| **Register** | Create an account with email |

---

## User Interfaces

- **UI 1 – Professional Dark** (default)
- **UI 2 – Concise White**

Movies are displayed in one row (UI 1) or two rows (UI 2).

---

## Data Storage

| State | Storage |
|---|---|
| Logged-out user | Browser Local Storage |
| Logged-in user | PostgreSQL (Railway) |

Preferences set before registration are automatically inherited into the user account upon sign-up.

---

## Tech Stack

- **Backend:** Flask 3, SQLAlchemy, PostgreSQL / SQLite
- **ML:** PyTorch (FM, SASRec), scikit-learn, scikit-surprise
- **Auth:** Session-based + Google OAuth
- **Deployment:** Docker (multi-stage), Gunicorn, Railway

---

## Local Setup

### 1. Create and activate a conda environment

```bash
conda create -n recsys python=3.10
conda activate recsys
```

### 2. Install dependencies

```bash
pip install --upgrade setuptools wheel pyquery
conda install -c conda-forge scikit-surprise
pip install -r requirements.txt
```

### 3. Run the development server

```bash
flask --app flaskr run --debug
```

The app will be available at `http://127.0.0.1:5000`.

---

## Docker

```bash
docker build -t recsys .
docker run -p 8080:8080 recsys
```

---

## Project Structure

```
flaskr/
├── __init__.py         # App factory, DB init, seeding
├── main.py             # Home/profile/feedback routes & recommendation dispatch
├── auth.py             # Register, login, Google OAuth, account merge
├── api.py              # REST API (ratings, genres, feedback, profile)
├── algo1.py            # Factorization Machine inference
├── algo2.py            # SASRec inference
├── models.py           # SQLAlchemy models (User, Movie, Rating, Feedback)
├── scrape.py           # Utility to re-scrape movie cover images
├── best_sasrec.pt      # Best SASRec checkpoint
├── static/
│   ├── ml_data/        # Dataset CSVs and model weights
│   │   ├── movie_info.csv
│   │   ├── genre.csv
│   │   ├── ratings.csv
│   │   ├── fm_weights_A.pt
│   │   ├── fm_weights_B.pt
│   │   └── sasrec_weights.pt
│   ├── css/
│   └── js/
├── templates/          # Jinja2 HTML templates
└── tools/
    ├── data_tool.py    # CSV loading helpers
    └── scrape_tool.py  # IMDb scraping utilities
```

---

## Dataset

Path: `./flaskr/static/ml_data/`

**ratings.csv** columns:
- `userId` – user ID
- `movieId` – movie ID
- `rating` – rating on a 5-star scale
- `timestamp` – seconds since epoch (Unix time)

**movie_info.csv** columns: `movieId`, `title`, `year`, `overview`, `cover_url`, `genres`

**genre.csv** – pipe-delimited list of genre names and IDs

```python
import pandas as pd
timestamp = 1717665888
dt_str = pd.to_datetime(timestamp, unit='s').strftime('%Y-%m-%d %H:%M:%S')
print(dt_str)  # 2024-06-06 09:24:48
```

---

## Environment Variables (Production)

| Variable | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string (Railway injects this) |
| `SECRET_KEY` | Flask session secret |
| `GOOGLE_CLIENT_ID` | Google OAuth client ID |
| `GOOGLE_CLIENT_SECRET` | Google OAuth client secret |
| `GOOGLE_REDIRECT_URI` | OAuth redirect URI |
