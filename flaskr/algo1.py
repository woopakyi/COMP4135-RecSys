import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


TOP_K = 12
USER_ID_FALLBACK = 611

_CACHE = {
    'ready': False,
    'error': None,
    'model': None,
    'num_features': 0,
    'n_users': 0,
    'n_movies': 0,
    'genres': [],
    'user_id_to_idx': {},
    'movie_id_to_idx': {},
    'movie_features': {},
    'movie_content': {},
    'ts_min': 0,
    'ts_max': 1,
    'needs_tfidf': False,
    'tfidf_start': 0,
    'movie_tfidf': {},
    'default_model_path': None,
}


class FM(nn.Module):
    def __init__(self, num_features, k):
        super().__init__()
        self.w0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.zeros(num_features))
        self.V = nn.Parameter(torch.randn(num_features, k) * 0.01)

    def forward(self, x):
        linear = self.w0 + x @ self.w
        x_v = x @ self.V
        interaction = 0.5 * (x_v.pow(2).sum(1) - (x.pow(2) @ self.V.pow(2)).sum(1))
        return linear + interaction


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _normalize_rating_to_5_scale(raw_rating):
    clamped = max(1, min(10, int(raw_rating)))
    return round(clamped / 2, 1)


def _safe_timestamp(raw_value):
    try:
        value = int(raw_value)
        return value if value > 0 else None
    except (TypeError, ValueError):
        return None


def _parse_user_rates(user_rates):
    rows = []
    for item in user_rates:
        parts = item.split('|')
        if len(parts) < 3:
            continue
        try:
            user_id = int(parts[0])
            movie_id = int(parts[1])
            rating_10 = int(parts[2])
        except ValueError:
            continue

        ts = _safe_timestamp(parts[3]) if len(parts) > 3 else None
        rows.append({
            'userId': user_id,
            'movieId': movie_id,
            'rating10': rating_10,
            'rating5': _normalize_rating_to_5_scale(rating_10),
            'timestamp': ts,
        })

    if not rows:
        return pd.DataFrame(columns=['userId', 'movieId', 'rating10', 'rating5', 'timestamp'])
    return pd.DataFrame(rows)


def _is_genre_match(movie_genres, interested_genres):
    if not movie_genres:
        return False
    return bool(set(movie_genres).intersection(set(interested_genres)))


def _fit_tfidf_by_movie(movies_df):
    overview_series = movies_df['overview'].fillna('').astype(str) if 'overview' in movies_df.columns else pd.Series([''] * len(movies_df))
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    matrix = vectorizer.fit_transform(overview_series)

    movie_tfidf = {}
    for i, mid in enumerate(movies_df['movieId'].astype(int).tolist()):
        movie_tfidf[mid] = matrix.getrow(i).toarray().ravel().astype(np.float32)
    return movie_tfidf, matrix.shape[1]


def _init_cache(movies_df, rates_df):
    if _CACHE['ready']:
        return

    try:
        root = _repo_root()
        ml_dir = os.path.join(root, 'flaskr', 'static', 'ml_data')
        weights_a = os.path.join(ml_dir, 'fm_weights_A.pt')
        weights_b = os.path.join(ml_dir, 'fm_weights_B.pt')

        model_path = weights_a if os.path.exists(weights_a) else weights_b if os.path.exists(weights_b) else None
        if model_path is None:
            raise FileNotFoundError('No FM weight file found in flaskr/static/ml_data')

        state_dict = torch.load(model_path, map_location='cpu')
        num_features = int(state_dict['w'].shape[0])
        k = int(state_dict['V'].shape[1])

        model = FM(num_features=num_features, k=k)
        model.load_state_dict(state_dict)
        model.eval()

        user_ids = sorted(rates_df['userId'].astype(int).unique().tolist())
        movie_ids = sorted(rates_df['movieId'].astype(int).unique().tolist())
        user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
        movie_id_to_idx = {mid: i for i, mid in enumerate(movie_ids)}

        genre_path = os.path.join(ml_dir, 'genre.csv')
        if os.path.exists(genre_path):
            genres_df = pd.read_csv(genre_path, delimiter='|', names=['name', 'id'])
            all_genres = genres_df['name'].astype(str).tolist()
        else:
            unique_genres = set()
            for gs in movies_df['genres']:
                if isinstance(gs, list):
                    unique_genres.update(gs)
            all_genres = sorted(unique_genres)

        year_series = pd.to_numeric(movies_df['year'], errors='coerce')
        year_min = float(year_series.min()) if year_series.notna().any() else 0.0
        year_max = float(year_series.max()) if year_series.notna().any() else 1.0
        year_med = float(year_series.median()) if year_series.notna().any() else 0.0
        year_den = (year_max - year_min) if year_max > year_min else 1.0

        movie_features = {}
        movie_content = {}
        for _, row in movies_df.iterrows():
            mid = int(row['movieId'])
            movie_genres = row['genres'] if isinstance(row['genres'], list) else []
            genre_set = set(movie_genres)
            genre_vec = np.array([1.0 if g in genre_set else 0.0 for g in all_genres], dtype=np.float32)

            y = row['year'] if pd.notna(row['year']) else year_med
            year_norm = float((float(y) - year_min) / year_den)
            year_norm = min(1.0, max(0.0, year_norm))

            movie_features[mid] = (genre_vec, year_norm)
            movie_content[mid] = np.concatenate([genre_vec, np.array([year_norm], dtype=np.float32)])

        ts_path = os.path.join(ml_dir, 'ratings.csv')
        ts_min, ts_max = 0, 1
        if os.path.exists(ts_path):
            ts_df = pd.read_csv(ts_path, usecols=['timestamp'])
            if not ts_df.empty:
                ts_min = int(ts_df['timestamp'].min())
                ts_max = int(ts_df['timestamp'].max())
                if ts_max <= ts_min:
                    ts_max = ts_min + 1

        base_features = len(user_id_to_idx) + len(movie_id_to_idx) + len(all_genres) + 2
        needs_tfidf = False
        movie_tfidf = {}
        tfidf_start = 0

        if num_features == base_features + 500:
            movie_tfidf, tfidf_dim = _fit_tfidf_by_movie(movies_df)
            if tfidf_dim != 500:
                raise ValueError('Expected TF-IDF dimension 500 for Model B')
            needs_tfidf = True
            tfidf_start = base_features
        elif num_features != base_features:
            raise ValueError(f'Feature mismatch: model expects {num_features}, reconstructed {base_features}')

        _CACHE.update({
            'ready': True,
            'error': None,
            'model': model,
            'num_features': num_features,
            'n_users': len(user_id_to_idx),
            'n_movies': len(movie_id_to_idx),
            'genres': all_genres,
            'user_id_to_idx': user_id_to_idx,
            'movie_id_to_idx': movie_id_to_idx,
            'movie_features': movie_features,
            'movie_content': movie_content,
            'ts_min': ts_min,
            'ts_max': ts_max,
            'needs_tfidf': needs_tfidf,
            'tfidf_start': tfidf_start,
            'movie_tfidf': movie_tfidf,
            'default_model_path': model_path,
        })
    except Exception as exc:
        _CACHE['error'] = str(exc)
        _CACHE['ready'] = False


def _build_features_for_movies(candidate_movie_ids, user_id, user_rates_df):
    n = len(candidate_movie_ids)
    if n == 0:
        return np.empty((0, _CACHE['num_features']), dtype=np.float32)

    x = np.zeros((n, _CACHE['num_features']), dtype=np.float32)

    user_idx = _CACHE['user_id_to_idx'].get(int(user_id))
    if user_idx is None:
        user_idx = _CACHE['user_id_to_idx'].get(USER_ID_FALLBACK)
    if user_idx is not None:
        x[:, user_idx] = 1.0

    movie_start = _CACHE['n_users']
    genre_start = movie_start + _CACHE['n_movies']
    genre_end = genre_start + len(_CACHE['genres'])
    year_idx = genre_end
    ts_idx = genre_end + 1

    latest_ts = user_rates_df['timestamp'].dropna().max() if not user_rates_df.empty else None
    if pd.isna(latest_ts):
        latest_ts = None
    if latest_ts is None:
        latest_ts = _CACHE['ts_max']
    ts_norm = float((int(latest_ts) - _CACHE['ts_min']) / (_CACHE['ts_max'] - _CACHE['ts_min']))
    ts_norm = min(1.0, max(0.0, ts_norm))
    x[:, ts_idx] = ts_norm

    zero_genres = np.zeros(len(_CACHE['genres']), dtype=np.float32)
    zero_tfidf = np.zeros(500, dtype=np.float32) if _CACHE['needs_tfidf'] else None

    for i, mid in enumerate(candidate_movie_ids):
        movie_idx = _CACHE['movie_id_to_idx'].get(int(mid))
        if movie_idx is not None:
            x[i, movie_start + movie_idx] = 1.0

        genre_vec, year_norm = _CACHE['movie_features'].get(int(mid), (zero_genres, 0.0))
        x[i, genre_start:genre_end] = genre_vec
        x[i, year_idx] = year_norm

        if _CACHE['needs_tfidf']:
            tfidf_vec = _CACHE['movie_tfidf'].get(int(mid), zero_tfidf)
            x[i, _CACHE['tfidf_start']:] = tfidf_vec

    return x


def _fm_predict(x_np, batch_size=1024):
    if x_np.shape[0] == 0:
        return np.array([], dtype=np.float32)

    model = _CACHE['model']
    preds = []
    with torch.no_grad():
        for start in range(0, x_np.shape[0], batch_size):
            batch = torch.from_numpy(x_np[start:start + batch_size])
            out = model(batch).cpu().numpy()
            preds.append(out)
    return np.concatenate(preds).astype(np.float32)


def _content_boost(candidate_movie_ids, user_rates_df):
    if user_rates_df.empty:
        return np.zeros(len(candidate_movie_ids), dtype=np.float32)

    rated = user_rates_df[['movieId', 'rating5']].drop_duplicates('movieId')
    vectors = []
    weights = []
    for _, row in rated.iterrows():
        mid = int(row['movieId'])
        vec = _CACHE['movie_content'].get(mid)
        if vec is None:
            continue
        w = float(row['rating5']) - 3.0
        if abs(w) < 1e-6:
            continue
        vectors.append(vec)
        weights.append(w)

    if not vectors:
        return np.zeros(len(candidate_movie_ids), dtype=np.float32)

    mat = np.vstack(vectors)
    profile = (mat.T @ np.array(weights, dtype=np.float32)).astype(np.float32)
    norm = float(np.linalg.norm(profile))
    if norm < 1e-8:
        return np.zeros(len(candidate_movie_ids), dtype=np.float32)
    profile = profile / norm

    candidate_vecs = []
    for mid in candidate_movie_ids:
        candidate_vecs.append(_CACHE['movie_content'].get(int(mid), np.zeros_like(profile)))
    cand_mat = np.vstack(candidate_vecs)
    cand_norm = np.linalg.norm(cand_mat, axis=1)
    scores = (cand_mat @ profile) / (cand_norm + 1e-8)
    return scores.astype(np.float32)


def _sort_movies_by_ids(movies_df, ordered_ids):
    if not ordered_ids:
        return pd.DataFrame(columns=movies_df.columns)
    order_map = {mid: i for i, mid in enumerate(ordered_ids)}
    subset = movies_df[movies_df['movieId'].isin(ordered_ids)].copy()
    subset['__rank'] = subset['movieId'].map(order_map)
    subset = subset.sort_values('__rank').drop(columns='__rank')
    return subset

# - Movies based on your Genre Ratings
def getMoviesByGenres(user_genres, movies_df, genres_df):
    """Movies based on your Genre Ratings section."""
    results = []
    if len(user_genres) > 0:
        genres_mask = genres_df['id'].isin([int(gid) for gid in user_genres])
        user_genres_mask = [1 if has is True else 0 for has in genres_mask]
        user_genres_df = pd.DataFrame(user_genres_mask, columns=['value'])
        user_genres_df = pd.concat([user_genres_df, genres_df['name']], axis=1)
        interested_genres = user_genres_df[user_genres_df['value'] == 1]['name'].tolist()
        results = movies_df[movies_df['genres'].apply(lambda x: _is_genre_match(x, interested_genres))]

    if len(results) > 0:
        return results.to_dict('records')
    return results

# - Recommended
def getRecommendationBy(user_rates, movies_df, rates_df):
    results = []
    if len(user_rates) == 0:
        return results, 'No recommendations.'

    _init_cache(movies_df, rates_df)
    if not _CACHE['ready']:
        return results, f"No recommendations. (FM unavailable: {_CACHE.get('error', 'unknown error')})"

    user_rates_df = _parse_user_rates(user_rates)
    if user_rates_df.empty:
        return results, 'No recommendations.'

    user_id = int(user_rates_df['userId'].iloc[0])
    rated_ids = set(user_rates_df['movieId'].astype(int).tolist())
    candidate_ids = [int(mid) for mid in movies_df['movieId'].astype(int).tolist() if int(mid) not in rated_ids]
    if not candidate_ids:
        return results, 'No recommendations.'

    x = _build_features_for_movies(candidate_ids, user_id, user_rates_df)
    fm_scores = _fm_predict(x)

    boost_scores = _content_boost(candidate_ids, user_rates_df)
    fm_norm = (fm_scores - fm_scores.mean()) / (fm_scores.std() + 1e-8)
    final_scores = fm_norm + 0.35 * boost_scores

    top_idx = np.argsort(-final_scores)[:TOP_K]
    top_movie_ids = [candidate_ids[i] for i in top_idx]
    results = _sort_movies_by_ids(movies_df, top_movie_ids)

    if len(results) > 0:
        return results.to_dict('records'), 'These movies are recommended based on your ratings.'
    return results, 'No recommendations.'

# - Liked with Similar Items
def getLikedSimilarBy(user_likes, movies_df):
    """Liked with Similar Items section based on genre profile similarity."""
    results = []
    if len(user_likes) > 0:
        item_rep_matrix, item_rep_vector, feature_list = item_representation_based_movie_genres(movies_df)
        user_profile = build_user_profile(user_likes, item_rep_vector, feature_list)
        results = generate_recommendation_results(user_profile, item_rep_matrix, item_rep_vector, 12)
    if len(results) > 0:
        return results.to_dict('records'), "The movies are similar to your liked movies."
    return results, "No similar movies found."


def item_representation_based_movie_genres(movies_df):
    movies_with_genres = movies_df.copy(deep=True)
    genre_list = []
    for index, row in movies_df.iterrows():
        for genre in row['genres']:
            movies_with_genres.at[index, genre] = 1
            if genre not in genre_list:
                genre_list.append(genre)

    movies_with_genres = movies_with_genres.fillna(0)
    movies_genre_matrix = movies_with_genres[genre_list].to_numpy()
    return movies_genre_matrix, movies_with_genres, genre_list


def build_user_profile(movie_ids, item_rep_vector, feature_list, normalized=True):
    user_movie_rating_df = item_rep_vector[item_rep_vector['movieId'].isin(movie_ids)]
    user_movie_df = user_movie_rating_df[feature_list].mean()
    user_profile = user_movie_df.T

    if normalized and sum(user_profile.values) > 0:
        user_profile = user_profile / sum(user_profile.values)

    return user_profile


def generate_recommendation_results(user_profile, item_rep_matrix, movies_data, k=12):
    u_v = user_profile.values
    u_v_matrix = [u_v]
    recommendation_table = cosine_similarity(u_v_matrix, item_rep_matrix)
    recommendation_table_df = movies_data.copy(deep=True)
    recommendation_table_df['similarity'] = recommendation_table[0]
    rec_result = recommendation_table_df.sort_values(by=['similarity'], ascending=False)[:k]
    return rec_result

# - Liked
def get_liked_movie_ids(user_rates):
    """Liked section source IDs (ratings >= 8 by current policy)."""
    liked_ids = []
    for rate in user_rates:
        parts = rate.split('|')
        if len(parts) < 3:
            continue
        movie_id = int(parts[1])
        rating = int(parts[2])
        if rating >= 8:
            liked_ids.append(movie_id)
    return liked_ids


def get_disliked_movie_ids(user_rates):
    """Disliked section source IDs (ratings between 1 and 3 inclusive)."""
    disliked_ids = []
    for rate in user_rates:
        parts = rate.split('|')
        if len(parts) < 3:
            continue
        movie_id = int(parts[1])
        rating = int(parts[2])
        if 1 <= rating <= 3:
            disliked_ids.append(movie_id)
    return disliked_ids
