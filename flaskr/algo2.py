import os
import math
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .main import normalize_rating_to_legacy_scale
from .main import item_representation_based_movie_genres
from .main import build_user_profile
from .main import generate_recommendation_results
from .main import is_genre_match
from .tools.data_tool import ratesFromUser


# ── SASRec model definition (must match training in algo2.ipynb) ─────────────
class _PointWiseFeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.fc1     = nn.Linear(d_model, d_model * 4)
        self.fc2     = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act     = nn.GELU()

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class _SASRecBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads,
                                          dropout=dropout, batch_first=True)
        self.ffn  = _PointWiseFeedForward(d_model, dropout)
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        residual = x
        x = self.ln1(x)
        x, _ = self.attn(x, x, x, attn_mask=attn_mask,
                         key_padding_mask=key_padding_mask, need_weights=False)
        x = self.drop(x) + residual
        residual = x
        x = self.ln2(x)
        x = self.ffn(x) + residual
        return x


class _SASRec(nn.Module):
    def __init__(self, num_items, d_model, num_heads, num_blocks,
                 max_seq_len, dropout=0.2):
        super().__init__()
        self.item_emb    = nn.Embedding(num_items, d_model, padding_idx=0)
        self.pos_emb     = nn.Embedding(max_seq_len + 1, d_model)
        self.dropout     = nn.Dropout(dropout)
        self.blocks      = nn.ModuleList([
            _SASRecBlock(d_model, num_heads, dropout) for _ in range(num_blocks)
        ])
        self.ln_out      = nn.LayerNorm(d_model)
        self.max_seq_len = max_seq_len

    def forward(self, seq):
        B, L = seq.shape
        positions = torch.arange(1, L + 1, device=seq.device).unsqueeze(0).expand(B, -1)
        x = self.dropout(self.item_emb(seq) + self.pos_emb(positions))
        causal_mask = torch.triu(
            torch.ones(L, L, device=seq.device, dtype=torch.bool), diagonal=1
        )
        pad_mask = (seq == 0)
        for block in self.blocks:
            x = block(x, attn_mask=causal_mask, key_padding_mask=pad_mask)
        return self.ln_out(x)


# ── Lazy-load cache ──────────────────────────────────────────────────────────
_SASREC_CACHE = {'ready': False, 'model': None, 'item_enc': None, 'cfg': None}

def _ml_data_dir():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'flaskr', 'static', 'ml_data')

def _load_sasrec():
    if _SASREC_CACHE['ready']:
        return True
    try:
        ml_dir = _ml_data_dir()
        cfg_path  = os.path.join(ml_dir, 'sasrec_config.pkl')
        enc_path  = os.path.join(ml_dir, 'sasrec_item_enc.pkl')
        wts_path  = os.path.join(ml_dir, 'sasrec_weights.pt')
        for p in (cfg_path, enc_path, wts_path):
            if not os.path.exists(p):
                return False
        with open(cfg_path, 'rb') as f:
            cfg = pickle.load(f)
        with open(enc_path, 'rb') as f:
            item_enc = pickle.load(f)
        model = _SASRec(**cfg)
        model.load_state_dict(torch.load(wts_path, map_location='cpu'))
        model.eval()
        _SASREC_CACHE['model']    = model
        _SASREC_CACHE['item_enc'] = item_enc
        _SASREC_CACHE['cfg']      = cfg
        _SASREC_CACHE['ready']    = True
        return True
    except Exception as e:
        print(f'[algo2] SASRec load error: {e}')
        return False


def _pad_or_truncate(seq, max_len, pad_val=0):
    seq = list(seq)
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [pad_val] * (max_len - len(seq)) + seq


def _predict_for_user(movie_id_sequence, top_k=12, exclude_seen=True):
    """Run SASRec inference. Returns list of (movieId, score) tuples."""
    if not _load_sasrec():
        return []
    model    = _SASREC_CACHE['model']
    item_enc = _SASREC_CACHE['item_enc']
    cfg      = _SASREC_CACHE['cfg']
    max_seq_len = cfg['max_seq_len']
    num_items   = cfg['num_items']

    raw_ids = np.array(movie_id_sequence)
    known_mask = np.isin(raw_ids, item_enc.classes_)
    known_ids  = raw_ids[known_mask]
    if len(known_ids) == 0:
        return []

    encoded = item_enc.transform(known_ids) + 1  # 1-indexed (0 = pad)
    seq_list   = _pad_or_truncate(encoded.tolist(), max_seq_len)
    seq_tensor = torch.tensor(seq_list, dtype=torch.long).unsqueeze(0)

    all_item_ids = torch.arange(1, num_items, dtype=torch.long)
    with torch.no_grad():
        h_last    = F.normalize(model(seq_tensor)[:, -1, :], dim=-1)
        item_embs = F.normalize(model.item_emb(all_item_ids), dim=-1)
        scores_t  = (item_embs @ h_last.T).squeeze(-1)
    scores_np = scores_t.cpu().numpy()

    if exclude_seen:
        for idx in encoded:
            scores_np[idx - 1] = -np.inf

    # Only consider positions with finite scores (not masked out)
    candidate_pos = np.where(np.isfinite(scores_np))[0]
    if len(candidate_pos) == 0:
        return []

    k = min(top_k, len(candidate_pos))
    candidate_scores = scores_np[candidate_pos]
    top_local = np.argpartition(candidate_scores, -k)[-k:]
    top_local = top_local[np.argsort(candidate_scores[top_local])[::-1]]
    top_pos       = candidate_pos[top_local]
    top_item_idx  = top_pos + 1
    top_movie_ids = item_enc.inverse_transform(top_item_idx - 1)
    return list(zip(top_movie_ids.tolist(), scores_np[top_pos].tolist()))


# Section mapping for Algorithm 2 developer:
# - Movies based on your Genre Ratings -> getMoviesByGenres
# - Recommended -> getRecommendationBy
# - Liked with Similar Items -> getLikedSimilarBy
# - Liked -> get_liked_movie_ids
# - Disliked -> get_disliked_movie_ids

# - Movies based on your Genre Ratings
def getMoviesByGenres(user_genres, movies_df, genres_df):
    """Movies based on your Genre Ratings section.

    For algo2, user_genres is a dict {genreId: score} where:
      score == 10  -> user liked this genre
      score == 1   -> user disliked this genre
      score == 0   -> not selected (ignored)

    A movie is included if it matches at least one liked genre.
    A movie is excluded if ALL of its genres are disliked (none are liked).
    """
    results = []
    if not user_genres:
        # No preferences at all — return a random sample of 12 movies
        top = movies_df.sample(n=min(12, len(movies_df)), random_state=42).copy()
        top['similarity'] = 0.0
        _write_algo2_section('Movies based on your Genre Ratings (Top 12)', top, mode='w')
        return top.to_dict('records')

    # Separate liked (10) and disliked (1) genre IDs
    liked_ids = {int(gid) for gid, score in user_genres.items() if int(score) == 10}
    disliked_ids = {int(gid) for gid, score in user_genres.items() if int(score) == 1}

    # Build name sets from genre IDs
    liked_names = set(genres_df[genres_df['id'].isin(liked_ids)]['name'].tolist())
    disliked_names = set(genres_df[genres_df['id'].isin(disliked_ids)]['name'].tolist())

    if not liked_ids:
        # No likes: return movies that contain none of the disliked genres
        if not disliked_names:
            # All genres unselected — return a random sample
            top = movies_df.sample(n=min(12, len(movies_df)), random_state=42).copy()
            top['similarity'] = 0.0
            _write_algo2_section('Movies based on your Genre Ratings (Top 12)', top, mode='w')
            return top.to_dict('records')
        def has_no_disliked(movie_genres):
            return not set(movie_genres).intersection(disliked_names)
        results = movies_df[movies_df['genres'].apply(has_no_disliked)].copy()
        results['similarity'] = 0.0
        top = results.head(12)
        _write_algo2_section('Movies based on your Genre Ratings (Top 12)', top, mode='w')
        return top.to_dict('records')

    def is_eligible(movie_genres):
        genre_set = set(movie_genres)
        # Must match at least one liked genre
        if not genre_set.intersection(liked_names):
            return False
        # Exclude if the movie contains ANY disliked genre
        if disliked_names and genre_set.intersection(disliked_names):
            return False
        return True

    results = movies_df[movies_df['genres'].apply(is_eligible)].copy()

    if len(results) > 0:
        # Score = fraction of liked genres matched by the movie (0.0 – 1.0)
        n_liked = max(len(liked_names), 1)
        results['similarity'] = results['genres'].apply(
            lambda g: len(set(g).intersection(liked_names)) / n_liked
        )
        results = results.sort_values('similarity', ascending=False)
        _write_algo2_section('Movies based on your Genre Ratings (Top 12)', results.head(12), mode='w')
        return results.to_dict('records')
    _write_algo2_section('Movies based on your Genre Ratings (Top 12)', pd.DataFrame(), mode='w')
    return results

def _genre_weight_from_rating(rating):
    """Rating-to-weight mapping (mirrors algo1 scheme)."""
    mapping = {10: 1.6, 9: 1.4, 8: 1.3, 7: 1.2, 6: 1.1,
               5: 1.0, 4: 0.9, 3: 0.8, 2: 0.7, 1: 0.5}
    return float(mapping.get(int(max(1, min(10, rating))), 1.0))


# - Recommended
def getRecommendationBy(user_rates, movies_df, rates_df, user_genres=None, genres_df=None):
    """Content-based recommendations combining genre preferences and movie rating history.

    Builds a genre-space user profile from two signals:
      1. Genre preferences: liked (10) → +1.0, disliked (1) → -0.5, unselected (0) → 0.0
      2. Movie ratings: each rated movie's genres are weighted by normalised rating,
         so that high-rated movies reinforce genres and low-rated movies penalise them.
    The two profiles are combined (alpha weighting), then cosine similarity is used
    to rank all unrated candidate movies.
    Works from genre preferences alone (no movie ratings needed).
    """
    if not user_rates:
        _write_algo2_section('Recommended (Top 12)', pd.DataFrame(), mode='a')
        return [], 'Rate a movie to get recommendations.'

    if genres_df is None or genres_df.empty:
        _write_algo2_section('Recommended (Top 12)', pd.DataFrame(), mode='a')
        return [], 'No recommendations.'

    all_genre_names = genres_df['name'].tolist()
    n_genres = len(all_genre_names)
    genre_name_index = {name: i for i, name in enumerate(all_genre_names)}

    # ── 1. Genre preference profile ──────────────────────────────────────────
    genre_pref_vec = np.zeros(n_genres, dtype=np.float32)
    if user_genres and isinstance(user_genres, dict):
        gid_to_name = dict(zip(genres_df['id'].tolist(), genres_df['name'].tolist()))
        for gid, score in user_genres.items():
            gname = gid_to_name.get(int(gid))
            if gname in genre_name_index:
                s = int(score)
                if s == 10:
                    genre_pref_vec[genre_name_index[gname]] = 1.0
                elif s == 1:
                    genre_pref_vec[genre_name_index[gname]] = -0.5

    # ── 2. Movie rating profile ───────────────────────────────────────────────
    movie_pref_vec = np.zeros(n_genres, dtype=np.float32)
    rated_ids = set()
    if user_rates:
        for record in user_rates:
            parts = record.split('|')
            if len(parts) < 3:
                continue
            try:
                movie_id = int(parts[1])
                rating   = int(parts[2])
            except ValueError:
                continue
            rated_ids.add(movie_id)
            weight = _genre_weight_from_rating(rating)
            movie_row = movies_df[movies_df['movieId'] == movie_id]
            if movie_row.empty:
                continue
            for gname in (movie_row.iloc[0]['genres'] or []):
                if gname in genre_name_index:
                    movie_pref_vec[genre_name_index[gname]] += weight

    # ── 3. Combine profiles ───────────────────────────────────────────────────
    gp_norm = float(np.linalg.norm(genre_pref_vec))
    mp_norm = float(np.linalg.norm(movie_pref_vec))
    has_genre = gp_norm > 1e-8
    has_movie = mp_norm > 1e-8

    alpha = 0.6  # weight for movie-rating profile when both signals exist
    if has_genre and has_movie:
        combined = alpha * (movie_pref_vec / mp_norm) + (1.0 - alpha) * (genre_pref_vec / gp_norm)
    elif has_movie:
        combined = movie_pref_vec / mp_norm
    elif has_genre:
        combined = genre_pref_vec / gp_norm
    else:
        _write_algo2_section('Recommended (Top 12)', pd.DataFrame(), mode='a')
        return [], 'No recommendations.'

    c_norm = float(np.linalg.norm(combined))
    if c_norm < 1e-8:
        _write_algo2_section('Recommended (Top 12)', pd.DataFrame(), mode='a')
        return [], 'No recommendations.'
    user_profile = combined / c_norm

    # ── 4. Score all candidate movies ─────────────────────────────────────────
    candidate_mask = ~movies_df['movieId'].isin(rated_ids)
    candidates = movies_df[candidate_mask]

    if candidates.empty:
        _write_algo2_section('Recommended (Top 12)', pd.DataFrame(), mode='a')
        return [], 'No recommendations.'

    # Build genre matrix (n_candidates × n_genres)
    movie_matrix = np.array([
        [1.0 if g in set(row) else 0.0 for g in all_genre_names]
        for row in candidates['genres']
    ], dtype=np.float32)

    row_norms = np.linalg.norm(movie_matrix, axis=1, keepdims=True)
    row_norms[row_norms < 1e-8] = 1.0
    movie_matrix_norm = movie_matrix / row_norms

    scores = movie_matrix_norm @ user_profile
    top_idx = np.argsort(-scores)[:12]

    top_movie_ids = candidates.iloc[top_idx]['movieId'].tolist()
    top_scores    = scores[top_idx].tolist()

    results = candidates.iloc[top_idx].copy()
    score_map = dict(zip(top_movie_ids, top_scores))
    results['similarity'] = results['movieId'].map(score_map)

    if len(results) > 0:
        _write_algo2_section('Recommended (Top 12)', results, mode='a')
        return results.to_dict('records'), 'These movies are recommended based on your ratings.'
    _write_algo2_section('Recommended (Top 12)', pd.DataFrame(), mode='a')
    return [], 'No recommendations.'

# - Liked with Similar Items
def getLikedSimilarBy(user_likes, movies_df):
    """
    Template for Algorithm 2 developer to implement liked items similarity.
    Current implementation uses content-based similarity with genre vectors.
    Feel free to replace with your own approach for better recommendations.
    
    Args:
        user_likes: List of movie IDs that user rated >= 8
        movies_df: DataFrame with all movies
    
    Returns:
        Tuple of (results list, message string)
    """
    results = []
    if len(user_likes) > 0:
        # Step 1: Representing items with multi-hot vectors
        item_rep_matrix, item_rep_vector, feature_list = item_representation_based_movie_genres(movies_df)
        # Step 2: Building user profile
        user_profile = build_user_profile(user_likes, item_rep_vector, feature_list)
        # Step 3: Predicting user interest in items
        results = generate_recommendation_results(user_profile, item_rep_matrix, item_rep_vector, 12)
    if len(results) > 0:
        _write_algo2_section('Liked with Similar Items (Top 12)', results, mode='a')
        return results.to_dict('records'), "The movies are similar to your liked movies."
    _write_algo2_section('Liked with Similar Items (Top 12)', pd.DataFrame(), mode='a')
    return results, "No similar movies found."

# ==================================================================================================
# Test file writer — mirrors Algo1RecTest.txt format for easy comparison
# ==================================================================================================

def _algo2_test_file_path(filename='Algo2RecTest.txt'):
    return os.path.join(os.path.dirname(__file__), filename)

def _write_section_header(f, title):
    f.write('\n' + '=' * 90 + '\n')
    f.write(title + '\n')
    f.write('=' * 90 + '\n')

def _write_section_rows(f, df):
    if df is None or len(df) == 0:
        f.write('No results.\n')
        return
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        movie_id = int(row.get('movieId', 0))
        title = str(row.get('title', ''))
        score = row.get('similarity', None)
        try:
            score_text = f"{float(score):.6f}" if (score is not None and score == score) else 'N/A'
        except (TypeError, ValueError):
            score_text = 'N/A'
        genres = row.get('genres', [])
        genres_text = ', '.join(genres) if isinstance(genres, list) else str(genres)
        f.write(f"{rank:02d}. movieId={movie_id} | score={score_text} | title={title} | genres={genres_text}\n")

def _write_algo2_section(title, df, mode='a', filename='Algo2RecTest.txt'):
    try:
        path = _algo2_test_file_path(filename)
        with open(path, mode, encoding='utf-8') as f:
            if mode == 'w':
                f.write('Algo2 Recommendation Debug Output\n')
            _write_section_header(f, title)
            _write_section_rows(f, df)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f'[algo2] Failed to write test file: {e}')


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

# - Disliked
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
