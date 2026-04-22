import pandas as pd
from surprise import Reader
from surprise import KNNWithMeans
from surprise import Dataset

from .main import normalize_rating_to_legacy_scale
from .main import item_representation_based_movie_genres
from .main import build_user_profile
from .main import generate_recommendation_results
from .main import is_genre_match
from .tools.data_tool import ratesFromUser


# Section mapping for Algorithm 2 developer:
# - Movies based on your Genre Ratings -> getMoviesByGenres
# - Recommended -> getRecommendationBy
# - Liked with Similar Items -> getLikedSimilarBy
# - Liked -> get_liked_movie_ids
# - Disliked -> get_disliked_movie_ids

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
        results = movies_df[movies_df['genres'].apply(lambda x: is_genre_match(x, interested_genres))]

    if len(results) > 0:
        return results.to_dict('records')
    return results

# - Recommended
def getRecommendationBy(user_rates, movies_df, rates_df):
    """
    Template algorithm file for Algorithm 2 developer.
    Keep this function signature stable for integration in main.py.
    """
    results = []

    if len(user_rates) > 0:
        reader = Reader(rating_scale=(1, 5))
        algo = KNNWithMeans(sim_options={'name': 'pearson', 'user_based': True})

        user_rates_df = ratesFromUser(user_rates)
        user_rates_df['rating'] = user_rates_df['rating'].apply(normalize_rating_to_legacy_scale)

        training_rates = pd.concat([rates_df, user_rates_df], ignore_index=True)
        training_data = Dataset.load_from_df(training_rates, reader=reader)
        trainset = training_data.build_full_trainset()
        algo.fit(trainset)

        all_movie_ids = movies_df['movieId'].unique()
        user_id = 611
        rated_movie_ids = user_rates_df[user_rates_df['userId'] == user_id]['movieId'].tolist()
        predictions = [algo.predict(user_id, movie_id) for movie_id in all_movie_ids if movie_id not in rated_movie_ids]
        predictions.sort(key=lambda x: x.est, reverse=True)

        top_movie_ids = [pred.iid for pred in predictions[:12]]
        results = movies_df[movies_df['movieId'].isin(top_movie_ids)]

    if len(results) > 0:
        return results.to_dict('records'), 'These movies are recommended based on your ratings.'
    return results, 'No recommendations.'

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
        return results.to_dict('records'), "The movies are similar to your liked movies."
    return results, "No similar movies found."

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
