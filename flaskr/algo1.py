import pandas as pd
from surprise import Reader
from surprise import KNNWithMeans
from surprise import Dataset
from sklearn.metrics.pairwise import cosine_similarity

from .main import normalize_rating_to_legacy_scale
from .tools.data_tool import ratesFromUser


def getRecommendationBy(user_rates, movies_df, rates_df):
    """
    Template algorithm file for Algorithm 1 developer.
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


def getLikedSimilarBy(user_likes, movies_df):
    """
    Template for Algorithm 1 developer to implement liked items similarity.
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


def item_representation_based_movie_genres(movies_df):
    """Step 1: Representing items with multi-hot vectors based on genres"""
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


def build_user_profile(movieIds, item_rep_vector, feature_list, weighted=True, normalized=True):
    """Step 2: Building user profile from liked movies"""
    user_movie_rating_df = item_rep_vector[item_rep_vector['movieId'].isin(movieIds)]
    user_movie_df = user_movie_rating_df[feature_list].mean()
    user_profile = user_movie_df.T
    
    if normalized:
        user_profile = user_profile / sum(user_profile.values)
        
    return user_profile


def generate_recommendation_results(user_profile, item_rep_matrix, movies_data, k=12):
    """Step 3: Predicting user preference for items using cosine similarity"""
    u_v = user_profile.values
    u_v_matrix = [u_v]
    recommendation_table = cosine_similarity(u_v_matrix, item_rep_matrix)
    recommendation_table_df = movies_data.copy(deep=True)
    recommendation_table_df['similarity'] = recommendation_table[0]
    rec_result = recommendation_table_df.sort_values(by=['similarity'], ascending=False)[:k]
    return rec_result
