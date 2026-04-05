import pandas as pd
from surprise import Reader
from surprise import KNNWithMeans
from surprise import Dataset

from .main import normalize_rating_to_legacy_scale
from .tools.data_tool import ratesFromUser


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
