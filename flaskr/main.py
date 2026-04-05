from flask import (
    Blueprint, render_template, request, current_app, jsonify
)
from datetime import datetime, timezone

from .tools.data_tool import *

from surprise import Reader
from surprise import KNNBasic, KNNWithMeans
from surprise import Dataset
from sklearn.metrics.pairwise import cosine_similarity

bp = Blueprint('main', __name__, url_prefix='/')

movies, genres, rates = loadData()

# These flags are intentionally controlled here for collaboration with algorithm developers.
algo1 = False
algo2 = False


@bp.route('/', methods=('GET', 'POST'))
def index():
    default_genres = genres.to_dict('records')
    selected_algorithm = request.cookies.get('user_algorithm', '')
    selected_ui = request.cookies.get('user_ui', '')
    setup_started = request.cookies.get('user_started', '') == '1'
    setup_complete = bool(selected_algorithm and selected_ui and setup_started)

    user_genre_scores = parse_genre_scores(request.cookies.get('user_genre_scores', ''))
    user_genres = [str(key) for key, value in user_genre_scores.items() if value > 0]

    user_rates = parse_cookie_list(request.cookies.get('user_rates'))
    liked_movie_ids = get_liked_movie_ids(user_rates)

    default_genres_movies = []
    recommendations_movies = []
    recommendations_message = 'Choose an algorithm and UI, then save to begin.'
    likes_similar_movies = []
    likes_similar_message = 'Choose an algorithm and UI, then save to begin.'
    likes_movies = []

    if setup_complete:
        default_genres_movies = getMoviesByGenres(user_genres)[:12]
        if len(default_genres_movies) == 0:
            default_genres_movies = movies.head(12).to_dict('records')

        recommendations_movies, recommendations_message = getRecommendationBy(user_rates, selected_algorithm)
        likes_similar_movies, likes_similar_message = getLikedSimilarBy(liked_movie_ids)
        likes_movies = getUserLikesBy([str(movie_id) for movie_id in liked_movie_ids])

        default_genres_movies = enrich_movies_with_user_feedback(default_genres_movies, user_rates)
        recommendations_movies = enrich_movies_with_user_feedback(recommendations_movies, user_rates)
        likes_similar_movies = enrich_movies_with_user_feedback(likes_similar_movies, user_rates)
        likes_movies = enrich_movies_with_user_feedback(likes_movies, user_rates)

    return render_template('index.html',
                           genres=default_genres,
                           selected_algorithm=selected_algorithm,
                           selected_ui=selected_ui,
                           setup_started=setup_started,
                           setup_complete=setup_complete,
                           user_genre_scores=user_genre_scores,
                           user_genres=user_genres,
                           user_rates=user_rates,
                           default_genres_movies=default_genres_movies,
                           recommendations=recommendations_movies,
                           recommendations_message=recommendations_message,
                           likes_similars=likes_similar_movies,
                           likes_similar_message=likes_similar_message,
                           likes=likes_movies,
                           )


@bp.route('/profile', methods=('GET',))
def profile():
    return render_template('profile.html')


@bp.route('/feedback', methods=('GET',))
def feedback():
    return render_template('feedback.html')


@bp.route('/evaluation', methods=('GET',))
def evaluation():
    return render_template('evaluation.html')


@bp.route('/debug/db', methods=('GET',))
def debug_database():
    from .models import db, User, Movie, Rating, GenreScore, Feedback, Vote

    def safe_count(model):
        try:
            return model.query.count()
        except Exception:
            return None

    uri = current_app.config.get('SQLALCHEMY_DATABASE_URI', '')
    backend = 'postgresql' if uri.startswith('postgresql') or uri.startswith('postgres://') else 'sqlite'

    return jsonify({
        'backend': backend,
        'database_uri_prefix': uri.split('://', 1)[0] if '://' in uri else uri,
        'counts': {
            'users': safe_count(User),
            'movies': safe_count(Movie),
            'ratings': safe_count(Rating),
            'genre_scores': safe_count(GenreScore),
            'feedback': safe_count(Feedback),
            'votes': safe_count(Vote),
        }
    })


def getUserLikesBy(user_likes):
    results = []

    if len(user_likes) > 0:
        mask = movies['movieId'].isin([int(movieId) for movieId in user_likes])
        results = movies.loc[mask]

        original_orders = pd.DataFrame()
        for _id in user_likes:
            movie = results.loc[results['movieId'] == int(_id)]
            if len(original_orders) == 0:
                original_orders = movie
            else:
                original_orders = pd.concat([movie, original_orders])
        results = original_orders

    if len(results) > 0:
        return results.to_dict('records')
    return results


def parse_cookie_list(raw_value):
    if not raw_value:
        return []
    return [item for item in raw_value.split(',') if item]


def parse_genre_scores(raw_scores):
    scores = {}
    if not raw_scores:
        return scores

    for item in raw_scores.split(','):
        if ':' not in item:
            continue
        genre_id, score = item.split(':', 1)
        if not genre_id:
            continue
        try:
            scores[int(genre_id)] = int(score)
        except ValueError:
            continue

    return scores


def parse_user_rates_map(user_rates):
    rates_map = {}
    for rate in user_rates:
        parts = rate.split('|')
        if len(parts) < 3:
            continue

        movie_id = int(parts[1])
        rating = int(parts[2])
        timestamp = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else None
        rates_map[movie_id] = {
            'rating': rating,
            'timestamp': timestamp,
            'time_ago': format_time_ago(timestamp),
        }

    return rates_map


def enrich_movies_with_user_feedback(movie_list, user_rates):
    if len(movie_list) == 0:
        return []

    rates_map = parse_user_rates_map(user_rates)
    enriched = []

    for movie in movie_list:
        movie_copy = dict(movie)
        movie_id = int(movie_copy['movieId'])
        feedback = rates_map.get(movie_id)
        if feedback:
            movie_copy['user_rating'] = feedback['rating']
            movie_copy['rated_time_ago'] = feedback['time_ago']
        else:
            movie_copy['user_rating'] = 0
            movie_copy['rated_time_ago'] = ''
        enriched.append(movie_copy)

    return enriched


def format_time_ago(timestamp):
    if not timestamp:
        return ''

    now = datetime.now(timezone.utc)
    rated_at = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    delta_seconds = int((now - rated_at).total_seconds())

    if delta_seconds < 60:
        return 'just rated'
    if delta_seconds < 3600:
        minutes = delta_seconds // 60
        return f'rated {minutes} minute(s) ago'
    if delta_seconds < 86400:
        hours = delta_seconds // 3600
        return f'rated {hours} hour(s) ago'

    days = delta_seconds // 86400
    return f'rated {days} day(s) ago'


def normalize_rating_to_legacy_scale(raw_rating):
    clamped = max(1, min(10, int(raw_rating)))
    return round(clamped / 2, 1)


def get_liked_movie_ids(user_rates):
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

def is_genre_match(movie_genres, interested_genres):
    return bool(set(movie_genres).intersection(set(interested_genres)))

def getMoviesByGenres(user_genres):
    results = []
    if len(user_genres) > 0:
        genres_mask = genres['id'].isin([int(id) for id in user_genres])
        user_genres = [1 if has is True else 0 for has in genres_mask]
        user_genres_df = pd.DataFrame(user_genres,columns=['value'])
        user_genres_df = pd.concat([user_genres_df, genres['name']], axis=1)
        interested_genres = user_genres_df[user_genres_df['value'] == 1]['name'].tolist()
        results = movies[movies['genres'].apply(lambda x: is_genre_match(x, interested_genres))]

    if len(results) > 0:
        return results.to_dict('records')
    return results

# Modify this function
def getRecommendationBy(user_rates, selected_algorithm='1'):
    if selected_algorithm == '1' and algo1:
        from . import algo1 as algo1_module
        return algo1_module.getRecommendationBy(user_rates, movies, rates)
    if selected_algorithm == '2' and algo2:
        from . import algo2 as algo2_module
        return algo2_module.getRecommendationBy(user_rates, movies, rates)

    return getRecommendationByDefault(user_rates)


def getRecommendationByDefault(user_rates):
    results = []
    if len(user_rates) > 0:
        # Initialize a reader with rating scale from 1 to 5
        reader = Reader(rating_scale=(1, 5))
        # Define the algorithm
        algo = KNNWithMeans(sim_options={'name': 'pearson', 'user_based': True})
        # Convert the user's ratings (stored in "user_rates") to the Dataset format
        user_rates = ratesFromUser(user_rates)
        user_rates['rating'] = user_rates['rating'].apply(normalize_rating_to_legacy_scale)
        # Add the user’s rating information into the Movielens dataset
        training_rates = pd.concat([rates, user_rates], ignore_index=True)
        # Load the combined data as a training dataset 
        training_data = Dataset.load_from_df(training_rates, reader=reader)
        # Build a full training set from the dataset
        trainset = training_data.build_full_trainset()
        # Fit the algorithm using the trainset
        algo.fit(trainset)
        all_movie_ids = movies['movieId'].unique()
        # Predict ratings for all movies for the specified user (assuming user ID 611)
        user_id = 611 
        rated_movie_ids = user_rates[user_rates['userId'] == user_id]['movieId'].tolist()
        predictions = [algo.predict(user_id, movie_id) for movie_id in all_movie_ids if movie_id not in rated_movie_ids]
        top_predictions = [pred for pred in predictions]
        # sort predicted ratings in a descending order
        top_predictions.sort(key=lambda x: x.est, reverse=True)
        # Select the top-K items (e.g., 12)
        top_movie_ids = [pred.iid for pred in top_predictions[:12]]
        results = movies[movies['movieId'].isin(top_movie_ids)]


    # Return the result
    if len(results) > 0:
        return results.to_dict('records'), "These movies are recommended based on your ratings."
    return results, "No recommendations."



# Modify this function
def getLikedSimilarBy(user_likes):
    results = []
    if len(user_likes) > 0:
        # Step 1: Representing items with multi-hot vectors
        item_rep_matrix, item_rep_vector, feature_list = item_representation_based_movie_genres(movies)
        # Step 2: Building user profile
        user_profile = build_user_profile(user_likes, item_rep_vector, feature_list)
        # Step 3: Predicting user interest in items
        results = generate_recommendation_results(user_profile, item_rep_matrix, item_rep_vector, 12)
    if len(results) > 0:
        return results.to_dict('records'), "The movies are similar to your liked movies."
    return results, "No similar movies found."


# Step 1: Representing items with multi-hot vectors
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

# Step 2: Building user profile
def build_user_profile(movieIds, item_rep_vector, feature_list, weighted=True, normalized=True):
    user_movie_rating_df = item_rep_vector[item_rep_vector['movieId'].isin(movieIds)]
    user_movie_df = user_movie_rating_df[feature_list].mean()
    user_profile = user_movie_df.T
    
    if normalized:
        user_profile = user_profile / sum(user_profile.values)
        
    return user_profile
# Step 3: Predicting user preference for items
def generate_recommendation_results(user_profile,item_rep_matrix, movies_data, k=12):
    u_v = user_profile.values
    u_v_matrix =  [u_v]
    recommendation_table =  cosine_similarity(u_v_matrix,item_rep_matrix)
    recommendation_table_df = movies_data.copy(deep=True)
    recommendation_table_df['similarity'] = recommendation_table[0]
    rec_result = recommendation_table_df.sort_values(by=['similarity'], ascending=False)[:k]
    return rec_result
