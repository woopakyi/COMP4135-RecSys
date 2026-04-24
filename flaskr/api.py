"""
API Endpoints for Movie Recommender System
Handles ratings, profile, feedback, and data migration
"""

from flask import Blueprint, jsonify, request, g
from .models import db, User, Movie, Rating, Feedback
from sqlalchemy import desc
import json
from datetime import datetime, timezone

api_bp = Blueprint('api', __name__, url_prefix='/api')


FEEDBACK_TYPE_CONFIG = {
    'algo_ui1': {
        'label': 'Algorithm Evaluation based on UI 1',
        'option_a_key': 'algo1',
        'option_a_label': 'Algo 1 (FM)',
        'option_b_key': 'algo2',
        'option_b_label': 'Algo 2 (SASRec)',
    },
    'algo_ui2': {
        'label': 'Algorithm Evaluation based on UI 2',
        'option_a_key': 'algo1',
        'option_a_label': 'Algo 1 (FM)',
        'option_b_key': 'algo2',
        'option_b_label': 'Algo 2 (SASRec)',
    },
    'ui_algo1': {
        'label': 'User Interface Evaluation based on Algo 1',
        'option_a_key': 'ui1',
        'option_a_label': 'UI 1 (Dark)',
        'option_b_key': 'ui2',
        'option_b_label': 'UI 2 (Light)',
    },
    'ui_algo2': {
        'label': 'User Interface Evaluation based on Algo 2',
        'option_a_key': 'ui1',
        'option_a_label': 'UI 1 (Dark)',
        'option_b_key': 'ui2',
        'option_b_label': 'UI 2 (Light)',
    },
}


# ==================== GENRES ====================
@api_bp.route('/genres', methods=['GET'])
def get_genres():
    """Get all available genres"""
    from .tools.data_tool import loadData
    _, genres, _ = loadData()
    return jsonify({'genres': genres.to_dict('records')})


@api_bp.route('/movies', methods=['GET'])
def get_movies_by_ids():
    """Get minimal movie metadata for provided IDs."""
    ids_text = request.args.get('ids', '').strip()
    if not ids_text:
        return jsonify({'movies': []})

    parsed_ids = []
    for token in ids_text.split(','):
        token = token.strip()
        if not token:
            continue
        try:
            parsed_ids.append(int(token))
        except ValueError:
            continue

    if not parsed_ids:
        return jsonify({'movies': []})

    movies = Movie.query.filter(Movie.id.in_(parsed_ids)).all()
    return jsonify({
        'movies': [{
            'id': movie.id,
            'title': movie.title,
            'year': movie.year,
            'image_url': movie.image_url,
        } for movie in movies]
    })


# ==================== RATINGS ====================
@api_bp.route('/ratings', methods=['GET'])
def get_ratings():
    """Get user's movie ratings"""
    if not g.user:
        return jsonify({'ratings': []}), 200

    ratings = Rating.query.filter_by(user_id=g.user.id).all()
    return jsonify({
        'ratings': [{
            'id': r.id,
            'movie_id': r.movie_id,
            'movie': {
                'id': r.movie.id,
                'title': r.movie.title,
                'year': r.movie.year,
                'image_url': r.movie.image_url
            },
            'rating': r.rating,
            'timestamp': r.timestamp.isoformat()
        } for r in ratings]
    })


@api_bp.route('/ratings', methods=['POST'])
def add_rating():
    """Add or update a movie rating"""
    if not g.user:
        return jsonify({'error': 'Not logged in'}), 401

    data = request.json
    movie_id = data.get('movie_id')
    rating = data.get('rating')

    if not movie_id or rating is None:
        return jsonify({'error': 'Missing movie_id or rating'}), 400

    if not (1 <= rating <= 10):
        return jsonify({'error': 'Rating must be between 1 and 10'}), 400

    movie = Movie.query.get(movie_id)
    if not movie:
        return jsonify({'error': 'Movie not found'}), 404

    existing = Rating.query.filter_by(user_id=g.user.id, movie_id=movie_id).first()
    if existing:
        existing.rating = rating
    else:
        existing = Rating(user_id=g.user.id, movie_id=movie_id, rating=rating)
        db.session.add(existing)

    db.session.commit()
    return jsonify({'success': True, 'rating_id': existing.id})


@api_bp.route('/ratings/<int:rating_id>', methods=['DELETE'])
def delete_rating(rating_id):
    """Delete a specific rating"""
    if not g.user:
        return jsonify({'error': 'Not logged in'}), 401

    rating = Rating.query.get(rating_id)
    if not rating or rating.user_id != g.user.id:
        return jsonify({'error': 'Rating not found'}), 404

    db.session.delete(rating)
    db.session.commit()
    return jsonify({'success': True})


@api_bp.route('/ratings/by-movie/<int:movie_id>', methods=['DELETE'])
def delete_rating_by_movie(movie_id):
    """Delete a user's rating by movie id."""
    if not g.user:
        return jsonify({'error': 'Not logged in'}), 401

    rating = Rating.query.filter_by(user_id=g.user.id, movie_id=movie_id).first()
    if not rating:
        return jsonify({'error': 'Rating not found'}), 404

    db.session.delete(rating)
    db.session.commit()
    return jsonify({'success': True})


@api_bp.route('/ratings/clear', methods=['POST'])
def clear_all_ratings():
    """Delete all ratings for the user"""
    if not g.user:
        return jsonify({'error': 'Not logged in'}), 401

    Rating.query.filter_by(user_id=g.user.id).delete()
    db.session.commit()
    return jsonify({'success': True})


@api_bp.route('/home/sections', methods=['POST'])
def refresh_home_sections():
    """Recompute homepage movie sections from current rating/genre state."""
    data = request.json or {}
    user_rates = data.get('user_rates', [])
    user_genres = data.get('user_genres', [])
    selected_algorithm = str(data.get('selected_algorithm') or request.cookies.get('user_algorithm', '1'))

    if not isinstance(user_genres, (list, dict)):
        user_genres = []
    if isinstance(user_genres, dict):
        user_genres = {int(k): v for k, v in user_genres.items()}
    if selected_algorithm not in {'1', '2'}:
        selected_algorithm = '1'

    from .main import (
        movies,
        getMoviesByGenres,
        getRecommendationBy,
        getLikedSimilarBy,
        getUserLikesBy,
        getUserDislikesBy,
        get_liked_movie_ids,
        get_disliked_movie_ids,
        enrich_movies_with_user_feedback,
    )

    liked_movie_ids = get_liked_movie_ids(user_rates, selected_algorithm)
    disliked_movie_ids = get_disliked_movie_ids(user_rates, selected_algorithm)

    default_genres_movies = getMoviesByGenres(user_genres, selected_algorithm)[:12]
    if len(default_genres_movies) == 0:
        default_genres_movies = movies.head(12).to_dict('records')

    recommendations_movies, recommendations_message = getRecommendationBy(user_rates, selected_algorithm, user_genres)
    likes_similar_movies, likes_similar_message = getLikedSimilarBy(liked_movie_ids, selected_algorithm)
    likes_movies = getUserLikesBy([str(movie_id) for movie_id in liked_movie_ids])
    dislikes_movies = getUserDislikesBy([str(movie_id) for movie_id in disliked_movie_ids])

    default_genres_movies = enrich_movies_with_user_feedback(default_genres_movies, user_rates)
    recommendations_movies = enrich_movies_with_user_feedback(recommendations_movies, user_rates)
    likes_similar_movies = enrich_movies_with_user_feedback(likes_similar_movies, user_rates)
    likes_movies = enrich_movies_with_user_feedback(likes_movies, user_rates)
    dislikes_movies = enrich_movies_with_user_feedback(dislikes_movies, user_rates)

    return jsonify({
        'default_genres_movies': default_genres_movies,
        'recommendations': recommendations_movies,
        'recommendations_message': recommendations_message,
        'likes_similars': likes_similar_movies,
        'likes_similar_message': likes_similar_message,
        'likes': likes_movies,
        'dislikes': dislikes_movies,
    })


# ==================== PROFILE ====================
@api_bp.route('/profile', methods=['GET'])
def get_profile():
    """Get user profile"""
    if not g.user:
        return jsonify({'error': 'Not logged in'}), 401

    return jsonify({
        'user': {
            'id': g.user.id,
            'username': g.user.username,
            'email': g.user.email,
            'algorithm_preference': g.user.algorithm_preference,
            'ui_preference': g.user.ui_preference,
            'preferences_saved': bool(g.user.preferences_saved),
            'genre_preferences': g.user.get_genre_preferences(),
            'feedback_progress_types': g.user.get_feedback_progress_types(),
            'participant_full_name': g.user.participant_full_name or '',
            'participant_contact_email': g.user.participant_contact_email or '',
            'admin': bool(g.user.admin),
            'created_at': g.user.created_at.isoformat()
        }
    })


@api_bp.route('/profile', methods=['PUT'])
def update_profile():
    """Update user profile"""
    if not g.user:
        return jsonify({'error': 'Not logged in'}), 401

    data = request.json

    if 'username' in data:
        # Check if username is taken
        if data['username'] != g.user.username:
            existing = User.query.filter_by(username=data['username']).first()
            if existing:
                return jsonify({'error': 'Username already taken'}), 400
        g.user.username = data['username']

    if 'algorithm_preference' in data:
        g.user.algorithm_preference = data['algorithm_preference']

    if 'ui_preference' in data:
        g.user.ui_preference = data['ui_preference']

    if 'genre_preferences' in data and isinstance(data.get('genre_preferences'), dict):
        normalized = {}
        for key, value in data['genre_preferences'].items():
            try:
                gid = int(key)
                normalized[str(gid)] = max(0, min(10, float(value)))
            except (ValueError, TypeError):
                continue
        g.user.set_genre_preferences(normalized)

    if 'preferences_saved' in data:
        g.user.preferences_saved = bool(data.get('preferences_saved'))

    db.session.commit()
    return jsonify({'success': True})


@api_bp.route('/preferences/reset', methods=['POST'])
def reset_preferences():
    """Reset logged-in user's saved preferences back to defaults."""
    if not g.user:
        return jsonify({'error': 'Not logged in'}), 401

    g.user.algorithm_preference = 'algo1'
    g.user.ui_preference = '1'
    g.user.preferences_saved = False
    g.user.set_genre_preferences({})
    db.session.commit()
    return jsonify({'success': True})


@api_bp.route('/genre-preferences', methods=['POST'])
def update_genre_preference():
    """Update a genre preference"""
    if not g.user:
        return jsonify({'error': 'Not logged in'}), 401

    data = request.json
    genre_id = data.get('genre_id')
    score = data.get('score', 0)

    if genre_id is None:
        return jsonify({'error': 'Missing genre_id'}), 400

    prefs = g.user.get_genre_preferences()
    prefs[str(genre_id)] = score
    g.user.set_genre_preferences(prefs)

    db.session.commit()
    return jsonify({'success': True})


# ==================== FEEDBACK ====================
@api_bp.route('/feedback', methods=['GET'])
def get_feedback():
    """Get user's feedback submissions"""
    try:
        if not g.user:
            return jsonify({'feedback': []}), 200

        feedback_list = Feedback.query.filter_by(user_id=g.user.id).all()
        return jsonify({
            'feedback': [{
                'id': f.id,
                'full_name': f.full_name,
                'contact_email': f.contact_email,
                'feedback_type': f.feedback_type,
                'rating_option_a': f.rating_option_a,
                'rating_option_b': f.rating_option_b,
                'feedback_text': f.feedback_text,
                'created_at': f.created_at.isoformat(),
            } for f in feedback_list]
        })
    except Exception:
        return jsonify({'feedback': []}), 200


@api_bp.route('/feedback/prefill', methods=['GET'])
def feedback_prefill():
    """Get default participant information and completed feedback types for signed-in users."""
    try:
        if not g.user:
            return jsonify({'logged_in': False, 'defaults': {}, 'submitted_types': []})

        user_rows = Feedback.query.filter_by(user_id=g.user.id).filter(
            Feedback.submission_type.in_(['participant', 'logged'])
        ).all()
        submitted_types = [row.feedback_type for row in user_rows]
        migrated_types = [t for t in g.user.get_feedback_progress_types() if isinstance(t, str)]
        submitted_types = list(dict.fromkeys(submitted_types + migrated_types))
        latest_row = Feedback.query.filter_by(user_id=g.user.id).filter(
            Feedback.submission_type.in_(['participant', 'logged'])
        ).order_by(desc(Feedback.created_at)).first()
        default_full_name = (
            latest_row.full_name if latest_row and latest_row.full_name
            else (g.user.participant_full_name or g.user.username)
        ) or ''
        default_contact_email = (
            latest_row.contact_email if latest_row and latest_row.contact_email
            else (g.user.participant_contact_email or g.user.email)
        ) or ''

        return jsonify({
            'logged_in': True,
            'defaults': {
                'full_name': default_full_name,
                'contact_email': default_contact_email,
            },
            'submitted_types': submitted_types,
        })
    except Exception:
        return jsonify({'logged_in': bool(g.user), 'defaults': {}, 'submitted_types': []})


@api_bp.route('/feedback', methods=['POST'])
def submit_feedback():
    """Submit feedback"""
    try:
        data = request.json or {}
        full_name = (data.get('full_name') or '').strip()
        contact_email = (data.get('contact_email') or '').strip()
        feedback_type = data.get('feedback_type')
        rating_option_a = data.get('rating_option_a')
        rating_option_b = data.get('rating_option_b')
        submission_type = data.get('submission_type')
        valid_types = set(FEEDBACK_TYPE_CONFIG.keys())
        valid_submission_types = {'participant', 'anonymous', 'logged'}

        if not feedback_type or rating_option_a is None or rating_option_b is None or not submission_type:
            return jsonify({'error': 'Missing required fields'}), 400

        if feedback_type not in valid_types:
            return jsonify({'error': 'Invalid feedback type'}), 400

        try:
            rating_option_a = int(rating_option_a)
            rating_option_b = int(rating_option_b)
        except (TypeError, ValueError):
            return jsonify({'error': 'Ratings must be integers'}), 400

        if not (1 <= rating_option_a <= 10) or not (1 <= rating_option_b <= 10):
            return jsonify({'error': 'Ratings must be between 1 and 10'}), 400

        if submission_type not in valid_submission_types:
            return jsonify({'error': 'Invalid submission type'}), 400

        if submission_type == 'logged':
            submission_type = 'participant'

        if submission_type == 'participant':
            if not full_name:
                full_name = g.user.username if g.user else ''
            if not contact_email:
                contact_email = g.user.email if g.user else ''
            if not full_name or not contact_email:
                return jsonify({'error': 'Full name and contact email are required'}), 400

        existing_for_type = None

        # For participant submissions, allow repeated submissions but keep a helpful message.
        if submission_type == 'participant':
            if g.user:
                existing_for_type = Feedback.query.filter_by(
                    user_id=g.user.id,
                    feedback_type=feedback_type
                ).filter(Feedback.submission_type.in_(['participant', 'logged'])).order_by(desc(Feedback.created_at)).first()

            feedback = Feedback(
                user_id=g.user.id if g.user else None,
                full_name=full_name,
                contact_email=contact_email,
                submission_type='participant',
                feedback_type=feedback_type,
                rating_option_a=rating_option_a,
                rating_option_b=rating_option_b,
                feedback_text=data.get('feedback_text'),
            )
        else:
            # Anonymous feedback
            feedback = Feedback(
                full_name=full_name or None,
                contact_email=contact_email or None,
                submission_type='anonymous',
                feedback_type=feedback_type,
                rating_option_a=rating_option_a,
                rating_option_b=rating_option_b,
                feedback_text=data.get('feedback_text'),
            )

        db.session.add(feedback)
        if submission_type == 'participant' and g.user:
            g.user.participant_full_name = full_name
            g.user.participant_contact_email = contact_email
            existing_types = g.user.get_feedback_progress_types()
            if feedback_type not in existing_types:
                existing_types.append(feedback_type)
                g.user.set_feedback_progress_types(existing_types)
        db.session.commit()
        response = {'success': True, 'feedback_id': feedback.id}
        if existing_for_type:
            response['message'] = 'You have already submitted this type of evaluation before. You are welcome to submit it again.'
        return jsonify(response)
    except Exception:
        db.session.rollback()
        return jsonify({'error': 'Failed to submit feedback'}), 500


@api_bp.route('/feedback/progress', methods=['GET'])
def feedback_progress():
    """Return feedback progress for profile view.

    Logged-in users get per-user completion, while guests get aggregate completion.
    Progress only counts participant submissions.
    """
    all_types = list(FEEDBACK_TYPE_CONFIG.keys())
    participant_types = ['participant', 'logged']
    if g.user:
        migrated_types = set(g.user.get_feedback_progress_types())
        submitted = {
            row.feedback_type for row in Feedback.query.filter_by(user_id=g.user.id).filter(
                Feedback.submission_type.in_(participant_types)
            ).all()
        }
        submitted = submitted.union(migrated_types)
        progress = [{
            'feedback_type': ft,
            'completed': ft in submitted,
            'submission_count': None,
        } for ft in all_types]
        completed_count = len(submitted.intersection(set(all_types)))
    else:
        submitted_rows = Feedback.query.with_entities(Feedback.feedback_type).filter(
            Feedback.submission_type.in_(participant_types)
        ).all()
        submission_counts = {ft: 0 for ft in all_types}
        for row in submitted_rows:
            if row.feedback_type in submission_counts:
                submission_counts[row.feedback_type] += 1

        progress = [{
            'feedback_type': ft,
            'completed': submission_counts[ft] > 0,
            'submission_count': submission_counts[ft],
        } for ft in all_types]
        completed_count = sum(1 for ft in all_types if submission_counts[ft] > 0)

    return jsonify({
        'scope': 'user' if g.user else 'global',
        'completed_count': completed_count,
        'total_count': len(all_types),
        'progress': progress,
    })


@api_bp.route('/evaluation/summary', methods=['GET'])
def evaluation_summary():
    """Return aggregated rating stats for evaluation charts."""
    try:
        source_filter = request.args.get('source', 'all')
        if source_filter == 'logged':
            source_filter = 'participant'
        if source_filter not in {'all', 'participant', 'anonymous'}:
            return jsonify({'error': 'Invalid source filter'}), 400

        types = list(FEEDBACK_TYPE_CONFIG.keys())

        def apply_source_filter(query):
            if source_filter == 'participant':
                return query.filter(Feedback.submission_type.in_(['participant', 'logged']))
            if source_filter == 'anonymous':
                return query.filter(Feedback.submission_type == 'anonymous')
            return query

        summary = {}
        for feedback_type in types:
            query = Feedback.query.filter_by(feedback_type=feedback_type)
            query = apply_source_filter(query)
            rows = query.all()
            valid_rows = [row for row in rows if row.rating_option_a is not None and row.rating_option_b is not None]
            total = len(valid_rows)

            sum_a = sum(int(row.rating_option_a) for row in valid_rows)
            sum_b = sum(int(row.rating_option_b) for row in valid_rows)
            avg_a = round(sum_a / total, 2) if total else 0
            avg_b = round(sum_b / total, 2) if total else 0

            prefer_a = sum(1 for row in valid_rows if int(row.rating_option_a) > int(row.rating_option_b))
            prefer_b = sum(1 for row in valid_rows if int(row.rating_option_b) > int(row.rating_option_a))
            ties = sum(1 for row in valid_rows if int(row.rating_option_a) == int(row.rating_option_b))

            cfg = FEEDBACK_TYPE_CONFIG[feedback_type]

            summary[feedback_type] = {
                'option_a_label': cfg['option_a_label'],
                'option_b_label': cfg['option_b_label'],
                'total': total,
                'avg_a': avg_a,
                'avg_b': avg_b,
                'delta': round(avg_a - avg_b, 2) if total else 0,
                'has_data': total > 0,
                'prefer_a_count': prefer_a,
                'prefer_b_count': prefer_b,
                'tie_count': ties,
            }

        return jsonify({'summary': summary, 'source': source_filter})
    except Exception:
        return jsonify({'summary': {}, 'source': request.args.get('source', 'all')}), 200


@api_bp.route('/evaluation/votes', methods=['GET'])
def evaluation_votes():
    """Return rating records for admin table only."""
    try:
        if not g.user or not getattr(g.user, 'admin', False):
            return jsonify({'error': 'Admin access required'}), 403

        source_filter = request.args.get('source', 'all')
        if source_filter == 'logged':
            source_filter = 'participant'
        type_filter = request.args.get('feedback_type', 'all')

        query = Feedback.query

        if source_filter == 'participant':
            query = query.filter(Feedback.submission_type.in_(['participant', 'logged']))
        elif source_filter == 'anonymous':
            query = query.filter(Feedback.submission_type == 'anonymous')

        if type_filter in FEEDBACK_TYPE_CONFIG:
            query = query.filter(Feedback.feedback_type == type_filter)

        rows = query.order_by(desc(Feedback.created_at)).all()
        return jsonify({
            'votes': [{
                'id': row.id,
                'user_id': row.user_id,
                'full_name': row.full_name,
                'contact_email': row.contact_email,
                'submission_type': 'participant' if row.submission_type == 'logged' else row.submission_type,
                'feedback_type': row.feedback_type,
                'option_a_label': FEEDBACK_TYPE_CONFIG.get(row.feedback_type, {}).get('option_a_label', 'Option A'),
                'option_b_label': FEEDBACK_TYPE_CONFIG.get(row.feedback_type, {}).get('option_b_label', 'Option B'),
                'rating_option_a': row.rating_option_a,
                'rating_option_b': row.rating_option_b,
                'feedback_text': row.feedback_text,
                'created_at': row.created_at.isoformat() if row.created_at else None,
            } for row in rows]
        })
    except Exception:
        return jsonify({'votes': []}), 200


@api_bp.route('/feedback/<int:feedback_id>', methods=['PUT'])
def update_feedback(feedback_id):
    """Feedback updates are disabled; users should submit a new record instead."""
    return jsonify({'error': 'Editing existing feedback is disabled. Please submit a new feedback record.'}), 403


# ==================== MIGRATION ====================
@api_bp.route('/migrate-ratings', methods=['POST'])
def migrate_ratings():
    """Migrate localStorage ratings to database (on login).

    Migrate local guest state to account on login.
    Ratings keep the latest record by timestamp when duplicates exist.
    """
    if not g.user:
        return jsonify({'error': 'Not logged in'}), 401

    data = request.json or {}
    ratings_data = data.get('ratings', [])
    genre_prefs = data.get('genrePreferences', {})
    algorithm = data.get('algorithm')
    ui_mode = data.get('uiMode')
    setup_saved = bool(data.get('setupSaved'))
    feedback_progress_types = data.get('feedbackProgressTypes', [])
    participant_profile = data.get('participantProfile', {}) if isinstance(data.get('participantProfile'), dict) else {}

    def parse_client_timestamp(raw_value):
        if raw_value is None:
            return None
        if isinstance(raw_value, (int, float)):
            return datetime.fromtimestamp(float(raw_value), tz=timezone.utc).replace(tzinfo=None)
        if isinstance(raw_value, str):
            text = raw_value.strip()
            if not text:
                return None
            if text.isdigit():
                return datetime.fromtimestamp(float(text), tz=timezone.utc).replace(tzinfo=None)
            try:
                return datetime.fromisoformat(text.replace('Z', '+00:00')).replace(tzinfo=None)
            except ValueError:
                return None
        return None

    created_count = 0
    updated_count = 0

    for rating in ratings_data:
        try:
            movie_id = int(rating.get('movieId'))
            rating_value = int(rating.get('rating'))
        except (TypeError, ValueError):
            continue

        if not (1 <= rating_value <= 10):
            continue

        incoming_timestamp = parse_client_timestamp(rating.get('timestamp'))
        existing = Rating.query.filter_by(user_id=g.user.id, movie_id=movie_id).first()

        if not existing:
            new_rating = Rating(
                user_id=g.user.id,
                movie_id=movie_id,
                rating=rating_value,
                timestamp=incoming_timestamp or datetime.utcnow()
            )
            db.session.add(new_rating)
            created_count += 1
            continue

        # Keep latest record when local and account contain same movie rating.
        if incoming_timestamp and (not existing.timestamp or incoming_timestamp > existing.timestamp):
            existing.rating = rating_value
            existing.timestamp = incoming_timestamp
            updated_count += 1

    if isinstance(genre_prefs, dict):
        normalized = {}
        for genre_id, score in genre_prefs.items():
            try:
                gid = int(genre_id)
                normalized[str(gid)] = max(0, min(10, float(score)))
            except (TypeError, ValueError):
                continue
        if normalized:
            g.user.set_genre_preferences(normalized)

    if algorithm in {'algo1', 'algo2'}:
        g.user.algorithm_preference = algorithm

    if ui_mode in {'1', '2'}:
        g.user.ui_preference = ui_mode

    g.user.preferences_saved = setup_saved

    if isinstance(feedback_progress_types, list):
        valid_types = [t for t in feedback_progress_types if isinstance(t, str) and t in FEEDBACK_TYPE_CONFIG]
        if valid_types:
            g.user.set_feedback_progress_types(valid_types)

    migrated_full_name = (participant_profile.get('full_name') or '').strip() if isinstance(participant_profile, dict) else ''
    migrated_contact_email = (participant_profile.get('contact_email') or '').strip() if isinstance(participant_profile, dict) else ''
    if migrated_full_name:
        g.user.participant_full_name = migrated_full_name
    if migrated_contact_email:
        g.user.participant_contact_email = migrated_contact_email

    db.session.commit()
    return jsonify({'success': True, 'created': created_count, 'updated': updated_count})
