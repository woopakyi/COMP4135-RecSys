"""
API Endpoints for Movie Recommender System
Handles ratings, profile, feedback, and data migration
"""

from flask import Blueprint, jsonify, request, g
from .models import db, User, Movie, Rating, GenreScore, Feedback
from werkzeug.security import generate_password_hash
from sqlalchemy import desc
import json

api_bp = Blueprint('api', __name__, url_prefix='/api')


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
            'genre_preferences': g.user.get_genre_preferences(),
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

    db.session.commit()
    return jsonify({'success': True})


@api_bp.route('/preferences/reset', methods=['POST'])
def reset_preferences():
    """Reset logged-in user's saved preferences back to defaults."""
    if not g.user:
        return jsonify({'error': 'Not logged in'}), 401

    g.user.algorithm_preference = 'algo1'
    g.user.ui_preference = '1'
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
    if not g.user:
        return jsonify({'feedback': []}), 200

    feedback_list = Feedback.query.filter_by(user_id=g.user.id).all()
    return jsonify({
        'feedback': [{
            'id': f.id,
            'feedback_type': f.feedback_type,
            'vote_choice': f.vote_choice,
            'voter_name': f.voter_name,
            'voter_email': f.voter_email,
            'feedback_text': f.feedback_text,
            'consent_agreed': f.consent_agreed,
            'created_at': f.created_at.isoformat(),
            'updated_at': f.updated_at.isoformat()
        } for f in feedback_list]
    })


@api_bp.route('/feedback/prefill', methods=['GET'])
def feedback_prefill():
    """Get default participant information and completed feedback types for logged users."""
    if not g.user:
        return jsonify({'logged_in': False, 'defaults': {}, 'submitted_types': []})

    user_rows = Feedback.query.filter_by(user_id=g.user.id).all()
    submitted_types = [row.feedback_type for row in user_rows]
    latest_row = Feedback.query.filter_by(user_id=g.user.id).order_by(desc(Feedback.updated_at), desc(Feedback.created_at)).first()

    return jsonify({
        'logged_in': True,
        'defaults': {
            'voter_name': (latest_row.voter_name if latest_row and latest_row.voter_name else g.user.username),
            'voter_email': (latest_row.voter_email if latest_row and latest_row.voter_email else g.user.email),
        },
        'submitted_types': submitted_types,
    })


@api_bp.route('/feedback', methods=['POST'])
def submit_feedback():
    """Submit feedback"""
    data = request.json
    feedback_type = data.get('feedback_type')
    vote_choice = data.get('vote_choice')
    submission_type = data.get('submission_type')
    valid_types = {'algo_ui1', 'algo_ui2', 'ui_algo1', 'ui_algo2'}
    valid_votes = {'algo1', 'algo2', 'ui1', 'ui2'}
    valid_submission_types = {'logged', 'anonymous'}

    if not feedback_type or not vote_choice or not submission_type:
        return jsonify({'error': 'Missing required fields'}), 400

    if feedback_type not in valid_types:
        return jsonify({'error': 'Invalid feedback type'}), 400

    if vote_choice not in valid_votes:
        return jsonify({'error': 'Invalid vote choice'}), 400

    if submission_type not in valid_submission_types:
        return jsonify({'error': 'Invalid submission type'}), 400

    if submission_type == 'logged' and not g.user:
        return jsonify({'error': 'Login required for logged submission'}), 401

    # For logged-in users, check if they already submitted this type
    if g.user and submission_type == 'logged':
        existing = Feedback.query.filter_by(
            user_id=g.user.id,
            feedback_type=feedback_type
        ).first()

        if existing:
            # Update existing feedback
            existing.vote_choice = vote_choice
            existing.voter_name = data.get('voter_name', g.user.username)
            existing.voter_email = data.get('voter_email', g.user.email)
            existing.feedback_text = data.get('feedback_text')
            existing.consent_agreed = data.get('consent_agreed', False)
            db.session.commit()
            return jsonify({
                'success': True,
                'feedback_id': existing.id,
                'updated_existing': True,
                'message': 'You have already submitted this evaluation type before. Your previous submission has been updated.'
            })

        # Create new feedback for logged user
        voter_name = data.get('voter_name', g.user.username)
        voter_email = data.get('voter_email', g.user.email)
        feedback = Feedback(
            user_id=g.user.id,
            submission_type=submission_type,
            feedback_type=feedback_type,
            vote_choice=vote_choice,
            voter_name=voter_name,
            voter_email=voter_email,
            feedback_text=data.get('feedback_text'),
            consent_agreed=data.get('consent_agreed', False)
        )
    else:
        # Anonymous feedback
        feedback = Feedback(
            submission_type=submission_type,
            feedback_type=feedback_type,
            vote_choice=vote_choice,
            voter_name=data.get('voter_name'),
            voter_email=data.get('voter_email'),
            feedback_text=data.get('feedback_text'),
            consent_agreed=False
        )

    db.session.add(feedback)
    db.session.commit()
    return jsonify({'success': True, 'feedback_id': feedback.id})


@api_bp.route('/feedback/progress', methods=['GET'])
def feedback_progress():
    """Return completion progress for each feedback type (logged users only)."""
    if not g.user:
        return jsonify({'error': 'Not logged in'}), 401

    all_types = ['algo_ui1', 'algo_ui2', 'ui_algo1', 'ui_algo2']
    submitted = {row.feedback_type for row in Feedback.query.filter_by(user_id=g.user.id).all()}
    progress = [{
        'feedback_type': ft,
        'completed': ft in submitted,
    } for ft in all_types]
    return jsonify({
        'completed_count': len(submitted.intersection(set(all_types))),
        'total_count': len(all_types),
        'progress': progress,
    })


@api_bp.route('/evaluation/summary', methods=['GET'])
def evaluation_summary():
    """Return aggregated vote counts for evaluation charts."""
    source_filter = request.args.get('source', 'all')
    if source_filter not in {'all', 'logged', 'anonymous'}:
        return jsonify({'error': 'Invalid source filter'}), 400

    types = ['algo_ui1', 'algo_ui2', 'ui_algo1', 'ui_algo2']

    def apply_source_filter(query):
        if source_filter == 'logged':
            return query.filter(Feedback.submission_type == 'logged')
        if source_filter == 'anonymous':
            return query.filter(Feedback.submission_type == 'anonymous')
        return query

    summary = {}
    for feedback_type in types:
        query = Feedback.query.filter_by(feedback_type=feedback_type)
        query = apply_source_filter(query)
        if feedback_type.startswith('algo_'):
            option_a_values = ['algo1']
            option_b_values = ['algo2']
        else:
            option_a_values = ['ui1']
            option_b_values = ['ui2']

        better_count = query.filter(Feedback.vote_choice.in_(option_a_values)).count()
        worse_count = query.filter(Feedback.vote_choice.in_(option_b_values)).count()
        total = better_count + worse_count

        summary[feedback_type] = {
            'better': better_count,
            'worse': worse_count,
            'total': total,
            'has_data': total > 0,
            'better_pct': round((better_count / total) * 100, 1) if total else 0,
            'worse_pct': round((worse_count / total) * 100, 1) if total else 0,
        }

    return jsonify({'summary': summary, 'source': source_filter})


@api_bp.route('/evaluation/votes', methods=['GET'])
def evaluation_votes():
    """Return vote records for admin table only."""
    if not g.user or not getattr(g.user, 'admin', False):
        return jsonify({'error': 'Admin access required'}), 403

    source_filter = request.args.get('source', 'all')
    type_filter = request.args.get('feedback_type', 'all')

    query = Feedback.query

    if source_filter in {'logged', 'anonymous'}:
        query = query.filter(Feedback.submission_type == source_filter)

    if type_filter in {'algo_ui1', 'algo_ui2', 'ui_algo1', 'ui_algo2'}:
        query = query.filter(Feedback.feedback_type == type_filter)

    rows = query.order_by(desc(Feedback.updated_at), desc(Feedback.created_at)).all()
    return jsonify({
        'votes': [{
            'id': row.id,
            'user_id': row.user_id,
            'submission_type': row.submission_type,
            'feedback_type': row.feedback_type,
            'vote_choice': row.vote_choice,
            'voter_name': row.voter_name,
            'voter_email': row.voter_email,
            'feedback_text': row.feedback_text,
            'consent_agreed': row.consent_agreed,
            'updated_at': row.updated_at.isoformat() if row.updated_at else None,
            'created_at': row.created_at.isoformat() if row.created_at else None,
        } for row in rows]
    })


@api_bp.route('/feedback/<int:feedback_id>', methods=['PUT'])
def update_feedback(feedback_id):
    """Update feedback (logged users only)"""
    if not g.user:
        return jsonify({'error': 'Not logged in'}), 401

    feedback = Feedback.query.get(feedback_id)
    if not feedback or feedback.user_id != g.user.id:
        return jsonify({'error': 'Feedback not found'}), 404

    data = request.json
    if 'vote_choice' in data:
        feedback.vote_choice = data['vote_choice']

    db.session.commit()
    return jsonify({'success': True})


# ==================== MIGRATION ====================
@api_bp.route('/migrate-ratings', methods=['POST'])
def migrate_ratings():
    """Migrate localStorage ratings to database (on login)"""
    if not g.user:
        return jsonify({'error': 'Not logged in'}), 401

    data = request.json
    ratings_data = data.get('ratings', [])
    genre_prefs = data.get('genrePreferences', {})
    algorithm = data.get('algorithm')
    ui_mode = data.get('uiMode')

    # Import ratings
    for rating in ratings_data:
        movie_id = rating.get('movieId')
        rating_value = rating.get('rating')

        if movie_id and rating_value:
            existing = Rating.query.filter_by(
                user_id=g.user.id,
                movie_id=movie_id
            ).first()

            if not existing:
                new_rating = Rating(
                    user_id=g.user.id,
                    movie_id=movie_id,
                    rating=rating_value
                )
                db.session.add(new_rating)

    # Import genre preferences
    if genre_prefs:
        existing_prefs = g.user.get_genre_preferences()
        for genre_id, score in genre_prefs.items():
            if str(genre_id) not in existing_prefs:
                existing_prefs[str(genre_id)] = score
        g.user.set_genre_preferences(existing_prefs)

    if algorithm in {'algo1', 'algo2'}:
        g.user.algorithm_preference = algorithm

    if ui_mode in {'1', '2'}:
        g.user.ui_preference = ui_mode

    db.session.commit()
    return jsonify({'success': True})
