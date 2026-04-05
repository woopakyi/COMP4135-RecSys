"""
API Endpoints for Movie Recommender System
Handles ratings, profile, feedback, and data migration
"""

from flask import Blueprint, jsonify, request, g
from .models import db, User, Movie, Rating, GenreScore, Feedback
from werkzeug.security import generate_password_hash
import json

api_bp = Blueprint('api', __name__, url_prefix='/api')


# ==================== GENRES ====================
@api_bp.route('/genres', methods=['GET'])
def get_genres():
    """Get all available genres"""
    from .tools.data_tool import loadData
    _, genres, _ = loadData()
    return jsonify({'genres': genres.to_dict('records')})


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

    if not (1 <= rating <= 5):
        return jsonify({'error': 'Rating must be between 1 and 5'}), 400

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
            'consent_agreed': f.consent_agreed,
            'created_at': f.created_at.isoformat(),
            'updated_at': f.updated_at.isoformat()
        } for f in feedback_list]
    })


@api_bp.route('/feedback', methods=['POST'])
def submit_feedback():
    """Submit feedback"""
    data = request.json
    feedback_type = data.get('feedback_type')
    vote_choice = data.get('vote_choice')
    submission_type = data.get('submission_type')

    if not feedback_type or not vote_choice:
        return jsonify({'error': 'Missing required fields'}), 400

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
            existing.consent_agreed = data.get('consent_agreed', False)
            db.session.commit()
            return jsonify({'success': True, 'feedback_id': existing.id})

        # Create new feedback for logged user
        feedback = Feedback(
            user_id=g.user.id,
            submission_type=submission_type,
            feedback_type=feedback_type,
            vote_choice=vote_choice,
            voter_name=data.get('voter_name', g.user.username),
            voter_email=data.get('voter_email', g.user.email),
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
            consent_agreed=False
        )

    db.session.add(feedback)
    db.session.commit()
    return jsonify({'success': True, 'feedback_id': feedback.id})


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

    db.session.commit()
    return jsonify({'success': True})
