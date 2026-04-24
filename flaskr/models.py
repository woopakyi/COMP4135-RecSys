from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import json

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    google_sub = db.Column(db.String(120), unique=True, nullable=True)
    password_hash = db.Column(db.String(255), nullable=False)
    algorithm_preference = db.Column(db.String(10), default='algo1')  # algo1 or algo2
    ui_preference = db.Column(db.String(2), default='1')  # 1 (dark) or 2 (light)
    preferences_saved = db.Column(db.Boolean, default=False)
    genre_preferences = db.Column(db.Text, default='{}')  # JSON string of {genre_id: score}
    feedback_progress_types = db.Column(db.Text, default='[]')  # JSON array of completed participant feedback types
    participant_full_name = db.Column(db.String(120), nullable=True)
    participant_contact_email = db.Column(db.String(120), nullable=True)
    admin = db.Column(db.Boolean, default=False)  # Admin user flag
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    ratings = db.relationship('Rating', backref='user', lazy=True, cascade='all, delete-orphan')
    feedback = db.relationship('Feedback', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set user password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Verify password hash"""
        return check_password_hash(self.password_hash, password)
    
    def get_genre_preferences(self):
        """Get genre preferences as dictionary"""
        try:
            return json.loads(self.genre_preferences) if self.genre_preferences else {}
        except:
            return {}
    
    def set_genre_preferences(self, preferences_dict):
        """Set genre preferences from dictionary"""
        self.genre_preferences = json.dumps(preferences_dict)

    def get_feedback_progress_types(self):
        """Get feedback progress types as list."""
        try:
            raw = json.loads(self.feedback_progress_types) if self.feedback_progress_types else []
            if isinstance(raw, list):
                return raw
            return []
        except Exception:
            return []

    def set_feedback_progress_types(self, feedback_types):
        """Set feedback progress types from iterable."""
        if not isinstance(feedback_types, (list, tuple, set)):
            feedback_types = []
        normalized = []
        for item in feedback_types:
            if isinstance(item, str) and item not in normalized:
                normalized.append(item)
        self.feedback_progress_types = json.dumps(normalized)


class Movie(db.Model):
    __tablename__ = 'movies'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    year = db.Column(db.Integer)
    genres = db.Column(db.String(255))  # comma-separated genre IDs
    imdb_id = db.Column(db.String(20))
    image_url = db.Column(db.String(500))
    
    # Relationships
    ratings = db.relationship('Rating', backref='movie', lazy=True, cascade='all, delete-orphan')


class Rating(db.Model):
    __tablename__ = 'ratings'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    movie_id = db.Column(db.Integer, db.ForeignKey('movies.id'), nullable=False)
    rating = db.Column(db.Float, nullable=False)  # 1-5 scale
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (db.UniqueConstraint('user_id', 'movie_id', name='unique_user_movie_rating'),)


class Feedback(db.Model):
    __tablename__ = 'feedback'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)  # NULL if anonymous

    # Contact details for participant submissions
    full_name = db.Column(db.String(120), nullable=True)
    contact_email = db.Column(db.String(120), nullable=True)
    
    # Submission details
    submission_type = db.Column(db.String(20), nullable=False)  # 'participant' or 'anonymous'
    
    # Feedback type and paired ratings
    feedback_type = db.Column(db.String(20), nullable=False)  # 'algorithm_eval', 'ui_eval'
    rating_option_a = db.Column(db.Integer, nullable=False)  # 1-10
    rating_option_b = db.Column(db.Integer, nullable=False)  # 1-10
    feedback_text = db.Column(db.Text, nullable=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        db.Index('idx_feedback_type_submission', 'feedback_type', 'submission_type'),
    )
