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
    password_hash = db.Column(db.String(255), nullable=False)
    algorithm_preference = db.Column(db.String(20), default='algo1')  # algo1 or algo2
    ui_preference = db.Column(db.String(20), default='1')  # 1 (dark) or 2 (light)
    genre_preferences = db.Column(db.Text, default='{}')  # JSON string of {genre_id: score}
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    ratings = db.relationship('Rating', backref='user', lazy=True, cascade='all, delete-orphan')
    feedback = db.relationship('Feedback', backref='user', lazy=True, cascade='all, delete-orphan')
    genre_scores = db.relationship('GenreScore', backref='user', lazy=True, cascade='all, delete-orphan')
    
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


class GenreScore(db.Model):
    __tablename__ = 'genre_scores'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    genre_id = db.Column(db.Integer, nullable=False)
    score = db.Column(db.Float, nullable=False)  # 0-10 scale
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (db.UniqueConstraint('user_id', 'genre_id', name='unique_user_genre_score'),)


class Feedback(db.Model):
    __tablename__ = 'feedback'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)  # NULL if anonymous
    
    # Submission details
    submission_type = db.Column(db.String(20), nullable=False)  # 'logged' or 'anonymous'
    voter_name = db.Column(db.String(120), nullable=True)  # Full name (may differ from account)
    voter_email = db.Column(db.String(120), nullable=True)  # Email (may differ from account)
    
    # Feedback type and vote
    feedback_type = db.Column(db.String(50), nullable=False)  # 'algo_ui1', 'algo_ui2', 'ui_algo1', 'ui_algo2'
    vote_choice = db.Column(db.String(20), nullable=False)  # 'better', 'worse', 'same'
    
    # Consent
    consent_agreed = db.Column(db.Boolean, default=False)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Constraint: logged-in users can only submit each feedback type once
    __table_args__ = (
        db.UniqueConstraint('user_id', 'feedback_type', name='unique_user_feedback_type'),
    )


class Vote(db.Model):
    __tablename__ = 'votes'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    algorithm = db.Column(db.String(20), nullable=False)  # algo1 or algo2
    ui_variant = db.Column(db.String(20), nullable=False)
    vote = db.Column(db.String(20), nullable=False)  # 'better', 'worse', 'same'
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
