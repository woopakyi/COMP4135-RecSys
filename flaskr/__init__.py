import os
import pandas as pd

from flask import Flask


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    
    # Database configuration - use PostgreSQL on Railway, SQLite locally
    database_url = os.getenv('DATABASE_URL')
    if database_url and database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    
    app.config.from_mapping(
        SECRET_KEY=os.getenv('SECRET_KEY', 'dev'),
        SQLALCHEMY_DATABASE_URI=database_url or f"sqlite:///{os.path.join(app.instance_path, 'flaskr.sqlite')}",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        GOOGLE_CLIENT_ID=os.getenv('GOOGLE_CLIENT_ID', ''),
        GOOGLE_CLIENT_SECRET=os.getenv('GOOGLE_CLIENT_SECRET', ''),
        GOOGLE_REDIRECT_URI=os.getenv('GOOGLE_REDIRECT_URI', ''),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # Initialize database
    from .models import db
    db.init_app(app)
    
    # Create tables on app startup
    with app.app_context():
        db.create_all()
        seed_movies_from_csv(app)
        try:
            from sqlalchemy import text
            db.session.execute(text("ALTER TABLE feedback ADD COLUMN IF NOT EXISTS feedback_text TEXT"))
            db.session.execute(text("ALTER TABLE feedback ADD COLUMN IF NOT EXISTS rating_option_a INTEGER"))
            db.session.execute(text("ALTER TABLE feedback ADD COLUMN IF NOT EXISTS rating_option_b INTEGER"))
            db.session.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS google_sub VARCHAR(120)"))
            db.session.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS preferences_saved BOOLEAN DEFAULT FALSE"))
            db.session.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS feedback_progress_types TEXT DEFAULT '[]'"))
            db.session.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS participant_full_name VARCHAR(120)"))
            db.session.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS participant_contact_email VARCHAR(120)"))
            db.session.execute(text("ALTER TABLE feedback ADD COLUMN IF NOT EXISTS full_name VARCHAR(120)"))
            db.session.execute(text("ALTER TABLE feedback ADD COLUMN IF NOT EXISTS contact_email VARCHAR(120)"))
            # Keep user table compact: voter_name/voter_email live in feedback table.
            if (app.config.get('SQLALCHEMY_DATABASE_URI') or '').startswith('postgresql://'):
                db.session.execute(text("ALTER TABLE users DROP COLUMN IF EXISTS full_name"))
                db.session.execute(text("ALTER TABLE users DROP COLUMN IF EXISTS contact_email"))
                db.session.execute(text("ALTER TABLE users DROP COLUMN IF EXISTS google_login_only"))
                db.session.execute(text("ALTER TABLE feedback DROP CONSTRAINT IF EXISTS unique_user_feedback_type"))
                db.session.execute(text("ALTER TABLE feedback ALTER COLUMN submission_type TYPE VARCHAR(20)"))
                db.session.execute(text("UPDATE feedback SET submission_type = 'participant' WHERE submission_type = 'logged'"))
                db.session.execute(text("ALTER TABLE feedback DROP COLUMN IF EXISTS updated_at"))
                db.session.execute(text("DROP INDEX IF EXISTS idx_feedback_updated_at"))
                db.session.execute(text("DROP TABLE IF EXISTS genre_scores"))
                db.session.execute(text("DROP TABLE IF EXISTS votes"))
            db.session.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_google_sub ON users(google_sub)"))
            db.session.execute(text("CREATE INDEX IF NOT EXISTS idx_feedback_type_submission ON feedback(feedback_type, submission_type)"))

            # One-time tidy-up: rebuild feedback table without legacy columns and with ordered fields.
            if (app.config.get('SQLALCHEMY_DATABASE_URI') or '').startswith('postgresql://'):
                from .models import Feedback
                cols = db.session.execute(text("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'feedback'
                """)).fetchall()
                col_names = {row[0] for row in cols}
                legacy_cols = {'voter_name', 'voter_email', 'consent_agreed'}
                needs_rebuild = bool(col_names.intersection(legacy_cols))
                if needs_rebuild:
                    db.session.execute(text('DROP TABLE IF EXISTS feedback CASCADE'))
                    db.session.commit()
                    Feedback.__table__.create(db.engine, checkfirst=True)
                    db.session.execute(text("CREATE INDEX IF NOT EXISTS idx_feedback_type_submission ON feedback(feedback_type, submission_type)"))
            db.session.commit()
        except Exception:
            db.session.rollback()

        # Check if schema is out of sync with PostgreSQL and fix it
        try:
            from .models import User
            User.query.first()  # Test query to check if columns exist
        except Exception as e:
            if 'UndefinedColumn' in str(type(e).__name__) or 'column' in str(e).lower():
                print("⚠ Database schema out of sync, recreating...")
                try:
                    db.drop_all()
                    db.create_all()
                    seed_movies_from_csv(app)
                    print("✅ Database schema recovered")
                except Exception as recovery_error:
                    print(f"❌ Recovery failed: {recovery_error}")
                    raise
            else:
                raise

    # Import and register functions
    from . import scrape
    app.register_blueprint(scrape.bp)

    from . import main
    app.register_blueprint(main.bp)
    
    from . import auth
    app.register_blueprint(auth.bp)

    from . import api
    app.register_blueprint(api.api_bp)

    return app


def seed_movies_from_csv(app):
    from .models import Movie, db

    if Movie.query.first() is not None:
        return

    csv_path = os.path.join(app.root_path, 'static', 'ml_data', 'movie_info.csv')
    movies_df = pd.read_csv(csv_path)

    movie_records = []
    for _, row in movies_df.iterrows():
        movie_records.append(Movie(
            id=int(row['movieId']),
            title=str(row['title']),
            year=int(row['year']) if pd.notna(row['year']) else None,
            genres=str(row['genres']),
            image_url=str(row['cover_url']),
        ))

    db.session.add_all(movie_records)
    db.session.commit()