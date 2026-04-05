import os

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

    # Import and register functions
    from . import scrape
    app.register_blueprint(scrape.bp)

    from . import main
    app.register_blueprint(main.bp)
    
    from . import auth
    app.register_blueprint(auth.bp)

    return app