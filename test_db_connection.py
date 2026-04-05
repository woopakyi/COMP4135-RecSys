#!/usr/bin/env python
"""Test PostgreSQL connection from Railway"""

import os
import sys

# Set the DATABASE_URL environment variable
os.environ['DATABASE_URL'] = 'postgresql://postgres:PgFmmvXhJzEgylpvTIIACKalNZdHcnbd@postgres.railway.internal:5432/railway'

try:
    from flaskr import create_app
    
    print("Creating Flask app...")
    app = create_app()
    
    with app.app_context():
        print("Testing database connection...")
        from flaskr.models import db, User, Movie, Rating
        
        # Test query
        user_count = User.query.count()
        movie_count = Movie.query.count()
        rating_count = Rating.query.count()
        
        print("✓ Database connection successful!")
        print(f"  Users: {user_count}")
        print(f"  Movies: {movie_count}")
        print(f"  Ratings: {rating_count}")
        print("\nAll tables created successfully on PostgreSQL!")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
