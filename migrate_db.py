#!/usr/bin/env python
"""
Database migration script - Run this to fix the schema
Usage: python migrate_db.py
"""

import os
import sys

# Add the flaskr package to path
sys.path.insert(0, os.path.dirname(__file__))

from flaskr import create_app
from flaskr.models import db

if __name__ == '__main__':
    app = create_app()
    
    with app.app_context():
        print("Dropping all tables...")
        db.drop_all()
        print("✓ Tables dropped")
        
        print("Creating new tables...")
        db.create_all()
        print("✓ Tables created")
        
        print("Seeding movies from CSV...")
        from flaskr import seed_movies_from_csv
        seed_movies_from_csv(app)
        print("✓ Movies seeded")
        
        print("\n✅ Database migration completed successfully!")
        print("The app should now work without errors.")
