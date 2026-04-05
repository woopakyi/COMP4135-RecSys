from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flaskr.models import db, User
from sqlalchemy.exc import IntegrityError

bp = Blueprint('auth', __name__, url_prefix='/auth')


@bp.route('/register', methods=['GET', 'POST'])
def register():
    """Register a new user"""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        # Validation
        errors = []
        
        if not username:
            errors.append('Username is required')
        elif len(username) < 3:
            errors.append('Username must be at least 3 characters')
            
        if not email:
            errors.append('Email is required')
        elif '@' not in email:
            errors.append('Invalid email format')
            
        if not password:
            errors.append('Password is required')
        elif len(password) < 6:
            errors.append('Password must be at least 6 characters')
            
        if password != confirm_password:
            errors.append('Passwords do not match')
        
        if errors:
            for error in errors:
                flash(error, 'error')
            return render_template('auth/register.html')
        
        # Create user
        try:
            user = User(
                username=username,
                email=email,
            )
            user.set_password(password)
            
            db.session.add(user)
            db.session.commit()
            
            flash(f'Welcome {username}! You can now log in.', 'success')
            return redirect(url_for('auth.login'))
            
        except IntegrityError:
            db.session.rollback()
            flash('Username or email already exists', 'error')
            return render_template('auth/register.html')
        except Exception as e:
            db.session.rollback()
            flash(f'Registration failed: {str(e)}', 'error')
            return render_template('auth/register.html')
    
    return render_template('auth/register.html')


@bp.route('/login', methods=['GET', 'POST'])
def login():
    """Log in a user"""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        if not username or not password:
            flash('Username and password are required', 'error')
            return render_template('auth/login.html')
        
        # Find user
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            # Set session
            session.clear()
            session['user_id'] = user.id
            session['username'] = user.username
            session['algorithm'] = user.algorithm
            session['ui_variant'] = user.ui_variant
            
            flash(f'Welcome back, {username}!', 'success')
            return redirect(url_for('main.index'))
        else:
            flash('Invalid username or password', 'error')
            return render_template('auth/login.html')
    
    return render_template('auth/login.html')


@bp.route('/logout')
def logout():
    """Log out the current user"""
    username = session.get('username', 'User')
    session.clear()
    flash(f'Goodbye, {username}!', 'success')
    return redirect(url_for('main.index'))


@bp.before_app_request
def load_logged_in_user():
    """Load user info into g for use in templates"""
    from flask import g
    
    user_id = session.get('user_id')
    
    if user_id is None:
        g.user = None
    else:
        g.user = db.session.get(User, user_id)
