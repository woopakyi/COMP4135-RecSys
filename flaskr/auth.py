from flask import Blueprint, render_template, request, redirect, url_for, session, flash, current_app
from flaskr.models import db, User, Rating, Feedback
from sqlalchemy.exc import IntegrityError
import secrets
import requests
from urllib.parse import urlencode

bp = Blueprint('auth', __name__, url_prefix='/auth')


def _google_configured():
    return bool(
        current_app.config.get('GOOGLE_CLIENT_ID')
        and current_app.config.get('GOOGLE_CLIENT_SECRET')
    )


def _google_redirect_uri():
    configured = current_app.config.get('GOOGLE_REDIRECT_URI')
    if configured:
        return configured
    return url_for('auth.google_callback', _external=True)


def _is_simple_email(email):
    if not email:
        return False
    local_part, separator, domain_part = email.partition('@')
    return bool(separator and local_part and domain_part and '@' not in domain_part)


def _merge_user_accounts(primary_user, secondary_user):
    """Merge secondary_user into primary_user and remove secondary_user."""
    if not primary_user or not secondary_user or primary_user.id == secondary_user.id:
        return

    # Keep admin if either account is admin.
    primary_user.admin = bool(primary_user.admin or secondary_user.admin)

    # Merge ratings: keep the newest timestamp when duplicates exist.
    primary_ratings = {r.movie_id: r for r in Rating.query.filter_by(user_id=primary_user.id).all()}
    secondary_ratings = Rating.query.filter_by(user_id=secondary_user.id).all()
    for rating in secondary_ratings:
        existing = primary_ratings.get(rating.movie_id)
        if not existing:
            rating.user_id = primary_user.id
        else:
            if rating.timestamp and (not existing.timestamp or rating.timestamp > existing.timestamp):
                existing.rating = rating.rating
                existing.timestamp = rating.timestamp
            db.session.delete(rating)

    # Merge feedback: preserve all submissions by reassigning ownership.
    secondary_feedback = Feedback.query.filter_by(user_id=secondary_user.id).all()
    for row in secondary_feedback:
        row.user_id = primary_user.id

    # Merge preference fields.
    if primary_user.algorithm_preference in (None, '') and secondary_user.algorithm_preference:
        primary_user.algorithm_preference = secondary_user.algorithm_preference
    if primary_user.ui_preference in (None, '') and secondary_user.ui_preference:
        primary_user.ui_preference = secondary_user.ui_preference

    primary_genres = primary_user.get_genre_preferences()
    secondary_genres = secondary_user.get_genre_preferences()
    for genre_id, score in secondary_genres.items():
        if genre_id not in primary_genres:
            primary_genres[genre_id] = score
    primary_user.set_genre_preferences(primary_genres)

    db.session.delete(secondary_user)


@bp.route('/register', methods=['GET', 'POST'])
def register():
    """Register a new user"""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        if email.endswith('@gmail.com'):
            return redirect(url_for('auth.google_login'))
        
        # Validation
        errors = []
        
        if not username:
            errors.append('Username is required')
        elif len(username) < 3:
            errors.append('Username must be at least 3 characters')
            
        if not email:
            errors.append('Email is required')
        elif not _is_simple_email(email):
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
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        
        if not email or not password:
            flash('Email and password are required', 'error')
            return render_template('auth/login.html')
        
        # Find user
        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            # Set session
            session.clear()
            session['user_id'] = user.id
            session['username'] = user.username
            session['algorithm'] = user.algorithm_preference
            session['ui_variant'] = user.ui_preference

            return redirect(url_for('main.index'))
        else:
            flash('Invalid email or password', 'error')
            return render_template('auth/login.html')
    
    return render_template('auth/login.html', google_login_enabled=_google_configured())


@bp.route('/google/login')
def google_login():
    """Start Google OAuth login flow."""
    if not _google_configured():
        flash('Google login is not configured on the server.', 'error')
        return redirect(url_for('auth.login'))

    state = secrets.token_urlsafe(24)
    session['google_oauth_state'] = state

    params = {
        'client_id': current_app.config.get('GOOGLE_CLIENT_ID'),
        'redirect_uri': _google_redirect_uri(),
        'response_type': 'code',
        'scope': 'openid email profile',
        'state': state,
        'prompt': 'select_account'
    }
    auth_url = 'https://accounts.google.com/o/oauth2/v2/auth?' + urlencode(params)
    return redirect(auth_url)


@bp.route('/google/link')
def google_link():
    """Start Google OAuth flow for linking Google account to current logged-in account."""
    user_id = session.get('user_id')
    if not user_id:
        flash('Please sign in first to link Google account.', 'error')
        return redirect(url_for('auth.login'))

    if not _google_configured():
        flash('Google login is not configured on the server.', 'error')
        return redirect(url_for('main.profile'))

    state = secrets.token_urlsafe(24)
    session['google_oauth_state'] = state
    session['google_link_user_id'] = user_id

    params = {
        'client_id': current_app.config.get('GOOGLE_CLIENT_ID'),
        'redirect_uri': _google_redirect_uri(),
        'response_type': 'code',
        'scope': 'openid email profile',
        'state': state,
        'prompt': 'select_account'
    }
    auth_url = 'https://accounts.google.com/o/oauth2/v2/auth?' + urlencode(params)
    return redirect(auth_url)


@bp.route('/google/callback')
def google_callback():
    """Handle Google OAuth callback."""
    if not _google_configured():
        flash('Google login is not configured on the server.', 'error')
        return redirect(url_for('auth.login'))

    state = request.args.get('state', '')
    expected_state = session.pop('google_oauth_state', '')
    if not state or state != expected_state:
        flash('Invalid OAuth state. Please try again.', 'error')
        return redirect(url_for('auth.login'))

    code = request.args.get('code')
    if not code:
        flash('Google authorization failed. Please try again.', 'error')
        return redirect(url_for('auth.login'))

    try:
        token_resp = requests.post(
            'https://oauth2.googleapis.com/token',
            data={
                'code': code,
                'client_id': current_app.config.get('GOOGLE_CLIENT_ID'),
                'client_secret': current_app.config.get('GOOGLE_CLIENT_SECRET'),
                'redirect_uri': _google_redirect_uri(),
                'grant_type': 'authorization_code',
            },
            timeout=15,
        )
        token_resp.raise_for_status()
        token_data = token_resp.json()
        access_token = token_data.get('access_token')
        if not access_token:
            flash('Google login failed to return access token.', 'error')
            return redirect(url_for('auth.login'))

        userinfo_resp = requests.get(
            'https://www.googleapis.com/oauth2/v3/userinfo',
            headers={'Authorization': f'Bearer {access_token}'},
            timeout=15,
        )
        userinfo_resp.raise_for_status()
        profile = userinfo_resp.json()
        email = (profile.get('email') or '').strip().lower()
        google_sub = (profile.get('sub') or '').strip()
        if not email:
            flash('Google account has no usable email.', 'error')
            return redirect(url_for('auth.login'))

        username = (profile.get('name') or email.split('@')[0] or 'google-user').strip()
        link_user_id = session.pop('google_link_user_id', None)

        if link_user_id:
            primary_user = db.session.get(User, link_user_id)
            if not primary_user:
                flash('Account link target not found. Please sign in and try again.', 'error')
                return redirect(url_for('auth.login'))

            if primary_user.google_sub and google_sub and primary_user.google_sub == google_sub:
                flash('This account is already linked with Google.', 'success')
                return redirect(url_for('main.profile'))

            secondary_user = None
            if google_sub:
                secondary_user = User.query.filter_by(google_sub=google_sub).first()
            if not secondary_user:
                secondary_user = User.query.filter_by(email=email).first()

            if secondary_user and secondary_user.id != primary_user.id:
                flash('This Google account is already linked with another user. Please use a different Google account or sign in with your current email.', 'error')
                return redirect(url_for('main.profile'))

            primary_user.google_sub = google_sub or primary_user.google_sub
            # Replace email and disable password login by randomizing password hash.
            primary_user.email = email
            primary_user.set_password(secrets.token_urlsafe(32))
            try:
                db.session.commit()
            except IntegrityError:
                db.session.rollback()
                flash('Switch failed due to account conflict. Please use another Google account.', 'error')
                return redirect(url_for('main.profile'))

            session.clear()
            session['user_id'] = primary_user.id
            session['username'] = primary_user.username
            session['algorithm'] = primary_user.algorithm_preference
            session['ui_variant'] = primary_user.ui_preference

            flash('Account switched to Google Sign-in successfully.', 'success')
            return redirect(url_for('main.profile'))

        user = None
        if google_sub:
            user = User.query.filter_by(google_sub=google_sub).first()
        if not user:
            user = User.query.filter_by(email=email).first()

        if not user:
            base_username = username
            suffix = 1
            while User.query.filter_by(username=username).first():
                suffix += 1
                username = f'{base_username}{suffix}'

            user = User(
                username=username,
                email=email,
                google_sub=google_sub or None,
            )
            user.set_password(secrets.token_urlsafe(32))
            db.session.add(user)
            db.session.commit()
        else:
            user.google_sub = google_sub or user.google_sub
            try:
                db.session.commit()
            except IntegrityError:
                db.session.rollback()
                flash('Google account link conflict detected. Please sign in with Google and retry linking.', 'error')
                return redirect(url_for('auth.login'))

        session.clear()
        session['user_id'] = user.id
        session['username'] = user.username
        session['algorithm'] = user.algorithm_preference
        session['ui_variant'] = user.ui_preference

        return redirect(url_for('main.index'))
    except Exception:
        db.session.rollback()
        flash('Google login failed. Please try again later.', 'error')
        return redirect(url_for('auth.login'))


@bp.route('/logout')
def logout():
    """Log out the current user"""
    session.clear()
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
