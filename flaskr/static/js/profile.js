const { createApp, ref, computed, onMounted } = Vue;
const profileRoot = document.getElementById('profile-app');
const FEEDBACK_PARTICIPANT_PROFILE_KEY = 'feedback_participant_profile';
const FEEDBACK_PARTICIPANT_TYPES_KEY = 'feedback_participant_submitted_types';

const app = createApp({
    data() {
        const initialLoggedIn = profileRoot?.dataset.loggedIn === 'true';
        const initialUsername = profileRoot?.dataset.initialUsername || '';
        return {
            isLoggedIn: initialLoggedIn,
            hasGoogleLinked: profileRoot?.dataset.hasGoogleLinked === 'true',
            profile: {
                username: initialUsername,
                algorithm: 'algo1',
                uiMode: '1'
            },
            genreScores: {},
            genres: [],
            ratings: [],
            currentRatingsPage: 1,
            ratingsPerPage: 10,
            feedbackProgress: {
                completed_count: 0,
                total_count: 4,
                progress: []
            },
            statusMessage: '',
            statusType: 'success',
            inlineStatus: {
                saveUsername: false,
                resetPreferences: false,
                clearAllRatings: false
            }
        };
    },
    computed: {
        hasRatings() {
            return this.ratings.length > 0;
        },
        paginatedRatings() {
            const start = (this.currentRatingsPage - 1) * this.ratingsPerPage;
            const end = start + this.ratingsPerPage;
            return this.ratings.slice(start, end);
        },
        totalRatingsPages() {
            return Math.ceil(this.ratings.length / this.ratingsPerPage);
        }
    },
    methods: {
        createGenreScoreDefaults() {
            return this.genres.reduce((scores, genre) => {
                scores[genre.id] = 0;
                return scores;
            }, {});
        },

        applyGenreScoreDefaults(overrides = {}) {
            this.genreScores = {
                ...this.createGenreScoreDefaults(),
                ...overrides
            };
        },

        applyUiTheme(uiValue) {
            document.body.classList.remove('ui-dark', 'ui-light');
            if (uiValue === '2') {
                document.body.classList.add('ui-light');
            } else {
                document.body.classList.add('ui-dark');
            }
        },

        persistUiMode(uiValue) {
            document.cookie = `user_ui=${uiValue}; path=/`;
            SettingsStorage.setUiMode(uiValue);
        },

        getCookieValue(name) {
            const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const match = document.cookie.match(new RegExp('(?:^|; )' + escaped + '=([^;]*)'));
            return match ? decodeURIComponent(match[1]) : '';
        },

        getGuestRatingsFromCookie() {
            const raw = this.getCookieValue('user_rates');
            if (!raw) {
                return [];
            }
            return raw.split(',').map((token) => {
                const parts = token.split('|');
                if (parts.length < 3) {
                    return null;
                }
                const movieId = parseInt(parts[1], 10);
                const rating = parseInt(parts[2], 10);
                const ts = parts[3] ? parseInt(parts[3], 10) : Math.floor(Date.now() / 1000);
                if (Number.isNaN(movieId) || Number.isNaN(rating)) {
                    return null;
                }
                return {
                    movieId,
                    rating,
                    timestamp: new Date(ts * 1000).toISOString()
                };
            }).filter(Boolean);
        },

        getGuestRateRecordsFromCookie() {
            const raw = this.getCookieValue('user_rates');
            if (!raw) {
                return [];
            }
            return raw.split(',').filter(Boolean);
        },

        writeGuestRateRecordsToCookie(records) {
            document.cookie = `user_rates=${encodeURIComponent(records.join(','))}; path=/`;
        },

        clearCookie(name) {
            document.cookie = `${name}=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/`;
        },

        async loadGenres() {
            try {
                const response = await fetch('/api/genres');
                const data = await response.json();
                this.genres = (data.genres || []).map((genre) => ({
                    ...genre,
                    name: genre.name === '(no genres listed)' ? '(no genres listed)' : genre.name
                }));
                this.applyGenreScoreDefaults(this.genreScores);
            } catch (error) {
                console.error('Error loading genres:', error);
            }
        },

        async loadUserProfile() {
            if (!this.isLoggedIn) {
                this.profile.algorithm = SettingsStorage.getAlgorithm() || 'algo1';
                this.profile.uiMode = SettingsStorage.getUiMode() || '1';
                this.genreScores = GenreStorage.getPreferencesForAPI();
                return;
            }

            try {
                const response = await fetch('/api/profile');
                if (!response.ok) throw new Error('Failed to load profile');
                const data = await response.json();
                
                this.profile.username = data.user.username || profileRoot?.dataset.initialUsername || '';
                this.profile.algorithm = data.user.algorithm_preference || 'algo1';
                this.profile.uiMode = data.user.ui_preference || '1';
                this.applyUiTheme(this.profile.uiMode);
                this.persistUiMode(this.profile.uiMode);
                this.applyGenreScoreDefaults(data.user.genre_preferences || {});
            } catch (error) {
                console.error('Error loading profile:', error);
                this.profile.username = profileRoot?.dataset.initialUsername || this.profile.username;
                this.showStatus('Error loading profile', 'error');
            }
        },

        async loadRatings() {
            if (!this.isLoggedIn) {
                let localRatings = RatingStorage.getRatingsArray();
                if (localRatings.length === 0) {
                    localRatings = this.getGuestRatingsFromCookie();
                }
                if (localRatings.length === 0) {
                    this.ratings = [];
                    return;
                }

                try {
                    const ids = localRatings.map((item) => item.movieId).join(',');
                    const response = await fetch(`/api/movies?ids=${encodeURIComponent(ids)}`);
                    const data = response.ok ? await response.json() : { movies: [] };
                    const movieMap = {};
                    (data.movies || []).forEach((movie) => {
                        movieMap[movie.id] = movie;
                    });

                    this.ratings = localRatings.map((item) => ({
                        id: `local-${item.movieId}`,
                        movie_id: item.movieId,
                        movie: {
                            id: item.movieId,
                            title: movieMap[item.movieId]?.title || `Movie #${item.movieId}`,
                            year: movieMap[item.movieId]?.year || 'N/A',
                            image_url: movieMap[item.movieId]?.image_url || ''
                        },
                        rating: item.rating,
                        timestamp: item.timestamp
                    }));
                } catch (error) {
                    console.error('Error loading local ratings:', error);
                    this.ratings = localRatings.map((item) => ({
                        id: `local-${item.movieId}`,
                        movie_id: item.movieId,
                        movie: {
                            id: item.movieId,
                            title: `Movie #${item.movieId}`,
                            year: 'N/A',
                            image_url: ''
                        },
                        rating: item.rating,
                        timestamp: item.timestamp
                    }));
                }
                return;
            }

            try {
                const response = await fetch('/api/ratings');
                if (!response.ok) throw new Error('Failed to load ratings');
                const data = await response.json();
                this.ratings = data.ratings || [];
            } catch (error) {
                console.error('Error loading ratings:', error);
            }
        },

        async loadFeedbackProgress() {
            if (!this.isLoggedIn) {
                this.feedbackProgress = this.getGuestFeedbackProgress();
                return;
            }

            try {
                const response = await fetch('/api/feedback/progress');
                if (!response.ok) throw new Error('Failed to load feedback progress');
                const data = await response.json();
                this.feedbackProgress = data;
            } catch (error) {
                console.error('Error loading feedback progress:', error);
            }
        },

        getGuestFeedbackProgress() {
            const allTypes = ['algo_ui1', 'algo_ui2', 'ui_algo1', 'ui_algo2'];
            let submittedTypes = [];

            try {
                const rawSubmittedTypes = localStorage.getItem(FEEDBACK_PARTICIPANT_TYPES_KEY) || '[]';
                const parsed = JSON.parse(rawSubmittedTypes);
                submittedTypes = Array.isArray(parsed) ? parsed : [];

                // Read the profile payload to keep the key contract consistent.
                // The profile table currently renders completion chips only.
                JSON.parse(localStorage.getItem(FEEDBACK_PARTICIPANT_PROFILE_KEY) || '{}');
            } catch (_error) {
                submittedTypes = [];
            }

            const submittedSet = new Set(submittedTypes.filter((item) => allTypes.includes(item)));
            const progress = allTypes.map((feedbackType) => ({
                feedback_type: feedbackType,
                completed: submittedSet.has(feedbackType),
                submission_count: null
            }));

            return {
                scope: 'local_storage',
                completed_count: progress.filter((item) => item.completed).length,
                total_count: allTypes.length,
                progress
            };
        },

        async updateProfile() {
            if (!this.isLoggedIn) {
                this.showStatus('Please log in to update profile', 'error');
                return;
            }

            try {
                const response = await fetch('/api/profile', {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        username: this.profile.username,
                        algorithm_preference: this.profile.algorithm,
                        ui_preference: this.profile.uiMode
                    })
                });

                if (!response.ok) throw new Error('Failed to update profile');
                this.showInlineStatus('saveUsername');
            } catch (error) {
                console.error('Error updating profile:', error);
                this.showStatus('Error updating username', 'error');
            }
        },

        async updatePreference(type, value) {
            if (type === 'algorithm') {
                this.profile.algorithm = value;
            } else if (type === 'ui') {
                this.profile.uiMode = value;
                this.applyUiTheme(value);
                this.persistUiMode(value);
            }

            if (!this.isLoggedIn) {
                if (type === 'algorithm') {
                    SettingsStorage.setAlgorithm(value);
                }
                return;
            }

            try {
                const response = await fetch('/api/profile', {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        [type === 'algorithm' ? 'algorithm_preference' : 'ui_preference']: value
                    })
                });

                if (!response.ok) throw new Error('Failed to update preference');
                this.showStatus('Preference updated', 'success');
            } catch (error) {
                console.error('Error updating preference:', error);
                this.showStatus('Error updating preference', 'error');
            }
        },

        async updateGenreScore(genreId, score) {
            // If score is not provided (called from older code), use the current score
            if (score === undefined) {
                score = this.genreScores[genreId] || 0;
            }
            this.genreScores[genreId] = score;

            if (!this.isLoggedIn) {
                GenreStorage.savePreference(genreId, score);
                return;
            }

            try {
                const response = await fetch('/api/genre-preferences', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        genre_id: genreId,
                        score: score
                    })
                });

                if (!response.ok) throw new Error('Failed to update genre preference');
                this.showStatus('Genre rating updated', 'success');
            } catch (error) {
                console.error('Error updating genre preference:', error);
                this.showStatus('Error updating genre rating', 'error');
            }
        },

        async resetPreferences() {
            if (!this.isLoggedIn) {
                SettingsStorage.setAlgorithm('algo1');
                SettingsStorage.setUiMode('1');
                GenreStorage.clearAllPreferences();
                document.cookie = 'user_started=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/';
                document.cookie = 'user_algorithm=1; path=/';
                document.cookie = 'user_ui=1; path=/';
                document.cookie = 'user_genre_scores=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/';
                this.profile.algorithm = 'algo1';
                this.profile.uiMode = '1';
                this.applyUiTheme('1');
                this.applyGenreScoreDefaults();
                this.showInlineStatus('resetPreferences');
                return;
            }

            try {
                const response = await fetch('/api/preferences/reset', { method: 'POST' });
                if (!response.ok) throw new Error('Failed to reset preferences');
                document.cookie = 'user_started=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/';
                document.cookie = 'user_algorithm=1; path=/';
                document.cookie = 'user_ui=1; path=/';
                document.cookie = 'user_genre_scores=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/';
                this.profile.algorithm = 'algo1';
                this.profile.uiMode = '1';
                this.applyUiTheme('1');
                this.applyGenreScoreDefaults();
                this.showInlineStatus('resetPreferences');
            } catch (error) {
                console.error('Error resetting preferences:', error);
                this.showStatus('Error resetting preferences', 'error');
            }
        },

        async deleteRating(ratingId) {
            const row = this.ratings.find((r) => r.id === ratingId);
            const movieId = row ? row.movie_id : null;

            if (!this.isLoggedIn) {
                const idText = String(ratingId);
                const movieId = parseInt(idText.replace('local-', ''), 10);
                if (!Number.isNaN(movieId)) {
                    RatingStorage.deleteRating(movieId);

                    const remainingRecords = this.getGuestRateRecordsFromCookie().filter((record) => {
                        const parts = record.split('|');
                        return parseInt(parts[1], 10) !== movieId;
                    });
                    if (remainingRecords.length > 0) {
                        this.writeGuestRateRecordsToCookie(remainingRecords);
                    } else {
                        this.clearCookie('user_rates');
                    }
                }
                this.ratings = this.ratings.filter(r => r.id !== ratingId);
                this.showStatus('Rating deleted', 'success');
                return;
            }

            try {
                const response = movieId
                    ? await fetch(`/api/ratings/by-movie/${movieId}`, { method: 'DELETE' })
                    : await fetch(`/api/ratings/${ratingId}`, { method: 'DELETE' });
                if (!response.ok) throw new Error('Failed to delete rating');
                
                if (movieId) {
                    RatingStorage.deleteRating(movieId);
                }
                const remainingRecords = this.getGuestRateRecordsFromCookie().filter((record) => {
                    const parts = record.split('|');
                    return parseInt(parts[1], 10) !== movieId;
                });
                if (remainingRecords.length > 0) {
                    this.writeGuestRateRecordsToCookie(remainingRecords);
                } else {
                    this.clearCookie('user_rates');
                }
                this.ratings = this.ratings.filter(r => r.id !== ratingId);
                this.showStatus('Rating deleted', 'success');
            } catch (error) {
                console.error('Error deleting rating:', error);
                this.showStatus('Error deleting rating', 'error');
            }
        },

        async clearAllRatings() {
            if (!this.isLoggedIn) {
                RatingStorage.clearAllRatings();
                this.clearCookie('user_rates');
                this.ratings = [];
                this.showInlineStatus('clearAllRatings');
                return;
            }

            try {
                const response = await fetch('/api/ratings/clear', { method: 'POST' });
                if (!response.ok) throw new Error('Failed to clear ratings');
                
                RatingStorage.clearAllRatings();
                this.clearCookie('user_rates');
                this.ratings = [];
                this.showInlineStatus('clearAllRatings');
            } catch (error) {
                console.error('Error clearing ratings:', error);
                this.showStatus('Error clearing ratings', 'error');
            }
        },

        formatDate(dateString) {
            const date = new Date(dateString);
            return date.toLocaleDateString();
        },

        formatRatingValue(value) {
            const rounded = Math.round(Number(value) || 0);
            return String(rounded);
        },

        formatFeedbackType(type) {
            const types = {
                'algo_ui1': 'Algorithm (UI 1)',
                'algo_ui2': 'Algorithm (UI 2)',
                'ui_algo1': 'UI (Algo 1)',
                'ui_algo2': 'UI (Algo 2)',
                'algorithm_eval': 'Algorithm Evaluation',
                'ui_eval': 'User Interface Evaluation'
            };
            return types[type] || type;
        },

        showStatus(message, type) {
            this.statusMessage = message;
            this.statusType = type;
            setTimeout(() => {
                this.statusMessage = '';
            }, 3000);
        },

        showInlineStatus(key) {
            if (this.inlineStatus.hasOwnProperty(key)) {
                this.inlineStatus[key] = true;
                setTimeout(() => {
                    this.inlineStatus[key] = false;
                }, 2000);
            }
        }
    },

    mounted() {
        this.loadGenres();
        this.loadUserProfile();
        this.loadRatings();
        this.loadFeedbackProgress();
    }
});

app.config.compilerOptions.delimiters = ['[[', ']]'];
app.mount('#profile-app');
