const { createApp, ref, computed, onMounted } = Vue;
const profileRoot = document.getElementById('profile-app');

const app = createApp({
    data() {
        return {
            isLoggedIn: !!document.querySelector('meta[name="user"]'),
            profile: {
                username: '',
                algorithm: 'algo1',
                uiMode: '1'
            },
            genreScores: {},
            genres: [],
            ratings: [],
            feedback: [],
            currentRatingsPage: 1,
            currentFeedbackPage: 1,
            ratingsPerPage: 10,
            feedbackPerPage: 10,
            feedbackProgress: {
                completed_count: 0,
                total_count: 4,
                progress: []
            },
            statusMessage: '',
            statusType: 'success'
        };
    },
    computed: {
        hasRatings() {
            return this.ratings.length > 0;
        },
        hasFeedback() {
            return this.feedback.length > 0;
        },
        paginatedRatings() {
            const start = (this.currentRatingsPage - 1) * this.ratingsPerPage;
            const end = start + this.ratingsPerPage;
            return this.ratings.slice(start, end);
        },
        paginatedFeedback() {
            const start = (this.currentFeedbackPage - 1) * this.feedbackPerPage;
            const end = start + this.feedbackPerPage;
            return this.feedback.slice(start, end);
        },
        totalRatingsPages() {
            return Math.ceil(this.ratings.length / this.ratingsPerPage);
        },
        totalFeedbackPages() {
            return Math.ceil(this.feedback.length / this.feedbackPerPage);
        }
    },
    methods: {
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
                this.genreScores = data.user.genre_preferences || {};
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

        async loadFeedback() {
            if (!this.isLoggedIn) return;

            try {
                const response = await fetch('/api/feedback');
                if (!response.ok) throw new Error('Failed to load feedback');
                const data = await response.json();
                this.feedback = data.feedback || [];
            } catch (error) {
                console.error('Error loading feedback:', error);
            }
        },

        async loadFeedbackProgress() {
            if (!this.isLoggedIn) {
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
                this.showStatus('Profile updated successfully', 'success');
            } catch (error) {
                console.error('Error updating profile:', error);
                this.showStatus('Error updating profile', 'error');
            }
        },

        async updatePreference(type, value) {
            if (type === 'algorithm') {
                this.profile.algorithm = value;
            } else if (type === 'ui') {
                this.profile.uiMode = value;
            }

            if (!this.isLoggedIn) {
                if (type === 'algorithm') {
                    SettingsStorage.setAlgorithm(value);
                } else if (type === 'ui') {
                    SettingsStorage.setUiMode(value);
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
            if (!confirm('Reset algorithm, UI, and genre preferences?')) return;

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
                this.genreScores = {};
                this.showStatus('Preferences reset', 'success');
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
                this.genreScores = {};
                this.showStatus('Preferences reset', 'success');
            } catch (error) {
                console.error('Error resetting preferences:', error);
                this.showStatus('Error resetting preferences', 'error');
            }
        },

        async deleteRating(ratingId) {
            if (!confirm('Are you sure you want to delete this rating?')) return;

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
                const response = await fetch(`/api/ratings/${ratingId}`, { method: 'DELETE' });
                if (!response.ok) throw new Error('Failed to delete rating');
                
                this.ratings = this.ratings.filter(r => r.id !== ratingId);
                this.showStatus('Rating deleted', 'success');
            } catch (error) {
                console.error('Error deleting rating:', error);
                this.showStatus('Error deleting rating', 'error');
            }
        },

        async clearAllRatings() {
            if (!confirm('Are you sure you want to delete ALL ratings? This cannot be undone.')) return;

            if (!this.isLoggedIn) {
                RatingStorage.clearAllRatings();
                this.clearCookie('user_rates');
                this.ratings = [];
                this.showStatus('All ratings cleared', 'success');
                return;
            }

            try {
                const response = await fetch('/api/ratings/clear', { method: 'POST' });
                if (!response.ok) throw new Error('Failed to clear ratings');
                
                this.ratings = [];
                this.showStatus('All ratings cleared', 'success');
            } catch (error) {
                console.error('Error clearing ratings:', error);
                this.showStatus('Error clearing ratings', 'error');
            }
        },

        editFeedback(feedback) {
            // Redirect to feedback edit page
            window.location.href = `/feedback?edit=${feedback.id}`;
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
                'ui_algo1': 'UI (Algorithm 1)',
                'ui_algo2': 'UI (Algorithm 2)'
            };
            return types[type] || type;
        },

        showStatus(message, type) {
            this.statusMessage = message;
            this.statusType = type;
            setTimeout(() => {
                this.statusMessage = '';
            }, 3000);
        }
    },

    mounted() {
        this.loadGenres();
        this.loadUserProfile();
        this.loadRatings();
        this.loadFeedback();
        this.loadFeedbackProgress();
    }
});

app.config.compilerOptions.delimiters = ['[[', ']]'];
app.mount('#profile-app');
