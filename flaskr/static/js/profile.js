const { createApp, ref, computed, onMounted } = Vue;

const app = createApp({
    delimiters: ['[[', ']]'],
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
        async loadGenres() {
            try {
                const response = await fetch('/api/genres');
                const data = await response.json();
                this.genres = data.genres || [];
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
                
                this.profile.username = data.user.username;
                this.profile.algorithm = data.user.algorithm_preference || 'algo1';
                this.profile.uiMode = data.user.ui_preference || '1';
                this.genreScores = data.user.genre_preferences || {};
            } catch (error) {
                console.error('Error loading profile:', error);
                this.showStatus('Error loading profile', 'error');
            }
        },

        async loadRatings() {
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
            } catch (error) {
                console.error('Error updating preference:', error);
            }
        },

        async updateGenreScore(genreId) {
            const score = this.genreScores[genreId] || 0;

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
            } catch (error) {
                console.error('Error updating genre preference:', error);
            }
        },

        async deleteRating(ratingId) {
            if (!confirm('Are you sure you want to delete this rating?')) return;

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
    }
});

app.mount('#profile-app');
