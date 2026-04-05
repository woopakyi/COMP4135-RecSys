/**
 * localStorage Utilities for Anonymous User Data Management
 * Stores movie ratings, genre preferences, and user settings for logged-out users
 */

const StorageKeys = {
    RATINGS: 'rec_user_ratings',           // Movie ratings: {movieId: rating}
    GENRE_PREFERENCES: 'rec_genre_prefs',  // Genre preferences: {genreId: score}
    ALGORITHM: 'rec_algorithm',            // Selected algorithm
    UI_MODE: 'rec_ui_mode',                // Selected UI mode (1 or 2)
    SELECTED_GENRES: 'rec_selected_genres' // Selected genres: [genreId, ...]
};

/**
 * Rating Management
 */
const RatingStorage = {
    /**
     * Save a movie rating to localStorage
     * @param {number} movieId - Movie ID
     * @param {number} rating - Rating (1-5)
     */
    saveRating(movieId, rating) {
        const ratings = this.getAllRatings();
        ratings[movieId.toString()] = {
            rating: rating,
            timestamp: new Date().toISOString()
        };
        localStorage.setItem(StorageKeys.RATINGS, JSON.stringify(ratings));
    },

    /**
     * Get a specific movie rating from localStorage
     * @param {number} movieId - Movie ID
     * @returns {number|null} Rating or null if not found
     */
    getRating(movieId) {
        const ratings = this.getAllRatings();
        return ratings[movieId]?.rating || null;
    },

    /**
     * Get all ratings from localStorage
     * @returns {Object} Object with movieId: {rating, timestamp}
     */
    getAllRatings() {
        try {
            const data = localStorage.getItem(StorageKeys.RATINGS);
            return data ? JSON.parse(data) : {};
        } catch {
            return {};
        }
    },

    /**
     * Get ratings as array format
     * @returns {Array} Array of {movieId, rating, timestamp}
     */
    getRatingsArray() {
        const ratings = this.getAllRatings();
        return Object.entries(ratings).map(([movieId, data]) => ({
            movieId: parseInt(movieId),
            rating: data.rating,
            timestamp: data.timestamp
        }));
    },

    /**
     * Delete a specific movie rating
     * @param {number} movieId - Movie ID
     */
    deleteRating(movieId) {
        const ratings = this.getAllRatings();
        delete ratings[movieId.toString()];
        localStorage.setItem(StorageKeys.RATINGS, JSON.stringify(ratings));
    },

    /**
     * Clear all ratings
     */
    clearAllRatings() {
        localStorage.removeItem(StorageKeys.RATINGS);
    }
};

/**
 * Genre Preference Management
 */
const GenreStorage = {
    /**
     * Save a genre preference to localStorage
     * @param {number} genreId - Genre ID
     * @param {number} score - Score (0-10)
     */
    savePreference(genreId, score) {
        const prefs = this.getAllPreferences();
        prefs[genreId.toString()] = {
            score: score,
            timestamp: new Date().toISOString()
        };
        localStorage.setItem(StorageKeys.GENRE_PREFERENCES, JSON.stringify(prefs));
    },

    /**
     * Get all genre preferences
     * @returns {Object} Object with genreId: {score, timestamp}
     */
    getAllPreferences() {
        try {
            const data = localStorage.getItem(StorageKeys.GENRE_PREFERENCES);
            return data ? JSON.parse(data) : {};
        } catch {
            return {};
        }
    },

    /**
     * Get preferences as object {genreId: score}
     * @returns {Object} Object with genreId: score
     */
    getPreferencesForAPI() {
        const prefs = this.getAllPreferences();
        const result = {};
        Object.entries(prefs).forEach(([genreId, data]) => {
            result[genreId] = data.score;
        });
        return result;
    },

    /**
     * Clear all genre preferences
     */
    clearAllPreferences() {
        localStorage.removeItem(StorageKeys.GENRE_PREFERENCES);
    }
};

/**
 * User Settings Management (Algorithm, UI Mode)
 */
const SettingsStorage = {
    /**
     * Save algorithm preference
     * @param {string} algorithm - 'algo1' or 'algo2'
     */
    setAlgorithm(algorithm) {
        localStorage.setItem(StorageKeys.ALGORITHM, algorithm);
    },

    /**
     * Get algorithm preference
     * @returns {string|null} Algorithm preference
     */
    getAlgorithm() {
        return localStorage.getItem(StorageKeys.ALGORITHM);
    },

    /**
     * Save UI mode preference
     * @param {string} mode - '1' for dark or '2' for light
     */
    setUiMode(mode) {
        localStorage.setItem(StorageKeys.UI_MODE, mode);
    },

    /**
     * Get UI mode preference
     * @returns {string|null} UI mode
     */
    getUiMode() {
        return localStorage.getItem(StorageKeys.UI_MODE);
    },

    /**
     * Save selected genres
     * @param {Array} genres - Array of genre IDs
     */
    setSelectedGenres(genres) {
        localStorage.setItem(StorageKeys.SELECTED_GENRES, JSON.stringify(genres));
    },

    /**
     * Get selected genres
     * @returns {Array} Array of genre IDs
     */
    getSelectedGenres() {
        try {
            const data = localStorage.getItem(StorageKeys.SELECTED_GENRES);
            return data ? JSON.parse(data) : [];
        } catch {
            return [];
        }
    }
};

/**
 * Data Migration: LocalStorage to Database
 */
const DataMigration = {
    /**
     * Prepare all localStorage data for migration to database
     * @returns {Object} Object containing all user data
     */
    getAllDataForMigration() {
        return {
            ratings: RatingStorage.getRatingsArray(),
            genrePreferences: GenreStorage.getPreferencesForAPI(),
            algorithm: SettingsStorage.getAlgorithm(),
            uiMode: SettingsStorage.getUiMode(),
            selectedGenres: SettingsStorage.getSelectedGenres()
        };
    },

    /**
     * Clear all localStorage data after successful migration
     */
    clearAllAfterMigration() {
        RatingStorage.clearAllRatings();
        GenreStorage.clearAllPreferences();
        localStorage.removeItem(StorageKeys.ALGORITHM);
        localStorage.removeItem(StorageKeys.UI_MODE);
        localStorage.removeItem(StorageKeys.SELECTED_GENRES);
    }
};

/**
 * Utility Functions
 */
const StorageUtils = {
    /**
     * Check if user has any data in localStorage
     * @returns {boolean}
     */
    hasAnyData() {
        const ratings = RatingStorage.getAllRatings();
        const genres = GenreStorage.getAllPreferences();
        return Object.keys(ratings).length > 0 || Object.keys(genres).length > 0;
    },

    /**
     * Get total storage usage estimate
     * @returns {number} Approximate bytes used
     */
    getStorageUsage() {
        let total = 0;
        for (const key of Object.values(StorageKeys)) {
            const item = localStorage.getItem(key);
            if (item) {
                total += item.length + key.length;
            }
        }
        return total;
    },

    /**
     * Export all localStorage data as JSON
     * @returns {Object} All stored data
     */
    exportData() {
        return {
            ratings: RatingStorage.getAllRatings(),
            genrePreferences: GenreStorage.getAllPreferences(),
            algorithm: SettingsStorage.getAlgorithm(),
            uiMode: SettingsStorage.getUiMode(),
            selectedGenres: SettingsStorage.getSelectedGenres(),
            exportDate: new Date().toISOString()
        };
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        StorageKeys,
        RatingStorage,
        GenreStorage,
        SettingsStorage,
        DataMigration,
        StorageUtils
    };
}
