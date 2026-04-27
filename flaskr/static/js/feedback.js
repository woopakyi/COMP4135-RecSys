const { createApp, ref, computed, onMounted } = Vue;

const app = createApp({
    delimiters: ['[[', ']]'],
    data() {
        return {
            isLoggedIn: !!document.querySelector('[data-user-logged-in]'),
            step: 1,
            submissionType: null,
            voterName: '',
            voterEmail: '',
            feedbackType: null,
            voteChoice: null,
            consentAgreed: false,
            isSubmitting: false,
            statusMessage: '',
            statusType: 'success'
        };
    },
    computed: {
        isAlgorithmFeedback() {
            return this.feedbackType?.startsWith('algo_');
        },
        isUiFeedback() {
            return this.feedbackType?.startsWith('ui_');
        },
        getUiContext() {
            if (this.feedbackType === 'algo_ui1') return 'UI 1 (Professional Dark)';
            if (this.feedbackType === 'algo_ui2') return 'UI 2 (Concise White)';
            return '';
        },
        getAlgorithmContext() {
            if (this.feedbackType === 'ui_algo1') return 'Algorithm 1';
            if (this.feedbackType === 'ui_algo2') return 'Algorithm 2';
            return '';
        }
    },
    methods: {
        selectSubmissionType(type) {
            this.submissionType = type;
        },

        selectFeedbackType(type) {
            this.feedbackType = type;
        },

        selectVote(choice) {
            this.voteChoice = choice;
        },

        proceedToStep(nextStep) {
            if (nextStep === 2 && this.submissionType === 'anonymous') {
                // Skip to step 3 for anonymous users
                this.step = 3;
            } else {
                this.step = nextStep;
            }
        },

        async submitFeedback() {
            if (this.submissionType === 'logged' && !this.consentAgreed) {
                this.showStatus('Please agree to the consent statement', 'error');
                return;
            }

            this.isSubmitting = true;

            try {
                const payload = {
                    submission_type: this.submissionType,
                    feedback_type: this.feedbackType,
                    vote_choice: this.voteChoice
                };

                if (this.submissionType === 'logged') {
                    payload.voter_name = this.voterName;
                    payload.voter_email = this.voterEmail;
                    payload.consent_agreed = this.consentAgreed;
                }

                const response = await fetch('/api/feedback', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.error || 'Failed to submit feedback');
                }

                this.step = 6;
                this.showStatus('Feedback submitted successfully!', 'success');
            } catch (error) {
                console.error('Error submitting feedback:', error);
                this.showStatus('Error: ' + error.message, 'error');
            } finally {
                this.isSubmitting = false;
            }
        },

        resetForm() {
            this.step = 1;
            this.submissionType = null;
            this.voterName = '';
            this.voterEmail = '';
            this.feedbackType = null;
            this.voteChoice = null;
            this.consentAgreed = false;
            this.statusMessage = '';
        },

        formatFeedbackType(type) {
            const types = {
                'algo_ui1': 'Algorithm Evaluation (UI 1 - Professional Dark)',
                'algo_ui2': 'Algorithm Evaluation (UI 2 - Concise White)',
                'ui_algo1': 'UI Evaluation (Algorithm 1)',
                'ui_algo2': 'UI Evaluation (Algorithm 2)'
            };
            return types[type] || type;
        },

        formatVote(choice, type) {
            if (type?.startsWith('algo_')) {
                if (choice === 'better') return 'Algorithm 1 is better';
                if (choice === 'worse') return 'Algorithm 2 is better';
                if (choice === 'same') return 'Both algorithms are equal';
            } else if (type?.startsWith('ui_')) {
                if (choice === 'better') return 'UI 1 (Dark) is better';
                if (choice === 'worse') return 'UI 2 (Light) is better';
                if (choice === 'same') return 'Both UIs are equal';
            }
            return choice;
        },

        showStatus(message, type) {
            this.statusMessage = message;
            this.statusType = type;
            if (type === 'success') {
                setTimeout(() => {
                    this.statusMessage = '';
                }, 3000);
            }
        }
    },

    mounted() {
        // Check if user is logged in
        const userElement = document.querySelector('[data-user-logged-in]');
        this.isLoggedIn = !!userElement;

        // If not logged in, skip direct to anonymous option
        if (!this.isLoggedIn) {
            // Only show anonymous option
        }

        // Check for edit parameter in URL
        const params = new URLSearchParams(window.location.search);
        if (params.has('edit')) {
            // TODO: Load feedback for editing
        }
    }
});

app.mount('#feedback-app');
