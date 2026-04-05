import { createApp, ref, computed } from 'https://unpkg.com/vue@3/dist/vue.esm-browser.js'

const app = createApp({
  setup() {
    const username = ref('')
    const email = ref('')
    const password = ref('')
    const confirmPassword = ref('')
    const errorMessage = ref('')

    const passwordStrength = computed(() => {
      let score = 0
      if (password.value.length >= 6) score += 35
      if (/[A-Z]/.test(password.value)) score += 20
      if (/[a-z]/.test(password.value)) score += 20
      if (/\d/.test(password.value)) score += 15
      if (/[^A-Za-z0-9]/.test(password.value)) score += 10
      return Math.min(score, 100)
    })

    const strengthLabel = computed(() => {
      if (passwordStrength.value < 40) return 'weak'
      if (passwordStrength.value < 70) return 'moderate'
      return 'strong'
    })

    function onSubmit(event) {
      errorMessage.value = ''
      if (username.value.length < 3) {
        event.preventDefault()
        errorMessage.value = 'Username must be at least 3 characters.'
        return
      }
      if (!email.value.includes('@')) {
        event.preventDefault()
        errorMessage.value = 'Please enter a valid email address.'
        return
      }
      if (password.value.length < 6) {
        event.preventDefault()
        errorMessage.value = 'Password must be at least 6 characters.'
        return
      }
      if (password.value !== confirmPassword.value) {
        event.preventDefault()
        errorMessage.value = 'Passwords do not match.'
      }
    }

    return {
      username,
      email,
      password,
      confirmPassword,
      errorMessage,
      passwordStrength,
      strengthLabel,
      onSubmit,
    }
  }
})

app.config.compilerOptions.delimiters = ['[[', ']]']
app.mount('#auth-register-app')
