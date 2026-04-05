import { createApp, ref } from 'https://unpkg.com/vue@3/dist/vue.esm-browser.js'

const app = createApp({
  setup() {
    const email = ref('')
    const password = ref('')
    const showPassword = ref(false)
    const errorMessage = ref('')

    function togglePassword() {
      showPassword.value = !showPassword.value
    }

    function onSubmit(event) {
      errorMessage.value = ''
      if (!email.value.includes('@')) {
        event.preventDefault()
        errorMessage.value = 'Please enter a valid email address.'
        return
      }
      if (password.value.length < 6) {
        event.preventDefault()
        errorMessage.value = 'Password must be at least 6 characters.'
      }
    }

    return { email, password, showPassword, errorMessage, togglePassword, onSubmit }
  }
})

app.config.compilerOptions.delimiters = ['[[', ']]']
app.mount('#auth-login-app')
