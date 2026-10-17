import process from 'node:process'
import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv } from 'vite'

// A production build bakes VITE_API_BASE_URL into the bundle, and the app
// falls back to http://127.0.0.1:8000 without it. Refuse to build a
// production bundle that would call a local or unencrypted backend.
function checkProductionApiUrl(value) {
  let url
  try {
    url = new URL(value)
  } catch {
    throw new Error('VITE_API_BASE_URL must be set to the deployed backend URL for a production build, e.g. the Cloudflare Tunnel URL https://<name>.trycloudflare.com (see docs/DEPLOYMENT.md)')
  }
  if (url.protocol !== 'https:' || ['localhost', '127.0.0.1', '0.0.0.0', '[::1]'].includes(url.hostname)) {
    throw new Error(`VITE_API_BASE_URL must be an https URL of the deployed backend, not ${value}`)
  }
}

// https://vite.dev/config/
export default defineConfig(({ command, mode }) => {
  if (command === 'build' && mode === 'production') {
    checkProductionApiUrl(loadEnv(mode, process.cwd(), '').VITE_API_BASE_URL)
  }
  return {
    plugins: [react()],
  }
})
