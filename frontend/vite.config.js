import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // В продакшене (GitHub Pages) — базовый путь /lol-ai-coaching/
  base: process.env.VITE_BASE_PATH || '/',
  server: {
    port: 5173,
    proxy: {
      '/analyze':          'http://localhost:8004',
      '/mistakes':         'http://localhost:8004',
      '/benchmarks':       'http://localhost:8004',
      '/mistakes/resolve': 'http://localhost:8004',
    }
  }
})
