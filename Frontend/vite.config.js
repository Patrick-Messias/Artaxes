import { defineConfig } from 'vite'
import react            from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Proxies /api/* → FastAPI on 8000, strips /api prefix
      '/api': {
        target:    'http://localhost:8000',
        rewrite:   path => path.replace(/^\/api/, '')
      }
    }
  }
})
