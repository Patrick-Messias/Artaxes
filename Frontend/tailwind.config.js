/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg:       '#0D0D0D',
        surface:  '#0A0A0A',
        border:   '#1E1E1E',
        muted:    '#1A1A1A',
        accent:   '#00D9FF',
        green:    '#4ECB71',
        red:      '#FF6B6B',
        purple:   '#C77DFF',
        yellow:   '#FFD93D',
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      }
    }
  },
  plugins: []
}













