import type { Config } from 'tailwindcss';

export default {
  content: ['./index.html', './src/**/*.{vue,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        arena: {
          navy: '#0D1B2A',
          cyan: '#00B4D8',
          mint: '#90E0EF',
          amber: '#FFB703',
          coral: '#FB8500'
        }
      },
      boxShadow: {
        glow: '0 0 0 1px rgba(0,180,216,.28), 0 8px 30px rgba(13,27,42,.35)'
      }
    }
  },
  plugins: []
} satisfies Config;
