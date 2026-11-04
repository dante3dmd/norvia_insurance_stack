/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      boxShadow: {
        glow: '0 0 0 1px rgba(148,163,184,0.15), 0 20px 50px rgba(15, 23, 42, 0.45)',
      },
    },
  },
  plugins: [],
};
