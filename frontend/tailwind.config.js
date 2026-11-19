/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        forest: {
          50: '#F2F8F5',
          100: '#E2F0EA',
          200: '#C4E1D4',
          300: '#9BCDBB',
          400: '#6CB39C',
          500: '#1F7A60',
          600: '#16634E',
          700: '#0F4C3B',
          800: '#0B3A2C',
          900: '#06261D',
          950: '#031711',
        },
        brand: {
          50: '#f0fdf4',
          100: '#dcfce7',
          500: '#15803d',
          600: '#166534',
          700: '#14532d',
          800: '#052e16',
        },
        clinical: {
          blue: '#1D4ED8',
          navy: '#0F172A',
          teal: '#0D9488',
          sand: '#F8FAF9',
          card: '#FFFFFF',
          border: '#E2E8F0',
          muted: '#64748B',
        },
      },
      boxShadow: {
        'subtle': '0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 1px 2px 0 rgba(0, 0, 0, 0.02)',
        'card': '0 1px 3px 0 rgba(15, 23, 42, 0.05), 0 1px 2px -1px rgba(15, 23, 42, 0.05)',
        'popover': '0 4px 6px -1px rgba(15, 23, 42, 0.08), 0 2px 4px -2px rgba(15, 23, 42, 0.05)',
      },
      fontFamily: {
        sans: [
          'Inter',
          '-apple-system',
          'BlinkMacSystemFont',
          '"Segoe UI"',
          'Roboto',
          'sans-serif',
        ],
      },
    },
  },
  plugins: [],
};
