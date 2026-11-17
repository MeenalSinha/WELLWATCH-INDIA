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
        brand: {
          50: '#f0fdf4',
          100: '#dcfce7',
          500: '#15803d',
          600: '#166534',
          700: '#14532d',
          800: '#052e16',
        },
        clinical: {
          blue: '#0284c7',
          navy: '#0f172a',
          teal: '#0d9488',
          sand: '#f8fafc',
          card: '#ffffff',
          border: '#e2e8f0',
        }
      },
    },
  },
  plugins: [],
};
