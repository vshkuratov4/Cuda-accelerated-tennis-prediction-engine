/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "media",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Validated categorical/status palette (see dataviz skill palette.md)
        series1: { DEFAULT: "#2a78d6", dark: "#3987e5" }, // player 1
        series2: { DEFAULT: "#eb6834", dark: "#d95926" }, // player 2
        good: { DEFAULT: "#0ca30c" },
        surface: { DEFAULT: "#fcfcfb", dark: "#1a1a19" },
        page: { DEFAULT: "#f9f9f7", dark: "#0d0d0d" },
      },
      fontFamily: {
        sans: ["system-ui", "-apple-system", "Segoe UI", "sans-serif"],
      },
    },
  },
  plugins: [],
};
