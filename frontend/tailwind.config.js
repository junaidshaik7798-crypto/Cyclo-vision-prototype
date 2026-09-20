/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        // CYCLO-VISION dark-ops palette
        abyss: "#050b14",
        deep: "#0a1420",
        panel: "#0f1c2c",
        edge: "#1b2c40",
        accent: "#38bdf8",
        alert: "#f97316",
        danger: "#ef4444",
        ok: "#22c55e",
      },
    },
  },
  plugins: [],
};