import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Tailwind/PostCSS are configured via postcss.config.js -- they are NOT
// Vite plugins, so they must not be imported here.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      // Demo/sample imagery is served by the backend as static files.
      "/data": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
});