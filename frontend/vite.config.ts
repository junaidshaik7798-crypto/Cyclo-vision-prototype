import { defineConfig, type ProxyOptions } from "vite";
import react from "@vitejs/plugin-react";

// The backend owns /api (JSON datasets) and /data (bundled demo imagery).
// `vite preview` does NOT reuse `server.proxy`, so both entries must be
// declared here: without the preview proxy the production build served on
// port 4173 answered every dataset request with 404 and the dashboard showed
// "could not fetch datasets" for all panels.
const backendProxy: Record<string, ProxyOptions> = {
  "/api": {
    target: "http://127.0.0.1:8000",
    changeOrigin: true,
  },
  // Demo/sample imagery is served by the backend as static files.
  "/data": {
    target: "http://127.0.0.1:8000",
    changeOrigin: true,
  },
};

// Tailwind/PostCSS are configured via postcss.config.js -- they are NOT
// Vite plugins, so they must not be imported here.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: "0.0.0.0",
    strictPort: true,
    proxy: backendProxy,
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
  preview: {
    host: "0.0.0.0",
    port: 4173,
    strictPort: true,
    proxy: backendProxy,
  },
});