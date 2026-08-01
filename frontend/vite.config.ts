import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev-server proxy so `npm run dev` can hit the FastAPI backend on :8000
// without CORS setup; the production build is served by FastAPI directly
// on a single port, so this proxy is dev-only.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
  build: {
    outDir: "dist",
  },
});
