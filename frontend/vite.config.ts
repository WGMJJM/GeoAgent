import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "VITE_");
  const backendTarget = env.VITE_BACKEND_URL || "http://127.0.0.1:8787";
  const websocketTarget = backendTarget.replace(/^http:/, "ws:").replace(/^https:/, "wss:");

  return {
    plugins: [react()],
    server: {
      host: "127.0.0.1",
      port: 5173,
      strictPort: true,
      proxy: {
        "/api": backendTarget,
        "/health": backendTarget,
        "/ws": { target: websocketTarget, ws: true },
      },
    },
    preview: { host: "127.0.0.1", port: 4173, strictPort: true },
  };
});
