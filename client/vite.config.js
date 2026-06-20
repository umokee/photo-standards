import react from "@vitejs/plugin-react";
import { fileURLToPath } from "url";
import { defineConfig } from "vite";
import viteTsconfigPaths from "vite-tsconfig-paths";
import { reactPreviewPlugin } from "./tools/react-preview/viteReactPreview";

export default defineConfig({
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  base: `./`,
  plugins: [react(), viteTsconfigPaths(), reactPreviewPlugin()],
  optimizeDeps: { exclude: ["fsevents"] },
  server: {
    host: "0.0.0.0",
    allowedHosts: true,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:3001",
        changeOrigin: true,
      },
      "/storage": {
        target: "http://127.0.0.1:3001",
        changeOrigin: true,
      },
      "/ws": {
        target: "ws://127.0.0.1:3001",
        ws: true,
        changeOrigin: true,
      },
    },
  },
});
