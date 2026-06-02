import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
  resolve: {
    alias: {
      // decimal.js-light uses extensionless "main" — fails on Linux/Vercel
      "decimal.js-light": "decimal.js-light/decimal.mjs",
    },
  },
  optimizeDeps: {
    include: ["recharts", "decimal.js-light"],
  },
  build: {
    commonjsOptions: {
      transformMixedEsModules: true,
    },
  },
});
