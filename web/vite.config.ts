import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

const rootDir = path.dirname(fileURLToPath(import.meta.url));

function resolveDecimalEntry(): string {
  const candidates = [
    path.join(rootDir, "node_modules/decimal.js-light/decimal.mjs"),
    path.join(rootDir, "node_modules/decimal.js-light/decimal.js"),
  ];
  for (const file of candidates) {
    if (fs.existsSync(file)) return file;
  }
  return candidates[0];
}

const decimalEntry = resolveDecimalEntry();

/** Recharts depends on decimal.js-light; its package.json "main" lacks a file extension (breaks on Linux). */
function decimalJsLightPlugin(): Plugin {
  return {
    name: "resolve-decimal-js-light",
    enforce: "pre",
    resolveId(source) {
      if (source === "decimal.js-light" || source.startsWith("decimal.js-light/")) {
        return decimalEntry;
      }
      return null;
    },
  };
}

export default defineConfig({
  plugins: [react(), decimalJsLightPlugin()],
  server: { port: 5173 },
  optimizeDeps: {
    include: ["recharts", "decimal.js-light"],
    needsInterop: ["decimal.js-light"],
  },
  build: {
    commonjsOptions: {
      transformMixedEsModules: true,
      include: [/decimal.js-light/, /recharts/, /node_modules/],
    },
  },
});
