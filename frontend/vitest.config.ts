import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";

const testArtifacts = fileURLToPath(new URL("../../Temp_Test/GeoAgent/", import.meta.url));

export default defineConfig({
  plugins: [react()],
  cacheDir: `${testArtifacts}/vitest-cache`,
  test: {
    environment: "jsdom",
    globals: true,
    restoreMocks: true,
    coverage: {
      reportsDirectory: `${testArtifacts}/coverage`,
    },
  },
});
