import { mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";

const temporaryRoot = fileURLToPath(new URL("../../Temp_Test/GeoAgent/node/", import.meta.url));
mkdirSync(temporaryRoot, { recursive: true });
for (const variable of ["TEMP", "TMP", "TMPDIR"]) {
  process.env[variable] = temporaryRoot;
}
