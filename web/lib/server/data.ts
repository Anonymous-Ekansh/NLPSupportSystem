import fs from "node:fs";
import path from "node:path";

/**
 * Read a file from `web/public/data/`. Returns `null` instead of throwing so a
 * page can show a neat empty state that names the missing file.
 */
export function readDataJson<T>(filename: string): T | null {
  try {
    const filePath = path.join(process.cwd(), "public", "data", filename);
    return JSON.parse(fs.readFileSync(filePath, "utf-8")) as T;
  } catch {
    return null;
  }
}

/**
 * Read a file from `web/public/evidence/` (written by the Python evaluation
 * scripts and copied by scripts/sync_evidence.py). Returns `null` when the
 * file has not been generated yet, so the page can say so instead of guessing.
 */
export function readEvidenceJson<T>(filename: string): T | null {
  try {
    const filePath = path.join(process.cwd(), "public", "evidence", filename);
    return JSON.parse(fs.readFileSync(filePath, "utf-8")) as T;
  } catch {
    return null;
  }
}
