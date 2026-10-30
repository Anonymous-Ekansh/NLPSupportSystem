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
