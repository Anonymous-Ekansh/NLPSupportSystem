import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import fs from "fs";
import path from "path";
import { ResolveResultSchema } from "@/lib/schemas";

const RequestSchema = z.object({
  query: z
    .string()
    .trim()
    .min(1, "Query must not be empty")
    .max(500, "Query must be ≤ 500 characters"),
});

// Load fallback demo cache from web/public/demo_cache.json
function getDemoCache(): Record<string, unknown> {
  try {
    const filePath = path.join(process.cwd(), "public", "demo_cache.json");
    if (fs.existsSync(filePath)) {
      return JSON.parse(fs.readFileSync(filePath, "utf-8"));
    }
  } catch (err) {
    console.error("Failed to read demo_cache.json:", err);
  }
  return {};
}

function findInDemoCache(query: string) {
  const cache = getDemoCache();
  const qClean = query.trim().toLowerCase();

  for (const [key, value] of Object.entries(cache)) {
    if (key.trim().toLowerCase() === qClean) {
      return value;
    }
  }
  return null;
}

export async function POST(req: NextRequest) {
  let bodyJson: unknown;
  try {
    bodyJson = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const parsed = RequestSchema.safeParse(bodyJson);
  if (!parsed.success) {
    return NextResponse.json(
      { error: parsed.error.issues[0]?.message || "Invalid input" },
      { status: 400 }
    );
  }

  const query = parsed.data.query;
  const apiUrl = process.env.NLP_API_URL || "http://localhost:8000";

  // Try proxying to backend with 8s timeout
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 8000);

  try {
    const upstreamRes = await fetch(`${apiUrl}/resolve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (upstreamRes.ok) {
      const data = await upstreamRes.json();
      const valid = ResolveResultSchema.safeParse(data);
      if (valid.success) {
        return NextResponse.json(valid.data);
      }
      console.warn("Backend response did not strictly match schema:", valid.error);
      return NextResponse.json(data);
    }

  } catch (err: unknown) {
    clearTimeout(timeoutId);
    const msg = err instanceof Error ? err.message : String(err);
    console.warn("Fetch to NLP backend failed or timed out:", msg);
  }

  // Fallback to demo cache if available
  const cached = findInDemoCache(query);
  if (cached) {
    const result = {
      ...cached,
      cached: true,
    };
    return NextResponse.json(result);
  }

  return NextResponse.json(
    {
      error:
        "NLP backend is unreachable or timed out, and query is not in demo cache. Please wake the service or try one of the 6 sample queries.",
    },
    { status: 502 }
  );
}
