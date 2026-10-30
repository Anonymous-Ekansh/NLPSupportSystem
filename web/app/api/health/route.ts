import { NextResponse } from "next/server";
import { fetchUpstream, sanitizeHealth } from "@/lib/server/upstream";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const upstream = await fetchUpstream(
      "/health",
      { method: "GET" },
      { attempts: 1, timeouts: [4500] }
    );

    if (upstream.ok) {
      const health = sanitizeHealth(await upstream.json());
      if (health.status === "ok") return NextResponse.json(health);
      return NextResponse.json(health, { status: 503 });
    }
  } catch {
    // The free host is probably waking.
  }

  return NextResponse.json(
    {
      status: "waking",
      model_loaded: false,
      retrieval_backend: "unknown",
      agents: ["a1", "a2"],
    },
    { status: 503 }
  );
}
