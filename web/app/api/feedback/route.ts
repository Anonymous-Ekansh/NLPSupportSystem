import { NextRequest, NextResponse } from "next/server";
import { fetchUpstream } from "@/lib/server/upstream";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: NextRequest) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      { error: "Invalid feedback payload.", code: "bad_request" },
      { status: 400 }
    );
  }

  const data = (body ?? {}) as Record<string, unknown>;
  const query = typeof data.query === "string" ? data.query.trim() : "";
  const rating = Number(data.rating);
  const note = typeof data.note === "string" ? data.note.slice(0, 1000) : undefined;

  if (
    !query ||
    query.length > 500 ||
    !Number.isInteger(rating) ||
    rating < 1 ||
    rating > 5
  ) {
    return NextResponse.json(
      { error: "Feedback needs a question and a rating from 1 to 5.", code: "bad_request" },
      { status: 422 }
    );
  }

  try {
    const upstream = await fetchUpstream(
      "/feedback",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, rating, note }),
      },
      { attempts: 2, timeouts: [6000, 4000], backoffs: [300] }
    );

    if (upstream.ok) {
      return NextResponse.json(await upstream.json().catch(() => ({ status: "saved" })));
    }

    if (upstream.status === 422) {
      return NextResponse.json(
        { error: "That feedback could not be saved.", code: "bad_request" },
        { status: 422 }
      );
    }
  } catch {
    // Feedback is best-effort: thank the customer even when storage is cold.
  }

  return NextResponse.json({ status: "accepted" });
}
