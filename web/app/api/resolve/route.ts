import { NextRequest, NextResponse } from "next/server";
import {
  fetchUpstream,
  readDemoResult,
  sanitizeResolve,
} from "@/lib/server/upstream";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

const MAX_QUERY_LENGTH = 500;

export async function POST(request: NextRequest) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      { error: "Please send a question in the request body.", code: "bad_request" },
      { status: 400 }
    );
  }

  const query = (body as { query?: unknown })?.query;
  if (typeof query !== "string" || query.trim().length === 0) {
    return NextResponse.json(
      { error: "Please type a question first.", code: "empty_query" },
      { status: 422 }
    );
  }

  const rawAnswer = (body as { answer?: unknown })?.answer;
  const answer =
    typeof rawAnswer === "string" && rawAnswer.trim().length > 0
      ? rawAnswer.trim().slice(0, MAX_QUERY_LENGTH)
      : null;

  const trimmed = query.trim();
  if (trimmed.length > MAX_QUERY_LENGTH) {
    return NextResponse.json(
      {
        error: `Questions are limited to ${MAX_QUERY_LENGTH} characters.`,
        code: "too_long",
        limit: MAX_QUERY_LENGTH,
      },
      { status: 422 }
    );
  }

  try {
    const upstream = await fetchUpstream(
      "/resolve",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(answer ? { query: trimmed, answer } : { query: trimmed }),
      },
      {
        attempts: 3,
        timeouts: [30000, 15000, 10000],
        backoffs: [500, 1500],
      }
    );

    if (upstream.status === 429) {
      return NextResponse.json(
        {
          error: "You're moving a little fast. Please wait a moment and try again.",
          code: "rate_limited",
        },
        { status: 429 }
      );
    }

    if (upstream.status === 422) {
      return NextResponse.json(
        {
          error: `Questions are limited to ${MAX_QUERY_LENGTH} characters.`,
          code: "too_long",
          limit: MAX_QUERY_LENGTH,
        },
        { status: 422 }
      );
    }

    if (!upstream.ok) {
      // A follow-up reply cannot be answered from the cache: the cached run is
      // the *clarification*, not the resolved follow-up.
      const fallback = answer ? null : readDemoResult(trimmed);
      if (fallback) return NextResponse.json(fallback);

      return NextResponse.json(
        {
          error: "The assistant couldn't answer that just now. Please try again.",
          code: "unavailable",
        },
        { status: 502 }
      );
    }

    const raw = await upstream.json();
    const result = sanitizeResolve(raw);

    if (!result.final_response && result.trace.length === 0) {
      const fallback = answer ? null : readDemoResult(trimmed);
      if (fallback) return NextResponse.json(fallback);

      return NextResponse.json(
        {
          error: "The assistant returned an empty answer. Please try again.",
          code: "empty_response",
        },
        { status: 502 }
      );
    }

    return NextResponse.json(result);
  } catch {
    const fallback = answer ? null : readDemoResult(trimmed);
    if (fallback) return NextResponse.json(fallback);

    return NextResponse.json(
      {
        error:
          "The assistant is taking longer than usual. It may still be waking up — please try again in a moment.",
        code: "unavailable",
      },
      { status: 503 }
    );
  }
}
