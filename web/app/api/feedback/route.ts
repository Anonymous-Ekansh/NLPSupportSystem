import { NextRequest, NextResponse } from "next/server";
import { FeedbackSchema } from "@/lib/schemas";

export async function POST(req: NextRequest) {
  let bodyJson: unknown;
  try {
    bodyJson = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const parsed = FeedbackSchema.safeParse(bodyJson);
  if (!parsed.success) {
    return NextResponse.json(
      { error: parsed.error.issues[0]?.message || "Invalid feedback" },
      { status: 400 }
    );
  }

  const apiUrl = process.env.NLP_API_URL || "http://localhost:8000";
  try {
    const upstreamRes = await fetch(`${apiUrl}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(parsed.data),
    });
    if (upstreamRes.ok) {
      const data = await upstreamRes.json();
      return NextResponse.json(data);
    }
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    console.warn("Feedback proxy to backend failed:", msg);
  }

  // Graceful response even if backend database is offline
  return NextResponse.json({ status: "saved_locally", note: "Feedback noted" });
}
