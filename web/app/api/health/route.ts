import { NextResponse } from "next/server";
import { HealthSchema } from "@/lib/schemas";

export async function GET() {
  const apiUrl = process.env.NLP_API_URL || "http://localhost:8000";
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 5000);

  try {
    const res = await fetch(`${apiUrl}/health`, {
      method: "GET",
      signal: controller.signal,
      cache: "no-store",
    });
    clearTimeout(timeoutId);

    if (res.ok) {
      const data = await res.json();
      const valid = HealthSchema.safeParse(data);
      return NextResponse.json(valid.success ? valid.data : data);
    }
  } catch {
    clearTimeout(timeoutId);
  }

  return NextResponse.json(
    {
      status: "unreachable",
      model_loaded: false,
      retrieval_backend: "offline",
    },
    { status: 503 }
  );
}
