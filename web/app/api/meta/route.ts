import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export async function GET() {
  const apiUrl = process.env.NLP_API_URL || "http://localhost:8000";
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 5000);

  try {
    const res = await fetch(`${apiUrl}/meta`, {
      method: "GET",
      signal: controller.signal,
      cache: "no-store",
    });
    clearTimeout(timeoutId);

    if (res.ok) {
      const data = await res.json();
      return NextResponse.json(data);
    }
  } catch {
    clearTimeout(timeoutId);
  }

  // Fallback to local config fixture if available
  try {
    const a1Path = path.join(process.cwd(), "fixtures", "a1.json");
    if (fs.existsSync(a1Path)) {
      const a1Data = JSON.parse(fs.readFileSync(a1Path, "utf-8"));
      return NextResponse.json({
        a1: {
          served_level: a1Data.served_level || "L2",
          tau: a1Data.threshold || a1Data.tau || 0.65,
          temperature: a1Data.temperature || 1.0,
          n_classes: 27,
        },
        retrieval_backend: "in-memory (cached)",
        planned_agents: [
          { name: "Verifier", status: "planned", planned_function: "Verification & fact checking" },
          { name: "Clarifier", status: "planned", planned_function: "Interactive clarification" },
          { name: "Escalator", status: "planned", planned_function: "Human escalation routing" }
        ]
      });
    }
  } catch {
    // Ignore fallback read errors
  }

  return NextResponse.json({
    a1: { tau: 0.65 },
    retrieval_backend: "in-memory",
  });
}
