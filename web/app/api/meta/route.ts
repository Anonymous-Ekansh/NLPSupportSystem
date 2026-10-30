import { NextResponse } from "next/server";
import {
  fetchUpstream,
  readConfigMeta,
  sanitizeMeta,
} from "@/lib/server/upstream";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const upstream = await fetchUpstream(
      "/meta",
      { method: "GET" },
      { attempts: 2, timeouts: [6000, 4000], backoffs: [300] }
    );

    if (upstream.ok) {
      return NextResponse.json(sanitizeMeta(await upstream.json()));
    }
  } catch {
    // Fall through to the checked-in calibration config.
  }

  try {
    return NextResponse.json(readConfigMeta());
  } catch {
    return NextResponse.json({
      a1: {
        served_level: null,
        temperature: null,
        tau: null,
        s_min: null,
        n_classes: null,
      },
      retrieval_backend: "unknown",
      fallback: true,
    });
  }
}
