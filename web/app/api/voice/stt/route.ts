import { NextResponse } from "next/server";

/**
 * Voice STT Route Skeleton
 *
 * Future Sarvam integration:
 * - Parse incoming multipart/form-data for audio blob.
 * - Retrieve process.env.SARVAM_API_KEY (server env only).
 * - Forward audio to Sarvam STT endpoint:
 *     POST https://api.sarvam.ai/speech-to-text
 *     Headers: { "api-subscription-key": process.env.SARVAM_API_KEY }
 * - Return { transcript: data.transcript, language_code: data.language_code }.
 *
 * Today: returns HTTP 501 { error: "voice_not_configured" } and makes no outbound fetch.
 */
export async function POST() {
  return NextResponse.json(
    { error: "voice_not_configured" },
    { status: 501 }
  );
}
