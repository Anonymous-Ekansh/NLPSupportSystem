import { NextResponse } from "next/server";

/**
 * Voice TTS Route Skeleton
 *
 * Future Sarvam integration:
 * - Parse JSON body: { text: string, languageCode?: string, speaker?: string }
 * - Retrieve process.env.SARVAM_API_KEY (server env only).
 * - Chunk text if exceeding Sarvam character limits.
 * - Forward request to Sarvam TTS endpoint:
 *     POST https://api.sarvam.ai/text-to-speech
 *     Headers: { "api-subscription-key": process.env.SARVAM_API_KEY, "Content-Type": "application/json" }
 *     Body: { inputs: [chunkText], target_language_code: languageCode || "en-IN", model: "bulbul:v1" }
 * - Return { audios: data.audios } (base64 audio strings).
 *
 * Today: returns HTTP 501 { error: "voice_not_configured" } and makes no outbound fetch.
 */
export async function POST() {
  return NextResponse.json(
    { error: "voice_not_configured" },
    { status: 501 }
  );
}
