/**
 * Sarvam AI Integration Skeleton (Planned Voice Provider)
 *
 * NOTE: This file is a typed skeleton only. No network calls are made.
 * All functions currently throw "Sarvam not connected".
 *
 * Intended Architecture & Flow:
 * 1. Speech-to-Text (STT):
 *    - The browser captures user audio via MediaRecorder (e.g., audio/webm or audio/wav).
 *    - The client sends audio Blob as multipart/form-data to the server proxy: POST /api/voice/stt.
 *    - The server route reads SARVAM_API_KEY from environment (server-side only, never exposed to client).
 *    - The server forwards audio to Sarvam STT:
 *        POST https://api.sarvam.ai/speech-to-text
 *        Header: api-subscription-key: <SARVAM_API_KEY>
 *    - Sarvam returns { transcript: string, language_code?: string }.
 *    - The transcript is then submitted to /api/resolve as plain text with source: "voice".
 *
 * 2. Text-to-Speech (TTS):
 *    - When the user clicks the play button on a bot message, the client calls POST /api/voice/tts
 *      with { text: string, languageCode?: string }.
 *    - The server forwards request to Sarvam TTS:
 *        POST https://api.sarvam.ai/text-to-speech
 *        Header: api-subscription-key: <SARVAM_API_KEY>
 *        Body: {
 *          "inputs": [chunkText],
 *          "target_language_code": "en-IN" | "hi-IN" | ...,
 *          "speaker": "meera" | "pavithra" | "maitreyi" | "arvind",
 *          "model": "bulbul:v1"
 *        }
 *    - Audio is returned as base64-encoded audio in the "audios" array.
 *    - Long responses must be chunked to respect Sarvam's per-request character limit (typically ~500 chars).
 *    - The browser receives audio and plays it via HTML5 Audio.
 *
 * Important:
 * - Sarvam voice endpoints authenticate with the "api-subscription-key" header.
 * - Always verify endpoint names, model names, supported languages, and limits against
 *   https://docs.sarvam.ai before implementing, as models and parameters change.
 */

export interface SarvamSTTRequest {
  audio: Blob | Buffer;
  languageCode?: string;
}

export interface SarvamSTTResponse {
  transcript: string;
  language_code?: string;
}

export interface SarvamTTSRequest {
  text: string;
  languageCode?: string;
  speaker?: string;
}

export interface SarvamTTSResponse {
  audios: string[]; // base64 encoded audio strings
}

/**
 * Skeleton for Sarvam Speech-to-Text
 */
export async function transcribeAudio(
  _req: SarvamSTTRequest
): Promise<SarvamSTTResponse> {
  void _req;
  throw new Error("Sarvam not connected");
}

/**
 * Skeleton for Sarvam Text-to-Speech
 */
export async function synthesizeSpeech(
  _req: SarvamTTSRequest
): Promise<SarvamTTSResponse> {
  void _req;
  throw new Error("Sarvam not connected");
}
