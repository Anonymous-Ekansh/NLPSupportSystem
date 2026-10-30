/**
 * Typed contract for the real /resolve, /meta, /health and /feedback API.
 * These types are shared by the route handlers and the browser. Never read
 * process.env from this file — the upstream URL is server-only.
 */

import { isFollowupEligible } from "@/lib/utils";

export interface Top3Intent {
  intent: string;
  prob: number;
}

export interface A1Output {
  query: string;
  intent: string;
  category: string;
  confidence: number;
  margin: number;
  top3: Top3Intent[];
  top1_similarity: number | null;
  needs_clarification: boolean;
}

export interface SupportingExample {
  instruction: string;
  response: string;
  similarity: number;
}

export interface SelfCheck {
  placeholder_safe: boolean;
  source_similarity_ok: boolean;
  response_intent_ok: boolean;
  check_passed: boolean;
  candidates_rejected: number;
  /** Present on the in-memory branch. */
  source_similarity?: number;
  /** Present on the in-memory branch. */
  response_predicted_intent?: string;
}

export interface A2Output {
  response: string;
  supporting_examples: SupportingExample[];
  used_intents: string[];
  retrieval_score: number;
  low_confidence: boolean;
  self_check: SelfCheck;
}

export interface TraceStep {
  agent: "IntentAnalyst" | "ResponseSpecialist" | string;
  input: Record<string, unknown>;
  output: unknown;
  latency_ms: number;
}

export interface MapPosition {
  x: number | null;
  y: number | null;
  confidence: number;
}

export interface ResolveResult {
  query: string;
  trace: TraceStep[];
  final_response: string;
  total_latency_ms: number;
  map: MapPosition;
  retrieval_backend: string;
  /** True when the live API was unavailable and a checked-in demo run was used. */
  cached?: boolean;
}

export interface MetaA1 {
  served_level: string | null;
  temperature: number | null;
  tau: number | null;
  s_min: number | null;
  n_classes: number | null;
}

export interface Meta {
  a1: MetaA1;
  retrieval_backend: string;
  /** True when served from the checked-in config because the API was cold. */
  fallback?: boolean;
}

export interface Health {
  status: "ok" | "waking" | "offline";
  model_loaded: boolean;
  retrieval_backend: string;
  agents?: string[];
}

export interface FeedbackPayload {
  query: string;
  rating: 1 | 5;
  note?: string;
}

/** A point from /map_points.json. */
export interface MapPoint {
  index: number;
  x: number;
  y: number;
  intent: string;
  category: string;
}

export interface MapPointsFile {
  n_total: number;
  n_projected: number;
  n_saved: number;
  method: string;
  coordinates: MapPoint[];
}

export const OUT_OF_SCOPE_RESPONSE =
  "I can only help with support topics like orders, refunds, accounts and shipping. Try rephrasing?";

export class ApiRequestError extends Error {
  status: number;
  code: string;

  constructor(message: string, status: number, code = "request_failed") {
    super(message);
    this.name = "ApiRequestError";
    this.status = status;
    this.code = code;
  }
}

async function parseError(response: Response): Promise<ApiRequestError> {
  let message = "Something went wrong. Please try again.";
  let code = "request_failed";

  try {
    const data = await response.json();
    if (typeof data?.error === "string") message = data.error;
    if (typeof data?.code === "string") code = data.code;
  } catch {
    // Keep the plain-English fallback.
  }

  if (response.status === 429) {
    message =
      "You're moving a little fast. Please wait a moment and try again.";
    code = "rate_limited";
  } else if (response.status === 422) {
    message =
      "That question is a little too long. Please keep it under 500 characters.";
    code = "too_long";
  } else if (response.status === 503 || response.status === 502) {
    message =
      "The assistant is taking longer than usual. Please try again in a moment.";
    code = "unavailable";
  }

  return new ApiRequestError(message, response.status, code);
}

export async function resolveQuestion(query: string): Promise<ResolveResult> {
  const response = await fetch("/api/resolve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });

  if (!response.ok) throw await parseError(response);
  return (await response.json()) as ResolveResult;
}

export async function sendFeedback(payload: FeedbackPayload) {
  const response = await fetch("/api/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) throw await parseError(response);
  return response.json() as Promise<{ status: string }>;
}

export async function fetchHealth(): Promise<Health> {
  const response = await fetch("/api/health", { cache: "no-store" });
  if (!response.ok) {
    const data = (await response.json().catch(() => null)) as Health | null;
    return (
      data ?? {
        status: "waking",
        model_loaded: false,
        retrieval_backend: "unknown",
      }
    );
  }
  return (await response.json()) as Health;
}

export async function fetchMeta(): Promise<Meta> {
  const response = await fetch("/api/meta", { cache: "no-store" });
  if (!response.ok) throw await parseError(response);
  return (await response.json()) as Meta;
}

export function getTraceStep(result: ResolveResult, agent: string) {
  return result.trace.find((step) =>
    step.agent.toLowerCase().includes(agent.toLowerCase())
  );
}

export function getA1Output(result: ResolveResult): A1Output | null {
  const step = getTraceStep(result, "IntentAnalyst");
  if (!step || !step.output || typeof step.output !== "object") return null;
  return step.output as A1Output;
}

export function getA2Output(result: ResolveResult): A2Output | null {
  const step = getTraceStep(result, "ResponseSpecialist");
  if (!step || !step.output || typeof step.output !== "object") return null;
  return step.output as A2Output;
}

export function isOutOfScope(
  result: ResolveResult,
  sMin: number | null | undefined
) {
  const a1 = getA1Output(result);
  if (!a1) return false;
  return (
    a1.needs_clarification &&
    typeof a1.top1_similarity === "number" &&
    typeof sMin === "number" &&
    a1.top1_similarity < sMin
  );
}

export function isUnsure(result: ResolveResult) {
  const a1 = getA1Output(result);
  return Boolean(a1?.needs_clarification);
}

export function followupIntents(result: ResolveResult): string[] {
  const a1 = getA1Output(result);
  if (!a1) return [];
  return a1.top3
    .map((item) => item.intent)
    .filter((intent) => intent !== a1.intent)
    .filter(isFollowupEligible)
    .slice(0, 3);
}
