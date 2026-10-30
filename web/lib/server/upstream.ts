import fs from "node:fs";
import path from "node:path";
import type {
  A1Output,
  A2Output,
  Health,
  MapPointsFile,
  Meta,
  ResolveResult,
  TraceStep,
} from "@/lib/api";

const API_BASE = (process.env.NLP_API_URL || "http://localhost:8000").replace(
  /\/+$/,
  ""
);

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

interface UpstreamOptions {
  attempts?: number;
  timeouts?: number[];
  backoffs?: number[];
}

export class UpstreamError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "UpstreamError";
  }
}

/**
 * Fetch the Python API with per-attempt timeouts and backoff. Rate-limit and
 * validation responses are returned as-is so the route can explain them.
 */
export async function fetchUpstream(
  pathname: string,
  init: RequestInit = {},
  options: UpstreamOptions = {}
) {
  const attempts = options.attempts ?? 3;
  const timeouts = options.timeouts ?? [30000, 15000, 10000];
  const backoffs = options.backoffs ?? [500, 1500];
  let lastError: unknown = new UpstreamError("Upstream did not respond");

  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const controller = new AbortController();
    const timeout = setTimeout(
      () => controller.abort(),
      timeouts[Math.min(attempt, timeouts.length - 1)]
    );

    try {
      const response = await fetch(`${API_BASE}${pathname}`, {
        ...init,
        cache: "no-store",
        signal: controller.signal,
      });
      clearTimeout(timeout);

      if (response.ok) return response;
      if (response.status === 422 || response.status === 429) return response;

      lastError = new UpstreamError(`Upstream responded ${response.status}`);
      if (response.status < 500) return response;
    } catch (error) {
      clearTimeout(timeout);
      lastError = error;
    }

    if (attempt < attempts - 1) {
      await sleep(backoffs[Math.min(attempt, backoffs.length - 1)]);
    }
  }

  throw lastError;
}

function asString(value: unknown, fallback = "") {
  return typeof value === "string" ? value : fallback;
}

function asNumber(value: unknown, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function asNullableNumber(value: unknown) {
  if (value === null || value === undefined) return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

/**
 * Whitelist the public contract. In particular, roadmap / planned-agent
 * fields from the backend are deliberately dropped.
 */
export function sanitizeResolve(raw: unknown): ResolveResult {
  const data = (raw ?? {}) as Record<string, unknown>;
  const rawSteps = Array.isArray(data.trace) ? data.trace : [];
  const trace: TraceStep[] = rawSteps
    .map((item) => {
      const step = (item ?? {}) as Record<string, unknown>;
      return {
        agent: asString(step.agent),
        input:
          step.input && typeof step.input === "object"
            ? (step.input as Record<string, unknown>)
            : {},
        output: step.output,
        latency_ms: asNumber(step.latency_ms),
      };
    })
    .filter(
      (step) =>
        step.agent === "IntentAnalyst" ||
        step.agent === "ResponseSpecialist"
    );

  for (const step of trace) {
    if (step.output && typeof step.output === "object") {
      // A1 / A2 output is consumed by typed accessors in lib/api.ts.
      const output = step.output as Record<string, unknown>;
      if (
        step.agent === "IntentAnalyst" &&
        !Array.isArray(output.top3)
      ) {
        (output as unknown as A1Output).top3 = [];
      }
      if (step.agent === "ResponseSpecialist") {
        const a2 = output as unknown as A2Output;
        if (!Array.isArray(a2.supporting_examples)) a2.supporting_examples = [];
        if (!Array.isArray(a2.used_intents)) a2.used_intents = [];
      }
    }
  }

  const map = (data.map ?? {}) as Record<string, unknown>;

  return {
    query: asString(data.query),
    trace,
    final_response: asString(data.final_response),
    total_latency_ms: asNumber(data.total_latency_ms),
    map: {
      x: asNullableNumber(map.x),
      y: asNullableNumber(map.y),
      confidence: asNumber(map.confidence),
    },
    retrieval_backend: asString(data.retrieval_backend, "in-memory"),
  };
}

export function sanitizeMeta(raw: unknown): Meta {
  const data = (raw ?? {}) as Record<string, unknown>;
  const a1 = (data.a1 ?? {}) as Record<string, unknown>;
  return {
    a1: {
      served_level: (a1.served_level as string | null) ?? null,
      temperature: asNullableNumber(a1.temperature),
      tau: asNullableNumber(a1.tau),
      s_min: asNullableNumber(a1.s_min),
      n_classes: asNullableNumber(a1.n_classes),
    },
    retrieval_backend: asString(data.retrieval_backend, "in-memory"),
  };
}

export function sanitizeHealth(raw: unknown): Health {
  const data = (raw ?? {}) as Record<string, unknown>;
  const healthy = data.status === "ok" && Boolean(data.model_loaded);
  return {
    status: healthy ? "ok" : "waking",
    model_loaded: Boolean(data.model_loaded),
    retrieval_backend: asString(data.retrieval_backend, "in-memory"),
    agents: Array.isArray(data.agents)
      ? data.agents.filter((agent) => agent === "a1" || agent === "a2")
      : ["a1", "a2"],
  };
}

export function readJsonFile<T>(relativePath: string): T {
  const filePath = path.join(process.cwd(), relativePath);
  return JSON.parse(fs.readFileSync(filePath, "utf-8")) as T;
}

export function readConfigMeta(): Meta {
  const config = readJsonFile<{
    served_level?: string;
    temperature?: number;
    tau?: number;
    s_min?: number;
    per_intent_f1?: Record<string, number>;
  }>("fixtures/a1.json");

  return {
    a1: {
      served_level: config.served_level ?? null,
      temperature: config.temperature ?? null,
      tau: config.tau ?? null,
      s_min: config.s_min ?? null,
      n_classes: config.per_intent_f1
        ? Object.keys(config.per_intent_f1).length
        : null,
    },
    retrieval_backend: "in-memory",
    fallback: true,
  };
}

export function readMapPoints(): MapPointsFile {
  for (const candidate of [
    "public/data/map_points.json",
    "public/map_points.json",
  ]) {
    try {
      return readJsonFile<MapPointsFile>(candidate);
    } catch {
      // Try the next known location.
    }
  }
  throw new Error("Map points are unavailable");
}

function normalizeQuestion(value: string) {
  return value
    .trim()
    .toLowerCase()
    .replace(/\s+/g, " ")
    .replace(/[?.!]+$/g, "");
}

/**
 * Last-resort demo fallback. Only exact matches from the checked-in cache are
 * used, so a judge can still see the full two-agent trace while free hosting
 * wakes up. The result is marked `cached` and labelled honestly in the UI.
 */
export function readDemoResult(query: string): ResolveResult | null {
  try {
    const cache = readJsonFile<Record<string, unknown>>(
      "public/demo_cache.json"
    );
    const target = normalizeQuestion(query);
    for (const [key, value] of Object.entries(cache)) {
      if (normalizeQuestion(key) !== target) continue;
      return { ...sanitizeResolve(value), cached: true };
    }
  } catch {
    // No cache, no fallback.
  }
  return null;
}

/** Strip roadmap fields from any raw resolve payload before it leaves the server. */
export function stripPrivateFields<T extends Record<string, unknown>>(value: T) {
  const clone = { ...value };
  delete clone.planned_agents;
  return clone;
}
