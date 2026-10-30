import { z } from "zod";

// ── A1 Schemas ────────────────────────────────────────────────────────
export const A1Top3ItemSchema = z.object({
  intent: z.string(),
  prob: z.number(),
});

export const A1OutputSchema = z.object({
  query: z.string(),
  intent: z.string(),
  category: z.string(),
  confidence: z.number(),
  margin: z.number(),
  top3: z.array(A1Top3ItemSchema),
  top1_similarity: z.number().nullable().optional(),
  needs_clarification: z.boolean(),
});

// ── A2 Schemas ────────────────────────────────────────────────────────
export const SupportingExampleSchema = z.object({
  instruction: z.string(),
  response: z.string(),
  similarity: z.number(),
});

export const SelfCheckSchema = z.object({
  placeholder_safe: z.boolean(),
  source_similarity_ok: z.boolean(),
  response_intent_ok: z.boolean(),
  check_passed: z.boolean(),
  candidates_rejected: z.number(),
});

export const A2OutputSchema = z.object({
  response: z.string(),
  supporting_examples: z.array(SupportingExampleSchema),
  used_intents: z.array(z.string()),
  retrieval_score: z.number(),
  low_confidence: z.boolean(),
  self_check: SelfCheckSchema,
});

// ── Trace & Pipeline Schemas ──────────────────────────────────────────
export const TraceStepSchema = z.object({
  agent: z.string(),
  input: z.record(z.string(), z.any()),
  output: z.any(),
  latency_ms: z.number(),
});

export const MapPositionSchema = z.object({
  x: z.number().nullable(),
  y: z.number().nullable(),
  confidence: z.number(),
});

export const PlannedAgentSchema = z.object({
  name: z.string(),
  status: z.string(),
  planned_function: z.string(),
});

export const ResolveResultSchema = z.object({
  query: z.string(),
  trace: z.array(TraceStepSchema),
  final_response: z.string(),
  total_latency_ms: z.number(),
  map: MapPositionSchema,
  retrieval_backend: z.string(),
  planned_agents: z.array(PlannedAgentSchema),
  cached: z.boolean().optional(),
  spokenAloud: z.boolean().optional(),
});

// ── Health, Meta & Feedback ───────────────────────────────────────────
export const HealthSchema = z.object({
  status: z.string(),
  agents: z.array(z.string()).optional(),
  model_loaded: z.boolean(),
  supabase_reachable: z.boolean().optional(),
  retrieval_backend: z.string().optional(),
});

export const MetaSchema = z.object({
  a1: z
    .object({
      served_level: z.string().nullable().optional(),
      temperature: z.number().nullable().optional(),
      tau: z.number().nullable().optional(),
      s_min: z.number().nullable().optional(),
      n_classes: z.number().nullable().optional(),
    })
    .optional(),
  retrieval_backend: z.string().optional(),
  planned_agents: z.array(PlannedAgentSchema).optional(),
}).passthrough();

export const FeedbackSchema = z.object({
  query: z.string(),
  rating: z.number().int().min(1).max(5),
  note: z.string().optional().nullable(),
});

// ── Inferred Types ────────────────────────────────────────────────────
export type A1Top3Item = z.infer<typeof A1Top3ItemSchema>;
export type A1Output = z.infer<typeof A1OutputSchema>;
export type SupportingExample = z.infer<typeof SupportingExampleSchema>;
export type SelfCheck = z.infer<typeof SelfCheckSchema>;
export type A2Output = z.infer<typeof A2OutputSchema>;
export type TraceStep = z.infer<typeof TraceStepSchema>;
export type MapPosition = z.infer<typeof MapPositionSchema>;
export type PlannedAgent = z.infer<typeof PlannedAgentSchema>;
export type ResolveResult = z.infer<typeof ResolveResultSchema>;
export type Health = z.infer<typeof HealthSchema>;
export type Meta = z.infer<typeof MetaSchema>;
export type Feedback = z.infer<typeof FeedbackSchema>;
