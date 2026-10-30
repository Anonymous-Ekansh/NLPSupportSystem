/**
 * Types for the checked-in JSON files in `web/public/data/`.
 *
 * These files are generated from the NLP pipeline (see
 * scripts/build_web_data.py). The web app never invents a metric: if a file
 * is missing, the page renders an empty state that names it.
 */

import type { MapPoint } from "@/lib/api";

export interface SplitSizes {
  train: number;
  dev: number;
  test: number;
}

export interface CleaningStats {
  raw_train_rows: number;
  raw_validation_rows: number;
  duplicates_removed: number;
  leaked_rows_removed: number;
  seed: number;
}

export interface CategoryCount {
  category: string;
  count: number;
}

export interface ClassBalanceRow {
  intent: string;
  category: string;
  count: number;
}

export interface StyleCount {
  style: string;
  count: number;
  pct: number;
}

export interface DatasetStats {
  total_examples: number;
  intents: number;
  categories: number;
  splits: SplitSizes;
  cleaning: CleaningStats;
  category_counts: CategoryCount[];
  class_balance: ClassBalanceRow[];
  class_balance_ratio: {
    min: number;
    max: number;
    ratio: number;
  };
  styles: StyleCount[];
  style_note?: string;
}

export interface CalibrationBin {
  confidence: number;
  accuracy: number;
  count: number;
}

export interface AnswerCheckRow {
  key: string;
  label: string;
  pass_rate: number;
  fail_rate: number;
}

export interface RobustnessRow {
  style: string;
  macro_f1: number;
  support: number;
}

export interface ResultsData {
  n_test: number;
  intent_macro_f1: number;
  intent_macro_f1_ci: [number, number];
  intent_accuracy: number;
  category_accuracy: number;
  answer_relevance: {
    cosine_sim: number;
    cosine_sim_ci: [number, number];
    rouge_l_f1: number;
    self_check_pass_rate: number;
    self_check_pass_rate_ci: [number, number];
    served_mode: string;
  };
  calibration: {
    tau: number;
    s_min: number;
    ece_before: number;
    ece_after: number;
    bins_before: CalibrationBin[];
    bins_after: CalibrationBin[];
  };
  answer_checks: {
    pass_rate: number;
    pass_rate_ci: [number, number];
    checks: AnswerCheckRow[];
    rejected_share: number;
  };
  robustness: RobustnessRow[];
  confidence_rates: {
    avg_confidence: number;
    needs_clarification: number;
    a1_low_conf: number;
    category_accuracy: number;
  };
}

export interface IntentMetric {
  intent: string;
  category: string;
  f1: number;
  precision: number;
  recall: number;
  support: number;
}

export interface PerIntentFile {
  metrics: IntentMetric[];
}

export interface ConfusionCategoryFile {
  labels: string[];
  matrix: number[][];
  note?: string;
}

export interface LadderRow {
  level: string;
  name: string;
  macro_f1: number;
  ci_lo: number;
  ci_hi: number;
  accuracy: number;
  served: boolean;
}

export interface LadderFile {
  rows: LadderRow[];
}

export interface RetrievalModeRow {
  key: string;
  mode: string;
  cosine_sim: number;
  rouge_l_f1: number;
  intent_consistency: number;
  check_passed_rate: number;
  served: boolean;
}

export interface RoutingRow {
  mode: string;
  key: string;
  cosine_sim: number;
  intent_consistency: number;
}

export interface RetrievalModesFile {
  modes: RetrievalModeRow[];
  routing: {
    with_a1: RoutingRow[];
    without_a1: RoutingRow[];
  };
  served_mode: string;
  note?: string;
}

export interface OodFile {
  auroc_maxprob: number;
  auroc_top1sim: number;
  flagged_pct_in_domain: number;
  flagged_pct_ood: number;
  threshold_s_min: number;
  n_ood: number;
}

export interface BrowseExample {
  query: string;
  category: string;
  intent: string;
  style: string;
}

export interface TraceHandoff {
  intent: string;
  category: string;
  confidence: number;
  needs_clarification: boolean;
  top3: { intent: string; prob: number }[];
}

export interface TraceExample {
  query: string;
  a1_output: {
    query?: string;
    intent: string;
    category: string;
    confidence: number;
    margin?: number;
    top3: { intent: string; prob: number }[];
    top1_similarity?: number | null;
    needs_clarification: boolean;
  };
  a2_output: {
    response: string;
    supporting_examples?: {
      instruction: string;
      response: string;
      similarity: number;
    }[];
    used_intents?: string[];
    retrieval_score?: number;
    low_confidence?: boolean;
    self_check?: {
      placeholder_safe?: boolean;
      source_similarity_ok?: boolean;
      response_intent_ok?: boolean;
      check_passed?: boolean;
      candidates_rejected?: number;
    };
  };
}

export interface ExamplesFile {
  traces: TraceExample[];
  browse: BrowseExample[];
}

export interface MapPointWithQuery extends MapPoint {
  query?: string;
}

export function handoffFromTrace(trace: TraceExample): TraceHandoff {
  const { intent, category, confidence, needs_clarification, top3 } =
    trace.a1_output;
  return { intent, category, confidence, needs_clarification, top3 };
}

/** Tolerate either a bare array or a `{ rows: [...] }` wrapper. */
export function rowsOf<T>(value: T[] | { rows: T[] } | null): T[] {
  if (!value) return [];
  return Array.isArray(value) ? value : value.rows;
}

/** Tolerate either a bare array or a `{ metrics: [...] }` wrapper. */
export function metricsOf(
  value: IntentMetric[] | PerIntentFile | null
): IntentMetric[] {
  if (!value) return [];
  return Array.isArray(value) ? value : value.metrics;
}

/** Tolerate either an array of traces or `{ traces, browse }`. */
export function examplesOf(
  value: TraceExample[] | ExamplesFile | null
): ExamplesFile {
  if (!value) return { traces: [], browse: [] };
  if (Array.isArray(value)) return { traces: value, browse: [] };
  return {
    traces: Array.isArray(value.traces) ? value.traces : [],
    browse: Array.isArray(value.browse) ? value.browse : [],
  };
}
