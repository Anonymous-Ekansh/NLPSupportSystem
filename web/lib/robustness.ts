/**
 * Types for `public/evidence/robustness.json`, `l2_decisions.json` and
 * `trace_examples.json`, written by `scripts/eval_l2.py`.
 *
 * Nothing here is computed in the browser: every number shown on /robustness
 * is read from those files. If a file is missing the page says so.
 */

export interface GroupMetrics {
  group?: string;
  label?: string;
  meaning?: string;
  n: number;
  low_sample: boolean;
  intent_accuracy: number;
  intent_macro_f1: number;
  category_accuracy: number;
  decisions: Record<string, number>;
  decision_shares: Record<string, number>;
  transform_coverage?: number;
  synthetic?: boolean;
  note?: string;
  multi_label?: boolean;
}

export interface OverallMetrics extends GroupMetrics {
  day1_reference?: {
    n_samples: number;
    intent_accuracy: number;
    intent_f1_macro: number;
    category_accuracy: number;
  };
  day1_note?: string;
  intent_accuracy_delta_vs_day1?: number;
  "matches_day1_within_0.01"?: boolean;
}

export interface RiskCoveragePoint {
  coverage: number;
  accuracy: number;
  n_answered: number;
  confidence_floor: number;
}

export interface SelectivePrediction {
  coverage: number;
  n_answered: number;
  accuracy_on_answered: number | null;
  answered_with_correct_intent: number;
  accuracy_if_all_answered: number;
  risk_coverage_curve?: RiskCoveragePoint[];
}

export interface StressCase {
  id: string;
  query: string;
  decision: string | null;
  correct: boolean;
  expected: string[];
}

export interface StressSet {
  n: number;
  low_sample: boolean;
  correctly_routed: number;
  correctly_routed_share: number;
  wrongly_answered: number;
  wrongly_answered_share: number;
  decisions: Record<string, number>;
  cases?: StressCase[];
  assumption?: string;
  probe_note?: string;
  preferred_decision?: string;
}

export interface AblationRow {
  n: number;
  level2_wrong_answer_share: number;
  level2_correct_routing_share: number;
  day1_answered_share: number;
  day1_asks_question_share: number;
  day1_action_claim_share: number;
}

export interface RobustnessFile {
  meta: {
    generated_at: string;
    split: string;
    n_split: number;
    seed: number;
    clean_sample_n: number;
    perturb_sample_n: number;
    low_sample_threshold: number;
    policy: { version?: string | number; path?: string; tuned_on?: string };
    thresholds?: Record<string, number>;
    runtime_s?: number;
    synthetic_note?: string;
  };
  overall: OverallMetrics;
  by_flag: GroupMetrics[];
  by_perturbation: GroupMetrics[];
  selective_prediction: SelectivePrediction;
  stress_sets: Record<string, StressSet>;
  false_action_claims: {
    answers_checked: number;
    false_action_claims: number;
    target: number;
    passed: boolean;
    examples: Array<{ query: string; match: string; response: string }>;
  };
  ablation_day1_vs_l2: Record<string, AblationRow>;
}

export interface L2DecisionsFile {
  meta: { generated_at: string; split: string; n: number; policy_version?: string };
  decisions: Record<string, number>;
  decision_shares: Record<string, number>;
  reason_codes: Record<string, number>;
  by_intent: Record<
    string,
    {
      n: number;
      low_sample: boolean;
      decision_shares: Record<string, number>;
      decisions: Record<string, number>;
      intent_accuracy: number;
    }
  >;
  by_category: Record<
    string,
    {
      n: number;
      low_sample: boolean;
      decision_shares: Record<string, number>;
      decisions: Record<string, number>;
    }
  >;
  selective_prediction: SelectivePrediction;
}

export interface TraceExample {
  label: string;
  why: string;
  query: string;
  decision: string;
  response: string;
  predicted_intent: string;
  human_intent: string;
  predicted_category: string;
  reason_codes: string[];
  total_latency_ms?: number;
  trace: Array<{
    agent: string;
    role?: string;
    decision?: string;
    evidence?: string[];
    output_summary?: string;
    latency_ms: number;
  }>;
}

export interface TraceExamplesFile {
  meta: { generated_at: string; split: string; note?: string };
  examples: TraceExample[];
}

/** Human labels for the stress sets, in display order. */
export const STRESS_LABELS: Record<string, string> = {
  ambiguous: "Ambiguous (two-intent blend)",
  missing_slot: "Missing essential detail",
  out_of_scope: "Out of scope",
  weak_evidence: "Weak evidence",
  contradictory_neighbour: "Contradictory retrieved evidence",
};

export const DECISION_TONES: Record<string, "success" | "warning" | "error"> = {
  ANSWER: "success",
  CLARIFY: "warning",
  ESCALATE: "error",
};

export function pct(value: number | null | undefined, digits = 1) {
  if (typeof value !== "number" || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}
