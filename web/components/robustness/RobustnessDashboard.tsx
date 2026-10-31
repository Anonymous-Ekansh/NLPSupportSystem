"use client";

import {
  AlertTriangle,
  CheckCircle2,
  FlaskConical,
  Info,
  ShieldCheck,
} from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Chip } from "@/components/ui/Chip";
import { Card } from "@/components/ui/Card";
import { DecisionBadge } from "@/components/decisions/DecisionBadge";
import {
  DECISION_TONES,
  STRESS_LABELS,
  pct,
  type GroupMetrics,
  type L2DecisionsFile,
  type RobustnessFile,
  type TraceExamplesFile,
} from "@/lib/robustness";
import { cn } from "@/lib/utils";

interface RobustnessDashboardProps {
  robustness: RobustnessFile;
  decisions: L2DecisionsFile | null;
  traces: TraceExamplesFile | null;
}

export function RobustnessDashboard({
  robustness,
  decisions,
  traces,
}: RobustnessDashboardProps) {
  const { overall, meta } = robustness;
  const day1 = overall.day1_reference;
  const delta = overall.intent_accuracy_delta_vs_day1;
  const selective = robustness.selective_prediction;
  const curve = selective.risk_coverage_curve ?? [];

  return (
    <div className="space-y-6">
      {/* ── provenance ──────────────────────────────────────────── */}
      <div className="rounded-[12px] border border-line bg-soft px-3.5 py-2.5">
        <div className="flex items-start gap-2">
          <Info
            size={14}
            aria-hidden="true"
            className="mt-0.5 shrink-0 text-ink-faint"
          />
          <p className="text-xs leading-[18px] text-ink-secondary">
            {meta.clean_sample_n.toLocaleString("en-US")} held-out{" "}
            <span className="font-medium text-ink">{meta.split}</span> questions
            (seed {meta.seed}), policy{" "}
            <span className="font-mono">{meta.policy.path}</span> tuned on{" "}
            {meta.policy.tuned_on}. Generated {meta.generated_at}. Groups with
            n &lt; {meta.low_sample_threshold} are marked low-sample.
          </p>
        </div>
      </div>

      {/* ── headline ────────────────────────────────────────────── */}
      <section>
        <SectionTitle
          title="Held-out accuracy"
          caption="Intent and category metrics on the same rows Day 1 reported on."
        />
        <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Metric label="intent accuracy" value={pct(overall.intent_accuracy)} />
          <Metric label="intent macro-F1" value={pct(overall.intent_macro_f1)} />
          <Metric
            label="category accuracy"
            value={pct(overall.category_accuracy)}
          />
          <Metric
            label="answers checked"
            value={String(robustness.false_action_claims.answers_checked)}
          />
        </div>

        {day1 ? (
          <div className="mt-2 flex flex-wrap items-center gap-1.5 text-xs text-ink-secondary">
            <span>
              Day 1 (full {day1.n_samples.toLocaleString("en-US")} rows):{" "}
              <span className="font-mono text-ink">
                {pct(day1.intent_accuracy)}
              </span>{" "}
              intent ·{" "}
              <span className="font-mono text-ink">
                {pct(day1.intent_f1_macro)}
              </span>{" "}
              macro-F1 ·{" "}
              <span className="font-mono text-ink">
                {pct(day1.category_accuracy)}
              </span>{" "}
              category
            </span>
            {typeof delta === "number" ? (
              <Chip
                tone={overall["matches_day1_within_0.01"] ? "success" : "warning"}
              >
                {overall["matches_day1_within_0.01"] ? "matches" : "differs"} ·{" "}
                {delta >= 0 ? "+" : ""}
                {(delta * 100).toFixed(2)} pts
              </Chip>
            ) : null}
          </div>
        ) : null}
      </section>

      {/* ── decisions ───────────────────────────────────────────── */}
      <section>
        <SectionTitle
          title="What the system decided"
          caption="Every held-out question gets exactly one of three outcomes."
        />
        <div className="mt-2 grid grid-cols-3 gap-2">
          {(["ANSWER", "CLARIFY", "ESCALATE"] as const).map((key) => (
            <div
              key={key}
              className="rounded-[12px] border border-line bg-card px-3 py-2.5"
            >
              <DecisionBadge decision={key} />
              <p className="mt-1.5 font-mono text-xl tabular-nums text-ink">
                {overall.decisions[key] ?? 0}
              </p>
              <p className="text-xs text-ink-secondary">
                {pct(overall.decision_shares[key] ?? 0)} of {overall.n}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* ── selective prediction ────────────────────────────────── */}
      <section>
        <SectionTitle
          title="Selective prediction"
          caption="ANSWER is the only committed decision; CLARIFY and ESCALATE abstain."
        />
        <div className="mt-2 grid gap-3 lg:grid-cols-[38fr_62fr]">
          <div className="space-y-2">
            <div className="grid grid-cols-2 gap-2">
              <Metric label="coverage" value={pct(selective.coverage)} />
              <Metric
                label="accuracy on answered"
                value={pct(selective.accuracy_on_answered)}
              />
            </div>
            <p className="text-xs leading-[18px] text-ink-secondary">
              It answered {selective.n_answered} of {overall.n} questions and was
              right on {selective.answered_with_correct_intent}. Answering
              everything would score {pct(selective.accuracy_if_all_answered)}.
            </p>
          </div>

          {curve.length > 1 ? (
            <Card className="p-3">
              <p className="mb-1 text-xs font-medium text-ink">
                Risk–coverage curve
              </p>
              <p className="mb-1.5 text-xs text-ink-secondary">
                Accuracy as the confidence floor rises (fewer answers, more
                certainty).
              </p>
              <div className="h-[190px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={curve}
                    margin={{ top: 4, right: 8, bottom: 0, left: -18 }}
                  >
                    <CartesianGrid stroke="#ECEBE8" vertical={false} />
                    <XAxis
                      dataKey="coverage"
                      tickFormatter={(value: number) => `${Math.round(value * 100)}%`}
                      stroke="#71717A"
                      fontSize={11}
                      tickLine={false}
                    />
                    <YAxis
                      domain={[0.8, 1]}
                      tickFormatter={(value: number) => `${Math.round(value * 100)}%`}
                      stroke="#71717A"
                      fontSize={11}
                      tickLine={false}
                    />
                    <Tooltip
                      contentStyle={{
                        border: "1px solid #ECEBE8",
                        borderRadius: 8,
                        fontSize: 12,
                      }}
                      formatter={(value: number) => [pct(value), "accuracy"]}
                      labelFormatter={(value: number) =>
                        `coverage ${pct(value)}`
                      }
                    />
                    <Line
                      type="monotone"
                      dataKey="accuracy"
                      stroke="#4F46E5"
                      strokeWidth={2}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </Card>
          ) : null}
        </div>
      </section>

      {/* ── stress sets ─────────────────────────────────────────── */}
      <section>
        <SectionTitle
          title="Stress tests"
          caption="Cases where a confident answer would be the wrong behaviour."
        />
        <div className="mt-2 overflow-x-auto">
          <table className="w-full min-w-[640px] border-collapse text-sm">
            <thead>
              <tr className="border-b border-line text-left text-xs uppercase tracking-[0.06em] text-ink-faint">
                <th className="py-1.5 pr-2 font-medium">set</th>
                <th className="py-1.5 pr-2 font-medium">n</th>
                <th className="py-1.5 pr-2 font-medium">correctly routed</th>
                <th className="py-1.5 pr-2 font-medium">wrongly answered</th>
                <th className="py-1.5 font-medium">decisions</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(robustness.stress_sets).map(([key, set]) => (
                <tr key={key} className="border-b border-line last:border-0">
                  <td className="py-2 pr-2">
                    <span className="font-medium text-ink">
                      {STRESS_LABELS[key] ?? key}
                    </span>
                    {set.low_sample ? (
                      <Chip tone="outline" className="ml-1.5">
                        low sample
                      </Chip>
                    ) : null}
                  </td>
                  <td className="py-2 pr-2 font-mono tabular-nums text-ink-secondary">
                    {set.n}
                  </td>
                  <td className="py-2 pr-2">
                    <ScoreCell value={set.correctly_routed_share} good />
                  </td>
                  <td className="py-2 pr-2">
                    <ScoreCell value={set.wrongly_answered_share} bad />
                  </td>
                  <td className="py-2 font-mono text-xs text-ink-secondary">
                    {Object.entries(set.decisions)
                      .map(([decision, count]) => `${decision} ${count}`)
                      .join(" · ")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {robustness.stress_sets.ambiguous?.assumption ? (
          <p className="mt-1.5 text-xs leading-[18px] text-ink-secondary">
            {robustness.stress_sets.ambiguous.assumption}
          </p>
        ) : null}
        {robustness.stress_sets.contradictory_neighbour?.probe_note ? (
          <p className="mt-1 text-xs leading-[18px] text-ink-secondary">
            {robustness.stress_sets.contradictory_neighbour.probe_note}
          </p>
        ) : null}
      </section>

      {/* ── ablation ────────────────────────────────────────────── */}
      <section>
        <SectionTitle
          title="Day 1 vs Level 2"
          caption="Same stress cases, both pipelines. Day 1 has no decision layer, so it always answers."
        />
        <div className="mt-2 overflow-x-auto">
          <table className="w-full min-w-[640px] border-collapse text-sm">
            <thead>
              <tr className="border-b border-line text-left text-xs uppercase tracking-[0.06em] text-ink-faint">
                <th className="py-1.5 pr-2 font-medium">set</th>
                <th className="py-1.5 pr-2 font-medium">L2 wrong-answer</th>
                <th className="py-1.5 pr-2 font-medium">L2 correct routing</th>
                <th className="py-1.5 pr-2 font-medium">Day 1 asks a question</th>
                <th className="py-1.5 font-medium">Day 1 action claim</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(robustness.ablation_day1_vs_l2).map(
                ([key, row]) => (
                  <tr key={key} className="border-b border-line last:border-0">
                    <td className="py-2 pr-2 font-medium text-ink">
                      {STRESS_LABELS[key] ?? key}
                    </td>
                    <td className="py-2 pr-2">
                      <ScoreCell value={row.level2_wrong_answer_share} bad />
                    </td>
                    <td className="py-2 pr-2">
                      <ScoreCell value={row.level2_correct_routing_share} good />
                    </td>
                    <td className="py-2 pr-2 font-mono tabular-nums text-ink-secondary">
                      {pct(row.day1_asks_question_share)}
                    </td>
                    <td className="py-2 font-mono tabular-nums text-ink-secondary">
                      {pct(row.day1_action_claim_share)}
                    </td>
                  </tr>
                )
              )}
            </tbody>
          </table>
        </div>
      </section>

      {/* ── claims ──────────────────────────────────────────────── */}
      <section>
        <SectionTitle
          title="Responsible behaviour"
          caption="No output may claim an order, payment, refund or account was changed."
        />
        <div
          className={cn(
            "mt-2 flex items-start gap-2 rounded-[12px] border px-3 py-2.5",
            robustness.false_action_claims.false_action_claims === 0
              ? "border-success/20 bg-success-soft"
              : "border-error/20 bg-error-soft"
          )}
        >
          {robustness.false_action_claims.false_action_claims === 0 ? (
            <ShieldCheck
              size={15}
              aria-hidden="true"
              className="mt-0.5 shrink-0 text-success"
            />
          ) : (
            <AlertTriangle
              size={15}
              aria-hidden="true"
              className="mt-0.5 shrink-0 text-error"
            />
          )}
          <p className="text-sm leading-[21px] text-ink">
            <span className="font-semibold">
              {robustness.false_action_claims.false_action_claims}
            </span>{" "}
            false action claims in{" "}
            {robustness.false_action_claims.answers_checked} ANSWER outputs
            (target {robustness.false_action_claims.target}).
          </p>
        </div>
      </section>

      {/* ── flag groups ─────────────────────────────────────────── */}
      <section>
        <SectionTitle
          title="Dataset variation groups"
          caption="Accuracy sliced by the dataset's own `flags` letters. Rows carry several letters, so the groups overlap."
        />
        <GroupTable
          rows={robustness.by_flag}
          firstHeader="flag"
          deriveLabel={(row) =>
            row.group ? `${row.group} · ${row.label ?? ""}` : "—"
          }
          deriveCaption={(row) => row.meaning ?? ""}
        />
      </section>

      {/* ── perturbations ───────────────────────────────────────── */}
      <section>
        <SectionTitle
          title="Synthetic perturbations"
          caption="Seeded, deterministic rewrites of held-out questions. Labels are inherited from the original row — these are synthetic inputs, not dataset ground truth."
        />
        <GroupTable
          rows={robustness.by_perturbation}
          firstHeader="perturbation"
          deriveLabel={(row) => row.group ?? "—"}
          deriveCaption={(row) =>
            typeof row.transform_coverage === "number"
              ? `applies to ${pct(row.transform_coverage)} of the split`
              : ""
          }
        />
      </section>

      {/* ── per-intent decisions ────────────────────────────────── */}
      {decisions ? (
        <section>
          <SectionTitle
            title="Decisions by intent"
            caption="Where the system chooses to abstain instead of answering."
          />
          <div className="mt-2 overflow-x-auto">
            <table className="w-full min-w-[560px] border-collapse text-sm">
              <thead>
                <tr className="border-b border-line text-left text-xs uppercase tracking-[0.06em] text-ink-faint">
                  <th className="py-1.5 pr-2 font-medium">intent</th>
                  <th className="py-1.5 pr-2 font-medium">n</th>
                  <th className="py-1.5 pr-2 font-medium">answer</th>
                  <th className="py-1.5 pr-2 font-medium">clarify</th>
                  <th className="py-1.5 font-medium">escalate</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(decisions.by_intent).map(([intent, row]) => (
                  <tr key={intent} className="border-b border-line last:border-0">
                    <td className="py-1.5 pr-2 font-mono text-xs text-ink">
                      {intent}
                      {row.low_sample ? (
                        <Chip tone="outline" className="ml-1.5">
                          low
                        </Chip>
                      ) : null}
                    </td>
                    <td className="py-1.5 pr-2 font-mono tabular-nums text-ink-secondary">
                      {row.n}
                    </td>
                    {(["ANSWER", "CLARIFY", "ESCALATE"] as const).map((key) => (
                      <td
                        key={key}
                        className="py-1.5 pr-2 font-mono tabular-nums text-ink-secondary"
                      >
                        {row.decisions[key] ?? 0}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {/* ── curated traces ──────────────────────────────────────── */}
      {traces?.examples?.length ? (
        <section>
          <SectionTitle
            title="Curated traces"
            caption="Five real runs covering accept, revise, clarify and escalate."
          />
          <div className="mt-2 grid gap-2 sm:grid-cols-2">
            {traces.examples.map((example) => (
              <div
                key={example.label}
                className="rounded-[12px] border border-line bg-card p-3"
              >
                <div className="flex items-center justify-between gap-2">
                  <DecisionBadge
                    decision={
                      example.decision as "ANSWER" | "CLARIFY" | "ESCALATE"
                    }
                  />
                  <span className="font-mono text-[11px] uppercase tracking-[0.06em] text-ink-faint">
                    {example.label}
                  </span>
                </div>
                <p className="mt-1.5 line-clamp-2 text-xs text-ink-secondary">
                  “{example.query}”
                </p>
                <p className="mt-1 line-clamp-3 text-xs leading-[18px] text-ink">
                  {example.response}
                </p>
                <div className="mt-1.5 flex flex-wrap gap-1">
                  {example.reason_codes.slice(0, 3).map((code) => (
                    <span
                      key={code}
                      className="rounded-[6px] border border-line bg-soft px-1.5 py-0.5 font-mono text-[11px] text-ink-secondary"
                    >
                      {code}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </section>
      ) : null}

      {robustness.false_action_claims.examples.length ? (
        <section>
          <SectionTitle
            title="Claim examples"
            caption="Every detected claim, so the guard can be checked by hand."
          />
          <ul className="mt-2 space-y-1.5">
            {robustness.false_action_claims.examples.map((item, index) => (
              <li
                key={index}
                className="rounded-[8px] border border-error/15 bg-error-soft px-2.5 py-1.5 text-xs text-ink"
              >
                <span className="font-mono text-[11px] text-error-ink">
                  {item.match}
                </span>{" "}
                — {item.query}
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      <p className="flex items-center gap-1.5 text-xs text-ink-faint">
        <FlaskConical size={12} aria-hidden="true" />
        {meta.synthetic_note}
      </p>
    </div>
  );
}

function GroupTable({
  rows,
  firstHeader,
  deriveLabel,
  deriveCaption,
}: {
  rows: GroupMetrics[];
  firstHeader: string;
  deriveLabel: (row: GroupMetrics) => string;
  deriveCaption: (row: GroupMetrics) => string;
}) {
  return (
    <div className="mt-2 overflow-x-auto">
      <table className="w-full min-w-[600px] border-collapse text-sm">
        <thead>
          <tr className="border-b border-line text-left text-xs uppercase tracking-[0.06em] text-ink-faint">
            <th className="py-1.5 pr-2 font-medium">{firstHeader}</th>
            <th className="py-1.5 pr-2 font-medium">n</th>
            <th className="py-1.5 pr-2 font-medium">intent</th>
            <th className="py-1.5 pr-2 font-medium">macro-F1</th>
            <th className="py-1.5 pr-2 font-medium">category</th>
            <th className="py-1.5 font-medium">decisions</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={row.group ?? deriveLabel(row)}
              className="border-b border-line last:border-0"
            >
              <td className="py-1.5 pr-2">
                <span className="font-medium text-ink">{deriveLabel(row)}</span>
                {row.low_sample ? (
                  <Chip tone="outline" className="ml-1.5">
                    low sample
                  </Chip>
                ) : null}
                <span className="block text-xs text-ink-secondary">
                  {deriveCaption(row)}
                </span>
              </td>
              <td className="py-1.5 pr-2 font-mono tabular-nums text-ink-secondary">
                {row.n}
              </td>
              <td className="py-1.5 pr-2 font-mono tabular-nums text-ink">
                {pct(row.intent_accuracy)}
              </td>
              <td className="py-1.5 pr-2 font-mono tabular-nums text-ink-secondary">
                {pct(row.intent_macro_f1)}
              </td>
              <td className="py-1.5 pr-2 font-mono tabular-nums text-ink-secondary">
                {pct(row.category_accuracy)}
              </td>
              <td className="py-1.5 font-mono text-xs text-ink-secondary">
                {Object.entries(row.decisions ?? {})
                  .map(([decision, count]) => `${decision} ${count}`)
                  .join(" · ")}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ScoreCell({ value, good, bad }: { value: number; good?: boolean; bad?: boolean }) {
  const tone =
    good && value >= 0.6
      ? "text-success-ink"
      : bad && value > 0.2
        ? "text-error-ink"
        : "text-ink";
  return (
    <span className={cn("font-mono text-sm tabular-nums", tone)}>
      {pct(value)}
    </span>
  );
}

function SectionTitle({ title, caption }: { title: string; caption: string }) {
  return (
    <div>
      <h2 className="text-base font-semibold text-ink">{title}</h2>
      <p className="mt-0.5 max-w-[720px] text-xs leading-[18px] text-ink-secondary">
        {caption}
      </p>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[12px] border border-line bg-card px-3 py-2.5">
      <p className="font-mono text-[11px] uppercase tracking-[0.06em] text-ink-faint">
        {label}
      </p>
      <p className="mt-0.5 font-mono text-xl tabular-nums text-ink">{value}</p>
    </div>
  );
}
