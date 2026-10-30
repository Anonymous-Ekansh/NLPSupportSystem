"use client";

import dynamic from "next/dynamic";
import type { LucideIcon } from "lucide-react";
import {
  BarChart3,
  CheckCircle2,
  ChevronDown,
  Gauge,
  Layers,
  ShieldCheck,
  Target,
} from "lucide-react";
import type {
  ConfusionCategoryFile,
  IntentMetric,
  LadderRow,
  OodFile,
  ResultsData,
  RetrievalModesFile,
} from "@/lib/data";
import { Card } from "@/components/ui/Card";
import { Chip } from "@/components/ui/Chip";
import { InfoHint } from "@/components/ui/InfoHint";
import { EmptyDataFile } from "@/components/ui/EmptyDataFile";
import { F1Table } from "@/components/results/F1Table";
import { ConfusionHeatmap } from "@/components/results/ConfusionHeatmap";
import { cn, formatCount, formatPercent, formatScore } from "@/lib/utils";

const CalibrationChart = dynamic(
  () => import("@/components/results/charts/CalibrationChart"),
  { ssr: false, loading: () => <ChartSkeleton /> }
);
const LadderChart = dynamic(
  () => import("@/components/results/charts/LadderChart"),
  { ssr: false, loading: () => <ChartSkeleton /> }
);
const RetrievalChart = dynamic(
  () => import("@/components/results/charts/RetrievalChart"),
  { ssr: false, loading: () => <ChartSkeleton /> }
);

export function ResultsDashboard({
  results,
  perIntent,
  confusion,
  ladder,
  retrieval,
  ood,
}: {
  results: ResultsData | null;
  perIntent: IntentMetric[];
  confusion: ConfusionCategoryFile | null;
  ladder: LadderRow[];
  retrieval: RetrievalModesFile | null;
  ood: OodFile | null;
}) {
  if (!results) {
    return (
      <EmptyDataFile
        file="results.json"
        detail="Add results.json to web/public/data/ to populate the held-out metrics."
        className="mt-6"
      />
    );
  }

  const relevance = results.answer_relevance;
  const checks = results.answer_checks;
  const correctIn100 = Math.round(results.intent_accuracy * 100);
  const categoryErrorEvery =
    results.category_accuracy < 1
      ? Math.round(1 / (1 - results.category_accuracy))
      : null;
  const weakestF1 =
    perIntent.length > 0
      ? Math.min(...perIntent.map((row) => row.f1))
      : null;

  return (
    <>
      <div className="mt-6 grid gap-4 lg:grid-cols-3">
        <ResultStat
          icon={Target}
          tone="a1"
          label="Intent Macro-F1"
          value={formatPercent(results.intent_macro_f1, 2)}
          sub={`95% CI ${formatPercent(
            results.intent_macro_f1_ci[0],
            2
          )}–${formatPercent(results.intent_macro_f1_ci[1], 2)}`}
          caption={`About ${correctIn100} in 100 held-out questions are routed to the right topic.`}
          hint="Macro-F1 averages every intent equally, so rare topics count as much as common ones."
        />
        <ResultStat
          icon={Layers}
          tone="brand"
          label="Category accuracy"
          value={formatPercent(results.category_accuracy, 2)}
          sub={`${formatCount(results.n_test)} held-out questions`}
          caption={
            categoryErrorEvery
              ? `The broader support category is wrong about once in ${formatCount(
                  categoryErrorEvery
                )} questions.`
              : "The broader support category is right on almost every question."
          }
          hint="A1 predicts a fine-grained intent; each intent maps to a broader category. This measures the group, not the fine topic."
        />
        <ResultStat
          icon={ShieldCheck}
          tone="a2"
          label="Answer relevance"
          value={formatScore(relevance.cosine_sim, 3)}
          sub={`cosine · 95% CI ${formatScore(
            relevance.cosine_sim_ci[0],
            3
          )}–${formatScore(relevance.cosine_sim_ci[1], 3)}`}
          caption={`A2's reply is close to the held-out reference reply. ${formatPercent(
            checks.pass_rate,
            1
          )} pass all three checks.`}
          hint="Relevance is cosine similarity between the returned answer and the reference answer. The pass rate is separate: it counts replies that clear all three self-checks."
          footer={
            <div className="mt-2 flex flex-wrap items-center gap-1.5">
              <Chip tone="success">
                Self-check {formatPercent(checks.pass_rate, 1)}
              </Chip>
              <span className="text-xs text-ink-secondary">
                ROUGE-L {formatScore(relevance.rouge_l_f1, 3)}
              </span>
            </div>
          }
        />
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-2">
        <Panel
          icon={Layers}
          title="F1 per intent"
          caption="Worst first by default. Click a column heading to sort."
          hint={
            weakestF1 !== null
              ? `F1 balances precision and recall for one intent. The weakest intent still scores ${formatPercent(
                  weakestF1,
                  1
                )}.`
              : "F1 balances precision and recall for one intent."
          }
        >
          {perIntent.length > 0 ? (
            <div className="mt-3">
              <F1Table metrics={perIntent} />
            </div>
          ) : (
            <EmptyDataFile file="per_intent_f1.json" className="mt-3" />
          )}
        </Panel>

        <Panel
          icon={Layers}
          title="Confusion by category"
          caption="Rows are true; columns are predicted. Hover a cell for the count."
          hint="A clean diagonal means categories are rarely confused. Off-diagonal cells show which groups get mixed up."
        >
          {confusion ? (
            <div className="mt-3">
              <ConfusionHeatmap data={confusion} />
            </div>
          ) : (
            <EmptyDataFile file="confusion_category.json" className="mt-3" />
          )}
        </Panel>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Panel
          icon={Gauge}
          title="Confidence vs accuracy"
          caption="Stated confidence against what actually happened."
          hint="A calibrated model that says 90% should be right about 90% of the time. The dashed line is perfect calibration."
        >
          <div className="mt-2">
            <CalibrationChart
              bins={results.calibration.bins_after}
              tau={results.calibration.tau}
            />
          </div>
          <p className="mt-2 border-t border-line pt-2.5 text-xs leading-[18px] text-ink-secondary">
            Expected calibration error is{" "}
            <span className="font-mono text-ink">
              {formatScore(results.calibration.ece_before, 4)}
            </span>{" "}
            before temperature scaling and{" "}
            <span className="font-mono text-ink">
              {formatScore(results.calibration.ece_after, 4)}
            </span>{" "}
            after. Lower is better.
          </p>
        </Panel>

        <Panel
          icon={BarChart3}
          title="Model comparison"
          caption="Macro-F1 on the same held-out split. Bars show 95% CIs."
          hint="The ladder shows what each modelling step adds. L2 is served because it is calibrated and shares the embedding space used for retrieval."
        >
          {ladder.length > 0 ? (
            <>
              <div className="mt-2">
                <LadderChart rows={ladder} />
              </div>
              <div className="mt-2 flex flex-wrap items-center gap-2 border-t border-line pt-2.5 text-xs text-ink-secondary">
                <Chip tone="brand">
                  L2 served · {formatPercent(results.intent_macro_f1, 2)}
                </Chip>
                <span>
                  L1 is a strong baseline but does not share A2&apos;s embedding
                  space.
                </span>
              </div>
            </>
          ) : (
            <EmptyDataFile file="ladder.json" className="mt-3" />
          )}
        </Panel>
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-[1.15fr_0.85fr]">
        <Panel
          icon={Layers}
          title="Answer finding"
          caption="Sparse, dense and hybrid retrieval, with and without A1 routing."
          hint="A1 routing filters A2 to the predicted intent. Global retrieval searches every past case. The comparison is shown as measured, including where global search is competitive."
        >
          {retrieval ? (
            <>
              <div className="mt-2">
                <RetrievalChart data={retrieval} />
              </div>
              <div className="mt-2 overflow-x-auto rounded-[10px] border border-line">
                <table className="w-full min-w-[400px] border-collapse text-left text-xs">
                  <thead className="bg-soft">
                    <tr>
                      <th className="px-2.5 py-1.5 font-medium uppercase tracking-[0.08em] text-ink-secondary">
                        Mode
                      </th>
                      <th className="px-2.5 py-1.5 text-right font-medium uppercase tracking-[0.08em] text-ink-secondary">
                        Relevance
                      </th>
                      <th className="px-2.5 py-1.5 text-right font-medium uppercase tracking-[0.08em] text-ink-secondary">
                        Intent consistency
                      </th>
                      <th className="px-2.5 py-1.5 text-right font-medium uppercase tracking-[0.08em] text-ink-secondary">
                        Self-check
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {retrieval.modes.map((mode) => (
                      <tr
                        key={mode.key}
                        className="border-t border-line last:border-b-0"
                      >
                        <td className="whitespace-nowrap px-2.5 py-1.5">
                          <span className="font-medium text-ink">
                            {mode.mode}
                          </span>
                          {mode.served ? (
                            <span className="ml-1.5 rounded-pill bg-brand/[0.08] px-1.5 py-0.5 text-[10px] font-medium text-brand-ink">
                              served
                            </span>
                          ) : null}
                        </td>
                        <td className="px-2.5 py-1.5 text-right font-mono tabular-nums text-ink-secondary">
                          {formatScore(mode.cosine_sim, 3)}
                        </td>
                        <td className="px-2.5 py-1.5 text-right font-mono tabular-nums text-ink-secondary">
                          {formatPercent(mode.intent_consistency, 2)}
                        </td>
                        <td className="px-2.5 py-1.5 text-right font-mono tabular-nums text-ink-secondary">
                          {formatPercent(mode.check_passed_rate, 1)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="mt-2 text-xs leading-[18px] text-ink-secondary">
                Routing narrows the search to A1&apos;s predicted topic and wins
                on sparse retrieval. Dense global search is a strong baseline
                here because A1 is already {formatPercent(results.intent_accuracy, 2)}{" "}
                accurate. The served pipeline keeps routing so every reply has
                the topic shown in its trace.
              </p>
            </>
          ) : (
            <EmptyDataFile file="retrieval_modes.json" className="mt-3" />
          )}
        </Panel>

        <Panel
          icon={CheckCircle2}
          title="Answer checks"
          caption="What happens when A2 checks its own candidate answer."
          hint="A2 runs three checks before returning a reply. If a candidate fails, it tries the next one; if none pass, it returns the best available match and says so in the trace."
        >
          <div className="mt-3 flex items-end justify-between gap-3 rounded-[12px] border border-line bg-soft px-3 py-2.5">
            <div>
              <p className="text-xs font-medium uppercase tracking-[0.08em] text-ink-secondary">
                All three checks
              </p>
              <p className="mt-1 text-xl font-semibold tabular-nums text-ink">
                {formatPercent(checks.pass_rate, 2)}
              </p>
            </div>
            <span className="text-right font-mono text-xs text-ink-secondary">
              95% CI {formatPercent(checks.pass_rate_ci[0], 1)}–
              {formatPercent(checks.pass_rate_ci[1], 1)}
            </span>
          </div>

          <ul className="mt-3 space-y-2.5">
            {checks.checks.map((check) => (
              <li key={check.key}>
                <div className="flex items-center justify-between gap-2 text-xs">
                  <span className="text-ink-secondary">{check.label}</span>
                  <span className="font-mono tabular-nums text-ink">
                    {formatPercent(check.fail_rate, 1)} fail
                  </span>
                </div>
                <div className="mt-1 h-1.5 overflow-hidden rounded-pill bg-[#F0F0ED]">
                  <div
                    className={cn(
                      "h-full rounded-pill",
                      check.fail_rate > 0.1
                        ? "bg-warning/70"
                        : check.fail_rate > 0.02
                          ? "bg-brand/60"
                          : "bg-success/70"
                    )}
                    style={{ width: `${Math.max(check.fail_rate * 100, 1)}%` }}
                  />
                </div>
              </li>
            ))}
          </ul>

          <p className="mt-3 border-t border-line pt-2.5 text-xs leading-[18px] text-ink-secondary">
            {formatPercent(checks.rejected_share, 1)} of questions had at least
            one candidate rejected. Many dataset answers intentionally keep
            customer-fill placeholders, so a failed placeholder check does not
            automatically make the answer wrong.
          </p>
        </Panel>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-[1fr_0.8fr]">
        <Panel
          icon={BarChart3}
          title="Robustness by style"
          caption="Macro-F1 on test questions with each style flag."
          hint="The same model is measured on colloquial, noisy and polite wording. Morphological variants are the hardest group."
        >
          <ul className="mt-3 space-y-2.5">
            {results.robustness.map((row) => (
              <li key={row.style}>
                <div className="flex items-center justify-between gap-2 text-xs">
                  <span className="text-ink-secondary">{row.style}</span>
                  <span className="font-mono tabular-nums text-ink">
                    {formatPercent(row.macro_f1, 2)}
                    <span className="ml-1.5 text-ink-faint">
                      n={formatCount(row.support)}
                    </span>
                  </span>
                </div>
                <div className="mt-1 h-1.5 overflow-hidden rounded-pill bg-[#F0F0ED]">
                  <div
                    className={cn(
                      "h-full rounded-pill",
                      row.macro_f1 >= 0.99
                        ? "bg-success/70"
                        : row.macro_f1 >= 0.95
                          ? "bg-brand/70"
                          : "bg-warning/70"
                    )}
                    style={{ width: `${Math.max(row.macro_f1 * 100, 2)}%` }}
                  />
                </div>
              </li>
            ))}
          </ul>
          <p className="mt-3 border-t border-line pt-2.5 text-xs leading-[18px] text-ink-secondary">
            Style is assigned once per example from the dataset&apos;s raw flag
            letters. Morphological variants are still the weakest group.
          </p>
        </Panel>

        <Panel
          icon={ShieldCheck}
          title="Out-of-scope questions"
          caption="Can the system tell when a question is outside support?"
          hint="OOD AUROC measures separation between in-domain and out-of-scope questions. The flag is the same s_min line used in the live trace."
        >
          {ood ? (
            <div className="mt-3 space-y-3">
              <div className="grid grid-cols-2 gap-2">
                <MiniStat
                  label="AUROC · max prob"
                  value={formatScore(ood.auroc_maxprob, 3)}
                />
                <MiniStat
                  label="AUROC · source sim"
                  value={formatScore(ood.auroc_top1sim, 3)}
                />
                <MiniStat
                  label="OOD flagged"
                  value={formatPercent(ood.flagged_pct_ood, 1)}
                />
                <MiniStat
                  label="In-domain flagged"
                  value={formatPercent(ood.flagged_pct_in_domain, 1)}
                />
              </div>
              <p className="text-xs leading-[18px] text-ink-secondary">
                On {ood.n_ood} out-of-scope questions,{" "}
                {formatPercent(ood.flagged_pct_ood, 1)} fall below the source
                line (s_min ={" "}
                <span className="font-mono">
                  {formatScore(ood.threshold_s_min, 3)}
                </span>
                ). Only {formatPercent(ood.flagged_pct_in_domain, 1)} of
                in-domain questions are flagged.
              </p>
            </div>
          ) : (
            <EmptyDataFile file="ood.json" className="mt-3" />
          )}
        </Panel>
      </div>

      <details className="group mt-4 rounded-card border border-line bg-card">
        <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-4 py-3.5 [&::-webkit-details-marker]:hidden">
          <div>
            <h2 className="text-base font-semibold text-ink">
              Known limitations
            </h2>
            <p className="mt-0.5 text-xs text-ink-secondary">
              Where the numbers stop being flattering.
            </p>
          </div>
          <ChevronDown
            size={16}
            aria-hidden="true"
            className="shrink-0 text-ink-faint transition-transform group-open:rotate-180"
          />
        </summary>
        <ul className="space-y-2.5 border-t border-line px-4 py-3.5">
          {[
            "Answers are retrieved from past support replies, not generated from scratch. They can inherit the source reply's tone and gaps.",
            "The self-check is strict. A useful reply can still fail the placeholder check because the dataset intentionally keeps customer-fill slots.",
            "Retrieval scores are backend-specific (RRF for hybrid, cosine for dense). They are not a 0–100 confidence and should not be compared across backends.",
            "A1 is measured on the intents in this support dataset. A new topic needs new labelled examples before it can be routed reliably.",
          ].map((item) => (
            <li key={item} className="flex gap-2 text-xs leading-[18px] text-ink-secondary">
              <span
                aria-hidden="true"
                className="mt-1.5 size-1.5 shrink-0 rounded-full bg-ink-faint"
              />
              {item}
            </li>
          ))}
        </ul>
      </details>
    </>
  );
}

function Panel({
  icon: Icon,
  title,
  caption,
  hint,
  children,
  className,
}: {
  icon: LucideIcon;
  title: string;
  caption: string;
  hint: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <Card className={cn("p-4 sm:p-5", className)}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-1.5">
            <Icon
              size={15}
              strokeWidth={1.9}
              aria-hidden="true"
              className="text-ink-faint"
            />
            <h2 className="text-base font-semibold text-ink">{title}</h2>
            <InfoHint text={hint} />
          </div>
          <p className="mt-0.5 text-xs leading-[18px] text-ink-secondary">
            {caption}
          </p>
        </div>
      </div>
      {children}
    </Card>
  );
}

function ResultStat({
  icon: Icon,
  tone,
  label,
  value,
  sub,
  caption,
  hint,
  footer,
}: {
  icon: React.ComponentType<any>;
  tone: "a1" | "a2" | "brand";
  label: string;
  value: string;
  sub: string;
  caption: string;
  hint: string;
  footer?: React.ReactNode;
}) {
  const toneClass =
    tone === "a1"
      ? "bg-a1/[0.08] text-a1-ink"
      : tone === "a2"
        ? "bg-a2/[0.08] text-a2-ink"
        : "bg-brand/[0.08] text-brand-ink";

  return (
    <Card className="p-5">
      <div className="flex items-start justify-between gap-3">
        <p className="text-xs font-medium uppercase tracking-[0.12em] text-ink-secondary">
          {label}
        </p>
        <span
          className={cn(
            "flex size-8 shrink-0 items-center justify-center rounded-[10px]",
            toneClass
          )}
        >
          <Icon size={16} strokeWidth={1.8} aria-hidden="true" />
        </span>
      </div>
      <p className="mt-3 text-[32px] font-semibold leading-none tracking-tight text-ink">
        {value}
      </p>
      <p className="mt-1.5 font-mono text-xs text-ink-secondary">{sub}</p>
      <p className="mt-2.5 text-xs leading-[18px] text-ink-secondary">
        {caption}
      </p>
      {footer}
      <div className="mt-2 flex items-center gap-1 text-[11px] text-ink-faint">
        <InfoHint text={hint} />
        What this means
      </div>
    </Card>
  );
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[10px] border border-line bg-soft px-2.5 py-2">
      <p className="text-[11px] font-medium uppercase tracking-[0.06em] text-ink-secondary">
        {label}
      </p>
      <p className="mt-1 font-mono text-base font-semibold tabular-nums text-ink">
        {value}
      </p>
    </div>
  );
}

function ChartSkeleton() {
  return (
    <div className="mt-2 h-[240px] animate-pulse rounded-[12px] border border-line bg-soft" />
  );
}

