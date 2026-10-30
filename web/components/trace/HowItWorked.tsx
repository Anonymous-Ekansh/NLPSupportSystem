"use client";

import { useEffect, useState, type ReactNode } from "react";
import {
  ArrowRight,
  ChevronRight,
  Clock3,
  Code2,
  ListTree,
  RotateCcw,
} from "lucide-react";
import {
  getA1Output,
  getA2Output,
  getTraceStep,
  isOutOfScope,
  type MapPoint,
  type Meta,
  type ResolveResult,
} from "@/lib/api";
import {
  cn,
  formatLatency,
  formatPercent,
  formatScore,
  humaniseCategory,
  humaniseIntent,
  topicLabel,
} from "@/lib/utils";
import { Card } from "@/components/ui/Card";
import { Chip, Dot } from "@/components/ui/Chip";
import { ConfidenceBar } from "@/components/ui/ConfidenceBar";
import { Skeleton } from "@/components/ui/Skeleton";
import { SegmentedToggle } from "@/components/ui/SegmentedToggle";
import { TraceNode } from "@/components/trace/TraceNode";
import { CheckRow } from "@/components/trace/CheckRow";
import { ScatterMap } from "@/components/trace/ScatterMap";
import { PlaceholderText } from "@/components/chat/PlaceholderText";
import { SessionCard } from "@/components/demo/SessionCard";

type TraceView = "simple" | "technical";

interface HowItWorkedProps {
  result: ResolveResult | null;
  meta: Meta | null;
  mapPoints: MapPoint[];
  session: {
    questions: number;
    averageLatency: number | null;
    helpfulVotes: number;
  };
  loading?: boolean;
  variant?: "panel" | "sheet";
  className?: string;
}

export function HowItWorked({
  result,
  meta,
  mapPoints,
  session,
  loading = false,
  variant = "panel",
  className,
}: HowItWorkedProps) {
  const [view, setView] = useState<TraceView>("simple");
  const [visible, setVisible] = useState(1);
  const [replayKey, setReplayKey] = useState(0);

  useEffect(() => {
    if (!result || loading) {
      setVisible(0);
      return;
    }

    const reduce = window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    ).matches;
    if (reduce) {
      setVisible(5);
      return;
    }

    setVisible(1);
    let current = 1;
    const timer = window.setInterval(() => {
      current += 1;
      setVisible(current);
      if (current >= 5) window.clearInterval(timer);
    }, 200);

    return () => window.clearInterval(timer);
  }, [result, loading, replayKey]);

  const a1 = result ? getA1Output(result) : null;
  const a2 = result ? getA2Output(result) : null;
  const a1Step = result ? getTraceStep(result, "IntentAnalyst") : null;
  const a2Step = result ? getTraceStep(result, "ResponseSpecialist") : null;
  const tau = meta?.a1?.tau ?? null;
  const sMin = meta?.a1?.s_min ?? null;
  const withheld = result ? isOutOfScope(result, sMin) : false;
  const selfCheck = a2?.self_check;
  const sourceSimilarity =
    selfCheck?.source_similarity ?? a2?.retrieval_score ?? null;
  const expectedIntent = a1?.intent ?? a2?.used_intents?.[0] ?? null;
  const supportingCount = a2?.supporting_examples?.length ?? 0;
  const bestSimilarity = a2?.supporting_examples?.[0]?.similarity;
  const show = (step: number) => visible >= step;

  const header = (
    <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
      {variant === "panel" ? (
        <div className="flex items-center gap-1.5">
          <ListTree
            size={16}
            strokeWidth={1.9}
            aria-hidden="true"
            className="text-ink-faint"
          />
          <h2 className="text-base font-semibold text-ink">How it worked</h2>
        </div>
      ) : (
        <span className="text-xs font-medium uppercase tracking-[0.12em] text-ink-secondary">
          Live trace
        </span>
      )}

      <div className="ml-auto flex items-center gap-2">
        <SegmentedToggle
          value={view}
          onChange={setView}
          label="Trace detail"
          options={[
            { value: "simple", label: "Simple" },
            { value: "technical", label: "Technical" },
          ]}
        />
        <button
          type="button"
          disabled={!result || loading}
          onClick={() => setReplayKey((key) => key + 1)}
          className="inline-flex items-center gap-1 rounded-pill border border-line bg-card px-2.5 py-1 text-xs font-medium text-ink-secondary transition-colors hover:text-ink disabled:cursor-not-allowed disabled:opacity-40"
        >
          <RotateCcw size={12} aria-hidden="true" />
          Replay
        </button>
      </div>
    </div>
  );

  if (loading) {
    const body = (
      <>
        {header}
        <div className="space-y-4" aria-label="Loading trace">
          {[0, 1, 2, 3, 4].map((index) => (
            <div key={index} className="flex gap-3">
              <Skeleton className="size-[26px] shrink-0 rounded-full" />
              <div className="min-w-0 flex-1 space-y-2">
                <Skeleton className="h-3 w-32" />
                <Skeleton className={index === 3 ? "h-20 w-full" : "h-8 w-full"} />
              </div>
            </div>
          ))}
        </div>
      </>
    );
    return variant === "panel" ? (
      <Card className={cn("p-4", className)}>{body}</Card>
    ) : (
      <div className={className}>{body}</div>
    );
  }

  if (!result) {
    const empty = (
      <>
        {header}
        <div className="rounded-[12px] border border-dashed border-line bg-soft px-4 py-8 text-center">
          <span className="mx-auto flex size-9 items-center justify-center rounded-full border border-line bg-card text-ink-faint">
            <ListTree size={16} aria-hidden="true" />
          </span>
          <p className="mt-2.5 text-sm font-medium text-ink">
            Your answer&apos;s work will appear here.
          </p>
          <p className="mx-auto mt-1 max-w-[250px] text-xs leading-[18px] text-ink-secondary">
            A1 reads the question, then A2 finds and checks the answer.
          </p>
          <div className="mt-3 flex items-center justify-center gap-1.5">
            <Chip tone="a1">
              <Dot tone="a1" />
              A1 Intent Analyst
            </Chip>
            <ArrowRight size={12} aria-hidden="true" className="text-ink-faint" />
            <Chip tone="a2">
              <Dot tone="a2" />
              A2 Response Specialist
            </Chip>
          </div>
        </div>
      </>
    );

    return variant === "panel" ? (
      <div className={cn("space-y-4", className)}>
        <Card className="p-4">{empty}</Card>
        <ScatterSection
          points={mapPoints}
          query={null}
          session={session}
        />
      </div>
    ) : (
      <div className={className}>{empty}</div>
    );
  }

  const simpleConfidence = a1 ? formatPercent(a1.confidence) : "—";
  const confidenceAbove =
    a1 && typeof tau === "number" ? a1.confidence >= tau : null;
  const sourceAbove =
    a1 &&
    typeof a1.top1_similarity === "number" &&
    typeof sMin === "number"
      ? a1.top1_similarity >= sMin
      : null;

  const stepNodes = [
    <TraceNode key="question" step={1} title="Your question" tone="neutral">
        <p className="text-sm leading-[21px] text-ink-secondary">
          “{result.query}”
        </p>
      </TraceNode>,

      <TraceNode
        step={2}
        title="A1 Intent Analyst"
        tone="a1"
        badge={
          <Chip tone="a1">
            <Dot tone="a1" />
            A1
          </Chip>
        }
      >
        {view === "simple" ? (
          <p className="text-sm leading-[21px] text-ink-secondary">
            Understood this as{" "}
            <Chip tone="a1" className="align-middle">
              {a1 ? topicLabel(a1.category, a1.intent) : "an unknown topic"}
            </Chip>{" "}
            with <span className="font-medium text-ink">{simpleConfidence}</span>{" "}
            confidence.
          </p>
        ) : a1 ? (
          <div className="space-y-3.5">
            <div className="grid grid-cols-2 gap-3">
              <Field label="predicted intent">
                <span className="font-mono text-xs text-ink">{a1.intent}</span>
              </Field>
              <Field label="category">
                <Chip tone="a1">{humaniseCategory(a1.category)}</Chip>
              </Field>
            </div>

            <div>
              <div className="mb-1.5 flex items-center justify-between gap-3">
                <span className="text-xs font-medium uppercase tracking-[0.1em] text-ink-secondary">
                  Confidence
                </span>
                <span className="font-mono text-xs tabular-nums text-ink">
                  {formatPercent(a1.confidence, 1)}
                </span>
              </div>
              <ConfidenceBar value={a1.confidence} threshold={tau} tone="a1" />
              <div className="mt-1.5 flex items-center justify-between gap-2">
                <p className="text-xs leading-[18px] text-ink-secondary">
                  How sure A1 is. Tick = τ {tau === null ? "—" : formatScore(tau, 3)}.
                </p>
                {confidenceAbove !== null ? (
                  <Chip tone={confidenceAbove ? "success" : "warning"}>
                    {confidenceAbove ? "Above" : "Below"} τ
                  </Chip>
                ) : null}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <Field label="margin" caption="Gap to the next-best intent.">
                <span className="font-mono text-xs text-ink">
                  {formatScore(a1.margin, 3)}
                </span>
              </Field>
              <Field label="latency" caption="A1 thinking time.">
                <span className="font-mono text-xs text-ink">
                  {formatLatency(a1Step?.latency_ms)}
                </span>
              </Field>
            </div>

            <div className="rounded-[12px] border border-line bg-soft p-2.5">
              <div className="mb-1.5 flex items-center justify-between gap-2">
                <span className="text-xs font-medium uppercase tracking-[0.1em] text-ink-secondary">
                  Source match
                </span>
                {sourceAbove !== null ? (
                  <Chip tone={sourceAbove ? "success" : "warning"}>
                    {sourceAbove ? "Above" : "Below"} s_min
                  </Chip>
                ) : null}
              </div>
              <ConfidenceBar
                value={a1.top1_similarity ?? 0}
                threshold={sMin}
                tone={sourceAbove === false ? "warning" : "a1"}
                showValue
              />
              <p className="mt-1.5 text-xs leading-[18px] text-ink-secondary">
                Closest past question. s_min is{" "}
                {sMin === null ? "unavailable" : formatScore(sMin, 3)}.
              </p>
            </div>

            <div>
              <p className="text-xs font-medium uppercase tracking-[0.1em] text-ink-secondary">
                Top 3 intents
              </p>
              <ul className="mt-1.5 space-y-1.5">
                {a1.top3.map((item, index) => (
                  <li key={item.intent}>
                    <div className="flex items-center justify-between gap-2">
                      <span
                        className={cn(
                          "truncate text-xs",
                          index === 0
                            ? "font-medium text-ink"
                            : "text-ink-secondary"
                        )}
                      >
                        {humaniseIntent(item.intent)}{" "}
                        <span className="font-mono text-[12px] text-ink-faint">
                          {item.intent}
                        </span>
                      </span>
                      <span className="shrink-0 font-mono text-xs tabular-nums text-ink-secondary">
                        {formatPercent(item.prob, 1)}
                      </span>
                    </div>
                    <div className="mt-1 h-1 overflow-hidden rounded-pill bg-[#F0F0ED]">
                      <div
                        className={cn(
                          "h-full rounded-pill",
                          index === 0 ? "bg-a1" : "bg-ink-faint/50"
                        )}
                        style={{ width: `${Math.max(item.prob * 100, 1)}%` }}
                      />
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        ) : (
          <p className="text-xs text-ink-secondary">
            A1 did not return an intent for this question.
          </p>
        )}
      </TraceNode>,

      <TraceNode step={3} title="Handoff" tone="neutral">
        <div className="flex flex-wrap items-center gap-1.5">
          <Chip tone="a1">
            <Dot tone="a1" />
            A1
          </Chip>
          <ArrowRight size={13} aria-hidden="true" className="text-ink-faint" />
          <Chip tone="a2">
            <Dot tone="a2" />
            A2
          </Chip>
          <span className="ml-1 text-sm text-ink-secondary">
            {view === "simple"
              ? "Passed the topic to the answer finder."
              : "Structured handoff to retrieval."}
          </span>
        </div>

        {view === "technical" && a1 ? (
          <div className="mt-3">
            <div className="grid grid-cols-2 gap-3">
              <Field label="intent">
                <span className="font-mono text-xs text-ink">{a1.intent}</span>
              </Field>
              <Field label="category">
                <span className="font-mono text-xs text-ink">{a1.category}</span>
              </Field>
              <Field label="confidence">
                <span className="font-mono text-xs text-ink">
                  {a1.confidence.toFixed(4)}
                </span>
              </Field>
              <Field label="needs_clarification">
                <Chip tone={a1.needs_clarification ? "warning" : "neutral"}>
                  {String(a1.needs_clarification)}
                </Chip>
              </Field>
              <div className="col-span-2">
                <Field label="top3">
                  <div className="flex flex-wrap gap-1.5">
                    {a1.top3.map((item) => (
                      <span
                        key={item.intent}
                        className="rounded-[8px] border border-line bg-soft px-2 py-0.5 font-mono text-xs text-ink-secondary"
                      >
                        {item.intent} · {item.prob.toFixed(4)}
                      </span>
                    ))}
                  </div>
                </Field>
              </div>
            </div>

            <details className="group mt-2">
              <summary className="flex cursor-pointer list-none items-center gap-1 text-xs font-medium text-ink-secondary hover:text-ink [&::-webkit-details-marker]:hidden">
                <Code2 size={12} aria-hidden="true" />
                View handoff JSON
                <ChevronRight
                  size={12}
                  aria-hidden="true"
                  className="transition-transform group-open:rotate-90"
                />
              </summary>
              <pre className="mt-2 max-h-56 overflow-auto rounded-[10px] border border-line bg-soft p-2.5 font-mono text-xs leading-[18px] text-ink-secondary">
                {JSON.stringify(
                  {
                    intent: a1.intent,
                    category: a1.category,
                    confidence: a1.confidence,
                    needs_clarification: a1.needs_clarification,
                    top3: a1.top3,
                  },
                  null,
                  2
                )}
              </pre>
            </details>
          </div>
        ) : null}
      </TraceNode>,

      <TraceNode
        step={4}
        title="A2 Response Specialist"
        tone="a2"
        badge={
          <Chip tone="a2">
            <Dot tone="a2" />
            A2
          </Chip>
        }
      >
        {view === "simple" ? (
          <div>
            <p className="text-sm leading-[21px] text-ink-secondary">
              Searched {supportingCount} past support{" "}
              {supportingCount === 1 ? "case" : "cases"} about this topic and
              picked the best match, then double-checked it.
            </p>
            <div className="mt-2 flex flex-wrap items-center gap-1.5">
              <Chip
                tone={selfCheck?.check_passed ? "success" : "error"}
              >
                Self-check {selfCheck?.check_passed ? "passed" : "failed"}
              </Chip>
              {typeof bestSimilarity === "number" ? (
                <span className="text-xs text-ink-secondary">
                  Closest case{" "}
                  <span className="font-mono text-ink">
                    {formatPercent(bestSimilarity, 1)}
                  </span>{" "}
                  similar
                </span>
              ) : null}
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <Field
                label="retrieval_backend"
                caption="Where past cases were searched."
              >
                <span className="font-mono text-xs text-ink">
                  {result.retrieval_backend}
                </span>
              </Field>
              <Field
                label="retrieval_score"
                caption="Closeness of the best past case."
              >
                <span className="font-mono text-xs text-ink">
                  {formatScore(a2?.retrieval_score, 4)}
                </span>
              </Field>
            </div>

            <div>
              <p className="text-xs font-medium uppercase tracking-[0.1em] text-ink-secondary">
                Retrieval score
              </p>
              <ConfidenceBar
                value={a2?.retrieval_score ?? 0}
                tone="a2"
                showValue
                className="mt-1.5"
              />
            </div>

            <div>
              <p className="text-xs font-medium uppercase tracking-[0.1em] text-ink-secondary">
                Intents searched
              </p>
              <div className="mt-1.5 flex flex-wrap gap-1.5">
                {(a2?.used_intents?.length
                  ? a2.used_intents
                  : a1
                    ? [a1.intent]
                    : []
                ).map((intent) => (
                  <Chip key={intent} tone="a2">
                    {intent}
                  </Chip>
                ))}
              </div>
            </div>

            <div className="rounded-[12px] border border-line bg-soft p-3">
              <div className="mb-1 flex items-center justify-between gap-2">
                <p className="text-xs font-medium uppercase tracking-[0.1em] text-ink-secondary">
                  Self-check
                </p>
                <Chip tone={selfCheck?.check_passed ? "success" : "error"}>
                  {selfCheck?.check_passed ? "Passed" : "Failed"}
                </Chip>
              </div>
              <CheckRow
                label="Placeholders resolved"
                detail={
                  selfCheck?.placeholder_safe
                    ? "No unresolved {{...}} tokens."
                    : "Customer-fill slots remain, shown as chips in the answer."
                }
                passed={Boolean(selfCheck?.placeholder_safe)}
              />
              <CheckRow
                label="Source similarity"
                detail={`Closest past case scored ${
                  sourceSimilarity === null
                    ? "—"
                    : formatScore(sourceSimilarity, 3)
                }.`}
                passed={Boolean(selfCheck?.source_similarity_ok)}
              />
              <CheckRow
                label="Response matches intent"
                detail={`Expected ${expectedIntent ?? "—"}${
                  selfCheck?.response_predicted_intent
                    ? ` · predicted ${selfCheck.response_predicted_intent}`
                    : ""
                }.`}
                passed={Boolean(selfCheck?.response_intent_ok)}
              />
              <div className="flex items-center justify-between gap-2 border-t border-line pt-2">
                <p className="text-xs text-ink-secondary">
                  Rejected candidates
                  <span className="ml-1 font-mono text-ink">
                    {selfCheck?.candidates_rejected ?? 0}
                  </span>
                </p>
                <span className="text-xs text-ink-faint">
                  Failed a check before this answer
                </span>
              </div>
            </div>

            <p className="text-xs leading-[18px] text-ink-secondary">
              {selfCheck?.check_passed
                ? "All three checks passed. This is the grounded answer."
                : "Not every check passed. A2 returned the best available match, shown honestly here."}
            </p>
          </div>
        )}
      </TraceNode>,

      <TraceNode
        step={5}
        title="Answer"
        tone={selfCheck?.check_passed ? "success" : "warning"}
        isLast
      >
        {withheld ? (
          <div className="rounded-[12px] border border-warning/20 bg-warning-soft p-3">
            <p className="text-xs font-medium text-warning-ink">
              Answer withheld from the chat
            </p>
            <p className="mt-1 text-xs leading-[18px] text-ink-secondary">
              The closest source match was below the s_min line, so the customer
              sees the support-scope message instead.
            </p>
            <details className="group mt-2">
              <summary className="flex cursor-pointer list-none items-center gap-1 text-xs font-medium text-ink-secondary hover:text-ink [&::-webkit-details-marker]:hidden">
                <Code2 size={12} aria-hidden="true" />
                View retrieved answer
                <ChevronRight
                  size={12}
                  aria-hidden="true"
                  className="transition-transform group-open:rotate-90"
                />
              </summary>
              <div className="mt-2 rounded-[10px] border border-line bg-card p-2.5 text-xs leading-[18px] text-ink-secondary">
                <PlaceholderText text={result.final_response || "No answer text returned."} />
              </div>
            </details>
          </div>
        ) : (
          <div className="rounded-[12px] border border-line bg-soft p-3 text-sm leading-[21px] text-ink">
            <PlaceholderText
              text={result.final_response || "No answer text returned."}
            />
          </div>
        )}

        <div className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1">
          <Clock3 size={12} aria-hidden="true" className="text-ink-faint" />
          <span className="font-mono text-xs text-ink">
            {formatLatency(result.total_latency_ms)}
          </span>
          <span className="text-xs text-ink-secondary">
            total · question to checked answer
          </span>
        </div>
      </TraceNode>,
  ];

  const body = (
    <>
      {header}
      <div aria-live="polite">
        {stepNodes.map((node, index) =>
          show(index + 1) ? (
            <div key={index} className="animate-fade-up">
              {node}
            </div>
          ) : null
        )}
      </div>

      <details className="group mt-4 border-t border-line pt-3">
        <summary className="flex cursor-pointer list-none items-center gap-1.5 text-xs font-medium text-ink-secondary hover:text-ink [&::-webkit-details-marker]:hidden">
          <Code2 size={13} aria-hidden="true" />
          Raw JSON
          <ChevronRight
            size={13}
            aria-hidden="true"
            className="transition-transform group-open:rotate-90"
          />
        </summary>
        <pre className="mt-2 max-h-72 overflow-auto rounded-[12px] border border-line bg-soft p-3 font-mono text-xs leading-[18px] text-ink-secondary">
          {JSON.stringify(result, null, 2)}
        </pre>
      </details>

      <ScatterSection
        points={mapPoints}
        query={result.map}
        session={session}
      />
    </>
  );

  return variant === "panel" ? (
    <Card className={cn("p-4", className)}>{body}</Card>
  ) : (
    <div className={className}>{body}</div>
  );
}

function Field({
  label,
  children,
  caption,
}: {
  label: string;
  children: ReactNode;
  caption?: string;
}) {
  return (
    <div className="min-w-0">
      <p className="font-mono text-xs text-ink-faint">{label}</p>
      <div className="mt-1">{children}</div>
      {caption ? (
        <p className="mt-1 text-xs leading-[18px] text-ink-secondary">
          {caption}
        </p>
      ) : null}
    </div>
  );
}

function ScatterSection({
  points,
  query,
  session,
}: {
  points: MapPoint[];
  query: { x: number | null; y: number | null } | null;
  session: {
    questions: number;
    averageLatency: number | null;
    helpfulVotes: number;
  };
}) {
  return (
    <div className="mt-4 space-y-4 border-t border-line pt-4">
      <section>
        <div className="flex items-center justify-between gap-2">
          <h3 className="text-sm font-semibold text-ink">
            Where your question sits
          </h3>
          <span className="text-xs text-ink-faint">2D semantic map</span>
        </div>
        <p className="mt-0.5 text-xs leading-[18px] text-ink-secondary">
          A projection of past support questions, coloured by category.
        </p>
        <div className="mt-2">
          <ScatterMap
            points={points}
            query={query}
            height={190}
            showLegend
            caption="Closer dots = similar past questions."
          />
        </div>
      </section>

      <SessionCard
        questions={session.questions}
        averageLatency={session.averageLatency}
        helpfulVotes={session.helpfulVotes}
      />
    </div>
  );
}
