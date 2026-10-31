"use client";

import { useState } from "react";
import { ChevronRight, Clock3, ShieldCheck } from "lucide-react";
import {
  getA1Output,
  getAudit,
  getRoutePlan,
  type AuditResult,
  type ResolveResult,
  type RoutePlan,
  type TraceStep,
} from "@/lib/api";
import {
  cn,
  formatLatency,
  formatPercent,
  formatScore,
  humaniseIntent,
} from "@/lib/utils";
import { Chip } from "@/components/ui/Chip";

interface AgentMeta {
  code: string;
  title: string;
  role: string;
  accent: string;
  dot: string;
}

const META: Record<string, AgentMeta> = {
  A1: {
    code: "A1",
    title: "Intent Analyst",
    role: "read the question, pick the topic",
    accent: "border-l-a1",
    dot: "bg-a1",
  },
  A3: {
    code: "A3",
    title: "Routing / Planning",
    role: "decide what happens next, and why",
    accent: "border-l-brand",
    dot: "bg-brand",
  },
  A2: {
    code: "A2",
    title: "Response Specialist",
    role: "retrieve a candidate answer",
    accent: "border-l-a2",
    dot: "bg-a2",
  },
  A4: {
    code: "A4",
    title: "Response Auditor",
    role: "check alignment, grounding, claims",
    accent: "border-l-success",
    dot: "bg-success",
  },
  A5: {
    code: "A5",
    title: "Coordinator",
    role: "make the one final decision",
    accent: "border-l-brand",
    dot: "bg-brand",
  },
};

function normaliseAgent(name: string) {
  if (META[name]) return name;
  if (name === "IntentAnalyst") return "A1";
  if (name === "ResponseSpecialist") return "A2";
  return name;
}

function decisionTone(value?: string) {
  const text = (value ?? "").toUpperCase();
  if (text === "ACCEPT" || text === "ANSWER") return "success" as const;
  if (text === "REVISE" || text === "CLARIFY" || text === "SKIPPED")
    return "warning" as const;
  if (text === "FLAG" || text === "ESCALATE") return "error" as const;
  return "neutral" as const;
}

interface AgentTimelineProps {
  result: ResolveResult;
  /** Open every card instead of only the interesting one. */
  defaultOpenAll?: boolean;
}

/**
 * Expandable A1–A5 timeline. Each card carries the agent's role, its own
 * decision, its latency and the evidence it used; expanding shows the inputs
 * and outputs that produced the decision (including A4's per-check results
 * and the before/after text when it revised).
 */
export function AgentTimeline({ result, defaultOpenAll = false }: AgentTimelineProps) {
  const audit = getAudit(result);
  const plan = getRoutePlan(result);

  const interesting = new Set<string>();
  if (defaultOpenAll) {
    result.trace.forEach((step) => interesting.add(step.agent));
  } else {
    // Auto-open the step a reviewer cares about: A4 when it changed the
    // answer, otherwise A3 when it routed away from a normal answer.
    if (audit && audit.verdict !== "accept") interesting.add("A4");
    else if (plan && plan.route !== "answer") interesting.add("A3");
  }

  const [open, setOpen] = useState<Set<string>>(interesting);
  const toggle = (agent: string) =>
    setOpen((previous) => {
      const next = new Set(previous);
      if (next.has(agent)) next.delete(agent);
      else next.add(agent);
      return next;
    });

  return (
    <ol className="space-y-2">
      {result.trace.map((step, index) => {
        const code = normaliseAgent(step.agent);
        const meta = META[code] ?? {
          code,
          title: step.agent,
          role: "agent",
          accent: "border-l-line",
          dot: "bg-ink-faint",
        };
        const isOpen = open.has(code);

        return (
          <li
            key={`${step.agent}-${index}`}
            className={cn(
              "rounded-[12px] border border-line border-l-2 bg-card",
              meta.accent
            )}
          >
            <button
              type="button"
              onClick={() => toggle(code)}
              aria-expanded={isOpen}
              className="flex w-full items-start gap-2.5 px-3 py-2.5 text-left"
            >
              <span
                className={cn(
                  "mt-0.5 flex size-6 shrink-0 items-center justify-center rounded-full font-mono text-[11px] font-semibold text-white",
                  meta.dot
                )}
              >
                {index + 1}
              </span>

              <span className="min-w-0 flex-1">
                <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
                  <span className="text-sm font-semibold text-ink">
                    {meta.code} · {meta.title}
                  </span>
                  {step.decision ? (
                    <Chip tone={decisionTone(step.decision)}>
                      {step.decision}
                    </Chip>
                  ) : null}
                </span>
                <span className="mt-0.5 block text-xs leading-[18px] text-ink-secondary">
                  {meta.role}
                  {step.output_summary && !isOpen ? (
                    <>
                      <span aria-hidden="true"> · </span>
                      <span className="text-ink-faint">
                        {step.output_summary.slice(0, 120)}
                      </span>
                    </>
                  ) : null}
                </span>
              </span>

              <span className="flex shrink-0 items-center gap-2">
                <span className="inline-flex items-center gap-1 font-mono text-xs tabular-nums text-ink-secondary">
                  <Clock3 size={11} aria-hidden="true" />
                  {formatLatency(step.latency_ms)}
                </span>
                <ChevronRight
                  size={14}
                  aria-hidden="true"
                  className={cn(
                    "text-ink-faint transition-transform",
                    isOpen && "rotate-90"
                  )}
                />
              </span>
            </button>

            {isOpen ? (
              <div className="animate-fade-in border-t border-line px-3 py-3">
                <AgentDetail
                  code={code}
                  step={step}
                  result={result}
                  plan={plan}
                  audit={audit}
                />

                {(step.evidence?.length ?? 0) > 0 ? (
                  <div className="mt-2.5">
                    <p className="font-mono text-[11px] uppercase tracking-[0.08em] text-ink-faint">
                      evidence
                    </p>
                    <div className="mt-1 flex flex-wrap gap-1">
                      {step.evidence!.slice(0, 8).map((item) => (
                        <span
                          key={item}
                          className="rounded-[6px] border border-line bg-soft px-1.5 py-0.5 font-mono text-[11px] text-ink-secondary"
                        >
                          {item}
                        </span>
                      ))}
                    </div>
                  </div>
                ) : null}

                <details className="group mt-2.5">
                  <summary className="flex cursor-pointer list-none items-center gap-1 text-xs font-medium text-ink-secondary hover:text-ink [&::-webkit-details-marker]:hidden">
                    Raw payload
                    <ChevronRight
                      size={11}
                      aria-hidden="true"
                      className="transition-transform group-open:rotate-90"
                    />
                  </summary>
                  <pre className="mt-1.5 max-h-48 overflow-auto rounded-[8px] border border-line bg-soft p-2 font-mono text-[11px] leading-[16px] text-ink-secondary">
                    {JSON.stringify(
                      { input: step.input, output: step.output },
                      null,
                      2
                    )}
                  </pre>
                </details>
              </div>
            ) : null}
          </li>
        );
      })}
    </ol>
  );
}

function AgentDetail({
  code,
  step,
  result,
  plan,
  audit,
}: {
  code: string;
  step: TraceStep;
  result: ResolveResult;
  plan: RoutePlan | null;
  audit: AuditResult | null;
}) {
  if (code === "A1") return <A1Detail result={result} />;
  if (code === "A3") return <A3Detail plan={plan} />;
  if (code === "A2") return <A2Detail result={result} />;
  if (code === "A4") return <A4Detail audit={audit} result={result} />;
  return (
    <div className="space-y-2">
      <p className="text-xs leading-[18px] text-ink-secondary">
        {step.output_summary || "Final decision issued."}
      </p>
      {(result.reason_codes?.length ?? 0) > 0 ? (
        <div className="flex flex-wrap gap-1">
          {result.reason_codes!.map((code) => (
            <span
              key={code}
              className="rounded-[6px] border border-line bg-soft px-1.5 py-0.5 font-mono text-[11px] text-ink-secondary"
            >
              {code}
            </span>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function A1Detail({ result }: { result: ResolveResult }) {
  const a1 = getA1Output(result);
  if (!a1) {
    return (
      <p className="text-xs text-ink-secondary">A1 output unavailable.</p>
    );
  }
  return (
    <div className="space-y-2">
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="intent" value={a1.intent} mono />
        <Stat label="category" value={a1.category} mono />
        <Stat label="confidence" value={formatPercent(a1.confidence, 1)} mono />
        <Stat label="margin" value={formatScore(a1.margin, 3)} mono />
      </div>
      {a1.top3?.length ? (
        <div>
          <p className="font-mono text-[11px] uppercase tracking-[0.08em] text-ink-faint">
            top 3
          </p>
          <div className="mt-1 space-y-1">
            {a1.top3.map((item, index) => (
              <div key={item.intent} className="flex items-center gap-2">
                <span
                  className={cn(
                    "min-w-0 flex-1 truncate text-xs",
                    index === 0 ? "font-medium text-ink" : "text-ink-secondary"
                  )}
                >
                  {humaniseIntent(item.intent)}{" "}
                  <span className="font-mono text-[11px] text-ink-faint">
                    {item.intent}
                  </span>
                </span>
                <span className="w-10 shrink-0 text-right font-mono text-xs tabular-nums text-ink-secondary">
                  {formatPercent(item.prob, 0)}
                </span>
              </div>
            ))}
          </div>
        </div>
      ) : null}
      <p className="text-xs text-ink-secondary">
        Closest past question{" "}
        <span className="font-mono text-ink">
          {a1.top1_similarity === null
            ? "—"
            : formatScore(a1.top1_similarity, 3)}
        </span>
        {a1.needs_clarification ? " · flagged as uncertain" : null}
      </p>
    </div>
  );
}

function A3Detail({ plan }: { plan: RoutePlan | null }) {
  if (!plan) {
    return <p className="text-xs text-ink-secondary">A3 plan unavailable.</p>;
  }
  return (
    <div className="space-y-2.5">
      <div className="flex flex-wrap items-center gap-1.5">
        <Chip tone={plan.route === "answer" ? "success" : plan.route === "clarify" ? "warning" : "error"}>
          route: {plan.route}
        </Chip>
        {plan.retrieval?.mode ? (
          <Chip tone="outline">
            {plan.retrieval.mode} · k={plan.retrieval.k}
          </Chip>
        ) : null}
        {plan.retrieval?.widen ? <Chip tone="outline">widened</Chip> : null}
      </div>

      {plan.reasons?.length ? (
        <ul className="space-y-1">
          {plan.reasons.map((reason) => (
            <li
              key={reason}
              className="flex gap-1.5 text-xs leading-[18px] text-ink-secondary"
            >
              <span aria-hidden="true" className="text-ink-faint">
                •
              </span>
              <span className="min-w-0">{reason}</span>
            </li>
          ))}
        </ul>
      ) : null}

      {plan.missing_slots?.length ? (
        <div className="rounded-[8px] border border-warning/20 bg-warning-soft px-2.5 py-2">
          <p className="text-xs font-medium text-warning-ink">
            Missing information
          </p>
          <ul className="mt-1 space-y-0.5">
            {plan.missing_slots.map((slot) => (
              <li key={slot.slot} className="text-xs leading-[18px] text-ink">
                <span className="font-medium">{slot.label}</span>
                <span className="font-mono text-[11px] text-ink-faint">
                  {" "}
                  ({slot.slot})
                </span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {plan.clarify_options?.length ? (
        <div>
          <p className="font-mono text-[11px] uppercase tracking-[0.08em] text-ink-faint">
            clarify options
          </p>
          <div className="mt-1 flex flex-wrap gap-1">
            {plan.clarify_options.map((option) => (
              <span
                key={option.intent}
                className="rounded-[6px] border border-line bg-soft px-1.5 py-0.5 font-mono text-[11px] text-ink-secondary"
              >
                {option.intent} · {formatPercent(option.prob, 0)}
              </span>
            ))}
          </div>
        </div>
      ) : null}

      {plan.reason_codes?.length ? (
        <div className="flex flex-wrap gap-1">
          {plan.reason_codes.map((code) => (
            <span
              key={code}
              className="rounded-[6px] border border-line bg-soft px-1.5 py-0.5 font-mono text-[11px] text-ink-secondary"
            >
              {code}
            </span>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function A2Detail({ result }: { result: ResolveResult }) {
  const step = result.trace.find((item) => normaliseAgent(item.agent) === "A2");
  const output = (step?.output ?? {}) as Record<string, unknown>;
  if (output.skipped) {
    return (
      <p className="text-xs leading-[18px] text-ink-secondary">
        Skipped — A3 routed to{" "}
        <span className="font-mono text-ink">{String(output.reason ?? "")}</span>
        , so no candidate answer was needed.
      </p>
    );
  }
  const examples = Array.isArray(output.supporting_examples)
    ? (output.supporting_examples as Array<Record<string, unknown>>)
    : [];
  const selfCheck = (output.self_check ?? {}) as Record<string, unknown>;

  return (
    <div className="space-y-2">
      <div className="grid grid-cols-2 gap-2">
        <Stat
          label="retrieval score"
          value={formatScore(Number(output.retrieval_score ?? 0), 3)}
          mono
        />
        <Stat
          label="self-check"
          value={selfCheck.check_passed ? "passed" : "failed"}
          mono
        />
      </div>
      <p className="text-xs leading-[18px] text-ink-secondary">
        {examples.length} supporting {examples.length === 1 ? "case" : "cases"}{" "}
        retrieved.
      </p>
      {examples.length ? (
        <ul className="space-y-1">
          {examples.slice(0, 2).map((example, index) => (
            <li
              key={index}
              className="rounded-[8px] border border-line bg-soft px-2 py-1.5 font-mono text-[11px] leading-[16px] text-ink-secondary"
            >
              {String(example.intent ?? "—")} ·{" "}
              {String(example.instruction ?? "").slice(0, 90)}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

function A4Detail({
  audit,
  result,
}: {
  audit: AuditResult | null;
  result: ResolveResult;
}) {
  if (!audit) {
    const step = result.trace.find((item) => normaliseAgent(item.agent) === "A4");
    const output = (step?.output ?? {}) as Record<string, unknown>;
    return (
      <p className="text-xs leading-[18px] text-ink-secondary">
        {output.skipped
          ? `Skipped — A3 routed to ${String(output.reason ?? "")}, so there was no candidate to audit.`
          : "A4 audit unavailable."}
      </p>
    );
  }

  const original =
    (result.trace.find((item) => normaliseAgent(item.agent) === "A2")?.output as
      | Record<string, unknown>
      | undefined)?.response ?? "";
  const revised = audit.revised_response;

  return (
    <div className="space-y-2.5">
      <div className="flex flex-wrap items-center gap-1.5">
        <Chip tone={decisionTone(audit.verdict)}>verdict: {audit.verdict}</Chip>
        {audit.retry_candidate_rank ? (
          <Chip tone="outline">retried rank {audit.retry_candidate_rank}</Chip>
        ) : null}
      </div>

      <ul className="space-y-1">
        {audit.checks?.map((check) => (
          <li key={check.name} className="flex items-start gap-1.5">
            <ShieldCheck
              size={12}
              aria-hidden="true"
              className={cn(
                "mt-1 shrink-0",
                check.passed ? "text-success" : "text-error"
              )}
            />
            <span className="min-w-0">
              <span className="text-xs font-medium text-ink">
                {check.name.replace(/_/g, " ")}
              </span>
              <span className="block text-xs leading-[18px] text-ink-secondary">
                {check.detail || (check.passed ? "passed" : "failed")}
              </span>
            </span>
          </li>
        ))}
      </ul>

      {audit.issues?.length ? (
        <div className="rounded-[8px] border border-error/15 bg-error-soft px-2.5 py-2">
          <p className="text-xs font-medium text-error-ink">Issues raised</p>
          <ul className="mt-1 space-y-0.5">
            {audit.issues.slice(0, 4).map((issue) => (
              <li key={issue} className="text-xs leading-[18px] text-ink">
                {issue}
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {revised ? (
        <div className="grid gap-2 sm:grid-cols-2">
          <div className="rounded-[8px] border border-error/15 bg-error-soft/60 p-2">
            <p className="font-mono text-[11px] uppercase tracking-[0.08em] text-error-ink">
              before (A2 candidate)
            </p>
            <p className="mt-1 line-clamp-4 text-xs leading-[18px] text-ink">
              {String(original || "—")}
            </p>
          </div>
          <div className="rounded-[8px] border border-success/20 bg-success-soft p-2">
            <p className="font-mono text-[11px] uppercase tracking-[0.08em] text-success-ink">
              after (A4 rewrite)
            </p>
            <p className="mt-1 line-clamp-4 text-xs leading-[18px] text-ink">
              {revised}
            </p>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function Stat({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="min-w-0">
      <p className="font-mono text-[11px] uppercase tracking-[0.06em] text-ink-faint">
        {label}
      </p>
      <p
        className={cn(
          "truncate text-xs text-ink",
          mono && "font-mono tabular-nums"
        )}
      >
        {value}
      </p>
    </div>
  );
}
