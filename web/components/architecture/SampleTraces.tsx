"use client";

import { CheckCircle2, ChevronDown, XCircle } from "lucide-react";
import type { TraceExample } from "@/lib/data";
import { handoffFromTrace } from "@/lib/data";
import { Chip } from "@/components/ui/Chip";
import { EmptyDataFile } from "@/components/ui/EmptyDataFile";
import {
  formatPercent,
  formatScore,
  humaniseIntent,
} from "@/lib/utils";

const CHECK_LABELS: Record<string, string> = {
  placeholder_safe: "Placeholders resolved",
  source_similarity_ok: "Source close enough",
  response_intent_ok: "Answer matches intent",
};

export function SampleTraces({
  traces,
  fileMissing,
}: {
  traces: TraceExample[];
  fileMissing: boolean;
}) {
  return (
    <section aria-labelledby="sample-traces-heading" className="mt-8">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h2
            id="sample-traces-heading"
            className="text-base font-semibold text-ink"
          >
            Sample traces
          </h2>
          <p className="mt-0.5 max-w-[640px] text-xs leading-[18px] text-ink-secondary">
            Five real held-out questions, each shown as it travels through both
            agents. Open a card to see the exact handoff, sources and checks.
          </p>
        </div>
        <span className="text-xs text-ink-faint">
          {traces.length > 0 ? `${traces.length} examples` : ""}
        </span>
      </div>

      {fileMissing || traces.length === 0 ? (
        <div className="mt-3">
          <EmptyDataFile
            file="examples.json"
            detail="Add examples.json with a `traces` array to show real end-to-end questions here."
          />
        </div>
      ) : (
        <div className="mt-3 space-y-2">
          {traces.map((trace, index) => (
            <TraceCard
              key={`${trace.query}-${index}`}
              trace={trace}
              index={index}
              defaultOpen={index === 0}
            />
          ))}
        </div>
      )}
    </section>
  );
}

function TraceCard({
  trace,
  index,
  defaultOpen,
}: {
  trace: TraceExample;
  index: number;
  defaultOpen: boolean;
}) {
  const a1 = trace.a1_output;
  const a2 = trace.a2_output;
  const checked = a2.self_check?.check_passed;

  return (
    <details
      open={defaultOpen}
      className="group rounded-card border border-line bg-card"
    >
      <summary className="flex cursor-pointer list-none items-center gap-3 px-3.5 py-3 [&::-webkit-details-marker]:hidden">
        <span className="font-mono text-xs text-ink-faint">
          {String(index + 1).padStart(2, "0")}
        </span>
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium text-ink">
            {trace.query}
          </p>
          <p className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs text-ink-secondary">
            <span className="font-mono">{a1.intent}</span>
            <span aria-hidden="true" className="text-ink-faint">
              ·
            </span>
            <span>{formatPercent(a1.confidence, 1)} confidence</span>
            {a1.needs_clarification ? (
              <>
                <span aria-hidden="true" className="text-ink-faint">
                  ·
                </span>
                <span className="text-warning-ink">Flagged by A1</span>
              </>
            ) : null}
          </p>
        </div>
        <div className="hidden shrink-0 items-center gap-1.5 sm:flex">
          <Chip tone={a1.needs_clarification ? "warning" : "a1"}>
            {humaniseIntent(a1.intent)}
          </Chip>
          {typeof checked === "boolean" ? (
            <Chip tone={checked ? "success" : "warning"}>
              {checked ? "Self-check passed" : "Check failed"}
            </Chip>
          ) : null}
        </div>
        <ChevronDown
          size={16}
          aria-hidden="true"
          className="shrink-0 text-ink-faint transition-transform group-open:rotate-180"
        />
      </summary>

      <div className="space-y-3 border-t border-line px-3.5 py-3.5">
        <Step n={1} title="A1 · Intent Analyst">
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Fact label="Intent" value={a1.intent} mono />
            <Fact label="Category" value={a1.category} mono />
            <Fact label="Confidence" value={formatPercent(a1.confidence, 2)} />
            <Fact
              label="Nearest match"
              value={
                a1.top1_similarity === null || a1.top1_similarity === undefined
                  ? "—"
                  : formatScore(a1.top1_similarity, 3)
              }
            />
          </div>
          <ul className="mt-2 space-y-1.5">
            {a1.top3.map((row) => (
              <li key={row.intent} className="flex items-center gap-2">
                <span className="w-[150px] shrink-0 truncate font-mono text-[11px] text-ink-secondary">
                  {row.intent}
                </span>
                <span className="h-1.5 flex-1 overflow-hidden rounded-pill bg-[#F0F0ED]">
                  <span
                    className="block h-full rounded-pill bg-a1/70"
                    style={{ width: `${Math.max(row.prob * 100, 1)}%` }}
                  />
                </span>
                <span className="w-12 shrink-0 text-right font-mono text-[11px] tabular-nums text-ink">
                  {formatPercent(row.prob, 1)}
                </span>
              </li>
            ))}
          </ul>
        </Step>

        <Step n={2} title="Handoff JSON">
          <pre className="max-h-[220px] overflow-auto whitespace-pre rounded-[10px] border border-line bg-soft px-2.5 py-2 font-mono text-[11px] leading-[16px] text-ink-secondary">
            {JSON.stringify(handoffFromTrace(trace), null, 2)}
          </pre>
        </Step>

        <Step n={3} title="A2 · Response Specialist">
          <div className="flex flex-wrap items-center gap-1.5">
            <Chip tone="neutral">
              {a2.used_intents?.length
                ? `Searched ${a2.used_intents.join(", ")}`
                : "Searched the routed intent"}
            </Chip>
            {typeof a2.retrieval_score === "number" ? (
              <Chip tone="outline">
                Retrieval score {formatScore(a2.retrieval_score, 3)}
              </Chip>
            ) : null}
            {a2.low_confidence ? (
              <Chip tone="warning">Low confidence</Chip>
            ) : null}
          </div>

          {a2.supporting_examples && a2.supporting_examples.length > 0 ? (
            <ul className="mt-2 space-y-1.5">
              {a2.supporting_examples.slice(0, 3).map((source, sourceIndex) => (
                <li
                  key={`${source.instruction}-${sourceIndex}`}
                  className="rounded-[10px] border border-line bg-white px-2.5 py-2"
                >
                  <div className="flex items-start justify-between gap-2">
                    <p className="min-w-0 text-[11px] font-medium leading-[16px] text-ink">
                      “{source.instruction}”
                    </p>
                    <span className="shrink-0 font-mono text-[11px] tabular-nums text-ink-secondary">
                      {formatScore(source.similarity, 3)}
                    </span>
                  </div>
                  <p className="mt-1 line-clamp-2 text-[11px] leading-[16px] text-ink-secondary">
                    {source.response}
                  </p>
                </li>
              ))}
            </ul>
          ) : null}

          {a2.self_check ? (
            <ul className="mt-2 space-y-1">
              {Object.entries(CHECK_LABELS).map(([key, label]) => {
                const value = a2.self_check?.[
                  key as keyof NonNullable<TraceExample["a2_output"]["self_check"]>
                ];
                if (typeof value !== "boolean") return null;
                return (
                  <li
                    key={key}
                    className="flex items-center justify-between gap-2 rounded-[8px] border border-line px-2.5 py-1.5"
                  >
                    <span className="text-[11px] text-ink-secondary">
                      {label}
                    </span>
                    <span
                      className={
                        value
                          ? "inline-flex items-center gap-1 text-[11px] font-medium text-success-ink"
                          : "inline-flex items-center gap-1 text-[11px] font-medium text-warning-ink"
                      }
                    >
                      {value ? (
                        <CheckCircle2 size={12} aria-hidden="true" />
                      ) : (
                        <XCircle size={12} aria-hidden="true" />
                      )}
                      {value ? "Pass" : "Fail"}
                    </span>
                  </li>
                );
              })}
            </ul>
          ) : null}
        </Step>

        <Step n={4} title="Answer returned">
          <p className="whitespace-pre-line rounded-[10px] border border-line bg-soft px-3 py-2.5 text-[13px] leading-[20px] text-ink">
            {a2.response}
          </p>
        </Step>
      </div>
    </details>
  );
}

function Step({
  n,
  title,
  children,
}: {
  n: number;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section>
      <div className="flex items-center gap-2">
        <span className="flex size-5 items-center justify-center rounded-full border border-line bg-soft font-mono text-[10px] font-medium text-ink-secondary">
          {n}
        </span>
        <h4 className="text-xs font-semibold uppercase tracking-[0.06em] text-ink-secondary">
          {title}
        </h4>
      </div>
      <div className="mt-1.5 pl-7">{children}</div>
    </section>
  );
}

function Fact({
  label,
  value,
  mono,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="rounded-[8px] border border-line bg-white px-2 py-1.5">
      <p className="text-[10px] uppercase tracking-[0.06em] text-ink-faint">
        {label}
      </p>
      <p
        className={
          mono
            ? "mt-0.5 truncate font-mono text-[11px] text-ink"
            : "mt-0.5 font-mono text-[11px] tabular-nums text-ink"
        }
      >
        {value}
      </p>
    </div>
  );
}
