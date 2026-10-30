"use client";

import React, { useState } from "react";
import {
  CheckCircle2,
  AlertTriangle,
  ArrowDown,
  Copy,
  Check,
  ChevronDown,
  ChevronUp,
  Clock,
  Database,
  Layers,
  Sparkles,
} from "lucide-react";
import { HighlightedText } from "./HighlightedText";
import { ConfidenceBar } from "./ConfidenceBar";
import type { ResolveResult, A1Output, A2Output } from "@/lib/schemas";

interface AgentTraceProps {
  run: ResolveResult | null;
  threshold?: number | null;
  onStartTour?: () => void;
  tourActive?: boolean;
}

export function AgentTrace({
  run,
  threshold,
  onStartTour,
  tourActive,
}: AgentTraceProps) {
  const [showHandoffJson, setShowHandoffJson] = useState(false);
  const [copiedHandoff, setCopiedHandoff] = useState(false);

  if (!run) {
    return (
      <div className="flex flex-col h-[760px] bg-[var(--card-bg)] rounded-xl border border-[var(--card-border)] p-8 items-center justify-center text-center shadow-sm">
        <div className="w-14 h-14 rounded-full bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center text-indigo-400 mb-4">
          <Layers className="w-6 h-6" />
        </div>
        <h3 className="text-xl font-bold mb-2">Agent Trace</h3>
        <p className="text-sm text-[var(--muted-text)] max-w-sm mb-6">
          Submit a query from the chat on the left or tap the guided tour to inspect step-by-step
          autonomous classification, handoff, and retrieval execution.
        </p>
        {onStartTour && (
          <button
            onClick={onStartTour}
            className="px-4 py-2 rounded-lg btn-teal text-sm font-semibold flex items-center gap-2 cursor-pointer shadow-sm hover:opacity-95"
          >
            <Sparkles className="w-4 h-4" />
            <span>Start Guided Tour</span>
          </button>
        )}
      </div>
    );
  }

  // Extract trace steps
  const trace = run.trace || [];
  const a1Step = trace.find((t) => t.agent === "IntentAnalyst");
  const a2Step = trace.find((t) => t.agent === "ResponseSpecialist");

  const a1Output: A1Output | undefined = a1Step?.output;
  const a2Output: A2Output | undefined = a2Step?.output;
  const handoffInput = a2Step?.input;

  const a1Latency = a1Step?.latency_ms ?? 0;
  const a2Latency = a2Step?.latency_ms ?? 0;
  const totalLatency = run.total_latency_ms;

  const needsClar = a1Output?.needs_clarification;
  const confPercent = a1Output ? Math.round(a1Output.confidence * 100) : 0;
  const threshPercent = threshold != null ? Math.round(threshold * 100) : 65;

  // Dynamic plain-English captions
  const a1Caption = a1Output
    ? confPercent >= threshPercent
      ? `A1 is ${confPercent}% confident this is '${a1Output.intent}', comfortably above the ${threshPercent}% threshold.`
      : `A1 is only ${confPercent}% confident it is '${a1Output.intent}', falling below the ${threshPercent}% threshold.`
    : "Classifying query intent...";

  const usedIntentsReason =
    a2Output?.used_intents.length === 1
      ? "Searched only the predicted intent"
      : "Searched the top-3 intents because A1 asked for clarification";

  const handleCopyHandoff = () => {
    if (!handoffInput) return;
    navigator.clipboard.writeText(JSON.stringify(handoffInput, null, 2));
    setCopiedHandoff(true);
    setTimeout(() => setCopiedHandoff(false), 2000);
  };

  return (
    <div className="flex flex-col h-[760px] bg-[var(--card-bg)] rounded-xl border border-[var(--card-border)] shadow-sm overflow-hidden">
      {/* Trace Header */}
      <div className="p-4 border-b border-[var(--card-border)] flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h2 className="text-xl font-bold tracking-tight">Agent trace</h2>
          <div className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-gray-800 text-xs text-gray-300 border border-gray-700">
            <Clock className="w-3.5 h-3.5 text-teal-400" />
            <span>Total: {totalLatency.toFixed(1)} ms</span>
          </div>
        </div>

        {onStartTour && (
          <button
            onClick={onStartTour}
            className={`px-3 py-1 rounded-lg text-xs font-semibold flex items-center gap-1.5 border transition-all ${
              tourActive
                ? "bg-teal-500/20 text-teal-300 border-teal-500"
                : "bg-gray-800/80 hover:bg-gray-700 text-gray-300 border-gray-700"
            }`}
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>{tourActive ? "Tour Active" : "Guided Tour"}</span>
          </button>
        )}
      </div>

      {/* Timeline Scroll Area */}
      <div className="flex-1 p-4 overflow-y-auto space-y-5 text-sm">
        {/* ── STEP 1: Query ─────────────────────────────────────────── */}
        <div className="relative pl-6 border-l-2 border-gray-700/80 space-y-1">
          <div className="absolute -left-[9px] top-0 w-4 h-4 rounded-full bg-indigo-500 ring-4 ring-[var(--card-bg)]" />
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-indigo-400">
              1. Query Input
            </span>
          </div>
          <div className="p-2.5 rounded-lg bg-[var(--card-hover)]/60 border border-[var(--card-border)] font-medium">
            &ldquo;{run.query}&rdquo;
          </div>
        </div>

        {/* ── STEP 2: Agent A1 (Intent Analyst) ──────────────────────── */}
        {a1Output && (
          <div className="relative pl-6 border-l-2 border-indigo-500/80 space-y-2">
            <div className="absolute -left-[9px] top-0 w-4 h-4 rounded-full bg-indigo-500 ring-4 ring-[var(--card-bg)]" />

            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold uppercase tracking-wider text-indigo-400">
                  2. A1 Intent Analyst
                </span>
                <span className="text-xs text-[var(--muted-text)] font-mono">
                  ({a1Latency.toFixed(1)} ms)
                </span>
              </div>
              {needsClar ? (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-amber-950/60 text-amber-300 border border-amber-500/60">
                  <AlertTriangle className="w-3 h-3" /> Needs Clarification
                </span>
              ) : (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-emerald-950/60 text-emerald-300 border border-emerald-500/60">
                  <CheckCircle2 className="w-3 h-3" /> Confident
                </span>
              )}
            </div>

            {/* One-line Plain English Caption */}
            <p className="text-xs text-[var(--muted-text)] italic">{a1Caption}</p>

            {/* A1 Card (Indigo Border) */}
            <div className="p-3.5 rounded-xl bg-gray-900/60 border-2 border-a1 space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <span className="text-xs text-[var(--muted-text)] block">Category</span>
                  <span className="font-semibold text-gray-200">{a1Output.category}</span>
                </div>
                <div>
                  <span className="text-xs text-[var(--muted-text)] block">Intent</span>
                  <span className="font-mono font-bold text-indigo-300">
                    {a1Output.intent}
                  </span>
                </div>
                <div>
                  <span className="text-xs text-[var(--muted-text)] block">Margin</span>
                  <span className="font-mono text-gray-300">
                    {a1Output.margin.toFixed(4)}
                  </span>
                </div>
                {a1Output.top1_similarity != null && (
                  <div>
                    <span className="text-xs text-[var(--muted-text)] block">Source Sim</span>
                    <span className="font-mono text-gray-300">
                      {a1Output.top1_similarity.toFixed(4)}
                    </span>
                  </div>
                )}
              </div>

              {/* Confidence Bar */}
              <ConfidenceBar
                confidence={a1Output.confidence}
                threshold={threshold ?? 0.65}
              />

              {/* Top-3 Candidates */}
              <div className="pt-2 border-t border-gray-800 space-y-1.5">
                <span className="text-xs text-[var(--muted-text)] block font-semibold">
                  Top-3 Hypotheses
                </span>
                <div className="grid grid-cols-1 gap-1 text-xs font-mono">
                  {a1Output.top3.map((t, idx) => (
                    <div
                      key={idx}
                      className="flex items-center justify-between px-2 py-1 rounded bg-gray-800/60"
                    >
                      <span className="truncate">{t.intent}</span>
                      <span className="text-indigo-300 font-semibold">
                        {(t.prob * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── STEP 3: Handoff Panel ─────────────────────────────────── */}
        <div className="relative pl-6 border-l-2 border-gray-700/80 space-y-2">
          <div className="absolute -left-[9px] top-0 w-4 h-4 rounded-full bg-gray-600 ring-4 ring-[var(--card-bg)]" />

          <button
            onClick={() => setShowHandoffJson(!showHandoffJson)}
            className="w-full flex items-center justify-between p-2 rounded-lg bg-[var(--card-hover)]/40 hover:bg-[var(--card-hover)] border border-[var(--card-border)] transition-colors text-xs font-semibold cursor-pointer"
          >
            <div className="flex items-center gap-1.5 text-gray-300">
              <ArrowDown className="w-3.5 h-3.5 text-teal-400" />
              <span>3. Handoff to A2 (A1 contract state)</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[var(--muted-text)] font-normal text-[11px]">
                {showHandoffJson ? "Hide JSON" : "Inspect payload"}
              </span>
              {showHandoffJson ? (
                <ChevronUp className="w-3.5 h-3.5" />
              ) : (
                <ChevronDown className="w-3.5 h-3.5" />
              )}
            </div>
          </button>

          {showHandoffJson && handoffInput && (
            <div className="relative p-3 rounded-lg bg-gray-950 border border-gray-800 text-xs font-mono overflow-x-auto">
              <button
                onClick={handleCopyHandoff}
                className="absolute top-2 right-2 px-2 py-1 rounded bg-gray-800 hover:bg-gray-700 text-gray-300 flex items-center gap-1 text-[11px]"
                title="Copy JSON payload"
              >
                {copiedHandoff ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                <span>{copiedHandoff ? "Copied" : "Copy"}</span>
              </button>
              <pre className="text-gray-300 max-h-48 overflow-y-auto">
                {JSON.stringify(handoffInput, null, 2)}
              </pre>
            </div>
          )}
        </div>

        {/* ── STEP 4: Agent A2 (Response Specialist) ────────────────── */}
        {a2Output && (
          <div className="relative pl-6 border-l-2 border-teal-500/80 space-y-2">
            <div className="absolute -left-[9px] top-0 w-4 h-4 rounded-full bg-teal-500 ring-4 ring-[var(--card-bg)]" />

            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold uppercase tracking-wider text-teal-400">
                  4. A2 Response Specialist
                </span>
                <span className="text-xs text-[var(--muted-text)] font-mono">
                  ({a2Latency.toFixed(1)} ms)
                </span>
              </div>
              <div className="flex items-center gap-1.5 text-xs">
                <span className="px-2 py-0.5 rounded bg-teal-950 text-teal-300 border border-teal-600 flex items-center gap-1">
                  <Database className="w-3 h-3" />
                  {run.retrieval_backend}
                </span>
              </div>
            </div>

            {/* Used Intents Reason Caption */}
            <p className="text-xs text-[var(--muted-text)] italic">{usedIntentsReason}</p>

            {/* A2 Panel (Teal Border) */}
            <div className="p-3.5 rounded-xl bg-gray-900/60 border-2 border-a2 space-y-3">
              {/* Answer card with Highlighted Placeholders */}
              <div className="p-3 rounded-lg bg-gray-800/80 border border-teal-500/30 space-y-1">
                <span className="text-xs text-teal-400 font-semibold block">Grounded Template</span>
                <div className="text-xs text-gray-200 whitespace-pre-line leading-relaxed">
                  <HighlightedText text={a2Output.response} />
                </div>
                {!a2Output.self_check.placeholder_safe && (
                  <span className="inline-block mt-1 text-[11px] text-amber-400 font-medium bg-amber-950/40 px-2 py-0.5 rounded border border-amber-600/40">
                    ⚠ raw template placeholders present
                  </span>
                )}
              </div>

              {/* Self-check badges */}
              <div className="pt-1">
                <span className="text-xs text-[var(--muted-text)] font-semibold block mb-1">
                  Self-Check Gate
                </span>
                <div className="flex flex-wrap gap-1.5 text-[11px] font-mono">
                  <span
                    className={`px-2 py-0.5 rounded border ${
                      a2Output.self_check.placeholder_safe
                        ? "bg-emerald-950/60 text-emerald-300 border-emerald-600"
                        : "bg-amber-950/60 text-amber-300 border-amber-600"
                    }`}
                  >
                    placeholder_safe: {String(a2Output.self_check.placeholder_safe)}
                  </span>
                  <span
                    className={`px-2 py-0.5 rounded border ${
                      a2Output.self_check.source_similarity_ok
                        ? "bg-emerald-950/60 text-emerald-300 border-emerald-600"
                        : "bg-amber-950/60 text-amber-300 border-amber-600"
                    }`}
                  >
                    source_sim: {String(a2Output.self_check.source_similarity_ok)}
                  </span>
                  <span
                    className={`px-2 py-0.5 rounded border ${
                      a2Output.self_check.response_intent_ok
                        ? "bg-emerald-950/60 text-emerald-300 border-emerald-600"
                        : "bg-amber-950/60 text-amber-300 border-amber-600"
                    }`}
                  >
                    intent_ok: {String(a2Output.self_check.response_intent_ok)}
                  </span>
                  <span className="px-2 py-0.5 rounded bg-gray-800 text-gray-300 border border-gray-700">
                    rejected: {a2Output.self_check.candidates_rejected}
                  </span>
                </div>
              </div>

              {/* Supporting Examples (3 cards with similarity) */}
              {a2Output.supporting_examples?.length > 0 && (
                <div className="pt-2 border-t border-gray-800 space-y-1.5">
                  <span className="text-xs text-[var(--muted-text)] font-semibold block">
                    Supporting Examples ({a2Output.supporting_examples.length})
                  </span>
                  <div className="space-y-1.5 max-h-40 overflow-y-auto">
                    {a2Output.supporting_examples.slice(0, 3).map((ex, i) => (
                      <div
                        key={i}
                        className="p-2 rounded bg-gray-800/50 border border-gray-700/60 text-xs space-y-1"
                      >
                        <div className="flex items-center justify-between text-teal-300 font-mono text-[11px]">
                          <span>Example #{i + 1}</span>
                          <span>Sim: {ex.similarity.toFixed(4)}</span>
                        </div>
                        <p className="text-gray-300 line-clamp-2">&ldquo;{ex.instruction}&rdquo;</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── STEP 5: Final Status Chip ─────────────────────────────── */}
        <div className="relative pl-6 border-l-2 border-transparent space-y-2">
          <div className="absolute -left-[9px] top-0 w-4 h-4 rounded-full bg-emerald-500 ring-4 ring-[var(--card-bg)]" />

          <div className="flex items-center gap-2">
            <span className="text-xs font-bold uppercase tracking-wider text-gray-400">
              5. Final Status:
            </span>
            {needsClar ? (
              <span className="px-3 py-1 rounded-full text-xs font-bold bg-amber-500/20 text-amber-400 border border-amber-500">
                Clarify
              </span>
            ) : (
              <span className="px-3 py-1 rounded-full text-xs font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500">
                Answer
              </span>
            )}
            {run.spokenAloud && (
              <span className="text-xs text-teal-400 font-medium">Spoken aloud</span>
            )}
          </div>
        </div>

        {/* ── Planned Level 2 Agents ─────────────────────────────────── */}
        <div className="pt-4 border-t border-[var(--card-border)] space-y-2">
          <span className="text-xs font-bold text-[var(--muted-text)] uppercase tracking-wider block">
            Planned Level 2 Agents
          </span>
          <div className="grid grid-cols-3 gap-2">
            {(run.planned_agents || [
              { name: "Verifier", status: "planned", planned_function: "Response fact-checking" },
              { name: "Clarifier", status: "planned", planned_function: "Clarifying question gen" },
              { name: "Escalator", status: "planned", planned_function: "Human escalation routing" },
            ]).map((agent, i) => (
              <div
                key={i}
                className="p-2.5 rounded-lg bg-gray-900/40 border border-gray-800 text-center opacity-65"
              >
                <span className="block text-xs font-semibold text-gray-400">{agent.name}</span>
                <span className="block text-[10px] text-gray-500 mt-0.5">{agent.status}</span>
              </div>
            ))}
          </div>

          {needsClar && (
            <p className="text-xs text-amber-400/90 italic pt-1">
              In Level 2 this query triggers a clarification question.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
