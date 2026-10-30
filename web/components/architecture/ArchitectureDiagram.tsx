"use client";

import { useEffect, useRef, useState } from "react";
import { ArrowRight, Download, X } from "lucide-react";
import { Chip } from "@/components/ui/Chip";
import { cn, formatCount, formatPercent } from "@/lib/utils";

export interface ArchitectureMetrics {
  macroF1: number | null;
  categoryAccuracy: number | null;
  answerCosine: number | null;
  selfCheckPass: number | null;
  nTest: number | null;
  trainRows: number | null;
  intents: number | null;
  categories: number | null;
}

type Tone = "neutral" | "a1" | "a2" | "data";

interface DiagramNode {
  id: string;
  kicker: string;
  title: string;
  subtitle: string;
  tone: Tone;
  x: number;
  y: number;
  w: number;
  h: number;
  purpose: string;
  input: string;
  output: string;
  algorithm: string;
  keyMetric: string;
  files: string[];
}

interface Edge {
  points: [number, number][];
  dashed?: boolean;
  label?: string;
}

const CANVAS_WIDTH = 1000;
const CANVAS_HEIGHT = 440;

const EDGES: Edge[] = [
  { points: [[140, 76], [166, 76]] },
  { points: [[302, 76], [328, 76]] },
  { points: [[490, 76], [516, 76]] },
  { points: [[642, 76], [668, 76]] },
  { points: [[844, 76], [864, 76]] },
  { points: [[111, 230], [111, 190], [236, 190], [236, 124]] },
  { points: [[370, 230], [370, 180], [411, 180], [411, 124]] },
  { points: [[665, 230], [665, 180], [751, 180], [751, 124]] },
  {
    points: [[665, 318], [665, 360]],
    dashed: true,
    label: "falls back to in-memory",
  },
];

const TONE: Record<
  Tone,
  { border: string; accent: string; fill: string; chip: "neutral" | "a1" | "a2" | "outline" }
> = {
  neutral: {
    border: "#ECEBE8",
    accent: "#71717A",
    fill: "#FFFFFF",
    chip: "outline",
  },
  a1: {
    border: "rgba(79,70,229,0.28)",
    accent: "#4F46E5",
    fill: "#FFFFFF",
    chip: "a1",
  },
  a2: {
    border: "rgba(13,148,136,0.28)",
    accent: "#0D9488",
    fill: "#FFFFFF",
    chip: "a2",
  },
  data: {
    border: "#ECEBE8",
    accent: "#A1A1AA",
    fill: "#FAFAF8",
    chip: "neutral",
  },
};

function buildNodes(metrics: ArchitectureMetrics): DiagramNode[] {
  const a1Metric =
    metrics.macroF1 !== null &&
    metrics.categoryAccuracy !== null &&
    metrics.nTest !== null
      ? `Macro-F1 ${formatPercent(
          metrics.macroF1,
          2
        )} and category accuracy ${formatPercent(
          metrics.categoryAccuracy,
          2
        )} on ${formatCount(metrics.nTest)} held-out questions.`
      : "Add results.json to web/public/data/ to show the measured Macro-F1 and category accuracy here.";

  const a2Metric =
    metrics.answerCosine !== null && metrics.selfCheckPass !== null
      ? `Answer relevance cosine ${metrics.answerCosine.toFixed(
          3
        )}; ${formatPercent(
          metrics.selfCheckPass,
          1
        )} of held-out replies pass all three checks.`
      : "Add results.json to web/public/data/ to show the measured answer relevance and self-check pass rate here.";

  const memoryMetric =
    metrics.trainRows !== null
      ? `A ${formatCount(
          metrics.trainRows
        )} × 384 float matrix ships with the model, so the fallback needs no database.`
      : "The cached train matrix ships with the model, so the fallback needs no database.";

  return [
    {
      id: "customer",
      kicker: "Input",
      title: "Customer query",
      subtitle: "free text, any wording",
      tone: "neutral",
      x: 16,
      y: 32,
      w: 124,
      h: 88,
      purpose:
        "The only thing a customer has to do: ask in their own words. No menu, no ticket form and no required keywords.",
      input: "Free text typed into the chat composer.",
      output: '{\n  "query": "where is my refund?"\n}',
      algorithm:
        "The text is trimmed and passed through the shared normalizer, which lowercases and preserves {{customer-fill}} placeholders.",
      keyMetric:
        "Questions can be a few words or several sentences; both map into the same embedding space.",
      files: ["web/components/chat/Composer.tsx", "nlp/normalize.py"],
    },
    {
      id: "orchestrator",
      kicker: "Orchestrator",
      title: "Runs A1 → A2",
      subtitle: "records every step",
      tone: "neutral",
      x: 170,
      y: 32,
      w: 132,
      h: 88,
      purpose:
        "Runs the two agents in order and records a trace for every step so the answer can be inspected afterwards.",
      input: '{\n  "query": "where is my refund?"\n}',
      output:
        '{\n  "trace": [A1, A2],\n  "final_response": "…",\n  "total_latency_ms": 412.6,\n  "retrieval_backend": "pgvector"\n}',
      algorithm:
        "A1 runs first. Its JSON output becomes A2's input. Each step stores its input, output and latency; the total is measured per request.",
      keyMetric:
        "If pgvector does not answer within 2 seconds, the same request continues on the in-memory index and the backend is labelled in the trace.",
      files: ["nlp/pipeline.py", "api/main.py"],
    },
    {
      id: "a1",
      kicker: "Agent 1",
      title: "A1 · Intent Analyst",
      subtitle:
        metrics.intents !== null && metrics.categories !== null
          ? `${metrics.intents} intents · ${metrics.categories} categories`
          : "intent + category",
      tone: "a1",
      x: 332,
      y: 32,
      w: 158,
      h: 88,
      purpose:
        "Reads the question and names the intent, its category and how confident it is. This is the routing step for the whole system.",
      input: '{\n  "query": "where is my refund?"\n}',
      output:
        '{\n  "intent": "track_refund",\n  "category": "REFUND",\n  "confidence": 0.9983,\n  "margin": 0.9971,\n  "top3": [\n    {"intent": "track_refund", "prob": 0.9983},\n    {"intent": "check_refund_policy", "prob": 0.0012},\n    {"intent": "get_refund", "prob": 0.0005}\n  ],\n  "top1_similarity": 0.9124,\n  "needs_clarification": false\n}',
      algorithm:
        `all-MiniLM-L6-v2 embeds the question → multinomial logistic regression over ${
        metrics.intents ?? "the"
      } intents → temperature scaling on dev. Confidence below tau, or similarity to the nearest training question below s_min, sets needs_clarification.`,
      keyMetric: a1Metric,
      files: ["nlp/a1.py", "outputs/models/a1.npz", "outputs/models/a1_config.json"],
    },
    {
      id: "handoff",
      kicker: "Contract · { }",
      title: "Handoff JSON",
      subtitle: "what A2 receives",
      tone: "data",
      x: 520,
      y: 32,
      w: 122,
      h: 88,
      purpose:
        "A small, explicit contract between the agents. A2 never looks inside A1's model; it only receives these fields.",
      input: "A1's output object.",
      output:
        '{\n  "query": "where is my refund?",\n  "intent": "track_refund",\n  "category": "REFUND",\n  "confidence": 0.9983,\n  "top3": […],\n  "needs_clarification": false\n}',
      algorithm:
        "Only the fields A2 needs. When A1 is unsure, needs_clarification is true and A2 searches A1's top-3 intents, weighting each candidate by the matching probability.",
      keyMetric:
        "The live trace renders this object exactly as received — no hidden state between the agents.",
      files: ["nlp/a2.py", "web/lib/data.ts", "web/components/trace/HowItWorked.tsx"],
    },
    {
      id: "a2",
      kicker: "Agent 2",
      title: "A2 · Response Specialist",
      subtitle: "retrieves + checks",
      tone: "a2",
      x: 672,
      y: 32,
      w: 172,
      h: 88,
      purpose:
        "Finds the closest real support reply for the routed intent, then checks it before returning anything to the customer.",
      input:
        '{\n  "query": "where is my refund?",\n  "a1_output": {\n    "intent": "track_refund",\n    "top3": […]…\n  }\n}',
      output:
        '{\n  "response": "To track your refund…",\n  "supporting_examples": [\n    {"instruction": "…", "response": "…", "similarity": 0.91}\n  ],\n  "used_intents": ["track_refund"],\n  "self_check": {\n    "placeholder_safe": true,\n    "source_similarity_ok": true,\n    "response_intent_ok": true,\n    "check_passed": true,\n    "candidates_rejected": 0\n  }\n}',
      algorithm:
        "Retrieval ladder: sparse TF-IDF, dense embeddings, or hybrid reciprocal-rank fusion (k = 60). Candidates are re-ranked so responses whose placeholders fit the query come first, then the first candidate passing all three checks is returned.",
      keyMetric: a2Metric,
      files: ["nlp/a2.py", "outputs/models/a2_specialist.joblib"],
    },
    {
      id: "answer",
      kicker: "Output",
      title: "Answer",
      subtitle: "sources + trace",
      tone: "neutral",
      x: 864,
      y: 32,
      w: 120,
      h: 88,
      purpose:
        "A past support reply shown with the sources it came from, the confidence behind the routing and a readable trace.",
      input: "A2's final output plus the orchestrator trace.",
      output:
        '{\n  "response": "To track your refund…",\n  "sources": 2,\n  "check_passed": true,\n  "retrieval_backend": "pgvector"\n}',
      algorithm:
        "Rendered from final_response. When A1 flags a question as out of scope, the chat withholds the answer and explains why; the trace still shows the real measured values.",
      keyMetric:
        "Every source is a real training row with its measured similarity, not a fabricated citation.",
      files: ["web/components/chat/ChatMessage.tsx", "web/components/chat/Sources.tsx"],
    },
    {
      id: "logs",
      kicker: "Telemetry",
      title: "Run logs",
      subtitle: "one row per /resolve",
      tone: "data",
      x: 16,
      y: 230,
      w: 190,
      h: 88,
      purpose:
        "Every resolved question is written to Supabase with its full trace, so failures can be inspected after the fact.",
      input: "Query + trace from the orchestrator.",
      output:
        '{\n  "id": "uuid",\n  "created_at": "2026-10-30T…",\n  "query": "where is my refund?",\n  "trace": [ … ]\n}',
      algorithm:
        "Best-effort insert into the runs table after the reply is built. A logging failure never fails the customer request.",
      keyMetric:
        "Runs and feedback live in separate tables; feedback is written by POST /feedback.",
      files: ["supabase/schema.sql", "api/main.py", "web/app/api/feedback/route.ts"],
    },
    {
      id: "embedding",
      kicker: "Encoder",
      title: "Embedding model",
      subtitle: "all-MiniLM-L6-v2 · 384-d",
      tone: "data",
      x: 250,
      y: 230,
      w: 240,
      h: 88,
      purpose:
        "Turns a sentence into a 384-dimensional vector so that similar meanings land close together, whatever the exact wording.",
      input: "Normalised question text.",
      output: "A 384-d L2-normalised float vector.",
      algorithm:
        "sentence-transformers/all-MiniLM-L6-v2, run locally through fastembed (ONNX, CPU). Chosen in an embedding bake-off on dev.",
      keyMetric:
        "Same encoder family powers A1's classifier and the pgvector index, so routing and retrieval agree on what 'similar' means.",
      files: ["nlp/embed.py", "outputs/models/embed_bakeoff.json"],
    },
    {
      id: "pgvector",
      kicker: "Retrieval",
      title: "Supabase pgvector",
      subtitle: "HNSW + full-text · RRF",
      tone: "data",
      x: 540,
      y: 230,
      w: 250,
      h: 88,
      purpose: `Holds ${
        metrics.trainRows !== null
          ? `all ${formatCount(metrics.trainRows)} training examples`
          : "all training examples"
      } and returns the closest past questions and replies for a new query.`,
      input: "Query embedding + sanitised query text + optional intent filter.",
      output:
        '{\n  "id": 8121,\n  "intent": "track_refund",\n  "instruction": "…",\n  "response": "…",\n  "similarity": 0.91,\n  "rrf_score": 0.0328\n}',
      algorithm:
        "Two legs: HNSW cosine search (m = 16, ef_construction = 64) and Postgres full-text search over a GIN-indexed tsvector. Results are fused with reciprocal-rank fusion (k = 60) by match_hybrid().",
      keyMetric:
        "Filtering by A1's intent happens inside the same SQL query, so retrieval never wanders into another topic.",
      files: ["supabase/schema.sql", "scripts/ingest.py", "api/main.py"],
    },
    {
      id: "memory",
      kicker: "Fallback",
      title: "In-memory index",
      subtitle: "no network needed",
      tone: "data",
      x: 562,
      y: 364,
      w: 206,
      h: 52,
      purpose:
        "Keeps the demo working when Supabase is unreachable, slow, or not configured at all.",
      input: "Query embedding.",
      output:
        '{\n  "response": "…",\n  "supporting_examples": […],\n  "retrieval_score": 0.84\n}',
      algorithm:
        "A single matrix multiply against the cached train embedding matrix, followed by the same intent filter and self-check path as pgvector.",
      keyMetric: memoryMetric,
      files: ["nlp/a2.py", "outputs/models/a2_specialist.joblib"],
    },
  ];
}

const NODE_BY_ID = (nodes: DiagramNode[], id: string) =>
  nodes.find((node) => node.id === id);

export function ArchitectureDiagram({
  metrics,
}: {
  metrics: ArchitectureMetrics;
}) {
  const nodes = buildNodes(metrics);
  const [openId, setOpenId] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);
  const closeRef = useRef<HTMLButtonElement>(null);

  const openNode = openId ? NODE_BY_ID(nodes, openId) : null;

  useEffect(() => {
    if (!openId) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpenId(null);
    };
    document.addEventListener("keydown", onKeyDown);
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeRef.current?.focus();
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.body.style.overflow = previous;
    };
  }, [openId]);

  const downloadPng = () => {
    setExporting(true);
    try {
      const canvas = renderDiagramCanvas(nodes);
      canvas.toBlob((blob) => {
        if (!blob) {
          setExporting(false);
          return;
        }
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = "resolveiq-architecture.png";
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
        setExporting(false);
      }, "image/png");
    } catch {
      setExporting(false);
    }
  };

  return (
    <>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs text-ink-secondary">
          Click any box for its purpose, input, output and the file that
          implements it.
          <span className="sm:hidden"> Scroll sideways to see every box.</span>
        </p>
        <button
          type="button"
          onClick={downloadPng}
          disabled={exporting}
          className="inline-flex items-center gap-1.5 rounded-pill border border-line bg-card px-3 py-1.5 text-xs font-medium text-ink transition-colors hover:border-brand/30 hover:text-brand-ink disabled:opacity-60"
        >
          <Download size={13} strokeWidth={1.9} aria-hidden="true" />
          {exporting ? "Preparing PNG…" : "Download PNG"}
        </button>
      </div>

      <div className="mt-3 overflow-x-auto rounded-card border border-line bg-card p-3 sm:p-4">
        <div className="relative mx-auto min-w-[760px] max-w-[1000px]">
          <div
            className="relative w-full"
            style={{ aspectRatio: `${CANVAS_WIDTH} / ${CANVAS_HEIGHT}` }}
          >
            <svg
              viewBox={`0 0 ${CANVAS_WIDTH} ${CANVAS_HEIGHT}`}
              aria-hidden="true"
              className="absolute inset-0 size-full"
            >
              <defs>
                <marker
                  id="arch-arrow"
                  viewBox="0 0 10 10"
                  refX="8"
                  refY="5"
                  markerWidth="6"
                  markerHeight="6"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="#A1A1AA" />
                </marker>
              </defs>
              {EDGES.map((edge, index) => {
                const points = edge.points
                  .map(([x, y]) => `${x},${y}`)
                  .join(" ");
                return (
                  <polyline
                    key={index}
                    points={points}
                    fill="none"
                    stroke="#C4C4C8"
                    strokeWidth={1.4}
                    strokeDasharray={edge.dashed ? "4 4" : undefined}
                    markerEnd="url(#arch-arrow)"
                  />
                );
              })}
              <text
                x={16}
                y={16}
                fill="#A1A1AA"
                fontSize={9}
                letterSpacing="1.6"
                fontFamily="Inter, sans-serif"
                fontWeight={600}
              >
                RESOLVEIQ · TWO-AGENT PIPELINE
              </text>
              <text
                x={676}
                y={344}
                fill="#71717A"
                fontSize={9.5}
                fontFamily="Inter, sans-serif"
              >
                falls back to in-memory
              </text>
            </svg>

            {nodes.map((node) => {
              const tone = TONE[node.tone];
              return (
                <button
                  key={node.id}
                  type="button"
                  onClick={() => setOpenId(node.id)}
                  aria-haspopup="dialog"
                  title={`${node.title} — open details`}
                  className={cn(
                    "group absolute flex flex-col justify-center overflow-hidden rounded-[12px] border bg-card px-3 text-left transition-all",
                    "hover:-translate-y-0.5 hover:shadow-float focus:outline-none focus-visible:ring-2 focus-visible:ring-brand/40"
                  )}
                  style={{
                    left: `${(node.x / CANVAS_WIDTH) * 100}%`,
                    top: `${(node.y / CANVAS_HEIGHT) * 100}%`,
                    width: `${(node.w / CANVAS_WIDTH) * 100}%`,
                    height: `${(node.h / CANVAS_HEIGHT) * 100}%`,
                    borderColor: tone.border,
                    backgroundColor: tone.fill,
                  }}
                >
                  <span
                    aria-hidden="true"
                    className="absolute inset-y-2 left-0 w-[2.5px] rounded-r-pill"
                    style={{ backgroundColor: tone.accent }}
                  />
                  <span className="text-[9px] font-semibold uppercase tracking-[0.14em] text-ink-faint">
                    {node.kicker}
                  </span>
                  <span
                    className={cn(
                      "mt-0.5 line-clamp-2 text-[11.5px] font-semibold leading-[14px] text-ink",
                      node.id === "handoff" && "font-mono text-[10.5px]"
                    )}
                  >
                    {node.title}
                  </span>
                  <span className="mt-0.5 line-clamp-2 text-[9.5px] leading-[12px] text-ink-secondary">
                    {node.subtitle}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      <p className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-ink-faint">
        <span className="inline-flex items-center gap-1.5">
          <span className="h-[2px] w-4 rounded-pill bg-[#C4C4C8]" />
          data flow
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-[2px] w-4 rounded-pill border-t border-dashed border-[#C4C4C8]" />
          automatic fallback
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="size-2 rounded-sm bg-a1" />
          A1 Intent Analyst
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="size-2 rounded-sm bg-a2" />
          A2 Response Specialist
        </span>
      </p>

      {openNode ? (
        <div className="fixed inset-0 z-50 flex justify-end">
          <button
            type="button"
            aria-label="Close details"
            onClick={() => setOpenId(null)}
            className="absolute inset-0 bg-ink/25"
          />
          <aside
            role="dialog"
            aria-modal="true"
            aria-labelledby="arch-drawer-title"
            className="relative z-10 flex h-full w-full max-w-[440px] flex-col border-l border-line bg-card shadow-float"
          >
            <div className="flex items-start justify-between gap-3 border-b border-line px-4 py-3.5">
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-ink-faint">
                  {openNode.kicker}
                </p>
                <h2
                  id="arch-drawer-title"
                  className="mt-1 text-base font-semibold text-ink"
                >
                  {openNode.title}
                </h2>
                <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                  <Chip tone={TONE[openNode.tone].chip}>
                    {openNode.subtitle}
                  </Chip>
                </div>
              </div>
              <button
                ref={closeRef}
                type="button"
                onClick={() => setOpenId(null)}
                aria-label="Close details"
                className="inline-flex size-8 shrink-0 items-center justify-center rounded-full text-ink-secondary transition-colors hover:bg-soft hover:text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-brand/40"
              >
                <X size={16} strokeWidth={1.8} aria-hidden="true" />
              </button>
            </div>

            <div className="flex-1 space-y-4 overflow-y-auto px-4 py-4">
              <section>
                <h3 className="text-xs font-semibold uppercase tracking-[0.08em] text-ink-secondary">
                  Purpose
                </h3>
                <p className="mt-1.5 text-xs leading-[19px] text-ink">
                  {openNode.purpose}
                </p>
              </section>

              <section>
                <h3 className="text-xs font-semibold uppercase tracking-[0.08em] text-ink-secondary">
                  Input
                </h3>
                <CodeBlock>{openNode.input}</CodeBlock>
              </section>

              <section>
                <h3 className="text-xs font-semibold uppercase tracking-[0.08em] text-ink-secondary">
                  Output
                </h3>
                <CodeBlock>{openNode.output}</CodeBlock>
              </section>

              <section>
                <h3 className="text-xs font-semibold uppercase tracking-[0.08em] text-ink-secondary">
                  Algorithm
                </h3>
                <p className="mt-1.5 text-xs leading-[19px] text-ink">
                  {openNode.algorithm}
                </p>
              </section>

              <section className="rounded-[10px] border border-line bg-soft px-3 py-2.5">
                <h3 className="text-xs font-semibold uppercase tracking-[0.08em] text-ink-secondary">
                  Key metric
                </h3>
                <p className="mt-1 text-xs leading-[19px] text-ink">
                  {openNode.keyMetric}
                </p>
              </section>

              <section>
                <h3 className="text-xs font-semibold uppercase tracking-[0.08em] text-ink-secondary">
                  In the repo
                </h3>
                <ul className="mt-1.5 space-y-1">
                  {openNode.files.map((file) => (
                    <li
                      key={file}
                      className="break-all rounded-[8px] border border-line bg-card px-2 py-1 font-mono text-[11px] text-ink-secondary"
                    >
                      {file}
                    </li>
                  ))}
                </ul>
              </section>
            </div>

            <div className="border-t border-line px-4 py-3">
              <button
                type="button"
                onClick={() => setOpenId(null)}
                className="inline-flex w-full items-center justify-center gap-1.5 rounded-pill border border-line bg-card px-3 py-1.5 text-xs font-medium text-ink transition-colors hover:bg-soft"
              >
                Close
                <ArrowRight size={12} aria-hidden="true" />
              </button>
            </div>
          </aside>
        </div>
      ) : null}
    </>
  );
}

function CodeBlock({ children }: { children: string }) {
  return (
    <pre className="mt-1.5 max-h-[220px] overflow-auto whitespace-pre rounded-[10px] border border-line bg-soft px-2.5 py-2 font-mono text-[11px] leading-[16px] text-ink-secondary">
      {children}
    </pre>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   PNG export. Drawn by hand on a <canvas> so the download needs no
   extra dependency and matches the on-page diagram geometry.
   ═══════════════════════════════════════════════════════════════════ */

function renderDiagramCanvas(nodes: DiagramNode[]): HTMLCanvasElement {
  const scale = 2;
  const canvas = document.createElement("canvas");
  canvas.width = CANVAS_WIDTH * scale;
  canvas.height = CANVAS_HEIGHT * scale;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas 2D unavailable");
  ctx.scale(scale, scale);

  ctx.fillStyle = "#FBFBFA";
  ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

  ctx.fillStyle = "#A1A1AA";
  ctx.font = "600 9px Inter, system-ui, sans-serif";
  ctx.fillText("RESOLVEIQ · TWO-AGENT PIPELINE", 16, 18);

  for (const edge of EDGES) {
    drawEdge(ctx, edge);
  }

  for (const node of nodes) {
    drawNode(ctx, node);
  }

  return canvas;
}

function drawEdge(ctx: CanvasRenderingContext2D, edge: Edge) {
  ctx.save();
  ctx.strokeStyle = "#C4C4C8";
  ctx.lineWidth = 1.4;
  ctx.setLineDash(edge.dashed ? [4, 4] : []);
  ctx.beginPath();
  edge.points.forEach(([x, y], index) => {
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.setLineDash([]);

  const [x1, y1] = edge.points[edge.points.length - 2];
  const [x2, y2] = edge.points[edge.points.length - 1];
  const angle = Math.atan2(y2 - y1, x2 - x1);
  ctx.fillStyle = "#A1A1AA";
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(
    x2 - 6 * Math.cos(angle - Math.PI / 7),
    y2 - 6 * Math.sin(angle - Math.PI / 7)
  );
  ctx.lineTo(
    x2 - 6 * Math.cos(angle + Math.PI / 7),
    y2 - 6 * Math.sin(angle + Math.PI / 7)
  );
  ctx.closePath();
  ctx.fill();

  if (edge.label) {
    ctx.fillStyle = "#71717A";
    ctx.font = "400 9.5px Inter, system-ui, sans-serif";
    ctx.fillText(edge.label, 676, 344);
  }
  ctx.restore();
}

function drawNode(ctx: CanvasRenderingContext2D, node: DiagramNode) {
  const tone = TONE[node.tone];
  const radius = 12;
  roundRect(ctx, node.x, node.y, node.w, node.h, radius);
  ctx.fillStyle = tone.fill;
  ctx.fill();
  ctx.strokeStyle = tone.border;
  ctx.lineWidth = 1;
  ctx.stroke();

  ctx.save();
  roundRect(ctx, node.x, node.y + 8, 2.5, node.h - 16, 1.5);
  ctx.fillStyle = tone.accent;
  ctx.fill();
  ctx.restore();

  const textX = node.x + 12;
  const maxWidth = node.w - 22;

  ctx.fillStyle = "#71717A";
  ctx.font = "600 9px Inter, system-ui, sans-serif";
  ctx.fillText(node.kicker.toUpperCase(), textX, node.y + 22);

  ctx.fillStyle = "#18181B";
  ctx.font =
    node.id === "handoff"
      ? "600 10.5px 'JetBrains Mono', ui-monospace, monospace"
      : "600 11.5px Inter, system-ui, sans-serif";
  const titleLines = wrapText(
    ctx,
    node.title,
    textX,
    node.y + 37,
    maxWidth,
    13,
    2
  );

  ctx.fillStyle = "#71717A";
  ctx.font = "400 9.5px Inter, system-ui, sans-serif";
  wrapText(
    ctx,
    node.subtitle,
    textX,
    node.y + 37 + titleLines * 13 + 2,
    maxWidth,
    11.5,
    2
  );
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number
) {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + width, y, x + width, y + height, r);
  ctx.arcTo(x + width, y + height, x, y + height, r);
  ctx.arcTo(x, y + height, x, y, r);
  ctx.arcTo(x, y, x + width, y, r);
  ctx.closePath();
}

function wrapText(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  maxWidth: number,
  lineHeight: number,
  maxLines: number
) {
  const words = text.split(" ");
  let line = "";
  let lineIndex = 0;
  for (const word of words) {
    const candidate = line ? `${line} ${word}` : word;
    if (ctx.measureText(candidate).width > maxWidth && line) {
      ctx.fillText(line, x, y + lineIndex * lineHeight);
      line = word;
      lineIndex += 1;
      if (lineIndex >= maxLines) return lineIndex;
    } else {
      line = candidate;
    }
  }
  ctx.fillText(line, x, y + lineIndex * lineHeight);
  return lineIndex + 1;
}
