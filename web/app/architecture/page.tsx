import type {
  ArchitectureMetrics,
} from "@/components/architecture/ArchitectureDiagram";
import { ArchitectureDiagram } from "@/components/architecture/ArchitectureDiagram";
import { SampleTraces } from "@/components/architecture/SampleTraces";
import { Card } from "@/components/ui/Card";
import { Chip } from "@/components/ui/Chip";
import type {
  DatasetStats,
  ExamplesFile,
  ResultsData,
  TraceExample,
} from "@/lib/data";
import { examplesOf } from "@/lib/data";
import { readDataJson } from "@/lib/server/data";

export const metadata = {
  title: "Architecture",
  description:
    "How ResolveIQ turns a customer question into a checked answer: A1 Intent Analyst, a handoff JSON, then A2 Response Specialist.",
};

const TECH_STACK = [
  "Next.js 14",
  "TypeScript",
  "Tailwind CSS",
  "FastAPI",
  "sentence-transformers",
  "scikit-learn",
  "Supabase pgvector",
  "Render",
  "Vercel",
];

function steps(dataset: DatasetStats | null) {
  const intents = dataset ? `${dataset.intents} intents` : "the support intents";
  const categories = dataset
    ? `${dataset.categories} categories`
    : "the support categories";
  return [
    {
      title: "Normalise the question",
      body: "Lowercase, trim and keep {{customer-fill}} placeholders intact. No keywords are extracted and no menu is shown.",
      agent: null,
    },
    {
      title: "A1 names the intent",
      body: `The question is embedded with all-MiniLM-L6-v2, classified into one of ${intents}, then mapped to one of ${categories}. A1 also returns confidence, margin and its three best guesses.`,
      agent: "A1",
    },
    {
      title: "A1 hands off a small JSON contract",
      body: "The handoff carries the query, intent, category, confidence and top-3. If confidence is below tau or the nearest training question is below s_min, it is marked needs_clarification and A2 searches all three hypotheses.",
      agent: "handoff",
    },
    {
      title: "A2 retrieves and checks",
      body: "A2 searches past support replies — Supabase pgvector (HNSW + full-text, hybrid RRF) when reachable, otherwise the in-memory matrix. Candidates are re-ranked and the first reply that passes the placeholder, source and intent checks wins.",
      agent: "A2",
    },
    {
      title: "The orchestrator returns the answer",
      body: "The customer sees the reply, its sources and a readable trace: A1's real confidence and top-3, A2's retrieval score and the three self-checks. Out-of-scope questions are withheld rather than guessed.",
      agent: null,
    },
  ] as const;
}

export default function ArchitecturePage() {
  const results = readDataJson<ResultsData>("results.json");
  const dataset = readDataJson<DatasetStats>("dataset_stats.json");
  const examplesFile = readDataJson<ExamplesFile | TraceExample[]>(
    "examples.json"
  );
  const examples = examplesOf(examplesFile);

  const metrics: ArchitectureMetrics = {
    intents: dataset?.intents ?? null,
    categories: dataset?.categories ?? null,
    macroF1: results?.intent_macro_f1 ?? null,
    categoryAccuracy: results?.category_accuracy ?? null,
    answerCosine: results?.answer_relevance.cosine_sim ?? null,
    selfCheckPass: results?.answer_checks.pass_rate ?? null,
    nTest: results?.n_test ?? null,
    trainRows: dataset?.splits.train ?? null,
  };
  const QUICK_STEPS = steps(dataset);

  return (
    <div className="mx-auto max-w-[1040px]">
      <header>
        <p className="text-xs font-medium uppercase tracking-[0.14em] text-ink-secondary">
          Architecture
        </p>
        <h1 className="mt-2 text-xl font-semibold tracking-tight text-ink sm:text-display sm:leading-[48px]">
          How it works
        </h1>
        <p className="mt-2 max-w-[720px] text-base text-ink-secondary">
          Two agents, one explicit handoff. A1 decides what the question is
          about; A2 finds a real past reply and checks it before the customer
          sees anything.
        </p>
      </header>

      <section aria-labelledby="diagram-heading" className="mt-6">
        <h2 id="diagram-heading" className="sr-only">
          Two-agent pipeline diagram
        </h2>
        <ArchitectureDiagram metrics={metrics} />
      </section>

      <section aria-labelledby="stack-heading" className="mt-6">
        <h2
          id="stack-heading"
          className="text-xs font-semibold uppercase tracking-[0.08em] text-ink-secondary"
        >
          Tech stack
        </h2>
        <div className="mt-2 flex flex-wrap gap-1.5">
          {TECH_STACK.map((item) => (
            <Chip key={item} tone="outline">
              {item}
            </Chip>
          ))}
        </div>
      </section>

      <section aria-labelledby="steps-heading" className="mt-6">
        <Card className="p-4 sm:p-5">
          <h2 id="steps-heading" className="text-base font-semibold text-ink">
            What happens to one question
          </h2>
          <p className="mt-0.5 text-xs leading-[18px] text-ink-secondary">
            The same five steps the trace shows, without the JSON.
          </p>

          <ol className="mt-4 space-y-3">
            {QUICK_STEPS.map((step, index) => (
              <li key={step.title} className="flex gap-3">
                <span
                  aria-hidden="true"
                  className="mt-0.5 flex size-6 shrink-0 items-center justify-center rounded-full border border-line bg-soft font-mono text-[11px] font-medium text-ink-secondary"
                >
                  {index + 1}
                </span>
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-1.5">
                    <h3 className="text-sm font-medium text-ink">
                      {step.title}
                    </h3>
                    {step.agent === "A1" ? (
                      <Chip tone="a1">A1</Chip>
                    ) : step.agent === "A2" ? (
                      <Chip tone="a2">A2</Chip>
                    ) : step.agent === "handoff" ? (
                      <Chip tone="neutral">handoff</Chip>
                    ) : null}
                  </div>
                  <p className="mt-0.5 text-xs leading-[18px] text-ink-secondary">
                    {step.body}
                  </p>
                </div>
              </li>
            ))}
          </ol>
        </Card>
      </section>

      <SampleTraces
        traces={examples.traces}
        fileMissing={examplesFile === null}
      />
    </div>
  );
}
