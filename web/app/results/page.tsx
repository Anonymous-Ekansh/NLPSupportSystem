import type {
  ConfusionCategoryFile,
  DatasetStats,
  IntentMetric,
  LadderFile,
  LadderRow,
  OodFile,
  PerIntentFile,
  ResultsData,
  RetrievalModesFile,
} from "@/lib/data";
import { metricsOf, rowsOf } from "@/lib/data";
import { readDataJson } from "@/lib/server/data";
import { ResultsDashboard } from "@/components/results/ResultsDashboard";

export const metadata = {
  title: "Results",
  description:
    "Held-out evaluation results for ResolveIQ's two agents: intent routing, answer relevance and out-of-scope detection.",
};

export default function ResultsPage() {
  const results = readDataJson<ResultsData>("results.json");
  const perIntentFile = readDataJson<PerIntentFile | IntentMetric[]>(
    "per_intent_f1.json"
  );
  const confusion = readDataJson<ConfusionCategoryFile>(
    "confusion_category.json"
  );
  const ladderFile = readDataJson<LadderFile | LadderRow[]>("ladder.json");
  const retrieval = readDataJson<RetrievalModesFile>("retrieval_modes.json");
  const ood = readDataJson<OodFile>("ood.json");

  const nTest = results?.n_test ?? readDataJson<DatasetStats>("dataset_stats.json")?.splits.test;

  return (
    <div className="mx-auto max-w-[1040px]">
      <header>
        <p className="text-xs font-medium uppercase tracking-[0.14em] text-ink-secondary">
          Results
        </p>
        <h1 className="mt-2 text-xl font-semibold tracking-tight text-ink sm:text-display sm:leading-[48px]">
          How well it works
        </h1>
        <p className="mt-2 max-w-[700px] text-base text-ink-secondary">
          Every number below is measured on a held-out test split that neither
          agent saw during training
          {nTest ? ` (${nTest.toLocaleString("en-US")} questions)` : ""}. No
          cherry-picked examples.
        </p>
      </header>

      <ResultsDashboard
        results={results}
        perIntent={metricsOf(perIntentFile)}
        confusion={confusion}
        ladder={rowsOf(ladderFile)}
        retrieval={retrieval}
        ood={ood}
      />
    </div>
  );
}
