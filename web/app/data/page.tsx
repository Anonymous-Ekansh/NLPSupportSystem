import {
  Boxes,
  Database,
  Layers,
  Tags,
} from "lucide-react";
import type { MapPointsFile } from "@/lib/api";
import {
  type BrowseExample,
  type DatasetStats,
  type ExamplesFile,
  type MapPointWithQuery,
} from "@/lib/data";
import { readDataJson } from "@/lib/server/data";
import { DataExplorer } from "@/components/data/DataExplorer";
import { StatCard } from "@/components/ui/StatCard";
import { EmptyDataFile } from "@/components/ui/EmptyDataFile";
import { formatCount } from "@/lib/utils";

export const metadata = {
  title: "Data explorer",
  description:
    "Explore the support questions, intents and semantic map behind ResolveIQ.",
};

export default function DataPage() {
  const dataset = readDataJson<DatasetStats>("dataset_stats.json");
  const map = readDataJson<MapPointsFile>("map_points.json");
  const examplesFile = readDataJson<ExamplesFile>("examples.json");

  const points = (map?.coordinates ?? []) as MapPointWithQuery[];
  const examples: BrowseExample[] = examplesFile?.browse ?? [];

  return (
    <div className="mx-auto max-w-[1040px]">
      <header>
        <p className="text-xs font-medium uppercase tracking-[0.14em] text-ink-secondary">
          Data explorer
        </p>
        <h1 className="mt-2 text-xl font-semibold tracking-tight text-ink sm:text-display sm:leading-[48px]">
          What the assistant learned from
        </h1>
        <p className="mt-2 max-w-[680px] text-base text-ink-secondary">
          Every answer starts from a real past support question. Explore the
          categories, intents and 2D neighbourhoods A2 searches.
        </p>
      </header>

      {dataset ? (
        <>
          <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard
              label="Total examples"
              value={formatCount(dataset.total_examples)}
              caption="Clean question / answer pairs."
              icon={Database}
            />
            <StatCard
              label="Intents"
              value={String(dataset.intents)}
              caption="Distinct support topics."
              icon={Tags}
              tone="a1"
            />
            <StatCard
              label="Categories"
              value={String(dataset.categories)}
              caption="Higher-level topic groups."
              icon={Boxes}
              tone="a2"
            />
            <StatCard
              label="Train / Dev / Test"
              value={`${formatCount(dataset.splits.train)} / ${formatCount(
                dataset.splits.dev
              )} / ${formatCount(dataset.splits.test)}`}
              caption="Held-out test never seen in training."
              icon={Layers}
              tone="success"
              className="[&_p:nth-of-type(2)]:text-base"
            />
          </div>

          <p className="mt-3 text-xs leading-[18px] text-ink-secondary">
            Raw {formatCount(dataset.cleaning.raw_train_rows)} train +{" "}
            {formatCount(dataset.cleaning.raw_validation_rows)} validation rows
            → {formatCount(dataset.cleaning.duplicates_removed)} duplicates and{" "}
            {formatCount(dataset.cleaning.leaked_rows_removed)} leaked rows
            removed → stratified, reproducible split (seed{" "}
            {dataset.cleaning.seed}).
          </p>
        </>
      ) : (
        <div className="mt-6">
          <EmptyDataFile
            file="dataset_stats.json"
            detail="Add dataset_stats.json to web/public/data/ to populate the corpus cards."
          />
        </div>
      )}

      <div className="mt-4">
        <DataExplorer dataset={dataset} mapPoints={points} examples={examples} />
      </div>
    </div>
  );
}
