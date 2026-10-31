import Link from "next/link";
import { FileWarning } from "lucide-react";
import { readEvidenceJson } from "@/lib/server/data";
import { RobustnessDashboard } from "@/components/robustness/RobustnessDashboard";
import type {
  L2DecisionsFile,
  RobustnessFile,
  TraceExamplesFile,
} from "@/lib/robustness";

export const metadata = {
  title: "Robustness",
  description:
    "How ResolveIQ behaves on noisy, ambiguous and unsupported input: held-out accuracy by variation type, decision distribution, selective prediction and Day 1 vs Level 2.",
};

export default function RobustnessPage() {
  const robustness = readEvidenceJson<RobustnessFile>("robustness.json");
  const decisions = readEvidenceJson<L2DecisionsFile>("l2_decisions.json");
  const traces = readEvidenceJson<TraceExamplesFile>("trace_examples.json");

  return (
    <div className="mx-auto max-w-[1040px]">
      <header>
        <p className="text-xs font-medium uppercase tracking-[0.14em] text-ink-secondary">
          Robustness
        </p>
        <h1 className="mt-2 text-xl font-semibold tracking-tight text-ink sm:text-display sm:leading-[48px]">
          Where it holds, and where it breaks
        </h1>
        <p className="mt-2 max-w-[720px] text-base text-ink-secondary">
          Spelling errors, abbreviations, colloquial phrasing and the dataset&apos;s
          own variation groups — measured on held-out questions, reported with
          sample sizes and without tuning on the test split.
        </p>
      </header>

      <div className="mt-6">
        {robustness ? (
          <RobustnessDashboard
            robustness={robustness}
            decisions={decisions}
            traces={traces}
          />
        ) : (
          <MissingEvidence />
        )}
      </div>
    </div>
  );
}

function MissingEvidence() {
  return (
    <div className="rounded-card border border-dashed border-line bg-soft px-4 py-10 text-center">
      <span className="mx-auto flex size-10 items-center justify-center rounded-full border border-line bg-card text-ink-faint">
        <FileWarning size={18} aria-hidden="true" />
      </span>
      <p className="mt-3 text-sm font-medium text-ink">
        No robustness evidence yet
      </p>
      <p className="mx-auto mt-1 max-w-[420px] text-xs leading-[18px] text-ink-secondary">
        Generate it with{" "}
        <code className="rounded-[4px] border border-line bg-card px-1 py-0.5 font-mono text-[11px]">
          python scripts/eval_l2.py --sync
        </code>
        . This page reads{" "}
        <code className="font-mono text-[11px]">
          public/evidence/robustness.json
        </code>{" "}
        and never invents numbers.
      </p>
      <Link
        href="/results"
        className="mt-3 inline-flex items-center rounded-pill border border-line bg-card px-3 py-1.5 text-xs font-medium text-ink-secondary transition-colors hover:text-ink"
      >
        See the Day 1 results instead
      </Link>
    </div>
  );
}
