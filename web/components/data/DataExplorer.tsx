"use client";

import { useMemo, useState } from "react";
import { ArrowUpDown, Search } from "lucide-react";
import type {
  BrowseExample,
  DatasetStats,
  MapPointWithQuery,
} from "@/lib/data";
import { CATEGORY_COLORS, CATEGORY_LABELS } from "@/components/trace/ScatterMap";
import { Card } from "@/components/ui/Card";
import { Chip } from "@/components/ui/Chip";
import { EmptyDataFile } from "@/components/ui/EmptyDataFile";
import { cn, formatCount, humaniseIntent } from "@/lib/utils";

const STYLE_COLORS: Record<string, string> = {
  Colloquial: "#64748B",
  Interrogative: "#7C8FA8",
  "Noisy / misspelled": "#A78B6F",
  Morphological: "#6B8E8A",
  Polite: "#8B7E9E",
  Other: "#A39A6A",
};

type CategorySort = "intents" | "examples" | "name";
type BalanceSort = "low" | "high";

export function DataExplorer({
  dataset,
  mapPoints,
  examples,
}: {
  dataset: DatasetStats | null;
  mapPoints: MapPointWithQuery[];
  examples: BrowseExample[];
}) {
  return (
    <div className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-2">
        {dataset ? (
          <CategoryBars dataset={dataset} />
        ) : (
          <EmptyDataFile file="dataset_stats.json" />
        )}
        {dataset ? (
          <StyleDonut dataset={dataset} />
        ) : (
          <EmptyDataFile file="dataset_stats.json" />
        )}
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        {dataset ? (
          <ClassBalance dataset={dataset} />
        ) : (
          <EmptyDataFile file="dataset_stats.json" />
        )}
        <EmbeddingMap points={mapPoints} />
      </div>

      {examples.length > 0 ? (
        <BrowseExamples examples={examples} />
      ) : (
        <EmptyDataFile
          file="examples.json"
          detail="Add examples.json with a `browse` array to enable the searchable table."
        />
      )}
    </div>
  );
}

function CategoryBars({ dataset }: { dataset: DatasetStats }) {
  const [sort, setSort] = useState<CategorySort>("intents");

  const rows = useMemo(() => {
    const byCategory = new Map<
      string,
      { category: string; examples: number; intents: Set<string> }
    >();

    for (const row of dataset.class_balance) {
      const existing = byCategory.get(row.category) ?? {
        category: row.category,
        examples: 0,
        intents: new Set<string>(),
      };
      existing.examples += row.count;
      existing.intents.add(row.intent);
      byCategory.set(row.category, existing);
    }

    const list = Array.from(byCategory.values()).map((row) => ({
      category: row.category,
      examples: row.examples,
      intents: row.intents.size,
    }));

    if (sort === "name") {
      return list.sort((a, b) => a.category.localeCompare(b.category));
    }
    if (sort === "examples") {
      return list.sort((a, b) => b.examples - a.examples);
    }
    return list.sort((a, b) => b.intents - a.intents || b.examples - a.examples);
  }, [dataset.class_balance, sort]);

  const maxIntents = Math.max(...rows.map((row) => row.intents), 1);

  return (
    <Card className="p-4">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h2 className="text-base font-semibold text-ink">
            Intents per category
          </h2>
          <p className="mt-0.5 text-xs leading-[18px] text-ink-secondary">
            How the {dataset.intents} support intents group into{" "}
            {dataset.categories} categories.
          </p>
        </div>
        <SortButton
          value={sort}
          onChange={setSort}
          options={[
            { value: "intents", label: "Most intents" },
            { value: "examples", label: "Most examples" },
            { value: "name", label: "A–Z" },
          ]}
        />
      </div>

      <ul className="mt-3 space-y-2.5">
        {rows.map((row) => (
          <li key={row.category}>
            <div className="flex items-center justify-between gap-2">
              <span className="inline-flex min-w-0 items-center gap-1.5 text-xs text-ink-secondary">
                <span
                  aria-hidden="true"
                  className="size-1.5 shrink-0 rounded-full"
                  style={{
                    backgroundColor: CATEGORY_COLORS[row.category] ?? "#A1A1AA",
                  }}
                />
                <span className="truncate">
                  {CATEGORY_LABELS[row.category] ?? row.category}
                </span>
              </span>
              <span className="shrink-0 font-mono text-xs tabular-nums text-ink">
                {row.intents}{" "}
                <span className="text-ink-secondary">
                  · {formatCount(row.examples)}
                </span>
              </span>
            </div>
            <div className="mt-1 h-1.5 overflow-hidden rounded-pill bg-[#F0F0ED]">
              <div
                className="h-full rounded-pill"
                style={{
                  width: `${(row.intents / maxIntents) * 100}%`,
                  backgroundColor:
                    CATEGORY_COLORS[row.category] ?? "#A1A1AA",
                  opacity: 0.82,
                }}
              />
            </div>
          </li>
        ))}
      </ul>
      <p className="mt-3 border-t border-line pt-2.5 text-xs leading-[18px] text-ink-secondary">
        Bar = intents. The smaller number is examples in the full corpus.
      </p>
    </Card>
  );
}

function StyleDonut({ dataset }: { dataset: DatasetStats }) {
  const total = dataset.styles.reduce((sum, row) => sum + row.count, 0) || 1;
  const radius = 60;
  const circumference = 2 * Math.PI * radius;
  let offset = 0;

  return (
    <Card className="flex h-full flex-col p-4">
      <h2 className="text-base font-semibold text-ink">Style variations</h2>
      <p className="mt-0.5 text-xs leading-[18px] text-ink-secondary">
        One primary style per example. The raw dataset tags can overlap.
      </p>

      <div className="mt-3 flex flex-1 flex-col items-center justify-center gap-4 sm:flex-row">
        <div className="relative shrink-0">
          <svg
            viewBox="0 0 160 160"
            className="size-[160px] -rotate-90"
            role="img"
            aria-label="Donut chart of dataset style variations"
          >
            <circle
              cx="80"
              cy="80"
              r={radius}
              fill="none"
              stroke="#F0F0ED"
              strokeWidth="18"
            />
            {dataset.styles.map((row) => {
              const length = (row.count / total) * circumference;
              const dash = `${length} ${circumference - length}`;
              const circle = (
                <circle
                  key={row.style}
                  cx="80"
                  cy="80"
                  r={radius}
                  fill="none"
                  stroke={STYLE_COLORS[row.style] ?? "#A1A1AA"}
                  strokeWidth="18"
                  strokeDasharray={dash}
                  strokeDashoffset={-offset}
                  strokeLinecap="butt"
                >
                  <title>{`${row.style}: ${formatCount(row.count)} examples`}</title>
                </circle>
              );
              offset += length;
              return circle;
            })}
          </svg>
          <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-base font-semibold tabular-nums text-ink">
              {formatCount(total)}
            </span>
            <span className="text-xs text-ink-secondary">examples</span>
          </div>
        </div>

        <ul className="w-full space-y-1.5">
          {dataset.styles.map((row) => (
            <li
              key={row.style}
              className="flex items-center justify-between gap-2 text-xs"
            >
              <span className="inline-flex min-w-0 items-center gap-1.5 text-ink-secondary">
                <span
                  aria-hidden="true"
                  className="size-1.5 shrink-0 rounded-full"
                  style={{
                    backgroundColor: STYLE_COLORS[row.style] ?? "#A1A1AA",
                  }}
                />
                <span className="truncate">{row.style}</span>
              </span>
              <span className="shrink-0 font-mono tabular-nums text-ink">
                {(row.pct * 100).toFixed(1)}%
              </span>
            </li>
          ))}
        </ul>
      </div>

      {dataset.style_note ? (
        <p className="mt-3 border-t border-line pt-2.5 text-xs leading-[18px] text-ink-secondary">
          {dataset.style_note}
        </p>
      ) : null}
    </Card>
  );
}

function ClassBalance({ dataset }: { dataset: DatasetStats }) {
  const [sort, setSort] = useState<BalanceSort>("high");
  const rows = useMemo(() => {
    const sorted = [...dataset.class_balance].sort((a, b) =>
      sort === "high" ? b.count - a.count : a.count - b.count
    );
    return sorted;
  }, [dataset.class_balance, sort]);
  const max = Math.max(...rows.map((row) => row.count), 1);

  return (
    <Card className="p-4">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h2 className="text-base font-semibold text-ink">Class balance</h2>
          <p className="mt-0.5 text-xs leading-[18px] text-ink-secondary">
            Examples per intent across train, dev and test.
          </p>
        </div>
        <SortButton
          value={sort}
          onChange={setSort}
          options={[
            { value: "high", label: "Largest first" },
            { value: "low", label: "Smallest first" },
          ]}
        />
      </div>

      <div className="mt-3 max-h-[320px] space-y-1.5 overflow-y-auto pr-1">
        {rows.map((row) => (
          <div key={row.intent}>
            <div className="flex items-center justify-between gap-2">
              <span className="truncate font-mono text-xs text-ink-secondary">
                {row.intent}
              </span>
              <span className="shrink-0 font-mono text-xs tabular-nums text-ink">
                {formatCount(row.count)}
              </span>
            </div>
            <div className="mt-0.5 h-1 overflow-hidden rounded-pill bg-[#F0F0ED]">
              <div
                className="h-full rounded-pill bg-brand/60"
                style={{ width: `${(row.count / max) * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      <p className="mt-3 border-t border-line pt-2.5 text-xs leading-[18px] text-ink-secondary">
        Smallest {formatCount(dataset.class_balance_ratio.min)} · largest{" "}
        {formatCount(dataset.class_balance_ratio.max)} ·{" "}
        {dataset.class_balance_ratio.ratio.toFixed(2)}× ratio. Not perfectly
        uniform, but every intent has enough examples to learn from.
      </p>
    </Card>
  );
}

function EmbeddingMap({ points }: { points: MapPointWithQuery[] }) {
  const [active, setActive] = useState<Set<string>>(
    () => new Set(points.map((point) => point.category))
  );
  const [hovered, setHovered] = useState<{
    point: MapPointWithQuery;
    left: number;
    top: number;
  } | null>(null);

  const categories = useMemo(
    () => Array.from(new Set(points.map((point) => point.category))).sort(),
    [points]
  );

  const { mapped, bounds } = useMemo(() => {
    if (points.length === 0) {
      return { mapped: [], bounds: { minX: 0, maxX: 1, minY: 0, maxY: 1 } };
    }
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const point of points) {
      if (point.x < minX) minX = point.x;
      if (point.x > maxX) maxX = point.x;
      if (point.y < minY) minY = point.y;
      if (point.y > maxY) maxY = point.y;
    }
    return { mapped: points, bounds: { minX, maxX, minY, maxY } };
  }, [points]);

  const width = 600;
  const height = 300;
  const padding = 18;
  const spanX = bounds.maxX - bounds.minX || 1;
  const spanY = bounds.maxY - bounds.minY || 1;
  const toX = (value: number) =>
    padding + ((value - bounds.minX) / spanX) * (width - padding * 2);
  const toY = (value: number) =>
    height - padding - ((value - bounds.minY) / spanY) * (height - padding * 2);

  const toggle = (category: string) => {
    setActive((previous) => {
      const next = new Set(previous);
      if (next.has(category)) next.delete(category);
      else next.add(category);
      return next;
    });
    setHovered(null);
  };

  return (
    <Card className="p-4">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h2 className="text-base font-semibold text-ink">Embedding map</h2>
          <p className="mt-0.5 text-xs leading-[18px] text-ink-secondary">
            A 2D projection of past questions. Hover or tap a dot for its
            intent and question.
          </p>
        </div>
        <Chip tone="neutral">{points.length > 0 ? "UMAP · 2D" : "No map data"}</Chip>
      </div>

      {points.length === 0 ? (
        <div className="mt-3">
          <EmptyDataFile
            file="map_points.json"
            detail="Add map_points.json to web/public/data/ to draw the semantic map."
          />
        </div>
      ) : (
        <>
          <div className="mt-2 flex flex-wrap gap-1.5">
            <FilterChip
              active={active.size === categories.length}
              onClick={() => setActive(new Set(categories))}
            >
              All
            </FilterChip>
            {categories.map((category) => (
              <button
                key={category}
                type="button"
                onClick={() => toggle(category)}
                aria-pressed={active.has(category)}
                className={cn(
                  "inline-flex items-center gap-1.5 rounded-pill border px-2.5 py-0.5 text-xs font-medium transition-colors",
                  active.has(category)
                    ? "border-brand/15 bg-brand/[0.08] text-brand-ink"
                    : "border-line bg-card text-ink-faint hover:text-ink-secondary"
                )}
              >
                <span
                  aria-hidden="true"
                  className="size-1.5 rounded-full"
                  style={{
                    backgroundColor: CATEGORY_COLORS[category] ?? "#A1A1AA",
                  }}
                />
                {CATEGORY_LABELS[category] ?? category}
              </button>
            ))}
          </div>

          <div className="relative mt-2">
            <svg
              viewBox={`0 0 ${width} ${height}`}
              role="img"
              aria-label="Scatter plot of past support questions coloured by category. Hover or tap a dot to read its intent and question."
              className="w-full rounded-[12px] border border-line bg-[#FAFAF8]"
              onMouseLeave={() => setHovered(null)}
            >
              {mapped
                .filter((point) => active.has(point.category))
                .map((point) => (
                  <circle
                    key={point.index}
                    cx={toX(point.x)}
                    cy={toY(point.y)}
                    r={1.8}
                    fill={CATEGORY_COLORS[point.category] ?? "#A1A1AA"}
                    opacity={0.55}
                    className="cursor-crosshair"
                    onMouseEnter={(event) => {
                      const rect = event.currentTarget.ownerSVGElement
                        ?.getBoundingClientRect();
                      if (!rect) return;
                      setHovered({
                        point,
                        left: (toX(point.x) / width) * 100,
                        top: (toY(point.y) / height) * 100,
                      });
                    }}
                    onPointerDown={(event) => {
                      const rect = event.currentTarget.ownerSVGElement
                        ?.getBoundingClientRect();
                      if (!rect) return;
                      setHovered({
                        point,
                        left: (toX(point.x) / width) * 100,
                        top: (toY(point.y) / height) * 100,
                      });
                    }}
                  >
                    <title>{`${humaniseIntent(point.intent)} — ${
                      point.query ?? ""
                    }`}</title>
                  </circle>
                ))}
            </svg>

            {hovered ? (
              <div
                role="status"
                className="pointer-events-none absolute z-10 w-[220px] -translate-x-1/2 -translate-y-full rounded-[10px] border border-line bg-card px-2.5 py-2 shadow-float"
                style={{
                  left: `${Math.min(88, Math.max(12, hovered.left))}%`,
                  top: `${Math.max(8, hovered.top)}%`,
                }}
              >
                <p className="text-xs font-medium text-ink">
                  {humaniseIntent(hovered.point.intent)}
                </p>
                <p className="mt-0.5 text-xs leading-[18px] text-ink-secondary">
                  “{hovered.point.query ?? "Sample question unavailable"}”
                </p>
                <p className="mt-1 font-mono text-[11px] text-ink-faint">
                  {CATEGORY_LABELS[hovered.point.category] ?? hovered.point.category}
                </p>
              </div>
            ) : null}
          </div>

          <p className="mt-1.5 text-xs leading-[18px] text-ink-secondary">
            Closer dots = more similar questions. {formatCount(points.length)} of
            the full corpus are projected.
          </p>
        </>
      )}
    </Card>
  );
}

function BrowseExamples({ examples }: { examples: BrowseExample[] }) {
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("ALL");
  const [style, setStyle] = useState("ALL");

  const categories = useMemo(
    () => Array.from(new Set(examples.map((row) => row.category))).sort(),
    [examples]
  );
  const styles = useMemo(
    () => Array.from(new Set(examples.map((row) => row.style))).sort(),
    [examples]
  );

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return examples.filter((row) => {
      const matchesQuery =
        !needle ||
        row.query.toLowerCase().includes(needle) ||
        row.intent.toLowerCase().includes(needle) ||
        row.category.toLowerCase().includes(needle);
      const matchesCategory = category === "ALL" || row.category === category;
      const matchesStyle = style === "ALL" || row.style === style;
      return matchesQuery && matchesCategory && matchesStyle;
    });
  }, [category, examples, query, style]);

  return (
    <Card className="p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-ink">
            Browse examples
          </h2>
          <p className="mt-0.5 text-xs leading-[18px] text-ink-secondary">
            Search a sample of real support questions by meaning, category or
            style.
          </p>
        </div>
        <span className="text-xs text-ink-secondary">
          {formatCount(filtered.length)} of {formatCount(examples.length)}
        </span>
      </div>

      <div className="mt-3 flex flex-col gap-2 sm:flex-row">
        <div className="relative min-w-0 flex-1">
          <Search
            size={14}
            strokeWidth={1.9}
            aria-hidden="true"
            className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-ink-faint"
          />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search questions, intents or categories…"
            aria-label="Search examples"
            className="h-9 w-full rounded-pill border border-line bg-card pl-8 pr-3 text-xs text-ink placeholder:text-ink-faint focus:border-brand/40 focus:outline-none"
          />
        </div>
        <Select
          label="Filter by category"
          value={category}
          onChange={setCategory}
          options={[
            { value: "ALL", label: "All categories" },
            ...categories.map((item) => ({
              value: item,
              label: CATEGORY_LABELS[item] ?? item,
            })),
          ]}
        />
        <Select
          label="Filter by style"
          value={style}
          onChange={setStyle}
          options={[
            { value: "ALL", label: "All styles" },
            ...styles.map((item) => ({ value: item, label: item })),
          ]}
        />
      </div>

      <div className="mt-3 max-h-[380px] overflow-auto rounded-[10px] border border-line">
        <table className="w-full min-w-[640px] border-collapse text-left">
          <thead className="sticky top-0 z-10 bg-soft">
            <tr>
              <th className="px-3 py-2 text-xs font-medium uppercase tracking-[0.08em] text-ink-secondary">
                Query
              </th>
              <th className="px-3 py-2 text-xs font-medium uppercase tracking-[0.08em] text-ink-secondary">
                Category
              </th>
              <th className="px-3 py-2 text-xs font-medium uppercase tracking-[0.08em] text-ink-secondary">
                Intent
              </th>
              <th className="px-3 py-2 text-xs font-medium uppercase tracking-[0.08em] text-ink-secondary">
                Style
              </th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((row, index) => (
              <tr
                key={`${row.query}-${index}`}
                className="border-t border-line last:border-b-0"
              >
                <td className="max-w-[340px] px-3 py-2 text-xs leading-[18px] text-ink">
                  <span className="line-clamp-2">{row.query}</span>
                </td>
                <td className="whitespace-nowrap px-3 py-2 text-xs text-ink-secondary">
                  {CATEGORY_LABELS[row.category] ?? row.category}
                </td>
                <td className="whitespace-nowrap px-3 py-2 font-mono text-xs text-ink-secondary">
                  {row.intent}
                </td>
                <td className="whitespace-nowrap px-3 py-2 text-xs text-ink-secondary">
                  <span className="inline-flex items-center gap-1.5">
                    <span
                      aria-hidden="true"
                      className="size-1.5 rounded-full"
                      style={{
                        backgroundColor:
                          STYLE_COLORS[row.style] ?? "#A1A1AA",
                      }}
                    />
                    {row.style}
                  </span>
                </td>
              </tr>
            ))}
            {filtered.length === 0 ? (
              <tr>
                <td
                  colSpan={4}
                  className="px-3 py-8 text-center text-xs text-ink-secondary"
                >
                  No examples match those filters.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function SortButton<T extends string>({
  value,
  onChange,
  options,
}: {
  value: T;
  onChange: (value: T) => void;
  options: { value: T; label: string }[];
}) {
  return (
    <label className="inline-flex items-center gap-1.5 text-xs text-ink-secondary">
      <ArrowUpDown size={12} aria-hidden="true" />
      <span className="sr-only">Sort rows</span>
      <select
        value={value}
        onChange={(event) => onChange(event.target.value as T)}
        className="h-7 rounded-pill border border-line bg-card px-2 pr-6 text-xs text-ink focus:border-brand/40 focus:outline-none"
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function Select({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <label className="sr-only">
      {label}
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        aria-label={label}
        className="mt-1 h-9 w-full rounded-pill border border-line bg-card px-2 text-xs text-ink focus:border-brand/40 focus:outline-none sm:w-auto"
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function FilterChip({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={cn(
        "rounded-pill border px-2.5 py-0.5 text-xs font-medium transition-colors",
        active
          ? "border-brand/15 bg-brand/[0.08] text-brand-ink"
          : "border-line bg-card text-ink-secondary hover:text-ink"
      )}
    >
      {children}
    </button>
  );
}
