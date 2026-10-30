"use client";

import { useMemo, useState } from "react";
import { ArrowDown, ArrowUp } from "lucide-react";
import type { IntentMetric } from "@/lib/data";
import { cn, formatPercent } from "@/lib/utils";

type SortKey = "intent" | "f1" | "precision" | "recall" | "support";
type Direction = "asc" | "desc";

export function F1Table({ metrics }: { metrics: IntentMetric[] }) {
  const [sortKey, setSortKey] = useState<SortKey>("f1");
  const [direction, setDirection] = useState<Direction>("asc");

  const rows = useMemo(() => {
    const sorted = [...metrics].sort((a, b) => {
      if (sortKey === "intent") return a.intent.localeCompare(b.intent);
      return a[sortKey] - b[sortKey];
    });
    return direction === "asc" ? sorted : sorted.reverse();
  }, [direction, metrics, sortKey]);

  const changeSort = (key: SortKey) => {
    if (sortKey === key) {
      setDirection((current) => (current === "asc" ? "desc" : "asc"));
      return;
    }
    setSortKey(key);
    setDirection(key === "intent" ? "asc" : key === "support" ? "desc" : "asc");
  };

  return (
    <div className="max-h-[430px] overflow-auto rounded-[10px] border border-line">
      <table className="w-full min-w-[520px] border-collapse text-left">
        <thead className="sticky top-0 z-10 bg-soft">
          <tr>
            <SortableHead
              label="Intent"
              active={sortKey === "intent"}
              direction={direction}
              onClick={() => changeSort("intent")}
            />
            <SortableHead
              label="F1"
              active={sortKey === "f1"}
              direction={direction}
              onClick={() => changeSort("f1")}
            />
            <SortableHead
              label="Precision"
              active={sortKey === "precision"}
              direction={direction}
              onClick={() => changeSort("precision")}
              className="hidden sm:table-cell"
            />
            <SortableHead
              label="Recall"
              active={sortKey === "recall"}
              direction={direction}
              onClick={() => changeSort("recall")}
              className="hidden sm:table-cell"
            />
            <SortableHead
              label="Support"
              active={sortKey === "support"}
              direction={direction}
              onClick={() => changeSort("support")}
              align="right"
            />
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={row.intent}
              className="border-t border-line last:border-b-0"
            >
              <td className="max-w-[190px] px-2.5 py-1.5">
                <p className="truncate font-mono text-xs text-ink">
                  {row.intent}
                </p>
                <p className="truncate text-[11px] text-ink-faint">
                  {row.category}
                </p>
              </td>
              <td className="px-2.5 py-1.5">
                <div className="flex items-center gap-2">
                  <div className="h-1.5 w-16 overflow-hidden rounded-pill bg-[#F0F0ED] sm:w-24">
                    <div
                      className={cn(
                        "h-full rounded-pill",
                        row.f1 >= 0.995
                          ? "bg-success/70"
                          : row.f1 >= 0.98
                            ? "bg-brand/70"
                            : "bg-warning/70"
                      )}
                      style={{ width: `${Math.max(row.f1 * 100, 2)}%` }}
                    />
                  </div>
                  <span className="font-mono text-xs tabular-nums text-ink">
                    {formatPercent(row.f1, 2)}
                  </span>
                </div>
              </td>
              <td className="hidden px-2.5 py-1.5 font-mono text-xs tabular-nums text-ink-secondary sm:table-cell">
                {formatPercent(row.precision, 2)}
              </td>
              <td className="hidden px-2.5 py-1.5 font-mono text-xs tabular-nums text-ink-secondary sm:table-cell">
                {formatPercent(row.recall, 2)}
              </td>
              <td className="px-2.5 py-1.5 text-right font-mono text-xs tabular-nums text-ink-secondary">
                {row.support}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SortableHead({
  label,
  active,
  direction,
  onClick,
  className,
  align = "left",
}: {
  label: string;
  active: boolean;
  direction: Direction;
  onClick: () => void;
  className?: string;
  align?: "left" | "right";
}) {
  return (
    <th
      className={cn(
        "px-2.5 py-2 text-xs font-medium uppercase tracking-[0.08em] text-ink-secondary",
        align === "right" ? "text-right" : "text-left",
        className
      )}
      aria-sort={
        active ? (direction === "asc" ? "ascending" : "descending") : "none"
      }
    >
      <button
        type="button"
        onClick={onClick}
        className={cn(
          "inline-flex items-center gap-1 transition-colors hover:text-ink",
          align === "right" && "flex-row-reverse"
        )}
      >
        {label}
        {active ? (
          direction === "asc" ? (
            <ArrowUp size={11} aria-hidden="true" />
          ) : (
            <ArrowDown size={11} aria-hidden="true" />
          )
        ) : (
          <ArrowDown size={11} aria-hidden="true" className="opacity-25" />
        )}
      </button>
    </th>
  );
}
