"use client";

import { useState } from "react";
import type { ConfusionCategoryFile } from "@/lib/data";
import { CATEGORY_LABELS } from "@/components/trace/ScatterMap";
import { cn } from "@/lib/utils";

export function ConfusionHeatmap({
  data,
}: {
  data: ConfusionCategoryFile;
}) {
  const [hovered, setHovered] = useState<{
    row: number;
    col: number;
  } | null>(null);

  const max = Math.max(...data.matrix.flat(), 1);

  return (
    <div>
      <div className="overflow-x-auto">
        <table
          className="w-full table-fixed border-separate border-spacing-0 text-left"
          aria-label="Confusion matrix by category. Rows are true categories and columns are predicted categories."
          onMouseLeave={() => setHovered(null)}
        >
          <thead>
            <tr>
              <th className="w-[74px] px-1 py-1 text-[10px] font-medium leading-[13px] text-ink-faint">
                true ↓ / pred →
              </th>
              {data.labels.map((label) => (
                <th
                  key={label}
                  title={CATEGORY_LABELS[label] ?? label}
                  className="overflow-hidden px-0.5 py-1 text-center text-[9px] font-medium tracking-tight text-ink-secondary"
                >
                  {(CATEGORY_LABELS[label] ?? label).slice(0, 6)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.matrix.map((row, rowIndex) => (
              <tr key={data.labels[rowIndex]}>
                <th className="w-[74px] max-w-[74px] truncate whitespace-nowrap bg-card px-1 py-0.5 text-right text-[10px] font-medium text-ink-secondary">
                  {CATEGORY_LABELS[data.labels[rowIndex]] ??
                    data.labels[rowIndex]}
                </th>
                {row.map((value, colIndex) => {
                  const alpha =
                    value === 0 ? 0 : 0.08 + 0.85 * (value / max);
                  const isDiagonal = rowIndex === colIndex;
                  return (
                    <td
                      key={`${rowIndex}-${colIndex}`}
                      onMouseEnter={() =>
                        setHovered({ row: rowIndex, col: colIndex })
                      }
                      title={`True ${data.labels[rowIndex]} → predicted ${data.labels[colIndex]}: ${value}`}
                      className={cn(
                        "relative h-8 overflow-hidden border border-white/60 text-center align-middle",
                        isDiagonal && "ring-1 ring-inset ring-brand/20"
                      )}
                      style={{
                        backgroundColor:
                          value === 0
                            ? "#FAFAF8"
                            : `rgba(79, 70, 229, ${alpha})`,
                      }}
                    >
                      <span
                        className={cn(
                          "font-mono text-[10px] tabular-nums",
                          alpha > 0.55 ? "text-white" : "text-ink-secondary"
                        )}
                      >
                        {value === 0 ? "" : value}
                      </span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-2 flex min-h-[18px] flex-wrap items-center justify-between gap-2 text-xs">
        <p className="text-ink-secondary" aria-live="polite">
          {hovered ? (
            <>
              True{" "}
              <span className="font-medium text-ink">
                {CATEGORY_LABELS[data.labels[hovered.row]] ??
                  data.labels[hovered.row]}
              </span>{" "}
              → predicted{" "}
              <span className="font-medium text-ink">
                {CATEGORY_LABELS[data.labels[hovered.col]] ??
                  data.labels[hovered.col]}
              </span>
              :{" "}
              <span className="font-mono text-ink">
                {data.matrix[hovered.row][hovered.col]}
              </span>
            </>
          ) : (
            "Hover a cell for the true and predicted category."
          )}
        </p>
        <span className="inline-flex items-center gap-1.5 text-ink-faint">
          fewer
          <span className="inline-flex items-center gap-[2px]">
            {[0.12, 0.3, 0.5, 0.7, 0.9].map((alpha) => (
              <span
                key={alpha}
                className="size-2 rounded-[2px]"
                style={{ backgroundColor: `rgba(79, 70, 229, ${alpha})` }}
              />
            ))}
          </span>
          more
        </span>
      </div>
      {data.note ? (
        <p className="mt-1 text-[11px] leading-[17px] text-ink-faint">
          {data.note}
        </p>
      ) : null}
    </div>
  );
}
