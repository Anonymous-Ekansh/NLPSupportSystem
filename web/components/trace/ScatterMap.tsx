"use client";

import { useMemo } from "react";
import type { MapPoint } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Chip } from "@/components/ui/Chip";

/**
 * Muted category palette. Deliberately excludes the agent colours
 * (indigo / teal) and the status colours, which carry a specific meaning.
 */
export const CATEGORY_COLORS: Record<string, string> = {
  ACCOUNT: "#64748B",
  ORDER: "#7C8FA8",
  REFUND: "#A78B6F",
  DELIVERY: "#6B8E8A",
  FEEDBACK: "#8B7E9E",
  SHIPPING: "#7A9A7E",
  CONTACT: "#A07A86",
  PAYMENT: "#A39A6A",
  INVOICE: "#6F8FA3",
  SUBSCRIPTION: "#8F7F9E",
  CANCEL: "#9A7B72",
};

export const CATEGORY_LABELS: Record<string, string> = {
  ACCOUNT: "Account",
  ORDER: "Order",
  REFUND: "Refund",
  DELIVERY: "Delivery",
  FEEDBACK: "Feedback",
  SHIPPING: "Shipping",
  CONTACT: "Contact",
  PAYMENT: "Payment",
  INVOICE: "Invoice",
  SUBSCRIPTION: "Subscription",
  CANCEL: "Cancel",
};

const VIEW_WIDTH = 600;
const VIEW_HEIGHT = 320;
const PADDING = 18;

interface ScatterMapProps {
  points: MapPoint[];
  query?: { x: number | null; y: number | null } | null;
  height?: number;
  showLegend?: boolean;
  className?: string;
  caption?: string;
}

export function ScatterMap({
  points,
  query,
  height = 220,
  showLegend = false,
  className,
  caption = "Closer dots = similar past questions.",
}: ScatterMapProps) {
  const { plotted, queryPoint } = useMemo(() => {
    if (points.length === 0) return { plotted: [], queryPoint: null };

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

    const spanX = maxX - minX || 1;
    const spanY = maxY - minY || 1;
    const toX = (x: number) =>
      PADDING + ((x - minX) / spanX) * (VIEW_WIDTH - PADDING * 2);
    const toY = (y: number) =>
      VIEW_HEIGHT -
      PADDING -
      ((y - minY) / spanY) * (VIEW_HEIGHT - PADDING * 2);

    const mapped = points.map((point) => ({
      ...point,
      cx: toX(point.x),
      cy: toY(point.y),
    }));

    return {
      plotted: mapped,
      queryPoint:
        query && typeof query.x === "number" && typeof query.y === "number"
          ? { cx: toX(query.x), cy: toY(query.y) }
          : null,
    };
  }, [points, query]);

  const categories = useMemo(
    () => Array.from(new Set(points.map((point) => point.category))).sort(),
    [points]
  );

  return (
    <div className={cn("min-w-0", className)}>
      <div className="relative">
        <svg
          role="img"
          aria-label="Scatter plot of past support questions by support category"
          viewBox={`0 0 ${VIEW_WIDTH} ${VIEW_HEIGHT}`}
          style={{ height }}
          className="w-full rounded-[12px] border border-line bg-[#FAFAF8]"
        >
          {plotted.map((point, index) => (
            <circle
              key={`${point.index}-${index}`}
              cx={point.cx}
              cy={point.cy}
              r={1.45}
              fill={CATEGORY_COLORS[point.category] ?? "#A1A1AA"}
              opacity={0.42}
            />
          ))}

          {queryPoint ? (
            <g>
              <circle
                cx={queryPoint.cx}
                cy={queryPoint.cy}
                r={6}
                fill="none"
                stroke="#4F46E5"
                strokeWidth={1.5}
                opacity={0.4}
                className="animate-ping-soft"
                style={{ transformBox: "fill-box", transformOrigin: "center" }}
              />
              <circle
                cx={queryPoint.cx}
                cy={queryPoint.cy}
                r={6.5}
                fill="#FFFFFF"
                stroke="#4F46E5"
                strokeWidth={2.5}
              />
              <circle
                cx={queryPoint.cx}
                cy={queryPoint.cy}
                r={2.4}
                fill="#4F46E5"
              >
                <title>Your question</title>
              </circle>
            </g>
          ) : null}
        </svg>

        {!queryPoint ? (
          <div className="absolute right-2 top-2">
            <Chip tone="outline">No position for this question</Chip>
          </div>
        ) : null}
      </div>

      <div className="mt-2 flex items-start justify-between gap-3">
        <p className="text-xs leading-[18px] text-ink-secondary">{caption}</p>
        {queryPoint ? (
          <span className="inline-flex shrink-0 items-center gap-1.5 text-xs text-ink-secondary">
            <span className="size-2 rounded-full border-2 border-brand bg-card" />
            You
          </span>
        ) : null}
      </div>

      {showLegend ? (
        <ul className="mt-2 flex flex-wrap gap-x-3 gap-y-1.5">
          {categories.map((category) => (
            <li
              key={category}
              className="inline-flex items-center gap-1.5 text-xs text-ink-secondary"
            >
              <span
                aria-hidden="true"
                className="size-1.5 rounded-full"
                style={{
                  backgroundColor: CATEGORY_COLORS[category] ?? "#A1A1AA",
                }}
              />
              {CATEGORY_LABELS[category] ?? category}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
