"use client";

import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { RetrievalModesFile } from "@/lib/data";
import { cn } from "@/lib/utils";

type Metric = "cosine" | "consistency";

interface Point {
  mode: string;
  label: string;
  with_a1: number;
  without_a1: number;
}

function ChartTooltip({
  active,
  payload,
  metric,
}: {
  active?: boolean;
  payload?: { payload?: Point }[];
  metric: Metric;
}) {
  if (!active || !payload?.length) return null;
  const point = payload[0]?.payload;
  if (!point) return null;
  const format = (value: number) => `${(value * 100).toFixed(1)}%`;
  return (
    <div className="rounded-[10px] border border-line bg-card px-2.5 py-2">
      <p className="text-xs font-medium text-ink">{point.label}</p>
      <p className="mt-0.5 font-mono text-xs text-brand-ink">
        With A1 routing {format(point.with_a1)}
      </p>
      <p className="font-mono text-xs text-ink-secondary">
        Without routing {format(point.without_a1)}
      </p>
      <p className="mt-0.5 text-[11px] text-ink-faint">
        {metric === "cosine" ? "Answer relevance (cosine)" : "Intent consistency"}
      </p>
    </div>
  );
}

export default function RetrievalChart({
  data,
}: {
  data: RetrievalModesFile;
}) {
  const [metric, setMetric] = useState<Metric>("cosine");

  const points: Point[] = data.modes.map((mode) => {
    const routed = data.routing.with_a1.find((row) => row.key === mode.key);
    const global = data.routing.without_a1.find((row) => row.key === mode.key);
    return {
      mode: mode.key,
      label: mode.served ? `${mode.mode} · served` : mode.mode,
      with_a1: metric === "cosine" ? routed?.cosine_sim ?? 0 : routed?.intent_consistency ?? 0,
      without_a1:
        metric === "cosine"
          ? global?.cosine_sim ?? 0
          : global?.intent_consistency ?? 0,
    };
  });

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="inline-flex rounded-pill border border-line bg-soft p-0.5">
          {(
            [
              { value: "cosine", label: "Answer relevance" },
              { value: "consistency", label: "Intent consistency" },
            ] as const
          ).map((option) => (
            <button
              key={option.value}
              type="button"
              onClick={() => setMetric(option.value)}
              aria-pressed={metric === option.value}
              className={cn(
                "rounded-pill px-2.5 py-1 text-xs font-medium transition-colors",
                metric === option.value
                  ? "bg-card text-ink shadow-float"
                  : "text-ink-secondary hover:text-ink"
              )}
            >
              {option.label}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-3 text-xs text-ink-secondary">
          <span className="inline-flex items-center gap-1.5">
            <span className="size-2 rounded-full bg-brand" />
            With A1 routing
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="size-2 rounded-full bg-[#D4D4D8]" />
            Without routing
          </span>
        </div>
      </div>

      <div
        className="mt-2 h-[230px] w-full"
        role="img"
        aria-label={`Grouped bar chart of ${metric === "cosine" ? "answer relevance" : "intent consistency"} with and without A1 routing`}
      >
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={points}
            margin={{ top: 8, right: 8, bottom: 0, left: 0 }}
            barGap={4}
          >
            <CartesianGrid vertical={false} stroke="#ECEBE8" />
            <XAxis
              dataKey="label"
              tickLine={false}
              axisLine={false}
              tick={{ fill: "#71717A", fontSize: 11 }}
            />
            <YAxis
              domain={[0, 1]}
              ticks={[0, 0.25, 0.5, 0.75, 1]}
              tickFormatter={(value: number) => `${Math.round(value * 100)}%`}
              tickLine={false}
              axisLine={false}
              tick={{ fill: "#71717A", fontSize: 12 }}
              width={44}
            />
            <Tooltip
              cursor={{ fill: "#F6F6F4" }}
              content={<ChartTooltip metric={metric} />}
            />
            <Bar
              dataKey="with_a1"
              name="With A1 routing"
              fill="#4F46E5"
              radius={[4, 4, 0, 0]}
              maxBarSize={42}
            />
            <Bar
              dataKey="without_a1"
              name="Without routing"
              fill="#D4D4D8"
              radius={[4, 4, 0, 0]}
              maxBarSize={42}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
