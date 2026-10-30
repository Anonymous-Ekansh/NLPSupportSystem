"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { LadderRow } from "@/lib/data";

interface Point extends LadderRow {
  errorY: [number, number];
}

function ChartTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: { payload?: Point }[];
}) {
  if (!active || !payload?.length) return null;
  const row = payload[0]?.payload;
  if (!row) return null;
  return (
    <div className="rounded-[10px] border border-line bg-card px-2.5 py-2">
      <p className="text-xs font-medium text-ink">
        {row.level} · {row.name}
      </p>
      <p className="mt-0.5 font-mono text-xs text-ink-secondary">
        Macro-F1 {(row.macro_f1 * 100).toFixed(2)}%
      </p>
      <p className="mt-0.5 font-mono text-[11px] text-ink-faint">
        95% CI {(row.ci_lo * 100).toFixed(2)}–{(row.ci_hi * 100).toFixed(2)}%
      </p>
    </div>
  );
}

export default function LadderChart({ rows }: { rows: LadderRow[] }) {
  const data: Point[] = rows.map((row) => ({
    ...row,
    errorY: [
      Math.max(0, row.macro_f1 - row.ci_lo),
      Math.max(0, row.ci_hi - row.macro_f1),
    ],
  }));

  return (
    <div
      className="h-[240px] w-full"
      role="img"
      aria-label="Bar chart of macro-F1 by model level with 95% confidence intervals; L2 is the served model"
    >
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          margin={{ top: 12, right: 8, bottom: 0, left: 0 }}
        >
          <CartesianGrid vertical={false} stroke="#ECEBE8" />
          <XAxis
            dataKey="level"
            tickLine={false}
            axisLine={false}
            tick={{ fill: "#71717A", fontSize: 12 }}
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
          <Tooltip cursor={{ fill: "#F6F6F4" }} content={<ChartTooltip />} />
          <Bar dataKey="macro_f1" radius={[4, 4, 0, 0]} maxBarSize={54}>
            {data.map((row) => (
              <Cell
                key={row.level}
                fill={row.served ? "#4F46E5" : "#D4D4D8"}
              />
            ))}
            <ErrorBar
              dataKey="errorY"
              width={4}
              strokeWidth={1.4}
              stroke="#71717A"
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
