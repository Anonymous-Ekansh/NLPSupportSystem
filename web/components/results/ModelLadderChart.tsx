"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface LadderRow {
  level: string;
  name: string;
  macro_f1: number;
  served: boolean;
}

interface TooltipPayload {
  payload?: LadderRow;
}

function ChartTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: TooltipPayload[];
}) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  if (!row) return null;
  return (
    <div className="rounded-[10px] border border-line bg-card px-2.5 py-2">
      <p className="text-xs font-medium text-ink">
        {row.level} · {row.name}
      </p>
      <p className="mt-0.5 font-mono text-xs text-ink-secondary">
        Macro-F1 {(row.macro_f1 * 100).toFixed(2)}%
      </p>
    </div>
  );
}

export function ModelLadderChart({ rows }: { rows: LadderRow[] }) {
  return (
    <div
      className="h-[230px] w-full"
      role="img"
      aria-label="Bar chart of macro-F1 by model level; L2 is the served model"
    >
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={rows}
          margin={{ top: 8, right: 4, bottom: 0, left: -12 }}
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
          <Tooltip
            cursor={{ fill: "#F6F6F4" }}
            content={<ChartTooltip />}
          />
          <Bar dataKey="macro_f1" radius={[4, 4, 0, 0]} maxBarSize={54}>
            {rows.map((row) => (
              <Cell
                key={row.level}
                fill={row.served ? "#4F46E5" : "#D4D4D8"}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
