"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { CalibrationBin } from "@/lib/data";

interface Point {
  confidence: number;
  accuracy: number;
  perfect: number;
  count: number;
}

function ChartTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: { payload?: Point }[];
}) {
  if (!active || !payload?.length) return null;
  const point = payload[0]?.payload;
  if (!point) return null;
  return (
    <div className="rounded-[10px] border border-line bg-card px-2.5 py-2">
      <p className="text-xs font-medium text-ink">
        Stated confidence {(point.confidence * 100).toFixed(1)}%
      </p>
      <p className="mt-0.5 font-mono text-xs text-ink-secondary">
        Observed accuracy {(point.accuracy * 100).toFixed(1)}%
      </p>
      <p className="mt-0.5 font-mono text-[11px] text-ink-faint">
        {point.count.toLocaleString("en-US")} questions
      </p>
    </div>
  );
}

export default function CalibrationChart({
  bins,
  tau,
}: {
  bins: CalibrationBin[];
  tau: number;
}) {
  const data: Point[] = bins
    .slice()
    .sort((a, b) => a.confidence - b.confidence)
    .map((bin) => ({
      confidence: bin.confidence,
      accuracy: bin.accuracy,
      perfect: bin.confidence,
      count: bin.count,
    }));

  return (
    <div
      className="h-[240px] w-full"
      role="img"
      aria-label={`Calibration line: observed accuracy against stated confidence, with the tau threshold at ${tau.toFixed(3)}`}
    >
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="#ECEBE8" />
          <XAxis
            dataKey="confidence"
            type="number"
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            tickFormatter={(value: number) => `${Math.round(value * 100)}%`}
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
          <Tooltip content={<ChartTooltip />} />
          <ReferenceLine
            x={tau}
            stroke="#D97706"
            strokeDasharray="4 4"
            label={{
              value: `τ ${tau.toFixed(3)}`,
              position: "insideTopRight",
              fill: "#B45309",
              fontSize: 11,
            }}
          />
          <ReferenceLine
            segment={[
              { x: 0, y: 0 },
              { x: 1, y: 1 },
            ]}
            stroke="#D4D4D8"
            strokeDasharray="3 5"
          />
          <Line
            type="monotone"
            dataKey="perfect"
            name="Perfect calibration"
            stroke="#D4D4D8"
            strokeWidth={1.2}
            strokeDasharray="3 5"
            dot={false}
            activeDot={false}
          />
          <Line
            type="monotone"
            dataKey="accuracy"
            name="Observed accuracy"
            stroke="#4F46E5"
            strokeWidth={2}
            dot={{ r: 2.4, fill: "#4F46E5" }}
            activeDot={{ r: 4 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
