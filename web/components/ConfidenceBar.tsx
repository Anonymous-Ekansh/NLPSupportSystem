import React from "react";

export function ConfidenceBar({
  confidence,
  threshold,
}: {
  confidence: number; // 0.0 to 1.0
  threshold?: number | null; // e.g. 0.65 or null
}) {
  const percent = Math.min(Math.max(Math.round(confidence * 100), 0), 100);
  const isBelow = threshold != null && confidence < threshold;
  const thresholdPercent =
    threshold != null ? Math.min(Math.max(Math.round(threshold * 100), 0), 100) : null;

  return (
    <div className="w-full space-y-1">
      <div className="flex justify-between items-center text-sm font-medium">
        <span className="text-[var(--muted-text)]">Confidence</span>
        <span
          className={`font-semibold ${
            isBelow ? "text-amber-400" : "text-emerald-400"
          }`}
        >
          {percent}% {isBelow ? "(low confidence)" : ""}
        </span>
      </div>

      <div className="relative w-full h-3.5 bg-gray-800/80 rounded-full overflow-hidden border border-[var(--card-border)]">
        {/* Progress fill */}
        <div
          className={`h-full transition-all duration-500 rounded-full ${
            isBelow ? "bg-amber-500" : "bg-emerald-500"
          }`}
          style={{ width: `${percent}%` }}
        />

        {/* Dashed Threshold Marker (hidden if threshold is not available) */}
        {thresholdPercent != null && (
          <div
            className="absolute top-0 bottom-0 w-0.5 border-r-2 border-dashed border-white/80 z-10"
            style={{ left: `${thresholdPercent}%` }}
            title={`Threshold: ${thresholdPercent}%`}
          />
        )}
      </div>

      {thresholdPercent != null && (
        <div className="flex justify-end text-xs text-[var(--muted-text)] font-mono">
          threshold {thresholdPercent}%
        </div>
      )}
    </div>
  );
}
