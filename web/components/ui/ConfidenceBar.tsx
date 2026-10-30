import { cn, clamp } from "@/lib/utils";

type Tone = "brand" | "a1" | "a2" | "success" | "warning" | "neutral";

const fills: Record<Tone, string> = {
  brand: "bg-brand",
  a1: "bg-a1",
  a2: "bg-a2",
  success: "bg-success",
  warning: "bg-warning",
  neutral: "bg-ink-faint",
};

interface ConfidenceBarProps {
  value: number;
  threshold?: number | null;
  tone?: Tone;
  label?: string;
  showValue?: boolean;
  height?: "sm" | "md";
  className?: string;
  ariaLabel?: string;
}

export function ConfidenceBar({
  value,
  threshold,
  tone = "brand",
  label,
  showValue = false,
  height = "md",
  className,
  ariaLabel,
}: ConfidenceBarProps) {
  const width = `${clamp(value) * 100}%`;
  const tick = threshold === null || threshold === undefined
    ? null
    : clamp(threshold);

  return (
    <div className={className}>
      {label || showValue ? (
        <div className="mb-1.5 flex items-center justify-between gap-3">
          <span className="text-xs text-ink-secondary">{label}</span>
          {showValue ? (
            <span className="font-mono text-xs tabular-nums text-ink">
              {(clamp(value) * 100).toFixed(1)}%
            </span>
          ) : null}
        </div>
      ) : null}
      <div
        role="progressbar"
        aria-label={ariaLabel ?? label ?? "Confidence"}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(clamp(value) * 100)}
        className={cn(
          "relative w-full overflow-visible rounded-pill bg-[#F0F0ED]",
          height === "md" ? "h-2" : "h-1.5"
        )}
      >
        <div
          className={cn(
            "h-full rounded-pill transition-[width] duration-300 ease-out",
            fills[tone]
          )}
          style={{ width }}
        />
        {tick !== null ? (
          <span
            aria-hidden="true"
            title={`Threshold ${(tick * 100).toFixed(0)}%`}
            className="absolute -top-1.5 bottom-[-6px] w-px bg-ink/70"
            style={{ left: `calc(${tick * 100}% - 0.5px)` }}
          />
        ) : null}
      </div>
    </div>
  );
}
