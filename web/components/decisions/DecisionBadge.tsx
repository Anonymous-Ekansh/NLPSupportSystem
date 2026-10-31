import type { Decision } from "@/lib/api";
import { cn } from "@/lib/utils";

const TONES: Record<
  Decision,
  { wrap: string; dot: string; label: string; blurb: string }
> = {
  ANSWER: {
    wrap: "border-success/25 bg-success-soft text-success-ink",
    dot: "bg-success",
    label: "Answer",
    blurb: "Evidence-backed answer, checked by A4",
  },
  CLARIFY: {
    wrap: "border-warning/25 bg-warning-soft text-warning-ink",
    dot: "bg-warning",
    label: "Clarify",
    blurb: "One focused question instead of a guess",
  },
  ESCALATE: {
    wrap: "border-error/20 bg-error-soft text-error-ink",
    dot: "bg-error",
    label: "Escalate",
    blurb: "Handed to a human — nothing was changed",
  },
};

interface DecisionBadgeProps {
  decision: Decision | null | undefined;
  size?: "sm" | "lg";
  showBlurb?: boolean;
  className?: string;
}

/**
 * The single most important thing on the page: which of the three outcomes
 * the system chose. Colour-coded and readable from across a room.
 */
export function DecisionBadge({
  decision,
  size = "sm",
  showBlurb = false,
  className,
}: DecisionBadgeProps) {
  if (!decision) return null;
  const tone = TONES[decision] ?? TONES.ANSWER;

  return (
    <div className={cn("inline-flex flex-col items-start gap-1", className)}>
      <span
        className={cn(
          "inline-flex items-center gap-1.5 rounded-pill border font-semibold uppercase tracking-[0.08em]",
          tone.wrap,
          size === "lg"
            ? "px-3.5 py-1.5 text-sm"
            : "px-2.5 py-0.5 text-xs"
        )}
      >
        <span aria-hidden="true" className={cn("size-1.5 rounded-full", tone.dot)} />
        {decision}
      </span>
      {showBlurb ? (
        <span className="text-xs leading-[18px] text-ink-secondary">
          {tone.blurb}
        </span>
      ) : null}
    </div>
  );
}
