import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

type TraceTone = "neutral" | "a1" | "a2" | "success" | "warning";

const circleTones: Record<TraceTone, string> = {
  neutral: "border-line bg-soft text-ink-secondary",
  a1: "border-a1/15 bg-a1/[0.08] text-a1-ink",
  a2: "border-a2/15 bg-a2/[0.08] text-a2-ink",
  success: "border-success/15 bg-success-soft text-success-ink",
  warning: "border-warning/15 bg-warning-soft text-warning-ink",
};

interface TraceNodeProps {
  step: number;
  title: ReactNode;
  tone?: TraceTone;
  isLast?: boolean;
  badge?: ReactNode;
  children: ReactNode;
  className?: string;
  /** Used by the replay animation to reveal nodes one at a time. */
  reveal?: boolean;
}

export function TraceNode({
  step,
  title,
  tone = "neutral",
  isLast = false,
  badge,
  children,
  className,
  reveal = true,
}: TraceNodeProps) {
  if (!reveal) return null;

  return (
    <div className={cn("relative pl-9", className)}>
      {!isLast ? (
        <span
          aria-hidden="true"
          className="absolute bottom-[-2px] left-[12px] top-7 w-px bg-line"
        />
      ) : null}
      <span
        aria-hidden="true"
        className={cn(
          "absolute left-0 top-0 flex size-[26px] items-center justify-center rounded-full border font-mono text-xs font-medium",
          circleTones[tone]
        )}
      >
        {step}
      </span>
      <div className={cn("min-w-0", !isLast && "pb-6")}>
        <div className="flex min-h-[26px] flex-wrap items-center gap-x-2 gap-y-1">
          <h3 className="text-sm font-semibold leading-[26px] text-ink">
            {title}
          </h3>
          {badge}
        </div>
        <div className="mt-1.5">{children}</div>
      </div>
    </div>
  );
}
