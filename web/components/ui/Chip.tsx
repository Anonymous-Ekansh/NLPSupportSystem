import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export type ChipTone =
  | "neutral"
  | "brand"
  | "a1"
  | "a2"
  | "success"
  | "warning"
  | "error"
  | "outline";

const tones: Record<ChipTone, string> = {
  neutral: "border-transparent bg-soft text-ink-secondary",
  brand: "border-brand/15 bg-brand/[0.08] text-brand-ink",
  a1: "border-a1/15 bg-a1/[0.08] text-a1-ink",
  a2: "border-a2/15 bg-a2/[0.08] text-a2-ink",
  success: "border-success/15 bg-success-soft text-success-ink",
  warning: "border-warning/15 bg-warning-soft text-warning-ink",
  error: "border-error/15 bg-error-soft text-error-ink",
  outline: "border-line bg-card text-ink-secondary",
};

interface ChipProps {
  children: ReactNode;
  tone?: ChipTone;
  className?: string;
  title?: string;
}

export function Chip({
  children,
  tone = "neutral",
  className,
  title,
}: ChipProps) {
  return (
    <span
      title={title}
      className={cn(
        "inline-flex max-w-full items-center gap-1.5 rounded-pill border px-2.5 py-0.5 text-xs font-medium",
        tones[tone],
        className
      )}
    >
      {children}
    </span>
  );
}

export function Dot({
  tone = "brand",
  className,
}: {
  tone?: "brand" | "a1" | "a2" | "success" | "warning" | "error" | "neutral";
  className?: string;
}) {
  const colors: Record<string, string> = {
    brand: "bg-brand",
    a1: "bg-a1",
    a2: "bg-a2",
    success: "bg-success",
    warning: "bg-warning",
    error: "bg-error",
    neutral: "bg-ink-faint",
  };
  return (
    <span
      aria-hidden="true"
      className={cn("inline-block size-1.5 rounded-full", colors[tone], className)}
    />
  );
}
