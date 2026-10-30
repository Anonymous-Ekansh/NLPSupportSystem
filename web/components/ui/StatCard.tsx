import type { LucideIcon } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { cn } from "@/lib/utils";

interface StatCardProps {
  label: string;
  value: string;
  caption: string;
  icon?: LucideIcon;
  tone?: "brand" | "a1" | "a2" | "success" | "warning" | "neutral";
  className?: string;
  mono?: boolean;
}

const toneStyles = {
  brand: "bg-brand/[0.08] text-brand-ink",
  a1: "bg-a1/[0.08] text-a1-ink",
  a2: "bg-a2/[0.08] text-a2-ink",
  success: "bg-success-soft text-success-ink",
  warning: "bg-warning-soft text-warning-ink",
  neutral: "bg-soft text-ink-secondary",
};

export function StatCard({
  label,
  value,
  caption,
  icon: Icon,
  tone = "brand",
  className,
  mono = false,
}: StatCardProps) {
  return (
    <Card className={cn("p-4", className)}>
      <div className="flex items-start justify-between gap-3">
        <p className="text-xs font-medium uppercase tracking-[0.12em] text-ink-secondary">
          {label}
        </p>
        {Icon ? (
          <span
            className={cn(
              "flex size-7 shrink-0 items-center justify-center rounded-[10px]",
              toneStyles[tone]
            )}
          >
            <Icon size={15} strokeWidth={1.8} aria-hidden="true" />
          </span>
        ) : null}
      </div>
      <p
        className={cn(
          "mt-3 text-xl font-semibold leading-none tracking-tight text-ink",
          mono && "font-mono"
        )}
      >
        {value}
      </p>
      <p className="mt-2 text-xs leading-[18px] text-ink-secondary">{caption}</p>
    </Card>
  );
}
