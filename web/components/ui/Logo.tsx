import { Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";

export function LogoMark({
  size = "md",
  className,
}: {
  size?: "sm" | "md";
  className?: string;
}) {
  return (
    <span
      aria-hidden="true"
      className={cn(
        "inline-flex shrink-0 items-center justify-center bg-brand text-white",
        size === "md" ? "size-8 rounded-[10px]" : "size-5 rounded-[7px]",
        className
      )}
    >
      <Sparkles
        size={size === "md" ? 17 : 11}
        strokeWidth={1.9}
      />
    </span>
  );
}

export function Logo() {
  return (
    <span className="flex items-center gap-2.5">
      <LogoMark />
      <span className="text-base font-semibold tracking-tight text-ink">
        Resolve<span className="text-brand">IQ</span>
      </span>
    </span>
  );
}
