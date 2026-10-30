import { cn } from "@/lib/utils";

export function Skeleton({
  className,
  label = "Loading",
}: {
  className?: string;
  label?: string;
}) {
  return (
    <span
      role="status"
      aria-label={label}
      className={cn(
        "block animate-pulse rounded-[6px] bg-[#F0F0ED]",
        className
      )}
    />
  );
}
