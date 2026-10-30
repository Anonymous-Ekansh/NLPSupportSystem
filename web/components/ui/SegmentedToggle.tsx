"use client";

import { cn } from "@/lib/utils";

interface SegmentedToggleProps<T extends string> {
  value: T;
  onChange: (value: T) => void;
  options: Array<{ value: T; label: string }>;
  label: string;
  className?: string;
}

export function SegmentedToggle<T extends string>({
  value,
  onChange,
  options,
  label,
  className,
}: SegmentedToggleProps<T>) {
  return (
    <div
      role="group"
      aria-label={label}
      className={cn(
        "inline-flex items-center rounded-pill border border-line bg-soft p-0.5",
        className
      )}
    >
      {options.map((option) => {
        const active = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            aria-pressed={active}
            onClick={() => onChange(option.value)}
            className={cn(
              "rounded-pill px-3 py-1 text-xs font-medium transition-colors",
              active
                ? "border border-line bg-card text-ink"
                : "border border-transparent text-ink-secondary hover:text-ink"
            )}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}
