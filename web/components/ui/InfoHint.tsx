"use client";

import { Info } from "lucide-react";

export function InfoHint({ text }: { text: string }) {
  return (
    <span className="group relative inline-flex align-middle">
      <button
        type="button"
        aria-label={text}
        className="inline-flex size-4 items-center justify-center rounded-full text-ink-faint transition-colors hover:text-ink-secondary"
      >
        <Info size={12} aria-hidden="true" />
      </button>
      <span
        role="tooltip"
        className="pointer-events-none absolute bottom-full left-1/2 z-30 mb-2 w-52 -translate-x-1/2 rounded-[10px] border border-line bg-card px-2.5 py-2 text-xs font-normal normal-case leading-[18px] tracking-normal text-ink-secondary opacity-0 shadow-float transition-opacity group-hover:opacity-100 group-focus-within:opacity-100"
      >
        {text}
      </span>
    </span>
  );
}
