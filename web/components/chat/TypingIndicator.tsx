"use client";

import { Clock3 } from "lucide-react";
import { LogoMark } from "@/components/ui/Logo";

export function TypingIndicator({ slow }: { slow: boolean }) {
  return (
    <div className="max-w-[92%]" role="status" aria-live="polite">
      <div className="flex items-center gap-1.5 pl-1">
        <LogoMark size="sm" />
        <span className="text-xs font-semibold text-ink-secondary">ResolveIQ</span>
      </div>
      <div className="mt-1.5 rounded-card rounded-tl-[6px] border border-line bg-card px-4 py-3.5">
        <div className="flex items-center gap-2.5">
          <span className="flex items-center gap-1" aria-hidden="true">
            {[0, 1, 2].map((index) => (
              <span
                key={index}
                className="size-1.5 animate-soft-pulse rounded-full bg-a2"
                style={{ animationDelay: `${index * 160}ms` }}
              />
            ))}
          </span>
          <div className="min-w-0">
            <p className="text-sm text-ink">
              {slow
                ? "Waking up the assistant..."
                : "Reading your question, then checking the answer..."}
            </p>
            {slow ? (
              <p className="mt-0.5 flex items-center gap-1 text-xs leading-[18px] text-ink-secondary">
                <Clock3 size={12} aria-hidden="true" />
                Free hosting, up to ~30s. Thanks for waiting.
              </p>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}
