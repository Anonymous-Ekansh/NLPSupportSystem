"use client";

import { useId, useState } from "react";
import { ChevronDown } from "lucide-react";
import type { SupportingExample } from "@/lib/api";
import { clamp } from "@/lib/utils";
import { cn } from "@/lib/utils";

export function Sources({ examples }: { examples: SupportingExample[] }) {
  const [open, setOpen] = useState(false);
  const panelId = useId();

  if (examples.length === 0) return null;

  return (
    <div>
      <button
        type="button"
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => setOpen((value) => !value)}
        className="inline-flex items-center gap-1 rounded-pill text-xs font-medium text-ink-secondary transition-colors hover:text-ink"
      >
        Sources
        <span className="font-mono text-ink-faint">{examples.length}</span>
        <ChevronDown
          size={13}
          aria-hidden="true"
          className={cn("transition-transform", open && "rotate-180")}
        />
      </button>

      {open ? (
        <div
          id={panelId}
          className="mt-2 animate-fade-in rounded-[12px] border border-line bg-soft px-3 py-1"
        >
          <ul>
            {examples.map((example, index) => (
              <li
                key={`${example.instruction}-${index}`}
                className="border-b border-line py-2.5 last:border-b-0"
              >
                <p className="text-xs font-medium leading-[18px] text-ink">
                  “{example.instruction}”
                </p>
                <p className="mt-0.5 line-clamp-2 text-xs leading-[18px] text-ink-secondary">
                  {example.response}
                </p>
                <div className="mt-1.5 flex items-center gap-2">
                  <span className="h-1 w-20 overflow-hidden rounded-pill bg-[#E8E8E4]">
                    <span
                      className="block h-full rounded-pill bg-a2"
                      style={{ width: `${clamp(example.similarity) * 100}%` }}
                    />
                  </span>
                  <span className="font-mono text-xs tabular-nums text-ink">
                    {(clamp(example.similarity) * 100).toFixed(1)}%
                  </span>
                  <span className="text-xs text-ink-faint">match</span>
                </div>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}
