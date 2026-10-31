"use client";

import { useState } from "react";
import { CornerDownLeft, MessageCircleQuestion } from "lucide-react";
import type { MissingSlot } from "@/lib/api";

interface ClarifyReplyProps {
  /** The original question; the reply is appended to it server-side. */
  query: string;
  missing?: MissingSlot[];
  loading?: boolean;
  onSend: (answer: string) => void;
}

/**
 * Reply box for a CLARIFY decision. Sends the original query plus the answer
 * as one stateless request, so the pipeline re-reads the whole exchange.
 */
export function ClarifyReply({
  query,
  missing = [],
  loading = false,
  onSend,
}: ClarifyReplyProps) {
  const [value, setValue] = useState("");
  const primary = missing[0];
  const placeholder = primary
    ? `Your ${primary.label}, e.g. 123456`
    : "Type the missing detail…";

  const submit = () => {
    const answer = value.trim();
    if (!answer || loading) return;
    setValue("");
    onSend(answer);
  };

  return (
    <div className="mt-3 rounded-[12px] border border-warning/25 bg-warning-soft p-2.5">
      <div className="flex items-center gap-1.5">
        <MessageCircleQuestion
          size={13}
          strokeWidth={2}
          aria-hidden="true"
          className="shrink-0 text-warning-ink"
        />
        <p className="text-xs font-medium text-warning-ink">
          Reply to continue
          {primary ? (
            <span className="font-normal text-ink-secondary">
              {" "}
              — this question is waiting on your {primary.label}
            </span>
          ) : null}
        </p>
      </div>

      <form
        className="mt-1.5 flex items-center gap-1.5"
        onSubmit={(event) => {
          event.preventDefault();
          submit();
        }}
      >
        <label className="sr-only" htmlFor={`clarify-${query.slice(0, 12)}`}>
          Answer the clarification question
        </label>
        <input
          id={`clarify-${query.slice(0, 12)}`}
          value={value}
          onChange={(event) => setValue(event.target.value)}
          placeholder={placeholder}
          disabled={loading}
          className="min-w-0 flex-1 rounded-[8px] border border-line bg-card px-2.5 py-1.5 text-sm text-ink placeholder:text-ink-faint focus:border-warning/40 focus:outline-none disabled:opacity-60"
        />
        <button
          type="submit"
          disabled={loading || value.trim().length === 0}
          className="inline-flex shrink-0 items-center gap-1 rounded-[8px] border border-warning/30 bg-card px-2.5 py-1.5 text-xs font-medium text-warning-ink transition-colors hover:bg-warning-soft disabled:cursor-not-allowed disabled:opacity-40"
        >
          <CornerDownLeft size={12} aria-hidden="true" />
          Send
        </button>
      </form>
    </div>
  );
}
