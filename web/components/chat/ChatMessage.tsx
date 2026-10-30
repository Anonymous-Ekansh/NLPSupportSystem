"use client";

import {
  AlertCircle,
  ArrowUpRight,
  Clock3,
  ListTree,
  RefreshCw,
  ThumbsDown,
  ThumbsUp,
} from "lucide-react";
import {
  followupIntents,
  getA1Output,
  getA2Output,
  type ResolveResult,
} from "@/lib/api";
import {
  cn,
  formatLatency,
  humaniseIntent,
  humaniseIntentPhrase,
  topicLabel,
} from "@/lib/utils";
import type { ChatMessageData } from "@/components/chat/types";
import { PlaceholderText } from "@/components/chat/PlaceholderText";
import { Sources } from "@/components/chat/Sources";
import { Chip, Dot } from "@/components/ui/Chip";
import { LogoMark } from "@/components/ui/Logo";

interface ChatMessageProps {
  message: ChatMessageData;
  rating?: 1 | 5;
  onFeedback: (messageId: string, rating: 1 | 5, result: ResolveResult) => void;
  onFollowup: (query: string) => void;
  onRetry: (query: string) => void;
  onOpenTrace: (result: ResolveResult) => void;
}

export function ChatMessage({
  message,
  rating,
  onFeedback,
  onFollowup,
  onRetry,
  onOpenTrace,
}: ChatMessageProps) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-card rounded-br-[6px] border border-brand/10 bg-brand/[0.08] px-4 py-3 text-sm leading-[21px] text-ink">
          <p className="whitespace-pre-wrap break-words">{message.text}</p>
        </div>
      </div>
    );
  }

  if (message.error) {
    return (
      <div className="max-w-[92%]">
        <AssistantLabel />
        <div className="mt-1.5 rounded-card rounded-tl-[6px] border border-error/15 bg-error-soft px-4 py-3.5">
          <div className="flex gap-2.5">
            <AlertCircle
              size={16}
              strokeWidth={1.9}
              aria-hidden="true"
              className="mt-0.5 shrink-0 text-error"
            />
            <div className="min-w-0">
              <p className="text-sm leading-[21px] text-ink">
                {message.text}
              </p>
              {message.retryQuery ? (
                <button
                  type="button"
                  onClick={() => onRetry(message.retryQuery ?? "")}
                  className="mt-2 inline-flex items-center gap-1.5 rounded-pill border border-line bg-card px-2.5 py-1 text-xs font-medium text-ink transition-colors hover:bg-soft"
                >
                  <RefreshCw size={12} aria-hidden="true" />
                  Try again
                </button>
              ) : null}
            </div>
          </div>
        </div>
      </div>
    );
  }

  const result = message.result;
  const a1 = result ? getA1Output(result) : null;
  const followups = result && !message.withheld ? followupIntents(result) : [];
  const supportingExamples = result
    ? getA2Output(result)?.supporting_examples ?? []
    : [];

  return (
    <div className="max-w-[92%]">
      <AssistantLabel />
      <div className="mt-1.5 rounded-card rounded-tl-[6px] border border-line bg-card p-4">
        {message.withheld ? (
          <Chip tone="warning" className="mb-3">
            <Dot tone="warning" />
            Outside support topics
          </Chip>
        ) : null}

        {message.unsure && a1 ? (
          <div className="mb-3 rounded-[12px] border border-warning/20 bg-warning-soft px-3 py-2.5">
            <div className="flex gap-2">
              <AlertCircle
                size={14}
                strokeWidth={2}
                aria-hidden="true"
                className="mt-0.5 shrink-0 text-warning"
              />
              <div className="min-w-0">
                <p className="text-xs leading-[18px] text-warning-ink">
                  I&apos;m not fully sure — did you mean:
                </p>
                <div className="mt-1.5 flex flex-wrap gap-1.5">
                  {a1.top3.map((item) => (
                    <button
                      key={item.intent}
                      type="button"
                      onClick={() =>
                        onFollowup(humaniseIntentPhrase(item.intent))
                      }
                      className="rounded-pill border border-warning/20 bg-card px-2.5 py-0.5 text-xs font-medium text-ink transition-colors hover:bg-warning-soft"
                    >
                      {humaniseIntentPhrase(item.intent)}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        ) : null}

        <p className="text-sm leading-[22px] text-ink">
          <PlaceholderText text={message.text} />
        </p>

        {result ? (
          <>
            <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-2 border-t border-line pt-3">
              {a1 ? (
                <Chip tone="neutral">
                  {topicLabel(a1.category, a1.intent)}
                </Chip>
              ) : null}
              {result.cached ? (
                <Chip
                  tone="outline"
                  title="The live API was waking, so this is a saved demo run."
                >
                  Saved demo
                </Chip>
              ) : null}
              <span className="inline-flex items-center gap-1 text-xs text-ink-secondary">
                <Clock3 size={12} aria-hidden="true" />
                {formatLatency(result.total_latency_ms)}
              </span>
              {!message.withheld ? (
                <Sources examples={supportingExamples} />
              ) : null}
            </div>

            <div className="mt-3 flex flex-wrap items-center gap-2">
              <span className="text-xs text-ink-secondary">
                Was this helpful?
              </span>
              <button
                type="button"
                aria-label="Mark answer helpful"
                aria-pressed={rating === 5}
                disabled={Boolean(rating)}
                onClick={() => onFeedback(message.id, 5, result)}
                className={cn(
                  "flex size-7 items-center justify-center rounded-full border transition-colors disabled:cursor-default",
                  rating === 5
                    ? "border-success/20 bg-success-soft text-success-ink"
                    : "border-transparent text-ink-faint hover:bg-soft hover:text-ink-secondary"
                )}
              >
                <ThumbsUp size={14} aria-hidden="true" />
              </button>
              <button
                type="button"
                aria-label="Mark answer not helpful"
                aria-pressed={rating === 1}
                disabled={Boolean(rating)}
                onClick={() => onFeedback(message.id, 1, result)}
                className={cn(
                  "flex size-7 items-center justify-center rounded-full border transition-colors disabled:cursor-default",
                  rating === 1
                    ? "border-error/15 bg-error-soft text-error-ink"
                    : "border-transparent text-ink-faint hover:bg-soft hover:text-ink-secondary"
                )}
              >
                <ThumbsDown size={14} aria-hidden="true" />
              </button>
              {rating ? (
                <span className="text-xs text-success-ink">
                  Thanks for the feedback.
                </span>
              ) : null}
            </div>

            {!message.withheld && followups.length > 0 ? (
              <div className="mt-3 flex flex-wrap items-center gap-1.5">
                <span className="mr-0.5 text-xs text-ink-faint">Related:</span>
                {followups.map((intent) => (
                  <button
                    key={intent}
                    type="button"
                    onClick={() => onFollowup(humaniseIntentPhrase(intent))}
                    className="inline-flex items-center gap-1 rounded-pill border border-line bg-card px-2.5 py-0.5 text-xs font-medium text-ink-secondary transition-colors hover:border-brand/20 hover:bg-brand/[0.06] hover:text-brand-ink"
                    title={`Ask about ${humaniseIntent(intent).toLowerCase()}`}
                  >
                    {humaniseIntentPhrase(intent)}
                    <ArrowUpRight size={11} aria-hidden="true" />
                  </button>
                ))}
              </div>
            ) : null}

            <button
              type="button"
              onClick={() => onOpenTrace(result)}
              className="mt-3 inline-flex w-full items-center justify-center gap-1.5 rounded-pill border border-line bg-soft px-3 py-1.5 text-xs font-medium text-ink-secondary transition-colors hover:text-ink lg:hidden"
            >
              <ListTree size={13} aria-hidden="true" />
              How I answered this
            </button>
          </>
        ) : null}
      </div>
    </div>
  );
}

function AssistantLabel() {
  return (
    <div className="flex items-center gap-1.5 pl-1">
      <LogoMark size="sm" />
      <span className="text-xs font-semibold text-ink-secondary">
        ResolveIQ
      </span>
    </div>
  );
}
