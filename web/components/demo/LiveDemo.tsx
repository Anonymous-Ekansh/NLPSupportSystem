"use client";

import { useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  Gavel,
  HelpCircle,
  Keyboard,
  Languages,
  PackageSearch,
  Plus,
  RotateCcw,
  Sparkles,
} from "lucide-react";
import {
  ApiRequestError,
  OUT_OF_SCOPE_RESPONSE,
  fetchMeta,
  isOutOfScope,
  isUnsure,
  resolveQuestion,
  sendFeedback,
  type MapPoint,
  type Meta,
  type ResolveResult,
} from "@/lib/api";
import { cn } from "@/lib/utils";
import { Composer } from "@/components/chat/Composer";
import { ChatMessage } from "@/components/chat/ChatMessage";
import { TypingIndicator } from "@/components/chat/TypingIndicator";
import type { ChatMessageData } from "@/components/chat/types";
import { Chip, Dot } from "@/components/ui/Chip";
import { HowItWorked } from "@/components/trace/HowItWorked";
import { BottomSheet } from "@/components/demo/BottomSheet";
import { SessionCard } from "@/components/demo/SessionCard";

const EXAMPLE_QUERIES = [
  {
    label: "How can I request a refund for my purchase?",
    hint: "Answer",
    icon: RotateCcw,
  },
  {
    label: "I need a refund but cannot find my order number",
    hint: "Clarify · missing detail",
    icon: PackageSearch,
  },
  {
    label: "refund",
    hint: "Clarify · ambiguous",
    icon: HelpCircle,
  },
  {
    label: "i wanna cancel my order pls its been a week",
    hint: "Typo & slang",
    icon: Keyboard,
  },
  {
    label: "bhai mera refund kab aayega please jaldi batao",
    hint: "Hinglish",
    icon: Languages,
  },
  {
    label: "asdkjhqwoieu 🎉🎉",
    hint: "Gibberish",
    icon: Sparkles,
  },
  {
    label: "I will sue you over this charge",
    hint: "Escalate · legal",
    icon: Gavel,
  },
];

interface LiveDemoProps {
  mapPoints: MapPoint[];
  initialMeta: Meta | null;
}

function messageId(prefix: string) {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function LiveDemo({ mapPoints, initialMeta }: LiveDemoProps) {
  const [messages, setMessages] = useState<ChatMessageData[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [slow, setSlow] = useState(false);
  const [selected, setSelected] = useState<ResolveResult | null>(null);
  const [meta, setMeta] = useState<Meta | null>(initialMeta);
  const [ratings, setRatings] = useState<Record<string, 1 | 5>>({});
  const [sheetOpen, setSheetOpen] = useState(false);
  const [sheetResult, setSheetResult] = useState<ResolveResult | null>(null);

  const scrollRef = useRef<HTMLDivElement>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const slowTimer = useRef<number | null>(null);
  const metaRef = useRef<Meta | null>(initialMeta);

  const hasChat = messages.length > 0;

  useEffect(() => {
    let active = true;
    fetchMeta()
      .then((liveMeta) => {
        if (!active) return;
        setMeta(liveMeta);
        metaRef.current = liveMeta;
      })
      .catch(() => {
        // The checked-in config is already in state.
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!hasChat) return;
    const reduce = window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    ).matches;
    const behavior: ScrollBehavior = reduce ? "auto" : "smooth";

    if (window.matchMedia("(max-width: 1023px)").matches) {
      endRef.current?.scrollIntoView({ behavior, block: "end" });
    } else if (scrollRef.current) {
      scrollRef.current.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior,
      });
    }
  }, [messages, loading, hasChat]);

  useEffect(() => {
    return () => {
      if (slowTimer.current) window.clearTimeout(slowTimer.current);
    };
  }, []);

  const clearSlowTimer = () => {
    if (slowTimer.current) {
      window.clearTimeout(slowTimer.current);
      slowTimer.current = null;
    }
  };

  const send = async (raw: string, appendUser = true, answer?: string) => {
    const query = raw.trim();
    if (!query || loading) return;

    if (appendUser) {
      setMessages((previous) => [
        ...previous,
        {
          id: messageId("user"),
          role: "user",
          text: answer ? answer : query,
          createdAt: Date.now(),
        },
      ]);
    }

    setInput("");
    setLoading(true);
    setSlow(false);
    setSelected(null);
    clearSlowTimer();
    slowTimer.current = window.setTimeout(() => setSlow(true), 2500);

    try {
      // A clarification reply is sent with the original question so the
      // server can re-run the whole exchange (nothing is stored server-side).
      const result = await resolveQuestion(query, { answer });
      const sMin = metaRef.current?.a1?.s_min ?? null;
      const withheld = isOutOfScope(result, sMin);
      const unsure = isUnsure(result) && !withheld;

      setMessages((previous) => [
        ...previous,
        {
          id: messageId("assistant"),
          role: "assistant",
          text: withheld ? OUT_OF_SCOPE_RESPONSE : result.final_response,
          result,
          withheld,
          unsure,
          createdAt: Date.now(),
        },
      ]);
      setSelected(result);
    } catch (error) {
      const text =
        error instanceof ApiRequestError
          ? error.message
          : "We lost the connection. Check your internet and try again.";
      setMessages((previous) => [
        ...previous,
        {
          id: messageId("error"),
          role: "assistant",
          text,
          error: true,
          retryQuery: query,
          createdAt: Date.now(),
        },
      ]);
    } finally {
      setLoading(false);
      setSlow(false);
      clearSlowTimer();
    }
  };

  const replyToClarify = (originalQuery: string, answer: string) => {
    // The customer's reply is its own bubble; the request carries both texts.
    void send(originalQuery, true, answer);
  };

  const startNewChat = () => {
    if (loading) return;
    setMessages([]);
    setInput("");
    setSelected(null);
    setRatings({});
    setSheetOpen(false);
    setSheetResult(null);
  };

  const handleFeedback = (
    id: string,
    rating: 1 | 5,
    result: ResolveResult
  ) => {
    setRatings((previous) =>
      previous[id] ? previous : { ...previous, [id]: rating }
    );
    void sendFeedback({ query: result.query, rating }).catch(() => {
      // Feedback is best-effort; the UI keeps the customer's choice.
    });
  };

  const openTrace = (result: ResolveResult) => {
    setSheetResult(result);
    setSheetOpen(true);
  };

  const successfulResults = messages
    .map((message) => message.result)
    .filter((result): result is ResolveResult => Boolean(result));
  const questionCount = messages.filter(
    (message) => message.role === "user"
  ).length;
  const latencies = successfulResults
    .map((result) => result.total_latency_ms)
    .filter((value) => value > 0);
  const averageLatency =
    latencies.length > 0
      ? latencies.reduce((sum, value) => sum + value, 0) / latencies.length
      : null;
  const helpfulVotes = Object.values(ratings).filter(
    (rating) => rating === 5
  ).length;
  const session = {
    questions: questionCount,
    averageLatency,
    helpfulVotes,
  };

  return (
    <>
      <div className="mb-6 flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs font-medium uppercase tracking-[0.14em] text-ink-secondary">
          Live demo
        </p>
        <div className="flex flex-wrap items-center gap-1.5 text-xs text-ink-secondary">
          <span className="inline-flex items-center gap-1">
            <Dot tone="a1" />
            A1
          </span>
          <ArrowRight size={12} aria-hidden="true" className="text-ink-faint" />
          <span className="inline-flex items-center gap-1">
            <Dot tone="brand" />
            A3
          </span>
          <ArrowRight size={12} aria-hidden="true" className="text-ink-faint" />
          <span className="inline-flex items-center gap-1">
            <Dot tone="a2" />
            A2
          </span>
          <ArrowRight size={12} aria-hidden="true" className="text-ink-faint" />
          <span className="inline-flex items-center gap-1">
            <Dot tone="success" />
            A4
          </span>
          <ArrowRight size={12} aria-hidden="true" className="text-ink-faint" />
          <span className="inline-flex items-center gap-1">
            <Dot tone="brand" />
            A5
          </span>
        </div>
      </div>

      {!hasChat ? (
        <div className="flex min-h-[calc(100dvh-190px)] flex-col items-center justify-center pb-14 text-center">
          <h1 className="max-w-[700px] text-display font-semibold tracking-[-0.02em] text-ink">
            How can we help today?
          </h1>
          <p className="mt-2 max-w-[540px] text-base text-ink-secondary">
            Ask about orders, refunds, accounts, payments, shipping and more.
          </p>

          <div className="mt-7 w-full max-w-[720px] text-left">
            <Composer
              value={input}
              onChange={setInput}
              onSubmit={(query) => void send(query)}
              loading={loading}
              variant="hero"
              autoFocus
            />
          </div>

          <div className="mt-4 grid max-w-[880px] grid-cols-1 gap-2 sm:grid-cols-2">
            {EXAMPLE_QUERIES.map((example) => {
              const Icon = example.icon;
              return (
                <button
                  key={example.label}
                  type="button"
                  onClick={() => void send(example.label)}
                  className="flex items-start gap-2 rounded-[12px] border border-line bg-card px-3 py-2 text-left transition-colors hover:border-brand/20 hover:bg-brand/[0.04]"
                >
                  <Icon
                    size={15}
                    strokeWidth={1.8}
                    aria-hidden="true"
                    className="mt-0.5 shrink-0 text-ink-faint"
                  />
                  <span className="min-w-0">
                    <span className="block truncate text-sm text-ink">
                      {example.label}
                    </span>
                    <span className="block font-mono text-[11px] uppercase tracking-[0.06em] text-ink-faint">
                      {example.hint}
                    </span>
                  </span>
                </button>
              );
            })}
          </div>

          <p className="mt-5 inline-flex items-center gap-1.5 text-xs text-ink-faint">
            <Dot tone="a1" />
            Five agents
            <span aria-hidden="true">·</span>
            every answer is checked before it is sent
            <Dot tone="a2" />
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 items-start gap-6 lg:grid-cols-[55fr_45fr]">
          <section className="flex min-w-0 flex-col rounded-card border border-line bg-card lg:h-[calc(100dvh-130px)] lg:min-h-[560px]">
            <header className="flex shrink-0 items-center justify-between gap-3 border-b border-line px-4 py-3">
              <div className="min-w-0">
                <h2 className="text-base font-semibold text-ink">Chat</h2>
                <p className="text-xs text-ink-secondary">
                  A1 → A3 → A2 → A4 → A5. Every step is traceable.
                </p>
              </div>
              <button
                type="button"
                onClick={startNewChat}
                disabled={loading}
                className="inline-flex shrink-0 items-center gap-1.5 rounded-pill border border-line bg-card px-3 py-1.5 text-xs font-medium text-ink-secondary transition-colors hover:bg-soft hover:text-ink disabled:cursor-not-allowed disabled:opacity-40"
              >
                <Plus size={13} aria-hidden="true" />
                New chat
              </button>
            </header>

            <div
              ref={scrollRef}
              aria-live="polite"
              aria-relevant="additions"
              className="min-h-0 flex-1 space-y-5 overflow-y-auto px-4 py-5 lg:px-5"
            >
              {messages.map((message) => (
                <ChatMessage
                  key={message.id}
                  message={message}
                  rating={ratings[message.id]}
                  onFeedback={handleFeedback}
                  onFollowup={(query) => void send(query)}
                  onRetry={(query) => void send(query, false)}
                  onOpenTrace={openTrace}
                  onClarifyReply={replyToClarify}
                  clarifyLoading={loading}
                />
              ))}
              {loading ? <TypingIndicator slow={slow} /> : null}
              <div ref={endRef} />
            </div>

            <div className="sticky bottom-[calc(4.5rem+env(safe-area-inset-bottom))] z-20 shrink-0 border-t border-line bg-card px-3 pb-1 pt-3 sm:px-4 lg:static">
              <Composer
                value={input}
                onChange={setInput}
                onSubmit={(query) => void send(query)}
                loading={loading}
                variant="chat"
              />
            </div>
          </section>

          <aside className="hidden min-w-0 lg:block">
            <div className="lg:sticky lg:top-6 lg:max-h-[calc(100dvh-3rem)] lg:overflow-y-auto lg:pb-2 lg:pr-1">
              <HowItWorked
                result={selected}
                meta={meta}
                mapPoints={mapPoints}
                session={session}
                loading={loading}
              />
            </div>
          </aside>
        </div>
      )}

      {hasChat ? (
        <SessionCard
          className="mt-4 lg:hidden"
          questions={session.questions}
          averageLatency={session.averageLatency}
          helpfulVotes={session.helpfulVotes}
        />
      ) : null}

      <BottomSheet
        open={sheetOpen}
        onClose={() => setSheetOpen(false)}
        title="How I answered this"
      >
        <HowItWorked
          result={sheetResult}
          meta={meta}
          mapPoints={mapPoints}
          session={session}
          variant="sheet"
        />
      </BottomSheet>
    </>
  );
}
