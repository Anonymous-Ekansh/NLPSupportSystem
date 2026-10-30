"use client";

import React, { useState, useRef, useEffect } from "react";
import { Mic, Send, Volume2, ThumbsUp, ThumbsDown, AlertCircle, RotateCcw, VolumeX } from "lucide-react";
import { HighlightedText } from "./HighlightedText";
import type { ChatMessage } from "@/lib/voice";
import type { ResolveResult } from "@/lib/schemas";

interface CustomerChatProps {
  messages: ChatMessage[];
  isLoading: boolean;
  onSendMessage: (text: string, source?: "typed" | "voice") => Promise<void>;
  onSelectRun: (run: ResolveResult) => void;
  activeRunId?: string;
  onRetry: (text: string) => void;
}

export const SAMPLE_QUERIES = [
  { label: "Refund", text: "How can I get a refund for my purchase?" },
  { label: "Typos", text: "filing consumer claim against ur organizzation" },
  { label: "{{Order Number}}", text: "i need to check status for order {{Order Number}}" },
  { label: "Ambiguous", text: "it did not work" },
  { label: "Out of scope", text: "what is the weather today" },
  { label: "Cancellation", text: "how do I cancel my subscription?" },
];

export function CustomerChat({
  messages,
  isLoading,
  onSendMessage,
  onSelectRun,
  activeRunId,
  onRetry,
}: CustomerChatProps) {
  const [input, setInput] = useState("");
  const [toast, setToast] = useState<string | null>(null);
  const [feedbackState, setFeedbackState] = useState<Record<string, number>>({});
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const voiceEnabled = process.env.NEXT_PUBLIC_VOICE_ENABLED === "true";

  const showToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 3500);
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || trimmed.length > 500 || isLoading) return;
    setInput("");
    onSendMessage(trimmed, "typed");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    } else if (e.key === "Escape") {
      setInput("");
    }
  };

  const handleMicClick = () => {
    if (!voiceEnabled) {
      showToast("Speech-to-text is not connected yet");
    } else {
      showToast("No voice provider configured");
    }
  };

  const handleTtsClick = () => {
    if (!voiceEnabled) {
      showToast("Speech-to-text is not connected yet");
    } else {
      showToast("No voice provider configured");
    }
  };

  const handleFeedback = async (msgId: string, query: string, rating: number) => {
    setFeedbackState((prev) => ({ ...prev, [msgId]: rating }));
    try {
      const res = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, rating }),
      });
      if (res.ok) {
        showToast(rating === 5 ? "Feedback recorded! Thanks 👍" : "Feedback recorded 👎");
      } else {
        showToast("Feedback could not be saved");
      }
    } catch {
      showToast("Feedback could not be saved");
    }
  };

  const charCount = input.length;
  const isTooLong = charCount > 500;

  return (
    <div className="flex flex-col h-[760px] bg-[var(--card-bg)] rounded-xl border border-[var(--card-border)] shadow-sm overflow-hidden">
      {/* Toast Notification */}
      {toast && (
        <div className="bg-teal-600 text-white text-sm font-medium px-4 py-2 text-center animate-fadeIn transition-opacity">
          {toast}
        </div>
      )}

      {/* Header */}
      <div className="p-4 border-b border-[var(--card-border)] flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold tracking-tight text-[var(--foreground)]">
            Customer chat
          </h2>
          <p className="text-xs text-[var(--muted-text)] mt-0.5">
            Interact with the autonomous 2-agent resolution pipeline
          </p>
        </div>
        <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-800 text-gray-400 border border-gray-700">
          <VolumeX className="w-3 h-3" />
          Voice: coming soon
        </span>
      </div>

      {/* Sample Query Chips */}
      <div className="px-4 py-2.5 bg-[var(--card-hover)]/40 border-b border-[var(--card-border)] flex items-center gap-2 overflow-x-auto text-xs">
        <span className="text-[var(--muted-text)] font-semibold shrink-0">Try sample:</span>
        {SAMPLE_QUERIES.map((sample, i) => (
          <button
            key={i}
            onClick={() => onSendMessage(sample.text, "typed")}
            disabled={isLoading}
            className="shrink-0 px-2.5 py-1 rounded-full bg-[var(--card-bg)] hover:bg-teal-500/10 hover:border-teal-500 border border-[var(--card-border)] text-[var(--foreground)] transition-colors cursor-pointer text-xs"
            title={sample.text}
          >
            {sample.label}
          </button>
        ))}
      </div>

      {/* Messages Scroll Area */}
      <div
        className="flex-1 p-4 overflow-y-auto space-y-4"
        aria-live="polite"
        role="log"
      >
        {messages.length === 0 && !isLoading && (
          <div className="h-full flex flex-col items-center justify-center text-center p-6 text-[var(--muted-text)]">
            <div className="w-12 h-12 rounded-full bg-teal-500/10 border border-teal-500/30 flex items-center justify-center mb-3 text-teal-400">
              💬
            </div>
            <p className="text-base font-medium">Try a sample query above</p>
          </div>
        )}

        {messages.map((msg) => {
          const isUser = msg.role === "user";
          const res = msg.result;
          const a1 = res?.trace?.[0]?.output;
          const needsClar = a1?.needs_clarification;
          const isSelected = msg.runId && msg.runId === activeRunId;

          if (isUser) {
            return (
              <div key={msg.id} className="flex justify-end">
                <div
                  onClick={() => res && onSelectRun(res)}
                  className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-base font-normal shadow-sm bg-teal-600/90 text-white cursor-pointer hover:opacity-95 transition-all ${
                    isSelected ? "ring-2 ring-white" : ""
                  }`}
                >
                  <div className="flex items-center gap-1.5 justify-end">
                    {msg.source === "voice" && (
                      <span className="inline-flex items-center text-xs opacity-90 gap-1 bg-black/20 px-1.5 py-0.5 rounded">
                        <Mic className="w-3 h-3" /> Voice
                      </span>
                    )}
                    <span>{msg.text}</span>
                  </div>
                </div>
              </div>
            );
          }

          // Error bubble
          if (msg.id.startsWith("err-")) {
            return (
              <div key={msg.id} className="flex justify-start">
                <div className="max-w-[90%] rounded-2xl p-4 text-sm bg-rose-950/40 border border-rose-600/50 text-rose-200 space-y-2">
                  <div className="flex items-center gap-2">
                    <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
                    <span>{msg.text}</span>
                  </div>
                  <button
                    onClick={() => onRetry("How can I get a refund for my purchase?")}
                    className="inline-flex items-center gap-1.5 px-3 py-1 rounded bg-rose-900/60 hover:bg-rose-900 text-rose-200 border border-rose-500 text-xs font-semibold cursor-pointer"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                    <span>Retry</span>
                  </button>
                </div>
              </div>
            );
          }

          // Bot bubble
          return (
            <div key={msg.id} className="flex justify-start">
              <div
                onClick={() => res && onSelectRun(res)}
                className={`max-w-[90%] rounded-2xl p-4 text-base bg-gray-800/80 text-[var(--foreground)] border border-[var(--card-border)] shadow-sm space-y-3 cursor-pointer transition-all hover:border-teal-500/50 ${
                  isSelected ? "ring-2 ring-teal-400 border-teal-500" : ""
                }`}
              >
                {/* Main Response Text with Highlighted Placeholders */}
                <div className="text-[17px] leading-relaxed whitespace-pre-line">
                  <HighlightedText text={msg.text} />
                </div>

                {/* Level 2 Clarification Note if needs_clarification is true */}
                {needsClar && (
                  <div className="p-2.5 rounded-lg bg-amber-950/40 border border-amber-600/50 text-amber-200 text-xs flex items-start gap-2">
                    <AlertCircle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                    <span>
                      A1 was not confident. In Level 2 the Clarifier agent would ask a question here.
                    </span>
                  </div>
                )}

                {/* Meta details footer: category · intent · confidence % · cached demo badge */}
                {a1 && (
                  <div className="pt-2 border-t border-[var(--card-border)]/60 flex flex-wrap items-center justify-between text-xs text-[var(--muted-text)] gap-2">
                    <div className="flex items-center gap-1.5 font-mono">
                      <span className="px-1.5 py-0.5 rounded bg-gray-900 border border-gray-700 text-indigo-300">
                        {a1.category}
                      </span>
                      <span>·</span>
                      <span className="text-gray-300">{a1.intent}</span>
                      <span>·</span>
                      <span
                        className={
                          a1.confidence < 0.65 ? "text-amber-400 font-semibold" : "text-emerald-400"
                        }
                      >
                        {Math.round(a1.confidence * 100)}%
                      </span>
                      {msg.cached && (
                        <span className="px-1.5 py-0.5 rounded-full bg-teal-900/50 text-teal-300 border border-teal-600 text-[10px] font-sans">
                          cached demo
                        </span>
                      )}
                    </div>

                    {/* Controls: Audio play button (disabled) & Feedback */}
                    <div className="flex items-center gap-2">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleTtsClick();
                        }}
                        disabled={!voiceEnabled}
                        title="Voice output: coming soon"
                        aria-label="Voice output: coming soon"
                        className="p-1 rounded hover:bg-gray-700/60 text-gray-400 disabled:opacity-50 cursor-pointer disabled:cursor-not-allowed"
                      >
                        <Volume2 className="w-3.5 h-3.5" />
                      </button>

                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleFeedback(msg.id, msg.text, 5);
                        }}
                        className={`p-1 rounded hover:bg-gray-700/60 cursor-pointer transition-colors ${
                          feedbackState[msg.id] === 5 ? "text-emerald-400" : "text-gray-400"
                        }`}
                        title="Helpful (5/5)"
                      >
                        <ThumbsUp className="w-3.5 h-3.5" />
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleFeedback(msg.id, msg.text, 1);
                        }}
                        className={`p-1 rounded hover:bg-gray-700/60 cursor-pointer transition-colors ${
                          feedbackState[msg.id] === 1 ? "text-rose-400" : "text-gray-400"
                        }`}
                        title="Unhelpful (1/5)"
                      >
                        <ThumbsDown className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          );
        })}

        {/* Typing indicator bubble */}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-gray-800/80 border border-[var(--card-border)] rounded-2xl px-4 py-3 flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-teal-400 animate-bounce" />
              <span className="w-2 h-2 rounded-full bg-teal-400 animate-bounce [animation-delay:0.2s]" />
              <span className="w-2 h-2 rounded-full bg-teal-400 animate-bounce [animation-delay:0.4s]" />
              <span className="text-xs text-[var(--muted-text)] ml-2">Agents resolving...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Row */}
      <div className="p-3 border-t border-[var(--card-border)] bg-[var(--card-bg)] space-y-1">
        {isTooLong && (
          <p className="text-xs text-rose-400 px-1">Query must be ≤ 500 characters ({charCount}/500)</p>
        )}
        <form onSubmit={handleSubmit} className="flex items-center gap-2">
          {/* Disabled round teal microphone button */}
          <button
            type="button"
            onClick={handleMicClick}
            disabled={!voiceEnabled}
            title="Voice input: coming soon"
            aria-label="Voice input: coming soon"
            className="w-11 h-11 rounded-full flex items-center justify-center btn-teal disabled:opacity-40 disabled:cursor-not-allowed shrink-0 transition-transform active:scale-95 shadow-sm"
          >
            <Mic className="w-5 h-5" />
          </button>

          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type or tap the mic"
            disabled={isLoading}
            className="flex-1 px-4 py-2.5 rounded-lg bg-[var(--card-hover)] border border-[var(--card-border)] text-base text-[var(--foreground)] focus:outline-none focus:ring-2 focus:ring-teal-500 placeholder-[var(--muted-text)]"
          />

          <button
            type="submit"
            disabled={!input.trim() || isTooLong || isLoading}
            className="h-11 px-5 rounded-lg btn-teal font-semibold text-sm flex items-center gap-1.5 disabled:opacity-40 disabled:cursor-not-allowed shrink-0 transition-all cursor-pointer"
          >
            <Send className="w-4 h-4" />
            <span>Send</span>
          </button>
        </form>
      </div>
    </div>
  );
}
