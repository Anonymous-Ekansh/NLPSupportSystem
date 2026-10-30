"use client";

import React, { useState, useCallback } from "react";
import { CustomerChat } from "@/components/CustomerChat";
import { AgentTrace } from "@/components/AgentTrace";
import { useApp } from "@/lib/store";
import type { ChatMessage } from "@/lib/voice";
import type { ResolveResult } from "@/lib/schemas";
import { ChevronRight, X, Sparkles } from "lucide-react";

const TOUR_STEPS = [
  {
    title: "1. Clear Query (Refund)",
    query: "How can I get a refund for my purchase?",
    caption:
      "Observe a high-confidence classification (100%), filtered retrieval on the top predicted intent, and a green 'Answer' chip.",
  },
  {
    title: "2. Ambiguous Query",
    query: "it did not work",
    caption:
      "Observe low confidence falling below the threshold (needs_clarification=true), top-3 intent exploration, and amber 'Clarify' chip.",
  },
  {
    title: "3. Out-of-Scope Query",
    query: "what is the weather today",
    caption:
      "Observe how an out-of-distribution query receives low confidence and flags for Level-2 clarification/escalation.",
  },
];

export default function ResolvePage() {
  const { currentRun, setCurrentRun, meta } = useApp();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [activeRunId, setActiveRunId] = useState<string | undefined>(undefined);

  // Guided tour state
  const [tourIndex, setTourIndex] = useState<number | null>(null);

  const sendMessage = useCallback(
    async (text: string, source: "typed" | "voice" = "typed") => {
      setIsLoading(true);
      const userMsgId = `user-${Date.now()}`;

      // Append user message immediately
      const userMsg: ChatMessage = {
        id: userMsgId,
        role: "user",
        text,
        source,
        createdAt: Date.now(),
      };
      setMessages((prev) => [...prev, userMsg]);

      try {
        const res = await fetch("/api/resolve", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: text }),
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.error || `HTTP error ${res.status}`);
        }

        const data: ResolveResult = await res.json();
        const runId = `run-${Date.now()}`;

        // Link run to user message & state
        userMsg.runId = runId;
        userMsg.result = data;

        const botMsg: ChatMessage = {
          id: `bot-${Date.now()}`,
          role: "bot",
          text: data.final_response,
          source: "typed",
          runId,
          createdAt: Date.now(),
          result: data,
          cached: data.cached,
        };

        setMessages((prev) => [...prev, botMsg]);
        setCurrentRun(data);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Failed to contact service";
        const errMsg: ChatMessage = {
          id: `err-${Date.now()}`,
          role: "bot",
          text: `Error resolving query: ${msg}. Click below to retry.`,
          source: "typed",
          createdAt: Date.now(),
        };
        setMessages((prev) => [...prev, errMsg]);
      } finally {
        setIsLoading(false);
      }
    },
    [setCurrentRun]
  );

  const handleSelectRun = (run: ResolveResult) => {
    setCurrentRun(run);
    setActiveRunId(`run-inspect-${Date.now()}`);
  };

  const startTour = () => {
    setTourIndex(0);
    sendMessage(TOUR_STEPS[0].query, "typed");
  };

  const nextTourStep = () => {
    if (tourIndex == null) return;
    const nextIdx = tourIndex + 1;
    if (nextIdx < TOUR_STEPS.length) {
      setTourIndex(nextIdx);
      sendMessage(TOUR_STEPS[nextIdx].query, "typed");
    } else {
      setTourIndex(null); // Finish tour
    }
  };

  const stopTour = () => {
    setTourIndex(null);
  };

  const currentTourStep = tourIndex != null ? TOUR_STEPS[tourIndex] : null;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-6 space-y-4">
      {/* System Explanation Sentence at the Top */}
      <div className="bg-[var(--card-hover)]/70 border border-[var(--card-border)] rounded-xl px-4 py-3 flex items-center justify-between text-sm text-[var(--foreground)] shadow-xs">
        <p className="font-medium">
          Two-agent NLP support system:{" "}
          <strong className="text-indigo-400">Agent A1 (Intent Analyst)</strong> classifies the customer intent, and{" "}
          <strong className="text-teal-400">Agent A2 (Response Specialist)</strong> retrieves a grounded answer with self-checking.
        </p>
      </div>

      {/* Guided Tour Banner */}
      {currentTourStep && (
        <div className="bg-teal-950/70 border-2 border-teal-500 rounded-xl p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-sm animate-fadeIn">
          <div className="space-y-1">
            <div className="flex items-center gap-2 text-teal-300 font-bold text-base">
              <Sparkles className="w-4 h-4 text-teal-400" />
              <span>Tour: {currentTourStep.title}</span>
            </div>
            <p className="text-gray-300 text-xs sm:text-sm">{currentTourStep.caption}</p>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <button
              onClick={nextTourStep}
              disabled={isLoading}
              className="px-3.5 py-1.5 rounded-lg btn-teal text-xs font-semibold flex items-center gap-1.5 cursor-pointer disabled:opacity-50"
            >
              <span>{tourIndex === TOUR_STEPS.length - 1 ? "Finish Tour" : "Next Query"}</span>
              <ChevronRight className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={stopTour}
              className="px-2.5 py-1.5 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-400 text-xs font-medium flex items-center gap-1 cursor-pointer"
            >
              <X className="w-3.5 h-3.5" />
              <span>Stop</span>
            </button>
          </div>
        </div>
      )}

      {/* Two columns on desktop, stacked on mobile */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
        {/* Left Column: Customer Chat */}
        <CustomerChat
          messages={messages}
          isLoading={isLoading}
          onSendMessage={sendMessage}
          onSelectRun={handleSelectRun}
          activeRunId={activeRunId}
          onRetry={(query) => sendMessage(query, "typed")}
        />

        {/* Right Column: Agent Trace */}
        <AgentTrace
          run={currentRun}
          threshold={meta?.a1?.tau}
          onStartTour={startTour}
          tourActive={tourIndex !== null}
        />
      </div>
    </div>
  );
}
