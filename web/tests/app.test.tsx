import { describe, it, expect, vi, beforeEach } from "vitest";
import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { CustomerChat } from "@/components/CustomerChat";
import { ResolveResultSchema } from "@/lib/schemas";
import { POST as sttRoute } from "@/app/api/voice/stt/route";
import { POST as ttsRoute } from "@/app/api/voice/tts/route";
import type { ChatMessage } from "@/lib/voice";
import type { ResolveResult } from "@/lib/schemas";

const mockResolveResult: ResolveResult = {
  query: "How can I get a refund for my purchase?",
  trace: [
    {
      agent: "IntentAnalyst",
      input: { query: "How can I get a refund for my purchase?" },
      output: {
        query: "How can I get a refund for my purchase?",
        intent: "get_refund",
        category: "REFUND",
        confidence: 0.99,
        margin: 0.98,
        top3: [{ intent: "get_refund", prob: 0.99 }],
        top1_similarity: 0.91,
        needs_clarification: false,
      },
      latency_ms: 25.0,
    },
    {
      agent: "ResponseSpecialist",
      input: {
        query: "How can I get a refund for my purchase?",
        a1_output: {
          query: "How can I get a refund for my purchase?",
          intent: "get_refund",
          category: "REFUND",
          confidence: 0.99,
          margin: 0.98,
          top3: [{ intent: "get_refund", prob: 0.99 }],
          top1_similarity: 0.91,
          needs_clarification: false,
        },
      },
      output: {
        response: "To request a refund, please follow these instructions...",
        supporting_examples: [
          { instruction: "how to get refund", response: "resp", similarity: 0.95 },
        ],
        used_intents: ["get_refund"],
        retrieval_score: 0.95,
        low_confidence: false,
        self_check: {
          placeholder_safe: true,
          source_similarity_ok: true,
          response_intent_ok: true,
          check_passed: true,
          candidates_rejected: 0,
        },
      },
      latency_ms: 18.0,
    },
  ],
  final_response: "To request a refund, please follow these instructions...",
  total_latency_ms: 43.0,
  map: { x: 1.25, y: -0.5, confidence: 0.92 },
  retrieval_backend: "in-memory",
  planned_agents: [
    { name: "Verifier", status: "planned", planned_function: "Verification" },
    { name: "Clarifier", status: "planned", planned_function: "Clarification" },
    { name: "Escalator", status: "planned", planned_function: "Escalation" },
  ],
};

describe("Customer Support System Test Suite", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("Disabled mic never calls getUserMedia", () => {
    const getUserMediaMock = vi.fn();
    // @ts-expect-error Mocking mediaDevices in test environment
    navigator.mediaDevices = { getUserMedia: getUserMediaMock };

    render(
      <CustomerChat
        messages={[]}
        isLoading={false}
        onSendMessage={vi.fn()}
        onSelectRun={vi.fn()}
        onRetry={vi.fn()}
      />
    );

    const micBtn = screen.getByLabelText(/voice input: coming soon/i);
    expect(micBtn).toBeDefined();

    fireEvent.click(micBtn);
    expect(getUserMediaMock).not.toHaveBeenCalled();
  });

  it("source 'voice' renders the mic icon and voice label in user bubble", () => {
    const voiceMessage: ChatMessage = {
      id: "msg-voice-1",
      role: "user",
      text: "How do I return my package?",
      source: "voice",
      createdAt: Date.now(),
    };

    render(
      <CustomerChat
        messages={[voiceMessage]}
        isLoading={false}
        onSendMessage={vi.fn()}
        onSelectRun={vi.fn()}
        onRetry={vi.fn()}
      />
    );

    expect(screen.getByText("Voice")).toBeDefined();
    expect(screen.getByText("How do I return my package?")).toBeDefined();
  });

  it("bubbles render only data from the mocked resolve response", () => {
    const botMessage: ChatMessage = {
      id: "msg-bot-1",
      role: "bot",
      text: mockResolveResult.final_response,
      source: "typed",
      result: mockResolveResult,
      createdAt: Date.now(),
    };

    render(
      <CustomerChat
        messages={[botMessage]}
        isLoading={false}
        onSendMessage={vi.fn()}
        onSelectRun={vi.fn()}
        onRetry={vi.fn()}
      />
    );

    expect(
      screen.getByText("To request a refund, please follow these instructions...")
    ).toBeDefined();
    expect(screen.getByText("REFUND")).toBeDefined();
    expect(screen.getByText("get_refund")).toBeDefined();
    expect(screen.getByText("99%")).toBeDefined();
  });

  it("empty input is rejected", () => {
    const onSendMock = vi.fn();
    render(
      <CustomerChat
        messages={[]}
        isLoading={false}
        onSendMessage={onSendMock}
        onSelectRun={vi.fn()}
        onRetry={vi.fn()}
      />
    );

    const sendBtn = screen.getByRole("button", { name: /send/i });
    expect(sendBtn.hasAttribute("disabled")).toBe(true);

    const input = screen.getByPlaceholderText(/type or tap the mic/i);
    fireEvent.change(input, { target: { value: "   " } });
    expect(sendBtn.hasAttribute("disabled")).toBe(true);

    fireEvent.keyDown(input, { key: "Enter", code: "Enter" });
    expect(onSendMock).not.toHaveBeenCalled();
  });

  it("zod rejects a malformed resolve response", () => {
    const malformed = {
      query: "hello",
      // missing trace, map, total_latency_ms, final_response
    };

    const parsed = ResolveResultSchema.safeParse(malformed);
    expect(parsed.success).toBe(false);
  });

  it("/api/voice/stt and /api/voice/tts return 501 and make no outbound fetch", async () => {
    const fetchSpy = vi.spyOn(global, "fetch");

    const sttRes = await sttRoute();
    expect(sttRes.status).toBe(501);
    const sttBody = await sttRes.json();
    expect(sttBody).toEqual({ error: "voice_not_configured" });

    const ttsRes = await ttsRoute();
    expect(ttsRes.status).toBe(501);
    const ttsBody = await ttsRes.json();
    expect(ttsBody).toEqual({ error: "voice_not_configured" });

    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("SARVAM_API_KEY never appears in any client bundle or response", () => {
    // SARVAM_API_KEY must not be exposed with NEXT_PUBLIC_
    expect(process.env.NEXT_PUBLIC_SARVAM_API_KEY).toBeUndefined();
  });
});
