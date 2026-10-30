"use client";

import React, { createContext, useContext, useMemo } from "react";
import type { ResolveResult } from "./schemas";

export type ChatMessage = {
  id: string;
  role: "user" | "bot";
  text: string;
  source: "typed" | "voice";
  runId?: string;
  createdAt: number;
  result?: ResolveResult;
  cached?: boolean;
};

export interface SpeechToText {
  start(): Promise<void>;
  stop(): Promise<{ transcript: string; languageCode?: string }>;
  supported(): boolean;
}

export interface TextToSpeech {
  speak(text: string, opts?: { languageCode?: string }): Promise<void>;
  cancel(): void;
  supported(): boolean;
}

export const stubSpeechToText: SpeechToText = {
  async start() {
    throw new Error("Not implemented");
  },
  async stop() {
    throw new Error("Not implemented");
  },
  supported() {
    return false;
  },
};

export const stubTextToSpeech: TextToSpeech = {
  async speak() {
    throw new Error("Not implemented");
  },
  cancel() {},
  supported() {
    return false;
  },
};

export interface VoiceContextValue {
  stt: SpeechToText;
  tts: TextToSpeech;
  voiceEnabled: boolean;
}

const VoiceContext = createContext<VoiceContextValue>({
  stt: stubSpeechToText,
  tts: stubTextToSpeech,
  voiceEnabled: false,
});

export function VoiceProvider({
  children,
  stt = stubSpeechToText,
  tts = stubTextToSpeech,
}: {
  children: React.ReactNode;
  stt?: SpeechToText;
  tts?: TextToSpeech;
}) {
  const voiceEnabled = process.env.NEXT_PUBLIC_VOICE_ENABLED === "true";

  const value = useMemo(
    () => ({
      stt,
      tts,
      voiceEnabled,
    }),
    [stt, tts, voiceEnabled]
  );

  return <VoiceContext.Provider value={value}>{children}</VoiceContext.Provider>;
}

export function useVoice() {
  return useContext(VoiceContext);
}
