"use client";

import {
  useEffect,
  useId,
  useRef,
  useState,
  type FormEvent,
} from "react";
import { ArrowUp, Mic } from "lucide-react";
import { cn } from "@/lib/utils";

interface SpeechResultAlternative {
  transcript: string;
}

interface SpeechResultItem {
  isFinal: boolean;
  0: SpeechResultAlternative;
  length: number;
}

interface SpeechRecognitionEventLike {
  results: {
    length: number;
    [index: number]: SpeechResultItem;
  };
}

interface SpeechRecognitionLike {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: (() => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
}

type SpeechRecognitionConstructor = new () => SpeechRecognitionLike;

function getSpeechConstructor(): SpeechRecognitionConstructor | null {
  if (typeof window === "undefined") return null;
  const speechWindow = window as typeof window & {
    SpeechRecognition?: SpeechRecognitionConstructor;
    webkitSpeechRecognition?: SpeechRecognitionConstructor;
  };
  return (
    speechWindow.SpeechRecognition ??
    speechWindow.webkitSpeechRecognition ??
    null
  );
}

interface ComposerProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: (value: string) => void;
  loading?: boolean;
  maxLength?: number;
  variant?: "hero" | "chat";
  autoFocus?: boolean;
  className?: string;
}

export function Composer({
  value,
  onChange,
  onSubmit,
  loading = false,
  maxLength = 500,
  variant = "chat",
  autoFocus = false,
  className,
}: ComposerProps) {
  const [speechSupported, setSpeechSupported] = useState(false);
  const [listening, setListening] = useState(false);
  const [speechError, setSpeechError] = useState<string | null>(null);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const counterId = useId();

  useEffect(() => {
    setSpeechSupported(Boolean(getSpeechConstructor()));
    return () => {
      recognitionRef.current?.stop();
    };
  }, []);

  const stopListening = () => {
    recognitionRef.current?.stop();
    setListening(false);
  };

  const toggleListening = () => {
    if (listening) {
      stopListening();
      return;
    }

    const Constructor = getSpeechConstructor();
    if (!Constructor) return;

    setSpeechError(null);
    const recognition = new Constructor();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onresult = (event) => {
      let transcript = "";
      for (let index = 0; index < event.results.length; index += 1) {
        transcript += event.results[index][0].transcript;
      }
      onChange(transcript.slice(0, maxLength));
    };
    recognition.onerror = () => {
      setSpeechError("The microphone is unavailable. You can type instead.");
      setListening(false);
    };
    recognition.onend = () => setListening(false);

    recognitionRef.current = recognition;
    try {
      recognition.start();
      setListening(true);
    } catch {
      setSpeechError("The microphone is unavailable. You can type instead.");
    }
  };

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmed = value.trim();
    if (!trimmed || loading) return;
    stopListening();
    onSubmit(trimmed);
  };

  const hero = variant === "hero";
  const nearLimit = value.length >= maxLength - 50;

  return (
    <form
      onSubmit={handleSubmit}
      className={cn("w-full", className)}
      aria-label="Ask a support question"
    >
      <div
        className={cn(
          "flex w-full items-center gap-2 border border-line bg-card transition-colors focus-within:border-brand/40",
          hero
            ? "rounded-pill px-3 py-2 shadow-float"
            : "rounded-pill px-2.5 py-1.5"
        )}
      >
        <label htmlFor={`composer-${counterId}`} className="sr-only">
          Your support question
        </label>
        <input
          id={`composer-${counterId}`}
          value={value}
          onChange={(event) => onChange(event.target.value.slice(0, maxLength))}
          maxLength={maxLength}
          autoFocus={autoFocus}
          autoComplete="off"
          enterKeyHint="send"
          aria-describedby={value.length > 0 ? counterId : undefined}
          placeholder="Ask about an order, refund, account or delivery..."
          className={cn(
            "min-w-0 flex-1 bg-transparent text-ink placeholder:text-ink-faint focus:outline-none",
            hero ? "h-11 px-2 text-base" : "h-9 px-2 text-sm"
          )}
        />

        {speechSupported ? (
          <button
            type="button"
            onClick={toggleListening}
            aria-label={listening ? "Stop voice input" : "Start voice input"}
            aria-pressed={listening}
            className={cn(
              "flex shrink-0 items-center justify-center rounded-full border transition-colors",
              hero ? "size-9" : "size-8",
              listening
                ? "border-error/20 bg-error-soft text-error-ink"
                : "border-transparent text-ink-faint hover:bg-soft hover:text-ink-secondary"
            )}
          >
            <Mic size={hero ? 17 : 15} strokeWidth={1.8} aria-hidden="true" />
          </button>
        ) : null}

        <button
          type="submit"
          disabled={!value.trim() || loading}
          aria-label="Send question"
          className={cn(
            "flex shrink-0 items-center justify-center rounded-full bg-brand text-white transition-colors hover:bg-brand-ink disabled:cursor-not-allowed disabled:bg-[#E4E4E1] disabled:text-[#A1A1AA]",
            hero ? "size-11" : "size-9"
          )}
        >
          <ArrowUp
            size={hero ? 19 : 16}
            strokeWidth={2}
            aria-hidden="true"
          />
        </button>
      </div>

      <div className="flex min-h-[18px] items-start justify-between gap-3 px-1">
        <p
          aria-live="polite"
          className={cn(
            "text-xs leading-[18px]",
            speechError ? "text-error-ink" : "text-ink-faint"
          )}
        >
          {listening
            ? "Listening — speak now."
            : speechError ?? (hero ? "Press Enter to ask." : "")}
        </p>
        {value.length > 0 ? (
          <p
            id={counterId}
            className={cn(
              "shrink-0 font-mono text-xs tabular-nums",
              nearLimit ? "text-warning-ink" : "text-ink-faint"
            )}
          >
            {value.length}/{maxLength}
          </p>
        ) : null}
      </div>
    </form>
  );
}
