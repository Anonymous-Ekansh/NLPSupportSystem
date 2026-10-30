"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useApp } from "@/lib/store";
import { Sun, Moon, VolumeX, RefreshCw } from "lucide-react";

export function Navbar() {
  const pathname = usePathname();
  const { health, isWaking, wakeService, theme, toggleTheme } = useApp();
  const voiceEnabled = process.env.NEXT_PUBLIC_VOICE_ENABLED === "true";

  // Determine health chip color & text
  let chipColor = "bg-gray-700/50 text-gray-300 border-gray-600";
  let statusText = "unknown";
  let isPulsing = false;

  if (isWaking) {
    chipColor = "bg-amber-950/60 text-amber-300 border-amber-600";
    statusText = "waking...";
    isPulsing = true;
  } else if (health) {
    if (health.model_loaded) {
      chipColor = "bg-emerald-950/70 text-emerald-300 border-emerald-500";
      statusText = "ready";
    } else if (health.status === "unreachable") {
      chipColor = "bg-rose-950/70 text-rose-300 border-rose-500";
      statusText = "unreachable";
    } else {
      chipColor = "bg-amber-950/60 text-amber-300 border-amber-600";
      statusText = health.status;
    }
  }

  const backendName = health?.retrieval_backend || "offline";

  return (
    <header className="sticky top-0 z-40 w-full border-b backdrop-blur border-[var(--card-border)] bg-[var(--card-bg)]/80">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 h-18 flex items-center justify-between">
        {/* Brand / Logo */}
        <div className="flex items-center gap-6">
          <Link href="/" className="flex items-center gap-2 group">
            <div className="w-8 h-8 rounded-lg bg-teal-500/20 border border-teal-500 flex items-center justify-center font-bold text-teal-400 group-hover:scale-105 transition-transform">
              AI
            </div>
            <span className="font-bold tracking-tight text-xl text-[var(--foreground)]">
              Support<span className="text-teal-400">Orchestrator</span>
            </span>
          </Link>

          {/* Navigation links */}
          <nav className="hidden md:flex items-center gap-1 font-medium text-base">
            <Link
              href="/"
              className={`px-3 py-1.5 rounded-md transition-colors ${
                pathname === "/"
                  ? "bg-teal-500/15 text-teal-400 font-semibold"
                  : "text-[var(--muted-text)] hover:text-[var(--foreground)] hover:bg-[var(--card-hover)]"
              }`}
            >
              Resolve
            </Link>
            <Link
              href="/map"
              className={`px-3 py-1.5 rounded-md transition-colors ${
                pathname === "/map"
                  ? "bg-teal-500/15 text-teal-400 font-semibold"
                  : "text-[var(--muted-text)] hover:text-[var(--foreground)] hover:bg-[var(--card-hover)]"
              }`}
            >
              Map
            </Link>
            <Link
              href="/evidence"
              className={`px-3 py-1.5 rounded-md transition-colors ${
                pathname === "/evidence"
                  ? "bg-teal-500/15 text-teal-400 font-semibold"
                  : "text-[var(--muted-text)] hover:text-[var(--foreground)] hover:bg-[var(--card-hover)]"
              }`}
            >
              Evidence
            </Link>
            <Link
              href="/data"
              className={`px-3 py-1.5 rounded-md transition-colors ${
                pathname === "/data"
                  ? "bg-teal-500/15 text-teal-400 font-semibold"
                  : "text-[var(--muted-text)] hover:text-[var(--foreground)] hover:bg-[var(--card-hover)]"
              }`}
            >
              Data
            </Link>
          </nav>
        </div>

        {/* Right controls: Health Chip, Voice Badge, Theme Toggle */}
        <div className="flex items-center gap-3">
          {/* Voice Feature Flag Badge */}
          <div
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${
              voiceEnabled
                ? "bg-teal-950/40 text-teal-300 border-teal-600"
                : "bg-gray-800/40 text-gray-400 border-gray-700"
            }`}
            title={voiceEnabled ? "Voice Enabled" : "Voice input/output is not enabled yet"}
          >
            <VolumeX className="w-3.5 h-3.5" />
            <span>Voice: {voiceEnabled ? "active" : "coming soon"}</span>
          </div>

          {/* Health Chip / Wake Service button */}
          <button
            onClick={wakeService}
            disabled={isWaking}
            className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold border transition-all cursor-pointer hover:opacity-90 active:scale-95 ${chipColor} ${
              isPulsing ? "animate-pulse" : ""
            }`}
            title="Click to check / wake NLP service"
          >
            <span
              className={`w-2 h-2 rounded-full ${
                health?.model_loaded
                  ? "bg-emerald-400"
                  : health?.status === "unreachable"
                  ? "bg-rose-500"
                  : "bg-amber-400"
              }`}
            />
            <span>NLP: {statusText}</span>
            <span className="opacity-75 font-normal">({backendName})</span>
            <RefreshCw className={`w-3 h-3 ${isWaking ? "animate-spin" : ""}`} />
          </button>

          {/* Theme toggle */}
          <button
            onClick={toggleTheme}
            aria-label="Toggle theme"
            className="p-2 rounded-lg border border-[var(--card-border)] bg-[var(--card-bg)] text-[var(--muted-text)] hover:text-[var(--foreground)] hover:bg-[var(--card-hover)] transition-colors"
          >
            {theme === "dark" ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>
        </div>
      </div>
    </header>
  );
}
