"use client";

import React, { createContext, useContext, useState, useEffect, useCallback } from "react";
import type { ResolveResult, Health, Meta } from "./schemas";

interface AppState {
  currentRun: ResolveResult | null;
  setCurrentRun: (run: ResolveResult | null) => void;
  health: Health | null;
  isWaking: boolean;
  wakeService: () => Promise<void>;
  meta: Meta | null;
  theme: "light" | "dark";
  toggleTheme: () => void;
}

const AppContext = createContext<AppState>({
  currentRun: null,
  setCurrentRun: () => {},
  health: null,
  isWaking: false,
  wakeService: async () => {},
  meta: null,
  theme: "dark",
  toggleTheme: () => {},
});

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [currentRun, setCurrentRun] = useState<ResolveResult | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [isWaking, setIsWaking] = useState(false);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [theme, setTheme] = useState<"light" | "dark">("dark");

  const toggleTheme = useCallback(() => {
    setTheme((prev) => {
      const next = prev === "dark" ? "light" : "dark";
      if (typeof document !== "undefined") {
        document.documentElement.classList.toggle("dark", next === "dark");
        document.documentElement.setAttribute("data-theme", next);
      }
      return next;
    });
  }, []);

  const wakeService = useCallback(async () => {
    setIsWaking(true);
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      setHealth(data);
    } catch {
      setHealth({
        status: "unreachable",
        model_loaded: false,
        retrieval_backend: "offline",
      });
    } finally {
      setIsWaking(false);
    }
  }, []);

  useEffect(() => {
    // Initial fetch of health and meta asynchronously
    fetch("/api/health")
      .then((r) => r.json())
      .then((d) => setHealth(d))
      .catch(() =>
        setHealth({
          status: "unreachable",
          model_loaded: false,
          retrieval_backend: "offline",
        })
      );

    fetch("/api/meta")
      .then((r) => r.json())
      .then((d) => setMeta(d))
      .catch(() => {});
  }, []);

  useEffect(() => {
    // Set initial dark theme on html
    if (typeof document !== "undefined") {
      document.documentElement.classList.add("dark");
      document.documentElement.setAttribute("data-theme", "dark");
    }
  }, []);

  return (
    <AppContext.Provider
      value={{
        currentRun,
        setCurrentRun,
        health,
        isWaking,
        wakeService,
        meta,
        theme,
        toggleTheme,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}

export function useApp() {
  return useContext(AppContext);
}
