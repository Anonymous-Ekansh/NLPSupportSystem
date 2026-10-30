"use client";

import { useEffect, useState } from "react";
import { fetchHealth } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Dot } from "@/components/ui/Chip";

type PillState = "checking" | "online" | "waking";

export function HealthPill() {
  const [state, setState] = useState<PillState>("checking");
  const [backend, setBackend] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    const check = async () => {
      try {
        const health = await fetchHealth();
        if (!active) return;
        setState(health.status === "ok" ? "online" : "waking");
        setBackend(health.retrieval_backend);
      } catch {
        if (!active) return;
        setState("waking");
        setBackend(null);
      }
    };

    void check();
    const interval = window.setInterval(check, 30000);
    const onVisible = () => {
      if (document.visibilityState === "visible") void check();
    };
    document.addEventListener("visibilitychange", onVisible);

    return () => {
      active = false;
      window.clearInterval(interval);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, []);

  const label =
    state === "checking"
      ? "Checking..."
      : state === "online"
        ? "Online"
        : "Waking up...";

  const title =
    state === "online"
      ? `The assistant is online${backend ? ` · ${backend} retrieval` : ""}`
      : state === "waking"
        ? "The assistant is waking up. Free hosting can take up to about 30 seconds."
        : "Checking the assistant status";

  return (
    <span
      role="status"
      title={title}
      className={cn(
        "inline-flex items-center gap-1.5 rounded-pill border px-2.5 py-1 text-xs font-medium transition-colors",
        state === "online" &&
          "border-success/15 bg-success-soft text-success-ink",
        state === "waking" &&
          "border-warning/15 bg-warning-soft text-warning-ink",
        state === "checking" &&
          "border-line bg-soft text-ink-secondary"
      )}
    >
      {state === "checking" ? (
        <span
          aria-hidden="true"
          className="size-1.5 animate-pulse rounded-full bg-ink-faint"
        />
      ) : (
        <Dot tone={state === "online" ? "success" : "warning"} />
      )}
      {label}
    </span>
  );
}
