"use client";

import { useEffect, useRef, type ReactNode } from "react";
import { X } from "lucide-react";

interface BottomSheetProps {
  open: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
}

export function BottomSheet({
  open,
  onClose,
  title,
  children,
}: BottomSheetProps) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const titleId = "bottom-sheet-title";

  useEffect(() => {
    if (!open) return;
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeRef.current?.focus();

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKeyDown);

    return () => {
      document.body.style.overflow = previous;
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 lg:hidden">
      <div
        aria-hidden="true"
        onClick={onClose}
        className="absolute inset-0 animate-fade-in bg-ink/30"
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className="absolute inset-x-0 bottom-0 flex max-h-[88dvh] animate-sheet-up flex-col rounded-t-[24px] border-t border-line bg-card shadow-sheet"
      >
        <div className="shrink-0 border-b border-line px-4 pb-3 pt-2">
          <span
            aria-hidden="true"
            className="mx-auto mb-2 block h-1 w-9 rounded-pill bg-[#E4E4E1]"
          />
          <div className="flex items-center justify-between gap-3">
            <h2 id={titleId} className="text-base font-semibold text-ink">
              {title}
            </h2>
            <button
              ref={closeRef}
              type="button"
              onClick={onClose}
              aria-label="Close trace"
              className="flex size-8 items-center justify-center rounded-full border border-line bg-card text-ink-secondary transition-colors hover:bg-soft hover:text-ink"
            >
              <X size={16} aria-hidden="true" />
            </button>
          </div>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-4 pb-[calc(1rem+env(safe-area-inset-bottom))] pt-4">
          {children}
        </div>
      </div>
    </div>
  );
}
