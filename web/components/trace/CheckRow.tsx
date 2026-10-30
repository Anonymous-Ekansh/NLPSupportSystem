import { CheckCircle2, XCircle } from "lucide-react";
import { cn } from "@/lib/utils";

interface CheckRowProps {
  label: string;
  detail: string;
  passed: boolean;
}

export function CheckRow({ label, detail, passed }: CheckRowProps) {
  const Icon = passed ? CheckCircle2 : XCircle;
  return (
    <div className="flex items-start gap-2 border-b border-line py-2 last:border-b-0">
      <Icon
        size={15}
        strokeWidth={1.9}
        aria-hidden="true"
        className={cn(
          "mt-0.5 shrink-0",
          passed ? "text-success" : "text-error"
        )}
      />
      <div className="min-w-0 flex-1">
        <p className="text-xs font-medium leading-[18px] text-ink">{label}</p>
        <p className="text-xs leading-[18px] text-ink-secondary">{detail}</p>
      </div>
      <span
        className={cn(
          "shrink-0 text-xs font-medium",
          passed ? "text-success-ink" : "text-error-ink"
        )}
      >
        {passed ? "Passed" : "Failed"}
      </span>
    </div>
  );
}
