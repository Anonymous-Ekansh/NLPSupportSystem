import { Activity } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { cn, formatLatency } from "@/lib/utils";

interface SessionCardProps {
  questions: number;
  averageLatency: number | null;
  helpfulVotes: number;
  className?: string;
}

export function SessionCard({
  questions,
  averageLatency,
  helpfulVotes,
  className,
}: SessionCardProps) {
  const stats = [
    {
      label: "Questions",
      value: String(questions),
      caption: "Asked in this chat",
    },
    {
      label: "Average time",
      value: formatLatency(averageLatency),
      caption: "Question to checked answer",
    },
    {
      label: "Helpful votes",
      value: String(helpfulVotes),
      caption: "Thumbs up from you",
    },
  ];

  return (
    <Card className={cn("p-4", className)}>
      <div className="flex items-center gap-1.5">
        <Activity
          size={14}
          strokeWidth={1.9}
          aria-hidden="true"
          className="text-ink-faint"
        />
        <h3 className="text-sm font-semibold text-ink">This session</h3>
      </div>
      <dl className="mt-3 grid grid-cols-3 gap-3">
        {stats.map((stat) => (
          <div key={stat.label}>
            <dt className="text-xs font-medium uppercase tracking-[0.1em] text-ink-secondary">
              {stat.label}
            </dt>
            <dd className="mt-1 text-base font-semibold tabular-nums text-ink">
              {stat.value}
            </dd>
            <p className="mt-0.5 text-xs leading-[18px] text-ink-faint">
              {stat.caption}
            </p>
          </div>
        ))}
      </dl>
      <p className="mt-3 border-t border-line pt-2.5 text-xs leading-[18px] text-ink-faint">
        Questions are logged to improve the system.
      </p>
    </Card>
  );
}
