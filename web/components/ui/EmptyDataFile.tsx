import { FileX2 } from "lucide-react";
import { Card } from "@/components/ui/Card";

export function EmptyDataFile({
  file,
  detail,
  className,
}: {
  file: string;
  detail?: string;
  className?: string;
}) {
  return (
    <Card className={className}>
      <div className="flex flex-col items-center justify-center px-4 py-8 text-center">
        <span className="flex size-9 items-center justify-center rounded-full border border-line bg-soft text-ink-faint">
          <FileX2 size={16} strokeWidth={1.8} aria-hidden="true" />
        </span>
        <p className="mt-2.5 text-sm font-medium text-ink">
          Waiting for {file}
        </p>
        <p className="mx-auto mt-1 max-w-[320px] text-xs leading-[18px] text-ink-secondary">
          {detail ??
            `Add ${file} to web/public/data/ to fill this card. No numbers are hard-coded.`}
        </p>
      </div>
    </Card>
  );
}
