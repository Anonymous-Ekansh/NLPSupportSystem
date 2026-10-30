import { Fragment } from "react";

/**
 * Renders {{Order Number}}-style dataset slots as intentional chips while
 * preserving the line breaks in the retrieved support answer.
 */
export function PlaceholderText({ text }: { text: string }) {
  const parts = text.split(/(\{\{[^}]+\}\})/g);

  return (
    <span className="whitespace-pre-wrap break-words">
      {parts.map((part, index) => {
        const match = part.match(/^\{\{([^}]+)\}\}$/);
        if (!match) {
          return <Fragment key={index}>{part}</Fragment>;
        }
        return (
          <span
            key={index}
            title="Customer detail to fill in"
            className="mx-0.5 inline-flex items-center rounded-pill border border-line bg-soft px-2 py-0.5 font-mono text-xs leading-[18px] text-ink-secondary"
          >
            [{match[1]}]
          </span>
        );
      })}
    </span>
  );
}
