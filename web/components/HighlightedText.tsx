import React from "react";

export function HighlightedText({ text }: { text: string }) {
  if (!text) return null;

  // Split by {{...}}
  const parts = text.split(/(\{\{[^}]+\}\})/g);

  return (
    <span>
      {parts.map((part, i) => {
        if (part.startsWith("{{") && part.endsWith("}}")) {
          const placeholderName = part.slice(2, -2).trim();
          return (
            <span
              key={i}
              className="inline-flex items-center px-2 py-0.5 mx-0.5 rounded font-mono text-sm font-semibold bg-amber-500/15 text-amber-300 border border-amber-500/40"
              title={`Placeholder: ${placeholderName}`}
            >
              &#123;&#123;{placeholderName}&#125;&#125;
            </span>
          );
        }
        return <span key={i}>{part}</span>;
      })}
    </span>
  );
}
