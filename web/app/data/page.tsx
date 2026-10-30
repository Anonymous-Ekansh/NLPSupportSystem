import React from "react";
import Link from "next/link";
import { ArrowLeft, Database } from "lucide-react";

export default function DataPage() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-16 text-center space-y-4">
      <div className="w-16 h-16 rounded-2xl bg-teal-500/10 border border-teal-500/30 flex items-center justify-center text-teal-400 mx-auto">
        <Database className="w-8 h-8" />
      </div>
      <h1 className="text-3xl font-bold tracking-tight">Dataset Explorer</h1>
      <p className="text-base text-[var(--muted-text)]">
        Coming in step 2
      </p>
      <div className="pt-4">
        <Link
          href="/"
          className="inline-flex items-center gap-2 px-4 py-2 rounded-lg btn-teal text-sm font-semibold"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Back to Resolve</span>
        </Link>
      </div>
    </div>
  );
}
