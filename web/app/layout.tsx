import type { Metadata } from "next";
import "./globals.css";
import { AppProvider } from "@/lib/store";
import { VoiceProvider } from "@/lib/voice";
import { Navbar } from "@/components/Navbar";

export const metadata: Metadata = {
  title: "NLP Support System — Two-Agent Customer Support Orchestrator",
  description:
    "Autonomous Level-1 two-agent customer support architecture featuring Agent A1 Intent Analyst and Agent A2 Response Specialist with self-checking and pgvector retrieval.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark h-full antialiased" data-theme="dark">
      <body className="min-h-full flex flex-col bg-[var(--background)] text-[var(--foreground)]">
        <AppProvider>
          <VoiceProvider>
            <Navbar />
            <main className="flex-1">{children}</main>
          </VoiceProvider>
        </AppProvider>
      </body>
    </html>
  );
}
