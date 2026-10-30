import type { Metadata, Viewport } from "next";
import "@fontsource/inter/400.css";
import "@fontsource/inter/500.css";
import "@fontsource/inter/600.css";
import "@fontsource/inter/700.css";
import "@fontsource/jetbrains-mono/400.css";
import "@fontsource/jetbrains-mono/500.css";
import "./globals.css";
import { Sidebar } from "@/components/shell/Sidebar";
import { MobileTabBar } from "@/components/shell/MobileTabBar";

export const metadata: Metadata = {
  title: {
    default: "ResolveIQ — Two-agent customer support",
    template: "%s · ResolveIQ",
  },
  description:
    "ResolveIQ understands a customer question, finds the right help answer, and checks it before replying. A1 Intent Analyst → A2 Response Specialist.",
};

export const viewport: Viewport = {
  themeColor: "#FBFBFA",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-dvh bg-canvas font-sans text-sm text-ink antialiased">
        <a className="skip-link" href="#main">
          Skip to content
        </a>
        <Sidebar />
        <main id="main" className="min-h-dvh md:pl-[200px]">
          <div className="mx-auto w-full max-w-content px-4 pb-28 pt-6 sm:px-6 md:py-8 md:pb-10 lg:px-8">
            {children}
          </div>
        </main>
        <MobileTabBar />
      </body>
    </html>
  );
}
