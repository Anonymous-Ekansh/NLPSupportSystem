"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BarChart3,
  Database,
  MessageSquareText,
  ShieldAlert,
  Workflow,
} from "lucide-react";
import { Logo } from "@/components/ui/Logo";
import { HealthPill } from "@/components/shell/HealthPill";
import { cn } from "@/lib/utils";

const navItems = [
  { href: "/", label: "Live demo", icon: MessageSquareText },
  { href: "/architecture", label: "Architecture", icon: Workflow },
  { href: "/data", label: "Data explorer", icon: Database },
  { href: "/results", label: "Results", icon: BarChart3 },
  { href: "/robustness", label: "Robustness", icon: ShieldAlert },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed inset-y-0 left-0 z-30 hidden w-[200px] flex-col border-r border-line bg-canvas md:flex">
      <div className="px-5 py-5">
        <Link
          href="/"
          className="inline-flex rounded-[12px]"
          aria-label="ResolveIQ home"
        >
          <Logo />
        </Link>
      </div>

      <nav aria-label="Main navigation" className="flex-1 px-3 pt-2">
        <ul className="space-y-1">
          {navItems.map((item) => {
            const active =
              item.href === "/"
                ? pathname === "/"
                : pathname.startsWith(item.href);
            const Icon = item.icon;
            return (
              <li key={item.href}>
                <Link
                  href={item.href}
                  aria-current={active ? "page" : undefined}
                  className={cn(
                    "flex items-center gap-2.5 rounded-[10px] px-2.5 py-2 text-sm font-medium transition-colors",
                    active
                      ? "bg-brand/[0.08] text-brand-ink"
                      : "text-ink-secondary hover:bg-soft hover:text-ink"
                  )}
                >
                  <Icon
                    size={17}
                    strokeWidth={1.8}
                    aria-hidden="true"
                    className={active ? "text-brand" : "text-ink-faint"}
                  />
                  {item.label}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      <div className="mt-auto border-t border-line p-3">
        <p className="px-1 pb-2 text-xs text-ink-faint">
          Jaypee AI Summit 2026
        </p>
        <div className="rounded-card border border-line bg-card p-2.5">
          <p className="mb-1.5 text-xs uppercase tracking-[0.12em] text-ink-secondary">
            Assistant
          </p>
          <HealthPill />
          <p className="mt-2 text-xs leading-[18px] text-ink-faint">
            Free hosting can take ~30s to wake.
          </p>
        </div>
      </div>
    </aside>
  );
}
