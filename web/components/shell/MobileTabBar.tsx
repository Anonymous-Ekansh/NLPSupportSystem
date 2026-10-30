"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BarChart3,
  Database,
  MessageSquareText,
  Workflow,
} from "lucide-react";
import { cn } from "@/lib/utils";

const items = [
  { href: "/", label: "Live demo", icon: MessageSquareText },
  { href: "/architecture", label: "Architecture", icon: Workflow },
  { href: "/data", label: "Data", icon: Database },
  { href: "/results", label: "Results", icon: BarChart3 },
];

export function MobileTabBar() {
  const pathname = usePathname();

  return (
    <nav
      aria-label="Mobile navigation"
      className="fixed inset-x-0 bottom-0 z-40 border-t border-line bg-card pb-[env(safe-area-inset-bottom)] md:hidden"
    >
      <ul className="grid h-16 grid-cols-4">
        {items.map((item) => {
          const active =
            item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
          const Icon = item.icon;
          return (
            <li key={item.href}>
              <Link
                href={item.href}
                aria-current={active ? "page" : undefined}
                className={cn(
                  "flex h-full flex-col items-center justify-center gap-1 text-xs font-medium transition-colors",
                  active ? "text-brand-ink" : "text-ink-secondary"
                )}
              >
                <Icon
                  size={19}
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
  );
}
