import type { ResolveResult } from "@/lib/api";

export interface ChatMessageData {
  id: string;
  role: "user" | "assistant";
  text: string;
  createdAt: number;
  result?: ResolveResult;
  withheld?: boolean;
  unsure?: boolean;
  error?: boolean;
  retryQuery?: string;
}
