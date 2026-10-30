export function cn(...values: Array<string | false | null | undefined>) {
  return values.filter(Boolean).join(" ");
}

export function formatLatency(ms: number | null | undefined) {
  if (!ms || ms <= 0) return "—";
  return `${Math.round(ms)} ms`;
}

export function formatPercent(
  value: number | null | undefined,
  digits = 0
) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatScore(value: number | null | undefined, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

export function formatCount(value: number) {
  return new Intl.NumberFormat("en-US").format(value);
}

export function titleCaseWords(value: string) {
  return value
    .toLowerCase()
    .split("_")
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

export const FOLLOWUP_PHRASES: Record<string, string> = {
  get_refund: "Get a refund",
  track_refund: "Track my refund",
  check_refund_policy: "Check the refund policy",
  cancel_order: "Cancel my order",
  change_order: "Change my order",
  track_order: "Track my order",
  delivery_period: "Check delivery time",
  delivery_options: "See delivery options",
  change_shipping_address: "Change my shipping address",
  set_up_shipping_address: "Set up a shipping address",
  payment_issue: "Report a payment issue",
  check_payment_methods: "See payment methods",
  check_invoice: "Check my invoice",
  get_invoice: "Get my invoice",
  recover_password: "Recover my password",
  registration_problems: "Fix a sign-up problem",
  switch_account: "Switch account type",
  delete_account: "Delete my account",
  create_account: "Create an account",
  edit_account: "Edit my account",
  contact_customer_service: "Contact customer service",
  complaint: "File a complaint",
  review: "Leave a review",
  newsletter_subscription: "Manage my newsletter",
  check_cancellation_fee: "Check my cancellation fee",
  place_order: "Place an order",
};

/** Intents that should not be offered as a casual follow-up suggestion. */
const FOLLOWUP_BLOCKLIST = new Set(["contact_human_agent"]);

export function humaniseIntent(intent: string | null | undefined) {
  if (!intent) return "Unknown topic";
  return titleCaseWords(intent);
}

export function humaniseIntentPhrase(intent: string | null | undefined) {
  if (!intent) return "";
  return FOLLOWUP_PHRASES[intent] ?? titleCaseWords(intent);
}

export function humaniseCategory(category: string | null | undefined) {
  if (!category) return "Other";
  return titleCaseWords(category);
}

export function topicLabel(
  category: string | null | undefined,
  intent: string | null | undefined
) {
  return `${humaniseCategory(category)} - ${intent ?? "unknown"}`;
}

export function isFollowupEligible(intent: string) {
  return !FOLLOWUP_BLOCKLIST.has(intent);
}

export function clamp(value: number, min = 0, max = 1) {
  return Math.min(max, Math.max(min, value));
}
