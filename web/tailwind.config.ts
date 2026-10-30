import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        canvas: "#FBFBFA",
        card: "#FFFFFF",
        line: "#ECEBE8",
        ink: {
          DEFAULT: "#18181B",
          secondary: "#71717A",
          // Kept at the same contrast as secondary so 12px captions meet AA.
          faint: "#71717A",
        },
        brand: {
          DEFAULT: "#4F46E5",
          ink: "#4338CA",
          soft: "#EEF2FF",
        },
        a1: {
          DEFAULT: "#4F46E5",
          ink: "#4338CA",
          soft: "#EEF2FF",
        },
        a2: {
          DEFAULT: "#0D9488",
          ink: "#0F766E",
          soft: "#F0FDFA",
        },
        success: {
          DEFAULT: "#16A34A",
          ink: "#15803D",
          soft: "#F0FDF4",
        },
        warning: {
          DEFAULT: "#D97706",
          ink: "#B45309",
          soft: "#FFFBEB",
        },
        error: {
          DEFAULT: "#DC2626",
          ink: "#B91C1C",
          soft: "#FEF2F2",
        },
        soft: "#F6F6F4",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: [
          "JetBrains Mono",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "monospace",
        ],
      },
      fontSize: {
        xs: ["12px", { lineHeight: "18px" }],
        sm: ["14px", { lineHeight: "21px" }],
        base: ["16px", { lineHeight: "25px" }],
        xl: ["28px", { lineHeight: "36px" }],
        display: ["40px", { lineHeight: "48px" }],
      },
      borderRadius: {
        card: "16px",
        pill: "999px",
      },
      boxShadow: {
        float:
          "0 10px 30px rgba(24, 24, 27, 0.07), 0 1px 2px rgba(24, 24, 27, 0.04)",
        sheet: "0 -12px 40px rgba(24, 24, 27, 0.12)",
      },
      maxWidth: {
        content: "1200px",
      },
      keyframes: {
        fadeUp: {
          "0%": { opacity: "0", transform: "translateY(6px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        sheetUp: {
          "0%": { transform: "translateY(100%)" },
          "100%": { transform: "translateY(0)" },
        },
        softPulse: {
          "0%, 100%": { opacity: "0.45" },
          "50%": { opacity: "1" },
        },
        pingSoft: {
          "0%": { transform: "scale(0.85)", opacity: "0.45" },
          "70%, 100%": { transform: "scale(1.9)", opacity: "0" },
        },
      },
      animation: {
        "fade-up": "fadeUp 220ms ease-out both",
        "fade-in": "fadeIn 180ms ease-out both",
        "sheet-up": "sheetUp 220ms ease-out both",
        "soft-pulse": "softPulse 1.4s ease-in-out infinite",
        "ping-soft": "pingSoft 1.8s cubic-bezier(0, 0, 0.2, 1) infinite",
      },
    },
  },
  plugins: [],
};

export default config;
