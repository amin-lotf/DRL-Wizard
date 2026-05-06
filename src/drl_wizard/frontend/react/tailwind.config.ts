import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        page: "#0c1110",
        panel: "#121a18",
        panelAlt: "#18211f",
        border: "#24312c",
        muted: "#8ea39a",
        text: "#edf5f1",
        accent: "#2dd4bf",
        accentSoft: "#0f766e",
        warning: "#fb923c",
        success: "#34d399",
        danger: "#f87171",
      },
      boxShadow: {
        panel: "0 18px 60px rgba(0, 0, 0, 0.28)",
      },
      borderRadius: {
        panel: "0.75rem",
      },
    },
  },
  plugins: [],
} satisfies Config;
