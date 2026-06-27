/**
 * Apex Front Office design system, ported verbatim from the Google Stitch
 * "NBA Draft AI War Room" project so the rendered app matches the design files.
 *
 * The color names are the Material-derived tokens the Stitch markup references
 * (bg-surface, text-primary, border-outline-variant, ...). The semantic intent
 * from the design's prose layer (basketball orange #FF6A2C, cool blue #4F8CFF,
 * tonal layers, hairlines) is preserved via the `tier` scale and the helper
 * classes in index.css.
 */
/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        "on-tertiary-fixed-variant": "#42474c",
        "primary-fixed-dim": "#ffb59b",
        "on-background": "#e1e2e9",
        "on-surface-variant": "#e2bfb4",
        "surface-container-lowest": "#0b0e13",
        "tertiary-fixed": "#dfe3e9",
        "surface-container": "#1d2025",
        "secondary-container": "#045ecf",
        "on-error": "#690005",
        "inverse-surface": "#e1e2e9",
        "on-secondary-fixed": "#001a43",
        "surface-container-low": "#191c21",
        secondary: "#afc6ff",
        "on-tertiary-fixed": "#171c20",
        "on-error-container": "#ffdad6",
        outline: "#a98a7f",
        "surface-tint": "#ffb59b",
        "tertiary-container": "#95999f",
        "on-secondary-container": "#d5e0ff",
        "on-secondary-fixed-variant": "#004398",
        "secondary-fixed": "#d9e2ff",
        "surface-container-high": "#272a30",
        "surface-variant": "#32353a",
        "outline-variant": "#5a4139",
        "on-primary-container": "#5c1b00",
        "surface-dim": "#111319",
        "on-tertiary": "#2c3136",
        "tertiary-fixed-dim": "#c3c7cd",
        "on-primary": "#5b1a00",
        primary: "#ffb59b",
        "error-container": "#93000a",
        "primary-container": "#ff6a2c",
        surface: "#111319",
        "surface-container-highest": "#32353a",
        "on-primary-fixed-variant": "#812900",
        tertiary: "#c3c7cd",
        background: "#111319",
        "on-surface": "#e1e2e9",
        "inverse-on-surface": "#2e3036",
        "on-tertiary-container": "#2d3136",
        error: "#ffb4ab",
        "primary-fixed": "#ffdbcf",
        "on-primary-fixed": "#380d00",
        "secondary-fixed-dim": "#afc6ff",
        "inverse-primary": "#a93800",
        "on-secondary": "#002d6c",
        "surface-bright": "#36393f",
        // Semantic accents from the design's prose layer (the "true" brand hues
        // the screenshots render with) + the 5-tier outcome scale.
        "brand-orange": "#ff6a2c",
        "brand-blue": "#4f8cff",
        "muted-pill": "#8a93a2",
        tier: {
          bust: "#ef4444", // red-500
          role: "#7d8694", // muted slate
          starter: "#ff6a2c", // basketball orange
          star: "#4f8cff", // cool blue
          super: "#ffb59b", // primary highlight
        },
      },
      borderRadius: {
        DEFAULT: "0.25rem",
        md: "0.5rem",
        lg: "0.75rem",
        xl: "1rem",
        full: "9999px",
      },
      spacing: {
        gutter: "16px",
        "card-gap": "20px",
        "container-padding": "24px",
        unit: "4px",
        "table-row-height": "40px",
      },
      fontFamily: {
        sans: ["Inter", "sans-serif"],
        "display-num": ["JetBrains Mono", "monospace"],
        "data-tabular": ["JetBrains Mono", "monospace"],
        "body-sm": ["Inter", "sans-serif"],
        "body-lg": ["Inter", "sans-serif"],
        "headline-md": ["Inter", "sans-serif"],
        "headline-lg": ["Inter", "sans-serif"],
        "label-caps": ["Inter", "sans-serif"],
      },
      fontSize: {
        "display-num": ["48px", { lineHeight: "1.1", letterSpacing: "-0.02em", fontWeight: "700" }],
        "body-sm": ["14px", { lineHeight: "1.5", fontWeight: "400" }],
        "body-lg": ["16px", { lineHeight: "1.6", fontWeight: "400" }],
        "data-tabular": ["14px", { lineHeight: "1", fontWeight: "500" }],
        "headline-sm": ["18px", { lineHeight: "1.3", fontWeight: "600" }],
        "headline-md": ["24px", { lineHeight: "1.3", fontWeight: "600" }],
        "headline-lg": ["32px", { lineHeight: "1.2", fontWeight: "600" }],
        "label-caps": ["11px", { lineHeight: "1", letterSpacing: "0.08em", fontWeight: "700" }],
      },
    },
  },
  plugins: [],
};
