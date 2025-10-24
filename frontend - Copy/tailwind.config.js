/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/app/**/*.{js,ts,jsx,tsx}",
    "./src/components/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      backgroundColor: {
        background: "var(--background)",
        primary: "var(--primary)",
        sidebar: "var(--sidebar)",
        "sidebar-primary": "var(--sidebar-primary)",
        "sidebar-accent": "var(--sidebar-accent)",
        card: "var(--card)",
        popover: "var(--popover)",
        input: "var(--input)",
      },
      textColor: {
        foreground: "var(--foreground)",
        primary: "var(--primary)",
        "sidebar-foreground": "var(--sidebar-foreground)",
        "sidebar-primary-foreground": "var(--sidebar-primary-foreground)",
        "sidebar-accent-foreground": "var(--sidebar-accent-foreground)",
      },
      borderColor: {
        border: "var(--border)",
        primary: "var(--primary)",
        "sidebar-border": "var(--sidebar-border)",
      },
      ringColor: {
        ring: "var(--ring)",
        primary: "var(--primary)",
        "sidebar-ring": "var(--sidebar-ring)",
      },
    },
  },
  plugins: [],
};