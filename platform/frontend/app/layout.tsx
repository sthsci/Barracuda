import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "BARRACUDA · Immune cell inference",
    template: "%s · BARRACUDA",
  },
  description: "Explore immune cell heterogeneity with transparent Bayesian event-count and contact-trajectory analyses.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
