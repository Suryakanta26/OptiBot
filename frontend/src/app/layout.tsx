import type { Metadata } from "next";
import "./globals.css";
import Nav from "@/components/Nav";

export const metadata: Metadata = {
  title: "OptiBot — Optimized GenAI Order Tracking",
  description:
    "Before/after demonstration of an optimized GenAI workflow: routing, prompt compression, RAG, caching, guardrails, and monitoring.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body suppressHydrationWarning>
        <Nav />
        <main className="page">{children}</main>
      </body>
    </html>
  );
}
