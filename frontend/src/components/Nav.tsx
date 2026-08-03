"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { getHealth } from "@/lib/api";
import type { HealthResponse } from "@/lib/types";
import { GearIcon } from "./icons";
import SettingsPanel from "./SettingsPanel";

const LINKS = [
  { href: "/", label: "Chat" },
  { href: "/dashboard", label: "Dashboard" },
  { href: "/comparison", label: "Baseline / Optimized" },
  { href: "/governance", label: "Governance" },
];

export default function Nav() {
  const pathname = usePathname();
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [down, setDown] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const load = useCallback(() => {
    getHealth()
      .then((h) => {
        setHealth(h);
        setDown(false);
      })
      .catch(() => setDown(true));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <nav className="nav">
      <div className="brand">
        Opti<span>Bot</span>
      </div>
      {LINKS.map((l) => (
        <Link
          key={l.href}
          href={l.href}
          className={`nav-link ${pathname === l.href ? "active" : ""}`}
        >
          {l.label}
        </Link>
      ))}
      <div className="nav-right">
        {down && <span className="pill bad">API offline</span>}
        {health && (
          <>
            {!health.api_key_configured && (
              // Clickable: the shortest path from "something is wrong" to the fix.
              <button
                className="pill warn pill-btn"
                onClick={() => setSettingsOpen(true)}
                type="button"
                title="No LiteLLM gateway key configured — click to set one"
              >
                No LiteLLM key
              </button>
            )}
            <span
              className="pill"
              title={`Embedding backend: ${health.embedding_backend}`}
            >
              {health.embedding_backend.startsWith("sentence-transformers")
                ? "semantic embeddings"
                : "lexical fallback"}
            </span>
            <span className="pill info">{health.rag_index.chunks} policy chunks</span>
          </>
        )}
        <button
          className="icon-btn"
          onClick={() => setSettingsOpen(true)}
          type="button"
          aria-label="LiteLLM gateway settings"
          title="LiteLLM gateway settings"
        >
          <GearIcon size={17} />
        </button>
      </div>
      {settingsOpen && (
        <SettingsPanel onClose={() => setSettingsOpen(false)} onSaved={load} />
      )}
    </nav>
  );
}
