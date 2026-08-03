"use client";

import { useCallback, useEffect, useState } from "react";
import { getAudit, getGovernance } from "@/lib/api";
import type { AuditRow, GovernanceSummary } from "@/lib/types";

function parseList(raw: string | null): string[] {
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.map(String) : [String(parsed)];
  } catch {
    return [raw];
  }
}

export default function GovernancePage() {
  const [summary, setSummary] = useState<GovernanceSummary | null>(null);
  const [rows, setRows] = useState<AuditRow[]>([]);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(() => {
    Promise.all([getGovernance(), getAudit(100)])
      .then(([s, a]) => {
        setSummary(s);
        setRows(a.audit);
        setErr(null);
      })
      .catch((e) => setErr(String(e)));
  }, []);

  useEffect(() => {
    load();
    const t = setInterval(load, 6000);
    return () => clearInterval(t);
  }, [load]);

  if (err) return <div className="banner warn">Could not load audit data: {err}</div>;
  if (!summary) return <div className="empty">Loading…</div>;

  return (
    <>
      <h1 className="page-title">Governance & Audit</h1>
      <p className="page-sub">
        Every interaction is written to an append-only audit log. Optimized-mode
        rows are PII-masked at write time; baseline rows deliberately are not,
        which is what the unmasked counter is measuring.
      </p>

      <div className="grid grid-4" style={{ marginBottom: 16 }}>
        <div className="card stat">
          <span className="stat-label">Audit coverage</span>
          <span className="stat-value" style={{ color: "#3fd18b" }}>
            {summary.audit_coverage_pct}%
          </span>
          <span className="stat-sub">
            {summary.audit_records} records / {summary.interactions} interactions
          </span>
        </div>
        <div className="card stat">
          <span className="stat-label">PII events masked</span>
          <span className="stat-value" style={{ color: "#3fd18b" }}>
            {summary.pii_masked_events}
          </span>
          <span className="stat-sub">of {summary.pii_events} containing PII</span>
        </div>
        <div className="card stat">
          <span className="stat-label">Unmasked PII stored</span>
          <span
            className="stat-value"
            style={{
              color: summary.pii_unmasked_events > 0 ? "#ef5f6b" : "#3fd18b",
            }}
          >
            {summary.pii_unmasked_events}
          </span>
          <span className="stat-sub">all from baseline mode</span>
        </div>
        <div className="card stat">
          <span className="stat-label">Requests blocked</span>
          <span className="stat-value" style={{ color: "#f5c451" }}>
            {summary.blocked_requests}
          </span>
          <span className="stat-sub">injection / abuse / rate limit</span>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 16 }}>
        <h2 className="card-title">Event breakdown</h2>
        {Object.keys(summary.event_counts).length === 0 ? (
          <div className="empty">No audit events yet.</div>
        ) : (
          Object.entries(summary.event_counts).map(([k, v]) => (
            <div className="kv" key={k}>
              <span className="kv-key">{k}</span>
              <span className="kv-val">{v}</span>
            </div>
          ))
        )}
      </div>

      <div className="card">
        <h2 className="card-title">Audit log (most recent 100)</h2>
        {rows.length === 0 ? (
          <div className="empty">No records yet — send some chat messages.</div>
        ) : (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Mode</th>
                  <th>Event</th>
                  <th>Model</th>
                  <th>Query (stored form)</th>
                  <th>PII</th>
                  <th>Guardrails</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => {
                  const pii = parseList(r.pii_detected);
                  const guards = parseList(r.guardrail_events);
                  return (
                    <tr key={r.id}>
                      <td className="mono">{r.timestamp.slice(11, 19)}</td>
                      <td>
                        <span
                          className={`pill ${r.mode === "optimized" ? "good" : "warn"}`}
                        >
                          {r.mode}
                        </span>
                      </td>
                      <td>{r.event_type}</td>
                      <td className="mono">{r.model ?? "—"}</td>
                      <td style={{ maxWidth: 340 }}>{r.query_masked ?? "—"}</td>
                      <td>
                        {pii.length === 0 ? (
                          <span className="muted">none</span>
                        ) : (
                          <span className={`pill ${r.pii_masked ? "good" : "bad"}`}>
                            {pii.join(", ")} {r.pii_masked ? "masked" : "UNMASKED"}
                          </span>
                        )}
                      </td>
                      <td>
                        {guards.length === 0 ? (
                          <span className="muted">—</span>
                        ) : (
                          guards.map((g) => (
                            <span key={g} className="pill warn" style={{ marginRight: 4 }}>
                              {g}
                            </span>
                          ))
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </>
  );
}
