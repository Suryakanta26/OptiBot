"use client";

import { useEffect, useState } from "react";
import { getSummary } from "@/lib/api";
import type { MetricsSummary } from "@/lib/types";

interface Row {
  metric: string;
  baseline: string;
  optimized: string;
  delta: string;
  better: boolean;
}

function pct(v: number) {
  return `${v > 0 ? "−" : v < 0 ? "+" : ""}${Math.abs(v).toFixed(1)}%`;
}

export default function ComparisonPage() {
  const [data, setData] = useState<MetricsSummary | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    const load = () =>
      getSummary()
        .then((d) => {
          setData(d);
          setErr(null);
        })
        .catch((e) => setErr(String(e)));
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, []);

  if (err) return <div className="banner warn">Could not load metrics: {err}</div>;
  if (!data) return <div className="empty">Loading…</div>;

  const { baseline: b, optimized: o, deltas: d } = data;

  const performance: Row[] = [
    {
      metric: "Avg tokens per request",
      baseline: String(b.avg_tokens),
      optimized: String(o.avg_tokens),
      delta: pct(d.tokens_reduction_pct),
      better: d.tokens_reduction_pct > 0,
    },
    {
      metric: "Avg latency",
      baseline: `${b.avg_latency_ms} ms`,
      optimized: `${o.avg_latency_ms} ms`,
      delta: pct(d.latency_reduction_pct),
      better: d.latency_reduction_pct > 0,
    },
    {
      metric: "Avg cost per interaction",
      baseline: `$${b.avg_cost_usd.toFixed(5)}`,
      optimized: `$${o.avg_cost_usd.toFixed(5)}`,
      delta: pct(d.cost_reduction_pct),
      better: d.cost_reduction_pct > 0,
    },
    {
      metric: "Cache hit rate",
      baseline: "0%",
      optimized: `${(o.cache_hit_rate * 100).toFixed(0)}%`,
      delta: `+${(o.cache_hit_rate * 100).toFixed(0)}pp`,
      better: o.cache_hit_rate > 0,
    },
  ];

  const trust: Row[] = [
    {
      metric: "Retrieval-grounded responses",
      baseline: "0%",
      optimized: `${(o.rag_rate * 100).toFixed(0)}%`,
      delta: `+${(o.rag_rate * 100).toFixed(0)}pp`,
      better: o.rag_rate > 0,
    },
    {
      metric: "Avg response confidence",
      baseline: b.avg_confidence.toFixed(2),
      optimized: o.avg_confidence.toFixed(2),
      delta: pct(-d.confidence_gain_pct),
      better: o.avg_confidence > b.avg_confidence,
    },
    {
      metric: "Output validated against order DB",
      baseline: "no",
      optimized: "yes",
      delta: "—",
      better: true,
    },
    {
      metric: "PII masked before storage",
      baseline: "no",
      optimized: "yes",
      delta: "—",
      better: true,
    },
    {
      metric: "Prompt injection blocked",
      baseline: "0",
      optimized: String(o.blocked),
      delta: `+${o.blocked}`,
      better: o.blocked > 0,
    },
  ];

  const value: Row[] = [
    {
      metric: "Projected cost / 10k queries",
      baseline: `$${d.monthly_cost_baseline_10k.toFixed(2)}`,
      optimized: `$${d.monthly_cost_optimized_10k.toFixed(2)}`,
      delta: pct(d.cost_reduction_pct),
      better: d.cost_reduction_pct > 0,
    },
    {
      metric: "Requests served without a model call",
      baseline: "0",
      optimized: String(o.cache_hits),
      delta: `+${o.cache_hits}`,
      better: o.cache_hits > 0,
    },
    {
      metric: "Distinct models in use",
      baseline: String(Object.keys(b.model_mix).length || 1),
      optimized: String(Object.keys(o.model_mix).length || 0),
      delta: "routed",
      better: Object.keys(o.model_mix).length > 1,
    },
  ];

  const sections: [string, string, Row[]][] = [
    ["Performance & Efficiency", "Prompt quality, token use, latency, retrieval effectiveness", performance],
    ["Trust & Governance", "Accuracy, hallucination controls, data protection, auditability", trust],
    ["Value & Operations", "Cost optimization, scalability, operational monitoring", value],
  ];

  const noData = b.requests === 0 || o.requests === 0;

  return (
    <>
      <h1 className="page-title">Baseline vs Optimized Comparison</h1>
      <p className="page-sub">
        Mapped to the evaluation lenses. Numbers are computed from{" "}
        {b.requests} baseline and {o.requests} optimized requests in this run.
      </p>

      {noData && (
        <div className="banner warn">
          Run the same set of questions through <strong>both</strong> modes on
          the Chat page (or run <code>python scripts/run_evaluation.py</code>)
          to populate this comparison.
        </div>
      )}

      {sections.map(([title, subtitle, rows]) => (
        <div className="card" style={{ marginBottom: 16 }} key={title}>
          <h2 className="card-title" style={{ marginBottom: 2 }}>
            {title}
          </h2>
          <p className="muted" style={{ fontSize: 12.5, margin: "0 0 10px" }}>
            {subtitle}
          </p>
          <div className="vs-row vs-head">
            <span>Metric</span>
            <span>Baseline</span>
            <span>Optimized</span>
            <span>Change</span>
          </div>
          {rows.map((r) => (
            <div className="vs-row" key={r.metric}>
              <span className="vs-metric">{r.metric}</span>
              <span className="vs-baseline">{r.baseline}</span>
              <span className="vs-optimized">{r.optimized}</span>
              <span
                className="vs-delta"
                style={{ color: r.better ? "#3fd18b" : "#94a1c0" }}
              >
                {r.delta}
              </span>
            </div>
          ))}
        </div>
      ))}

      <div className="card">
        <h2 className="card-title">User Experience</h2>
        <p className="muted" style={{ fontSize: 12.5, margin: "0 0 10px" }}>
          Adoption potential, workflow integration, fallback design, transparency
        </p>
        <div className="kv">
          <span className="kv-key">Source citations on policy answers</span>
          <span className="kv-val">baseline: none · optimized: filename cited</span>
        </div>
        <div className="kv">
          <span className="kv-key">Uncertainty signalling</span>
          <span className="kv-val">
            baseline: none · optimized: disclaimer below 0.7 confidence
          </span>
        </div>
        <div className="kv">
          <span className="kv-key">Escalation path</span>
          <span className="kv-val">
            baseline: none · optimized: routed to human with 4h window
          </span>
        </div>
        <div className="kv">
          <span className="kv-key">Pipeline transparency</span>
          <span className="kv-val">per-layer trace shown on every response</span>
        </div>
      </div>
    </>
  );
}
