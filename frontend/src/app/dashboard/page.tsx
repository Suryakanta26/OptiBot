"use client";

import { useCallback, useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getSummary, resetMetrics } from "@/lib/api";
import type { MetricsSummary } from "@/lib/types";

const BASELINE = "#f2795b";
const OPTIMIZED = "#3fd18b";
const TIER_COLORS = ["#5b8cff", "#3fd18b", "#f5c451", "#ef5f6b"];

const axis = { stroke: "#94a1c0", fontSize: 12 };
const tooltipStyle = {
  background: "#131a2e",
  border: "1px solid #26314f",
  borderRadius: 8,
  fontSize: 13,
};

export default function DashboardPage() {
  const [data, setData] = useState<MetricsSummary | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(() => {
    getSummary()
      .then((d) => {
        setData(d);
        setErr(null);
      })
      .catch((e) => setErr(String(e)));
  }, []);

  useEffect(() => {
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, [load]);

  if (err) return <div className="banner warn">Could not load metrics: {err}</div>;
  if (!data) return <div className="empty">Loading metrics…</div>;

  const { baseline, optimized, deltas, cache } = data;
  const total = baseline.requests + optimized.requests;

  const tokenChart = [
    { name: "Avg tokens", Baseline: baseline.avg_tokens, Optimized: optimized.avg_tokens },
  ];
  const latencyChart = [
    { name: "Avg latency (ms)", Baseline: baseline.avg_latency_ms, Optimized: optimized.avg_latency_ms },
  ];
  const costChart = [
    {
      name: "Cost / 10k queries ($)",
      Baseline: deltas.monthly_cost_baseline_10k,
      Optimized: deltas.monthly_cost_optimized_10k,
    },
  ];

  const modelMix = Object.entries(optimized.model_mix).map(([name, value]) => ({
    name,
    value,
  }));

  return (
    <>
      <div style={{ display: "flex", alignItems: "flex-start" }}>
        <div>
          <h1 className="page-title">Monitoring Dashboard</h1>
          <p className="page-sub">
            Live metrics from every request. Refreshes every 5 seconds.
          </p>
        </div>
        <button
          className="btn btn-ghost"
          style={{ marginLeft: "auto" }}
          onClick={() => resetMetrics().then(load)}
        >
          Reset run
        </button>
      </div>

      {total === 0 && (
        <div className="banner info">
          No requests recorded yet. Send a few messages on the Chat page in both
          modes, then come back — the comparison needs data from both pipelines.
        </div>
      )}

      <div className="grid grid-4" style={{ marginBottom: 16 }}>
        <div className="card stat">
          <span className="stat-label">Token reduction</span>
          <span className="stat-value" style={{ color: OPTIMIZED }}>
            {deltas.tokens_reduction_pct}%
          </span>
          <span className="stat-sub">
            {baseline.avg_tokens} → {optimized.avg_tokens} avg
          </span>
        </div>
        <div className="card stat">
          <span className="stat-label">Cost reduction</span>
          <span className="stat-value" style={{ color: OPTIMIZED }}>
            {deltas.cost_reduction_pct}%
          </span>
          <span className="stat-sub">
            ${baseline.avg_cost_usd.toFixed(5)} → ${optimized.avg_cost_usd.toFixed(5)} per query
          </span>
        </div>
        <div className="card stat">
          <span className="stat-label">Latency reduction</span>
          <span className="stat-value" style={{ color: OPTIMIZED }}>
            {deltas.latency_reduction_pct}%
          </span>
          <span className="stat-sub">
            {baseline.avg_latency_ms} → {optimized.avg_latency_ms} ms
          </span>
        </div>
        <div className="card stat">
          <span className="stat-label">Cache hit rate</span>
          <span className="stat-value" style={{ color: OPTIMIZED }}>
            {(optimized.cache_hit_rate * 100).toFixed(0)}%
          </span>
          <span className="stat-sub">
            {cache.hits}/{cache.lookups} lookups · {cache.entries} entries
          </span>
        </div>
      </div>

      <div className="grid grid-3" style={{ marginBottom: 16 }}>
        {[
          { title: "Tokens per request", data: tokenChart },
          { title: "Latency per request", data: latencyChart },
          { title: "Projected cost / 10k queries", data: costChart },
        ].map((chart) => (
          <div className="card" key={chart.title}>
            <h2 className="card-title">{chart.title}</h2>
            <ResponsiveContainer width="100%" height={190}>
              <BarChart data={chart.data}>
                <CartesianGrid strokeDasharray="3 3" stroke="#26314f" />
                <XAxis dataKey="name" {...axis} />
                <YAxis {...axis} />
                <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "#1a2340" }} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="Baseline" fill={BASELINE} radius={[4, 4, 0, 0]} />
                <Bar dataKey="Optimized" fill={OPTIMIZED} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        ))}
      </div>

      <div className="grid grid-2">
        <div className="card">
          <h2 className="card-title">Model routing mix (optimized)</h2>
          {modelMix.length === 0 ? (
            <div className="empty">No optimized requests yet.</div>
          ) : (
            <ResponsiveContainer width="100%" height={230}>
              <PieChart>
                <Pie
                  data={modelMix}
                  dataKey="value"
                  nameKey="name"
                  outerRadius={80}
                  label={(e: { name?: string; value?: number }) =>
                    `${e.name} (${e.value})`
                  }
                >
                  {modelMix.map((_, i) => (
                    <Cell key={i} fill={TIER_COLORS[i % TIER_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip contentStyle={tooltipStyle} />
              </PieChart>
            </ResponsiveContainer>
          )}
          <p className="muted" style={{ fontSize: 12.5, marginTop: 8 }}>
            Baseline sends every query to a single expensive model. The optimized
            pipeline routes by classified complexity.
          </p>
        </div>

        <div className="card">
          <h2 className="card-title">Run totals</h2>
          <div className="kv">
            <span className="kv-key">Baseline requests</span>
            <span className="kv-val">{baseline.requests}</span>
          </div>
          <div className="kv">
            <span className="kv-key">Optimized requests</span>
            <span className="kv-val">{optimized.requests}</span>
          </div>
          <div className="kv">
            <span className="kv-key">Total tokens (baseline)</span>
            <span className="kv-val">{baseline.total_tokens.toLocaleString()}</span>
          </div>
          <div className="kv">
            <span className="kv-key">Total tokens (optimized)</span>
            <span className="kv-val">{optimized.total_tokens.toLocaleString()}</span>
          </div>
          <div className="kv">
            <span className="kv-key">Spend this run</span>
            <span className="kv-val">
              ${baseline.total_cost_usd.toFixed(4)} vs ${optimized.total_cost_usd.toFixed(4)}
            </span>
          </div>
          <div className="kv">
            <span className="kv-key">RAG-grounded responses</span>
            <span className="kv-val">
              {optimized.rag_used}/{optimized.requests} ({(optimized.rag_rate * 100).toFixed(0)}%)
            </span>
          </div>
          <div className="kv">
            <span className="kv-key">Blocked by guardrails</span>
            <span className="kv-val">{optimized.blocked}</span>
          </div>
          <div className="kv">
            <span className="kv-key">Tier mix</span>
            <span className="kv-val">
              {Object.entries(optimized.tier_mix)
                .map(([k, v]) => `${k}:${v}`)
                .join("  ") || "—"}
            </span>
          </div>
        </div>
      </div>
    </>
  );
}
