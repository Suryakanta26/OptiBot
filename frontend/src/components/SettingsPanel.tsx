"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  getLlmConfig,
  getLlmModels,
  saveLlmConfig,
  testLlmConfig,
} from "@/lib/api";
import type {
  LlmConfigResponse,
  LlmTestResponse,
  ModelSlot,
} from "@/lib/types";
import {
  AlertIcon,
  CheckIcon,
  CloseIcon,
  GearIcon,
  KeyIcon,
  RefreshIcon,
  ShieldIcon,
  SpinnerIcon,
} from "./icons";

const SLOTS: ModelSlot[] = ["baseline", "simple", "complex"];

type PerSlot<T> = Record<ModelSlot, T>;

const emptyPerSlot = <T,>(value: T): PerSlot<T> => ({
  baseline: value,
  simple: value,
  complex: value,
});

interface Props {
  onClose: () => void;
  /** Called after a successful save so the nav can refresh its status pills. */
  onSaved?: () => void;
}

export default function SettingsPanel({ onClose, onSaved }: Props) {
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [data, setData] = useState<LlmConfigResponse | null>(null);

  const [baseUrl, setBaseUrl] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [picked, setPicked] = useState<PerSlot<string>>(emptyPerSlot(""));
  const [custom, setCustom] = useState<PerSlot<string>>(emptyPerSlot(""));
  const [testing, setTesting] = useState<PerSlot<boolean>>(emptyPerSlot(false));
  const [results, setResults] = useState<PerSlot<LlmTestResponse | null>>(
    emptyPerSlot<LlmTestResponse | null>(null),
  );

  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const closeRef = useRef<HTMLButtonElement>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setMessage("");
    setError("");
    try {
      const config = await getLlmConfig();
      setData(config);
      setBaseUrl(config.config.base_url);
      setApiKey("");
      setCustom(emptyPerSlot(""));
      setResults(emptyPerSlot<LlmTestResponse | null>(null));
      const next = emptyPerSlot("");
      for (const slot of config.config.slots) {
        // Keep the active alias selectable even when the gateway does not list
        // it — otherwise the <select> silently snaps to a different model.
        next[slot.slot] = slot.model;
      }
      setPicked(next);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not load the LiteLLM configuration from the backend.",
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const effective = (slot: ModelSlot) => custom[slot].trim() || picked[slot];

  const slotConfig = (slot: ModelSlot) =>
    data?.config.slots.find((s) => s.slot === slot);

  const hasChanges = Boolean(
    data &&
      (apiKey.trim().length > 0 ||
        baseUrl.trim() !== data.config.base_url ||
        SLOTS.some((slot) => {
          const next = effective(slot);
          return next && next !== slotConfig(slot)?.model;
        })),
  );

  async function refreshModels() {
    setRefreshing(true);
    setError("");
    try {
      const fresh = await getLlmModels();
      setData((prev) => (prev ? { ...prev, ...fresh } : prev));
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Could not refresh the model list.",
      );
    } finally {
      setRefreshing(false);
    }
  }

  async function runTest(slot: ModelSlot) {
    setTesting((prev) => ({ ...prev, [slot]: true }));
    setResults((prev) => ({ ...prev, [slot]: null }));
    try {
      // Send the pending key and URL so a value can be validated before saving.
      const result = await testLlmConfig({
        slot,
        model_id: effective(slot),
        api_key: apiKey.trim() || undefined,
        base_url: baseUrl.trim() || undefined,
      });
      setResults((prev) => ({ ...prev, [slot]: result }));
    } catch (err) {
      setResults((prev) => ({
        ...prev,
        [slot]: {
          status: "error",
          message:
            err instanceof Error
              ? err.message
              : "The test request did not reach the backend.",
          slot,
          model: effective(slot),
          latency_ms: null,
          response: null,
          usage: null,
          config_used: {},
        },
      }));
    } finally {
      setTesting((prev) => ({ ...prev, [slot]: false }));
    }
  }

  async function save() {
    setSaving(true);
    setMessage("");
    setError("");
    try {
      // Send only what actually changed. Submitting all three slots every time
      // would pin slots the user never touched, so they would stop tracking
      // LITELLM_MODEL_* in backend/.env.
      const models: Partial<Record<ModelSlot, string>> = {};
      for (const slot of SLOTS) {
        const next = effective(slot);
        if (next && next !== slotConfig(slot)?.model) models[slot] = next;
      }
      const trimmedBase = baseUrl.trim();
      const result = await saveLlmConfig({
        models: Object.keys(models).length > 0 ? models : undefined,
        base_url:
          trimmedBase && trimmedBase !== data?.config.base_url
            ? trimmedBase
            : undefined,
        api_key: apiKey.trim() || undefined,
      });
      // 200 with status "error" is the backend's convention for a rejected
      // config, so checking res.ok alone would report a phantom success.
      if (result.status === "success") {
        // Reload first: load() clears the banners, so setting the message before
        // it would wipe the confirmation the user needs to read.
        await load();
        setMessage(result.message);
        onSaved?.();
      } else {
        setError(result.message);
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not save the LiteLLM configuration.",
      );
    } finally {
      setSaving(false);
    }
  }

  const models = data?.models ?? [];
  const live = data?.models_source === "litellm";
  const keyStatus = data?.config.api_key_status;
  const tls = data?.config.tls;

  return (
    <div
      className="modal-layer"
      role="dialog"
      aria-modal="true"
      aria-labelledby="llm-settings-title"
    >
      <button
        className="modal-backdrop"
        onClick={onClose}
        aria-label="Close settings"
        type="button"
      />
      <section className="modal">
        <header className="modal-head">
          <div className="modal-title">
            <h2 id="llm-settings-title">
              <GearIcon size={17} /> LiteLLM Gateway
            </h2>
            <p>
              Models, providers and routing live on the gateway. OptiBot only
              chooses aliases and holds the bearer token.
            </p>
          </div>
          <button
            ref={closeRef}
            className="icon-btn"
            onClick={onClose}
            aria-label="Close"
            type="button"
          >
            <CloseIcon size={17} />
          </button>
        </header>

        <div className="modal-body">
          {loading ? (
            <div className="modal-loading">
              <SpinnerIcon size={18} /> Loading gateway configuration…
            </div>
          ) : (
            <>
              {!live && (
                <div className="result-box bad">
                  <AlertIcon size={17} />
                  <div>
                    <strong>Could not reach the LiteLLM gateway</strong>
                    <span>
                      {data?.warning ??
                        "Showing built-in aliases instead of the gateway's live list."}
                    </span>
                    <span className="muted">
                      Saving still works, but chat will fail until the gateway is
                      reachable.
                    </span>
                  </div>
                </div>
              )}

              <div className="field">
                <label htmlFor="llm-base-url">Gateway base URL</label>
                <input
                  id="llm-base-url"
                  value={baseUrl}
                  onChange={(e) => setBaseUrl(e.target.value)}
                  placeholder="http://localhost:4000"
                  spellCheck={false}
                  // Browser autofill will happily rewrite a URL-shaped field,
                  // silently repointing the gateway.
                  autoComplete="off"
                />
                <small className="form-help">
                  Include any mount prefix (e.g.{" "}
                  <code className="mono">https://host/litellm</code>).{" "}
                  <code className="mono">/v1</code> is appended automatically —
                  you can paste it either way. Chat posts to{" "}
                  <code className="mono">{data?.config.chat_url}</code>.
                </small>
              </div>

              <div className="field">
                <label htmlFor="llm-api-key">LiteLLM API key</label>
                <div className="key-status">
                  <KeyIcon size={16} />
                  {keyStatus?.is_set ? (
                    <span>
                      {keyStatus.env_var} is set ({keyStatus.masked_value}),
                      from {keyStatus.source === "runtime" ? "this panel" : "backend/.env"}.
                    </span>
                  ) : (
                    <span>{keyStatus?.env_var} is not set.</span>
                  )}
                </div>
                <input
                  id="llm-api-key"
                  type="password"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="Paste a key to use for new requests"
                  autoComplete="off"
                />
                <small className="form-help">
                  Applied in-process only — never written to disk. Add{" "}
                  <code className="mono">{keyStatus?.env_var}=…</code> to{" "}
                  <code className="mono">backend/.env</code> to survive a
                  restart. Provider keys (Gemini, Anthropic, OpenAI…) live on the
                  gateway itself, not here.
                </small>
              </div>

              {tls && (
                <div className="field">
                  <label>TLS verification</label>
                  <div className="key-status">
                    <ShieldIcon size={16} />
                    <span>
                      {tls.verify ? "On" : "Off"}
                      {tls.ca_bundle
                        ? ` — CA bundle: ${tls.ca_bundle}`
                        : ` (${tls.source})`}
                    </span>
                  </div>
                  {tls.warning && (
                    <div className="result-box bad">
                      <AlertIcon size={17} />
                      <div>
                        <strong>{tls.warning}</strong>
                      </div>
                    </div>
                  )}
                  <small className="form-help">{tls.note}</small>
                </div>
              )}

              <div className="slot-list">
                {SLOTS.map((slot) => {
                  const config = slotConfig(slot);
                  const result = results[slot];
                  const options = models.includes(picked[slot])
                    ? models
                    : [picked[slot], ...models].filter(Boolean);
                  return (
                    <div className="slot-row" key={slot}>
                      <div className="slot-row-head">
                        <div>
                          <strong>{config?.label ?? slot}</strong>
                          <div className="muted slot-desc">
                            {config?.description}
                          </div>
                        </div>
                        {config?.price_source === "default" && (
                          <span
                            className="pill warn"
                            title="No local price for this alias — cost figures fall back to $3.00/$15.00 per Mtok and the baseline-vs-optimized comparison will be wrong."
                          >
                            price source: default
                          </span>
                        )}
                      </div>

                      <div className="form-grid-2">
                        <div className="field">
                          <label htmlFor={`model-${slot}`}>Model</label>
                          <select
                            id={`model-${slot}`}
                            value={picked[slot]}
                            onChange={(e) =>
                              setPicked((prev) => ({
                                ...prev,
                                [slot]: e.target.value,
                              }))
                            }
                          >
                            {options.length === 0 && (
                              <option value="">(no models)</option>
                            )}
                            {options.map((model) => (
                              <option key={model} value={model}>
                                {model}
                              </option>
                            ))}
                          </select>
                        </div>
                        <div className="field">
                          <label htmlFor={`custom-${slot}`}>Custom alias</label>
                          <input
                            id={`custom-${slot}`}
                            value={custom[slot]}
                            onChange={(e) =>
                              setCustom((prev) => ({
                                ...prev,
                                [slot]: e.target.value,
                              }))
                            }
                            placeholder="Override with any alias the gateway accepts"
                            spellCheck={false}
                          />
                        </div>
                      </div>

                      <div className="slot-actions">
                        <span className="muted mono">
                          {effective(slot) || "—"}
                          {config &&
                            ` · $${config.price_in_per_mtok.toFixed(2)}/$${config.price_out_per_mtok.toFixed(2)} per Mtok`}
                        </span>
                        <button
                          className="btn btn-ghost"
                          type="button"
                          onClick={() => runTest(slot)}
                          disabled={testing[slot] || !effective(slot)}
                        >
                          {testing[slot] ? (
                            <SpinnerIcon size={15} />
                          ) : (
                            <CheckIcon size={15} />
                          )}{" "}
                          Test
                        </button>
                      </div>

                      {result && (
                        <div
                          className={`result-box ${result.status === "success" ? "ok" : "bad"}`}
                        >
                          {result.status === "success" ? (
                            <CheckIcon size={17} />
                          ) : (
                            <AlertIcon size={17} />
                          )}
                          <div>
                            <strong>{result.message}</strong>
                            {result.response && (
                              <span>Replied: “{result.response}”</span>
                            )}
                            {result.usage && (
                              <span className="mono">
                                {result.latency_ms} ms ·{" "}
                                {result.usage.total_tokens} tokens · $
                                {result.usage.cost_usd.toFixed(6)} (
                                {result.usage.cost_basis})
                              </span>
                            )}
                            {Object.keys(result.config_used).length > 0 && (
                              <code>
                                {Object.entries(result.config_used)
                                  .map(([k, v]) => `${k}=${v}`)
                                  .join(" | ")}
                              </code>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              <small className="form-help">
                Typical lab setup uses two models: the fast one for{" "}
                <strong>simple</strong>, the strong one for{" "}
                <strong>baseline</strong> and <strong>complex</strong>. Baseline
                spends the expensive model on every query on purpose — that is
                the behaviour the dashboard measures against.
                {live
                  ? ` ${models.length} model${models.length === 1 ? "" : "s"} reported by the gateway at ${data?.config.base_url}.`
                  : ""}
              </small>

              {message && (
                <div className="result-box ok">
                  <CheckIcon size={17} />
                  <div>
                    <strong>{message}</strong>
                  </div>
                </div>
              )}
              {error && (
                <div className="result-box bad">
                  <AlertIcon size={17} />
                  <div>
                    <strong>{error}</strong>
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        <footer className="modal-foot">
          <span className="muted mono modal-foot-note">
            {data?.config.persisted
              ? `Saved to ${data.config.runtime_config_path}`
              : "No saved overrides yet"}
          </span>
          <button
            className="btn btn-ghost"
            type="button"
            onClick={refreshModels}
            disabled={loading || refreshing}
            title="Re-fetch the model list from the gateway"
          >
            {refreshing ? <SpinnerIcon size={15} /> : <RefreshIcon size={15} />}{" "}
            Refresh models
          </button>
          <button
            className="btn"
            type="button"
            onClick={save}
            disabled={loading || saving || !hasChanges}
          >
            {saving ? <SpinnerIcon size={15} /> : null} Save &amp; apply
          </button>
        </footer>
      </section>
    </div>
  );
}
