"""Request/response contracts for the API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Mode = Literal["baseline", "optimized"]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    mode: Mode = "optimized"
    session_id: str = Field(default="demo-session", max_length=64)


class TraceStep(BaseModel):
    layer: str
    detail: str
    duration_ms: int = 0


class ChatMetrics(BaseModel):
    mode: Mode
    model: str | None = None
    tier: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0
    cache_hit: bool = False
    cache_level: Literal["none", "exact", "semantic"] = "none"
    cache_similarity: float = 0.0
    cache_entry_age_ms: int = 0
    rag_used: bool = False
    rag_sources: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    guardrail_events: list[str] = Field(default_factory=list)
    pii_detected: list[str] = Field(default_factory=list)
    pii_masked: bool = False
    blocked: bool = False


class ChatResponse(BaseModel):
    response: str
    metrics: ChatMetrics
    trace: list[TraceStep] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    provider: Literal["litellm"] = "litellm"
    # Refers to the LiteLLM gateway key, which can be set at runtime from the
    # settings panel — so this is evaluated per request, not frozen at import.
    api_key_configured: bool
    api_key_source: Literal["runtime", "env", "none"] = "none"
    litellm_base_url: str = ""
    # None means "not probed on this request". Probing costs a round trip and
    # /api/health is called on every page mount, so it is opt-in via ?probe=true.
    litellm_reachable: bool | None = None
    embedding_backend: str
    rag_index: dict[str, Any]
    models: dict[str, str]


# --------------------------------------------------------------------------- #
# LiteLLM gateway configuration (the settings panel behind the gear icon)
# --------------------------------------------------------------------------- #

ModelSlot = Literal["baseline", "simple", "complex"]


class ApiKeyStatus(BaseModel):
    env_var: str = "LITELLM_API_KEY"
    is_set: bool
    source: Literal["runtime", "env", "none"]
    masked_value: str = ""


class TlsStatus(BaseModel):
    verify: bool
    ca_bundle: str | None = None
    source: Literal["default", "LITELLM_CA_BUNDLE", "LITELLM_SSL_VERIFY"]
    warning: str | None = None
    note: str


class SlotConfig(BaseModel):
    slot: ModelSlot
    label: str
    description: str
    model: str
    source: Literal["runtime", "env", "fallback"]
    price_in_per_mtok: float
    price_out_per_mtok: float
    price_source: Literal["exact", "prefix", "default"]


class LlmConfig(BaseModel):
    provider: Literal["litellm"] = "litellm"
    base_url: str
    chat_url: str
    models_url: str
    slots: list[SlotConfig]
    api_key_status: ApiKeyStatus
    tls: TlsStatus
    request_timeout_s: float
    max_attempts: int
    runtime_config_path: str
    persisted: bool


class LlmModelsResponse(BaseModel):
    status: Literal["success", "error"] = "success"
    models: list[str] = Field(default_factory=list)
    models_source: Literal["litellm", "fallback"] = "fallback"
    warning: str | None = None


class LlmConfigResponse(LlmModelsResponse):
    config: LlmConfig


class LlmConfigUpdate(BaseModel):
    """A partial update — the panel edits each slot independently."""

    models: dict[ModelSlot, str] | None = None
    base_url: str | None = None
    # repr=False so a pydantic repr in a traceback cannot leak the secret.
    api_key: str | None = Field(default=None, repr=False)


class LlmConfigSaveResponse(BaseModel):
    status: Literal["success", "error"]
    message: str
    config: LlmConfig | None = None


class LlmTestRequest(BaseModel):
    slot: ModelSlot | None = None
    model_id: str | None = None
    # Unsaved overrides, so the panel can test a pending key or URL *before*
    # applying it. Neither is persisted or applied by the test call.
    api_key: str | None = Field(default=None, repr=False)
    base_url: str | None = None


class LlmUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    cost_basis: str | None = None


class LlmTestResponse(BaseModel):
    status: Literal["success", "error"]
    message: str
    slot: ModelSlot | None = None
    model: str | None = None
    latency_ms: int | None = None
    response: str | None = None
    usage: LlmUsage | None = None
    config_used: dict[str, str] = Field(default_factory=dict)
