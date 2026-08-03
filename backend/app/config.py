"""Central configuration for OptiBot.

Everything that differs between the *baseline* and *optimized* pipelines is
declared here so the two paths stay honestly comparable: same data, same
question, different machinery.

This module is the *environment* layer only — every value is read once at
import. Anything the operator can change while the app is running (the LiteLLM
key, the base URL, the per-slot model aliases) lives in ``app.llm_settings``,
which layers a mutable runtime store on top of these defaults.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
DATA_DIR = BACKEND_DIR / "app" / "data"
POLICY_DIR = DATA_DIR / "policies"
DB_PATH = BACKEND_DIR / "optibot.db"

# Also loaded in app/__init__.py, which has to read it before litellm is imported.
load_dotenv(BACKEND_DIR / ".env")


class Settings:
    """Runtime settings, read once at import."""

    # --- LiteLLM gateway -------------------------------------------------
    # OptiBot talks to exactly one thing: a LiteLLM gateway. Provider keys
    # (Gemini, Anthropic, OpenAI, Vertex...) live on that gateway, never here.
    litellm_base_url_env: str = os.getenv("LITELLM_BASE_URL", "http://localhost:4000")
    litellm_api_key_env: str | None = os.getenv("LITELLM_API_KEY") or None

    # --- Model slots (aliases exactly as the gateway exposes them) --------
    # Startup defaults only. Whatever the operator picks in the settings panel
    # is persisted to llm_runtime_path and takes precedence — see
    # app.llm_settings.model_for(). Defaulting to "" rather than a hardcoded
    # alias keeps the per-slot fallback in exactly one place.
    #
    # Baseline deliberately sends *every* query to the expensive model.
    # Optimized routes by classified complexity.
    # OPTIBOT_MODEL_* is accepted as a legacy alias so an existing
    # backend/.env keeps working after the move off the direct Anthropic path.
    model_baseline: str = (
        os.getenv("LITELLM_MODEL_BASELINE") or os.getenv("OPTIBOT_MODEL_BASELINE") or ""
    )
    model_simple: str = (
        os.getenv("LITELLM_MODEL_SIMPLE") or os.getenv("OPTIBOT_MODEL_SIMPLE") or ""
    )
    model_complex: str = (
        os.getenv("LITELLM_MODEL_COMPLEX") or os.getenv("OPTIBOT_MODEL_COMPLEX") or ""
    )

    # --- TLS to the gateway ----------------------------------------------
    # Deliberately env-only and read once: litellm caches its HTTP client per
    # (api key, base url) and the TLS setting is *not* part of that cache key,
    # so a runtime toggle would silently no-op. See services/gateway.py.
    litellm_ssl_verify_raw: str = os.getenv("LITELLM_SSL_VERIFY", "")
    litellm_ca_bundle: str | None = os.getenv("LITELLM_CA_BUNDLE") or None

    # --- Gateway timeouts / retries --------------------------------------
    litellm_timeout_s: float = float(os.getenv("LITELLM_TIMEOUT_S", "120"))
    litellm_connect_timeout_s: float = float(os.getenv("LITELLM_CONNECT_TIMEOUT_S", "10"))
    litellm_models_timeout_s: float = float(os.getenv("LITELLM_MODELS_TIMEOUT_S", "10"))
    litellm_test_timeout_s: float = float(os.getenv("LITELLM_TEST_TIMEOUT_S", "15"))
    litellm_probe_timeout_s: float = float(os.getenv("LITELLM_PROBE_TIMEOUT_S", "3"))
    litellm_max_attempts: int = int(os.getenv("LITELLM_MAX_ATTEMPTS", "3"))

    # Where the settings panel persists its model-slot choices. Never holds a key.
    llm_runtime_path: Path = Path(
        os.getenv("OPTIBOT_LLM_RUNTIME_PATH", str(BACKEND_DIR / "llm_runtime.json"))
    )

    max_tokens_baseline: int = 1024
    max_tokens_optimized: int = 700

    # --- Semantic cache --------------------------------------------------
    # Calibrated against all-MiniLM-L6-v2 on a labelled pair set (see
    # scripts/calibrate_cache.py). The should-hit and should-miss classes
    # overlap slightly, so this is tuned for precision over recall: a false
    # cache hit serves one customer another customer's answer, which costs
    # far more than a missed hit. 0.80 gave zero false hits on that set while
    # still matching genuine paraphrases (~0.82+).
    cache_threshold: float = float(os.getenv("OPTIBOT_CACHE_THRESHOLD", "0.80"))
    cache_ttl_order: int = int(os.getenv("OPTIBOT_CACHE_TTL_ORDER", "300"))
    cache_ttl_policy: int = int(os.getenv("OPTIBOT_CACHE_TTL_POLICY", "1800"))
    cache_max_entries: int = 500
    semantic_cache_backend: str = os.getenv("OPTIBOT_SEMANTIC_CACHE_BACKEND", "local")

    # --- RAG -------------------------------------------------------------
    rag_top_k: int = int(os.getenv("OPTIBOT_RAG_TOP_K", "3"))
    rag_min_score: float = float(os.getenv("OPTIBOT_RAG_MIN_SCORE", "0.15"))
    rag_chunk_tokens: int = 300
    rag_chunk_overlap: int = 50

    # --- Guardrails ------------------------------------------------------
    max_input_chars: int = 500
    rate_limit_per_minute: int = 20
    low_confidence_threshold: float = 0.7

    # --- Misc ------------------------------------------------------------
    # There is no `api_key` / `has_api_key` here on purpose. The gateway key can
    # be set at runtime from the settings panel, so a value frozen at import
    # would report the wrong thing forever. Ask app.llm_settings instead.
    # Comma-separated, for a lab deployment where the UI is not on port 3000.
    cors_origins: list[str] = [
        origin.strip()
        for origin in os.getenv(
            "OPTIBOT_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
        ).split(",")
        if origin.strip()
    ]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
