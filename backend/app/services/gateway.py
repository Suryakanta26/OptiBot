"""Transport to the LiteLLM gateway — the only module that touches litellm.

OptiBot sends every model call through a LiteLLM gateway, which in a corporate
AI lab means: a private-CA TLS certificate, a bearer token that some gateways
only accept under a non-standard header, gateway-side model aliases rather than
provider model IDs, and a hardened ``/v1/models`` endpoint.

The reference implementation for all of this is ``usecase3-main`` (Node), and the
non-obvious parts are annotated below with what they mirror and why. Everything
litellm-shaped lives here; ``llm_client.py`` keeps the domain concepts (routing
slots, pricing, the ``LLMResult`` contract).
"""

from __future__ import annotations

import logging
import math
import random
import ssl
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import litellm

from app import llm_settings
from app.config import settings

log = logging.getLogger("optibot.gateway")

litellm.drop_params = True
litellm.suppress_debug_info = True
# We own the retry policy (see _is_rate_limited). litellm's own retry wrapper
# would multiply with ours — 3 attempts each becomes 9 requests at a gateway
# that is already rate-limiting us.
litellm.num_retries = 0

# The OpenAI SDK raises when api_key is None, but a bare local proxy needs no
# auth. LiteLLM ignores the value, so a sentinel is safer than None: passing
# None makes litellm fall back to a stray OPENAI_API_KEY from the environment.
_NOAUTH_SENTINEL = "sk-litellm-noauth"

_LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "[::1]", "0.0.0.0")


# --------------------------------------------------------------------------- #
# TLS bootstrap — module scope, exactly once, before any completion() call
# --------------------------------------------------------------------------- #

# A stray SSL_VERIFY is evicted in app.config, which has to happen before litellm
# is imported at all — see the comment there.

_FALSY = {"false", "0", "no", "off"}


def _resolve_verify() -> tuple[bool | str, str, str | None]:
    """Returns ``(httpx verify value, source label, warning)``."""
    bundle = settings.litellm_ca_bundle
    if bundle:
        if Path(bundle).is_file():
            return bundle, "LITELLM_CA_BUNDLE", None
        # Do not let a typo take the whole app down at import: httpx raises a bare
        # OSError for a missing bundle. Fall back to system CAs, say so loudly, and
        # surface it in the settings panel so it is diagnosable rather than fatal.
        return (
            True,
            "default",
            f"LITELLM_CA_BUNDLE points at {bundle!r}, which is not a readable file. "
            "Falling back to the system CA store — gateway calls will fail if the "
            "gateway uses a private CA.",
        )
    if settings.litellm_ssl_verify_raw.strip().lower() in _FALSY:
        return False, "LITELLM_SSL_VERIFY", None
    return True, "default", None


VERIFY, VERIFY_SOURCE, VERIFY_WARNING = _resolve_verify()

_TIMEOUT = httpx.Timeout(
    settings.litellm_timeout_s, connect=settings.litellm_connect_timeout_s
)


def _build_sessions(verify: bool | str) -> tuple[httpx.Client, httpx.AsyncClient]:
    return (
        httpx.Client(verify=verify, timeout=_TIMEOUT),
        httpx.AsyncClient(verify=verify, timeout=_TIMEOUT),
    )


try:
    _sync_session, _async_session = _build_sessions(VERIFY)
except (OSError, ssl.SSLError) as exc:
    # A CA bundle that exists but is not a valid PEM raises here rather than at
    # the is_file() check above. Degrade to the system store with a loud message
    # instead of taking the process down at import, so the settings panel can
    # still show what is wrong.
    VERIFY_WARNING = (
        f"LITELLM_CA_BUNDLE at {VERIFY!r} could not be loaded ({exc}). Falling back "
        "to the system CA store — gateway calls will fail if the gateway uses a "
        "private CA."
    )
    VERIFY, VERIFY_SOURCE = True, "default"
    _sync_session, _async_session = _build_sessions(True)

if VERIFY_WARNING:
    log.error(VERIFY_WARNING)

# Both assignments are required. litellm.ssl_verify covers litellm's own httpx
# handlers, but the openai-compatible route we use gets its socket from
# litellm.client_session, which litellm passes as `http_client=` into the OpenAI
# SDK. Setting only one makes LITELLM_SSL_VERIFY=false appear to do nothing.
litellm.ssl_verify = VERIFY
litellm.client_session = _sync_session
litellm.aclient_session = _async_session

if VERIFY is False:
    log.warning(
        "LITELLM_SSL_VERIFY=false — TLS certificate verification is DISABLED for "
        "outbound HTTPS to the LiteLLM gateway. Prefer "
        "LITELLM_CA_BUNDLE=/path/to/corporate-ca.pem when you can obtain the bundle."
    )
elif VERIFY_SOURCE == "LITELLM_CA_BUNDLE":
    log.info("Verifying the LiteLLM gateway against CA bundle %s.", VERIFY)


def tls_status() -> dict:
    """What the settings panel shows in its read-only TLS row."""
    return {
        "verify": VERIFY is not False,
        "ca_bundle": VERIFY if isinstance(VERIFY, str) else None,
        "source": VERIFY_SOURCE,
        "warning": VERIFY_WARNING,
        "note": (
            "Read-only: set LITELLM_CA_BUNDLE or LITELLM_SSL_VERIFY in backend/.env "
            "and restart the backend. It cannot be changed at runtime — LiteLLM "
            "caches its HTTP client per (key, base URL) and the TLS setting is not "
            "part of that cache key, so a live toggle would silently do nothing."
        ),
    }


# --------------------------------------------------------------------------- #
# Result / error types
# --------------------------------------------------------------------------- #


class GatewayError(RuntimeError):
    """A gateway failure already phrased for a human. See describe_error()."""


@dataclass
class GatewayResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    reported_cost_usd: float | None  # None = the gateway did not report one
    stop_reason: str | None = None


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #


def auth_headers(key: str | None) -> dict[str, str]:
    """Both header forms, always.

    LiteLLM accepts ``Authorization: Bearer <key>`` (OpenAI-compatible) or
    ``x-litellm-api-key: <key>`` (LiteLLM native). Some corporate gateways only
    honour the latter, so send both whenever a key is configured. Mirrors
    usecase3-main/backend/src/litellm.ts.
    """
    if not key:
        return {}
    return {"Authorization": f"Bearer {key}", "x-litellm-api-key": key}


def _is_loopback(base: str) -> bool:
    host = httpx.URL(base).host if base else ""
    return host in _LOOPBACK_HOSTS


def _resolve_key(base: str, override: str | None) -> str:
    key = (override or llm_settings.api_key() or "").strip()
    if key:
        return key
    if _is_loopback(base):
        # A bare `litellm --config ...` on localhost needs no auth.
        return _NOAUTH_SENTINEL
    raise GatewayError(
        "No LiteLLM API key is configured. Set it with the gear icon in the "
        "OptiBot header (applies immediately, in-process only), or add "
        "LITELLM_API_KEY to backend/.env to survive a restart."
    )


# --------------------------------------------------------------------------- #
# Chat completion
# --------------------------------------------------------------------------- #

_RATE_LIMIT_MARKERS = (
    '"code":"429"',
    '"code": "429"',
    "429 too many requests",
    "rate limit",
    "ratelimiterror",
    "resource_exhausted",
    "quota exceeded",
)


def _is_rate_limited(exc: Exception) -> bool:
    """usecase3 sniffs the response body for a wrapped 429; we sniff the exception.

    ``litellm.completion`` raises rather than returning a response, and gateways
    routinely wrap an upstream 429 in a non-429 envelope, so the status code
    alone is not enough.
    """
    if isinstance(exc, litellm.exceptions.RateLimitError):
        return True
    if getattr(exc, "status_code", None) == 429:
        return True
    blob = f"{exc}".lower()
    return any(marker in blob for marker in _RATE_LIMIT_MARKERS)


def _jitter(base_s: float) -> float:
    return base_s + random.uniform(-0.2 * base_s, 0.2 * base_s)


def chat(
    *,
    model: str,
    messages: list[dict],
    max_tokens: int | None = None,
    temperature: float | None = None,
    timeout_s: float | None = None,
    api_key_override: str | None = None,
    base_url_override: str | None = None,
    diagnose: bool = False,
) -> GatewayResult:
    """One chat completion through the gateway.

    ``api_key_override`` / ``base_url_override`` let the settings panel test a
    *pending* value without applying or persisting it — do not remove them in
    the name of simplification. ``diagnose`` adds a socket-level probe to a
    connection-failure message; it costs a round trip, so only the Test button
    asks for it.
    """
    base = (
        llm_settings.normalize_base_url(base_url_override)
        if base_url_override and base_url_override.strip()
        else llm_settings.base_url()
    )
    key = _resolve_key(base, api_key_override)

    kwargs: dict = {
        # The alias goes through raw. The gateway resolves its own model_list, so
        # `openai/genailab-maas-gpt-4o` is right and `openai/gemini/...` is the
        # single most common cause of a 404 that reads like a missing model.
        #
        # `openai/` rather than `litellm_proxy/`: both land in the same
        # openai-compatible code path, but litellm_proxy/ silently resolves its
        # base URL and key from LITELLM_PROXY_* — a second, invisible precedence
        # layer competing with the settings panel.
        "model": f"openai/{model}",
        "messages": messages,
        # Neither prefix appends /v1, so api_base must already carry it.
        "api_base": f"{base}/v1",
        # Always pass both explicitly and never None: litellm falls back to
        # OPENAI_API_KEY / OPENAI_API_BASE when either is None, which could route
        # a cleared key straight to api.openai.com.
        "api_key": key,
        "extra_headers": {"x-litellm-api-key": key},
        "timeout": timeout_s or settings.litellm_timeout_s,
        "num_retries": 0,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature

    attempts = max(1, settings.litellm_max_attempts)
    started = time.perf_counter()
    response = None

    for attempt in range(1, attempts + 1):
        try:
            response = litellm.completion(**kwargs)
            break
        except Exception as exc:  # noqa: BLE001 - re-raised as one type
            # Deliberately only 429 is retried. An unreachable gateway surfaces
            # as APIError(status_code=500), and retrying that turns one 10s
            # connect timeout into 30s of a hung "Test connection" button for the
            # most common misconfiguration there is.
            if attempt == attempts or not _is_rate_limited(exc):
                raise GatewayError(
                    describe_error(exc, base, model, diagnose_connection=diagnose)
                ) from exc
            time.sleep(_jitter(0.5 * attempt))  # 0.5s, 1.0s, +/-20%

    latency_ms = int((time.perf_counter() - started) * 1000)

    choice = response.choices[0]
    text = (getattr(choice.message, "content", None) or "").strip()
    finish_reason = getattr(choice, "finish_reason", None)
    if not text:
        # A blank body would otherwise flow into parse_structured() and be
        # recorded as a 0.6-confidence prose answer — a silent quality
        # regression rather than a failure.
        raise GatewayError(
            f"'{model}' returned empty content (finish_reason={finish_reason!r}). "
            "Common causes: a content filter on the upstream provider, or "
            "max_tokens consumed entirely by reasoning tokens."
        )

    usage = getattr(response, "usage", None)
    return GatewayResult(
        text=text,
        model=model,
        input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        latency_ms=latency_ms,
        reported_cost_usd=reported_cost(response),
        stop_reason=finish_reason,
    )


# --------------------------------------------------------------------------- #
# Cost reported by the gateway
# --------------------------------------------------------------------------- #


def _extra(obj) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    return dict(
        getattr(obj, "model_extra", None) or getattr(obj, "__pydantic_extra__", None) or {}
    )


def _finite_positive(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0 else None


def reported_cost(response) -> float | None:
    """LiteLLM puts the cost in one of several places depending on version.

    Two Python-specific traps: ``usage`` is a pydantic model, so proxy-added
    fields like ``response_cost`` live in ``model_extra`` and a plain ``getattr``
    returns None even when the field is present in the JSON; and a gateway that
    does not price an alias reports ``0.0``, which must not count as an answer.
    """
    usage = getattr(response, "usage", None)
    usage_extra = _extra(usage)
    response_extra = _extra(response)
    hidden = getattr(response, "_hidden_params", None) or {}

    for candidate in (
        getattr(usage, "response_cost", None),
        usage_extra.get("response_cost"),
        getattr(usage, "cost", None),
        usage_extra.get("cost"),
        getattr(response, "response_cost", None),
        response_extra.get("response_cost"),
        hidden.get("response_cost") if isinstance(hidden, dict) else None,
    ):
        cost = _finite_positive(candidate)
        if cost is not None:
            return cost
    return None


# --------------------------------------------------------------------------- #
# Error phrasing
# --------------------------------------------------------------------------- #


def describe_error(
    exc: Exception, base: str, model: str, *, diagnose_connection: bool = False
) -> str:
    """Turn a litellm exception into something actionable.

    Keyed on the exception classes first (more reliable than string matching),
    with a substring fallback for gateways that flatten provider errors into the
    message. The base URL is injected by us on purpose: litellm reduces an
    unreachable host to "APIError - Connection error." with neither the URL nor
    the underlying httpx cause attached.

    ``diagnose_connection`` adds a live probe for the real socket-level reason.
    It costs another round trip, so only the settings panel's Test button asks
    for it — a failing chat turn should not pay for diagnostics.
    """
    exceptions = litellm.exceptions
    blob = f"{exc}"
    low = blob.lower()

    auth_markers = ("invalid x-api-key", "api key not valid", "api_key_invalid",
                    "authentication_error", "401", "invalid_api_key")
    if isinstance(exc, (exceptions.AuthenticationError, exceptions.PermissionDeniedError)) or any(
        m in low for m in auth_markers
    ):
        return (
            f"The gateway at {base} rejected the request (authentication). Check the "
            "LiteLLM API key, and confirm the upstream provider keys are configured "
            "on the gateway itself — OptiBot never holds provider keys."
        )

    if isinstance(exc, exceptions.Timeout) or "timed out" in low or "timeout" in low:
        return (
            f"Model '{model}' at {base} did not finish within "
            f"{settings.litellm_timeout_s:g} seconds. The gateway is reachable; try a "
            "faster/non-reasoning model or increase LITELLM_TIMEOUT_S."
        )

    if isinstance(exc, exceptions.APIConnectionError) or "connection error" in low:
        detail = probe(base) if diagnose_connection else None
        suffix = f" Probe says: {detail}." if detail else ""
        return (
            f"Cannot reach the LiteLLM gateway at {base}. Confirm the URL (include any "
            "mount prefix, e.g. https://host/litellm), that it is reachable from this "
            f"host, and — for a private CA — set LITELLM_CA_BUNDLE in backend/.env.{suffix}"
        )

    if isinstance(exc, exceptions.NotFoundError) or "404" in low or "not found" in low:
        return (
            f"The gateway at {base} does not expose a model named '{model}'. Pick one "
            "from the list in the settings panel, or check the alias spelling."
        )

    if isinstance(exc, exceptions.RateLimitError) or "429" in low:
        return (
            f"The gateway rate-limited the request for '{model}' after "
            f"{settings.litellm_max_attempts} attempts. Try again shortly."
        )

    if isinstance(exc, exceptions.ContextWindowExceededError):
        return f"The prompt exceeded the context window of '{model}'."

    return f"{model}: {blob}"


def probe(base: str | None = None) -> str | None:
    """A raw request to /v1/models, purely for its error message.

    litellm destroys the useful part of a connection failure. A bare httpx call
    tells us whether it was DNS, a refused connection, or
    CERTIFICATE_VERIFY_FAILED — which is the difference between "wrong URL" and
    "set LITELLM_CA_BUNDLE".
    """
    url = f"{llm_settings.normalize_base_url(base)}/v1/models" if base else llm_settings.models_url()
    try:
        with httpx.Client(verify=VERIFY, timeout=settings.litellm_probe_timeout_s) as client:
            response = client.get(url, headers=auth_headers(llm_settings.api_key()))
        return None if response.status_code < 400 else f"HTTP {response.status_code}"
    except httpx.HTTPError as exc:
        return f"{type(exc).__name__}: {exc}"


def is_reachable(base: str | None = None) -> bool:
    return probe(base) is None


# --------------------------------------------------------------------------- #
# Model discovery
# --------------------------------------------------------------------------- #

# Some LiteLLM deployments return 405 unless the listing flags are present.
# These match what the gateway's own swagger UI sends and are no-ops against
# vanilla LiteLLM. Mirrors usecase3-main/backend/src/litellm.ts.
_MODELS_QUERY = {
    "return_wildcard_routes": "false",
    "include_model_access_groups": "false",
    "only_model_access_groups": "false",
    "include_metadata": "false",
}


def list_model_ids(*, base_url: str | None = None, api_key: str | None = None) -> list[str]:
    """The aliases the gateway actually exposes.

    Plain httpx rather than litellm's own ``get_models()``: that helper hits
    ``{base}/models`` without the hardening flags above and re-prefixes every id
    with ``litellm_proxy/``, which is exactly wrong for a picker.
    """
    base = (
        llm_settings.normalize_base_url(base_url)
        if base_url and base_url.strip()
        else llm_settings.base_url()
    )
    key = (api_key or "").strip() or llm_settings.api_key()
    url = f"{base}/v1/models"

    try:
        with httpx.Client(verify=VERIFY, timeout=settings.litellm_models_timeout_s) as client:
            response = client.get(
                url,
                params=_MODELS_QUERY,
                headers={"Accept": "application/json", **auth_headers(key)},
            )
    except httpx.HTTPError as exc:
        raise GatewayError(f"Cannot reach {url} — {type(exc).__name__}: {exc}") from exc

    if response.status_code >= 400:
        raise GatewayError(
            f"GET {url} failed with HTTP {response.status_code}: {response.text[:300]}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise GatewayError(
            f"GET {url} returned non-JSON ({response.text[:160]!r}). Is that really "
            "a LiteLLM gateway?"
        ) from exc

    ids = [
        str(entry.get("id")).strip()
        for entry in (payload.get("data") or [])
        if isinstance(entry, dict) and str(entry.get("id") or "").strip()
    ]
    return sorted(dict.fromkeys(ids))


def resolve_models(
    *, base_url: str | None = None, api_key: str | None = None
) -> tuple[list[str], str, str | None]:
    """``(models, source, warning)`` — never raises, so the panel always renders."""
    try:
        ids = list_model_ids(base_url=base_url, api_key=api_key)
    except GatewayError as exc:
        return list(llm_settings.FALLBACK_MODELS), "fallback", str(exc)

    if not ids:
        # A hardened gateway with an access-group filter answers 200 with an
        # empty list, which would otherwise be an empty dropdown and no reason.
        return (
            list(llm_settings.FALLBACK_MODELS),
            "fallback",
            "The gateway returned an empty model list — your key may not have "
            "access to any model group.",
        )
    return ids, "litellm", None
