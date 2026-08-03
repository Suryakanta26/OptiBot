"""Layer 1's execution half — model routing, plus token/cost accounting.

The transport lives in ``services/gateway.py``; this module owns the domain: which
model a given (mode, tier) should use, and what the call cost.

Prices are declared locally rather than read from the gateway or from LiteLLM's
cost map. Both lag new model releases, and a demo whose headline metric is cost
reduction cannot have its cost numbers silently fall back to something plausible
but wrong. Rates are provider list prices in USD per million tokens.

Note the deliberate policy in ``complete()``: the local table *outranks* the
gateway's own ``response_cost``. A corporate gateway commonly prices some aliases
and reports 0.0 for others, and mixing sources across the baseline and optimized
arms would corrupt the comparison in a way nothing in the dashboard could show.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from app import llm_settings
from app.services import gateway  # noqa: F401 - importing it runs the TLS bootstrap
from app.services.langchain_model import GatewayChatModel

# USD per 1M tokens (input, output).
PRICING: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    # Google Gemini
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    # OpenAI, including the TCS GenAI Lab MaaS alias
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "genailab-maas-gpt-4o": (2.50, 10.00),
}

# Substring fallbacks, longest first, for the lab-prefixed and version-suffixed
# aliases a gateway exposes (genailab-maas-gpt-4o, vertex-gemini-2.5-pro,
# gemini-2.5-flash-002). Order matters: "gemini-2.5-flash-lite" must be tested
# before "gemini-2.5-flash" or the lite tier gets priced 3x too high.
_PRICING_PATTERNS: tuple[tuple[str, tuple[float, float]], ...] = tuple(
    sorted(
        {
            "gemini-2.5-flash-lite": (0.10, 0.40),
            "gemini-2.5-flash": (0.30, 2.50),
            "gemini-2.5-pro": (1.25, 10.00),
            "gemini-3": (1.25, 10.00),
            "gpt-4o-mini": (0.15, 0.60),
            "gpt-4o": (2.50, 10.00),
            "claude-haiku": (1.00, 5.00),
            "claude-sonnet": (3.00, 15.00),
            "claude-opus": (5.00, 25.00),
        }.items(),
        key=lambda item: -len(item[0]),
    )
)

_DEFAULT_PRICE = (3.00, 15.00)


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: float
    stop_reason: str | None = None
    # How cost_usd was derived: local-exact, local-prefix, local-default or
    # gateway. Surfaced in the settings panel so a mispriced alias is visible.
    cost_basis: str = "local-exact"


def price_for_with_source(model: str) -> tuple[tuple[float, float], str]:
    """``((input_rate, output_rate), "exact" | "prefix" | "default")``."""
    key = model.split("/")[-1].strip().lower()
    if key in PRICING:
        return PRICING[key], "exact"
    for needle, rate in _PRICING_PATTERNS:
        if needle in key:
            return rate, "prefix"
    return _DEFAULT_PRICE, "default"


def price_for(model: str) -> tuple[float, float]:
    rate, _ = price_for_with_source(model)
    return rate


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    in_rate, out_rate = price_for(model)
    return round(
        (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate,
        8,
    )


def select_model(mode: str, tier: str) -> str:
    """The routing decision itself.

    Baseline ignores the tier entirely — one expensive model for every query,
    which is the behaviour being measured against.
    """
    if mode == "baseline":
        return llm_settings.model_for("baseline")
    return llm_settings.model_for("simple" if tier == "simple" else "complex")


def resolve_cost(
    model: str, input_tokens: int, output_tokens: int, reported_cost: float | None
) -> tuple[float, str]:
    """Local price table first; the gateway's figure only for unknown aliases.

    See the module docstring for why this order is deliberate rather than a
    missed optimisation.
    """
    _, price_source = price_for_with_source(model)
    if price_source == "default" and reported_cost is not None:
        return reported_cost, "gateway"
    return estimate_cost(model, input_tokens, output_tokens), f"local-{price_source}"


def complete(
    *,
    model: str,
    system: str,
    messages: list[dict],
    max_tokens: int,
) -> LLMResult:
    langchain_messages = [SystemMessage(content=system)]
    for message in messages:
        # Optimized history currently contains user/assistant turns only.
        if message.get("role") == "assistant":
            from langchain_core.messages import AIMessage

            langchain_messages.append(AIMessage(content=message.get("content", "")))
        else:
            langchain_messages.append(HumanMessage(content=message.get("content", "")))
    try:
        response = GatewayChatModel(model=model, max_tokens=max_tokens).invoke(langchain_messages)
    except gateway.GatewayError as exc:
        # LLMError stays the only exception type crossing into chat_service, so
        # its two handlers (and the audit trail they write) keep working.
        raise LLMError(str(exc)) from exc

    usage = response.usage_metadata or {}
    metadata = response.response_metadata or {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    cost_usd, cost_basis = resolve_cost(
        model, input_tokens, output_tokens, metadata.get("reported_cost_usd")
    )

    return LLMResult(
        text=str(response.content),
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=int(metadata.get("latency_ms", 0) or 0),
        cost_usd=cost_usd,
        stop_reason=metadata.get("stop_reason"),
        cost_basis=cost_basis,
    )


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_structured(text: str) -> tuple[str, float, list[str]]:
    """Extract (response, confidence, sources) from the optimized model output.

    The optimized prompt asks for JSON. Models occasionally wrap it in prose or
    a code fence, so parse defensively and degrade to treating the whole reply
    as the response rather than failing the request.
    """
    if not text:
        return "", 0.0, []

    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate).strip()

    match = _JSON_BLOCK_RE.search(candidate)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict) and "response" in data:
                confidence = data.get("confidence", 0.8)
                try:
                    confidence = float(confidence)
                except (TypeError, ValueError):
                    confidence = 0.8
                sources = data.get("sources") or []
                if not isinstance(sources, list):
                    sources = [str(sources)]
                return (
                    str(data["response"]).strip(),
                    max(0.0, min(1.0, confidence)),
                    [str(s) for s in sources],
                )
        except json.JSONDecodeError:
            pass

    # Not JSON — the model answered in prose. Usable, but flag it as lower
    # confidence since the output contract was not honoured.
    return candidate, 0.6, []
