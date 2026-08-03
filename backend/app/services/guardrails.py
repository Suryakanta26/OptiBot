"""Layer 5 — input and output guardrails.

Input side: prompt-injection detection, length limits, abuse filtering.
Output side: the important one — every order ID the model states is checked
against the database before the response reaches the customer. A model that
invents ORD-99999 gets that claim stripped rather than delivered.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

from app.config import settings
from app.services.order_service import extract_order_ids, order_exists

# --------------------------------------------------------------------------
# Input guardrails
# --------------------------------------------------------------------------

_INJECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("instruction_override", re.compile(
        r"\b(ignore|disregard|forget|override)\b.{0,30}\b"
        r"(previous|prior|above|earlier|all)\b.{0,20}\b(instruction|prompt|rule|direction)",
        re.IGNORECASE)),
    ("system_prompt_probe", re.compile(
        r"\b(system\s*prompt|initial\s+instruction|your\s+instructions|"
        r"reveal\s+your|print\s+your\s+(?:prompt|rules)|what\s+are\s+your\s+instructions)\b",
        re.IGNORECASE)),
    ("role_switch", re.compile(
        r"\b(you\s+are\s+now|from\s+now\s+on\s+you|act\s+as\s+(?:a\s+)?(?:developer|admin|root)|"
        r"pretend\s+(?:to\s+be|you\s+are)|enter\s+(?:developer|debug|god)\s+mode|jailbreak|DAN\s+mode)\b",
        re.IGNORECASE)),
    ("delimiter_injection", re.compile(
        r"(</?(?:system|assistant|instructions?)>|\[\/?INST\]|###\s*system)",
        re.IGNORECASE)),
    ("data_exfiltration", re.compile(
        r"\b(list|show|dump|give\s+me)\b.{0,25}\b(all\s+(?:customers?|orders?|emails?)|"
        r"every\s+(?:customer|order)|entire\s+database|all\s+users)\b",
        re.IGNORECASE)),
    ("credential_request", re.compile(
        r"\b(api[\s_-]?key|password|secret\s+key|access\s+token|credential)\b",
        re.IGNORECASE)),
]

_ABUSE_RE = re.compile(r"\b(fuck|shit|bitch|asshole|bastard)\w*\b", re.IGNORECASE)


@dataclass
class InputVerdict:
    allowed: bool
    query: str
    triggers: list[str] = field(default_factory=list)
    message: str | None = None

    @property
    def blocked(self) -> bool:
        return not self.allowed


class RateLimiter:
    """Fixed-window limiter, per session."""

    def __init__(self, limit: int, window_seconds: int = 60) -> None:
        self._limit = limit
        self._window = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def check(self, session_id: str) -> bool:
        now = time.time()
        bucket = self._events[session_id]
        while bucket and now - bucket[0] > self._window:
            bucket.popleft()
        if len(bucket) >= self._limit:
            return False
        bucket.append(now)
        return True

    def reset(self) -> None:
        self._events.clear()


rate_limiter = RateLimiter(settings.rate_limit_per_minute)


def check_input(query: str, session_id: str) -> InputVerdict:
    triggers: list[str] = []
    text = (query or "").strip()

    if not text:
        return InputVerdict(False, text, ["empty_input"], "Please enter a question.")

    if not rate_limiter.check(session_id):
        return InputVerdict(
            False, text, ["rate_limit"],
            "You're sending messages faster than we can answer. Please wait a moment.",
        )

    for label, pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            triggers.append(label)

    if triggers:
        return InputVerdict(
            False, text, triggers,
            "I can only help with ShopFast orders, returns, shipping, and warranty "
            "questions. Ask me about an order number and I'll look it up.",
        )

    if _ABUSE_RE.search(text):
        # This is a terminal verdict: the orchestrator returns it before any
        # classification, retrieval, caching, or model invocation.
        return InputVerdict(
            False,
            text,
            ["abusive_language"],
            "I can help with ShopFast orders, returns, shipping, and warranty "
            "questions, but I can't process abusive messages. Please rephrase "
            "your request respectfully.",
        )

    if len(text) > settings.max_input_chars:
        text = text[: settings.max_input_chars]
        triggers.append("input_truncated")

    return InputVerdict(True, text, triggers)


# --------------------------------------------------------------------------
# Output guardrails
# --------------------------------------------------------------------------

_FORBIDDEN_RE = re.compile(
    r"\b(amazon|ebay|walmart|temu|internal\s+(?:tool|system|database)|"
    r"our\s+(?:vendor|supplier)\s+portal)\b",
    re.IGNORECASE,
)

_TRACKING_RE = re.compile(r"\b(?:FX|1Z|94)\d{8,12}\b")

_LOW_CONFIDENCE_PREFIX = "I'm not fully certain, but "


@dataclass
class OutputVerdict:
    response: str
    triggers: list[str] = field(default_factory=list)
    invalid_order_ids: list[str] = field(default_factory=list)
    invalid_tracking: list[str] = field(default_factory=list)


def check_output(
    response: str,
    confidence: float,
    *,
    allowed_order_ids: list[str],
    allowed_tracking: list[str],
) -> OutputVerdict:
    """Verify the model only asserted facts the database supports.

    ``allowed_order_ids`` / ``allowed_tracking`` are what the pipeline actually
    resolved for this request. Anything the model mentions outside those sets
    is a hallucination, and gets removed rather than shipped.
    """
    triggers: list[str] = []
    text = response or ""

    allowed_orders = {o.upper() for o in allowed_order_ids}
    invalid_orders: list[str] = []
    for order_id in extract_order_ids(text):
        # An ID is acceptable if this request resolved it, or if it at least
        # exists (the customer may have named a valid order we did not fetch).
        if order_id.upper() not in allowed_orders and not order_exists(order_id):
            invalid_orders.append(order_id)

    if invalid_orders:
        triggers.append("fabricated_order_id")
        for bad in invalid_orders:
            text = re.sub(
                rf"\b{re.escape(bad)}\b", "[unverified order number]", text,
                flags=re.IGNORECASE,
            )

    allowed_track = {t.upper() for t in allowed_tracking}
    invalid_tracking = [
        t for t in _TRACKING_RE.findall(text) if t.upper() not in allowed_track
    ]
    if invalid_tracking:
        triggers.append("fabricated_tracking_number")
        for bad in invalid_tracking:
            text = text.replace(bad, "[unverified tracking number]")

    if _FORBIDDEN_RE.search(text):
        triggers.append("forbidden_content")
        text = _FORBIDDEN_RE.sub("[redacted]", text)

    if confidence < settings.low_confidence_threshold:
        triggers.append("low_confidence")
        if not text.startswith(_LOW_CONFIDENCE_PREFIX):
            text = _LOW_CONFIDENCE_PREFIX + text[0].lower() + text[1:] if text else text

    return OutputVerdict(
        response=text,
        triggers=triggers,
        invalid_order_ids=invalid_orders,
        invalid_tracking=invalid_tracking,
    )
