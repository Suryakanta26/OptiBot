"""Cache behavior across LangGraph-checkpointed chat turns."""

import json
from unittest.mock import Mock

from app.services import chat_service, guardrails, llm_client
from app.services.cache_service import semantic_cache


def test_repeated_policy_question_in_same_session_skips_second_llm_call(monkeypatch):
    semantic_cache.clear()
    guardrails.rate_limiter.reset()
    complete = Mock(
        return_value=llm_client.LLMResult(
            text=json.dumps(
                {
                    "response": "Returns are accepted within 30 days.",
                    "confidence": 0.95,
                    "sources": ["return_policy.md"],
                }
            ),
            model="test-model",
            input_tokens=20,
            output_tokens=10,
            latency_ms=5,
            cost_usd=0.001,
        )
    )
    monkeypatch.setattr(llm_client, "complete", complete)
    monkeypatch.setattr(
        chat_service.rag_service,
        "retrieve",
        lambda _query: [
            {
                "source": "return_policy.md",
                "heading": "Return window",
                "text": "Returns are accepted within 30 days.",
            }
        ],
    )

    session = "repeated-policy-cache"
    first = chat_service.handle("What is your return policy?", session, "optimized")
    second = chat_service.handle("What is your return policy?", session, "optimized")

    assert first.metrics.cache_hit is False
    assert second.metrics.cache_hit is True
    assert second.metrics.cache_level == "exact"
    assert second.metrics.model == "cache"
    assert second.metrics.total_tokens == 0
    assert complete.call_count == 1
