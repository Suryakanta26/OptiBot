"""Input-guardrail behavior, including the pre-LLM blocking boundary."""

from unittest.mock import Mock

from app.services import chat_service, guardrails, llm_client


def test_abusive_language_is_blocked():
    verdict = guardrails.check_input("You are a shit", "abuse-verdict")

    assert verdict.blocked is True
    assert verdict.triggers == ["abusive_language"]
    assert "rephrase" in (verdict.message or "").lower()


def test_abusive_language_never_calls_llm(monkeypatch):
    complete = Mock(side_effect=AssertionError("LLM must not be called"))
    monkeypatch.setattr(llm_client, "complete", complete)

    result = chat_service.handle("You are a shit", "abuse-pipeline", "optimized")

    assert result.metrics.blocked is True
    assert result.metrics.guardrail_events == ["abusive_language"]
    assert result.metrics.model is None
    assert result.metrics.total_tokens == 0
    complete.assert_not_called()
