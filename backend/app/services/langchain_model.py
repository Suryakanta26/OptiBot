"""LangChain chat-model interface backed by OptiBot's hardened LiteLLM gateway.

The dedicated ChatLiteLLM integration cannot currently express all of this
repository's private-CA and dual-header transport rules. This small adapter
keeps those rules in ``gateway.py`` while exposing the standard BaseChatModel
interface to prompts, LangGraph, callbacks, and future streaming work.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.services import gateway


class GatewayChatModel(BaseChatModel):
    model: str
    max_tokens: int

    @property
    def _llm_type(self) -> str:
        return "optibot-litellm-gateway"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "max_tokens": self.max_tokens}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        role_map = {"human": "user", "ai": "assistant", "system": "system"}
        payload = [
            {"role": role_map.get(message.type, message.type), "content": str(message.content)}
            for message in messages
        ]
        result = gateway.chat(
            model=self.model,
            messages=payload,
            max_tokens=int(kwargs.get("max_tokens", self.max_tokens)),
        )
        message = AIMessage(
            content=result.text,
            usage_metadata={
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "total_tokens": result.input_tokens + result.output_tokens,
            },
            response_metadata={
                "model": result.model,
                "latency_ms": result.latency_ms,
                "reported_cost_usd": result.reported_cost_usd,
                "stop_reason": result.stop_reason,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])
