"""Pipeline orchestration — the baseline and optimized paths side by side.

Both paths answer the same question from the same data. The difference is
everything between receiving the query and returning the answer, which is what
the dashboard measures.

baseline
    rate limit -> dump the whole order record -> verbose system prompt ->
    one expensive model -> return whatever came back -> log it unmasked

optimized
    input guardrails -> classify -> resolve only the needed order fields ->
    retrieve policy chunks -> semantic cache -> route to a model by tier ->
    compressed structured prompt -> parse -> output guardrails -> cache ->
    PII-mask -> audit log
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from app.config import settings
from app.models.schemas import ChatMetrics, ChatResponse, TraceStep
from app.services import guardrails, llm_client, metrics_service, order_service
from app.services import pii_detector, prompts, rag_service
from app.services.cache_service import build_identity, semantic_cache
from app.services.classifier import classify


class _Timer:
    """Collects per-layer timings for the trace shown in the UI."""

    def __init__(self) -> None:
        self.steps: list[TraceStep] = []
        self._mark = time.perf_counter()

    def step(self, layer: str, detail: str) -> None:
        now = time.perf_counter()
        self.steps.append(
            TraceStep(layer=layer, detail=detail, duration_ms=int((now - self._mark) * 1000))
        )
        self._mark = now


def _collect_order_context(order_ids: list[str], *, verbose: bool):
    """Resolve every referenced order, tracking which IDs were real."""
    contexts, resolved, tracking, missing = [], [], [], []
    for order_id in order_ids:
        ctx = order_service.build_order_context(order_id, verbose=verbose)
        if ctx is None:
            missing.append(order_id)
            continue
        contexts.append(ctx)
        resolved.append(order_id.upper())
        order = order_service.get_order(order_id)
        if order and order.get("tracking_number"):
            tracking.append(order["tracking_number"])
    return contexts, resolved, tracking, missing


# --------------------------------------------------------------------------
# Baseline
# --------------------------------------------------------------------------

def _run_baseline(query: str, session_id: str) -> ChatResponse:
    timer = _Timer()
    metrics = ChatMetrics(mode="baseline")

    # The only control in the baseline path — without it a demo can be
    # accidentally DoS'd, which would say nothing about GenAI quality.
    if not guardrails.rate_limiter.check(session_id):
        metrics.blocked = True
        metrics.guardrail_events = ["rate_limit"]
        return ChatResponse(
            response="Too many requests. Please wait a moment.",
            metrics=metrics,
            trace=timer.steps,
        )

    order_ids = order_service.extract_order_ids(query)
    order_blob = None
    if order_ids:
        # Whole record, PII and every tracking scan included.
        order_blob = order_service.build_order_context(order_ids[0], verbose=True)
    timer.step("context", f"raw order dump for {order_ids[:1] or 'n/a'}")

    system, messages = prompts.build_baseline_messages(query, order_blob)
    model = llm_client.select_model("baseline", "n/a")
    timer.step("routing", f"single model for every query -> {model}")

    try:
        result = llm_client.complete(
            model=model, system=system, messages=messages,
            max_tokens=settings.max_tokens_baseline,
        )
    except llm_client.LLMError as exc:
        metrics.model = model
        metrics_service.record_interaction(
            {"session_id": session_id, "mode": "baseline", "query_masked": query,
             "model": model, "error": str(exc), "blocked": 0}
        )
        # A failed turn still has to leave an audit trail — "no record" is the
        # one outcome an auditor cannot distinguish from "never happened".
        metrics_service.record_audit(
            {"session_id": session_id, "mode": "baseline", "event_type": "error",
             "query_masked": query, "model": model,
             "pii_detected": pii_detector.scan(query), "pii_masked": 0,
             "detail": str(exc)}
        )
        return ChatResponse(
            response="", metrics=metrics, trace=timer.steps, error=str(exc)
        )

    timer.step("llm", f"{model} answered in {result.latency_ms}ms")

    metrics.model = result.model
    metrics.input_tokens = result.input_tokens
    metrics.output_tokens = result.output_tokens
    metrics.total_tokens = result.input_tokens + result.output_tokens
    metrics.latency_ms = result.latency_ms
    metrics.cost_usd = result.cost_usd
    # No output validation and no confidence contract in the baseline; the
    # score is nominal so the dashboard has a comparable column.
    metrics.confidence = 0.5

    # PII is detected for the metric but deliberately NOT masked before
    # storage — this is the gap the optimized path closes.
    pii = pii_detector.scan(query + " " + result.text)
    metrics.pii_detected = pii
    metrics.pii_masked = False

    metrics_service.record_interaction(
        {
            "session_id": session_id, "mode": "baseline", "query_masked": query,
            "tier": None, "model": result.model,
            "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
            "total_tokens": metrics.total_tokens, "latency_ms": result.latency_ms,
            "cost_usd": result.cost_usd, "cache_hit": 0, "rag_used": 0,
            "rag_sources": [], "confidence": metrics.confidence,
            "guardrail_events": [], "blocked": 0, "error": None,
        }
    )
    metrics_service.record_audit(
        {
            "session_id": session_id, "mode": "baseline", "event_type": "chat",
            "query_masked": query,            # unmasked on purpose
            "response_masked": result.text,   # unmasked on purpose
            "model": result.model, "pii_detected": pii, "pii_masked": 0,
            "guardrail_events": [], "detail": "no guardrails applied",
        }
    )
    timer.step("logging", f"stored unmasked; PII categories present: {pii or 'none'}")

    return ChatResponse(
        response=result.text, metrics=metrics, trace=timer.steps, sources=[]
    )


# --------------------------------------------------------------------------
# Optimized
# --------------------------------------------------------------------------

def _run_optimized(
    query: str,
    session_id: str,
    chat_history: list[dict] | None = None,
) -> ChatResponse:
    timer = _Timer()
    metrics = ChatMetrics(mode="optimized")

    # --- Layer 5a: input guardrails ------------------------------------
    verdict = guardrails.check_input(query, session_id)
    metrics.guardrail_events = list(verdict.triggers)
    if verdict.blocked:
        metrics.blocked = True
        masked = pii_detector.mask(query)
        metrics.pii_detected = masked.found
        metrics.pii_masked = True
        timer.step("guardrails-in", f"blocked: {', '.join(verdict.triggers)}")

        metrics_service.record_interaction(
            {"session_id": session_id, "mode": "optimized",
             "query_masked": masked.text, "guardrail_events": verdict.triggers,
             "blocked": 1, "confidence": 0.0}
        )
        metrics_service.record_audit(
            {"session_id": session_id, "mode": "optimized",
             "event_type": "blocked_input", "query_masked": masked.text,
             "pii_detected": masked.found, "pii_masked": 1,
             "guardrail_events": verdict.triggers,
             "detail": "request rejected before reaching the model"}
        )
        return ChatResponse(
            response=verdict.message or "That request cannot be processed.",
            metrics=metrics, trace=timer.steps,
        )

    query = verdict.query
    timer.step("guardrails-in", "passed" if not verdict.triggers
               else f"passed with warnings: {', '.join(verdict.triggers)}")

    # --- Layer 1a: classification --------------------------------------
    cls = classify(query)
    metrics.tier = cls.tier
    timer.step("classifier", f"{cls.tier} ({cls.reason})")

    # --- Context: only the fields this query needs ---------------------
    order_contexts, resolved_ids, tracking, missing = _collect_order_context(
        cls.order_ids, verbose=False
    ) if cls.needs_order_data else ([], [], [], [])

    if missing:
        metrics.guardrail_events.append("unknown_order_id")
    if cls.needs_order_data:
        timer.step("context", f"resolved {len(resolved_ids)} order(s)"
                              + (f", {len(missing)} unknown" if missing else ""))

    # --- Layer 3: RAG ---------------------------------------------------
    policy_chunks: list[dict] = []
    if cls.needs_rag:
        policy_chunks = rag_service.retrieve(query)
        metrics.rag_used = bool(policy_chunks)
        metrics.rag_sources = sorted({c["source"] for c in policy_chunks})
        timer.step(
            "rag",
            f"{len(policy_chunks)} chunk(s) from {', '.join(metrics.rag_sources) or 'no match'}",
        )

    # A complete policy question is grounded entirely by the retrieved policy
    # chunks. Letting unrelated thread history affect its prompt would also
    # make the cache identity change on every turn. Follow-ups and order turns
    # still retain history because their meaning can depend on prior messages.
    effective_history = (
        None
        if cls.tier == "medium" and cls.needs_rag and not cls.needs_order_data
        else chat_history
    )

    # --- Layer 4: exact cache, then semantic policy cache ---------------
    cache_identity = build_identity(
        query=query,
        tier=cls.tier,
        order_contexts=order_contexts,
        policy_chunks=policy_chunks,
        conversation_context=effective_history,
    )
    if cls.is_cacheable:
        lookup = semantic_cache.lookup_identity(cache_identity)
        metrics.cache_similarity = lookup.similarity
        if lookup.hit and lookup.entry is not None:
            entry = lookup.entry
            metrics.cache_hit = True
            metrics.cache_level = lookup.level
            metrics.cache_entry_age_ms = lookup.age_ms
            metrics.model = "cache"
            metrics.confidence = entry.confidence
            metrics.latency_ms = int((time.perf_counter() - timer._mark) * 1000)
            timer.step("cache", f"HIT (similarity {lookup.similarity}) — no model call")

            masked_q = pii_detector.mask(query)
            masked_r = pii_detector.mask(entry.response)
            metrics.pii_detected = sorted(set(masked_q.found + masked_r.found))
            metrics.pii_masked = True

            metrics_service.record_interaction(
                {"session_id": session_id, "mode": "optimized",
                 "query_masked": masked_q.text, "tier": cls.tier,
                 "model": "cache", "input_tokens": 0, "output_tokens": 0,
                 "total_tokens": 0, "latency_ms": metrics.latency_ms,
                 "cost_usd": 0.0, "cache_hit": 1,
                 "rag_used": int(metrics.rag_used), "rag_sources": metrics.rag_sources,
                 "confidence": entry.confidence,
                 "guardrail_events": metrics.guardrail_events, "blocked": 0,
                 "error": None}
            )
            metrics_service.record_audit(
                {"session_id": session_id, "mode": "optimized",
                 "event_type": "cache_hit", "query_masked": masked_q.text,
                 "response_masked": masked_r.text, "model": "cache",
                 "pii_detected": metrics.pii_detected, "pii_masked": 1,
                 "guardrail_events": metrics.guardrail_events,
                 "detail": f"similarity {lookup.similarity}"}
            )
            return ChatResponse(
                response=entry.response, metrics=metrics, trace=timer.steps,
                sources=entry.sources,
            )
        timer.step("cache", f"MISS (best similarity {lookup.similarity})")

    # --- Layer 2 + 1b: compressed prompt, routed model ------------------
    system, messages = prompts.build_optimized_messages(
        query,
        order_contexts,
        policy_chunks,
        should_escalate=cls.should_escalate,
        chat_history=effective_history,
    )
    model = llm_client.select_model("optimized", cls.tier)
    timer.step("routing", f"tier '{cls.tier}' -> {model}")

    try:
        result = llm_client.complete(
            model=model, system=system, messages=messages,
            max_tokens=settings.max_tokens_optimized,
        )
    except llm_client.LLMError as exc:
        masked_q = pii_detector.mask(query)
        metrics.model = model
        metrics.pii_detected = masked_q.found
        metrics.pii_masked = True
        metrics_service.record_interaction(
            {"session_id": session_id, "mode": "optimized",
             "query_masked": masked_q.text, "tier": cls.tier,
             "model": model, "error": str(exc), "blocked": 0}
        )
        metrics_service.record_audit(
            {"session_id": session_id, "mode": "optimized", "event_type": "error",
             "query_masked": masked_q.text, "model": model,
             "pii_detected": masked_q.found, "pii_masked": 1,
             "guardrail_events": metrics.guardrail_events, "detail": str(exc)}
        )
        return ChatResponse(
            response="", metrics=metrics, trace=timer.steps, error=str(exc)
        )

    timer.step("llm", f"{model} answered in {result.latency_ms}ms")

    answer, confidence, sources = llm_client.parse_structured(result.text)

    # --- Layer 5b: output guardrails ------------------------------------
    out = guardrails.check_output(
        answer, confidence,
        allowed_order_ids=resolved_ids, allowed_tracking=tracking,
    )
    answer = out.response
    metrics.guardrail_events.extend(out.triggers)
    timer.step(
        "guardrails-out",
        f"{', '.join(out.triggers)}" if out.triggers else "response verified against order data",
    )

    metrics.model = result.model
    metrics.input_tokens = result.input_tokens
    metrics.output_tokens = result.output_tokens
    metrics.total_tokens = result.input_tokens + result.output_tokens
    metrics.latency_ms = result.latency_ms
    metrics.cost_usd = result.cost_usd
    metrics.confidence = confidence

    # --- Layer 4 write ---------------------------------------------------
    # Only cache answers that survived validation. Caching a response that
    # tripped a guardrail would serve the same bad answer repeatedly.
    if cls.is_cacheable and not out.triggers and confidence >= settings.low_confidence_threshold:
        semantic_cache.store_identity(cache_identity, answer, confidence, sources)
        timer.step(
            "cache",
            "response stored in exact cache"
            + (" + semantic policy cache" if cache_identity.semantic_allowed else ""),
        )

    # --- Layer 5c: PII masking before anything is persisted -------------
    masked_q = pii_detector.mask(query)
    masked_r = pii_detector.mask(answer)
    metrics.pii_detected = sorted(set(masked_q.found + masked_r.found))
    metrics.pii_masked = True

    metrics_service.record_interaction(
        {
            "session_id": session_id, "mode": "optimized",
            "query_masked": masked_q.text, "tier": cls.tier, "model": result.model,
            "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
            "total_tokens": metrics.total_tokens, "latency_ms": result.latency_ms,
            "cost_usd": result.cost_usd, "cache_hit": 0,
            "rag_used": int(metrics.rag_used), "rag_sources": metrics.rag_sources,
            "confidence": confidence, "guardrail_events": metrics.guardrail_events,
            "blocked": 0, "error": None,
        }
    )
    metrics_service.record_audit(
        {
            "session_id": session_id, "mode": "optimized", "event_type": "chat",
            "query_masked": masked_q.text, "response_masked": masked_r.text,
            "model": result.model, "pii_detected": metrics.pii_detected,
            "pii_masked": 1, "guardrail_events": metrics.guardrail_events,
            "detail": f"tier={cls.tier}; sources={','.join(sources) or 'none'}",
        }
    )
    timer.step("logging", "PII masked, interaction + audit records written")

    return ChatResponse(
        response=answer, metrics=metrics, trace=timer.steps,
        sources=sources or metrics.rag_sources,
    )


class OptimizedState(MessagesState):
    """Serializable state checkpointed independently for each chat session."""

    current_query: str
    session_id: str
    response: ChatResponse


def _message_history(messages: list[Any]) -> list[dict]:
    history: list[dict] = []
    # The final HumanMessage is the current request; only earlier messages are
    # injected as history. Bound the context even though the checkpointer keeps
    # the complete thread for future summarization/persistence upgrades.
    for message in messages[:-1][-8:]:
        role = "assistant" if isinstance(message, AIMessage) else "user"
        history.append({"role": role, "content": str(message.content)})
    return history


def _execute_optimized(state: OptimizedState) -> dict:
    response = _run_optimized(
        state["current_query"],
        state["session_id"],
        chat_history=_message_history(state["messages"]),
    )
    # Only masked content is checkpointed. The unmasked response remains in the
    # request-scoped response field returned to FastAPI.
    checkpointed = pii_detector.mask(response.response).text
    return {"messages": [AIMessage(content=checkpointed)], "response": response}


_workflow_builder = StateGraph(OptimizedState)
_workflow_builder.add_node("optimized_pipeline", _execute_optimized)
_workflow_builder.add_edge(START, "optimized_pipeline")
_workflow_builder.add_edge("optimized_pipeline", END)
optimized_workflow = _workflow_builder.compile(checkpointer=InMemorySaver())


def handle(query: str, session_id: str, mode: str) -> ChatResponse:
    if mode == "baseline":
        return _run_baseline(query, session_id)
    masked_query = pii_detector.mask(query).text
    result = optimized_workflow.invoke(
        {
            "messages": [HumanMessage(content=masked_query)],
            "current_query": query,
            "session_id": session_id,
        },
        config={"configurable": {"thread_id": session_id}},
    )
    return result["response"]
