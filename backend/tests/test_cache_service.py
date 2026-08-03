from app.services.cache_service import LayeredResponseCache, build_identity


def _identity(query="What is the return policy?", *, tier="medium", orders=None, policies=None):
    return build_identity(
        query=query,
        tier=tier,
        order_contexts=orders or [],
        policy_chunks=policies or [{"source": "return.md", "heading": "Window", "text": "14 days"}],
    )


def test_exact_cache_runs_before_semantic():
    cache = LayeredResponseCache()
    identity = _identity()
    cache.store_identity(identity, "Fourteen days.", 0.95, ["return.md"])

    hit = cache.lookup_identity(identity)

    assert hit.hit is True
    assert hit.level == "exact"
    assert hit.similarity == 1.0


def test_policy_paraphrase_can_use_semantic_cache(monkeypatch):
    cache = LayeredResponseCache()
    original = _identity("What is the return policy?")
    paraphrase = _identity("What is the return policy for an item?")
    cache.store_identity(original, "Fourteen days.", 0.95, ["return.md"])
    monkeypatch.setattr("app.services.cache_service.cosine", lambda *_: 1.0)

    hit = cache.lookup_identity(paraphrase)

    assert hit.hit is True
    assert hit.level == "semantic"


def test_order_context_change_invalidates_exact_cache():
    cache = LayeredResponseCache()
    old = _identity(
        "Where is ORD-10042?", tier="simple", orders=[{"id": "ORD-10042", "status": "shipped"}], policies=[]
    )
    new = _identity(
        "Where is ORD-10042?", tier="simple", orders=[{"id": "ORD-10042", "status": "delivered"}], policies=[]
    )
    cache.store_identity(old, "It shipped.", 0.95, ["order_database"])

    assert cache.lookup_identity(new).hit is False


def test_semantic_cache_disabled_for_order_and_conversation_context():
    order = _identity("Where is ORD-10042?", tier="simple", orders=[{"id": "ORD-10042"}], policies=[])
    follow_up = build_identity(
        query="When will it arrive?",
        tier="medium",
        order_contexts=[],
        policy_chunks=[],
        conversation_context=[{"role": "user", "content": "Where is ORD-10042?"}],
    )

    assert order.semantic_allowed is False
    assert follow_up.semantic_allowed is False
