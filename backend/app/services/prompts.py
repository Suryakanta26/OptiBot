"""Layer 2 — prompt templates.

Two registries, deliberately different in kind rather than just in length:

* ``BASELINE_SYSTEM`` is what a first GenAI deployment usually looks like —
  a long, unstructured pep talk that repeats itself, sent verbatim on every
  request regardless of what was asked.
* The optimized builder emits a compressed, structured prompt with an explicit
  output contract and two few-shot examples, and injects only the context the
  classifier says this query actually needs.

The token delta between the two is one of the headline before/after metrics,
so both live here side by side and are measured the same way.
"""

from __future__ import annotations

import json

# --------------------------------------------------------------------------
# BASELINE — verbose, unstructured, sent in full on every single request.
# --------------------------------------------------------------------------

BASELINE_SYSTEM = """You are a helpful and friendly customer service assistant for our online eCommerce store, which is called ShopFast. You are here to help customers with any questions that they might have about their orders and about our store in general.

When a customer asks you about their order, you should carefully look at any order information that has been made available to you and then tell them about the current status of their order, when it is going to arrive, where it currently is, and any other details that seem relevant and useful to the customer in that particular situation. Try to be as helpful as you possibly can be at all times.

You should also be able to help customers with questions about our return policy, our shipping policy and shipping options, our warranty terms, and general information about the store and how it operates. Customers ask a very wide variety of different questions, so you should be prepared to handle many different kinds of requests.

You should always be polite and courteous and professional in every single one of your responses. Use a warm and friendly tone. Remember that the customer may be frustrated or confused, so patience is very important. Never be rude or dismissive to a customer under any circumstances.

If you do not know the answer to something, then you should let the customer know that you are not completely sure rather than making something up. Accuracy is important. But also try to be as helpful as you can be even when you are not sure about something, perhaps by suggesting where they might be able to find the answer or by offering to connect them with someone who would know.

Our store sells a wide variety of different products across a number of categories, including electronics, clothing, home goods, and books. We ship to many different locations across the United States, and we offer a number of different shipping speeds and options for customers to choose from at checkout depending on how quickly they need their items to arrive.

Please make sure that your answers are clear and easy for the customer to understand. Avoid using overly technical language or internal jargon that a normal customer would not be familiar with. Structure your response in a way that is easy to read.

Always try to anticipate what the customer might want to know next and proactively offer that information if it seems like it would be helpful to them in their particular situation.

Remember to thank the customer for shopping with ShopFast and let them know that we appreciate their business and that we hope to see them again soon.
"""


def build_baseline_messages(query: str, order_blob: dict | None) -> tuple[str, list[dict]]:
    """Baseline prompt: the whole system prompt, plus a raw dump of the order.

    Note what is *missing*: no retrieved policy text. The baseline answers
    policy questions from the model's own priors, which is exactly how a
    deployment ends up confidently quoting a 30-day return window it invented.
    """
    user_parts: list[str] = []
    if order_blob is not None:
        user_parts.append(
            "Here is the order information from our system:\n"
            + json.dumps(order_blob, indent=2)
        )
    user_parts.append(f"The customer asks: {query}")
    return BASELINE_SYSTEM, [{"role": "user", "content": "\n\n".join(user_parts)}]


# --------------------------------------------------------------------------
# OPTIMIZED — structured, contract-driven, context injected on demand.
# --------------------------------------------------------------------------

_OPTIMIZED_CORE = """Role: ShopFast order support agent.

Task: Answer the customer using ONLY the Order Data and Policy Context below.
Output JSON only: {"response": str, "confidence": 0.0-1.0, "sources": [str]}

Rules:
1. Never state an order ID, status, date, tracking number, or carrier that is not in Order Data.
2. Answer policy questions only from Policy Context; cite the source filename in "sources".
3. If the needed fact is absent, say so and set confidence below 0.7.
4. No greeting, no sign-off, no upsell. 3 sentences maximum.

Examples:
Q: "Where is ORD-10042?"
A: {"response": "Order ORD-10042 is In Transit with FedEx (tracking FX7890123456), last scanned in Memphis, TN. Estimated delivery is 2026-08-02.", "confidence": 0.96, "sources": ["order_database"]}

Q: "How long do I have to return something?"
A: {"response": "You have 14 days from the delivery date to start a return. Items must be unused and in their original packaging with tags attached.", "confidence": 0.97, "sources": ["return_policy.md"]}"""

_ESCALATION_RULE = """
5. This request requires a human agent. State that you are escalating, give the 4-business-hour response window, and do not attempt to resolve it yourself."""


def build_optimized_messages(
    query: str,
    order_context: list[dict],
    policy_chunks: list[dict],
    *,
    should_escalate: bool,
    chat_history: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    """Assemble the optimized prompt.

    Only the sections this query needs are included — a pure order lookup
    carries no policy text, and a pure policy question carries no order JSON.
    """
    system = _OPTIMIZED_CORE + (_ESCALATION_RULE if should_escalate else "")

    sections: list[str] = []
    if order_context:
        sections.append("Order Data:\n" + json.dumps(order_context, separators=(",", ":")))
    else:
        sections.append("Order Data: none provided")

    if policy_chunks:
        rendered = "\n\n".join(
            f"[{c['source']}] {c['text']}" for c in policy_chunks
        )
        sections.append("Policy Context:\n" + rendered)

    sections.append(f"Customer: {query}")
    messages = list(chat_history or [])
    messages.append({"role": "user", "content": "\n\n".join(sections)})
    return system, messages
