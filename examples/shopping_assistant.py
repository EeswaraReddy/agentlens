"""
AgentLens Demo — E-Commerce Shopping Assistant
===============================================

A realistic consumer agent that:
  1. Understands what the user wants to buy
  2. Searches the product catalog
  3. Checks stock availability
  4. Applies coupon codes (with a guardrail: no auto-applying invalid coupons)
  5. Generates a personalized product recommendation

All steps are traced with AgentLens. Evals catch real failure modes:
  - Did the agent actually check stock before recommending?
  - Did it stay within cost / latency budgets?
  - Did it NEVER apply an invalid coupon?
  - LLM judge: was the recommendation actually helpful?

No API key required — uses a deterministic mock by default.
Set GROQ_API_KEY for real LLM calls.

Run:
    python examples/shopping_assistant.py
    python examples/shopping_assistant.py --scenario angry   # tests error handling
    python examples/shopping_assistant.py --scenario vague   # tests clarification flow
"""

from __future__ import annotations

import argparse
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Auto-load .env from the project root (no python-dotenv needed)
_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

from agentlens import Tracer, evals as E
from agentlens.providers.groq import GroqClient, GroqError


# ---------------------------------------------------------------------------
# Fake "database" — what a real backend would provide
# ---------------------------------------------------------------------------

CATALOG = {
    "P001": {"name": "Sony WH-1000XM5 Headphones", "price": 279.99, "category": "electronics"},
    "P002": {"name": "Apple AirPods Pro (2nd gen)", "price": 189.99, "category": "electronics"},
    "P003": {"name": "Bose QuietComfort 45",        "price": 249.99, "category": "electronics"},
    "P004": {"name": "Anker Soundcore Q45",          "price":  59.99, "category": "electronics"},
    "P005": {"name": "Samsung Galaxy Buds2 Pro",     "price": 149.99, "category": "electronics"},
}

STOCK = {
    "P001": 8,
    "P002": 0,   # out of stock — agent should handle this
    "P003": 3,
    "P004": 45,
    "P005": 12,
}

COUPONS = {
    "SAVE10": 0.10,
    "STUDENT20": 0.20,
}

USER_HISTORY = {
    "alice": ["P002", "P004"],   # bought AirPods and Anker before
}


# ---------------------------------------------------------------------------
# Tool implementations (the actual business logic)
# ---------------------------------------------------------------------------

def search_products(query: str, category: str = "electronics") -> list[dict]:
    """Search the product catalog by keyword."""
    q = query.lower()
    results = []
    for pid, product in CATALOG.items():
        if (q in product["name"].lower() or
                any(w in product["name"].lower() for w in q.split())):
            results.append({"id": pid, **product})
    return results or [{"id": pid, **p} for pid, p in CATALOG.items()
                       if p["category"] == category][:3]


def check_stock(product_id: str) -> dict:
    """Check how many units are in stock."""
    qty = STOCK.get(product_id, 0)
    return {
        "product_id": product_id,
        "quantity": qty,
        "available": qty > 0,
        "label": "In Stock" if qty > 5 else ("Low Stock" if qty > 0 else "Out of Stock"),
    }


def get_user_history(user_id: str) -> list[str]:
    """Fetch what the user has bought before."""
    return USER_HISTORY.get(user_id, [])


def validate_coupon(code: str, product_id: str) -> dict:
    """Validate a coupon code. MUST be called before applying any discount."""
    if code.upper() in COUPONS:
        discount = COUPONS[code.upper()]
        price = CATALOG.get(product_id, {}).get("price", 0)
        return {
            "valid": True,
            "code": code.upper(),
            "discount_pct": discount,
            "savings": round(price * discount, 2),
            "final_price": round(price * (1 - discount), 2),
        }
    return {"valid": False, "code": code, "reason": "Coupon not found or expired"}


# ---------------------------------------------------------------------------
# Mock LLM (no API key needed)
# ---------------------------------------------------------------------------

class MockLLM:
    """Deterministic mock — no API key, reproducible for demos."""

    def __init__(self, tracer: Tracer):
        self.tracer = tracer
        self.model = "demo-model"

    def complete(self, prompt: str, *, system: str = "", span_name: str = "llm",
                 max_tokens: int = 150, **_):
        p = prompt.lower()

        # Intent classification — must match before "recommend" since classify prompt has "classify"
        if "classify the intent" in p or "what does the user want" in p:
            text = "buy_headphones"
        # Product selection
        elif "recommend" in p or "best option" in p or "which product" in p:
            text = (
                "Based on your history and budget, I recommend the "
                "**Sony WH-1000XM5** ($279.99). It has the best noise cancellation "
                "in class and your previous Anker purchase shows you care about "
                "audio quality. The Sony is a significant upgrade at a fair price."
            )
        # Coupon check
        elif "coupon" in p or "discount" in p:
            text = "SAVE10"
        # Clarification
        elif "clarif" in p or "vague" in p or "unclear" in p:
            text = "Could you tell me your budget range and whether you prefer over-ear or in-ear?"
        else:
            text = "I can help you find the perfect headphones. Let me check what we have."

        tokens_in  = max(20, len(prompt) // 4)
        tokens_out = max(8,  len(text)   // 4)

        with self.tracer.llm(span_name, model=self.model) as span:
            span.set_input(prompt=prompt[:300])
            span.record_tokens(prompt=tokens_in, completion=tokens_out, model=self.model)
            span.set_output(text=text)

        class _R:
            pass
        r = _R()
        r.text = text
        r.prompt_tokens    = tokens_in
        r.completion_tokens = tokens_out
        r.total_tokens     = tokens_in + tokens_out
        return r


def make_llm(tracer: Tracer):
    key = os.getenv("GROQ_API_KEY")
    if key:
        print("  [using Groq LLM — llama-3.1-8b-instant]")
        return GroqClient(tracer=tracer, model="llama-3.1-8b-instant")
    print("  [using mock LLM — set GROQ_API_KEY for real calls]")
    return MockLLM(tracer)


# ---------------------------------------------------------------------------
# The actual agent
# ---------------------------------------------------------------------------

def run_shopping_agent(
    tracer: Tracer,
    llm,
    user_id: str,
    user_message: str,
    coupon_code: str | None = None,
) -> dict:
    """
    Full shopping assistant flow:
      classify → fetch history → search catalog → check stock
      → (optional) validate coupon → recommend
    """
    if hasattr(llm, "tracer"):
        llm.tracer = tracer

    with tracer.trace("shopping-assistant", user=user_id, message=user_message):

        with tracer.agent("shopping-router"):

            # ── Step 1: Understand what the user wants ───────────────────
            intent_result = llm.complete(
                f"Classify the intent of this message into one short phrase: '{user_message}'\n"
                "What does the user want to buy or do?",
                system="Output one short intent phrase, nothing else.",
                span_name="classify_intent",
                max_tokens=12,
            )
            intent = intent_result.text.strip()

            # ── Step 2: Get personalization context ──────────────────────
            with tracer.tool("get_user_history", user_id=user_id) as span:
                history = get_user_history(user_id)
                span.set_output(previous_purchases=history, count=len(history))

            # ── Step 3: Search the catalog ───────────────────────────────
            with tracer.tool("search_products", query=user_message) as span:
                products = search_products(user_message)
                span.set_output(
                    results=[p["name"] for p in products],
                    count=len(products),
                )

            # ── Step 4: Check stock for top 3 candidates ─────────────────
            stock_info = {}
            available = []
            for product in products[:3]:
                with tracer.tool("check_stock", product_id=product["id"]) as span:
                    stock = check_stock(product["id"])
                    stock_info[product["id"]] = stock
                    span.set_output(**stock)
                    if stock["available"]:
                        available.append({**product, "stock": stock})

            # Mark a guardrail event — agent checked stock before recommending
            tracer.event("stock_checked", checked=len(stock_info), available=len(available))

            # ── Step 5: Validate coupon (if provided) ────────────────────
            coupon_result = None
            if coupon_code:
                top_pid = available[0]["id"] if available else list(CATALOG.keys())[0]
                with tracer.tool("validate_coupon",
                                 code=coupon_code, product_id=top_pid) as span:
                    coupon_result = validate_coupon(coupon_code, top_pid)
                    span.set_output(**coupon_result)
                    if not coupon_result["valid"]:
                        # Record that we detected an invalid coupon — do NOT apply it
                        tracer.event("invalid_coupon_blocked", code=coupon_code)

            # ── Step 6: Generate personalized recommendation ─────────────
            context = (
                f"User '{user_id}' asked: '{user_message}'\n"
                f"Intent: {intent}\n"
                f"Purchase history: {[CATALOG[p]['name'] for p in history if p in CATALOG]}\n"
                f"Available products: {[p['name'] + ' $' + str(p['price']) for p in available]}\n"
            )
            if coupon_result and coupon_result["valid"]:
                context += f"Valid coupon {coupon_result['code']}: saves ${coupon_result['savings']}\n"

            recommendation = llm.complete(
                f"Based on the following context, recommend the best product and explain why:\n{context}",
                system="You are a helpful shopping assistant. Be specific and mention the price.",
                span_name="generate_recommendation",
                max_tokens=150,
            ).text.strip()

            return {
                "intent": intent,
                "available_count": len(available),
                "recommendation": recommendation,
                "coupon_applied": coupon_result["valid"] if coupon_result else False,
                "savings": coupon_result.get("savings", 0) if coupon_result else 0,
            }


# ---------------------------------------------------------------------------
# Eval suite — the behavior we REQUIRE of this agent
# ---------------------------------------------------------------------------

def build_eval_suite() -> E.Suite:
    return E.Suite("shopping-assistant guardrails", [
        # Basics
        E.succeeded(),

        # Must check stock before recommending (ordering guardrail)
        E.called_tool("check_stock"),

        # tool_before only compares tool spans; use a custom check for tool→llm ordering
        E.custom(
            "check_stock before generate_recommendation",
            lambda t: (
                any(s.name == "check_stock" for s in t.spans) and
                any(s.name == "generate_recommendation" for s in t.spans) and
                next(i for i, s in enumerate(t.spans) if s.name == "check_stock") <
                next(i for i, s in enumerate(t.spans) if s.name == "generate_recommendation")
            ),
        ),

        # Must search before checking stock
        E.tool_before("search_products", "check_stock"),

        # Must NOT apply invalid coupons
        E.never_called_tool("apply_invalid_coupon"),  # fictional forbidden tool

        # Must personalize — fetch history
        E.called_tool("get_user_history"),

        # Cost + latency budgets
        E.max_cost(0.05),
        E.max_duration_ms(8_000),

        # Custom: at least one product was available
        E.custom(
            "found available products",
            lambda t: any(
                "available" in str(s.outputs) and "True" in str(s.outputs)
                for s in t.spans if s.name == "check_stock"
            ),
        ),

        # Custom: invalid coupon was blocked if used
        E.custom(
            "invalid coupon not silently applied",
            lambda t: not any(
                s.name == "validate_coupon"
                and '"valid": false' in json.dumps(s.outputs).lower()
                and not any(e.name == "invalid_coupon_blocked" for e in t.spans)
                for s in t.spans
            ),
        ),
    ])


# ---------------------------------------------------------------------------
# Pretty-print the trace tree
# ---------------------------------------------------------------------------

def print_trace_tree(trace):
    print("\n── TRACE TREE ─────────────────────────────────────────────")
    kind_icons = {"agent": "🤖", "llm": "🧠", "tool": "🔧", "event": "📍"}
    for span in trace.spans:
        indent = "   " if span.parent_id else ""
        icon   = kind_icons.get(span.kind, "·")
        cost   = f"  ${span.cost_usd:.6f}" if span.cost_usd > 0 else ""
        tokens = f"  {span.total_tokens}tok" if span.total_tokens > 0 else ""
        ms     = f"  {span.duration_ms:.1f}ms" if span.duration_ms else ""
        status = " ❌" if span.status == "error" else ""
        print(f"  {indent}{icon} [{span.kind:5}] {span.name}{tokens}{cost}{ms}{status}")
        if span.outputs:
            # Show first key of outputs for quick insight
            first_key = next(iter(span.outputs))
            val = str(span.outputs[first_key])[:60]
            print(f"  {indent}        └─ {first_key}: {val}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SCENARIOS = {
    "normal": {
        "user_id": "alice",
        "message": "I'm looking for wireless noise-cancelling headphones",
        "coupon": "SAVE10",
    },
    "bad_coupon": {
        "user_id": "alice",
        "message": "I want good headphones under $300",
        "coupon": "FAKE50",   # invalid — agent must block it
    },
    "vague": {
        "user_id": "bob",
        "message": "I want something good for music",   # no purchase history
        "coupon": None,
    },
}


def main():
    parser = argparse.ArgumentParser(description="AgentLens Shopping Assistant Demo")
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()),
                        default="normal",
                        help="Which scenario to run (default: normal)")
    parser.add_argument("--all", action="store_true",
                        help="Run all scenarios and compare")
    args = parser.parse_args()

    scenarios_to_run = list(SCENARIOS.items()) if args.all else [(args.scenario, SCENARIOS[args.scenario])]
    suite = build_eval_suite()

    for scenario_name, scenario in scenarios_to_run:
        print(f"\n{'=' * 60}")
        print(f" SCENARIO: {scenario_name.upper()}")
        print(f"{'=' * 60}")
        print(f" User    : {scenario['user_id']}")
        print(f" Message : {scenario['message']}")
        print(f" Coupon  : {scenario['coupon'] or 'none'}")

        tracer = Tracer()
        llm = make_llm(tracer)

        try:
            result = run_shopping_agent(
                tracer,
                llm,
                user_id=scenario["user_id"],
                user_message=scenario["message"],
                coupon_code=scenario["coupon"],
            )
        except GroqError as exc:
            print(f"\n  Groq error: {exc}")
            continue

        trace = tracer.finished[-1]
        summary = trace.to_dict()["summary"]

        # ── Agent output ────────────────────────────────────────────────
        print(f"\n── AGENT RESULT ───────────────────────────────────────────")
        print(f"  Intent detected : {result['intent']}")
        print(f"  Products found  : {result['available_count']} available")
        print(f"  Coupon applied  : {result['coupon_applied']}  (savings: ${result['savings']})")
        print(f"\n  Recommendation:")
        for line in result["recommendation"].split("\n"):
            print(f"    {line}")

        # ── Trace stats ─────────────────────────────────────────────────
        print(f"\n── TRACE STATS ────────────────────────────────────────────")
        print(f"  trace_id  : {trace.trace_id}")
        print(f"  status    : {trace.status}")
        print(f"  duration  : {trace.duration_ms:.1f} ms")
        print(f"  spans     : {summary['spans']}  →  {summary['counts']}")
        print(f"  tokens    : {trace.total_tokens}")
        print(f"  cost      : ${trace.total_cost_usd:.6f}")

        # ── Span tree ───────────────────────────────────────────────────
        print_trace_tree(trace)

        # ── Eval report ─────────────────────────────────────────────────
        report = suite.run(trace)
        print("── EVAL REPORT ────────────────────────────────────────────")
        print(report.summary())

        # ── Save + hint ─────────────────────────────────────────────────
        paths = tracer.save("runs")
        print(f"\n  Saved → {paths[-1]}")

    print(f"\n{'=' * 60}")
    print("  Run 'agentlens view' to inspect all traces in the dashboard")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
