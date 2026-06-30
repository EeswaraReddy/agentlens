"""
Real-LLM example: a Groq-powered agent, fully traced, scored by an LLM judge.

Groq's OpenAI-compatible API is fast and has a free tier, which makes it great
for both real-time agent calls and LLM-as-judge evals.

Setup:
    pip install -e ".[viewer]"      # viewer optional
    export GROQ_API_KEY=...          # free key at https://console.groq.com

Run:
    python examples/groq_agent.py

Without a key, it prints instructions instead of failing.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentlens import Tracer, evals as E
from agentlens.providers.groq import GroqClient, GroqError


# A trivial "tool" the agent can use.
def lookup_order(order_id: str) -> str:
    return f"Order {order_id}: shipped, arriving in 2 days."


def run_agent(tracer: Tracer, groq: GroqClient, user_msg: str, order_id: str) -> str:
    groq.tracer = tracer  # so each Groq call is captured as an llm span
    with tracer.trace("groq-support-agent", user_message=user_msg):
        with tracer.agent("support-router"):
            # Step 1 (real LLM): decide intent
            intent = groq.complete(
                f"Classify the intent of this message in one word: '{user_msg}'",
                system="You output a single lowercase word, nothing else.",
                span_name="classify_intent",
                max_tokens=8,
            ).text.strip()

            # Step 2 (tool): look up the order
            with tracer.tool("lookup_order", order_id=order_id) as span:
                order_info = lookup_order(order_id)
                span.set_output(info=order_info)

            # Step 3 (real LLM): compose the customer reply
            reply = groq.complete(
                f"Customer asked: '{user_msg}'. Known facts: {order_info}. "
                "Write a friendly one-sentence reply.",
                system="You are a concise, helpful support agent.",
                span_name="compose_reply",
                max_tokens=80,
            ).text.strip()

            tracer.event("intent_detected", intent=intent)
            return reply


def main():
    groq = GroqClient(model="llama-3.1-8b-instant")
    if not groq.api_key:
        print("No GROQ_API_KEY set.\n")
        print("1) Get a free key: https://console.groq.com")
        print("2) export GROQ_API_KEY=...   (PowerShell: $env:GROQ_API_KEY='...')")
        print("3) python examples/groq_agent.py")
        return

    tracer = Tracer()
    try:
        reply = run_agent(tracer, groq, "Where is my order A1029?", "A1029")
    except GroqError as exc:
        print(f"Groq call failed: {exc}")
        return

    trace = tracer.finished[-1]
    print(f"Agent reply: {reply}\n")
    s = trace.to_dict()["summary"]
    print(f"trace: {trace.trace_id} · {s['spans']} spans · "
          f"{trace.total_tokens} tokens · ${trace.total_cost_usd:.6f}\n")

    # Mix deterministic checks with an LLM-as-judge check (also via Groq).
    suite = E.Suite("groq-agent evals", [
        E.succeeded(),
        E.called_tool("lookup_order"),
        E.max_cost(0.01),
        E.llm_judge(
            "The reply directly addresses the customer's order-status question, "
            "is friendly, and does not invent facts beyond the known order info.",
            model="llama-3.3-70b-versatile",
            threshold=0.7,
        ),
    ])
    print(suite.run(trace).summary())
    tracer.save("runs")
    print("\nSaved. Inspect with: agentlens view")


if __name__ == "__main__":
    main()
