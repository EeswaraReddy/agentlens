"""
A tiny, fully mocked "support agent" instrumented with AgentLens.

No cloud, no API keys, no real LLM — it simulates token usage and tool calls so
you can see tracing, cost accounting, and evals end to end:

    python examples/support_agent.py
"""

import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentlens import Tracer, evals as E


# ---- fake "tools" the agent can call -----------------------------------
def lookup_order(order_id: str) -> dict:
    return {"order_id": order_id, "status": "shipped", "eta": "2 days"}


def knowledge_base(query: str) -> str:
    return "Returns are accepted within 30 days of delivery."


def send_reply(text: str) -> str:
    return "sent"


def run_support_agent(tracer: Tracer, user_message: str, order_id: str):
    """Simulate one agent turn, fully traced."""
    with tracer.trace("support-agent", user_message=user_message):
        with tracer.agent("support-router"):

            # 1. classify intent (LLM)
            with tracer.llm("classify_intent", model="gpt-4o-mini") as span:
                span.set_input(message=user_message)
                span.record_tokens(prompt=120, completion=12, model="gpt-4o-mini")
                intent = "order_status"
                span.set_output(intent=intent)

            # 2. look up the order (tool)
            with tracer.tool("lookup_order", order_id=order_id) as span:
                result = lookup_order(order_id)
                span.set_output(**result)

            # 3. consult policy KB (tool)
            with tracer.tool("knowledge_base", query="returns") as span:
                kb = knowledge_base("returns")
                span.set_output(answer=kb)

            # 4. compose the answer (LLM)
            with tracer.llm("compose_reply", model="gpt-4o") as span:
                span.set_input(intent=intent, order=result)
                span.record_tokens(prompt=300, completion=80, model="gpt-4o")
                reply = (f"Your order {order_id} is {result['status']} "
                         f"and should arrive in {result['eta']}.")
                span.set_output(reply=reply)

            # 5. send it (tool)
            with tracer.tool("send_reply") as span:
                span.set_output(status=send_reply(reply))


def build_suite() -> E.Suite:
    """The behavior we require of the support agent."""
    return E.Suite("support-agent guardrails", [
        E.succeeded(),
        E.called_tool("lookup_order"),
        E.called_tool("send_reply"),
        E.never_called_tool("issue_refund"),     # agent must not refund
        E.tool_before("lookup_order", "send_reply"),
        E.max_cost(0.01),
        E.max_duration_ms(5000),
        E.custom("reply references the order",
                 lambda t: any("order" in str(s.outputs).lower() for s in t.spans)),
    ])


def main():
    tracer = Tracer()
    run_support_agent(tracer, "Where is my order A1029?", order_id="A1029")

    trace = tracer.finished[-1]
    print("=" * 60)
    print("TRACE SUMMARY")
    print("=" * 60)
    s = trace.to_dict()["summary"]
    print(f"  trace_id   : {trace.trace_id}")
    print(f"  status     : {trace.status}")
    print(f"  duration   : {trace.duration_ms:.1f} ms")
    print(f"  spans      : {s['spans']}  {s['counts']}")
    print(f"  tokens     : {trace.total_tokens}")
    print(f"  cost       : ${trace.total_cost_usd:.6f}")

    print("\n" + "=" * 60)
    report = build_suite().run(trace)
    print(report.summary())

    paths = tracer.save("runs")
    print(f"\nSaved trace to: {paths[-1]}")
    print("View it: agentlens view   (or  python -m agentlens.cli view)")

    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
