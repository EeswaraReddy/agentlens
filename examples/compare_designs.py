"""AgentLens — compare agent designs side by side.

Runs the same task through three agent designs, traces every call, runs the
same eval suite against each, and prints a comparison table:
spans, tokens, cost, latency, evals passed.

Uses the Groq client when GROQ_API_KEY is set, otherwise a deterministic mock
so the comparison runs anywhere.

Variants:
  - minimal   : single LLM call + tool
  - rag       : classify -> tool lookup -> compose reply  (2 LLM calls)
  - expensive : same as rag, plus a redundant refine pass (3 LLM calls)

Run:
    python examples/compare_designs.py
    python examples/compare_designs.py --otlp
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentlens import Tracer, Trace, evals as E
from agentlens.providers.groq import GroqClient


TASK = "Where is my order A1029?"
ORDER_ID = "A1029"


def lookup_order(order_id: str) -> str:
    return "Order " + order_id + ": shipped, arriving in 2 days."


class MockClient:
    """Used when GROQ_API_KEY is missing so the comparison still runs."""

    def __init__(self, tracer: Tracer):
        self.tracer = tracer
        self.model = "demo-model"

    def complete(self, prompt: str, *, system: Optional[str] = None,
                 span_name: str = "llm", max_tokens: int = 80, **_):
        p = prompt.lower()
        if "classify" in p:
            text = "order_status"
        elif "refine" in p or "improve" in p:
            text = "Your order has shipped and will arrive in 2 days."
        else:
            text = "Your order has shipped and arrives in 2 days."
        prompt_tokens = max(20, min(220, len(prompt) // 4))
        completion_tokens = max(4, min(60, len(text) // 4))

        with self.tracer.llm(span_name, model=self.model) as span:
            span.set_input(prompt=prompt[:200])
            span.record_tokens(prompt=prompt_tokens, completion=completion_tokens,
                               model=self.model)
            span.set_output(text=text)

        class _R:
            pass
        r = _R()
        r.text = text
        r.prompt_tokens = prompt_tokens
        r.completion_tokens = completion_tokens
        r.total_tokens = prompt_tokens + completion_tokens
        return r


def make_client(tracer: Tracer):
    if os.getenv("GROQ_API_KEY"):
        return GroqClient(tracer=tracer, model="llama-3.1-8b-instant")
    return MockClient(tracer)


def agent_minimal(tracer: Tracer, client) -> str:
    with tracer.trace("variant:minimal", task=TASK):
        with tracer.agent("router"):
            reply = client.complete(
                "Customer asked: '" + TASK + "'. Reply in one short sentence.",
                system="You are a concise support agent.",
                span_name="reply",
                max_tokens=80,
            ).text
            with tracer.tool("lookup_order", order_id=ORDER_ID) as span:
                span.set_output(info=lookup_order(ORDER_ID))
            return reply


def agent_rag(tracer: Tracer, client) -> str:
    with tracer.trace("variant:rag", task=TASK):
        with tracer.agent("router"):
            intent = client.complete(
                "Classify the intent of '" + TASK + "' in one lowercase word.",
                system="Output one lowercase word, nothing else.",
                span_name="classify_intent",
                max_tokens=8,
            ).text.strip()
            with tracer.tool("lookup_order", order_id=ORDER_ID) as span:
                info = lookup_order(ORDER_ID)
                span.set_output(info=info)
            reply = client.complete(
                "Customer asked: '" + TASK + "'. Facts: " + info +
                ". Write a friendly one-sentence reply.",
                system="You are a concise support agent.",
                span_name="compose_reply",
                max_tokens=80,
            ).text
            tracer.event("intent_detected", intent=intent)
            return reply


def agent_expensive(tracer: Tracer, client) -> str:
    with tracer.trace("variant:expensive", task=TASK):
        with tracer.agent("router"):
            intent = client.complete(
                "Classify the intent of '" + TASK + "' in one lowercase word.",
                system="Output one lowercase word, nothing else.",
                span_name="classify_intent",
                max_tokens=8,
            ).text.strip()
            with tracer.tool("lookup_order", order_id=ORDER_ID) as span:
                info = lookup_order(ORDER_ID)
                span.set_output(info=info)
            draft = client.complete(
                "Customer asked: '" + TASK + "'. Facts: " + info + ". Write a reply.",
                system="You are a support agent.",
                span_name="compose_reply",
                max_tokens=80,
            ).text
            reply = client.complete(
                "Refine and improve this reply: " + draft,
                system="Improve clarity but keep the meaning.",
                span_name="refine_reply",
                max_tokens=80,
            ).text
            tracer.event("intent_detected", intent=intent)
            return reply


SUITE = E.Suite("support-agent guardrails", [
    E.succeeded(),
    E.called_tool("lookup_order"),
    E.never_called_tool("issue_refund"),
    E.max_cost(0.01),
    E.max_duration_ms(10_000),
])


@dataclass
class Row:
    name: str
    spans: int
    llm_calls: int
    tokens: int
    cost: float
    duration_ms: float
    evals: str
    reply: str


def run_variant(name: str, fn: Callable, client) -> Row:
    tracer = Tracer()
    client.tracer = tracer
    t0 = time.perf_counter()
    reply = fn(tracer, client)
    elapsed = (time.perf_counter() - t0) * 1000.0
    trace: Trace = tracer.finished[-1]
    report = SUITE.run(trace)
    counts = trace.counts()
    return Row(
        name=name,
        spans=len(trace.spans),
        llm_calls=counts.get("llm", 0),
        tokens=trace.total_tokens,
        cost=trace.total_cost_usd,
        duration_ms=elapsed,
        evals=str(report.num_passed) + "/" + str(len(report.results)),
        reply=reply,
    )


def print_table(rows: List[Row]) -> None:
    fmt = "{:<12}{:>6}{:>5}{:>8}{:>12}{:>9}{:>8}"
    header = fmt.format("variant", "spans", "llm", "tokens", "cost", "ms", "evals")
    print()
    print(header)
    print("-" * len(header))
    for r in rows:
        cost_str = "$" + format(r.cost, ".6f")
        ms_str = format(r.duration_ms, ".1f")
        print(fmt.format(r.name, r.spans, r.llm_calls, r.tokens,
                         cost_str, ms_str, r.evals))
    print()
    nonzero = [r for r in rows if r.cost > 0]
    if len(nonzero) >= 2:
        cheap = min(nonzero, key=lambda r: r.cost)
        pricey = max(nonzero, key=lambda r: r.cost)
        multi = pricey.cost / cheap.cost
        print("  '" + pricey.name + "' costs " + format(multi, ".1f") +
              "x more than '" + cheap.name + "' for the same eval result.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--otlp", action="store_true",
                        help="Also emit every trace as real OTel spans")
    args = parser.parse_args()

    placeholder = Tracer()
    client = make_client(placeholder)
    mode = "Groq (real LLM)" if os.getenv("GROQ_API_KEY") else "mock LLM"
    print("mode :", mode)
    print("task :", TASK)

    rows: List[Row] = []
    all_traces = []
    for name, fn in (("minimal", agent_minimal),
                     ("rag", agent_rag),
                     ("expensive", agent_expensive)):
        row = run_variant(name, fn, client)
        rows.append(row)
        all_traces.append(client.tracer.finished[-1])
        client.tracer.save("runs")

    print_table(rows)

    if args.otlp:
        try:
            from agentlens.export import install_otlp_bridge, emit_to_otlp
            install_otlp_bridge(service_name="agentlens-compare",
                                console=not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"))
            total = 0
            for tr in all_traces:
                total += emit_to_otlp(tr, service_name="agentlens-compare")
            print("emitted", total, "OTel spans")
        except ImportError as exc:
            print("OTLP bridge unavailable:", exc)

    print("inspect any trace: agentlens view")


if __name__ == "__main__":
    main()
