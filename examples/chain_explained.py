"""
How the shopping agent links to AgentLens — step by step trace of execution.

Run this to see exactly what happens internally at each step:
    python examples/chain_explained.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─────────────────────────────────────────────────────────────
# STEP 1: Import — what you get
# ─────────────────────────────────────────────────────────────
from agentlens import Tracer           # agentlens/tracer.py  → class Tracer
from agentlens import evals as E       # agentlens/evals.py   → check functions
from agentlens.trace import Trace, Span  # agentlens/trace.py → dataclasses
from agentlens.pricing import estimate_cost  # agentlens/pricing.py → cost table

print("─── STEP 1: What Tracer looks like before anything runs ───")
tracer = Tracer()
print(f"  tracer._current  = {tracer._current}")   # None — no active trace
print(f"  tracer._stack    = {tracer._stack}")      # [] — empty span stack
print(f"  tracer.finished  = {tracer.finished}")    # [] — no completed traces
print()

# ─────────────────────────────────────────────────────────────
# STEP 2: tracer.trace() — opens a Trace
# ─────────────────────────────────────────────────────────────
print("─── STEP 2: tracer.trace() opens a Trace object ───")

with tracer.trace("shopping-assistant", user="alice") as active_trace:
    print(f"  type(active_trace)       = {type(active_trace)}")
    print(f"  active_trace.trace_id    = {active_trace.trace_id}")
    print(f"  active_trace.name        = {active_trace.name}")
    print(f"  active_trace.spans       = {active_trace.spans}")   # empty yet
    print(f"  tracer._current is trace = {tracer._current is active_trace}")
    print(f"  tracer._stack            = {tracer._stack}")  # still empty
    print()

    # ─────────────────────────────────────────────────────────
    # STEP 3: tracer.agent() — opens an agent Span, pushes its ID to stack
    # ─────────────────────────────────────────────────────────
    print("─── STEP 3: tracer.agent() creates a Span, pushes to stack ───")

    with tracer.agent("shopping-router") as agent_handle:
        agent_span = tracer._current.spans[0]  # first span added
        print(f"  agent_span.kind      = {agent_span.kind}")
        print(f"  agent_span.span_id   = {agent_span.span_id}")
        print(f"  agent_span.parent_id = {agent_span.parent_id}")  # None — root
        print(f"  tracer._stack        = {tracer._stack}")          # [agent_span.id]
        print()

        # ─────────────────────────────────────────────────────
        # STEP 4: tracer.llm() — child span, parent = agent span
        # ─────────────────────────────────────────────────────
        print("─── STEP 4: tracer.llm() creates a child LLM span ───")

        with tracer.llm("classify_intent", model="llama-3.1-8b-instant") as llm_handle:
            llm_span = tracer._current.spans[1]
            print(f"  llm_span.kind          = {llm_span.kind}")
            print(f"  llm_span.span_id       = {llm_span.span_id}")
            print(f"  llm_span.parent_id     = {llm_span.parent_id}")
            print(f"  parent matches agent?  = {llm_span.parent_id == agent_span.span_id}")
            print(f"  tracer._stack          = {tracer._stack}")  # [agent_id, llm_id]
            print()

            # This is what the Groq client does internally when you call .complete()
            print("  → Simulating: span.record_tokens(prompt=82, completion=8)")
            llm_handle.record_tokens(prompt=82, completion=8, model="llama-3.1-8b-instant")

            print(f"  llm_span.prompt_tokens    = {llm_span.prompt_tokens}")
            print(f"  llm_span.completion_tokens= {llm_span.completion_tokens}")
            print(f"  llm_span.cost_usd         = {llm_span.cost_usd}")  # auto-computed!
            print(f"  (pricing: 82/1M * $0.05 + 8/1M * $0.08 = tiny)")
            print()

            llm_handle.set_output(text="buy_headphones")

        # LLM span just closed — stack is back to [agent_id]
        print(f"  after llm exits, tracer._stack = {tracer._stack}")
        print(f"  llm_span.duration_ms = {llm_span.duration_ms:.4f}ms  (measured by perf_counter)")
        print()

        # ─────────────────────────────────────────────────────
        # STEP 5: tracer.tool() — another child span
        # ─────────────────────────────────────────────────────
        print("─── STEP 5: tracer.tool() creates a tool span ───")

        with tracer.tool("search_products", query="wireless headphones") as tool_handle:
            tool_span = tracer._current.spans[2]
            print(f"  tool_span.kind       = {tool_span.kind}")
            print(f"  tool_span.parent_id  = {tool_span.parent_id}")
            print(f"  same parent as llm?  = {tool_span.parent_id == llm_span.parent_id}")
            # Both are siblings under the agent span
            tool_handle.set_output(results=["Sony WH-1000XM5", "Bose QC45"])
            print(f"  tool_span.outputs    = {tool_span.outputs}")
        print()

        # ─────────────────────────────────────────────────────
        # STEP 6: tracer.event() — instantaneous marker, no push
        # ─────────────────────────────────────────────────────
        print("─── STEP 6: tracer.event() — instantaneous marker ───")
        tracer.event("stock_checked", checked=2, available=2)
        event_span = tracer._current.spans[3]
        print(f"  event_span.kind      = {event_span.kind}")
        print(f"  event_span.duration  = {event_span.duration_ms}")  # always 0
        print(f"  event_span.attrs     = {event_span.attributes}")
        print()

    # agent span just closed — stack is empty again
    print(f"─── After agent exits: tracer._stack = {tracer._stack} ───")
    print()

# trace just closed — moved to tracer.finished
print("─── STEP 7: trace exits — moved to tracer.finished ───")
print(f"  tracer._current  = {tracer._current}")       # None again
print(f"  len(tracer.finished) = {len(tracer.finished)}")  # 1
print()

# ─────────────────────────────────────────────────────────────
# STEP 7: The finished Trace — what it looks like
# ─────────────────────────────────────────────────────────────
finished = tracer.finished[0]
print("─── STEP 7: Finished Trace aggregates ───")
print(f"  status           = {finished.status}")
print(f"  total_tokens     = {finished.total_tokens}")   # sum of all llm spans
print(f"  total_cost_usd   = {finished.total_cost_usd:.8f}")  # sum of all costs
print(f"  counts           = {finished.counts()}")       # by kind
print(f"  span tree:")
for sp in finished.spans:
    indent = "      " if sp.parent_id else "  "
    print(f"  {indent}[{sp.kind:5}] {sp.name}  parent={sp.parent_id or 'ROOT'}")
print()

# ─────────────────────────────────────────────────────────────
# STEP 8: Evals — checks run against the finished Trace
# ─────────────────────────────────────────────────────────────
print("─── STEP 8: Evals run against the Trace ───")
print("  Each check is just a function: Trace → CheckResult")
print()

suite = E.Suite("demo checks", [
    E.succeeded(),
    E.called_tool("search_products"),
    E.never_called_tool("issue_refund"),
    E.tool_before("search_products", "stock_checked"),  # will fail — event not a tool
    E.max_cost(0.01),
    E.custom("has llm span", lambda t: any(s.kind == "llm" for s in t.spans)),
])

report = suite.run(finished)
print(report.summary())
print()

# ─────────────────────────────────────────────────────────────
# STEP 9: Save — what goes to disk
# ─────────────────────────────────────────────────────────────
print("─── STEP 9: tracer.save() → JSON ───")
import json
d = finished.to_dict()
# show the structure without the full span details
top_level = {k: v for k, v in d.items() if k != "spans"}
print("  top-level keys:", list(d.keys()))
print("  summary block :", json.dumps(d["summary"], indent=4))
print()
print("  first span as JSON:")
print(json.dumps(d["spans"][0], indent=4))
