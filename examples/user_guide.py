"""
AgentLens — Practical User Guide
=================================
Run this to see every available feature with real output.
No API key needed for most sections.

    python examples/user_guide.py

Sections:
  1. Basic tracing        — instrument any agent in 5 lines
  2. Cost tracking        — know what every run costs
  3. Eval gates           — assert behavior, use in CI
  4. LLM-as-judge         — score quality with a real model  (needs GROQ_API_KEY)
  5. Design comparison    — find the cheapest agent that still passes
  6. Custom pricing       — add your own model costs
  7. Error handling       — what happens when a span fails
  8. What's NOT available — honest gaps
"""

import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# load .env
_env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(_env):
    for line in open(_env):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from agentlens import Tracer, evals as E
from agentlens.pricing import set_price, estimate_cost


def section(title):
    print(f"\n{'━'*60}")
    print(f"  {title}")
    print(f"{'━'*60}")


# ══════════════════════════════════════════════════════════════
# 1. BASIC TRACING — instrument any agent in 5 lines
# ══════════════════════════════════════════════════════════════
section("1. BASIC TRACING")
print("""
What it does: records every LLM call and tool call as a span,
with timing, inputs, outputs, and parent/child structure.

When to use: any time you want to know "what did my agent actually do?"
""")

tracer = Tracer()

with tracer.trace("my-agent", user="alice", session="demo"):
    with tracer.agent("router"):                          # groups related steps
        with tracer.llm("classify", model="gpt-4o-mini") as s:
            s.set_input(message="Where is my order?")
            s.record_tokens(prompt=120, completion=12, model="gpt-4o-mini")
            s.set_output(intent="order_status")

        with tracer.tool("lookup_order", order_id="A1029") as s:
            s.set_output(status="shipped", eta="2 days")

        tracer.event("intent_confirmed", intent="order_status")  # marker

trace = tracer.finished[-1]
print(f"  ✓ trace_id  : {trace.trace_id}")
print(f"  ✓ spans     : {trace.counts()}")
print(f"  ✓ duration  : {trace.duration_ms:.1f}ms")
for sp in trace.spans:
    pad = "     " if sp.parent_id else "  "
    print(f"  {pad}[{sp.kind:5}] {sp.name}  → out={sp.outputs}")


# ══════════════════════════════════════════════════════════════
# 2. COST TRACKING — know what every run costs
# ══════════════════════════════════════════════════════════════
section("2. COST TRACKING")
print("""
What it does: auto-computes USD cost from token counts using a
built-in price table. Rolls up to the trace level.

When to use: catch runaway costs before they hit your bill.
Budget alerts via eval checks (see section 3).
""")

tracer2 = Tracer()

# Simulate a multi-model agent to compare costs
with tracer2.trace("cost-demo"):
    with tracer2.agent("pipeline"):
        with tracer2.llm("cheap-classify", model="gpt-4o-mini") as s:
            s.record_tokens(prompt=200, completion=15, model="gpt-4o-mini")

        with tracer2.llm("expensive-generate", model="gpt-4o") as s:
            s.record_tokens(prompt=800, completion=300, model="gpt-4o")

        with tracer2.llm("claude-refine", model="claude-3-5-sonnet") as s:
            s.record_tokens(prompt=1200, completion=400, model="claude-3-5-sonnet")

t2 = tracer2.finished[-1]
print(f"  Per-span cost breakdown:")
for sp in t2.spans:
    if sp.kind == "llm":
        print(f"    {sp.name:<25} {sp.total_tokens:>5} tokens  →  ${sp.cost_usd:.6f}")
print(f"\n  {'Total':<25} {t2.total_tokens:>5} tokens  →  ${t2.total_cost_usd:.6f}")
llm_spans = [s for s in t2.spans if s.kind == "llm"]
base = llm_spans[0].cost_usd or 0.000001
print(f"\n  ⚡ gpt-4o (generate) is {llm_spans[1].cost_usd / base:.0f}x more expensive than classify")
print(f"  ⚡ claude (refine)   is {llm_spans[2].cost_usd / base:.0f}x more expensive than classify")

print("""
  Built-in models with pricing:
    gpt-4o, gpt-4o-mini, claude-3-5-sonnet, claude-3-5-haiku,
    claude-opus-4, llama-3.1-8b-instant, llama-3.3-70b-versatile

  Add your own model:  (see section 6)
    from agentlens.pricing import set_price
    set_price("my-model", prompt_per_1m=1.00, completion_per_1m=3.00)
""")


# ══════════════════════════════════════════════════════════════
# 3. EVAL GATES — assert behavior, use in CI
# ══════════════════════════════════════════════════════════════
section("3. EVAL GATES")
print("""
What it does: run declarative checks against a trace.
assert report.passed  →  fails your CI build if agent misbehaves.

When to use:
  - After every prompt change (regression gate)
  - In pytest to catch drift before deploy
  - Cost/latency budget enforcement
""")

tracer3 = Tracer()
with tracer3.trace("order-agent"):
    with tracer3.agent("support"):
        with tracer3.llm("classify", model="gpt-4o-mini") as s:
            s.record_tokens(prompt=100, completion=10, model="gpt-4o-mini")
        with tracer3.tool("lookup_order", order_id="A1") as s:
            s.set_output(status="shipped")
        with tracer3.tool("send_reply") as s:
            s.set_output(status="sent")

t3 = tracer3.finished[-1]

suite = E.Suite("order-agent guardrails", [
    E.succeeded(),                              # run completed without exception
    E.called_tool("lookup_order"),              # must check order
    E.called_tool("send_reply"),                # must reply to customer
    E.never_called_tool("issue_refund"),        # must NOT refund without approval
    E.tool_before("lookup_order", "send_reply"),# must look up BEFORE replying
    E.max_cost(0.01),                           # stay under 1 cent
    E.max_duration_ms(5000),                    # stay under 5 seconds
    E.max_tokens(500),                          # token budget
    E.custom("reply was sent",
             lambda t: any(s.name == "send_reply" and
                          s.outputs.get("status") == "sent"
                          for s in t.spans)),
])

report = suite.run(t3)
print(report.summary())
print()
print("  Use in pytest:")
print("    def test_agent_behavior():")
print("        report = suite.run(trace)")
print("        assert report.passed, report.summary()")


# ══════════════════════════════════════════════════════════════
# 4. LLM-AS-JUDGE — score quality with a real model
# ══════════════════════════════════════════════════════════════
section("4. LLM-AS-JUDGE (needs GROQ_API_KEY)")

has_key = bool(os.getenv("GROQ_API_KEY"))
if has_key:
    print("  GROQ_API_KEY found — running real judge eval...\n")
    tracer4 = Tracer()
    with tracer4.trace("support-reply"):
        with tracer4.agent("support"):
            with tracer4.llm("reply", model="llama-3.1-8b-instant") as s:
                s.record_tokens(prompt=150, completion=60, model="llama-3.1-8b-instant")
                s.set_output(reply="Your order A1029 has shipped and arrives in 2 days. "
                                   "Feel free to reach out if you need anything else!")
            with tracer4.tool("lookup_order") as s:
                s.set_output(status="shipped", eta="2 days")

    t4 = tracer4.finished[-1]
    judge_suite = E.Suite("quality", [
        E.succeeded(),
        E.llm_judge(
            "The reply directly addresses the order status question, "
            "is friendly, and does not invent facts beyond the known order info.",
            model="llama-3.3-70b-versatile",
            threshold=0.7,
        ),
    ])
    print(judge_suite.run(t4).summary())
else:
    print("""
  No GROQ_API_KEY — showing what the judge eval looks like:

    E.llm_judge(
        "The reply answers the question and is friendly.",
        model="llama-3.3-70b-versatile",   # the judge model
        threshold=0.7,                      # pass if score >= 0.7
    )

  The judge receives the full trace transcript and returns:
    {"score": 0.85, "reason": "Reply is accurate and friendly"}

  Free Groq key at: https://console.groq.com
""")


# ══════════════════════════════════════════════════════════════
# 5. DESIGN COMPARISON — find cheapest agent that still passes
# ══════════════════════════════════════════════════════════════
section("5. DESIGN COMPARISON")
print("""
What it does: run the same task through multiple agent designs,
score all with the same eval suite, compare cost/latency/quality.

When to use: before committing to an architecture — find the
cheapest design that passes all your evals.
""")

same_suite = E.Suite("guardrails", [
    E.succeeded(),
    E.called_tool("lookup_order"),
    E.max_cost(0.01),
])

results = []
for design, tokens_in, tokens_out, model in [
    ("1-call  (cheap)",  120,  40, "gpt-4o-mini"),
    ("2-call  (mid)",    300, 120, "gpt-4o-mini"),
    ("3-call  (pricey)", 800, 300, "gpt-4o"),
]:
    t = Tracer()
    with t.trace(f"design:{design}"):
        with t.agent("router"):
            with t.llm("reply", model=model) as s:
                s.record_tokens(prompt=tokens_in, completion=tokens_out, model=model)
            with t.tool("lookup_order") as s:
                s.set_output(status="shipped")
    tr = t.finished[-1]
    r  = same_suite.run(tr)
    results.append((design, tr.total_cost_usd, tr.total_tokens, r.num_passed, len(r.results)))

print(f"  {'Design':<20} {'Cost':>12}  {'Tokens':>7}  {'Evals':>6}")
print(f"  {'-'*50}")
for name, cost, tokens, passed, total in results:
    print(f"  {name:<20} ${cost:.6f}  {tokens:>7}  {passed}/{total}")

cheapest = min(results, key=lambda x: x[1])
print(f"\n  ✓ '{cheapest[0]}' is the cheapest design that still passes all evals")
print(f"    Run: python examples/compare_designs.py  (full 3-variant comparison)")


# ══════════════════════════════════════════════════════════════
# 6. CUSTOM PRICING
# ══════════════════════════════════════════════════════════════
section("6. CUSTOM PRICING")
print("""
What it does: override or add any model's price per 1M tokens.
Useful for: fine-tuned models, private deployments, Bedrock pricing.
""")

# Before
cost_before = estimate_cost("gpt-4o-mini", 1000, 500)
print(f"  gpt-4o-mini cost before override: ${cost_before:.6f}")

# Override
set_price("gpt-4o-mini", prompt_per_1m=0.10, completion_per_1m=0.40)  # hypothetical
cost_after = estimate_cost("gpt-4o-mini", 1000, 500)
print(f"  gpt-4o-mini cost after  override: ${cost_after:.6f}")

# Add a new model
set_price("my-fine-tuned-llama", prompt_per_1m=0.50, completion_per_1m=1.50)
cost_custom = estimate_cost("my-fine-tuned-llama", 1000, 500)
print(f"  my-fine-tuned-llama cost:          ${cost_custom:.6f}")

# Reset for other sections
set_price("gpt-4o-mini", prompt_per_1m=0.15, completion_per_1m=0.60)
print("""
  Usage:
    from agentlens.pricing import set_price
    set_price("my-model", prompt_per_1m=2.00, completion_per_1m=6.00)
    # Now any span that records tokens for "my-model" gets auto-costed
""")


# ══════════════════════════════════════════════════════════════
# 7. ERROR HANDLING — what happens when a span fails
# ══════════════════════════════════════════════════════════════
section("7. ERROR HANDLING")
print("""
What it does: errors inside a span are caught, recorded on the span
(status=error, error=message), and re-raised. The trace still saves.
The E.succeeded() eval catches it.
""")

tracer7 = Tracer()
try:
    with tracer7.trace("error-demo"):
        with tracer7.agent("router"):
            with tracer7.llm("classify", model="gpt-4o-mini") as s:
                s.record_tokens(prompt=50, completion=5, model="gpt-4o-mini")
            with tracer7.tool("broken_tool") as s:
                raise ValueError("Database connection timeout")   # simulated failure
except ValueError:
    pass  # in real code you'd handle this

t7 = tracer7.finished[-1]
print(f"  trace status : {t7.status}")
for sp in t7.spans:
    err = f"  ← ERROR: {sp.error}" if sp.status == "error" else ""
    print(f"  [{sp.kind:5}] {sp.name}  status={sp.status}{err}")

print()
report7 = E.Suite("error check", [E.succeeded()]).run(t7)
print(f"  {report7.summary()}")
print("""
  The broken span is still in the trace with full context.
  You can inspect what the agent did UP TO the failure point.
""")


# ══════════════════════════════════════════════════════════════
# 8. WHAT'S NOT AVAILABLE YET — honest gaps
# ══════════════════════════════════════════════════════════════
section("8. WHAT'S NOT AVAILABLE YET  (honest gaps)")
print("""
  These are in the roadmap or simply don't exist yet:

  ✗ REAL-TIME STREAMING TRACES
      Spans are recorded in-memory and flushed at the end.
      You can't watch a trace build live (e.g. token streaming).
      Workaround: none yet — post-run inspection only.

  ✗ ASYNC / CONCURRENT AGENT SUPPORT
      The Tracer uses a single _current and _stack.
      If two agents run in parallel (asyncio tasks), their spans
      will collide on the same stack.
      Workaround: create one Tracer per concurrent run.

  ✗ AUTOMATIC LLM INTERCEPTION
      You must manually wrap LLM calls with tracer.llm().
      It does NOT auto-patch openai.ChatCompletion or requests.
      Workaround: use the adapters for Strands/LangGraph/OpenAI Agents,
      or wrap manually.

  ✗ PII DETECTION / REDACTION
      Inputs and outputs are stored as-is. No scrubbing.
      Workaround: call s.set_input(message=redact(msg)) yourself.

  ✗ ALERTS / WEBHOOKS
      No built-in "email me when cost > $X" or Slack notifications.
      Workaround: check report.passed in your own alert logic.

  ✗ TRACE DIFFING BETWEEN RUNS
      Can't compare two runs of the same agent side-by-side in the UI.
      Workaround: compare_designs.py does it in the terminal.

  ✗ MULTI-TENANT USER SEPARATION IN THE SERVER
      The server has project-level isolation but no per-user rows.
      Workaround: use metadata fields (user=...) and filter by them.

  ✗ PERSISTENT SPAN CONTEXT ACROSS HTTP CALLS
      If your agent spans multiple services/microservices, there's
      no W3C TraceContext propagation built in.
      Workaround: export to OTel backend (Tempo/Jaeger) which handles this.

  ✗ BUILT-IN RETRY TRACING
      If an LLM call retries internally, each attempt isn't a separate span.
      Workaround: wrap your retry loop manually.

  ✗ BROWSER / FRONTEND TRACING
      No JS SDK. Terminal and Python only.
""")


# ══════════════════════════════════════════════════════════════
# QUICK REFERENCE
# ══════════════════════════════════════════════════════════════
section("QUICK REFERENCE — copy-paste snippets")
print("""
  ── Install ──────────────────────────────────────────────────
  pip install agentlens                      # core only
  pip install "agentlens[server]"            # + dashboard
  pip install "agentlens[viewer,otlp]"       # + viewer + OpenTelemetry

  ── Instrument ───────────────────────────────────────────────
  tracer = Tracer()
  with tracer.trace("name", user="alice"):
      with tracer.llm("step", model="gpt-4o-mini") as s:
          s.record_tokens(prompt=100, completion=20, model="gpt-4o-mini")
          s.set_output(text="...")
      with tracer.tool("my_tool", arg="val") as s:
          s.set_output(result="...")
      tracer.event("checkpoint", key="val")

  ── Eval in CI ───────────────────────────────────────────────
  suite = E.Suite("name", [
      E.succeeded(), E.called_tool("X"), E.never_called_tool("Y"),
      E.tool_before("A", "B"), E.max_cost(0.01), E.max_tokens(500),
      E.max_duration_ms(3000), E.had_event("approved"),
      E.custom("label", lambda t: True/False),
      E.llm_judge("rubric text", threshold=0.7),
  ])
  assert suite.run(trace).passed

  ── Save / inspect ───────────────────────────────────────────
  tracer.save("runs/")                       # → runs/<id>.json
  agentlens ls                               # list all traces
  python examples/inspect_trace.py --all     # full terminal view

  ── Add custom model price ───────────────────────────────────
  from agentlens.pricing import set_price
  set_price("my-model", prompt_per_1m=2.0, completion_per_1m=6.0)

  ── Ship to AWS / Grafana / Jaeger ───────────────────────────
  from agentlens.export import install_otlp_bridge, emit_to_otlp
  install_otlp_bridge(endpoint="http://localhost:4317")
  emit_to_otlp(trace)
""")
