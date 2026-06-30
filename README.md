# AgentLens

Observability + eval harness for AI agents. Trace every LLM and tool call,
assert behavior with declarative evals, and inspect runs in a local web viewer.

Most AI agent projects don't fail on prompts or models — they fail on the
**ops layer**: no tracing, no evals, no guardrails. AgentLens is that layer, in
a dependency-free core that runs anywhere and exports to OpenTelemetry or
Amazon Bedrock AgentCore.

```
   instrument            assert                 inspect
   ──────────            ──────                 ───────
   tracer.llm(...)   →   evals.max_cost(0.01)   →   agentlens view
   tracer.tool(...)      evals.called_tool(...)
```

## Why it exists

- **Observability is table stakes.** You can't fix what you can't see. Every
  LLM and tool call becomes a structured span with timing, tokens, and cost.
- **Eval-first.** Encode the behavior you require ("never refund", "look up the
  order before replying", "stay under $0.01") and fail loudly on drift.
- **Zero cloud to try.** The core has no dependencies. Run the example, see
  traces, run evals — no API keys.

## Install

```bash
# core only (no dependencies)
pip install agentlens

# with the local web viewer
pip install "agentlens[viewer]"

# with all integrations
pip install "agentlens[viewer,otlp,langgraph,openai-agents]"
```

From source:

```bash
pip install -e .
pip install -e ".[viewer]"
```

## 60-second tour

```bash
python examples/support_agent.py   # runs a mocked, fully-traced agent + evals
agentlens ls                       # list saved traces
agentlens view                     # open http://localhost:8800
```

## Instrument your own agent

```python
from agentlens import Tracer, evals as E

tracer = Tracer()

with tracer.trace("support-agent", user="alice"):
    with tracer.agent("router"):
        with tracer.llm("classify", model="gpt-4o-mini") as span:
            # ... call your model ...
            span.record_tokens(prompt=120, completion=18, model="gpt-4o-mini")
            span.set_output(intent="order_status")

        with tracer.tool("lookup_order", order_id="A1") as span:
            span.set_output(status="shipped")

tracer.save("runs")   # writes runs/<trace_id>.json
```

Then assert behavior:

```python
suite = E.Suite("guardrails", [
    E.succeeded(),
    E.called_tool("lookup_order"),
    E.never_called_tool("issue_refund"),
    E.tool_before("lookup_order", "send_reply"),
    E.max_cost(0.01),
    E.max_duration_ms(5000),
])
report = suite.run(tracer.finished[-1])
print(report.summary())
assert report.passed          # use in CI as a regression gate
```

## Built-in eval checks

| Check | Asserts |
|-------|---------|
| `succeeded()` | the run finished without errors |
| `called_tool(name)` | a tool was used |
| `never_called_tool(name)` | a forbidden tool was **not** used |
| `tool_before(a, b)` | ordering guardrail |
| `max_cost(usd)` | run stayed under a cost budget |
| `max_tokens(n)` | run stayed under a token budget |
| `max_duration_ms(ms)` | latency budget |
| `had_event(name)` | a marker (e.g. `human_approval`) occurred |
| `custom(name, fn)` | any predicate over the trace |

## Real LLM calls with Groq (fast + free tier)

AgentLens ships a dependency-free, traced Groq client (Groq's API is
OpenAI-compatible and runs on fast LPU hardware). Each call becomes an `llm`
span with the real token usage Groq returns:

```python
from agentlens import Tracer
from agentlens.providers.groq import GroqClient

tracer = Tracer()
groq = GroqClient(tracer=tracer, model="llama-3.1-8b-instant")  # reads GROQ_API_KEY

with tracer.trace("chat"):
    reply = groq.complete("Summarize AgentLens in one line.",
                          system="Be concise.")
print(reply.text, reply.total_tokens)
```

Get a free key at https://console.groq.com, then `export GROQ_API_KEY=...`.
Full traced agent + evals: `examples/groq_agent.py`.

## LLM-as-judge evals

Beyond deterministic checks, score a run against a natural-language rubric. The
judge defaults to Groq (fast/cheap), or pass your own client:

```python
from agentlens import evals as E

suite = E.Suite("quality", [
    E.succeeded(),
    E.llm_judge(
        "The reply answers the order-status question, is friendly, and invents "
        "no facts beyond the known order info.",
        model="llama-3.3-70b-versatile",
        threshold=0.7,
    ),
])
report = suite.run(trace)     # judge returns {"score", "reason"}; passes if score >= threshold
```

## Auto-instrument a Strands agent (one line)

No manual span code — subscribe AgentLens to the Strands lifecycle hooks and
every model and tool call is captured automatically:

```python
from strands import Agent, tool
from agentlens import Tracer
from agentlens.adapters.strands import make_agentlens_hook

tracer = Tracer()
hook = make_agentlens_hook(tracer, name="support", model="claude-3-5-sonnet")

agent = Agent(tools=[...], hooks=[hook])
agent("Where is my order A1029?")

tracer.save("runs")
```

The adapter maps Strands events to spans: `Before/AfterInvocationEvent` →
trace + root agent span, `Before/AfterModelCallEvent` → `llm` span with token
usage, `Before/AfterToolCallEvent` → `tool` span. See
`examples/strands_agent.py`.

## Auto-instrument a LangGraph / LangChain agent

LangGraph runs on LangChain callbacks, so AgentLens plugs in as a callback
handler — pass it in the run config:

```python
from agentlens import Tracer
from agentlens.adapters.langgraph import make_langgraph_handler

tracer = Tracer()
handler = make_langgraph_handler(tracer, name="my-graph")

with tracer.trace("my-graph"):
    graph.invoke({"messages": [...]}, config={"callbacks": [handler]})
```

LLM, tool, and chain/node steps each become spans with token usage. Install the
extra: `pip install "agentlens[langgraph]"`.

## Auto-instrument an OpenAI Agents SDK agent

The OpenAI Agents SDK has its own tracing system. AgentLens plugs in as a
`TracingProcessor`, so every run, model call, tool call, and handoff lands in
the same trace model — with cost accounting and evals:

```python
from agents import Agent, Runner, add_trace_processor
from agentlens import Tracer
from agentlens.adapters.openai_agents import make_agentlens_processor

tracer = Tracer()
add_trace_processor(make_agentlens_processor(tracer))

agent = Agent(name="Assistant", instructions="You are helpful.")
Runner.run_sync(agent, "Hello!")

tracer.save("runs")
```

Install: `pip install "agentlens[openai-agents]"`.

## Ship to Grafana Tempo / Jaeger / X-Ray / AgentCore (real OTLP)

AgentLens has a full OpenTelemetry bridge: replay any trace as real OTel spans
into any OTLP-compatible backend.

```python
from agentlens.export import install_otlp_bridge, emit_to_otlp

# Configure once
install_otlp_bridge(
    service_name="my-service",
    endpoint="http://localhost:4317",   # or OTEL_EXPORTER_OTLP_ENDPOINT
)

# Emit any AgentLens trace as real OTel spans
emit_to_otlp(tracer.finished[-1])
```

Works with Grafana Tempo, Jaeger, SigNoz, AWS X-Ray (via the AWS Distro for
OpenTelemetry collector), Amazon Bedrock AgentCore observability, Honeycomb,
Datadog, Langfuse — anything that speaks OTLP.

Install: `pip install "agentlens[otlp]"`. Try it locally:

```bash
docker run -d -p 4317:4317 -p 4318:4318 otel/opentelemetry-collector
python examples/otlp_bridge.py
```

Or print spans to stdout (no collector needed):

```bash
python examples/otlp_bridge.py --console
```

## Lightweight OTLP-JSON export (offline)

For environments without the OTel SDK, the AgentCore exporter still produces
OTLP-shaped JSON-lines locally so you can see exactly what would ship:

```python
from agentlens.export import to_otel_spans, AgentCoreExporter

otel_spans = to_otel_spans(tracer.finished[-1])     # GenAI-convention dicts
info = AgentCoreExporter().export(tracer.finished[-1])  # writes runs/agentcore_export.jsonl
```

## What's inside

| Path | What |
|------|------|
| `agentlens/trace.py` | `Trace` / `Span` data model + aggregates |
| `agentlens/tracer.py` | instrumentation API (context-manager + manual span) |
| `agentlens/pricing.py` | token → cost estimation (override-able) |
| `agentlens/evals.py` | declarative eval checks + suite runner |
| `agentlens/export.py` | OTel conversion + AgentCore exporter + real OTLP bridge |
| `agentlens/adapters/strands.py` | auto-instrumentation for Strands agents |
| `agentlens/adapters/langgraph.py` | auto-instrumentation for LangGraph/LangChain |
| `agentlens/adapters/openai_agents.py` | auto-instrumentation for the OpenAI Agents SDK |
| `agentlens/providers/groq.py` | traced, dependency-free Groq client |
| `agentlens/viewer.py` | local FastAPI web viewer |
| `agentlens/cli.py` | `agentlens view` / `agentlens ls` |
| `examples/` | runnable samples (mock, Groq, Strands, LangGraph, OAI Agents, OTLP) |
| `tests/` | pytest suite (25 tests) |

## Run the tests

```bash
pip install pytest
pytest -q
```

## Roadmap

- Trace diffing between runs (regression view)
- A hosted demo deployable to AgentCore
- More built-in eval checks (PII, format adherence)

MIT licensed.
