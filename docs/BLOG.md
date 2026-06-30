# Why I stopped building agents and built the thing that watches them

Around 88% of AI agent projects never make it to production. After building a
few myself, I stopped believing it was a model problem. The demos worked. The
prompts were fine. What was missing was everything *around* the agent — the
ability to see what it did, prove it behaved, and stop it when it didn't.

So I built **AgentLens**: an observability and eval layer for agents. This is
the part nobody films a flashy demo about, and it's exactly the part that
decides whether an agent survives contact with real users.

## The three things every production agent needs

**1. You have to see every step.**
An agent run is a tree: it reasons, calls a model, calls a tool, calls another
model. When something goes wrong at 2am, "the agent gave a bad answer" is
useless. You need the full trace — each LLM call with its tokens and cost, each
tool call with its arguments and result, timing on all of it. AgentLens captures
that as a span tree with zero ceremony:

```python
with tracer.llm("classify", model="gpt-4o-mini") as span:
    span.record_tokens(prompt=120, completion=18, model="gpt-4o-mini")
```

Tokens and cost roll up automatically, so every run tells you what it spent.

**2. You have to assert behavior, not eyeball it.**
"Looks good" doesn't scale. The agents that ship treat behavior like code: write
checks, run them on every change, fail loudly on drift. AgentLens makes those
checks declarative:

```python
suite = E.Suite("guardrails", [
    E.never_called_tool("issue_refund"),
    E.tool_before("lookup_order", "send_reply"),
    E.max_cost(0.01),
])
```

Drop that in CI and a prompt tweak that suddenly makes your agent 5x more
expensive, or skip a required step, fails the build instead of surprising a
customer.

**3. You have to inspect runs without grepping JSON.**
AgentLens ships a small local viewer — a timeline of spans, color-coded by kind,
with tokens, cost, and status per step. It reads the same trace files you'd
export to OpenTelemetry or Amazon Bedrock AgentCore in production.

## The design choice I'd defend

The core has **no dependencies**. That was deliberate. An observability tool you
can't drop into any environment isn't an observability tool, it's another
integration project. The instrumentation is plain context managers; the viewer
is the only thing that pulls in a web framework, and it's optional.

## What I learned

Building the watcher taught me more about agents than building the agents did.
Once you can see cost per step, you start designing cheaper agents. Once you have
eval gates, you refactor fearlessly. The ops layer isn't overhead — it's what
makes the rest of the work safe.

*Code, a one-command demo, and the eval harness are in the repo. Run
`python examples/support_agent.py` and then `agentlens view`.*
