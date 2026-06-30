# Launch posts

## Hacker News — Show HN

**Title:** Show HN: AgentLens – Observability and evals for AI agents (Python, MIT)

**Body:**

I built AgentLens because most AI agent projects I'd seen die before production,
and almost never because of the prompts or the model. They die on the ops layer:
no tracing, no evals, no guardrails.

It's a small, dependency-free Python library that does three things:

- **Tracing** — wrap any agent and every LLM and tool call becomes a span with
  timing, tokens, and auto-computed cost.
- **Evals** — declarative checks (`never_called_tool`, `tool_before`,
  `max_cost`, `max_duration_ms`, plus an LLM-as-judge powered by Groq) you can
  run in CI as regression gates.
- **Export** — a local web viewer for runs, and a real OpenTelemetry bridge
  that ships traces to Grafana Tempo, Jaeger, AWS X-Ray (via the AWS Distro),
  or Amazon Bedrock AgentCore observability.

Adapters auto-instrument the popular frameworks — Strands SDK, LangGraph /
LangChain, and the OpenAI Agents SDK — so you don't have to write span code.

The core has zero dependencies on purpose. Drop it into any environment.

Repo: https://github.com/EeswaraReddy/agentlens
Install: `pip install agentlens`

Feedback welcome, especially on the eval check vocabulary and the OTel
attribute conventions.

---

## LinkedIn

I just open-sourced AgentLens — observability and evals for AI agents.

After building a few agents, I'm convinced the reason most projects don't reach
production isn't the model or the prompts. It's the missing ops layer: nobody
can see what the agent did, prove it behaved, or stop it when it didn't.

AgentLens is that layer:
• Tracing — every LLM and tool call becomes a span with tokens and cost
• Evals — declarative checks like "never call issue_refund" or "stay under
  $0.01", run in CI as regression gates
• A local web viewer for inspecting any run
• A real OpenTelemetry bridge to Grafana Tempo, Jaeger, AWS X-Ray, or Amazon
  Bedrock AgentCore observability
• Adapters for Strands, LangGraph, and the OpenAI Agents SDK
• An LLM-as-judge eval check (powered by Groq — fast and free tier)
• Zero dependencies in the core

The biggest lesson from building it: a team of agents is less about the model
and more about orchestration and boundaries. A clear state machine and a hard
approval gate did more for reliability than any prompt tweak.

`pip install agentlens`
Repo: https://github.com/EeswaraReddy/agentlens

Would love your feedback, especially on the eval vocabulary. What checks would
you want in your CI pipeline for an agent?

#AI #AgenticAI #LLMOps #Observability #OpenSource #AWS #Python

---

## X / Twitter (thread)

1/ I open-sourced AgentLens today — observability + evals for AI agents in
Python. 🧵

Why: ~88% of AI agent projects die before production, and almost never because
of the prompt or the model. It's the ops layer.

2/ Pillar 1 — tracing. Wrap any agent and every LLM + tool call becomes a span
with timing, tokens, and auto-computed cost. One context manager.

3/ Pillar 2 — evals. Declarative checks: never_called_tool("issue_refund"),
tool_before("lookup", "reply"), max_cost(0.01). Run them in CI as regression
gates. There's also an LLM-as-judge check powered by Groq.

4/ Pillar 3 — export. A local viewer for runs, plus a real OpenTelemetry
bridge that ships traces to Grafana Tempo, Jaeger, AWS X-Ray, or Amazon
Bedrock AgentCore.

5/ Adapters auto-instrument Strands, LangGraph/LangChain, and the OpenAI
Agents SDK. No manual span code.

6/ Best decision: the core has zero dependencies. Drop it into anything.

7/ Building the watcher taught me more about agents than building the agents
did.

`pip install agentlens`
github.com/EeswaraReddy/agentlens
