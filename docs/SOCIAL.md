# Social posts

## LinkedIn

~88% of AI agent projects never reach production. After building a few, I'm
convinced it's almost never a model problem — it's the missing ops layer.

So I built AgentLens: observability + evals for AI agents.

It does three things every production agent needs:
• Tracing — every LLM and tool call becomes a span with timing, tokens, and cost
• Evals — declarative checks like "never call issue_refund" or "stay under $0.01"
  that you run in CI
• A local viewer — inspect any run without grepping JSON

The design choice I'd defend: the core has zero dependencies. An observability
tool you can't drop into any environment isn't a tool, it's another integration
project.

Building the watcher taught me more about agents than building the agents did.
Once you can see cost per step, you design cheaper agents. Once you have eval
gates, you refactor fearlessly.

Repo + one-command demo in the comments. 👇

#AI #AgenticAI #LLMOps #Observability #AWS #Python

---

## X / Twitter (thread)

1/ ~88% of AI agent projects die before production. It's almost never the model.
It's that nobody can see what the agent did, prove it behaved, or stop it when it
didn't. So I built AgentLens. 🧵

2/ Pillar 1 — tracing. Every LLM + tool call becomes a span: timing, tokens,
cost. One context manager, rolls up automatically.

3/ Pillar 2 — evals. Declarative checks: never_called_tool("issue_refund"),
tool_before("lookup","reply"), max_cost(0.01). Run them in CI as a regression
gate.

4/ Pillar 3 — a local viewer. Span timeline, color-coded, cost per step. Reads
the same files you'd export to OpenTelemetry / Bedrock AgentCore.

5/ Best decision: the core has zero dependencies. Drop it into anything.

6/ Building the watcher taught me more about agents than building agents did.
Repo + one-command demo below. 👇
