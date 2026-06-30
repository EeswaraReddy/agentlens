# Changelog

All notable changes to AgentLens are documented here.
This project follows [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-06-30

Initial release.

### Added
- Core tracing primitives (`Trace`, `Span`) and a `Tracer` with context-manager
  and manual span APIs.
- Token + cost accounting via an overridable pricing table.
- Declarative eval suite: `succeeded`, `called_tool`, `never_called_tool`,
  `tool_before`, `max_cost`, `max_tokens`, `max_duration_ms`, `had_event`,
  `custom`, and `llm_judge` (LLM-as-judge).
- Local FastAPI web viewer + `agentlens view` / `agentlens ls` CLI.
- Strands SDK adapter (`agentlens.adapters.strands`) — `HookProvider` mapping
  `BeforeInvocation/AfterInvocation/BeforeModelCall/AfterModelCall/BeforeToolCall/AfterToolCall`
  events to spans.
- LangGraph / LangChain adapter (`agentlens.adapters.langgraph`) —
  `BaseCallbackHandler` mapping LLM, tool, and chain callbacks to spans.
- OpenAI Agents SDK adapter (`agentlens.adapters.openai_agents`) —
  `TracingProcessor` mapping the SDK's trace/span lifecycle to AgentLens.
- Traced Groq provider (`agentlens.providers.groq`) — dependency-free
  (stdlib `urllib`) OpenAI-compatible client that auto-records `llm` spans
  with real token usage.
- Exporters: OTLP-shaped JSON-lines for offline use; a real OpenTelemetry
  bridge (`install_otlp_bridge` + `emit_to_otlp`) that ships traces to any
  OTLP backend (Grafana Tempo, Jaeger, AWS X-Ray, Amazon Bedrock AgentCore,
  Honeycomb, Datadog, Langfuse).
- Runnable examples: `support_agent.py`, `compare_designs.py`,
  `export_to_agentcore.py`, `otlp_bridge.py`, `groq_agent.py`,
  `strands_agent.py`.
- 25 unit tests covering core, evals, exporters, and all adapters.
