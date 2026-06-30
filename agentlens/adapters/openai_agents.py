"""
OpenAI Agents SDK adapter.

The OpenAI Agents SDK has its own tracing system based on a `TracingProcessor`
interface (on_trace_start / on_span_start / on_span_end / on_trace_end). This
adapter implements that interface and routes every event into AgentLens so you
get the same unified trace model (with cost accounting + evals) regardless of
which SDK you used to build the agent.

    from agents import Agent, Runner, add_trace_processor
    from agentlens import Tracer
    from agentlens.adapters.openai_agents import make_agentlens_processor

    tracer = Tracer()
    add_trace_processor(make_agentlens_processor(tracer))

    agent = Agent(name="Assistant", instructions="You are helpful.")
    Runner.run_sync(agent, "Hello!")

    tracer.save("runs")

Imports `agents` lazily so AgentLens core stays dependency-free.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..tracer import Tracer
from ..trace import Span
from ..pricing import estimate_cost


def _base_processor_cls():
    """Import the SDK's TracingProcessor base lazily."""
    try:
        from agents.tracing.processor_interface import TracingProcessor  # type: ignore
    except ImportError:
        try:
            from agents.tracing import TracingProcessor  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The OpenAI Agents adapter needs the openai-agents SDK. "
                "Install with: pip install openai-agents"
            ) from exc
    return TracingProcessor


def _classify_span(sdk_span: Any) -> str:
    """Map an SDK Span's type to an AgentLens span kind."""
    # SDK spans expose a `.span_data` with a `type` string in most versions.
    sd = getattr(sdk_span, "span_data", None)
    t = (getattr(sd, "type", None)
         or getattr(sdk_span, "type", None)
         or "").lower()
    if "generation" in t or "response" in t or t == "llm":
        return "llm"
    if "function" in t or "tool" in t:
        return "tool"
    if "guardrail" in t or "handoff" in t:
        return "event"
    return "agent"


def _span_name(sdk_span: Any, kind: str) -> str:
    sd = getattr(sdk_span, "span_data", None)
    for attr in ("name", "tool_name", "function_name", "type"):
        v = getattr(sd, attr, None) if sd is not None else None
        if v:
            return str(v)
    return kind


def _extract_usage(sdk_span: Any) -> Dict[str, int]:
    sd = getattr(sdk_span, "span_data", None)
    usage = getattr(sd, "usage", None) if sd is not None else None
    if usage is None:
        return {"prompt": 0, "completion": 0, "model": ""}
    def _g(k1, k2):
        if isinstance(usage, dict):
            return int(usage.get(k1) or usage.get(k2) or 0)
        return int(getattr(usage, k1, None) or getattr(usage, k2, None) or 0)
    model = ""
    if sd is not None:
        model = (getattr(sd, "model", None)
                 or (sd.get("model") if isinstance(sd, dict) else None)
                 or "")
    return {
        "prompt": _g("input_tokens", "prompt_tokens"),
        "completion": _g("output_tokens", "completion_tokens"),
        "model": str(model or ""),
    }


def make_agentlens_processor(tracer: Tracer, name: str = "openai-agent"):
    """Build a TracingProcessor bound to the given AgentLens tracer (factory)."""
    Base = _base_processor_cls()

    class AgentLensProcessor(Base):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__()
            self.tracer = tracer
            self.default_name = name
            self._spans: Dict[str, Span] = {}

        # ---- trace lifecycle -----------------------------------------
        def on_trace_start(self, trace: Any) -> None:
            workflow = getattr(trace, "workflow_name", None) or self.default_name
            self.tracer.start_trace(workflow, sdk="openai-agents")

        def on_trace_end(self, trace: Any) -> None:
            self.tracer.end_trace()

        # ---- span lifecycle ------------------------------------------
        def on_span_start(self, span: Any) -> None:
            kind = _classify_span(span)
            name = _span_name(span, kind)
            sp = self.tracer.start_span(name, kind)
            self._spans[str(getattr(span, "span_id", id(span)))] = sp

        def on_span_end(self, span: Any) -> None:
            key = str(getattr(span, "span_id", id(span)))
            sp = self._spans.pop(key, None)
            if sp is None:
                return

            err = getattr(span, "error", None)
            status = "error" if err else "ok"

            # token usage for LLM spans
            if sp.kind == "llm":
                u = _extract_usage(span)
                sp.prompt_tokens += u["prompt"]
                sp.completion_tokens += u["completion"]
                if u["model"]:
                    sp.model = u["model"]
                    sp.cost_usd = estimate_cost(sp.model, sp.prompt_tokens, sp.completion_tokens)

            self.tracer.end_span(sp, status=status, error=str(err) if err else None)

        # ---- required no-ops -----------------------------------------
        def shutdown(self) -> None:
            self._spans.clear()

        def force_flush(self) -> None:
            pass

    return AgentLensProcessor()
