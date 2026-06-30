"""
Strands Agents adapter.

Auto-instruments a Strands agent (or multi-agent orchestrator) into AgentLens
traces by subscribing to the SDK's lifecycle hooks. No manual span code needed
in your agent.

    from strands import Agent
    from agentlens import Tracer
    from agentlens.adapters.strands import AgentLensHook

    tracer = Tracer()
    agent = Agent(tools=[...], hooks=[AgentLensHook(tracer, name="support")])
    agent("Where is my order?")

    tracer.save("runs")

Maps:
    BeforeInvocationEvent / AfterInvocationEvent  -> trace + root "agent" span
    BeforeModelCallEvent  / AfterModelCallEvent   -> "llm" span (+ token usage)
    BeforeToolCallEvent   / AfterToolCallEvent    -> "tool" span

This module imports `strands` lazily, so AgentLens core stays dependency-free.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..tracer import Tracer
from ..trace import Span


def _import_hook_base():
    """Import the Strands HookProvider base + event types lazily."""
    try:
        from strands.hooks import (  # type: ignore
            HookProvider,
            HookRegistry,
            BeforeInvocationEvent,
            AfterInvocationEvent,
            BeforeModelCallEvent,
            AfterModelCallEvent,
            BeforeToolCallEvent,
            AfterToolCallEvent,
        )
    except ImportError as exc:  # pragma: no cover - only when strands absent
        raise ImportError(
            "AgentLensHook requires the Strands SDK. Install with: pip install strands-agents"
        ) from exc
    return {
        "HookProvider": HookProvider,
        "HookRegistry": HookRegistry,
        "BeforeInvocationEvent": BeforeInvocationEvent,
        "AfterInvocationEvent": AfterInvocationEvent,
        "BeforeModelCallEvent": BeforeModelCallEvent,
        "AfterModelCallEvent": AfterModelCallEvent,
        "BeforeToolCallEvent": BeforeToolCallEvent,
        "AfterToolCallEvent": AfterToolCallEvent,
    }


def _extract_token_usage(event: Any) -> Dict[str, int]:
    """Best-effort token extraction from an AfterModelCallEvent.

    Strands surfaces usage on the model result; shapes vary by provider, so we
    probe a few common locations and fall back to zero.
    """
    usage = None
    for path in ("result", "response", "message", "stop_response"):
        obj = getattr(event, path, None)
        if obj is None:
            continue
        usage = getattr(obj, "usage", None) or (
            obj.get("usage") if isinstance(obj, dict) else None
        )
        if usage:
            break
    if not usage:
        return {"prompt": 0, "completion": 0}

    def _get(u: Any, *keys: str) -> int:
        for k in keys:
            if isinstance(u, dict) and k in u:
                return int(u[k] or 0)
            v = getattr(u, k, None)
            if v is not None:
                return int(v)
        return 0

    return {
        "prompt": _get(usage, "inputTokens", "input_tokens", "prompt_tokens"),
        "completion": _get(usage, "outputTokens", "output_tokens", "completion_tokens"),
    }


def make_agentlens_hook(tracer: Tracer, name: str = "strands-agent",
                        model: Optional[str] = None, **metadata: Any):
    """Build an AgentLensHook instance bound to the given tracer.

    Returned as a factory so the Strands base class is only imported when used.
    """
    H = _import_hook_base()
    HookProvider = H["HookProvider"]

    class AgentLensHook(HookProvider):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            self.tracer = tracer
            self.name = name
            self.model = model
            self.metadata = metadata
            self._root: Optional[Span] = None
            self._model_span: Optional[Span] = None
            self._tool_spans: Dict[str, Span] = {}

        def register_hooks(self, registry: Any, **kwargs: Any) -> None:
            registry.add_callback(H["BeforeInvocationEvent"], self._before_invocation)
            registry.add_callback(H["AfterInvocationEvent"], self._after_invocation)
            registry.add_callback(H["BeforeModelCallEvent"], self._before_model)
            registry.add_callback(H["AfterModelCallEvent"], self._after_model)
            registry.add_callback(H["BeforeToolCallEvent"], self._before_tool)
            registry.add_callback(H["AfterToolCallEvent"], self._after_tool)

        # ---- invocation ------------------------------------------------
        def _before_invocation(self, event: Any) -> None:
            agent_name = getattr(getattr(event, "agent", None), "name", None) or self.name
            self.tracer.start_trace(self.name, agent=agent_name, **self.metadata)
            self._root = self.tracer.start_span(agent_name, "agent")

        def _after_invocation(self, event: Any) -> None:
            if self._root is not None:
                self.tracer.end_span(self._root)
                self._root = None
            self.tracer.end_trace()

        # ---- model -----------------------------------------------------
        def _before_model(self, event: Any) -> None:
            self._model_span = self.tracer.start_span("model_call", "llm")

        def _after_model(self, event: Any) -> None:
            if self._model_span is None:
                return
            tokens = _extract_token_usage(event)
            handle_model = self._model_span
            handle_model.prompt_tokens += tokens["prompt"]
            handle_model.completion_tokens += tokens["completion"]
            if self.model:
                handle_model.model = self.model
                from ..pricing import estimate_cost
                handle_model.cost_usd = estimate_cost(
                    self.model, handle_model.prompt_tokens, handle_model.completion_tokens
                )
            err = getattr(event, "exception", None)
            self.tracer.end_span(
                handle_model,
                status="error" if err else "ok",
                error=str(err) if err else None,
            )
            self._model_span = None

        # ---- tools -----------------------------------------------------
        def _tool_key(self, event: Any) -> str:
            tu = getattr(event, "tool_use", {}) or {}
            return str(tu.get("toolUseId") or tu.get("name") or id(event))

        def _before_tool(self, event: Any) -> None:
            tu = getattr(event, "tool_use", {}) or {}
            tool_name = tu.get("name", "tool")
            span = self.tracer.start_span(tool_name, "tool", **(tu.get("input") or {}))
            self._tool_spans[self._tool_key(event)] = span

        def _after_tool(self, event: Any) -> None:
            key = self._tool_key(event)
            span = self._tool_spans.pop(key, None)
            if span is None:
                return
            result = getattr(event, "result", None)
            err = getattr(event, "exception", None)
            status = "ok"
            outputs: Dict[str, Any] = {}
            if isinstance(result, dict):
                status = "error" if result.get("status") == "error" else "ok"
                outputs["result"] = result.get("content", result)
            if err:
                status = "error"
            self.tracer.end_span(
                span, status=status, error=str(err) if err else None, **outputs
            )

    return AgentLensHook()
