"""
LangGraph / LangChain adapter.

LangGraph runs on LangChain's callback system, so AgentLens plugs in as a
`BaseCallbackHandler`. Pass it via the run config and every LLM, tool, and chain
step is captured as a span with token usage — no manual instrumentation.

    from agentlens import Tracer
    from agentlens.adapters.langgraph import AgentLensCallbackHandler

    tracer = Tracer()
    handler = AgentLensCallbackHandler(tracer, name="my-graph")

    with tracer.trace("my-graph"):
        graph.invoke({"messages": [...]}, config={"callbacks": [handler]})

    tracer.save("runs")

Imports langchain lazily so AgentLens core stays dependency-free.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from ..tracer import Tracer
from ..trace import Span
from ..pricing import estimate_cost


def _base_handler_cls():
    try:
        from langchain_core.callbacks import BaseCallbackHandler  # type: ignore
    except ImportError:
        try:
            from langchain.callbacks.base import BaseCallbackHandler  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The LangGraph adapter needs LangChain. Install with: "
                "pip install langchain-core"
            ) from exc
    return BaseCallbackHandler


def make_langgraph_handler(tracer: Tracer, name: str = "langgraph",
                           model: Optional[str] = None, **metadata: Any):
    """Build a LangChain callback handler bound to the tracer (factory)."""
    Base = _base_handler_cls()

    class AgentLensCallbackHandler(Base):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__()
            self.tracer = tracer
            self.name = name
            self.model = model
            self.metadata = metadata
            self._spans: Dict[str, Span] = {}
            self._owns_trace = False

        def _ensure_trace(self) -> None:
            if self.tracer._current is None:  # noqa: SLF001
                self.tracer.start_trace(self.name, **self.metadata)
                self._owns_trace = True

        # ---- LLM ------------------------------------------------------
        def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str],
                         *, run_id: UUID, **kwargs: Any) -> None:
            self._ensure_trace()
            span = self.tracer.start_span("llm", "llm", prompts=prompts[:1])
            self._spans[str(run_id)] = span

        def on_chat_model_start(self, serialized: Dict[str, Any], messages: Any,
                                *, run_id: UUID, **kwargs: Any) -> None:
            self._ensure_trace()
            span = self.tracer.start_span("chat_model", "llm")
            self._spans[str(run_id)] = span

        def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
            span = self._spans.pop(str(run_id), None)
            if span is None:
                return
            prompt_t, completion_t = self._token_usage(response)
            span.prompt_tokens += prompt_t
            span.completion_tokens += completion_t
            mdl = self.model or self._model_name(response)
            if mdl:
                span.model = mdl
                span.cost_usd = estimate_cost(mdl, span.prompt_tokens, span.completion_tokens)
            text = self._first_text(response)
            self.tracer.end_span(span, status="ok", **({"text": text} if text else {}))

        def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
            span = self._spans.pop(str(run_id), None)
            if span is not None:
                self.tracer.end_span(span, status="error", error=str(error))

        # ---- tools ----------------------------------------------------
        def on_tool_start(self, serialized: Dict[str, Any], input_str: str,
                          *, run_id: UUID, **kwargs: Any) -> None:
            self._ensure_trace()
            tool_name = (serialized or {}).get("name", "tool")
            span = self.tracer.start_span(tool_name, "tool", input=input_str)
            self._spans[str(run_id)] = span

        def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
            span = self._spans.pop(str(run_id), None)
            if span is not None:
                self.tracer.end_span(span, status="ok", result=str(output)[:500])

        def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
            span = self._spans.pop(str(run_id), None)
            if span is not None:
                self.tracer.end_span(span, status="error", error=str(error))

        # ---- chains / graph nodes ------------------------------------
        def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any],
                           *, run_id: UUID, **kwargs: Any) -> None:
            self._ensure_trace()
            node = (serialized or {}).get("name") or "chain"
            span = self.tracer.start_span(node, "agent")
            self._spans[str(run_id)] = span

        def on_chain_end(self, outputs: Dict[str, Any], *, run_id: UUID, **kwargs: Any) -> None:
            span = self._spans.pop(str(run_id), None)
            if span is not None:
                self.tracer.end_span(span, status="ok")
            # close the trace once the outermost chain finishes
            if self._owns_trace and not self._spans:
                self.tracer.end_trace()
                self._owns_trace = False

        # ---- helpers --------------------------------------------------
        @staticmethod
        def _token_usage(response: Any):
            out = getattr(response, "llm_output", None) or {}
            usage = out.get("token_usage") or out.get("usage") or {}
            p = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            c = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            return p, c

        @staticmethod
        def _model_name(response: Any) -> Optional[str]:
            out = getattr(response, "llm_output", None) or {}
            return out.get("model_name") or out.get("model")

        @staticmethod
        def _first_text(response: Any) -> Optional[str]:
            try:
                gens = getattr(response, "generations", None)
                if gens and gens[0]:
                    g = gens[0][0]
                    return getattr(g, "text", None) or getattr(getattr(g, "message", None), "content", None)
            except Exception:
                return None
            return None

    return AgentLensCallbackHandler()
