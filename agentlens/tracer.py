"""
Tracer — the instrumentation API.

Usage:

    from agentlens import Tracer

    tracer = Tracer()
    with tracer.trace("support-agent", user="alice") as t:
        with tracer.agent("router"):
            with tracer.llm("classify", model="gpt-4o-mini") as span:
                ... call your model ...
                span.record_tokens(prompt=120, completion=18)
            with tracer.tool("lookup_order", order_id="A1") as span:
                span.set_output(result="shipped")

    tracer.save("runs")   # writes runs/<trace_id>.json
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from .trace import Span, Trace
from .pricing import estimate_cost


class _SpanHandle:
    """Thin wrapper handed to `with` blocks for recording data on a span."""

    def __init__(self, span: Span):
        self._span = span

    def set_input(self, **kwargs: Any) -> None:
        self._span.inputs.update(kwargs)

    def set_output(self, **kwargs: Any) -> None:
        self._span.outputs.update(kwargs)

    def set_attr(self, **kwargs: Any) -> None:
        self._span.attributes.update(kwargs)

    def record_tokens(self, prompt: int = 0, completion: int = 0,
                      model: Optional[str] = None) -> None:
        self._span.prompt_tokens += prompt
        self._span.completion_tokens += completion
        if model:
            self._span.model = model
        if self._span.model:
            self._span.cost_usd = estimate_cost(
                self._span.model, self._span.prompt_tokens, self._span.completion_tokens
            )


class Tracer:
    def __init__(self) -> None:
        self._current: Optional[Trace] = None
        self._stack: List[str] = []
        self.finished: List[Trace] = []

    # ---- trace (top-level run) -----------------------------------------
    @contextmanager
    def trace(self, name: str, **metadata: Any) -> Iterator[Trace]:
        tr = Trace(name=name, metadata=dict(metadata))
        self._current = tr
        self._stack = []
        start = time.perf_counter()
        try:
            yield tr
        except Exception as exc:
            tr.status = "error"
            tr.metadata["error"] = str(exc)
            raise
        finally:
            tr.duration_ms = (time.perf_counter() - start) * 1000.0
            if any(s.status == "error" for s in tr.spans):
                tr.status = "error"
            self.finished.append(tr)
            self._current = None
            self._stack = []

    # ---- spans ----------------------------------------------------------
    @contextmanager
    def _span(self, name: str, kind: str, **inputs: Any) -> Iterator[_SpanHandle]:
        if self._current is None:
            raise RuntimeError("No active trace. Open `with tracer.trace(...)` first.")
        sp = Span(
            name=name,
            kind=kind,
            parent_id=self._stack[-1] if self._stack else None,
            inputs=dict(inputs),
        )
        self._current.spans.append(sp)
        self._stack.append(sp.span_id)
        start = time.perf_counter()
        try:
            yield _SpanHandle(sp)
        except Exception as exc:
            sp.status = "error"
            sp.error = str(exc)
            raise
        finally:
            sp.duration_ms = (time.perf_counter() - start) * 1000.0
            self._stack.pop()

    def agent(self, name: str, **inputs: Any):
        return self._span(name, "agent", **inputs)

    def llm(self, name: str, model: Optional[str] = None, **inputs: Any):
        cm = self._span(name, "llm", **inputs)
        return cm

    def tool(self, name: str, **inputs: Any):
        return self._span(name, "tool", **inputs)

    def event(self, name: str, **attributes: Any) -> None:
        """Record an instantaneous marker (e.g. an approval gate)."""
        if self._current is None:
            raise RuntimeError("No active trace.")
        sp = Span(
            name=name, kind="event",
            parent_id=self._stack[-1] if self._stack else None,
            duration_ms=0.0, attributes=dict(attributes),
        )
        self._current.spans.append(sp)

    # ---- manual span API (for event/callback-driven instrumentation) ----
    # Context managers don't span separate callbacks (e.g. Strands hooks fire
    # Before* and After* as distinct calls), so these let an adapter open and
    # close spans imperatively.
    def start_trace(self, name: str, **metadata: Any) -> Trace:
        tr = Trace(name=name, metadata=dict(metadata))
        self._current = tr
        self._stack = []
        tr._perf_start = time.perf_counter()  # type: ignore[attr-defined]
        return tr

    def end_trace(self) -> Optional[Trace]:
        tr = self._current
        if tr is None:
            return None
        start = getattr(tr, "_perf_start", None)
        if start is not None:
            tr.duration_ms = (time.perf_counter() - start) * 1000.0
        if any(s.status == "error" for s in tr.spans):
            tr.status = "error"
        self.finished.append(tr)
        self._current = None
        self._stack = []
        return tr

    def start_span(self, name: str, kind: str, **inputs: Any) -> Span:
        if self._current is None:
            raise RuntimeError("No active trace. Call start_trace() first.")
        sp = Span(
            name=name, kind=kind,
            parent_id=self._stack[-1] if self._stack else None,
            inputs=dict(inputs),
        )
        sp._perf_start = time.perf_counter()  # type: ignore[attr-defined]
        self._current.spans.append(sp)
        self._stack.append(sp.span_id)
        return sp

    def end_span(self, span: Span, status: str = "ok",
                 error: Optional[str] = None, **outputs: Any) -> None:
        start = getattr(span, "_perf_start", None)
        if start is not None:
            span.duration_ms = (time.perf_counter() - start) * 1000.0
        if status != "ok":
            span.status = status
        if error:
            span.error = error
        span.outputs.update(outputs)
        if self._stack and self._stack[-1] == span.span_id:
            self._stack.pop()

    # ---- persistence ----------------------------------------------------
    def save(self, directory: str = "runs") -> List[str]:
        os.makedirs(directory, exist_ok=True)
        paths = []
        for tr in self.finished:
            path = os.path.join(directory, f"{tr.trace_id}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(tr.to_dict(), f, indent=2)
            paths.append(path)
        return paths
