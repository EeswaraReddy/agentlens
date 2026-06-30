"""
Core tracing primitives.

A Trace is one agent run. It contains a tree of Spans. Each Span captures a unit
of work — an agent turn, an LLM call, or a tool call — with timing, status,
inputs/outputs, token usage and cost.

Dependency-free on purpose: this works in any Python environment and can later
export to OpenTelemetry or Amazon Bedrock AgentCore observability.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


SpanKind = str  # "agent" | "llm" | "tool" | "event"


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class Span:
    name: str
    kind: SpanKind
    span_id: str = field(default_factory=_new_id)
    parent_id: Optional[str] = None
    start_ts: float = field(default_factory=time.time)
    duration_ms: Optional[float] = None
    status: str = "ok"                       # "ok" | "error"
    error: Optional[str] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    # model accounting
    model: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.duration_ms is not None:
            d["duration_ms"] = round(self.duration_ms, 2)
        d["cost_usd"] = round(self.cost_usd, 6)
        d["total_tokens"] = self.total_tokens
        return d


@dataclass
class Trace:
    name: str
    trace_id: str = field(default_factory=_new_id)
    start_ts: float = field(default_factory=time.time)
    duration_ms: Optional[float] = None
    status: str = "ok"
    spans: List[Span] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---- aggregates -----------------------------------------------------
    @property
    def total_tokens(self) -> int:
        return sum(s.total_tokens for s in self.spans)

    @property
    def total_cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.spans)

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for s in self.spans:
            out[s.kind] = out.get(s.kind, 0) + 1
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "start_ts": self.start_ts,
            "duration_ms": round(self.duration_ms, 2) if self.duration_ms else None,
            "status": self.status,
            "metadata": self.metadata,
            "summary": {
                "spans": len(self.spans),
                "counts": self.counts(),
                "total_tokens": self.total_tokens,
                "total_cost_usd": round(self.total_cost_usd, 6),
                "errors": sum(1 for s in self.spans if s.status == "error"),
            },
            "spans": [s.to_dict() for s in self.spans],
        }
