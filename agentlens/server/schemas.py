"""Pydantic request/response models for the server API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ProjectOut(BaseModel):
    id: str
    name: str
    api_key: str


class ProjectMeta(BaseModel):
    id: str
    name: str


class IngestRequest(BaseModel):
    """A serialized AgentLens Trace, as produced by `trace.to_dict()`."""
    trace: Dict[str, Any]


class IngestResponse(BaseModel):
    trace_id: str
    spans_stored: int


class TraceSummary(BaseModel):
    trace_id: str
    name: str
    status: str
    duration_ms: float
    total_tokens: int
    cost_usd: float
    span_count: int
    error_count: int
    start_ts: str


class TraceListResponse(BaseModel):
    items: List[TraceSummary]
    total: int
    page: int
    page_size: int


class SpanDetail(BaseModel):
    span_id: str
    parent_id: Optional[str]
    name: str
    kind: str
    status: str
    duration_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    model: Optional[str]
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class TraceDetail(BaseModel):
    trace_id: str
    name: str
    status: str
    duration_ms: float
    total_tokens: int
    cost_usd: float
    span_count: int
    error_count: int
    start_ts: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    spans: List[SpanDetail]


class StatsResponse(BaseModel):
    total_traces: int
    total_tokens: int
    total_cost_usd: float
    error_rate: float
    avg_duration_ms: float
    traces_per_day: List[Dict[str, Any]]   # [{date, count, cost, errors}]
    top_models: List[Dict[str, Any]]       # [{model, calls, tokens, cost}]


class ErrorResponse(BaseModel):
    detail: str
