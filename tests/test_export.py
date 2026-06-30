"""Tests for the manual span API and exporters."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentlens import Tracer
from agentlens.export import to_otel_spans, AgentCoreExporter


def _manual_trace():
    tr = Tracer()
    tr.start_trace("manual-run", user="bob")
    root = tr.start_span("agent", "agent")
    llm = tr.start_span("model_call", "llm")
    llm.prompt_tokens = 100
    llm.completion_tokens = 50
    llm.model = "gpt-4o"
    tr.end_span(llm, status="ok", reply="hi")
    tool = tr.start_span("lookup", "tool", q="x")
    tr.end_span(tool, status="ok", result="found")
    tr.end_span(root)
    tr.end_trace()
    return tr.finished[-1]


def test_manual_span_api_builds_tree():
    trace = _manual_trace()
    assert trace.status == "ok"
    assert len(trace.spans) == 3
    root = next(s for s in trace.spans if s.kind == "agent")
    llm = next(s for s in trace.spans if s.kind == "llm")
    tool = next(s for s in trace.spans if s.kind == "tool")
    assert llm.parent_id == root.span_id
    assert tool.parent_id == root.span_id
    assert trace.total_tokens == 150
    assert trace.duration_ms is not None


def test_manual_span_error_marks_trace():
    tr = Tracer()
    tr.start_trace("t")
    s = tr.start_span("boom", "tool")
    tr.end_span(s, status="error", error="bad")
    tr.end_trace()
    trace = tr.finished[-1]
    assert trace.status == "error"


def test_to_otel_spans_uses_genai_conventions():
    trace = _manual_trace()
    spans = to_otel_spans(trace)
    assert len(spans) == 3
    llm = next(s for s in spans if s["attributes"]["agentlens.kind"] == "llm")
    assert llm["attributes"]["gen_ai.operation.name"] == "chat"
    assert llm["attributes"]["gen_ai.usage.input_tokens"] == 100
    assert llm["attributes"]["gen_ai.usage.output_tokens"] == 50
    assert llm["attributes"]["gen_ai.request.model"] == "gpt-4o"
    # parent/child linkage preserved
    assert all("trace_id" in s and "span_id" in s for s in spans)


def test_agentcore_exporter_offline_fallback(tmp_path):
    trace = _manual_trace()
    out = tmp_path / "agentcore.jsonl"
    exporter = AgentCoreExporter(endpoint=None, fallback_path=str(out))
    info = exporter.export(trace)
    assert info["transport"] == "file"
    assert info["spans"] == 3
    line = out.read_text(encoding="utf-8").strip()
    record = json.loads(line)
    assert record["trace_id"] == trace.trace_id
    assert record["resource"]["service.name"] == "manual-run"
    assert len(record["spans"]) == 3
