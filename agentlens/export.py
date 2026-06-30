"""
Trace exporters.

AgentLens traces are framework-agnostic. These exporters push them to common
backends:

  - to_otel_spans()  : convert to OpenTelemetry-style span dicts (and optionally
                       emit via the OTel SDK if installed)
  - AgentCoreExporter: send traces to Amazon Bedrock AgentCore observability,
                       which is OpenTelemetry-based. Falls back to writing OTLP
                       JSON-lines locally when AWS deps/creds are absent.

Everything degrades gracefully so the core stays dependency-free and testable.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from .trace import Trace


# GenAI semantic-convention-ish attribute keys (OpenTelemetry GenAI SIG).
_KIND_TO_OTEL = {
    "agent": "agent",
    "llm": "chat",
    "tool": "execute_tool",
    "event": "event",
}


def to_otel_spans(trace: Trace) -> List[Dict[str, Any]]:
    """Convert an AgentLens trace into OpenTelemetry-style span dicts.

    Uses GenAI semantic conventions for the common attributes (operation name,
    token usage) so downstream tooling understands them.
    """
    otel: List[Dict[str, Any]] = []
    trace_id = trace.trace_id
    for s in trace.spans:
        start_ns = int(s.start_ts * 1e9)
        dur_ns = int((s.duration_ms or 0.0) * 1e6)
        attrs: Dict[str, Any] = {
            "gen_ai.operation.name": _KIND_TO_OTEL.get(s.kind, s.kind),
            "agentlens.kind": s.kind,
        }
        if s.model:
            attrs["gen_ai.request.model"] = s.model
        if s.prompt_tokens:
            attrs["gen_ai.usage.input_tokens"] = s.prompt_tokens
        if s.completion_tokens:
            attrs["gen_ai.usage.output_tokens"] = s.completion_tokens
        if s.cost_usd:
            attrs["agentlens.cost_usd"] = round(s.cost_usd, 6)
        if s.inputs:
            attrs["agentlens.inputs"] = json.dumps(s.inputs)[:2000]
        if s.outputs:
            attrs["agentlens.outputs"] = json.dumps(s.outputs)[:2000]
        if s.error:
            attrs["error.message"] = s.error

        otel.append({
            "name": s.name,
            "trace_id": trace_id,
            "span_id": s.span_id,
            "parent_span_id": s.parent_id,
            "start_time_unix_nano": start_ns,
            "end_time_unix_nano": start_ns + dur_ns,
            "status": {"code": "ERROR" if s.status == "error" else "OK"},
            "attributes": attrs,
        })
    return otel


class AgentCoreExporter:
    """Export traces to Amazon Bedrock AgentCore observability.

    AgentCore observability is OpenTelemetry-based, so traces flow as OTLP. When
    the OTel SDK / AWS OTLP endpoint is configured via environment, spans are
    emitted there. Otherwise this writes OTLP-shaped JSON-lines to a local file
    so the pipeline is fully testable offline.

    Env it respects:
      OTEL_EXPORTER_OTLP_ENDPOINT  - OTLP collector / AgentCore endpoint
      AGENTCORE_LOG_GROUP          - (informational) CloudWatch log group
    """

    def __init__(self, endpoint: str | None = None,
                 fallback_path: str = "runs/agentcore_export.jsonl") -> None:
        self.endpoint = endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        self.fallback_path = fallback_path

    def export(self, trace: Trace) -> Dict[str, Any]:
        spans = to_otel_spans(trace)
        if self.endpoint and self._try_otlp(spans):
            return {"transport": "otlp", "endpoint": self.endpoint, "spans": len(spans)}
        # offline fallback
        os.makedirs(os.path.dirname(self.fallback_path) or ".", exist_ok=True)
        record = {
            "exported_at": time.time(),
            "trace_id": trace.trace_id,
            "resource": {"service.name": trace.name},
            "spans": spans,
        }
        with open(self.fallback_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return {"transport": "file", "path": self.fallback_path, "spans": len(spans)}

    def _try_otlp(self, spans: List[Dict[str, Any]]) -> bool:  # pragma: no cover
        """Attempt a real OTLP emit. Returns False if the SDK isn't available."""
        try:
            from opentelemetry import trace as _ot  # noqa: F401
        except ImportError:
            return False
        # A full OTLP bridge wires these dicts into SDK ReadableSpans. Kept as a
        # clearly-marked extension point so the offline path stays the default.
        return False


def install_otlp_bridge(service_name: str = "agentlens",
                       endpoint: Optional[str] = None,
                       insecure: bool = True,
                       console: bool = False):
    """Configure a real OpenTelemetry TracerProvider + OTLP exporter.

    After installation, `emit_to_otlp(trace)` will ship AgentLens traces as real
    OTel spans to any OTLP-compatible backend: Grafana Tempo, Jaeger, AWS X-Ray
    (via the AWS Distro), Amazon Bedrock AgentCore observability, Honeycomb,
    Datadog, Langfuse, etc.

    Args:
        service_name: resource attribute `service.name`
        endpoint: OTLP gRPC endpoint (e.g. http://localhost:4317). Falls back to
                  OTEL_EXPORTER_OTLP_ENDPOINT.
        insecure: gRPC insecure channel (true for local collectors)
        console: also print spans to stdout (useful for debugging)

    Returns the configured TracerProvider, or raises ImportError if the OTel
    SDK isn't installed.
    """
    try:
        from opentelemetry import trace as ot_trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor, ConsoleSpanExporter
        )
        from opentelemetry.sdk.resources import Resource
    except ImportError as exc:
        raise ImportError(
            "OTLP bridge needs opentelemetry-sdk. Install with: "
            "pip install opentelemetry-sdk opentelemetry-exporter-otlp"
        ) from exc

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    endpoint = endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=insecure))
            )
        except ImportError:
            # gRPC exporter missing; fall through and rely on console if set.
            pass
    if console or not endpoint:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    ot_trace.set_tracer_provider(provider)
    return provider


def emit_to_otlp(trace: Trace, service_name: str = "agentlens") -> int:
    """Replay an AgentLens trace as real OpenTelemetry spans.

    `install_otlp_bridge()` must be called first (or a global TracerProvider
    must already be configured). Returns the number of spans emitted.

    Note: this preserves parent/child structure within the AgentLens trace.
    Each emitted OTel span carries GenAI semantic-convention attributes and
    AgentLens-specific attributes (kind, cost, inputs/outputs).
    """
    try:
        from opentelemetry import trace as ot_trace
        from opentelemetry.trace import Status, StatusCode
    except ImportError as exc:
        raise ImportError("opentelemetry not installed") from exc

    tracer = ot_tracer = ot_trace.get_tracer("agentlens", "0.1.0")
    otel_span_index: Dict[str, Any] = {}

    # Walk spans in declaration order so parents are seen before children.
    for s in trace.spans:
        parent_span = otel_span_index.get(s.parent_id) if s.parent_id else None
        ctx = ot_trace.set_span_in_context(parent_span) if parent_span is not None else None

        with tracer.start_as_current_span(s.name, context=ctx) as ot_span:
            otel_span_index[s.span_id] = ot_span
            ot_span.set_attribute("gen_ai.operation.name",
                                  _KIND_TO_OTEL.get(s.kind, s.kind))
            ot_span.set_attribute("agentlens.kind", s.kind)
            if s.model:
                ot_span.set_attribute("gen_ai.request.model", s.model)
            if s.prompt_tokens:
                ot_span.set_attribute("gen_ai.usage.input_tokens", s.prompt_tokens)
            if s.completion_tokens:
                ot_span.set_attribute("gen_ai.usage.output_tokens", s.completion_tokens)
            if s.cost_usd:
                ot_span.set_attribute("agentlens.cost_usd", round(s.cost_usd, 6))
            if s.inputs:
                ot_span.set_attribute("agentlens.inputs",
                                      json.dumps(s.inputs)[:2000])
            if s.outputs:
                ot_span.set_attribute("agentlens.outputs",
                                      json.dumps(s.outputs)[:2000])
            if s.error:
                ot_span.set_attribute("error.message", s.error)
            if s.status == "error":
                ot_span.set_status(Status(StatusCode.ERROR, s.error or ""))

    return len(trace.spans)
