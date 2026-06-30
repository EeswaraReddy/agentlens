"""
Verify the OTLP bridge calls real OpenTelemetry APIs correctly.

We inject a fake `opentelemetry` package whose `start_as_current_span` records
each call. This lets us assert the bridge sets the right GenAI/AgentLens
attributes and preserves parent/child structure.
"""

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


_recorded_spans = []


class _FakeOtelSpan:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.attributes = {}
        self.status = ("OK", "")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def set_attribute(self, k, v):
        self.attributes[k] = v

    def set_status(self, status):
        # Status is the StatusCode + description tuple in our fake
        self.status = (getattr(status, "_code", "OK"), getattr(status, "_desc", ""))


def _install_fake_otel():
    """Inject minimal opentelemetry modules used by the bridge."""
    ot = types.ModuleType("opentelemetry")
    ot_trace = types.ModuleType("opentelemetry.trace")

    class StatusCode:
        OK = "OK"
        ERROR = "ERROR"

    class Status:
        def __init__(self, code, desc=""):
            self._code = code
            self._desc = desc

    _stack = []

    class _Tracer:
        def start_as_current_span(self, name, context=None):
            parent = context  # fake: 'context' carries the parent span directly
            span = _FakeOtelSpan(name, parent=parent)
            _recorded_spans.append(span)
            _stack.append(span)
            class _Cm:
                def __enter__(_self): return span
                def __exit__(_self, *a):
                    _stack.pop()
                    return False
            return _Cm()

    def get_tracer(name, version=None):
        return _Tracer()

    def set_span_in_context(span):
        return span  # we pass the span directly through `context=`

    ot_trace.Status = Status
    ot_trace.StatusCode = StatusCode
    ot_trace.get_tracer = get_tracer
    ot_trace.set_span_in_context = set_span_in_context
    ot.trace = ot_trace
    sys.modules["opentelemetry"] = ot
    sys.modules["opentelemetry.trace"] = ot_trace


def test_otlp_bridge_emits_spans_with_attributes_and_nesting():
    _recorded_spans.clear()
    _install_fake_otel()

    from agentlens import Tracer
    from agentlens.export import emit_to_otlp

    # Build a 3-level trace: agent -> (llm, tool)
    tr = Tracer()
    tr.start_trace("svc")
    a = tr.start_span("router", "agent")
    llm = tr.start_span("classify", "llm")
    llm.prompt_tokens = 100
    llm.completion_tokens = 30
    llm.model = "gpt-4o"
    from agentlens.pricing import estimate_cost
    llm.cost_usd = estimate_cost("gpt-4o", 100, 30)
    tr.end_span(llm, status="ok", text="hi")
    tool = tr.start_span("lookup", "tool", q="A1")
    tr.end_span(tool, status="error", error="bad")
    tr.end_span(a)
    tr.end_trace()
    trace = tr.finished[-1]

    n = emit_to_otlp(trace, service_name="svc")
    assert n == 3
    assert len(_recorded_spans) == 3

    by_name = {s.name: s for s in _recorded_spans}
    # GenAI conventions on the LLM span
    llm_otel = by_name["classify"]
    assert llm_otel.attributes["gen_ai.operation.name"] == "chat"
    assert llm_otel.attributes["agentlens.kind"] == "llm"
    assert llm_otel.attributes["gen_ai.request.model"] == "gpt-4o"
    assert llm_otel.attributes["gen_ai.usage.input_tokens"] == 100
    assert llm_otel.attributes["gen_ai.usage.output_tokens"] == 30
    assert "agentlens.cost_usd" in llm_otel.attributes
    # Tool error surfaced
    tool_otel = by_name["lookup"]
    assert tool_otel.status[0] == "ERROR"
    assert tool_otel.attributes["error.message"] == "bad"
    # Parent linkage preserved (children carry the router span as parent)
    router_otel = by_name["router"]
    assert llm_otel.parent is router_otel
    assert tool_otel.parent is router_otel
