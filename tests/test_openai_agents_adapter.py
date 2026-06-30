"""Tests for the OpenAI Agents SDK adapter using a faked `agents` package."""

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _install_fake_agents():
    agents = types.ModuleType("agents")
    tracing = types.ModuleType("agents.tracing")
    proc_iface = types.ModuleType("agents.tracing.processor_interface")

    class TracingProcessor:
        pass

    proc_iface.TracingProcessor = TracingProcessor
    tracing.TracingProcessor = TracingProcessor
    tracing.processor_interface = proc_iface
    agents.tracing = tracing
    sys.modules["agents"] = agents
    sys.modules["agents.tracing"] = tracing
    sys.modules["agents.tracing.processor_interface"] = proc_iface


class FakeTrace:
    def __init__(self, workflow_name="my-workflow", trace_id="t1"):
        self.workflow_name = workflow_name
        self.trace_id = trace_id


class FakeSpanData:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeSpan:
    def __init__(self, span_id, type_, **data):
        self.span_id = span_id
        self.span_data = FakeSpanData(type=type_, **data)
        self.error = None


def test_openai_agents_processor_routes_lifecycle_to_spans():
    _install_fake_agents()
    from agentlens import Tracer
    from agentlens.adapters.openai_agents import make_agentlens_processor

    tracer = Tracer()
    proc = make_agentlens_processor(tracer)

    trace = FakeTrace(workflow_name="support-agent")
    proc.on_trace_start(trace)

    agent_span = FakeSpan("s-agent", "agent", name="Assistant")
    proc.on_span_start(agent_span)

    gen_span = FakeSpan("s-gen", "generation", model="gpt-4o",
                        usage={"input_tokens": 220, "output_tokens": 40})
    proc.on_span_start(gen_span)
    proc.on_span_end(gen_span)

    tool_span = FakeSpan("s-tool", "function", name="lookup_order")
    proc.on_span_start(tool_span)
    proc.on_span_end(tool_span)

    proc.on_span_end(agent_span)
    proc.on_trace_end(trace)
    proc.shutdown()

    t = tracer.finished[-1]
    kinds = sorted(s.kind for s in t.spans)
    assert kinds == ["agent", "llm", "tool"]
    assert t.name == "support-agent"
    assert t.metadata.get("sdk") == "openai-agents"

    llm = next(s for s in t.spans if s.kind == "llm")
    assert llm.prompt_tokens == 220
    assert llm.completion_tokens == 40
    assert llm.model == "gpt-4o"
    assert llm.cost_usd > 0

    tool = next(s for s in t.spans if s.kind == "tool")
    assert tool.name == "lookup_order"


def test_openai_agents_processor_records_error():
    _install_fake_agents()
    from agentlens import Tracer
    from agentlens.adapters.openai_agents import make_agentlens_processor

    tracer = Tracer()
    proc = make_agentlens_processor(tracer)
    proc.on_trace_start(FakeTrace())

    bad = FakeSpan("s-bad", "function", name="flaky")
    bad.error = RuntimeError("boom")
    proc.on_span_start(bad)
    proc.on_span_end(bad)
    proc.on_trace_end(FakeTrace())

    t = tracer.finished[-1]
    tool = next(s for s in t.spans if s.kind == "tool")
    assert tool.status == "error"
    assert t.status == "error"
