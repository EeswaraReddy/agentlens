"""Tests for AgentLens core: tracing, cost, evals."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentlens import Tracer, evals as E
from agentlens.pricing import estimate_cost


def test_trace_collects_spans():
    tr = Tracer()
    with tr.trace("t"):
        with tr.agent("a"):
            with tr.tool("lookup"):
                pass
    trace = tr.finished[-1]
    assert trace.status == "ok"
    assert len(trace.spans) == 2
    assert trace.counts() == {"agent": 1, "tool": 1}


def test_parent_child_nesting():
    tr = Tracer()
    with tr.trace("t"):
        with tr.agent("a") as _:
            with tr.tool("lookup"):
                pass
    spans = tr.finished[-1].spans
    agent_span = next(s for s in spans if s.kind == "agent")
    tool_span = next(s for s in spans if s.kind == "tool")
    assert tool_span.parent_id == agent_span.span_id
    assert agent_span.parent_id is None


def test_token_and_cost_accounting():
    tr = Tracer()
    with tr.trace("t"):
        with tr.llm("call", model="gpt-4o") as span:
            span.record_tokens(prompt=1000, completion=1000, model="gpt-4o")
    trace = tr.finished[-1]
    assert trace.total_tokens == 2000
    # gpt-4o: 2.50 prompt + 10.00 completion per 1M
    expected = estimate_cost("gpt-4o", 1000, 1000)
    assert abs(trace.total_cost_usd - expected) < 1e-9
    assert trace.total_cost_usd > 0


def test_error_marks_trace_failed():
    tr = Tracer()
    try:
        with tr.trace("t"):
            with tr.tool("boom"):
                raise ValueError("kaboom")
    except ValueError:
        pass
    trace = tr.finished[-1]
    assert trace.status == "error"
    assert any(s.status == "error" and s.error == "kaboom" for s in trace.spans)


def _good_trace():
    tr = Tracer()
    with tr.trace("support"):
        with tr.tool("lookup_order"):
            pass
        with tr.tool("send_reply"):
            pass
    return tr.finished[-1]


def test_evals_pass():
    trace = _good_trace()
    suite = E.Suite("s", [
        E.succeeded(),
        E.called_tool("lookup_order"),
        E.never_called_tool("issue_refund"),
        E.tool_before("lookup_order", "send_reply"),
    ])
    report = suite.run(trace)
    assert report.passed
    assert report.num_passed == 4


def test_evals_catch_violation():
    trace = _good_trace()
    suite = E.Suite("s", [
        E.never_called_tool("lookup_order"),   # this SHOULD fail
        E.called_tool("issue_refund"),         # this SHOULD fail
    ])
    report = suite.run(trace)
    assert not report.passed
    assert report.num_passed == 0


def test_ordering_check_detects_wrong_order():
    tr = Tracer()
    with tr.trace("t"):
        with tr.tool("send_reply"):
            pass
        with tr.tool("lookup_order"):
            pass
    trace = tr.finished[-1]
    report = E.Suite("s", [E.tool_before("lookup_order", "send_reply")]).run(trace)
    assert not report.passed


def test_event_marker():
    tr = Tracer()
    with tr.trace("t"):
        tr.event("human_approval", approver="alice")
    trace = tr.finished[-1]
    assert E.Suite("s", [E.had_event("human_approval")]).run(trace).passed
