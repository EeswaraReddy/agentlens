"""Tests for the traced Groq client and the LLM-as-judge eval."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentlens import Tracer, evals as E
from agentlens.providers.groq import GroqClient, Completion


def test_groq_client_traces_call(monkeypatch):
    tracer = Tracer()
    client = GroqClient(api_key="fake-key", model="llama-3.1-8b-instant", tracer=tracer)

    # Mock the HTTP layer so no network is hit.
    fake_body = {
        "choices": [{"message": {"content": "Hi there friend"}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4},
    }
    monkeypatch.setattr(client, "_post", lambda payload: fake_body)

    with tracer.trace("chat"):
        comp = client.complete("Say hi")

    assert comp.text == "Hi there friend"
    assert comp.total_tokens == 16

    trace = tracer.finished[-1]
    llm = next(s for s in trace.spans if s.kind == "llm")
    assert llm.prompt_tokens == 12
    assert llm.completion_tokens == 4
    assert llm.model == "llama-3.1-8b-instant"
    assert llm.cost_usd > 0
    assert llm.outputs.get("text") == "Hi there friend"


def test_groq_client_without_active_trace_still_works(monkeypatch):
    client = GroqClient(api_key="fake-key")
    monkeypatch.setattr(client, "_post", lambda payload: {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    })
    comp = client.complete("hi")
    assert comp.text == "ok"


def test_groq_missing_key_raises():
    import pytest
    from agentlens.providers.groq import GroqError
    client = GroqClient(api_key=None)
    # ensure env var doesn't accidentally satisfy it
    client.api_key = None
    with pytest.raises(GroqError):
        client._post({"x": 1})


class FakeJudge:
    """Stands in for a Groq client in llm_judge tests."""
    def __init__(self, score: float):
        self._score = score

    def complete(self, prompt, system=None, **kwargs):
        class R:
            text = '{"score": %s, "reason": "looks good"}' % self._score
        return R()


def _trace_with_reply():
    tr = Tracer()
    with tr.trace("support"):
        with tr.tool("lookup_order") as s:
            s.set_output(status="shipped")
    return tr.finished[-1]


def test_llm_judge_passes_above_threshold():
    trace = _trace_with_reply()
    check = E.llm_judge("agent answered correctly", client=FakeJudge(0.9), threshold=0.7)
    report = E.Suite("s", [check]).run(trace)
    assert report.passed
    assert "score=0.90" in report.results[0].detail


def test_llm_judge_fails_below_threshold():
    trace = _trace_with_reply()
    check = E.llm_judge("agent answered correctly", client=FakeJudge(0.3), threshold=0.7)
    report = E.Suite("s", [check]).run(trace)
    assert not report.passed


def test_llm_judge_handles_code_fenced_json():
    class FencedJudge:
        def complete(self, prompt, system=None, **kwargs):
            class R:
                text = '```json\n{"score": 0.8, "reason": "ok"}\n```'
            return R()
    trace = _trace_with_reply()
    check = E.llm_judge("rubric", client=FencedJudge(), threshold=0.5)
    assert E.Suite("s", [check]).run(trace).passed
