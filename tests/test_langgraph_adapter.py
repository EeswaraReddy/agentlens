"""
Tests for the LangGraph/LangChain adapter using a faked langchain_core module.
"""

import os
import sys
import types
from uuid import uuid4

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _install_fake_langchain():
    lc = types.ModuleType("langchain_core")
    cb = types.ModuleType("langchain_core.callbacks")

    class BaseCallbackHandler:
        pass

    cb.BaseCallbackHandler = BaseCallbackHandler
    lc.callbacks = cb
    sys.modules["langchain_core"] = lc
    sys.modules["langchain_core.callbacks"] = cb


class FakeLLMResult:
    def __init__(self, text, prompt_tokens, completion_tokens, model="llama-3.3-70b-versatile"):
        gen = types.SimpleNamespace(text=text, message=None)
        self.generations = [[gen]]
        self.llm_output = {
            "token_usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
            "model_name": model,
        }


def test_langgraph_adapter_maps_callbacks_to_spans():
    _install_fake_langchain()
    from agentlens import Tracer
    from agentlens.adapters.langgraph import make_langgraph_handler

    tracer = Tracer()
    h = make_langgraph_handler(tracer, name="graph")

    chain_id = uuid4()
    llm_id = uuid4()
    tool_id = uuid4()

    h.on_chain_start({"name": "router"}, {"input": "hi"}, run_id=chain_id)
    h.on_llm_start({}, ["classify this"], run_id=llm_id)
    h.on_llm_end(FakeLLMResult("intent=order", 100, 20), run_id=llm_id)
    h.on_tool_start({"name": "lookup_order"}, "A1029", run_id=tool_id)
    h.on_tool_end("shipped", run_id=tool_id)
    h.on_chain_end({"output": "done"}, run_id=chain_id)

    trace = tracer.finished[-1]
    kinds = sorted(s.kind for s in trace.spans)
    assert kinds == ["agent", "llm", "tool"]

    llm = next(s for s in trace.spans if s.kind == "llm")
    assert llm.prompt_tokens == 100
    assert llm.completion_tokens == 20
    assert llm.model == "llama-3.3-70b-versatile"
    assert llm.cost_usd > 0

    tool = next(s for s in trace.spans if s.kind == "tool")
    assert tool.name == "lookup_order"
    assert "shipped" in str(tool.outputs)


def test_langgraph_adapter_records_tool_error():
    _install_fake_langchain()
    from agentlens import Tracer
    from agentlens.adapters.langgraph import make_langgraph_handler

    tracer = Tracer()
    h = make_langgraph_handler(tracer, name="graph")
    chain_id = uuid4()
    tool_id = uuid4()

    h.on_chain_start({"name": "router"}, {}, run_id=chain_id)
    h.on_tool_start({"name": "flaky"}, "x", run_id=tool_id)
    h.on_tool_error(RuntimeError("boom"), run_id=tool_id)
    h.on_chain_end({}, run_id=chain_id)

    trace = tracer.finished[-1]
    tool = next(s for s in trace.spans if s.kind == "tool")
    assert tool.status == "error"
    assert trace.status == "error"
