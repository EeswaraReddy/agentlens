"""
Tests for the Strands adapter using a faked `strands.hooks` module.

We don't install the real SDK in CI, so we inject a minimal stand-in that
provides the HookProvider base + event classes the adapter imports. This
verifies the Before*/After* callback wiring maps correctly onto AgentLens spans.
"""

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _install_fake_strands():
    """Create a fake `strands.hooks` module with the needed symbols."""
    strands = types.ModuleType("strands")
    hooks = types.ModuleType("strands.hooks")

    class HookProvider:  # minimal base
        pass

    class HookRegistry:
        pass

    # event marker classes
    for cls_name in [
        "BeforeInvocationEvent", "AfterInvocationEvent",
        "BeforeModelCallEvent", "AfterModelCallEvent",
        "BeforeToolCallEvent", "AfterToolCallEvent",
    ]:
        setattr(hooks, cls_name, type(cls_name, (), {}))

    hooks.HookProvider = HookProvider
    hooks.HookRegistry = HookRegistry
    strands.hooks = hooks
    sys.modules["strands"] = strands
    sys.modules["strands.hooks"] = hooks
    return hooks


class FakeRegistry:
    """Collects callbacks like the real HookRegistry, keyed by event type."""
    def __init__(self):
        self.callbacks = {}

    def add_callback(self, event_type, fn):
        self.callbacks.setdefault(event_type, []).append(fn)

    def fire(self, event_type, event):
        for fn in self.callbacks.get(event_type, []):
            fn(event)


class Obj:
    """Generic attribute bag for fake events."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_strands_adapter_maps_lifecycle_to_spans():
    hooks = _install_fake_strands()
    from agentlens import Tracer
    from agentlens.adapters.strands import make_agentlens_hook

    tracer = Tracer()
    hook = make_agentlens_hook(tracer, name="support", model="gpt-4o")

    reg = FakeRegistry()
    hook.register_hooks(reg)

    # Simulate the Strands lifecycle:
    reg.fire(hooks.BeforeInvocationEvent, Obj(agent=Obj(name="support")))

    reg.fire(hooks.BeforeModelCallEvent, Obj())
    reg.fire(hooks.AfterModelCallEvent,
             Obj(result=Obj(usage={"inputTokens": 200, "outputTokens": 60}), exception=None))

    reg.fire(hooks.BeforeToolCallEvent,
             Obj(tool_use={"toolUseId": "t1", "name": "lookup_order", "input": {"order_id": "A1"}}))
    reg.fire(hooks.AfterToolCallEvent,
             Obj(tool_use={"toolUseId": "t1", "name": "lookup_order"},
                 result={"status": "success", "content": "shipped"}, exception=None))

    reg.fire(hooks.AfterInvocationEvent, Obj(agent=Obj(name="support")))

    trace = tracer.finished[-1]
    kinds = sorted(s.kind for s in trace.spans)
    assert kinds == ["agent", "llm", "tool"]

    llm = next(s for s in trace.spans if s.kind == "llm")
    assert llm.prompt_tokens == 200
    assert llm.completion_tokens == 60
    assert llm.cost_usd > 0          # gpt-4o priced

    tool = next(s for s in trace.spans if s.kind == "tool")
    assert tool.name == "lookup_order"
    assert tool.inputs == {"order_id": "A1"}
    assert tool.status == "ok"
    assert trace.status == "ok"


def test_strands_adapter_records_tool_error():
    hooks = _install_fake_strands()
    from agentlens import Tracer
    from agentlens.adapters.strands import make_agentlens_hook

    tracer = Tracer()
    hook = make_agentlens_hook(tracer, name="svc")
    reg = FakeRegistry()
    hook.register_hooks(reg)

    reg.fire(hooks.BeforeInvocationEvent, Obj(agent=Obj(name="svc")))
    reg.fire(hooks.BeforeToolCallEvent, Obj(tool_use={"toolUseId": "t9", "name": "flaky"}))
    reg.fire(hooks.AfterToolCallEvent,
             Obj(tool_use={"toolUseId": "t9", "name": "flaky"},
                 result={"status": "error", "content": "boom"}, exception=None))
    reg.fire(hooks.AfterInvocationEvent, Obj(agent=Obj(name="svc")))

    trace = tracer.finished[-1]
    tool = next(s for s in trace.spans if s.kind == "tool")
    assert tool.status == "error"
    assert trace.status == "error"
