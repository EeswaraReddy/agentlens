"""
Auto-instrument a real Strands agent with AgentLens — one line, zero manual spans.

Requires the Strands SDK and AWS Bedrock access:
    pip install strands-agents
    # configure AWS credentials / region

Run:
    python examples/strands_agent.py

If Strands isn't installed, this prints how to use it instead of failing.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentlens import Tracer, evals as E


USAGE = '''\
from strands import Agent, tool
from agentlens import Tracer
from agentlens.adapters.strands import make_agentlens_hook

@tool
def lookup_order(order_id: str) -> str:
    return "shipped"

tracer = Tracer()
hook = make_agentlens_hook(tracer, name="support", model="claude-3-5-sonnet")

agent = Agent(tools=[lookup_order], hooks=[hook])
agent("Where is order A1029?")

tracer.save("runs")          # every model + tool call captured automatically
print(tracer.finished[-1].to_dict()["summary"])
'''


def main():
    try:
        import strands  # noqa: F401
    except ImportError:
        print("Strands SDK not installed. Install with:  pip install strands-agents\n")
        print("Then instrument any agent in one line:\n")
        print(USAGE)
        return

    from strands import Agent, tool
    from agentlens.adapters.strands import make_agentlens_hook

    @tool
    def lookup_order(order_id: str) -> str:
        """Look up an order's status."""
        return "shipped, arriving in 2 days"

    tracer = Tracer()
    hook = make_agentlens_hook(tracer, name="support", model="claude-3-5-sonnet")
    agent = Agent(tools=[lookup_order], hooks=[hook])

    agent("Where is my order A1029?")

    trace = tracer.finished[-1]
    print(trace.to_dict()["summary"])
    report = E.Suite("support guardrails", [
        E.succeeded(),
        E.called_tool("lookup_order"),
        E.max_cost(0.10),
    ]).run(trace)
    print(report.summary())
    tracer.save("runs")


if __name__ == "__main__":
    main()
