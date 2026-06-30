"""
AgentLens — observability + eval harness for AI agents.

Trace every LLM and tool call, assert behavior with declarative evals, and
inspect runs in a local web viewer. Dependency-free core; works in any Python
environment and exports to OpenTelemetry / Amazon Bedrock AgentCore.
"""

from .trace import Trace, Span
from .tracer import Tracer
from . import evals
from . import pricing
from . import export

__version__ = "0.1.0"

__all__ = ["Tracer", "Trace", "Span", "evals", "pricing", "export", "__version__"]
