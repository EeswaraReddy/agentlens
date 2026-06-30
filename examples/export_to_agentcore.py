"""
Export an AgentLens trace to Amazon Bedrock AgentCore observability.

AgentCore observability is OpenTelemetry-based, so traces leave as OTLP spans.
With no OTLP endpoint configured this writes OTLP-shaped JSON-lines locally so
you can see exactly what would be sent — fully offline.

    python examples/export_to_agentcore.py
    # then inspect runs/agentcore_export.jsonl

To send for real, set OTEL_EXPORTER_OTLP_ENDPOINT to your AgentCore/collector
endpoint and configure AWS credentials.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentlens import Tracer
from agentlens.export import to_otel_spans, AgentCoreExporter


def build_sample_trace(tracer: Tracer):
    with tracer.trace("invoice-agent", env="prod"):
        with tracer.agent("router"):
            with tracer.llm("classify", model="claude-3-5-sonnet") as s:
                s.record_tokens(prompt=210, completion=24, model="claude-3-5-sonnet")
                s.set_output(intent="invoice_query")
            with tracer.tool("fetch_invoice", invoice_id="INV-77") as s:
                s.set_output(total="$420.00", status="paid")


def main():
    tracer = Tracer()
    build_sample_trace(tracer)
    trace = tracer.finished[-1]

    print("OpenTelemetry spans (GenAI conventions):")
    print(json.dumps(to_otel_spans(trace), indent=2)[:900] + "\n...\n")

    exporter = AgentCoreExporter()  # offline fallback unless OTLP endpoint set
    info = exporter.export(trace)
    print("Export result:", info)
    if info["transport"] == "file":
        print(f"\nWrote OTLP JSON-lines to {info['path']}")
        print("Set OTEL_EXPORTER_OTLP_ENDPOINT to ship to AgentCore for real.")


if __name__ == "__main__":
    main()
