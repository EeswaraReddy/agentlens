"""
Real OpenTelemetry export — ship AgentLens traces to any OTLP backend.

Works with anything OTLP-compatible:
  - Grafana Tempo / Jaeger / SigNoz (run locally with docker)
  - AWS X-Ray (via the AWS Distro for OpenTelemetry collector)
  - Amazon Bedrock AgentCore observability
  - Honeycomb, Datadog, Langfuse, etc.

Setup:
    pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc

Run with a local collector (defaults to localhost:4317):
    docker run -d --name otel -p 4317:4317 -p 4318:4318 otel/opentelemetry-collector

    python examples/otlp_bridge.py
    # or point elsewhere:
    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 python examples/otlp_bridge.py

Run without a collector to print spans to the console:
    python examples/otlp_bridge.py --console
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentlens import Tracer
from agentlens.export import install_otlp_bridge, emit_to_otlp


def build_sample_trace(tracer: Tracer) -> None:
    with tracer.trace("invoice-agent", env="prod"):
        with tracer.agent("router"):
            with tracer.llm("classify", model="claude-3-5-sonnet") as s:
                s.record_tokens(prompt=180, completion=22, model="claude-3-5-sonnet")
                s.set_output(intent="invoice_query")
            with tracer.tool("fetch_invoice", invoice_id="INV-77") as s:
                s.set_output(total="$420.00", status="paid")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--console", action="store_true",
                        help="Print spans to stdout instead of OTLP")
    parser.add_argument("--endpoint", default=None,
                        help="OTLP endpoint (default: OTEL_EXPORTER_OTLP_ENDPOINT or localhost:4317)")
    args = parser.parse_args()

    endpoint = args.endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not args.console and not endpoint:
        endpoint = "http://localhost:4317"

    try:
        provider = install_otlp_bridge(
            service_name="agentlens-demo",
            endpoint=None if args.console else endpoint,
            console=args.console,
        )
    except ImportError as exc:
        print(exc)
        return

    tracer = Tracer()
    build_sample_trace(tracer)
    trace = tracer.finished[-1]

    n = emit_to_otlp(trace, service_name="agentlens-demo")
    print(f"emitted {n} OTel spans  (service=agentlens-demo)")
    if endpoint and not args.console:
        print(f"shipped to OTLP endpoint: {endpoint}")
    time.sleep(0.5)  # let the batch processor flush
    provider.shutdown()


if __name__ == "__main__":
    main()
