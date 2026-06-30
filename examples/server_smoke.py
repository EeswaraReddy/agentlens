"""Smoke test: end-to-end against a running AgentLens server.

Spins up a Tracer, runs a small instrumented agent, ships the trace to the
server, then queries it back. Use this to verify a deployment.

    python examples/server_smoke.py --url http://localhost:8810 --key al_...
"""

import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentlens import Tracer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8810")
    p.add_argument("--key", required=True)
    args = p.parse_args()

    tr = Tracer()
    with tr.trace("demo-agent", user="alice"):
        with tr.agent("router"):
            with tr.llm("classify", model="gpt-4o-mini") as s:
                s.record_tokens(prompt=120, completion=18, model="gpt-4o-mini")
                s.set_output(intent="order_status")
            with tr.tool("lookup_order", order_id="A1029") as s:
                s.set_output(status="shipped")
    ids = tr.export_to(args.url, api_key=args.key)
    print("shipped trace_ids:", ids)

    # query it back
    def get(path):
        req = urllib.request.Request(args.url + path, headers={"X-API-Key": args.key})
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read().decode("utf-8"))

    lst = get("/api/v1/traces")
    print("traces:", lst["total"], "first:", lst["items"][0]["name"])
    det = get("/api/v1/traces/" + lst["items"][0]["trace_id"])
    kinds = sorted({s["kind"] for s in det["spans"]})
    print("spans:", len(det["spans"]), "kinds:", kinds)
    stats = get("/api/v1/stats")
    print(f"stats: traces={stats['total_traces']} "
          f"tokens={stats['total_tokens']} cost=${stats['total_cost_usd']:.6f}")


if __name__ == "__main__":
    main()
