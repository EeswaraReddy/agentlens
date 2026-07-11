"""
AgentLens MCP Server
====================
Exposes AgentLens as an MCP (Model Context Protocol) tool server.
Any MCP-compatible client — Claude Desktop, Kiro, Cursor, or another
agent — can discover and invoke these tools to trace, query, and eval
AI agent runs.

Tools exposed:
  ingest_trace        push a completed trace into AgentLens
  list_traces         query saved traces with filters
  get_trace           full detail on one trace (all spans)
  get_stats           aggregate stats for a project
  run_evals           run a declarative eval suite against a trace
  list_local_traces   read traces from local runs/ directory (no server needed)

Usage (stdio — works with any MCP client):
  python -m agentlens.mcp_server

Usage (HTTP SSE — for remote clients):
  python -m agentlens.mcp_server --transport sse --port 8801

Config (environment):
  AGENTLENS_SERVER_URL   base URL of a running AgentLens server
                         e.g. http://localhost:8800  (default)
  AGENTLENS_API_KEY      API key for that server
                         (auto-read from agentlens.db if local)
  AGENTLENS_RUNS_DIR     directory for local JSON traces (default: runs)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
import sys
import urllib.error
import urllib.request
from typing import Any

# ── resolve package path when run directly ──────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── MCP SDK import (mcp package) ────────────────────────────────────────────
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    print(
        "MCP SDK not installed.\n"
        "Install with:  pip install mcp\n"
        "  or:          pip install 'mcp[cli]'",
        file=sys.stderr,
    )
    sys.exit(1)


# ── config helpers ───────────────────────────────────────────────────────────

def _load_env():
    """Load .env from project root if present."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(root, ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

_load_env()

SERVER_URL  = os.getenv("AGENTLENS_SERVER_URL", "http://localhost:8800")
RUNS_DIR    = os.getenv("AGENTLENS_RUNS_DIR", "runs")


def _api_key() -> str | None:
    """Return API key: env var → agentlens.db → None."""
    k = os.getenv("AGENTLENS_API_KEY")
    if k:
        return k
    # try to read from local sqlite db
    for db_path in ["agentlens.db", "agentlens/agentlens.db"]:
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                row = conn.execute(
                    "SELECT api_key FROM projects WHERE name='default' LIMIT 1"
                ).fetchone()
                conn.close()
                if row:
                    return row[0]
            except Exception:
                pass
    return None


# ── HTTP helper ──────────────────────────────────────────────────────────────

def _request(method: str, path: str, body: dict | None = None) -> dict:
    """Make an authenticated request to the AgentLens server."""
    key = _api_key()
    if not key:
        return {"error": "No API key found. Set AGENTLENS_API_KEY or run agentlens server first."}

    url = SERVER_URL.rstrip("/") + path
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={
            "Content-Type":  "application/json",
            "X-API-Key":     key,
            "User-Agent":    "agentlens-mcp/0.1",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        msg = exc.read().decode("utf-8", "replace")
        return {"error": f"HTTP {exc.code}: {msg}"}
    except urllib.error.URLError as exc:
        return {"error": f"Cannot reach server at {SERVER_URL}: {exc.reason}. "
                         "Is 'agentlens server' running?"}


# ── local trace helpers ──────────────────────────────────────────────────────

def _local_trace_paths() -> list[str]:
    runs = RUNS_DIR if os.path.isabs(RUNS_DIR) else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), RUNS_DIR
    )
    return sorted(glob.glob(os.path.join(runs, "*.json")), reverse=True)


def _load_local_trace(trace_id: str) -> dict | None:
    for path in _local_trace_paths():
        if trace_id in os.path.basename(path):
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return None


# ── eval runner ──────────────────────────────────────────────────────────────

def _run_evals_on_trace(trace_dict: dict, checks: list[str]) -> dict:
    """Run built-in eval checks against a trace dict."""
    from agentlens.trace import Trace, Span
    from agentlens import evals as E

    # Reconstruct Trace from dict
    spans = []
    for s in trace_dict.get("spans", []):
        sp = Span(
            name=s.get("name", ""),
            kind=s.get("kind", "agent"),
            span_id=s.get("span_id", ""),
            parent_id=s.get("parent_id"),
            status=s.get("status", "ok"),
            error=s.get("error"),
            inputs=s.get("inputs", {}),
            outputs=s.get("outputs", {}),
            model=s.get("model"),
            prompt_tokens=s.get("prompt_tokens", 0),
            completion_tokens=s.get("completion_tokens", 0),
            cost_usd=s.get("cost_usd", 0.0),
            duration_ms=s.get("duration_ms"),
            attributes=s.get("attributes", {}),
        )
        spans.append(sp)

    trace = Trace(
        name=trace_dict.get("name", ""),
        trace_id=trace_dict.get("trace_id", ""),
        status=trace_dict.get("status", "ok"),
        duration_ms=trace_dict.get("duration_ms"),
        spans=spans,
        metadata=trace_dict.get("metadata", {}),
    )

    # Build check list
    check_map = {
        "succeeded":    E.succeeded(),
        "max_cost_1c":  E.max_cost(0.01),
        "max_cost_5c":  E.max_cost(0.05),
        "max_cost_10c": E.max_cost(0.10),
        "fast":         E.max_duration_ms(5000),
        "under_500tok": E.max_tokens(500),
    }
    # parse checks like "called_tool:lookup_order", "never_called_tool:refund"
    eval_checks = []
    for c in checks:
        if c in check_map:
            eval_checks.append(check_map[c])
        elif c.startswith("called_tool:"):
            eval_checks.append(E.called_tool(c.split(":", 1)[1]))
        elif c.startswith("never_called_tool:"):
            eval_checks.append(E.never_called_tool(c.split(":", 1)[1]))
        elif c.startswith("tool_before:"):
            parts = c.split(":", 1)[1].split(",")
            if len(parts) == 2:
                eval_checks.append(E.tool_before(parts[0].strip(), parts[1].strip()))
        elif c.startswith("max_cost:"):
            eval_checks.append(E.max_cost(float(c.split(":", 1)[1])))
        elif c.startswith("max_tokens:"):
            eval_checks.append(E.max_tokens(int(c.split(":", 1)[1])))
        elif c.startswith("max_ms:"):
            eval_checks.append(E.max_duration_ms(float(c.split(":", 1)[1])))
        elif c.startswith("had_event:"):
            eval_checks.append(E.had_event(c.split(":", 1)[1]))

    if not eval_checks:
        eval_checks = [E.succeeded(), E.max_cost(0.05)]

    suite  = E.Suite("mcp-eval", eval_checks)
    report = suite.run(trace)
    return {
        "passed":   report.passed,
        "score":    f"{report.num_passed}/{len(report.results)}",
        "results":  [{"name": r.name, "passed": r.passed, "detail": r.detail}
                     for r in report.results],
        "summary":  report.summary(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MCP Server
# ══════════════════════════════════════════════════════════════════════════════

def build_server() -> Server:
    server = Server("agentlens")

    # ── tool definitions ─────────────────────────────────────────────────────
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="list_local_traces",
                description=(
                    "List all AI agent traces saved locally in the runs/ directory. "
                    "Returns trace_id, name, status, tokens, cost for each run. "
                    "No server required. Use this first to see what traces exist."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Max traces to return (default 20)",
                            "default": 20,
                        },
                        "name_filter": {
                            "type": "string",
                            "description": "Optional: filter by trace name substring",
                        },
                    },
                },
            ),
            Tool(
                name="get_local_trace",
                description=(
                    "Get the full span tree for a specific local trace. "
                    "Shows every LLM call, tool call, event — with inputs, outputs, "
                    "tokens, cost, and timing. Use list_local_traces first to get trace_id."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "trace_id": {
                            "type": "string",
                            "description": "The trace_id to inspect (from list_local_traces)",
                        },
                    },
                    "required": ["trace_id"],
                },
            ),
            Tool(
                name="run_evals",
                description=(
                    "Run behavioral eval checks against a saved trace. "
                    "Checks: 'succeeded', 'max_cost_1c', 'max_cost_5c', 'fast', 'under_500tok', "
                    "'called_tool:<name>', 'never_called_tool:<name>', "
                    "'tool_before:<a>,<b>', 'max_cost:<usd>', 'max_tokens:<n>', "
                    "'max_ms:<ms>', 'had_event:<name>'. "
                    "Returns pass/fail per check and overall result."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "trace_id": {
                            "type": "string",
                            "description": "trace_id to evaluate",
                        },
                        "checks": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "List of checks. Examples: "
                                "['succeeded', 'called_tool:lookup_order', "
                                "'never_called_tool:issue_refund', 'max_cost:0.01']"
                            ),
                        },
                    },
                    "required": ["trace_id"],
                },
            ),
            Tool(
                name="get_trace_cost_breakdown",
                description=(
                    "Get a cost and token breakdown per span for a trace. "
                    "Shows which LLM calls are expensive and how tokens are distributed. "
                    "Useful for optimizing agent cost."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "trace_id": {"type": "string"},
                    },
                    "required": ["trace_id"],
                },
            ),
            Tool(
                name="list_server_traces",
                description=(
                    "Query traces from the running AgentLens server (requires server). "
                    "Supports search, status filter, and pagination."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "q":        {"type": "string", "description": "Search in trace name"},
                        "status":   {"type": "string", "enum": ["ok", "error"]},
                        "page":     {"type": "integer", "default": 1},
                        "page_size":{"type": "integer", "default": 25},
                    },
                },
            ),
            Tool(
                name="get_server_stats",
                description=(
                    "Get aggregate stats from the AgentLens server: total traces, "
                    "tokens, cost, error rate, top models, traces per day."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "description": "Lookback window in days (default 14)",
                            "default": 14,
                        },
                    },
                },
            ),
        ]

    # ── tool handlers ─────────────────────────────────────────────────────────
    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        result = _dispatch(name, arguments or {})
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    return server


def _dispatch(name: str, args: dict) -> Any:
    """Route tool calls to implementations."""

    if name == "list_local_traces":
        limit  = int(args.get("limit", 20))
        needle = (args.get("name_filter") or "").lower()
        paths  = _local_trace_paths()
        rows   = []
        for p in paths:
            if len(rows) >= limit:
                break
            try:
                with open(p, encoding="utf-8") as f:
                    t = json.load(f)
            except Exception:
                continue
            if needle and needle not in t.get("name", "").lower():
                continue
            s = t.get("summary", {})
            rows.append({
                "trace_id":    t.get("trace_id"),
                "name":        t.get("name"),
                "status":      t.get("status"),
                "duration_ms": t.get("duration_ms"),
                "spans":       s.get("spans", 0),
                "tokens":      s.get("total_tokens", 0),
                "cost_usd":    s.get("total_cost_usd", 0.0),
                "errors":      s.get("errors", 0),
            })
        return {"traces": rows, "count": len(rows), "total_available": len(paths)}

    if name == "get_local_trace":
        tid   = args.get("trace_id", "")
        trace = _load_local_trace(tid)
        if not trace:
            return {"error": f"Trace '{tid}' not found in {RUNS_DIR}/"}
        # format as readable span tree
        spans_out = []
        for sp in trace.get("spans", []):
            spans_out.append({
                "span_id":    sp.get("span_id"),
                "parent_id":  sp.get("parent_id"),
                "kind":       sp.get("kind"),
                "name":       sp.get("name"),
                "status":     sp.get("status"),
                "duration_ms":sp.get("duration_ms"),
                "tokens":     sp.get("total_tokens", 0),
                "cost_usd":   sp.get("cost_usd", 0.0),
                "model":      sp.get("model"),
                "inputs":     sp.get("inputs", {}),
                "outputs":    sp.get("outputs", {}),
                "error":      sp.get("error"),
            })
        return {
            "trace_id":   trace.get("trace_id"),
            "name":       trace.get("name"),
            "status":     trace.get("status"),
            "duration_ms":trace.get("duration_ms"),
            "metadata":   trace.get("metadata", {}),
            "summary":    trace.get("summary", {}),
            "spans":      spans_out,
        }

    if name == "run_evals":
        tid    = args.get("trace_id", "")
        checks = args.get("checks") or ["succeeded"]
        trace  = _load_local_trace(tid)
        if not trace:
            # try server
            result = _request("GET", f"/api/v1/traces/{tid}")
            if "error" in result:
                return {"error": f"Trace '{tid}' not found locally or on server."}
            trace = result
        return _run_evals_on_trace(trace, checks)

    if name == "get_trace_cost_breakdown":
        tid   = args.get("trace_id", "")
        trace = _load_local_trace(tid)
        if not trace:
            return {"error": f"Trace '{tid}' not found."}
        spans = trace.get("spans", [])
        llm_spans = [s for s in spans if s.get("kind") == "llm"]
        tool_spans = [s for s in spans if s.get("kind") == "tool"]
        total_cost   = sum(s.get("cost_usd", 0) for s in spans)
        total_tokens = sum(s.get("total_tokens", 0) for s in spans)
        breakdown = []
        for sp in llm_spans:
            breakdown.append({
                "name":          sp.get("name"),
                "model":         sp.get("model"),
                "prompt_tokens": sp.get("prompt_tokens", 0),
                "completion_tokens": sp.get("completion_tokens", 0),
                "cost_usd":      sp.get("cost_usd", 0.0),
                "pct_of_total":  round(sp.get("cost_usd", 0) / total_cost * 100, 1)
                                 if total_cost > 0 else 0,
                "duration_ms":   sp.get("duration_ms"),
            })
        return {
            "trace_id":      tid,
            "total_cost_usd":round(total_cost, 6),
            "total_tokens":  total_tokens,
            "llm_calls":     len(llm_spans),
            "tool_calls":    len(tool_spans),
            "breakdown":     sorted(breakdown, key=lambda x: x["cost_usd"], reverse=True),
            "tip": (
                f"Most expensive span: {breakdown[0]['name']} "
                f"({breakdown[0]['pct_of_total']}% of total cost)"
                if breakdown else "No LLM spans found."
            ),
        }

    if name == "list_server_traces":
        params = []
        if args.get("q"):          params.append(f"q={args['q']}")
        if args.get("status"):     params.append(f"status={args['status']}")
        if args.get("page"):       params.append(f"page={args['page']}")
        if args.get("page_size"):  params.append(f"page_size={args['page_size']}")
        qs = "?" + "&".join(params) if params else ""
        return _request("GET", f"/api/v1/traces{qs}")

    if name == "get_server_stats":
        days = int(args.get("days", 14))
        return _request("GET", f"/api/v1/stats?days={days}")

    return {"error": f"Unknown tool: {name}"}


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

async def _run_stdio():
    server = build_server()
    async with stdio_server() as streams:
        await server.run(streams[0], streams[1], server.create_initialization_options())


async def _run_sse(port: int):
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Route, Mount
    import uvicorn

    server = build_server()
    sse    = SseServerTransport("/messages")

    async def handle_sse(request):
        async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
            await server.run(streams[0], streams[1], server.create_initialization_options())

    starlette_app = Starlette(routes=[
        Route("/sse", endpoint=handle_sse),
        Mount("/messages", app=sse.handle_post_message),
    ])
    print(f"AgentLens MCP server → http://localhost:{port}/sse", file=sys.stderr)
    config = uvicorn.Config(starlette_app, host="0.0.0.0", port=port, log_level="info")
    await uvicorn.Server(config).serve()


def main():
    parser = argparse.ArgumentParser(description="AgentLens MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio",
                        help="Transport: stdio (default) or sse (HTTP)")
    parser.add_argument("--port", type=int, default=8801,
                        help="Port for SSE transport (default 8801)")
    args = parser.parse_args()

    import asyncio
    if args.transport == "sse":
        asyncio.run(_run_sse(args.port))
    else:
        asyncio.run(_run_stdio())


if __name__ == "__main__":
    main()
