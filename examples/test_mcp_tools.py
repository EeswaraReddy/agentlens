"""Quick test of the MCP tool dispatch — no transport needed."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentlens.mcp_server import _dispatch

print("=== Tool: list_local_traces ===")
r = _dispatch("list_local_traces", {"limit": 5})
for t in r["traces"]:
    print(f"  {t['trace_id']}  {t['name']:<25}  {t['tokens']} tok  ${t['cost_usd']:.6f}")
print(f"  ... {r['total_available']} total traces available")

trace_id = r["traces"][0]["trace_id"]

print()
print("=== Tool: get_local_trace (span tree) ===")
detail = _dispatch("get_local_trace", {"trace_id": trace_id})
print(f"  name: {detail['name']}  status: {detail['status']}")
for sp in detail["spans"]:
    pad = "     " if sp.get("parent_id") else "  "
    icon = {"agent":"🤖","llm":"🧠","tool":"🔧","event":"📍"}.get(sp["kind"],"·")
    print(f"  {pad}{icon} [{sp['kind']:5}] {sp['name']}  {sp['tokens']}tok  ${sp['cost_usd']:.6f}")
    if sp.get("outputs"):
        first = next(iter(sp["outputs"].items()))
        print(f"  {pad}       └ {first[0]}: {str(first[1])[:60]}")

print()
print("=== Tool: get_trace_cost_breakdown ===")
b = _dispatch("get_trace_cost_breakdown", {"trace_id": trace_id})
print(f"  total: {b['total_tokens']} tokens  ${b['total_cost_usd']}")
for sp in b["breakdown"]:
    print(f"  {sp['name']:<30} {sp['pct_of_total']:>5}%  ${sp['cost_usd']:.6f}  {sp['duration_ms']:.0f}ms")
print(f"  💡 {b['tip']}")

print()
print("=== Tool: run_evals ===")
e = _dispatch("run_evals", {
    "trace_id": trace_id,
    "checks": ["succeeded", "max_cost:0.05", "called_tool:check_stock",
               "never_called_tool:issue_refund", "max_ms:5000"]
})
print(e["summary"])
print(f"  overall passed: {e['passed']}")
