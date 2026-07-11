"""
Inspect a saved trace from the terminal — no UI needed.

Usage:
    python examples/inspect_trace.py                      # most recent trace
    python examples/inspect_trace.py runs/52465347dea4.json
    python examples/inspect_trace.py --all                # summary of every trace
"""

import glob
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt_cost(usd: float) -> str:
    return "$%.6f" % usd


def print_trace(t: dict, verbose: bool = True):
    s = t.get("summary", {})
    print()
    print("=" * 60)
    print(f"  trace_id : {t['trace_id']}")
    print(f"  name     : {t['name']}")
    print(f"  status   : {t['status']}")
    print(f"  duration : {t.get('duration_ms', 0):.1f} ms")
    print(f"  spans    : {s.get('spans', 0)}  →  {s.get('counts', {})}")
    print(f"  tokens   : {s.get('total_tokens', 0)}")
    print(f"  cost     : {fmt_cost(s.get('total_cost_usd', 0))}")
    if t.get("metadata"):
        print(f"  metadata : {t['metadata']}")
    print()

    if not verbose:
        return

    kind_icons = {"agent": "🤖", "llm": "🧠", "tool": "🔧", "event": "📍"}
    spans = t.get("spans", [])

    # build parent index for indentation
    id_to_idx = {sp["span_id"]: i for i, sp in enumerate(spans)}

    for sp in spans:
        icon  = kind_icons.get(sp["kind"], "·")
        depth = 1 if sp.get("parent_id") else 0
        pad   = "   " * depth

        cost_str   = f"  cost={fmt_cost(sp['cost_usd'])}" if sp["cost_usd"] > 0 else ""
        token_str  = f"  {sp['total_tokens']}tok" if sp["total_tokens"] > 0 else ""
        ms_str     = f"  {sp['duration_ms']:.1f}ms" if sp.get("duration_ms") else ""
        status_str = "  ❌ ERROR: " + sp["error"] if sp["status"] == "error" else ""

        print(f"  {pad}{icon} [{sp['kind']:5}] {sp['name']}{token_str}{cost_str}{ms_str}{status_str}")

        # show inputs
        if sp.get("inputs"):
            for k, v in sp["inputs"].items():
                val = str(v)
                if len(val) > 100:
                    val = val[:100] + "..."
                print(f"  {pad}        ┌ in.{k} = {val}")

        # show outputs
        if sp.get("outputs"):
            for k, v in sp["outputs"].items():
                val = str(v)
                if len(val) > 100:
                    val = val[:100] + "..."
                print(f"  {pad}        └ out.{k} = {val}")

        # show attributes / events
        if sp.get("attributes") and sp["kind"] == "event":
            print(f"  {pad}        └ attrs = {sp['attributes']}")

        print()


def print_all_summary(traces: list[dict]):
    fmt = "{:<14}{:<26}{:<8}{:>6}{:>8}{:>12}{:>10}"
    header = fmt.format("TRACE ID", "NAME", "STATUS", "SPANS", "TOKENS", "COST", "ERRORS")
    print()
    print(header)
    print("-" * len(header))
    for t in traces:
        s = t.get("summary", {})
        print(fmt.format(
            t["trace_id"][:14],
            t["name"][:24],
            t["status"],
            s.get("spans", 0),
            s.get("total_tokens", 0),
            fmt_cost(s.get("total_cost_usd", 0)),
            s.get("errors", 0),
        ))

    total_tokens = sum(t.get("summary", {}).get("total_tokens", 0) for t in traces)
    total_cost   = sum(t.get("summary", {}).get("total_cost_usd", 0) for t in traces)
    print("-" * len(header))
    print(f"  {len(traces)} traces  ·  {total_tokens} total tokens  ·  {fmt_cost(total_cost)} total cost")
    print()


def main():
    args = sys.argv[1:]

    if "--all" in args:
        paths = sorted(glob.glob("runs/*.json"), reverse=True)
        traces = []
        for p in paths:
            try:
                traces.append(load(p))
            except Exception:
                pass
        print_all_summary(traces)
        return

    # specific file or most recent
    if args and not args[0].startswith("--"):
        path = args[0]
    else:
        paths = sorted(glob.glob("runs/*.json"), reverse=True)
        if not paths:
            print("No traces found in runs/")
            return
        path = paths[0]

    print(f"Loading: {path}")
    t = load(path)
    print_trace(t, verbose=True)


if __name__ == "__main__":
    main()
