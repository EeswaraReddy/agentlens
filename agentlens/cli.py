"""
AgentLens command-line interface.

    agentlens view [--dir runs] [--port 8800]   # launch the local viewer
    agentlens ls   [--dir runs]                 # list saved traces
"""

import argparse
import glob
import json
import os
import sys


def _cmd_view(args: argparse.Namespace) -> int:
    from .viewer import serve
    serve(directory=args.dir, port=args.port)
    return 0


def _cmd_ls(args: argparse.Namespace) -> int:
    paths = sorted(glob.glob(os.path.join(args.dir, "*.json")), reverse=True)
    if not paths:
        print(f"No traces found in '{args.dir}/'.")
        return 0
    print(f"{'TRACE ID':<14}{'NAME':<24}{'STATUS':<8}{'SPANS':>6}{'TOKENS':>8}{'COST':>10}")
    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                t = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        s = t.get("summary", {})
        print(f"{t.get('trace_id',''):<14}{t.get('name','')[:22]:<24}"
              f"{t.get('status',''):<8}{s.get('spans',0):>6}"
              f"{s.get('total_tokens',0):>8}{('$%.5f' % s.get('total_cost_usd',0)):>10}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="agentlens", description="Observability + evals for AI agents")
    sub = parser.add_subparsers(dest="command")

    p_view = sub.add_parser("view", help="Launch the local trace viewer")
    p_view.add_argument("--dir", default="runs", help="Directory of trace JSON files")
    p_view.add_argument("--port", type=int, default=8800)
    p_view.set_defaults(func=_cmd_view)

    p_ls = sub.add_parser("ls", help="List saved traces")
    p_ls.add_argument("--dir", default="runs")
    p_ls.set_defaults(func=_cmd_ls)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
