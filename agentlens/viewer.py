"""
Local web viewer for AgentLens traces.

Reads saved trace JSON files from a directory and serves a single-page UI to
inspect runs: span timeline, tokens, cost, status, and eval-friendly detail.

    agentlens view --dir runs --port 8800

Requires the optional 'viewer' extra:  pip install "agentlens[viewer]"
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List


def load_traces(directory: str) -> List[Dict[str, Any]]:
    traces = []
    for path in sorted(glob.glob(os.path.join(directory, "*.json")), reverse=True):
        try:
            with open(path, "r", encoding="utf-8") as f:
                traces.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    return traces


def create_app(directory: str = "runs"):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title="AgentLens Viewer")

    @app.get("/api/traces")
    def api_traces():
        return JSONResponse(load_traces(directory))

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(_INDEX_HTML)

    return app


def serve(directory: str = "runs", port: int = 8800) -> None:
    try:
        import uvicorn
    except ImportError:
        raise SystemExit(
            'The viewer needs FastAPI/uvicorn. Install with: pip install "agentlens[viewer]"'
        )
    app = create_app(directory)
    print(f"AgentLens viewer → http://localhost:{port}  (reading '{directory}/')")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


_INDEX_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>AgentLens</title>
<style>
:root{--bg:#0b0e14;--panel:#11151f;--border:#1f2733;--text:#dbe3ee;--muted:#7d8aa0;
--accent:#5cc8ff;--ok:#39d98a;--err:#ff6b6b;--llm:#bd93f9;--tool:#ffb86c;--agent:#5cc8ff;}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:-apple-system,Segoe UI,Roboto,sans-serif;display:flex;height:100vh}
.side{width:320px;border-right:1px solid var(--border);overflow-y:auto;flex-shrink:0}
.side h1{font-size:15px;padding:16px;border-bottom:1px solid var(--border);letter-spacing:.5px}
.side h1 small{color:var(--muted);font-weight:400}
.item{padding:12px 16px;border-bottom:1px solid var(--border);cursor:pointer}
.item:hover{background:var(--panel)}
.item.active{background:var(--panel);border-left:3px solid var(--accent)}
.item .nm{font-size:13px;font-weight:600}
.item .meta{font-size:11px;color:var(--muted);margin-top:4px}
.badge{display:inline-block;padding:1px 7px;border-radius:10px;font-size:10px;font-weight:700}
.badge.ok{background:rgba(57,217,138,.15);color:var(--ok)}
.badge.error{background:rgba(255,107,107,.15);color:var(--err)}
.main{flex:1;overflow-y:auto;padding:24px}
.empty{color:var(--muted);margin-top:40px;text-align:center}
.cards{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:14px 18px;min-width:120px}
.card .k{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}
.card .v{font-size:20px;font-weight:700;margin-top:4px}
.span{background:var(--panel);border:1px solid var(--border);border-radius:8px;margin:6px 0;padding:10px 14px}
.span .row{display:flex;align-items:center;gap:10px}
.kind{font-size:10px;font-weight:700;padding:2px 8px;border-radius:6px;text-transform:uppercase}
.kind.llm{background:rgba(189,147,249,.15);color:var(--llm)}
.kind.tool{background:rgba(255,184,108,.15);color:var(--tool)}
.kind.agent{background:rgba(92,200,255,.15);color:var(--agent)}
.kind.event{background:rgba(125,138,160,.15);color:var(--muted)}
.span .nm{font-weight:600;font-size:13px}
.span .dur{margin-left:auto;color:var(--muted);font-size:12px}
.span .io{margin-top:8px;font-family:ui-monospace,Menlo,monospace;font-size:11px;color:var(--muted);white-space:pre-wrap;word-break:break-word}
.bar{height:4px;background:var(--accent);border-radius:2px;margin-top:8px;opacity:.5}
.tok{color:var(--muted);font-size:11px;margin-left:6px}
.refresh{float:right;background:transparent;border:1px solid var(--border);color:var(--text);border-radius:6px;padding:4px 10px;cursor:pointer;font-size:12px;margin:12px 16px}
</style></head><body>
<div class="side">
  <h1>AgentLens <small>traces</small><button class="refresh" onclick="load()">refresh</button></h1>
  <div id="list"></div>
</div>
<div class="main" id="main"><div class="empty">Select a trace on the left.</div></div>
<script>
let traces=[],sel=null;
async function load(){
  const r=await fetch('/api/traces');traces=await r.json();
  const list=document.getElementById('list');
  if(!traces.length){list.innerHTML='<div class="empty">No traces yet.<br>Run an agent, then refresh.</div>';return;}
  list.innerHTML=traces.map((t,i)=>`<div class="item ${i===sel?'active':''}" onclick="pick(${i})">
    <div class="nm">${t.name} <span class="badge ${t.status}">${t.status}</span></div>
    <div class="meta">${t.summary.spans} spans · ${t.summary.total_tokens} tok · $${t.summary.total_cost_usd.toFixed(5)}</div>
  </div>`).join('');
  if(sel===null&&traces.length){pick(0);}
}
function pick(i){sel=i;load2();
  document.querySelectorAll('.item').forEach((e,j)=>e.classList.toggle('active',j===i));}
function esc(s){return (s||'').replace(/</g,'&lt;');}
function load2(){
  const t=traces[sel];if(!t)return;
  const maxd=Math.max(...t.spans.map(s=>s.duration_ms||0),1);
  const cards=`<div class="cards">
    <div class="card"><div class="k">status</div><div class="v">${t.status}</div></div>
    <div class="card"><div class="k">duration</div><div class="v">${(t.duration_ms||0).toFixed(0)}<small> ms</small></div></div>
    <div class="card"><div class="k">spans</div><div class="v">${t.summary.spans}</div></div>
    <div class="card"><div class="k">tokens</div><div class="v">${t.summary.total_tokens}</div></div>
    <div class="card"><div class="k">cost</div><div class="v">$${t.summary.total_cost_usd.toFixed(5)}</div></div>
    <div class="card"><div class="k">errors</div><div class="v">${t.summary.errors}</div></div>
  </div>`;
  const spans=t.spans.map(s=>{
    const io=[];
    if(Object.keys(s.inputs||{}).length)io.push('in  '+esc(JSON.stringify(s.inputs)));
    if(Object.keys(s.outputs||{}).length)io.push('out '+esc(JSON.stringify(s.outputs)));
    if(s.error)io.push('ERR '+esc(s.error));
    const tok=s.total_tokens?`<span class="tok">${s.total_tokens} tok · $${(s.cost_usd||0).toFixed(5)}</span>`:'';
    const w=((s.duration_ms||0)/maxd*100).toFixed(0);
    return `<div class="span"><div class="row">
      <span class="kind ${s.kind}">${s.kind}</span>
      <span class="nm">${esc(s.name)}</span>${tok}
      <span class="dur">${(s.duration_ms||0).toFixed(1)} ms · ${s.status}</span></div>
      <div class="bar" style="width:${w}%"></div>
      ${io.length?'<div class="io">'+io.join('\\n')+'</div>':''}</div>`;
  }).join('');
  document.getElementById('main').innerHTML=cards+spans;
}
load();setInterval(load,3000);
</script></body></html>"""
