"""
AgentLens server — FastAPI app.

Endpoints (all `/api/v1`):

  POST /traces              ingest a trace                 [X-API-Key]
  GET  /traces              list traces (filter+paginate)  [X-API-Key]
  GET  /traces/{id}         trace + spans detail           [X-API-Key]
  GET  /stats               aggregate dashboard stats      [X-API-Key]
  GET  /project             project info                   [X-API-Key]

Plus:

  GET  /health              liveness
  GET  /                    enterprise dashboard (UI)
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from .db import Project, SpanRow, TraceRow, ensure_default_project, get_session, init_db, SessionLocal
from .schemas import (
    IngestRequest, IngestResponse, ProjectMeta, SpanDetail, StatsResponse,
    TraceDetail, TraceListResponse, TraceSummary,
)
from .security import require_project


def create_app() -> FastAPI:
    app = FastAPI(
        title="AgentLens",
        version="0.2.0",
        description="Observability + evals control plane for AI agents.",
    )

    # CORS open by default for dashboards on other origins; lock down in prod.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def _startup() -> None:
        init_db()
        with SessionLocal() as s:
            ensure_default_project(s)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        return JSONResponse(status_code=500, content={"detail": f"internal error: {exc}"})

    # ---- health / project ------------------------------------------------
    @app.get("/health")
    def health():
        return {"status": "ok", "service": "agentlens", "version": app.version}

    @app.get("/api/v1/project", response_model=ProjectMeta)
    def project_info(project: Project = Depends(require_project)):
        return ProjectMeta(id=project.id, name=project.name)

    # ---- ingest ----------------------------------------------------------
    @app.post("/api/v1/traces", response_model=IngestResponse,
              status_code=status.HTTP_201_CREATED)
    def ingest_trace(
        body: IngestRequest,
        project: Project = Depends(require_project),
        session: Session = Depends(get_session),
    ):
        return _store_trace(session, project, body.trace)

    # ---- list ------------------------------------------------------------
    @app.get("/api/v1/traces", response_model=TraceListResponse)
    def list_traces(
        project: Project = Depends(require_project),
        session: Session = Depends(get_session),
        q: Optional[str] = Query(default=None, description="search in trace name"),
        status_filter: Optional[str] = Query(default=None, alias="status"),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=25, ge=1, le=200),
    ):
        query = session.query(TraceRow).filter(TraceRow.project_id == project.id)
        if q:
            query = query.filter(TraceRow.name.contains(q))
        if status_filter:
            query = query.filter(TraceRow.status == status_filter)
        total = query.count()
        rows = (query.order_by(TraceRow.start_ts.desc())
                .offset((page - 1) * page_size).limit(page_size).all())
        return TraceListResponse(
            items=[_row_to_summary(r) for r in rows],
            total=total, page=page, page_size=page_size,
        )

    # ---- detail ----------------------------------------------------------
    @app.get("/api/v1/traces/{trace_id}", response_model=TraceDetail)
    def get_trace(
        trace_id: str,
        project: Project = Depends(require_project),
        session: Session = Depends(get_session),
    ):
        row = (session.query(TraceRow)
               .filter(TraceRow.id == trace_id, TraceRow.project_id == project.id)
               .one_or_none())
        if row is None:
            raise HTTPException(status_code=404, detail="trace not found")
        raw = json.loads(row.raw_json)
        return TraceDetail(
            trace_id=row.id,
            name=row.name,
            status=row.status,
            duration_ms=row.duration_ms,
            total_tokens=row.total_tokens,
            cost_usd=row.cost_usd,
            span_count=row.span_count,
            error_count=row.error_count,
            start_ts=row.start_ts.isoformat() + "Z",
            metadata=raw.get("metadata", {}),
            spans=[
                SpanDetail(
                    span_id=s.get("span_id"),
                    parent_id=s.get("parent_id"),
                    name=s.get("name"),
                    kind=s.get("kind"),
                    status=s.get("status", "ok"),
                    duration_ms=s.get("duration_ms") or 0.0,
                    prompt_tokens=s.get("prompt_tokens") or 0,
                    completion_tokens=s.get("completion_tokens") or 0,
                    cost_usd=s.get("cost_usd") or 0.0,
                    model=s.get("model"),
                    inputs=s.get("inputs") or {},
                    outputs=s.get("outputs") or {},
                    error=s.get("error"),
                )
                for s in raw.get("spans", [])
            ],
        )

    # ---- stats -----------------------------------------------------------
    @app.get("/api/v1/stats", response_model=StatsResponse)
    def stats(
        project: Project = Depends(require_project),
        session: Session = Depends(get_session),
        days: int = Query(default=14, ge=1, le=90),
    ):
        return _compute_stats(session, project, days)

    # ---- UI --------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        html_path = Path(__file__).parent / "ui" / "index.html"
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    return app


# ---- helpers --------------------------------------------------------------
def _row_to_summary(r: TraceRow) -> TraceSummary:
    return TraceSummary(
        trace_id=r.id,
        name=r.name,
        status=r.status,
        duration_ms=r.duration_ms or 0.0,
        total_tokens=r.total_tokens or 0,
        cost_usd=r.cost_usd or 0.0,
        span_count=r.span_count or 0,
        error_count=r.error_count or 0,
        start_ts=(r.start_ts or datetime.utcnow()).isoformat() + "Z",
    )


def _store_trace(session: Session, project: Project, trace: Dict[str, Any]) -> IngestResponse:
    trace_id = trace.get("trace_id")
    if not trace_id:
        raise HTTPException(status_code=400, detail="trace.trace_id is required")

    spans = trace.get("spans", []) or []
    summary = trace.get("summary", {}) or {}

    # idempotency: re-ingesting the same trace_id overwrites
    session.query(SpanRow).filter(SpanRow.trace_id == trace_id).delete()
    session.query(TraceRow).filter(TraceRow.id == trace_id,
                                   TraceRow.project_id == project.id).delete()

    start_ts = datetime.utcnow()
    if trace.get("start_ts"):
        try:
            start_ts = datetime.fromtimestamp(float(trace["start_ts"]), tz=timezone.utc).replace(tzinfo=None)
        except (TypeError, ValueError):
            pass

    row = TraceRow(
        id=trace_id,
        project_id=project.id,
        name=trace.get("name", "trace"),
        status=trace.get("status", "ok"),
        duration_ms=trace.get("duration_ms") or 0.0,
        total_tokens=summary.get("total_tokens") or 0,
        cost_usd=summary.get("total_cost_usd") or 0.0,
        span_count=summary.get("spans") or len(spans),
        error_count=summary.get("errors") or 0,
        start_ts=start_ts,
        raw_json=json.dumps(trace),
    )
    session.add(row)

    for s in spans:
        sid = s.get("span_id") or ""
        if not sid:
            continue
        session.add(SpanRow(
            id=trace_id + ":" + sid,   # globally unique
            trace_id=trace_id,
            parent_id=s.get("parent_id"),
            name=s.get("name", "span"),
            kind=s.get("kind", "agent"),
            status=s.get("status", "ok"),
            duration_ms=s.get("duration_ms") or 0.0,
            prompt_tokens=s.get("prompt_tokens") or 0,
            completion_tokens=s.get("completion_tokens") or 0,
            cost_usd=s.get("cost_usd") or 0.0,
            model=s.get("model"),
            raw_json=json.dumps(s),
        ))
    session.commit()
    return IngestResponse(trace_id=trace_id, spans_stored=len(spans))


def _compute_stats(session: Session, project: Project, days: int) -> StatsResponse:
    since = datetime.utcnow() - timedelta(days=days)
    q = session.query(TraceRow).filter(
        TraceRow.project_id == project.id, TraceRow.start_ts >= since
    )
    rows = q.all()

    total = len(rows)
    tokens = sum(r.total_tokens or 0 for r in rows)
    cost = sum(r.cost_usd or 0.0 for r in rows)
    errors = sum(1 for r in rows if r.status == "error")
    avg_dur = (sum(r.duration_ms or 0.0 for r in rows) / total) if total else 0.0

    # per-day breakdown
    per_day: Dict[str, Dict[str, float]] = {}
    for r in rows:
        d = (r.start_ts or datetime.utcnow()).date().isoformat()
        slot = per_day.setdefault(d, {"date": d, "count": 0, "cost": 0.0, "errors": 0})
        slot["count"] += 1
        slot["cost"] += r.cost_usd or 0.0
        slot["errors"] += 1 if r.status == "error" else 0

    # top models (from spans)
    span_q = (session.query(
                  SpanRow.model,
                  func.count(SpanRow.id),
                  func.sum(SpanRow.prompt_tokens + SpanRow.completion_tokens),
                  func.sum(SpanRow.cost_usd))
              .join(TraceRow, TraceRow.id == SpanRow.trace_id)
              .filter(TraceRow.project_id == project.id,
                      TraceRow.start_ts >= since,
                      SpanRow.kind == "llm",
                      SpanRow.model.isnot(None))
              .group_by(SpanRow.model)
              .order_by(func.sum(SpanRow.cost_usd).desc())
              .limit(5)).all()

    top_models = [
        {"model": m or "unknown", "calls": int(c or 0),
         "tokens": int(t or 0), "cost": float(co or 0.0)}
        for (m, c, t, co) in span_q
    ]

    return StatsResponse(
        total_traces=total,
        total_tokens=tokens,
        total_cost_usd=round(cost, 6),
        error_rate=round(errors / total, 4) if total else 0.0,
        avg_duration_ms=round(avg_dur, 2),
        traces_per_day=sorted(per_day.values(), key=lambda x: x["date"]),
        top_models=top_models,
    )


# importable app for `uvicorn agentlens.server.app:app`
app = create_app()
