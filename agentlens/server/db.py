"""
Database layer for the AgentLens server.

Uses SQLAlchemy with SQLite by default, ready for Postgres in production.
The schema is intentionally simple and indexed for the queries the dashboard
runs:

  - projects(id, name, api_key)
  - traces(id, project_id, name, status, duration_ms, total_tokens, cost_usd,
           start_ts, span_count, error_count, raw_json)
  - spans(id, trace_id, parent_id, name, kind, status, duration_ms,
          prompt_tokens, completion_tokens, cost_usd, model, raw_json)
"""

from __future__ import annotations

import os
import secrets
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, String, Integer, Float, Text, DateTime, ForeignKey, Index, create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship


DATABASE_URL = os.getenv("AGENTLENS_DATABASE_URL", "sqlite:///./agentlens.db")

Base = declarative_base()

_engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    future=True,
)
SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False, future=True)


class Project(Base):
    __tablename__ = "projects"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    api_key = Column(String, nullable=False, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    traces = relationship("TraceRow", back_populates="project", cascade="all, delete-orphan")


class TraceRow(Base):
    __tablename__ = "traces"

    id = Column(String, primary_key=True)              # trace_id
    project_id = Column(String, ForeignKey("projects.id"), nullable=False, index=True)
    name = Column(String, nullable=False, index=True)
    status = Column(String, nullable=False, index=True)        # ok | error
    duration_ms = Column(Float, default=0.0)
    total_tokens = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    span_count = Column(Integer, default=0)
    error_count = Column(Integer, default=0)
    start_ts = Column(DateTime, default=datetime.utcnow, index=True)
    received_at = Column(DateTime, default=datetime.utcnow)
    raw_json = Column(Text, nullable=False)            # full serialized trace

    project = relationship("Project", back_populates="traces")
    spans = relationship("SpanRow", back_populates="trace", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_traces_project_start", "project_id", "start_ts"),
        Index("ix_traces_project_status", "project_id", "status"),
    )


class SpanRow(Base):
    __tablename__ = "spans"

    id = Column(String, primary_key=True)              # span_id
    trace_id = Column(String, ForeignKey("traces.id"), nullable=False, index=True)
    parent_id = Column(String, nullable=True)
    name = Column(String, nullable=False)
    kind = Column(String, nullable=False, index=True)
    status = Column(String, nullable=False)
    duration_ms = Column(Float, default=0.0)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    model = Column(String, nullable=True)
    raw_json = Column(Text, nullable=False)

    trace = relationship("TraceRow", back_populates="spans")


def init_db() -> None:
    """Create all tables (idempotent). Safe to call on every startup."""
    Base.metadata.create_all(_engine)


def get_session():
    """FastAPI dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_default_project(session) -> Project:
    """Create the 'default' project the first time the server starts."""
    proj = session.query(Project).filter_by(name="default").one_or_none()
    if proj is None:
        proj = Project(
            id=secrets.token_hex(8),
            name="default",
            api_key="al_" + secrets.token_urlsafe(24),
        )
        session.add(proj)
        session.commit()
    return proj
