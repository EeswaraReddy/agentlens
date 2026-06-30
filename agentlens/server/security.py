"""API-key authentication for the AgentLens server.

Clients pass their project's key in the `X-API-Key` header. The header maps to
exactly one Project; all data access is scoped through it. No keys, no data.
"""

from __future__ import annotations

from fastapi import Header, HTTPException, Depends, status
from sqlalchemy.orm import Session

from .db import Project, get_session


def require_project(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    session: Session = Depends(get_session),
) -> Project:
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
        )
    project = session.query(Project).filter_by(api_key=x_api_key).one_or_none()
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return project
