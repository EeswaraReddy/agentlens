"""AgentLens server — production backend (FastAPI + SQLAlchemy).

The server provides a multi-tenant REST API for ingesting traces, browsing
them, and exposing aggregate statistics, plus an enterprise web dashboard.

Install the server extras:

    pip install "agentlens[server]"

Run:

    agentlens server                 # starts on http://localhost:8800
"""
