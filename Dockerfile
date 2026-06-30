# AgentLens production server image.
# Build: docker build -t agentlens:0.2.0 .
# Run:   docker run -p 8800:8800 -v $(pwd)/data:/data agentlens:0.2.0

FROM python:3.12-slim

WORKDIR /app

# System deps kept minimal; Postgres extras installed at runtime via extras.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE MANIFEST.in ./
COPY agentlens ./agentlens

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir ".[server]"

ENV AGENTLENS_DATABASE_URL=sqlite:////data/agentlens.db
ENV PYTHONUNBUFFERED=1
EXPOSE 8800
VOLUME ["/data"]

CMD ["uvicorn", "agentlens.server.app:app", "--host", "0.0.0.0", "--port", "8800"]
