# syntax=docker/dockerfile:1
# Multi-stage Dockerfile for TicketRouter Serving Service

# Stage 1: Build stage with uv
FROM python:3.12-slim AS builder

WORKDIR /build

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:0.5.26 /uv /bin/uv

# Copy project specification
COPY pyproject.toml .

# Install dependencies into virtualenv without dev dependencies
ENV UV_PROJECT_ENVIRONMENT="/app/.venv"
RUN uv sync --frozen --no-dev --no-install-project || uv sync --no-dev

# Stage 2: Minimal, secure non-root runner
FROM python:3.12-slim AS runner

WORKDIR /app

# Install curl for healthcheck probe
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root system user
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/false appuser

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV HOST=0.0.0.0
ENV DATA_DIR=/app/data
ENV SQLITE_DB_PATH=/tmp/predictions.db

# Copy application code, baseline models, and data
COPY ticketrouter /app/ticketrouter
COPY data /app/data
COPY models /app/models
COPY metrics /app/metrics

# Set permissions for non-root user
RUN chown -R appuser:appgroup /app /tmp

USER appuser

EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["uvicorn", "ticketrouter.serve:app", "--host", "0.0.0.0", "--port", "8000"]
