FROM python:3.13.2-slim AS builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency file and install
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --locked

# Runtime stage
FROM python:3.13.2-slim AS python-runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpng16-16 \
    libjpeg62-turbo \
    libwebp7 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"

# Copy application code
COPY . .

# Environment variables (can be overridden)
ENV WORKERS=1 \
    LOG_LEVEL=info \
    FACE_MATCH_THRESHOLD=0.6

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose application port
EXPOSE 8000

# Start application
ENTRYPOINT ["/entrypoint.sh"]