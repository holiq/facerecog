FROM python:3.13.2-slim AS builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency file and install
COPY pyproject.toml .
RUN uv sync --no-dev --system

# Pre-download insightface model (default: buffalo_l)
RUN python -c "from insightface.app import FaceAnalysis; app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); app.prepare(ctx_id=-1)"

# Runtime stage
FROM python:3.13.2-slim AS python-runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpng16-16 \
    libjpeg62-turbo \
    libwebp-dev \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy pre-downloaded insightface model
COPY --from=builder /root/.insightface /root/.insightface

# Copy application code
COPY . .

# Environment variables (can be overridden)
ENV WORKERS=1 \
    LOG_LEVEL=info \
    FACE_MATCH_THRESHOLD=0.6

# Expose application port
EXPOSE 8000

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Start application
ENTRYPOINT ["/entrypoint.sh"]