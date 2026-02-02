FROM python:3.13.2-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y cmake build-essential git

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Runtime stage  
FROM python:3.13.2-slim AS python-runtime

WORKDIR /app

# Install runtime dependencies including curl for health checks
RUN apt-get update && apt-get install -y \
    libpng16-16 \
    libjpeg62-turbo \
    libwebp-dev \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

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