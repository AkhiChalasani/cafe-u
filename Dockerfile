FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY engine/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY engine/ engine/
COPY agent/dist/cafeu.min.js agent/dist/cafeu.min.js 2>/dev/null || true

# Create data dirs
RUN mkdir -p /data /app/engine/rules/rag_cache

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8080/health').read().decode())"

EXPOSE 8080

# Run with uvicorn
CMD ["uvicorn", "engine.api.server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
