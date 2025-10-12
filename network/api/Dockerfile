FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 UVICORN_HOST=0.0.0.0 UVICORN_PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
      fastapi uvicorn[standard] httpx \
      prometheus-client prometheus-fastapi-instrumentator

COPY server.py /app/server.py

EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=5s --retries=5 \
  CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["uvicorn","server:app","--host","0.0.0.0","--port","8000"]
