FROM python:3.11-slim
WORKDIR /app
COPY main.py /app/
RUN pip install --no-cache-dir fastapi uvicorn[standard] httpx prometheus-client

# Healthcheck tool
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=5s --retries=5 \
  CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]