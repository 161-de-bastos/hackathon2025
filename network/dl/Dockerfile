FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 UVICORN_HOST=0.0.0.0 UVICORN_PORT=9000

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu124 \
      "torch==2.4.*+cu124"

RUN pip install --no-cache-dir \
      fastapi uvicorn[standard] pydantic>=2 prometheus-client \
      "injusToS @ git+https://github.com/161-de-bastos/hackathon2025@servidores"

COPY server.py /app/server.py

EXPOSE 9000
HEALTHCHECK --interval=20s --timeout=5s --retries=5 \
  CMD curl -fsS http://localhost:9000/healthz || exit 1

CMD ["uvicorn","server:app","--host","0.0.0.0","--port","9000"]