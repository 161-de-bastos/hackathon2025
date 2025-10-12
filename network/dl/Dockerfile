FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=9000

# base OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Minimal runtime deps for the DL server ---
# Torch CUDA 12.4 + server deps + your library from Git (branch/tag)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu124 \
      "torch==2.4.*+cu124" && \
    pip install --no-cache-dir \
      fastapi uvicorn[standard] pydantic>=2 prometheus-client \
      "injusToS @ git+https://github.com/161-de-bastos/hackathon2025@servidores"

# Your server file must define `app = FastAPI(...)`
COPY server.py /app/server.py

EXPOSE 9000
HEALTHCHECK --interval=20s --timeout=5s --retries=5 \
  CMD curl -fsS http://localhost:9000/healthz || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9000"]