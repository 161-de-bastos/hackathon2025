FROM python:3.11-slim
WORKDIR /app

# (opcional pero útil) evita el prompt interactivo en apt
ENV DEBIAN_FRONTEND=noninteractive

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt || true

COPY app.py /app/

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Por si no está en requirements.txt:
RUN pip install --no-cache-dir streamlit requests prometheus-client

ENV API_URL=http://api:8000
# (opcional) streamlit sin telemetría
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501
CMD ["streamlit","run","app.py","--server.address=0.0.0.0","--server.port=8501"]

# Healthcheck correcto
HEALTHCHECK --interval=20s --timeout=5s --retries=5 --start-period=15s \
  CMD curl -fsS http://localhost:8501/_stcore/health || exit 1