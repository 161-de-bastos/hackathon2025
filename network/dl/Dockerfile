FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3 python3-pip ca-certificates && \
    rm -rf /var/lib/apt/lists/*

ARG GIT_REF=main
ARG REPO_URL=https://github.com/161-de-bastos/hackathon2025.git
ENV PYTHONUNBUFFERED=1 \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=9000 \
    REPO_DIR=/app/hackathon2025

WORKDIR /app

RUN git clone --depth 1 --branch ${GIT_REF} ${REPO_URL} ${REPO_DIR}

# Reinstalar si luego cambias GIT_REF (mejor caché):
# RUN --mount=type=cache,target=/root/.cache/pip true

RUN if [ -f ${REPO_DIR}/requirements.txt ]; then \
      pip3 install --no-cache-dir -r ${REPO_DIR}/requirements.txt ; \
    fi


RUN pip3 install --no-cache-dir \
      "torch==2.4.*+cu124" --index-url https://download.pytorch.org/whl/cu124 \
      fastapi uvicorn[standard] pydantic==2.* prometheus-client


COPY main.py /app/main.py
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

EXPOSE 9000
HEALTHCHECK --interval=20s --timeout=5s --retries=5 \
  CMD curl -fsS http://localhost:9000/healthz || exit 1
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9000"]