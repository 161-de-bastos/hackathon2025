import os, json, asyncio, uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

# --- Config por ENV ---
DL_URL          = os.getenv("DL_URL", "http://dl:9000")
DL_PREDICT_PATH = os.getenv("DL_PREDICT_PATH", "/predict")
TIMEOUT_SECS    = float(os.getenv("API_TIMEOUT", "30"))
RETRIES         = int(os.getenv("API_RETRIES", "2"))

# --- Esquemas ligeros (sin hints estrictos) ---
class AnalyzeIn(BaseModel):
    text = Field(..., min_length=1)
    preprocess_kwargs = Field(default_factory=dict)
    infer_kwargs = Field(default_factory=dict)

# puedes devolver dict plano → no definimos response_model
# class AnalyzeOut(BaseModel):
#     result = Field(...)

# --- Cliente HTTP global + lifespan ---
http_client = None
_client_lock = asyncio.Lock()

@asynccontextmanager
async def lifespan(app):
    global http_client
    Instrumentator().instrument(app).expose(app)

    # Un único httpx.AsyncClient para toda la app
    async with _client_lock:
        if http_client is None:
            http_client = httpx.AsyncClient(
                base_url=DL_URL,
                timeout=httpx.Timeout(TIMEOUT_SECS),
                limits=httpx.Limits(max_keepalive_connections=50, max_connections=200),
            )
    yield

    # Cierre ordenado
    async with _client_lock:
        if http_client is not None:
            await http_client.aclose()

app = FastAPI(title="API gateway → DL", version="1.0.0", lifespan=lifespan)

# --- Utilidad: reintentos simples con backoff ---
async def _post_predict(payload, request_id):
    backoff = 0.5
    last_exc = None
    for attempt in range(RETRIES + 1):
        try:
            r = await http_client.post(
                DL_PREDICT_PATH,
                json=payload,
                headers={"X-Request-ID": request_id},
            )
            # DL devuelve dict plano; si viene error, propaga
            if r.status_code >= 400:
                raise HTTPException(status_code=r.status_code, detail=r.text)
            return r.json()
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            last_exc = e
            if attempt < RETRIES:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 4.0)
            else:
                raise HTTPException(status_code=504, detail=f"DL timeout/network: {e}")
    # por si acaso
    raise HTTPException(status_code=502, detail=f"DL unreachable: {last_exc}")

# --- Endpoints ---
@app.get("/healthz")
async def healthz():
    try:
        r = await http_client.get("/healthz")
        ok = r.status_code == 200
        extra = r.json() if ok else {"dl_status": r.status_code}
    except Exception as e:
        ok = False
        extra = {"error": str(e)}
    return {"ok": ok, "dl": extra, "dl_url": f"{DL_URL}{DL_PREDICT_PATH}"}

@app.post("/analyze")  # devuelve dict plano del DL
async def analyze(body: AnalyzeIn, x_request_id: str | None = Header(default=None), request: Request = None):
    # request id para rastreo de punta a punta
    rid = x_request_id or request.headers.get("X-Request-ID") or str(uuid.uuid4())
    payload = {
        "text": body.text,
        "preprocess_kwargs": body.preprocess_kwargs or {},
        "infer_kwargs": body.infer_kwargs or {},
    }
    out = await _post_predict(payload, rid)
    # Puedes añadir metadata si quieres:
    # out = {"_request_id": rid, **out}
    return out
