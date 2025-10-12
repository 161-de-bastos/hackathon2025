import os, httpx
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, generate_latest
from fastapi.responses import PlainTextResponse

DL_URL = os.getenv("DL_URL", "http://dl:9000")
app = FastAPI()
BRIDGE_REQS = Counter("bridge_requests_total", "Bridge requests to DL")

class TOS(BaseModel):
    text: str

@app.post("/process")
async def process(data: TOS):
    BRIDGE_REQS.inc()
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{DL_URL}/infer", json=data.dict())
        r.raise_for_status()
        return r.json()

@app.get("/healthz")
def healthz(): return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type="text/plain")
