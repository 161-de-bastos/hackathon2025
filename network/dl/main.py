import os, importlib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, generate_latest

# Dónde está tu función de inferencia: "mod.path:func"
ENTRYPOINT = os.getenv("MANN_ENTRYPOINT", "hackathon2025.api.mann_infer:infer")
REPO_DIR = os.getenv("REPO_DIR", "/app/hackathon2025")

mod_name, func_name = ENTRYPOINT.split(":")
import sys
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
infer_func = getattr(importlib.import_module(mod_name), func_name)

app = FastAPI()
REQS = Counter("mann_infer_requests_total", "Total infer requests")

class InferenceIn(BaseModel):
    text: str

@app.post("/infer")
def infer(payload: InferenceIn):
    REQS.inc()
    # Llama a tu repo
    result = infer_func(payload.text)
    return result

@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type="text/plain")

@app.get("/healthz")
def healthz():
    return {"status": "ok"}