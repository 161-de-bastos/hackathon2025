import importlib
import os
import json
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

LOAD_ENTRYPOINT       = os.getenv("LOAD_ENTRYPOINT",       "malvaditos.api:load")
PREPROCESS_ENTRYPOINT = os.getenv("PREPROCESS_ENTRYPOINT", "malvaditos.api:preprocess")
INFER_ENTRYPOINT      = os.getenv("INFER_ENTRYPOINT",      "malvaditos.api:infer")

def _import_callable(spec):
    mod_path, fn_name = spec.split(":")
    mod = importlib.import_module(mod_path)
    fn = getattr(mod, fn_name, None)
    if not callable(fn):
        raise RuntimeError("Function '%s' not found/callable in '%s'" % (fn_name, mod_path))
    return fn

load_fn       = _import_callable(LOAD_ENTRYPOINT)
preprocess_fn = _import_callable(PREPROCESS_ENTRYPOINT)
infer_fn      = _import_callable(INFER_ENTRYPOINT)

_lock = asyncio.Lock()

@asynccontextmanager
async def lifespan(app):
    # métricas
    Instrumentator().instrument(app).expose(app)
    # estado inicial
    app.state.lit = None
    app.state.tok = None
    app.state.kb  = None
    # carga del modelo al arrancar
    async with _lock:
        t = load_fn()  # debe devolver (lit, tok, kb)
        if not isinstance(t, (tuple, list)) or len(t) < 3:
            raise RuntimeError("load() must return (lit, tok, kb_struct)")
        app.state.lit, app.state.tok, app.state.kb = t[0], t[1], t[2]
    # app listo
    yield
    # (opcional) cleanup al apagar
    # e.g., app.state.lit = app.state.tok = app.state.kb = None

app = FastAPI(title="DL Inference (malvaditos)", version="1.0.0", lifespan=lifespan)

class PredictIn(BaseModel):
    text = Field(..., min_length=1)
    preprocess_kwargs = Field(default_factory=dict)
    infer_kwargs = Field(default_factory=dict)

class PredictOut(BaseModel):
    result = Field(...)


@app.get("/healthz")
def healthz():
    ok = all(getattr(app.state, k, None) is not None for k in ("lit","tok","kb"))
    return {
        "ok": ok,
        "loaded": ok,
        "load_entrypoint": LOAD_ENTRYPOINT,
        "preprocess_entrypoint": PREPROCESS_ENTRYPOINT,
        "infer_entrypoint": INFER_ENTRYPOINT,
    }

@app.post("/load")
async def reload_model():
    async with _lock:
        t = load_fn()
        if not isinstance(t, (tuple, list)) or len(t) < 3:
            raise RuntimeError("load() must return (lit, tok, kb_struct)")
        app.state.lit, app.state.tok, app.state.kb = t[0], t[1], t[2]
    return {"ok": True}

@app.post("/predict", response_model=PredictOut)
def predict(req: PredictIn):
    if app.state.lit is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        idxin, sents = preprocess_fn(req.text, **(req.preprocess_kwargs or {}))
        out = infer_fn(sents, app.state.lit, app.state.tok, app.state.kb, **(req.infer_kwargs or {}))
    except Exception as e:
        raise HTTPException(status_code=500, detail="predict failed: %s" % e)
    return {"result": out}