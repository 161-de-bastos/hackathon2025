import importlib
import os
import json
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Header, Request
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
Instrumentator().instrument(app).expose(app)

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

@app.post("/predict")
async def predict(request: Request):
    if app.state.lit is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    body = await request.json()
    text = (body.get("text") or "").strip()
    preprocess_kwargs = body.get("preprocess_kwargs") or {}
    infer_kwargs      = body.get("infer_kwargs") or {}

    if not text:
        raise HTTPException(status_code=422, detail="Field 'text' is required")

    try:
        _, sents = preprocess_fn(text, **preprocess_kwargs)
        out = infer_fn(sents, app.state.lit, app.state.tok, app.state.kb, **infer_kwargs)
        return out  # ← dict plano, sin envolver
    except Exception as e:
        raise HTTPException(status_code=500, detail="predict failed: %s" % e)