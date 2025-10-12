# fastapi/app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os, requests

app = FastAPI()

MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://cuda-server:9000/predict")
SECRET_KEY = os.getenv("API_SECRET_KEY", None)

class PredictRequest(BaseModel):
    secret_key: str | None
    data: dict  # el cuerpo que se mandará al servidor CUDA

@app.post("/predict")
def forward_prediction(req: PredictRequest):
    if SECRET_KEY and req.secret_key != SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        response = requests.post(MODEL_SERVER_URL, json=req.data, timeout=60)
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error contacting CUDA server: {e}")
