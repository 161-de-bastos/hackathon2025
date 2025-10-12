from fastapi import FastAPI, Request
import torch
from torch.nn import functional as F
import json

app = FastAPI()

# Cargar modelo en GPU
MODEL_PATH = "model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(MODEL_PATH, map_location=device)
model.eval()

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    text = data["text"]
    
    # 🔹 Aquí conviertes texto a tensores (según tu modelo)
    inputs = torch.tensor(model.tokenize(text)).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1).cpu().numpy().tolist()
    
    return {"prediction": probs}
