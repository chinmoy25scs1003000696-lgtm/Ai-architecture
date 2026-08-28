from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

app = FastAPI(
    title="Enterprise AI Solution API",
    description="Week 5 Capstone Project - Inference & Evaluation Service",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    status: str

@app.get("/")
def read_root():
    return {
        "project": "Enterprise AI Solution",
        "status": "Healthy",
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    if not payload.features:
        raise HTTPException(status_code=400, detail="Features list cannot be empty.")
    
    simulated_score = float(np.mean(payload.features))
    prediction = 1 if simulated_score > 0.5 else 0
    confidence = float(min(0.99, max(0.51, abs(simulated_score))))
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "status": "Success"
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "inference-engine"}
