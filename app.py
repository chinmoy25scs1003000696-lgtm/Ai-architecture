# ==============================================================================
# Intelligent Multi-Modal Logistics Optimization Engine (IMLOE) - Core Pipeline
# ==============================================================================
# Description: This script sets up a complete, runnable foundation for the IMLOE 
# AI architecture. It includes synthetic data generation for IoT/weather telemetry, 
# a PyTorch Temporal Fusion Transformer (TFT) baseline model for demand forecasting, 
# and a FastAPI server to expose endpoints for real-time inference and health checks.
# ==============================================================================

from datetime import datetime
from typing import List
import numpy as np
import pandas as pd
from pydantic import BaseModel
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException

# ------------------------------------------------------------------------------
# 1. Synthetic Data Generation (Simulating Ingestion & ETL Layer)
# ------------------------------------------------------------------------------
def generate_synthetic_telemetry(num_records: int = 1000) -> pd.DataFrame:
    np.random.seed(42)
    timestamps = pd.date_range(start="2026-01-01", periods=num_records, freq="H")
    sku_ids = [f"SKU_{i:03d}" for i in range(10)]
    
    data = {
        "timestamp": np.random.choice(timestamps, num_records),
        "sku_id": np.random.choice(sku_ids, num_records),
        "temperature": np.random.uniform(15.0, 35.0, num_records),
        "traffic_delay_mins": np.random.exponential(10.0,size=num_records),
        "historical_demand": np.random.poisson(lam=120, size=num_records)
    }
    df = pd.DataFrame(data)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# ------------------------------------------------------------------------------
# 2. PyTorch Machine Learning Engine (Predictive Demand Forecasting Module)
# ------------------------------------------------------------------------------
class SimpleDemandForecaster(nn.Module):
    """
    A lightweight neural network simulating the multi-horizon forecasting 
    capabilities of a Temporal Fusion Transformer.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(SimpleDemandForecaster, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        out = self.fc(last_step)
        return out

# ------------------------------------------------------------------------------
# 3. FastAPI Service (Exposing Model Inference via REST API)
# ------------------------------------------------------------------------------
app = FastAPI(
    title="IMLOE Inference API",
    description="Backend AI service for real-time logistics and demand forecasting.",
    version="1.0.0"
)

model = SimpleDemandForecaster(input_dim=3)
model.eval()

class PredictionRequest(BaseModel):
    sku_id: str
    recent_temperatures: List[float]
    recent_traffic_delays: List[float]
    recent_demands: List[float]

class PredictionResponse(BaseModel):
    sku_id: str
    predicted_demand: float
    timestamp: str

@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "system": "Intelligent Multi-Modal Logistics Optimization Engine",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/predict/demand", response_model=PredictionResponse)
def predict_demand(payload: PredictionRequest):
    if len(payload.recent_demands) == 0:
        raise HTTPException(status_code=400, detail="Telemetry input sequences cannot be empty.")
    
    features = list(zip(payload.recent_temperatures, payload.recent_traffic_delays, payload.recent_demands))
    input_tensor = torch.tensor([features], dtype=torch.float32)
    
    with torch.no_grad():
        prediction = model(input_tensor)
        forecast_value = float(prediction.item())

    return PredictionResponse(
        sku_id=payload.sku_id,
        predicted_demand=max(0.0, round(forecast_value, 2)),
        timestamp=datetime.utcnow().isoformat()
    )

if __name__ == "__main__":
    import uvicorn
    print("Initializing IMLOE Pipeline & Test Data Ingestion...")
    sample_df = generate_synthetic_telemetry(num_records=5)
    print("Sample Ingested Data Preview:\n", sample_df.head())
    
    print("\nStarting Local API Server on port 8000...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
