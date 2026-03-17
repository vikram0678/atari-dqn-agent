"""
FastAPI inference server.
Endpoints:
  GET  /health   → health check
  POST /predict  → get action from game state
"""

import os
import gymnasium as gym
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

try:
    import ale_py
    gym.register_envs(ale_py)
except Exception:
    pass

from src.api.schemas import PredictRequest, PredictResponse, HealthResponse
from src.api.model_loader import model_loader


# ── Startup / Shutdown ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    model_path = os.getenv("MODEL_PATH", "models/latest_model.pth")
    game       = os.getenv("ATARI_GAME_ID", "ALE/Pong-v5")
    n_actions  = int(os.getenv("N_ACTIONS", "6"))

    try:
        model_loader.load(model_path, n_actions=n_actions)
        print(f"✓ Model loaded: {model_path}")
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        print("API will start but /predict will fail until model is available.")

    yield  # App runs here

    print("Shutting down API.")


# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="Atari DQN Agent API",
    description="REST API for serving a trained DQN agent on Atari games.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        model_loaded=model_loader.is_loaded,
        game=os.getenv("ATARI_GAME_ID", "ALE/Pong-v5"),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Predict best action for a given game state.

    Request body:
        { "state": [[[float, ...], ...], ...] }  shape: (4, 84, 84)

    Response:
        { "action": int, "q_values": [float, ...] }
    """
    if not model_loader.is_loaded:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Check MODEL_PATH environment variable.",
        )

    try:
        action, q_values = model_loader.predict(request.state)
        return PredictResponse(action=action, q_values=q_values)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


@app.get("/")
def root():
    return {
        "message": "Atari DQN Agent API",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }
