from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from pitchmodel.predict import predict_whiff_probability
from pitchmodel.schemas import PitchRequest, PredictionResponse
from pitchmodel.train import load_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    yield

app = FastAPI(title="Pitch Whiff Model", lifespan=lifespan)

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(pitch: PitchRequest, request: Request) -> PredictionResponse:
    probability = predict_whiff_probability(request.app.state.model, pitch.model_dump())
    return PredictionResponse(whiff_probability=probability)
