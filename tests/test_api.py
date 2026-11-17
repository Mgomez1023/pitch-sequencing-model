import pytest
from fastapi.testclient import TestClient

from pitchmodel.api import app
from pitchmodel.predict import predict_whiff_probability
from pitchmodel.train import load_model

VALID = {
    "pitch_type": "SL", "release_speed": 85.0, "release_spin_rate": 2500.0,
    "pfx_x": 0.4, "pfx_z": 0.1, "plate_x": 0.3, "plate_z": 1.8,
    "balls": 1, "strikes": 2, "stand": "R", "p_throws": "R",
}


@pytest.fixture(scope="module")
def client():
    # The `with` block matters: it's what runs the lifespan, which loads the model.
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_a_probability(client):
    response = client.post("/predict", json=VALID)
    assert response.status_code == 200
    p = response.json()["whiff_probability"]
    assert 0.0 <= p <= 1.0


def test_api_matches_direct_prediction(client):
    response = client.post("/predict", json=VALID)
    expected = predict_whiff_probability(load_model(), VALID)
    assert response.json()["whiff_probability"] == pytest.approx(expected)


def test_invalid_pitch_gets_422(client):
    response = client.post("/predict", json={**VALID, "balls": 4})
    assert response.status_code == 422


def test_missing_field_gets_422(client):
    bad = {k: v for k, v in VALID.items() if k != "plate_x"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422