import pytest

from pitchmodel.predict import predict_whiff_probability
from pitchmodel.train import fit_model
from test_train import make_training_frame   # reuse the synthetic data from test_train.py

SAMPLE_PITCH = {
    "pitch_type": "SL", "release_speed": 85.0, "release_spin_rate": 2500.0,
    "pfx_x": 0.4, "pfx_z": 0.1, "plate_x": 0.3, "plate_z": 1.8,
    "balls": 1, "strikes": 2, "stand": "R", "p_throws": "R",
}


@pytest.fixture(scope="module")
def model():
    return fit_model(make_training_frame())


def test_returns_a_probability(model):
    p = predict_whiff_probability(model, SAMPLE_PITCH)
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0


def test_missing_field_gives_a_clear_error(model):
    bad = {k: v for k, v in SAMPLE_PITCH.items() if k != "release_speed"}
    with pytest.raises(ValueError, match="release_speed"):
        predict_whiff_probability(model, bad)


def test_unseen_pitch_type_does_not_crash(model):
    p = predict_whiff_probability(model, {**SAMPLE_PITCH, "pitch_type": "KN"})
    assert 0.0 <= p <= 1.0