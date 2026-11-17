import pytest
from pydantic import ValidationError
from pitchmodel.schemas import PitchRequest

VALID = {
    "pitch_type": "SL", "release_speed": 85.0, "release_spin_rate": 2500.0,
    "pfx_x": 0.4, "pfx_z": 0.1, "plate_x": 0.3, "plate_z": 1.8,
    "balls": 1, "strikes": 2, "stand": "R", "p_throws": "R",
}


def test_valid_pitch_is_accepted():
    PitchRequest(**VALID)


@pytest.mark.parametrize("field, value", [
    ("balls", 4), ("balls", -1), ("strikes", 3),
    ("release_speed", -5.0), ("release_speed", 500.0),
    ("stand", "X"), ("p_throws", "S"),
    ("pitch_type", "ZZ"),
])
def test_invalid_values_are_rejected(field, value):
    with pytest.raises(ValidationError):
        PitchRequest(**{**VALID, field: value})


def test_missing_field_is_rejected():
    bad = {k: v for k, v in VALID.items() if k != "plate_x"}
    with pytest.raises(ValidationError):
        PitchRequest(**bad)