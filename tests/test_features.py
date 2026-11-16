import numpy as np
import pandas as pd
from pitchmodel.features import add_sequence_features, add_whiff_target

def make_pitches() -> pd.DataFrame:
    """Two at-bats in one game, already sorted chronologically."""
    return pd.DataFrame({
        "game_pk":       [1, 1, 1, 1, 1],
        "at_bat_number": [1, 1, 1, 2, 2],
        "pitch_number":  [1, 2, 3, 1, 2],
        "pitch_type":    ["FF", "FF", "SL", "CH", "FF"],
        "release_speed": [95.0, 96.0, 85.0, 83.0, 94.0],
        "plate_x":       [0.0, 0.5, 1.0, -0.5, 0.0],
        "plate_z":       [2.0, 3.0, 1.5, 2.5, 3.5],
        "description":   ["ball", "swinging_strike", "foul", "ball", "hit_into_play"],
    })


def test_whiff_target_uses_savant_definition():
    df = add_whiff_target(pd.DataFrame(
        {"description": ["swinging_strike", "foul_tip", "foul", "hit_into_play", "ball"]}))
    assert df["whiff"].tolist() == [1, 1, 0, 0, 0]


def test_first_pitch_of_each_at_bat_has_no_previous_pitch():
    out = add_sequence_features(add_whiff_target(make_pitches()))
    assert out.loc[out["pitch_number"] == 1, "prev_pitch_type"].isna().all()


def test_previous_pitch_comes_from_same_at_bat_only():
    out = add_sequence_features(add_whiff_target(make_pitches()))
    assert out.loc[1, "prev_pitch_type"] == "FF"
    assert out.loc[2, "prev_pitch_type"] == "FF"
    assert pd.isna(out.loc[3, "prev_pitch_type"])   # row 2's CH must not leak into the next at-bat


def test_speed_change_is_current_minus_previous():
    out = add_sequence_features(add_whiff_target(make_pitches()))
    assert out.loc[1, "speed_change"] == 1.0     # 96 - 95
    assert out.loc[2, "speed_change"] == -11.0   # 85 - 96


def test_prev_whiff_looks_back_one_pitch():
    out = add_sequence_features(add_whiff_target(make_pitches()))
    assert out.loc[1, "prev_whiff"] == 0   # previous pitch was a ball
    assert out.loc[2, "prev_whiff"] == 1   # previous pitch was a swinging strike


def test_same_type_as_prev():
    out = add_sequence_features(add_whiff_target(make_pitches()))
    assert out.loc[1, "same_type_as_prev"] == 1.0
    assert out.loc[2, "same_type_as_prev"] == 0.0
    assert np.isnan(out.loc[0, "same_type_as_prev"])


def test_input_is_not_modified():
    df = make_pitches()
    cols_before = list(df.columns)
    add_sequence_features(add_whiff_target(df))
    assert list(df.columns) == cols_before