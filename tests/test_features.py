import numpy as np
import pandas as pd
from pitchmodel.features import (
    add_sequence_features, 
    add_whiff_target, 
    get_swings, 
    split_by_date
)
from pitchmodel.config import SWING_DESCRIPTIONS, TRAIN_CUTOFF

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


def make_swing_data() -> pd.DataFrame:
    """make_pitches() plus every column get_swings needs.

    Two games on either side of the 2026-09-01 cutoff. Pitch 0 is a ball (not a swing).
    Pitch 3 is a swing that is also the first pitch of its at-bat (prev_pitch_type is NaN).
    """
    df = make_pitches()
    df["game_pk"] = [1, 1, 1, 2, 2]
    df["game_date"] = ["2026-08-30"] * 3 + ["2026-09-02"] * 2
    df["description"] = ["ball", "swinging_strike", "foul", "foul", "hit_into_play"]
    df["release_spin_rate"] = 2300.0
    df["pfx_x"] = 0.5
    df["pfx_z"] = 1.2
    df["balls"] = 0
    df["strikes"] = [0, 1, 2, 0, 1]
    df["stand"] = "R"
    df["p_throws"] = "R"
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    return get_swings(add_sequence_features(add_whiff_target(df)))


def test_get_swings_keeps_only_swings():
    out = prepare(make_swing_data())
    assert set(out["description"]) <= set(SWING_DESCRIPTIONS)
    assert len(out) == 4   # the one "ball" is gone


def test_get_swings_keeps_first_pitch_swings():
    # A swing with no previous pitch must survive: dropping on sequence features
    # would silently make Model A and Model B score on different rows.
    out = prepare(make_swing_data())
    assert out["prev_pitch_type"].isna().sum() == 1


def test_get_swings_drops_rows_missing_a_baseline_feature():
    df = make_swing_data()
    df.loc[1, "release_spin_rate"] = np.nan
    out = prepare(df)
    assert len(out) == 3


def test_get_swings_sets_categorical_dtypes():
    out = prepare(make_swing_data())
    for col in ["pitch_type", "stand", "p_throws", "prev_pitch_type"]:
        assert isinstance(out[col].dtype, pd.CategoricalDtype), col


def test_get_swings_parses_game_date():
    out = prepare(make_swing_data())
    assert pd.api.types.is_datetime64_any_dtype(out["game_date"])


def test_split_by_date_separates_before_and_after_cutoff():
    swings = prepare(make_swing_data())
    train, test = split_by_date(swings)
    cutoff = pd.Timestamp(TRAIN_CUTOFF)
    assert (train["game_date"] < cutoff).all()
    assert (test["game_date"] >= cutoff).all()
    assert len(train) == 2 and len(test) == 2
    assert len(train) + len(test) == len(swings)