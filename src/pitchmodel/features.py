import pandas as pd
from pitchmodel.config import (
    WHIFF_DESCRIPTIONS, 
    SWING_DESCRIPTIONS, 
    FEATURES,
    TRAIN_CUTOFF,
    CATEGORICAL_FEATURES
)

def add_whiff_target(df: pd.DataFrame) -> pd.DataFrame:
    """ Add a 0/1 'whiff' column (Savant Definiton) """
    df = df.copy()
    df["whiff"] = df["description"].isin(WHIFF_DESCRIPTIONS).astype(int)
    return df


def add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """ Add previous-pitch features. Expects 'df' sorted chronologically and 'whiff' present """
    group = ["game_pk", "at_bat_number"]
    df = df.copy()
    g = df.groupby(group)

    df["prev_pitch_type"] = g["pitch_type"].shift(1)
    df["prev_speed"] = g["release_speed"].shift(1)
    df["speed_change"] = df["release_speed"] - df["prev_speed"]
    df["plate_z_change"] = df["plate_z"] - g["plate_z"].shift(1)
    df["plate_x_change"] = df["plate_x"] - g["plate_x"].shift(1)
    df["prev_whiff"] = g["whiff"].shift(1)

    df["same_type_as_prev"] = (
        (df["pitch_type"] == df["prev_pitch_type"]).astype(float).where(df["prev_pitch_type"].notna())
    )

    return df

def get_swings(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows where the batter swung, with complete baseline features."""
    swings = df[df["description"].isin(SWING_DESCRIPTIONS)]
    swings = swings.dropna(subset=FEATURES+["whiff"]).copy()

    for feature in CATEGORICAL_FEATURES + ["prev_pitch_type"]:
        if feature in swings.columns:
            swings[feature] = swings[feature].astype("category")


    swings["game_date"] = pd.to_datetime(swings["game_date"])

    return swings

def split_by_date(swings: pd.DataFrame, cutoff: str = TRAIN_CUTOFF) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Return (train, test): games before the cutoff vs. on/after it. """
    cutoff_ts = pd.Timestamp(cutoff)
    train = swings[swings["game_date"] < cutoff_ts]
    test = swings[swings["game_date"] >= cutoff_ts]
    return train, test
