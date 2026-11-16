import pandas as pd
from pitchmodel.config import WHIFF_DESCRIPTIONS

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
