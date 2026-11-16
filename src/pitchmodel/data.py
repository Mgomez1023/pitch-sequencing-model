from pathlib import Path
import pandas as pd
from pitchmodel.config import DATA_DIR, RAW_COLUMNS

def load_raw(data_dir: Path=DATA_DIR) -> pd.DataFrame:
    """ Load every monthly parquet file and return one DataFrame """
    files = sorted(data_dir.glob("statcast_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No statcast_*.parquet files found in {data_dir}")
    return pd.concat(
        (pd.read_parquet(file, columns=RAW_COLUMNS) for file in files),
        ignore_index=True,
    )

def filter_regular_season(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["game_type"] == "R"].copy()

def load_pitches(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """ Load, Filter to the regular season, sort chronologically, reset the index """
    data = load_raw(data_dir)
    data = filter_regular_season(data)
    return data.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)

