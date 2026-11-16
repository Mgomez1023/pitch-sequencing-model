import pandas as pd
from pitchmodel.config import FEATURES

def predict_whiff_probability(model, pitch: dict) -> float:
    """Probability the batter whiffs, given the pitch's features and that he swings."""
    missing = [f for f in FEATURES 
               if f not in pitch]
    if missing:
        raise ValueError(f"Missing fields: {missing}")

    row = pd.DataFrame([pitch])[FEATURES]
    return float(model.predict_proba(row)[0,1])