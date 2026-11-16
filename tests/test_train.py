import numpy as np
import pandas as pd

from pitchmodel.config import CATEGORICAL_FEATURES, FEATURES
from pitchmodel.train import evaluate, fit_model, load_model, save_model


def make_training_frame(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "pitch_type": rng.choice(["FF", "SL", "CH"], n),
        "release_speed": rng.normal(90, 5, n),
        "release_spin_rate": rng.normal(2300, 200, n),
        "pfx_x": rng.normal(0, 1, n),
        "pfx_z": rng.normal(1, 1, n),
        "plate_x": rng.normal(0, 1, n),
        "plate_z": rng.normal(2.5, 1, n),
        "balls": rng.integers(0, 4, n),
        "strikes": rng.integers(0, 3, n),
        "stand": rng.choice(["L", "R"], n),
        "p_throws": rng.choice(["L", "R"], n),
        "whiff": rng.integers(0, 2, n),
    })
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")
    return df


def test_model_outputs_probabilities():
    df = make_training_frame()
    model = fit_model(df)
    probs = model.predict_proba(df[FEATURES])[:, 1]
    assert probs.shape == (len(df),)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_evaluate_returns_expected_metrics():
    df = make_training_frame()
    model = fit_model(df.iloc[:200])
    metrics = evaluate(model, df.iloc[:200], df.iloc[200:])
    assert {"auc", "log_loss", "baseline_log_loss"} <= set(metrics)
    assert all(np.isfinite(v) for v in metrics.values())


def test_saved_model_gives_identical_predictions(tmp_path):
    df = make_training_frame()
    model = fit_model(df)
    path = tmp_path / "model.joblib"
    save_model(model, path)
    reloaded = load_model(path)
    assert np.allclose(model.predict_proba(df[FEATURES]), reloaded.predict_proba(df[FEATURES]))