import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss
import joblib, json
from pitchmodel.config import FEATURES, RANDOM_STATE, MODEL_PATH, METRICS_PATH
from pitchmodel.data import load_pitches
from pitchmodel.features import add_whiff_target, add_sequence_features, get_swings, split_by_date

def fit_model(train, features=FEATURES, random_state=RANDOM_STATE) -> HistGradientBoostingClassifier:
    """Fit the whiff model on the training swings."""
    model = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=random_state)
    X_train = train[features]
    y_train = train["whiff"]

    model.fit(X_train, y_train)

    return model

def evaluate(model, train, test, features=FEATURES) -> dict[str, float]:
    """Return auc, log_loss and baseline_log_loss on the test set."""
    X_test = test[features]
    y_test = test["whiff"]
    y_train = train["whiff"]
    baseline_probs = np.full(len(y_test), y_train.mean())
    probs = model.predict_proba(X_test)[:,1]

    return {
        "auc": roc_auc_score(y_test, probs),
        "log_loss": log_loss(y_test, probs),
        "baseline_log_loss": log_loss(y_test, baseline_probs)
    }

def save_model(model, path=MODEL_PATH) -> None:
    """Write the model to disk (create the folder if needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path=MODEL_PATH):
    """Read a saved model back."""
    return joblib.load(path)

def main() -> None:
    """Whole pipeline: load -> features -> split -> fit -> evaluate -> save model + metrics.json."""
    df = load_pitches()
    df = add_sequence_features(add_whiff_target(df))
    swings = get_swings(df)

    train, test = split_by_date(swings)

    model = fit_model(train)
    metrics = evaluate(model, train, test)
    save_model(model)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(metrics)


if __name__ == "__main__":
    main()