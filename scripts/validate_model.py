"""
CI model validation: trains the ensemble on synthetic data and asserts the
pipeline produces sane, better-than-random results end to end.

This is a smoke test of the modelling code path, not a benchmark - it proves
fit/predict/evaluate/save/load all work before anything is deployed.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.training import CreditRiskEnsemble

MIN_AUC = 0.75  # Synthetic data is easily separable; well below this means breakage


def make_synthetic_data(n: int = 2000, seed: int = 42):
    """Generate a separable binary classification dataset."""
    rng = np.random.default_rng(seed)
    n_features = 12
    X = pd.DataFrame(
        rng.normal(size=(n, n_features)),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    logits = X["feature_0"] * 1.5 - X["feature_1"] + 0.5 * X["feature_2"]
    probs = 1 / (1 + np.exp(-logits))
    y = (rng.uniform(size=n) < probs).astype(int)
    return X, y


def main() -> int:
    X, y = make_synthetic_data()
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]

    print("Training ensemble on synthetic data...")
    model = CreditRiskEnsemble(
        params={
            "xgb": {"n_estimators": 50},
            "lgb": {"n_estimators": 50},
        }
    )
    model.fit(X_train, y_train)

    metrics = model.evaluate(X_test, y_test)
    print(f"Metrics: {metrics.to_dict()}")

    if metrics.auc_roc < MIN_AUC:
        print(f"FAIL: AUC {metrics.auc_roc:.3f} below threshold {MIN_AUC}")
        return 1

    # Round-trip persistence
    with tempfile.TemporaryDirectory() as tmp:
        model.save(version="ci", model_dir=Path(tmp))
        reloaded = CreditRiskEnsemble()
        reloaded.load(version="ci", model_dir=Path(tmp))
        original = model.predict_proba(X_test.head(10))
        restored = reloaded.predict_proba(X_test.head(10))
        if not np.allclose(original, restored):
            print("FAIL: predictions differ after save/load round-trip")
            return 1

    print("Model validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
