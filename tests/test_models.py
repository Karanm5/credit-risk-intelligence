"""Unit tests for the credit risk ensemble model."""

import numpy as np
import pandas as pd
import pytest

from src.models.training import CreditRiskEnsemble, ModelMetrics

FAST_PARAMS = {
    "xgb": {"n_estimators": 20, "max_depth": 3},
    "lgb": {"n_estimators": 20, "max_depth": 3},
}


@pytest.fixture(scope="module")
def training_data():
    """Small separable dataset for fast model tests."""
    rng = np.random.default_rng(42)
    n = 600
    X = pd.DataFrame(
        rng.normal(size=(n, 6)), columns=[f"feature_{i}" for i in range(6)]
    )
    logits = 2.0 * X["feature_0"] - 1.5 * X["feature_1"]
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logits))).astype(int)
    return X, y


@pytest.fixture(scope="module")
def fitted_model(training_data):
    X, y = training_data
    return CreditRiskEnsemble(params=FAST_PARAMS).fit(X, y)


class TestCreditRiskEnsemble:
    def test_predict_before_fit_raises(self):
        model = CreditRiskEnsemble()
        with pytest.raises(RuntimeError):
            model.predict_proba(pd.DataFrame({"feature_0": [0.0]}))

    def test_predict_proba_in_unit_interval(self, fitted_model, training_data):
        X, _ = training_data
        probs = fitted_model.predict_proba(X.head(50))
        assert probs.shape == (50,)
        assert ((probs >= 0) & (probs <= 1)).all()

    def test_predict_alias_matches_predict_proba(self, fitted_model, training_data):
        X, _ = training_data
        assert np.allclose(
            fitted_model.predict(X.head(10)), fitted_model.predict_proba(X.head(10))
        )

    def test_evaluate_beats_random(self, fitted_model, training_data):
        X, y = training_data
        metrics = fitted_model.evaluate(X, y)
        assert isinstance(metrics, ModelMetrics)
        assert metrics.auc_roc > 0.7
        assert 0 <= metrics.ks_statistic <= 1
        assert metrics.gini_coefficient == pytest.approx(2 * metrics.auc_roc - 1)
        assert "auc_roc" in metrics.to_dict()

    def test_save_and_load_round_trip(self, fitted_model, training_data, tmp_path):
        X, _ = training_data
        fitted_model.save(version="test", model_dir=tmp_path)

        reloaded = CreditRiskEnsemble()
        reloaded.load(version="test", model_dir=tmp_path)

        assert reloaded.feature_names == fitted_model.feature_names
        assert np.allclose(
            reloaded.predict_proba(X.head(20)), fitted_model.predict_proba(X.head(20))
        )

    def test_load_missing_artefact_raises(self, tmp_path):
        model = CreditRiskEnsemble()
        with pytest.raises(FileNotFoundError):
            model.load(version="does-not-exist", model_dir=tmp_path)
