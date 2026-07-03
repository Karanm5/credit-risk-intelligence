"""Unit tests for the FastAPI serving layer."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import src.serving.api as api_module
from src.models.training import CreditRiskEnsemble

FAST_PARAMS = {
    "xgb": {"n_estimators": 20, "max_depth": 3},
    "lgb": {"n_estimators": 20, "max_depth": 3},
}


class StubFeatureStore:
    """In-memory feature store standing in for Snowflake."""

    def __init__(self, feature_names):
        self.feature_names = feature_names

    def _row(self, customer_id):
        rng = np.random.default_rng(abs(hash(customer_id)) % (2**32))
        row = {name: float(rng.normal()) for name in self.feature_names}
        row["customer_id"] = customer_id
        return row

    def get_features_for_customer(self, customer_id):
        if customer_id == "CUST_MISSING":
            return None
        return self._row(customer_id)

    def get_features_batch(self, customer_ids):
        rows = [self._row(cid) for cid in customer_ids if cid != "CUST_MISSING"]
        return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def trained_model():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.normal(size=(400, 5)), columns=[f"feature_{i}" for i in range(5)]
    )
    y = (X["feature_0"] > 0).astype(int).to_numpy()
    return CreditRiskEnsemble(params=FAST_PARAMS).fit(X, y)


@pytest.fixture
def degraded_client():
    """Client with no model loaded (fresh deployment state)."""
    api_module.model = None
    api_module.feature_store = None
    api_module.shap_explainer = None
    return TestClient(api_module.app)


@pytest.fixture
def ready_client(trained_model):
    """Client with a trained model and stub feature store injected."""
    api_module.model = trained_model
    api_module.feature_store = StubFeatureStore(trained_model.feature_names)
    api_module.shap_explainer = None  # SHAP path returns empty explanations
    yield TestClient(api_module.app)
    api_module.model = None
    api_module.feature_store = None


class TestHealthAndMetrics:
    def test_health_reports_degraded_without_model(self, degraded_client):
        response = degraded_client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "degraded"
        assert body["model_loaded"] is False

    def test_health_reports_healthy_with_model(self, ready_client):
        body = ready_client.get("/health").json()
        assert body["status"] == "healthy"
        assert body["model_loaded"] is True

    def test_metrics_endpoint_serves_prometheus_format(self, degraded_client):
        response = degraded_client.get("/metrics")
        assert response.status_code == 200
        assert "credit_risk" in response.text


class TestPredict:
    def test_predict_returns_503_when_not_ready(self, degraded_client):
        response = degraded_client.post("/predict", json={"customer_id": "CUST_001"})
        assert response.status_code == 503

    def test_predict_returns_valid_score(self, ready_client):
        response = ready_client.post(
            "/predict",
            json={"customer_id": "CUST_001", "include_explanation": False},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["customer_id"] == "CUST_001"
        assert 0.0 <= body["risk_score"] <= 1.0
        assert body["risk_level"] in {"low", "medium", "high"}

    def test_predict_unknown_customer_returns_404(self, ready_client):
        response = ready_client.post("/predict", json={"customer_id": "CUST_MISSING"})
        assert response.status_code == 404

    def test_predict_validation_error_on_missing_field(self, ready_client):
        response = ready_client.post("/predict", json={})
        assert response.status_code == 422
