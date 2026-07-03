"""
Unit tests for feature engineering modules.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.features.graph import GraphFeatureEngineer, TransactionGraphBuilder
from src.features.temporal import TemporalFeatureEngineer, TemporalWindow


class TestTemporalWindow:
    """Tests for TemporalWindow parsing."""

    def test_parse_hours(self):
        window = TemporalWindow.from_string("1h")
        assert window.name == "1h"
        assert window.seconds == 3600

    def test_parse_days(self):
        window = TemporalWindow.from_string("7d")
        assert window.name == "7d"
        assert window.seconds == 7 * 86400

    def test_parse_weeks(self):
        window = TemporalWindow.from_string("2w")
        assert window.name == "2w"
        assert window.seconds == 2 * 604800

    def test_invalid_unit(self):
        with pytest.raises(ValueError):
            TemporalWindow.from_string("5x")


class TestTemporalFeatureEngineer:
    """Tests for temporal feature engineering."""

    @pytest.fixture
    def sample_transactions(self):
        """Create sample transaction data."""
        np.random.seed(42)
        n_transactions = 100

        base_time = datetime(2024, 1, 1)

        data = {
            "transaction_id": [f"TXN_{i}" for i in range(n_transactions)],
            "customer_id": np.random.choice(["C001", "C002", "C003"], n_transactions),
            "merchant_id": np.random.choice(
                ["M001", "M002", "M003", "M004"], n_transactions
            ),
            "amount": np.random.exponential(100, n_transactions),
            "timestamp": [
                base_time + timedelta(hours=np.random.randint(0, 720))
                for _ in range(n_transactions)
            ],
        }

        return pd.DataFrame(data)

    @pytest.fixture
    def engineer(self):
        """Create feature engineer instance."""
        return TemporalFeatureEngineer(windows=["1h", "24h", "7d"])

    def test_compute_all_features(self, engineer, sample_transactions):
        """Test that all features are computed."""
        features = engineer.compute_all_features(sample_transactions)

        # Check that features DataFrame has correct shape
        assert len(features) == len(sample_transactions)

        # Check velocity features exist
        assert "txn_count_1h" in features.columns
        assert "txn_count_24h" in features.columns
        assert "txn_count_7d" in features.columns

        # Check amount features exist
        assert "avg_amount_1h" in features.columns
        assert "max_amount_7d" in features.columns

    def test_velocity_features_positive(self, engineer, sample_transactions):
        """Test that velocity features are non-negative."""
        features = engineer.compute_all_features(sample_transactions)

        for col in features.columns:
            if "txn_count" in col:
                assert (features[col] >= 0).all()

    def test_pattern_features(self, engineer, sample_transactions):
        """Test temporal pattern features."""
        features = engineer.compute_all_features(sample_transactions)

        # Hour of day should be 0-23
        assert features["hour_of_day"].min() >= 0
        assert features["hour_of_day"].max() <= 23

        # Day of week should be 0-6
        assert features["day_of_week"].min() >= 0
        assert features["day_of_week"].max() <= 6

        # Weekend flag should be binary
        assert features["is_weekend"].isin([0, 1]).all()

    def test_customer_summary(self, engineer, sample_transactions):
        """Test customer-level summary."""
        summary = engineer.compute_customer_summary(sample_transactions)

        # Should have one row per customer
        assert len(summary) == sample_transactions["customer_id"].nunique()

        # Account age should be positive
        assert (summary["account_age_days"] >= 0).all()


class TestGraphFeatureEngineer:
    """Tests for graph-based feature engineering."""

    @pytest.fixture
    def sample_transactions(self):
        """Create sample transaction data for graph."""
        data = {
            "customer_id": ["C001", "C001", "C001", "C002", "C002", "C003"],
            "merchant_id": ["M001", "M002", "M001", "M001", "M003", "M002"],
            "amount": [100, 200, 150, 300, 50, 75],
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="D"),
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def graph_builder(self):
        """Create graph builder instance."""
        return TransactionGraphBuilder()

    @pytest.fixture
    def graph_engineer(self):
        """Create graph feature engineer instance."""
        return GraphFeatureEngineer()

    def test_build_graph(self, graph_builder, sample_transactions):
        """Test graph construction."""
        graph = graph_builder.build_graph(sample_transactions)

        # Check nodes exist
        assert graph.number_of_nodes() > 0

        # Check edges exist
        assert graph.number_of_edges() > 0

        # Check customer and merchant nodes
        assert len(graph_builder.customer_nodes) == 3
        assert len(graph_builder.merchant_nodes) == 3

    def test_compute_centrality_features(self, graph_engineer, sample_transactions):
        """Test centrality feature computation."""
        features = graph_engineer.compute_all_features(sample_transactions)

        # Check features exist
        assert "pagerank_score" in features.columns
        assert "degree_centrality" in features.columns
        assert "weighted_degree" in features.columns

        # PageRank should be positive
        assert (features["pagerank_score"] >= 0).all()
        assert (features["pagerank_score"] <= 1).all()

    def test_compute_community_features(self, graph_engineer, sample_transactions):
        """Test community detection features."""
        features = graph_engineer.compute_all_features(sample_transactions)

        # Check features exist
        assert "community_id" in features.columns
        assert "community_size" in features.columns

        # Community size should be positive
        assert (features["community_size"] >= 0).all()

    def test_risk_propagation(self, graph_engineer, sample_transactions):
        """Test risk propagation features."""
        merchant_risk = {"M001": 0.8, "M002": 0.3, "M003": 0.5}

        features = graph_engineer.compute_all_features(
            sample_transactions, merchant_risk_scores=merchant_risk
        )

        # Check features exist
        assert "merchant_risk_exposure" in features.columns
        assert "high_risk_merchant_ratio" in features.columns

        # C001 shops at M001 (high risk) - should have higher exposure
        c001_exposure = features[features["customer_id"] == "C001"][
            "merchant_risk_exposure"
        ].values[0]
        assert c001_exposure > 0


class TestFeatureIntegration:
    """Integration tests for feature pipeline."""

    @pytest.fixture
    def full_transaction_data(self):
        """Create realistic transaction dataset."""
        np.random.seed(42)
        n_customers = 50
        n_merchants = 20
        n_transactions = 1000

        customers = [f"C{str(i).zfill(4)}" for i in range(n_customers)]
        merchants = [f"M{str(i).zfill(4)}" for i in range(n_merchants)]

        base_time = datetime(2024, 1, 1)

        data = {
            "transaction_id": [f"TXN_{i}" for i in range(n_transactions)],
            "customer_id": np.random.choice(customers, n_transactions),
            "merchant_id": np.random.choice(merchants, n_transactions),
            "amount": np.random.exponential(150, n_transactions),
            "timestamp": [
                base_time + timedelta(hours=np.random.randint(0, 2160))
                for _ in range(n_transactions)
            ],
        }

        return pd.DataFrame(data)

    def test_full_feature_pipeline(self, full_transaction_data):
        """Test complete feature engineering pipeline."""
        # Temporal features
        temporal_eng = TemporalFeatureEngineer()
        temporal_features = temporal_eng.compute_customer_summary(full_transaction_data)

        # Graph features
        graph_eng = GraphFeatureEngineer()
        graph_features = graph_eng.compute_all_features(full_transaction_data)

        # Merge features
        combined = temporal_features.merge(
            graph_features, on="customer_id", how="outer"
        )

        # Check we have features for all customers
        assert len(combined) == full_transaction_data["customer_id"].nunique()

        # Check no NaN in critical columns
        assert combined["customer_id"].notna().all()

    def test_feature_reproducibility(self, full_transaction_data):
        """Test that features are reproducible."""
        temporal_eng = TemporalFeatureEngineer()

        features1 = temporal_eng.compute_customer_summary(full_transaction_data)
        features2 = temporal_eng.compute_customer_summary(full_transaction_data)

        # Sort both DataFrames
        features1 = features1.sort_values("customer_id").reset_index(drop=True)
        features2 = features2.sort_values("customer_id").reset_index(drop=True)

        # Compare numeric columns
        numeric_cols = features1.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            np.testing.assert_array_almost_equal(
                features1[col].values, features2[col].values, decimal=5
            )
