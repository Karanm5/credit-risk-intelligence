"""
Pytest configuration and shared fixtures.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture(scope="session")
def sample_transactions() -> pd.DataFrame:
    """
    Generate sample transaction data for testing.
    Session-scoped for efficiency.
    """
    np.random.seed(42)
    n_transactions = 500
    n_customers = 50
    n_merchants = 20
    
    customers = [f"CUST_{str(i).zfill(4)}" for i in range(n_customers)]
    merchants = [f"MERCH_{str(i).zfill(4)}" for i in range(n_merchants)]
    categories = ["grocery", "restaurant", "retail", "travel", "entertainment", "utilities"]
    
    base_time = datetime(2024, 1, 1)
    
    data = {
        "transaction_id": [f"TXN_{str(i).zfill(6)}" for i in range(n_transactions)],
        "customer_id": np.random.choice(customers, n_transactions),
        "merchant_id": np.random.choice(merchants, n_transactions),
        "merchant_category": np.random.choice(categories, n_transactions),
        "amount": np.random.exponential(150, n_transactions).round(2),
        "timestamp": [
            base_time + timedelta(
                days=np.random.randint(0, 90),
                hours=np.random.randint(0, 24),
                minutes=np.random.randint(0, 60)
            )
            for _ in range(n_transactions)
        ],
        "is_online": np.random.choice([True, False], n_transactions, p=[0.4, 0.6]),
        "currency": "GBP"
    }
    
    df = pd.DataFrame(data)
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    return df


@pytest.fixture(scope="session")
def sample_customers() -> pd.DataFrame:
    """Generate sample customer data."""
    np.random.seed(42)
    n_customers = 50
    
    data = {
        "customer_id": [f"CUST_{str(i).zfill(4)}" for i in range(n_customers)],
        "registration_date": [
            datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 365))
            for _ in range(n_customers)
        ],
        "age_band": np.random.choice(["18-25", "26-35", "36-45", "46-55", "55+"], n_customers),
        "income_band": np.random.choice(["low", "medium", "high"], n_customers, p=[0.3, 0.5, 0.2]),
        "region": np.random.choice(["London", "South East", "Midlands", "North", "Scotland"], n_customers)
    }
    
    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def sample_labels() -> pd.DataFrame:
    """Generate sample labels (defaults) for training."""
    np.random.seed(42)
    n_customers = 50
    
    # 10% default rate
    is_default = np.random.choice([0, 1], n_customers, p=[0.9, 0.1])
    
    data = {
        "customer_id": [f"CUST_{str(i).zfill(4)}" for i in range(n_customers)],
        "is_default": is_default,
        "event_date": datetime(2024, 3, 31)
    }
    
    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def merchant_risk_scores() -> Dict[str, float]:
    """Generate sample merchant risk scores."""
    np.random.seed(42)
    n_merchants = 20
    
    return {
        f"MERCH_{str(i).zfill(4)}": np.random.beta(2, 5)  # Skewed towards low risk
        for i in range(n_merchants)
    }


@pytest.fixture
def mock_snowflake_config() -> Dict[str, Any]:
    """Mock Snowflake configuration for testing."""
    return {
        "account": "test_account",
        "user": "test_user",
        "password": "test_password",
        "warehouse": "TEST_WH",
        "database": "TEST_DB",
        "schema": "TEST_SCHEMA",
        "role": "TEST_ROLE"
    }


@pytest.fixture
def mock_feature_vector() -> np.ndarray:
    """Generate a mock feature vector for model testing."""
    np.random.seed(42)
    n_features = 30
    return np.random.randn(1, n_features)


@pytest.fixture(autouse=True)
def env_setup(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "test_account")
    monkeypatch.setenv("SNOWFLAKE_USER", "test_user")
    monkeypatch.setenv("SNOWFLAKE_PASSWORD", "test_password")
    monkeypatch.setenv("DATABRICKS_HOST", "https://test.databricks.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", "test_token")
    monkeypatch.setenv("DATABRICKS_CLUSTER_ID", "test_cluster")


# Markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
