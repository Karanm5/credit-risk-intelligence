"""Unit tests for the Snowflake connector and feature store (mock-based)."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.data.snowflake_connector import (
    FeatureStore,
    SnowflakeConnectionError,
    SnowflakeConnector,
)


class TestSnowflakeConnector:
    def test_connect_without_credentials_raises(self, monkeypatch):
        monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "")
        monkeypatch.setenv("SNOWFLAKE_USER", "")
        monkeypatch.setenv("SNOWFLAKE_PASSWORD", "")
        # Rebuild settings so blanked env vars take effect
        from src.config import settings as settings_module

        settings_module.get_settings.cache_clear()

        connector = SnowflakeConnector()
        with pytest.raises(SnowflakeConnectionError):
            connector.connect()

        settings_module.get_settings.cache_clear()

    def test_close_is_safe_when_not_connected(self):
        SnowflakeConnector().close()  # must not raise


class TestFeatureStore:
    @pytest.fixture
    def mock_connector(self):
        connector = MagicMock(spec=SnowflakeConnector)
        connector.query.return_value = pd.DataFrame(
            {
                "CUSTOMER_ID": ["CUST_001"],
                "TXN_COUNT_24H": [5],
                "AVG_AMOUNT_7D": [120.5],
            }
        )
        return connector

    def test_get_features_returns_single_row(self, mock_connector):
        store = FeatureStore(connector=mock_connector)
        frame = store.get_features("CUST_001")
        assert len(frame) == 1
        mock_connector.query.assert_called_once()
        # Customer IDs must be passed as bound parameters, not interpolated
        sql, kwargs = (
            mock_connector.query.call_args.args[0],
            mock_connector.query.call_args.kwargs,
        )
        assert "CUST_001" not in sql
        assert kwargs["params"] == ["CUST_001"]

    def test_get_features_missing_customer_raises(self, mock_connector):
        mock_connector.query.return_value = pd.DataFrame()
        store = FeatureStore(connector=mock_connector)
        with pytest.raises(KeyError):
            store.get_features("CUST_MISSING")

    def test_get_features_for_customer_dict(self, mock_connector):
        store = FeatureStore(connector=mock_connector)
        features = store.get_features_for_customer("CUST_001")
        assert features["TXN_COUNT_24H"] == 5

    def test_get_features_for_customer_missing_returns_none(self, mock_connector):
        mock_connector.query.return_value = pd.DataFrame()
        store = FeatureStore(connector=mock_connector)
        assert store.get_features_for_customer("CUST_MISSING") is None

    def test_is_connected_false_without_credentials(self, mock_connector):
        mock_connector.connect.side_effect = SnowflakeConnectionError("no creds")
        store = FeatureStore(connector=mock_connector)
        assert store.is_connected() is False
