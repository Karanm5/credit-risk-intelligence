"""
Snowflake integration for the Credit Risk Intelligence Platform.

Provides a thin connection wrapper and a FeatureStore abstraction for
retrieving pre-computed features at inference and training time.

The snowflake-connector-python dependency is imported lazily so that this
module can be imported (and unit tested) without Snowflake installed.
"""

import logging
from typing import Any, List, Optional

import pandas as pd

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class SnowflakeConnectionError(RuntimeError):
    """Raised when a Snowflake connection cannot be established."""


class SnowflakeConnector:
    """Manages a connection to Snowflake and executes queries."""

    def __init__(self) -> None:
        self.settings = get_settings().snowflake
        self._connection: Optional[Any] = None

    def connect(self) -> Any:
        """
        Establish (or reuse) a Snowflake connection.

        Raises:
            SnowflakeConnectionError: If credentials are missing or the
                connection attempt fails.
        """
        if self._connection is not None:
            return self._connection

        if not self.settings.is_configured:
            raise SnowflakeConnectionError(
                "Snowflake credentials are not configured. "
                "Set SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER and SNOWFLAKE_PASSWORD."
            )

        try:
            import snowflake.connector
        except ImportError as exc:
            raise SnowflakeConnectionError(
                "snowflake-connector-python is not installed."
            ) from exc

        try:
            self._connection = snowflake.connector.connect(
                account=self.settings.account,
                user=self.settings.user,
                password=self.settings.password.get_secret_value(),
                warehouse=self.settings.warehouse,
                database=self.settings.database,
                schema=self.settings.schema_name,
                role=self.settings.role,
            )
        except Exception as exc:
            raise SnowflakeConnectionError(
                f"Failed to connect to Snowflake: {exc}"
            ) from exc

        logger.info("Connected to Snowflake account %s", self.settings.account)
        return self._connection

    def query(self, sql: str, params: Optional[List[Any]] = None) -> pd.DataFrame:
        """Execute a (parameterised) query and return the result as a DataFrame."""
        connection = self.connect()
        cursor = connection.cursor()
        try:
            cursor.execute(sql, params)
            return cursor.fetch_pandas_all()
        finally:
            cursor.close()

    def close(self) -> None:
        """Close the connection if open."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None


class FeatureStore:
    """
    Retrieves pre-computed features from the Snowflake feature store.

    Features are materialised into the GOLD layer by the feature pipelines
    (see src/features) and read here for training and real-time scoring.
    """

    FEATURE_TABLE = "CUSTOMER_FEATURES"

    def __init__(self, connector: Optional[SnowflakeConnector] = None) -> None:
        self.connector = connector or SnowflakeConnector()

    def is_connected(self) -> bool:
        """Whether a live Snowflake connection can be established."""
        try:
            self.connector.connect()
            return True
        except SnowflakeConnectionError:
            return False

    def get_features(self, customer_id: str) -> pd.DataFrame:
        """
        Fetch the latest feature vector for a single customer.

        Args:
            customer_id: Customer identifier.

        Returns:
            Single-row DataFrame of features.

        Raises:
            KeyError: If the customer has no feature record.
        """
        frame = self.get_features_batch([customer_id])
        if frame.empty:
            raise KeyError(f"No features found for customer {customer_id}")
        return frame

    def get_features_for_customer(self, customer_id: str) -> Optional[dict]:
        """
        Fetch the latest feature vector for a customer as a dict.

        Returns:
            Feature dict, or None if the customer has no record.
        """
        frame = self.get_features_batch([customer_id])
        if frame.empty:
            return None
        return frame.iloc[0].to_dict()

    def get_features_batch(self, customer_ids: List[str]) -> pd.DataFrame:
        """
        Fetch latest feature vectors for multiple customers.

        Args:
            customer_ids: Customer identifiers.

        Returns:
            DataFrame with one row per found customer.
        """
        placeholders = ", ".join(["%s"] * len(customer_ids))
        # Safe: table name is a class constant and customer IDs are bound params
        sql = (  # nosec B608
            f"SELECT * FROM {self.FEATURE_TABLE} "
            f"WHERE CUSTOMER_ID IN ({placeholders}) "
            "QUALIFY ROW_NUMBER() OVER ("
            "PARTITION BY CUSTOMER_ID ORDER BY FEATURE_TIMESTAMP DESC) = 1"
        )
        return self.connector.query(sql, params=list(customer_ids))
