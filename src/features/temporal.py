"""
Temporal feature engineering for credit risk assessment.
Computes time-based features from transaction data.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TemporalWindow:
    """Configuration for a temporal aggregation window."""

    name: str
    seconds: int

    @classmethod
    def from_string(cls, window_str: str) -> "TemporalWindow":
        """Parse window string like '1h', '7d', '30d' into TemporalWindow."""
        unit = window_str[-1]
        value = int(window_str[:-1])

        multipliers = {
            "h": 3600,
            "d": 86400,
            "w": 604800,
            "m": 2592000,  # 30 days
        }

        if unit not in multipliers:
            raise ValueError(f"Unknown time unit: {unit}")

        return cls(name=window_str, seconds=value * multipliers[unit])


def _rolling_time_agg(
    values: pd.Series,
    timestamps: pd.Series,
    window_seconds: int,
    agg: str,
    min_periods: int = 1,
) -> pd.Series:
    """
    Time-windowed rolling aggregate for a single customer's transactions.

    Assumes `timestamps` are sorted ascending within the group (guaranteed by
    the sort in compute_all_features). Uses a DatetimeIndex so pandas performs
    a true time-based window rather than a fixed row count.

    Args:
        values: Series of values to aggregate (original DataFrame index).
        timestamps: Series of datetimes aligned with `values`.
        window_seconds: Window length in seconds.
        agg: Aggregation name, e.g. 'mean', 'max', 'std', 'sum', 'count'.
        min_periods: Minimum observations in window required for a value.

    Returns:
        Series of aggregated values, indexed like `values`.
    """
    indexed = pd.Series(
        values.to_numpy(),
        index=pd.DatetimeIndex(timestamps.to_numpy()),
    )
    rolling = indexed.rolling(f"{window_seconds}s", min_periods=min_periods)
    aggregated = getattr(rolling, agg)()
    return pd.Series(aggregated.to_numpy(), index=values.index)


class TemporalFeatureEngineer:
    """
    Computes temporal features from transaction data.

    Features include:
    - Transaction velocity (count per window)
    - Amount statistics (mean, std, max, min)
    - Rate of change metrics
    - Temporal patterns (time of day, day of week)
    """

    DEFAULT_WINDOWS = ["1h", "6h", "24h", "7d", "30d"]

    def __init__(self, windows: Optional[List[str]] = None):
        """
        Initialise temporal feature engineer.

        Args:
            windows: List of time windows (e.g., ['1h', '24h', '7d'])
        """
        window_strs = windows or self.DEFAULT_WINDOWS
        self.windows = [TemporalWindow.from_string(w) for w in window_strs]

    def compute_all_features(
        self,
        transactions: pd.DataFrame,
        customer_id_col: str = "customer_id",
        timestamp_col: str = "timestamp",
        amount_col: str = "amount",
    ) -> pd.DataFrame:
        """
        Compute all temporal features for transaction data.

        Args:
            transactions: DataFrame with transaction data
            customer_id_col: Name of customer ID column
            timestamp_col: Name of timestamp column
            amount_col: Name of amount column

        Returns:
            DataFrame with computed features
        """
        # Ensure timestamp is datetime
        transactions = transactions.copy()
        transactions[timestamp_col] = pd.to_datetime(transactions[timestamp_col])

        # Sort by customer and timestamp (required for time-based rolling)
        transactions = transactions.sort_values([customer_id_col, timestamp_col])

        # Compute features
        velocity_features = self._compute_velocity_features(
            transactions, customer_id_col, timestamp_col
        )

        amount_features = self._compute_amount_features(
            transactions, customer_id_col, timestamp_col, amount_col
        )

        volatility_features = self._compute_volatility_features(
            transactions, customer_id_col, timestamp_col, amount_col
        )

        pattern_features = self._compute_pattern_features(
            transactions, customer_id_col, timestamp_col
        )

        # Assemble on the shared index (all frames preserve the original index)
        features = transactions[[customer_id_col, timestamp_col]].copy()

        for feature_df in [
            velocity_features,
            amount_features,
            volatility_features,
            pattern_features,
        ]:
            feature_cols = [
                c
                for c in feature_df.columns
                if c not in (customer_id_col, timestamp_col)
            ]
            features = features.join(feature_df[feature_cols])

        return features

    def _compute_velocity_features(
        self,
        df: pd.DataFrame,
        customer_id_col: str,
        timestamp_col: str,
    ) -> pd.DataFrame:
        """Compute transaction velocity features."""

        result = df[[customer_id_col, timestamp_col]].copy()
        grouped = df.groupby(customer_id_col, group_keys=False, sort=False)

        for window in self.windows:
            # Count transactions in the trailing time window (inclusive of current)
            result[f"txn_count_{window.name}"] = grouped[timestamp_col].transform(
                lambda x, w=window: _rolling_time_agg(
                    pd.Series(1.0, index=x.index), x, w.seconds, "sum"
                )
            )

        # Compute velocity change (current vs previous transaction's window count)
        for window in self.windows:
            col_name = f"txn_count_{window.name}"
            prev = (
                result.groupby(customer_id_col, sort=False)[col_name]
                .shift(1)
                .fillna(1)  # Avoid division by zero for first transaction
            )
            result[f"txn_velocity_change_{window.name}"] = (
                (result[col_name] - prev) / prev
            ).replace([np.inf, -np.inf], 0)

        return result

    def _compute_amount_features(
        self,
        df: pd.DataFrame,
        customer_id_col: str,
        timestamp_col: str,
        amount_col: str,
    ) -> pd.DataFrame:
        """Compute amount-based features over trailing time windows."""

        result = df[[customer_id_col, timestamp_col]].copy()
        grouped = df.groupby(customer_id_col, group_keys=False, sort=False)

        for window in self.windows:
            specs = [
                (f"avg_amount_{window.name}", "mean", 1),
                (f"max_amount_{window.name}", "max", 1),
                (f"std_amount_{window.name}", "std", 2),
                (f"total_amount_{window.name}", "sum", 1),
            ]
            for col, agg, min_periods in specs:
                result[col] = grouped.apply(
                    lambda g, a=agg, mp=min_periods, w=window: _rolling_time_agg(
                        g[amount_col], g[timestamp_col], w.seconds, a, mp
                    )
                )
            result[f"std_amount_{window.name}"] = result[
                f"std_amount_{window.name}"
            ].fillna(0)

        return result

    def _compute_volatility_features(
        self,
        df: pd.DataFrame,
        customer_id_col: str,
        timestamp_col: str,
        amount_col: str,
    ) -> pd.DataFrame:
        """Compute spending volatility features."""

        result = df[[customer_id_col, timestamp_col]].copy()
        grouped = df.groupby(customer_id_col, group_keys=False, sort=False)

        # Coefficient of variation (std/mean) - measures relative volatility
        for window in self.windows:
            rolling_mean = grouped.apply(
                lambda g, w=window: _rolling_time_agg(
                    g[amount_col], g[timestamp_col], w.seconds, "mean", 1
                )
            )
            rolling_std = grouped.apply(
                lambda g, w=window: _rolling_time_agg(
                    g[amount_col], g[timestamp_col], w.seconds, "std", 2
                )
            ).fillna(0)

            result[f"spending_cv_{window.name}"] = (
                rolling_std / rolling_mean.replace(0, 1)
            ).fillna(0)

        # Amount deviation from personal average
        by_customer = df.groupby(customer_id_col, sort=False)[amount_col]
        personal_avg = by_customer.transform("mean")
        personal_std = by_customer.transform("std").fillna(1).replace(0, 1)

        result["amount_zscore"] = (
            (df[amount_col] - personal_avg) / personal_std
        ).fillna(0)

        return result

    def _compute_pattern_features(
        self,
        df: pd.DataFrame,
        customer_id_col: str,
        timestamp_col: str,
    ) -> pd.DataFrame:
        """Compute temporal pattern features."""

        result = df[[customer_id_col, timestamp_col]].copy()

        # Extract time components
        result["hour_of_day"] = df[timestamp_col].dt.hour
        result["day_of_week"] = df[timestamp_col].dt.dayofweek
        result["day_of_month"] = df[timestamp_col].dt.day
        result["is_weekend"] = result["day_of_week"].isin([5, 6]).astype(int)
        result["is_night"] = (
            result["hour_of_day"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
        )
        result["is_business_hours"] = (
            result["hour_of_day"].isin(range(9, 18)).astype(int)
        )

        # Rolling ratios over day-plus windows
        grouped = result.groupby(customer_id_col, group_keys=False, sort=False)

        for window in [w for w in self.windows if w.seconds >= 86400]:
            result[f"weekend_ratio_{window.name}"] = grouped.apply(
                lambda g, w=window: _rolling_time_agg(
                    g["is_weekend"].astype(float),
                    g[timestamp_col],
                    w.seconds,
                    "mean",
                    1,
                )
            )
            result[f"night_ratio_{window.name}"] = grouped.apply(
                lambda g, w=window: _rolling_time_agg(
                    g["is_night"].astype(float), g[timestamp_col], w.seconds, "mean", 1
                )
            )

        return result

    def compute_customer_summary(
        self,
        transactions: pd.DataFrame,
        customer_id_col: str = "customer_id",
        timestamp_col: str = "timestamp",
        amount_col: str = "amount",
    ) -> pd.DataFrame:
        """
        Compute summary features at customer level (latest values).

        Args:
            transactions: Transaction DataFrame
            customer_id_col: Customer ID column name
            timestamp_col: Timestamp column name
            amount_col: Amount column name

        Returns:
            DataFrame with one row per customer containing latest features
        """
        # Compute all features
        all_features = self.compute_all_features(
            transactions, customer_id_col, timestamp_col, amount_col
        )

        # Get latest row per customer
        latest_idx = all_features.groupby(customer_id_col)[timestamp_col].idxmax()

        summary = all_features.loc[latest_idx].reset_index(drop=True)

        # Add account age
        transactions = transactions.copy()
        transactions[timestamp_col] = pd.to_datetime(transactions[timestamp_col])
        first_txn = transactions.groupby(customer_id_col)[timestamp_col].min()
        summary = summary.merge(
            first_txn.rename("first_transaction").reset_index(),
            on=customer_id_col,
        )
        summary["account_age_days"] = (
            summary[timestamp_col] - summary["first_transaction"]
        ).dt.days

        summary.drop(columns=["first_transaction"], inplace=True)

        return summary
