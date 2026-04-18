"""
Temporal feature engineering for credit risk assessment.
Computes time-based features from transaction data.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

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
            'h': 3600,
            'd': 86400,
            'w': 604800,
            'm': 2592000  # 30 days
        }
        
        if unit not in multipliers:
            raise ValueError(f"Unknown time unit: {unit}")
        
        return cls(name=window_str, seconds=value * multipliers[unit])


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
        amount_col: str = "amount"
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
        
        # Sort by customer and timestamp
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
        
        # Merge all features
        features = transactions[[customer_id_col, timestamp_col]].copy()
        
        for feature_df in [velocity_features, amount_features, 
                          volatility_features, pattern_features]:
            features = features.merge(
                feature_df, 
                on=[customer_id_col, timestamp_col], 
                how="left"
            )
        
        return features
    
    def _compute_velocity_features(
        self, 
        df: pd.DataFrame,
        customer_id_col: str,
        timestamp_col: str
    ) -> pd.DataFrame:
        """Compute transaction velocity features."""
        
        result = df[[customer_id_col, timestamp_col]].copy()
        
        for window in self.windows:
            # Count transactions in window
            result[f"txn_count_{window.name}"] = (
                df.groupby(customer_id_col)[timestamp_col]
                .transform(
                    lambda x: x.apply(
                        lambda t: ((x >= t - timedelta(seconds=window.seconds)) & (x <= t)).sum()
                    )
                )
            )
        
        # Compute velocity change (current vs previous period)
        for window in self.windows:
            col_name = f"txn_count_{window.name}"
            if col_name in result.columns:
                # Shift to get previous period
                prev_col = f"prev_{col_name}"
                result[prev_col] = (
                    result.groupby(customer_id_col)[col_name]
                    .shift(1)
                    .fillna(1)  # Avoid division by zero
                )
                
                # Rate of change
                result[f"txn_velocity_change_{window.name}"] = (
                    (result[col_name] - result[prev_col]) / result[prev_col]
                ).replace([np.inf, -np.inf], 0)
                
                # Drop temporary column
                result.drop(columns=[prev_col], inplace=True)
        
        return result
    
    def _compute_amount_features(
        self, 
        df: pd.DataFrame,
        customer_id_col: str,
        timestamp_col: str,
        amount_col: str
    ) -> pd.DataFrame:
        """Compute amount-based features."""
        
        result = df[[customer_id_col, timestamp_col]].copy()
        
        for window in self.windows:
            window_mask = lambda x, t: (
                (x[timestamp_col] >= t - timedelta(seconds=window.seconds)) & 
                (x[timestamp_col] <= t)
            )
            
            # For each row, compute aggregates over the window
            # Using rolling with time-based windows
            df_indexed = df.set_index(timestamp_col)
            
            rolling_window = f"{window.seconds}s"
            
            grouped = df.groupby(customer_id_col)
            
            # Mean amount
            result[f"avg_amount_{window.name}"] = grouped[amount_col].transform(
                lambda x: x.rolling(window=rolling_window, min_periods=1, on=df.loc[x.index, timestamp_col]).mean()
            )
            
            # Max amount
            result[f"max_amount_{window.name}"] = grouped[amount_col].transform(
                lambda x: x.rolling(window=rolling_window, min_periods=1, on=df.loc[x.index, timestamp_col]).max()
            )
            
            # Standard deviation
            result[f"std_amount_{window.name}"] = grouped[amount_col].transform(
                lambda x: x.rolling(window=rolling_window, min_periods=2, on=df.loc[x.index, timestamp_col]).std()
            ).fillna(0)
            
            # Sum (total spending)
            result[f"total_amount_{window.name}"] = grouped[amount_col].transform(
                lambda x: x.rolling(window=rolling_window, min_periods=1, on=df.loc[x.index, timestamp_col]).sum()
            )
        
        return result
    
    def _compute_volatility_features(
        self, 
        df: pd.DataFrame,
        customer_id_col: str,
        timestamp_col: str,
        amount_col: str
    ) -> pd.DataFrame:
        """Compute spending volatility features."""
        
        result = df[[customer_id_col, timestamp_col]].copy()
        
        grouped = df.groupby(customer_id_col)
        
        # Coefficient of variation (std/mean) - measures relative volatility
        for window in self.windows:
            rolling_window = f"{window.seconds}s"
            
            mean_col = f"_mean_{window.name}"
            std_col = f"_std_{window.name}"
            
            result[mean_col] = grouped[amount_col].transform(
                lambda x: x.rolling(window=rolling_window, min_periods=1, on=df.loc[x.index, timestamp_col]).mean()
            )
            
            result[std_col] = grouped[amount_col].transform(
                lambda x: x.rolling(window=rolling_window, min_periods=2, on=df.loc[x.index, timestamp_col]).std()
            ).fillna(0)
            
            # Coefficient of variation
            result[f"spending_cv_{window.name}"] = (
                result[std_col] / result[mean_col].replace(0, 1)
            ).fillna(0)
            
            # Drop temporary columns
            result.drop(columns=[mean_col, std_col], inplace=True)
        
        # Amount deviation from personal average
        personal_avg = grouped[amount_col].transform("mean")
        personal_std = grouped[amount_col].transform("std").fillna(1)
        
        result["amount_zscore"] = (df[amount_col] - personal_avg) / personal_std
        result["amount_zscore"] = result["amount_zscore"].fillna(0)
        
        return result
    
    def _compute_pattern_features(
        self, 
        df: pd.DataFrame,
        customer_id_col: str,
        timestamp_col: str
    ) -> pd.DataFrame:
        """Compute temporal pattern features."""
        
        result = df[[customer_id_col, timestamp_col]].copy()
        
        # Extract time components
        result["hour_of_day"] = df[timestamp_col].dt.hour
        result["day_of_week"] = df[timestamp_col].dt.dayofweek
        result["day_of_month"] = df[timestamp_col].dt.day
        result["is_weekend"] = result["day_of_week"].isin([5, 6]).astype(int)
        result["is_night"] = result["hour_of_day"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
        result["is_business_hours"] = result["hour_of_day"].isin(range(9, 18)).astype(int)
        
        # Rolling ratios
        grouped = result.groupby(customer_id_col)
        
        for window in [w for w in self.windows if w.seconds >= 86400]:  # Only for day+ windows
            rolling_window = f"{window.seconds}s"
            
            # Weekend transaction ratio
            result[f"weekend_ratio_{window.name}"] = grouped["is_weekend"].transform(
                lambda x: x.rolling(window=rolling_window, min_periods=1, on=df.loc[x.index, timestamp_col]).mean()
            )
            
            # Night transaction ratio
            result[f"night_ratio_{window.name}"] = grouped["is_night"].transform(
                lambda x: x.rolling(window=rolling_window, min_periods=1, on=df.loc[x.index, timestamp_col]).mean()
            )
        
        return result
    
    def compute_customer_summary(
        self, 
        transactions: pd.DataFrame,
        customer_id_col: str = "customer_id",
        timestamp_col: str = "timestamp",
        amount_col: str = "amount"
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
        latest_idx = (
            all_features.groupby(customer_id_col)[timestamp_col]
            .idxmax()
        )
        
        summary = all_features.loc[latest_idx].reset_index(drop=True)
        
        # Add account age
        first_txn = transactions.groupby(customer_id_col)[timestamp_col].min()
        summary = summary.merge(
            first_txn.rename("first_transaction").reset_index(),
            on=customer_id_col
        )
        summary["account_age_days"] = (
            summary[timestamp_col] - summary["first_transaction"]
        ).dt.days
        
        summary.drop(columns=["first_transaction"], inplace=True)
        
        return summary
