"""
Configuration management for Credit Risk Intelligence Platform.
Uses pydantic-settings (v2) for validation and environment variable loading.

Connection credentials default to empty strings so the application can be
imported and started without a full production environment; components that
require them (Snowflake, Databricks) validate at connection time instead.
"""

from functools import lru_cache
from typing import List

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class SnowflakeSettings(BaseSettings):
    """Snowflake connection configuration (env prefix: SNOWFLAKE_)."""

    model_config = SettingsConfigDict(env_prefix="SNOWFLAKE_", extra="ignore")

    account: str = ""
    user: str = ""
    password: SecretStr = SecretStr("")
    warehouse: str = "COMPUTE_WH"
    database: str = "CREDIT_RISK"
    schema_name: str = Field(default="FEATURES", validation_alias="SNOWFLAKE_SCHEMA")
    role: str = "ANALYST"

    @property
    def is_configured(self) -> bool:
        """Whether the minimum credentials are present."""
        return bool(self.account and self.user and self.password.get_secret_value())


class DatabricksSettings(BaseSettings):
    """Databricks connection configuration (env prefix: DATABRICKS_)."""

    model_config = SettingsConfigDict(env_prefix="DATABRICKS_", extra="ignore")

    host: str = ""
    token: SecretStr = SecretStr("")
    cluster_id: str = ""
    delta_path: str = "dbfs:/mnt/credit-risk/delta"

    @property
    def is_configured(self) -> bool:
        """Whether the minimum credentials are present."""
        return bool(self.host and self.token.get_secret_value())


class MLflowSettings(BaseSettings):
    """MLflow tracking and registry configuration (env prefix: MLFLOW_)."""

    model_config = SettingsConfigDict(env_prefix="MLFLOW_", extra="ignore")

    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "credit-risk-model"
    registry_uri: str = "http://localhost:5000"


class KafkaSettings(BaseSettings):
    """Kafka streaming configuration (env prefix: KAFKA_)."""

    model_config = SettingsConfigDict(env_prefix="KAFKA_", extra="ignore")

    bootstrap_servers: str = "localhost:9092"
    transactions_topic: str = "transactions"
    consumer_group: str = "credit-risk-consumer"


class ModelSettings(BaseSettings):
    """Model training and serving configuration (env prefix: MODEL_)."""

    model_config = SettingsConfigDict(env_prefix="MODEL_", extra="ignore")

    # Training parameters
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5

    # XGBoost parameters
    xgb_n_estimators: int = 500
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8

    # LightGBM parameters
    lgb_n_estimators: int = 500
    lgb_max_depth: int = 8
    lgb_learning_rate: float = 0.05
    lgb_num_leaves: int = 31

    # Neural network parameters
    nn_hidden_layers: List[int] = Field(default_factory=lambda: [128, 64, 32])
    nn_dropout: float = 0.3
    nn_epochs: int = 100
    nn_batch_size: int = 256

    # Serving parameters
    model_version: str = "latest"
    prediction_threshold: float = 0.5


class FeatureSettings(BaseSettings):
    """Feature engineering configuration (env prefix: FEATURE_)."""

    model_config = SettingsConfigDict(env_prefix="FEATURE_", extra="ignore")

    # Temporal windows
    velocity_windows: List[str] = Field(
        default_factory=lambda: ["1h", "6h", "24h", "7d", "30d"]
    )

    # Graph parameters
    pagerank_alpha: float = 0.85
    community_resolution: float = 1.0

    # Feature store
    feature_freshness_hours: int = 24
    cache_ttl_seconds: int = 3600


class APISettings(BaseSettings):
    """API server configuration (env prefix: API_)."""

    model_config = SettingsConfigDict(env_prefix="API_", extra="ignore")

    # Binding all interfaces is required inside the container; external
    # exposure is controlled by Docker/K8s networking, not the app.
    host: str = "0.0.0.0"  # nosec B104
    port: int = 8000
    workers: int = 4
    debug: bool = False

    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60

    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090


class Settings(BaseSettings):
    """Main application settings aggregating all configurations."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Environment
    environment: str = Field(default="development", validation_alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    # Sub-configurations (built lazily so import never requires credentials)
    snowflake: SnowflakeSettings = Field(default_factory=SnowflakeSettings)
    databricks: DatabricksSettings = Field(default_factory=DatabricksSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)
    kafka: KafkaSettings = Field(default_factory=KafkaSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    features: FeatureSettings = Field(default_factory=FeatureSettings)
    api: APISettings = Field(default_factory=APISettings)


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache for singleton pattern.
    """
    return Settings()
