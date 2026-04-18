"""
Configuration management for Credit Risk Intelligence Platform.
Uses Pydantic for validation and environment variable loading.
"""

from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr
from typing import Optional
from functools import lru_cache


class SnowflakeSettings(BaseSettings):
    """Snowflake connection configuration."""
    
    account: str = Field(..., env="SNOWFLAKE_ACCOUNT")
    user: str = Field(..., env="SNOWFLAKE_USER")
    password: SecretStr = Field(..., env="SNOWFLAKE_PASSWORD")
    warehouse: str = Field(default="COMPUTE_WH", env="SNOWFLAKE_WAREHOUSE")
    database: str = Field(default="CREDIT_RISK", env="SNOWFLAKE_DATABASE")
    schema_name: str = Field(default="FEATURES", env="SNOWFLAKE_SCHEMA")
    role: str = Field(default="ANALYST", env="SNOWFLAKE_ROLE")
    
    class Config:
        env_prefix = "SNOWFLAKE_"


class DatabricksSettings(BaseSettings):
    """Databricks connection configuration."""
    
    host: str = Field(..., env="DATABRICKS_HOST")
    token: SecretStr = Field(..., env="DATABRICKS_TOKEN")
    cluster_id: str = Field(..., env="DATABRICKS_CLUSTER_ID")
    delta_path: str = Field(
        default="dbfs:/mnt/credit-risk/delta",
        env="DATABRICKS_DELTA_PATH"
    )
    
    class Config:
        env_prefix = "DATABRICKS_"


class MLflowSettings(BaseSettings):
    """MLflow tracking and registry configuration."""
    
    tracking_uri: str = Field(
        default="http://localhost:5000",
        env="MLFLOW_TRACKING_URI"
    )
    experiment_name: str = Field(
        default="credit-risk-model",
        env="MLFLOW_EXPERIMENT_NAME"
    )
    registry_uri: str = Field(
        default="http://localhost:5000",
        env="MLFLOW_REGISTRY_URI"
    )
    
    class Config:
        env_prefix = "MLFLOW_"


class KafkaSettings(BaseSettings):
    """Kafka streaming configuration."""
    
    bootstrap_servers: str = Field(
        default="localhost:9092",
        env="KAFKA_BOOTSTRAP_SERVERS"
    )
    transactions_topic: str = Field(
        default="transactions",
        env="KAFKA_TRANSACTIONS_TOPIC"
    )
    consumer_group: str = Field(
        default="credit-risk-consumer",
        env="KAFKA_CONSUMER_GROUP"
    )
    
    class Config:
        env_prefix = "KAFKA_"


class ModelSettings(BaseSettings):
    """Model training and serving configuration."""
    
    # Training parameters
    test_size: float = Field(default=0.2)
    random_state: int = Field(default=42)
    cv_folds: int = Field(default=5)
    
    # XGBoost parameters
    xgb_n_estimators: int = Field(default=500)
    xgb_max_depth: int = Field(default=6)
    xgb_learning_rate: float = Field(default=0.05)
    xgb_subsample: float = Field(default=0.8)
    
    # LightGBM parameters
    lgb_n_estimators: int = Field(default=500)
    lgb_max_depth: int = Field(default=8)
    lgb_learning_rate: float = Field(default=0.05)
    lgb_num_leaves: int = Field(default=31)
    
    # Neural network parameters
    nn_hidden_layers: list = Field(default=[128, 64, 32])
    nn_dropout: float = Field(default=0.3)
    nn_epochs: int = Field(default=100)
    nn_batch_size: int = Field(default=256)
    
    # Serving parameters
    model_version: str = Field(default="latest")
    prediction_threshold: float = Field(default=0.5)
    
    class Config:
        env_prefix = "MODEL_"


class FeatureSettings(BaseSettings):
    """Feature engineering configuration."""
    
    # Temporal windows
    velocity_windows: list = Field(default=["1h", "6h", "24h", "7d", "30d"])
    
    # Graph parameters
    pagerank_alpha: float = Field(default=0.85)
    community_resolution: float = Field(default=1.0)
    
    # Feature store
    feature_freshness_hours: int = Field(default=24)
    cache_ttl_seconds: int = Field(default=3600)
    
    class Config:
        env_prefix = "FEATURE_"


class APISettings(BaseSettings):
    """API server configuration."""
    
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=4)
    debug: bool = Field(default=False)
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100)
    rate_limit_window: int = Field(default=60)
    
    # Monitoring
    enable_metrics: bool = Field(default=True)
    metrics_port: int = Field(default=9090)
    
    class Config:
        env_prefix = "API_"


class Settings(BaseSettings):
    """Main application settings aggregating all configurations."""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Sub-configurations
    snowflake: SnowflakeSettings = SnowflakeSettings()
    databricks: DatabricksSettings = DatabricksSettings()
    mlflow: MLflowSettings = MLflowSettings()
    kafka: KafkaSettings = KafkaSettings()
    model: ModelSettings = ModelSettings()
    features: FeatureSettings = FeatureSettings()
    api: APISettings = APISettings()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache for singleton pattern.
    """
    return Settings()
