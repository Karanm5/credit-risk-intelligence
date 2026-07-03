"""
End-to-end training pipeline for Credit Risk Intelligence Platform.
Orchestrates data loading, feature engineering, model training, and evaluation.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import click
import mlflow
import numpy as np
import pandas as pd
import yaml

from src.config.settings import get_settings
from src.data.snowflake_connector import FeatureStore, SnowflakeConnector
from src.features.graph import GraphFeatureEngineer
from src.features.temporal import TemporalFeatureEngineer
from src.models.training import CreditRiskEnsemble, ModelMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

settings = get_settings()


class TrainingPipeline:
    """
    End-to-end training pipeline for credit risk model.

    Steps:
    1. Load training data from feature store
    2. Apply feature engineering
    3. Train ensemble model
    4. Evaluate and log metrics
    5. Save model artifacts
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialise training pipeline.

        Args:
            config_path: Path to model configuration YAML
        """
        self.config = self._load_config(config_path)
        self.feature_store = FeatureStore()
        self.temporal_engineer = TemporalFeatureEngineer()
        self.graph_engineer = GraphFeatureEngineer()
        self.model: Optional[CreditRiskEnsemble] = None

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = (
                Path(__file__).parent.parent.parent / "configs" / "model_config.yaml"
            )

        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def run(self) -> ModelMetrics:
        """
        Execute the full training pipeline.

        Returns:
            ModelMetrics with evaluation results
        """
        logger.info("Starting training pipeline...")
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        with mlflow.start_run(run_name=f"training_pipeline_{run_id}"):
            # Log configuration
            mlflow.log_params(
                {
                    "train_start_date": self.config["data"]["train_start_date"],
                    "train_end_date": self.config["data"]["train_end_date"],
                    "cv_folds": self.config["model"]["cv_folds"],
                }
            )

            # Step 1: Load data
            logger.info("Step 1: Loading training data...")
            X_train, y_train = self._load_training_data()
            logger.info(
                f"Loaded {len(X_train)} samples with {X_train.shape[1]} features"
            )

            # Step 2: Feature engineering (if needed)
            logger.info("Step 2: Applying feature engineering...")
            X_train = self._apply_feature_engineering(X_train)

            # Step 3: Train model
            logger.info("Step 3: Training ensemble model...")
            self.model = CreditRiskEnsemble(
                experiment_name=self.config["mlflow"]["experiment_name"]
            )
            metrics = self.model.train(X_train, y_train)

            # Step 4: Evaluate
            logger.info("Step 4: Evaluating model...")
            self._log_evaluation_metrics(metrics)

            # Step 5: Save model
            logger.info("Step 5: Saving model artifacts...")
            model_path = self._save_model(run_id)

            # Check performance thresholds
            self._check_thresholds(metrics)

            logger.info(f"Training pipeline completed. Model saved to {model_path}")

            return metrics

    def _load_training_data(self) -> tuple:
        """Load training data from feature store."""
        start_date = datetime.strptime(
            self.config["data"]["train_start_date"], "%Y-%m-%d"
        )
        end_date = datetime.strptime(self.config["data"]["train_end_date"], "%Y-%m-%d")

        # Get features to include
        feature_groups = self.config["features"]["include_groups"]

        # Build feature list based on groups
        feature_list = self._get_feature_list(feature_groups)

        # Load from feature store
        df = self.feature_store.get_training_dataset(
            start_date=start_date,
            end_date=end_date,
            feature_list=feature_list,
            label_column=self.config["data"]["label_column"],
        )

        if df.empty:
            raise ValueError("No training data found in feature store")

        # Split features and labels
        label_col = self.config["data"]["label_column"]
        id_cols = [self.config["data"]["customer_id_column"], "event_date"]

        y = df[label_col]
        X = df.drop(columns=[label_col] + id_cols, errors="ignore")

        # Handle missing values
        X = self._handle_missing_values(X)

        return X, y

    def _get_feature_list(self, feature_groups: list) -> list:
        """Map feature groups to specific feature columns."""
        feature_mapping = {
            "temporal": [
                "txn_count_1h",
                "txn_count_6h",
                "txn_count_24h",
                "txn_count_7d",
                "txn_count_30d",
            ],
            "amount": [
                "avg_amount_24h",
                "avg_amount_7d",
                "avg_amount_30d",
                "max_amount_7d",
                "max_amount_30d",
                "std_amount_7d",
                "std_amount_30d",
                "total_amount_30d",
            ],
            "velocity": [
                "txn_velocity_change_7d",
                "txn_velocity_change_30d",
                "amount_velocity_change_7d",
            ],
            "volatility": ["spending_cv_7d", "spending_cv_30d", "amount_zscore"],
            "diversity": [
                "unique_merchants_7d",
                "unique_merchants_30d",
                "unique_categories_30d",
                "merchant_concentration_30d",
            ],
            "pattern": [
                "weekend_ratio_7d",
                "weekend_ratio_30d",
                "night_ratio_7d",
                "night_ratio_30d",
            ],
            "graph": [
                "pagerank_score",
                "degree_centrality",
                "weighted_degree",
                "betweenness_centrality",
                "community_size",
                "community_density",
                "clustering_coefficient",
                "num_merchants",
                "avg_neighbor_degree",
            ],
            "risk_propagation": [
                "merchant_risk_exposure",
                "high_risk_merchant_ratio",
                "max_merchant_risk",
                "weighted_avg_merchant_risk",
                "peer_risk_exposure",
                "high_risk_peer_ratio",
            ],
        }

        features = []
        for group in feature_groups:
            if group in feature_mapping:
                features.extend(feature_mapping[group])

        # Remove excluded features
        excluded = self.config["features"].get("exclude_features", [])
        features = [f for f in features if f not in excluded]

        return features

    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values according to configuration."""
        strategy = self.config["features"].get("missing_strategy", "median")

        if strategy == "median":
            return X.fillna(X.median())
        elif strategy == "mean":
            return X.fillna(X.mean())
        elif strategy == "zero":
            return X.fillna(0)
        elif strategy == "drop":
            return X.dropna()
        else:
            return X.fillna(X.median())

    def _apply_feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply any additional feature engineering."""
        # Features are pre-computed in feature store
        # Add any transformations here if needed

        # Log feature statistics
        mlflow.log_metrics(
            {
                "num_features": X.shape[1],
                "num_samples": X.shape[0],
                "feature_missing_rate": X.isnull().mean().mean(),
            }
        )

        return X

    def _log_evaluation_metrics(self, metrics: ModelMetrics) -> None:
        """Log evaluation metrics to MLflow."""
        mlflow.log_metrics(
            {
                "auc_roc": metrics.auc_roc,
                "auc_pr": metrics.auc_pr,
                "ks_statistic": metrics.ks_statistic,
                "gini_coefficient": metrics.gini_coefficient,
                "precision_at_10": metrics.precision_at_10,
                "recall_at_10": metrics.recall_at_10,
            }
        )

        logger.info(f"Model Performance:")
        logger.info(f"  AUC-ROC: {metrics.auc_roc:.4f}")
        logger.info(f"  AUC-PR: {metrics.auc_pr:.4f}")
        logger.info(f"  KS Statistic: {metrics.ks_statistic:.4f}")
        logger.info(f"  Gini Coefficient: {metrics.gini_coefficient:.4f}")

    def _save_model(self, run_id: str) -> str:
        """Save model artifacts to disk and MLflow."""
        model_dir = Path("models") / run_id
        model_dir.mkdir(parents=True, exist_ok=True)

        self.model.save(str(model_dir))

        # Log to MLflow
        mlflow.log_artifacts(str(model_dir), "model")

        return str(model_dir)

    def _check_thresholds(self, metrics: ModelMetrics) -> None:
        """Check if model meets performance thresholds."""
        thresholds = self.config["evaluation"]["thresholds"]

        passed = True

        if metrics.auc_roc < thresholds.get("auc_roc", 0):
            logger.warning(
                f"AUC-ROC {metrics.auc_roc:.4f} below threshold {thresholds['auc_roc']}"
            )
            passed = False

        if metrics.ks_statistic < thresholds.get("ks_statistic", 0):
            logger.warning(
                f"KS Statistic {metrics.ks_statistic:.4f} below threshold {thresholds['ks_statistic']}"
            )
            passed = False

        if metrics.gini_coefficient < thresholds.get("gini_coefficient", 0):
            logger.warning(
                f"Gini {metrics.gini_coefficient:.4f} below threshold {thresholds['gini_coefficient']}"
            )
            passed = False

        if passed:
            logger.info("All performance thresholds passed!")
        else:
            logger.warning("Some performance thresholds not met!")


@click.command()
@click.option("--config", "-c", default=None, help="Path to model configuration YAML")
@click.option("--dry-run", is_flag=True, help="Run without saving model")
def main(config: Optional[str], dry_run: bool):
    """Run the credit risk model training pipeline."""
    try:
        pipeline = TrainingPipeline(config_path=config)
        metrics = pipeline.run()

        click.echo(f"\nTraining completed successfully!")
        click.echo(f"AUC-ROC: {metrics.auc_roc:.4f}")
        click.echo(f"Gini: {metrics.gini_coefficient:.4f}")

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise click.ClickException(str(e))


if __name__ == "__main__":
    main()
