"""
Model training for the Credit Risk Intelligence Platform.

Implements the stacked ensemble described in the README:
XGBoost + LightGBM base learners with a logistic regression meta-learner.

Heavy dependencies (xgboost, lightgbm) are imported lazily inside methods so
that importing this module does not require them to be installed. This keeps
lightweight environments (CI unit tests, tooling) fast.
"""

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path("models")


@dataclass
class ModelMetrics:
    """Evaluation metrics for a trained credit risk model."""

    auc_roc: float
    ks_statistic: float
    gini_coefficient: float
    precision_at_10: float

    def to_dict(self) -> Dict[str, float]:
        """Return metrics as a plain dictionary (for MLflow logging)."""
        return asdict(self)


class CreditRiskEnsemble:
    """
    Stacked ensemble for credit default prediction.

    Base learners: XGBoost and LightGBM gradient-boosted trees.
    Meta-learner: logistic regression over base learner out-of-fold
    probabilities.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialise the ensemble.

        Args:
            params: Optional hyperparameter overrides with keys
                'xgb', 'lgb', and 'meta'.
        """
        self.params = params or {}
        self.xgb_model = None
        self.lgb_model = None
        self.meta_model = None
        self.feature_names: Optional[list] = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "CreditRiskEnsemble":
        """
        Fit base learners and meta-learner.

        Args:
            X: Feature matrix.
            y: Binary default labels.

        Returns:
            The fitted ensemble (self).
        """
        import lightgbm as lgb
        import xgboost as xgb
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict

        self.feature_names = list(X.columns)

        xgb_params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "eval_metric": "auc",
            **self.params.get("xgb", {}),
        }
        lgb_params = {
            "n_estimators": 500,
            "max_depth": 8,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "verbose": -1,
            **self.params.get("lgb", {}),
        }

        self.xgb_model = xgb.XGBClassifier(**xgb_params)
        self.lgb_model = lgb.LGBMClassifier(**lgb_params)

        # Out-of-fold predictions to train the meta-learner without leakage
        logger.info("Generating out-of-fold predictions for meta-learner")
        xgb_oof = cross_val_predict(self.xgb_model, X, y, cv=5, method="predict_proba")[
            :, 1
        ]
        lgb_oof = cross_val_predict(self.lgb_model, X, y, cv=5, method="predict_proba")[
            :, 1
        ]

        meta_features = np.column_stack([xgb_oof, lgb_oof])
        self.meta_model = LogisticRegression(**self.params.get("meta", {}))
        self.meta_model.fit(meta_features, y)

        # Refit base learners on the full data for serving
        logger.info("Refitting base learners on full training data")
        self.xgb_model.fit(X, y)
        self.lgb_model.fit(X, y)

        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict default probabilities.

        Args:
            X: Feature matrix with the same columns used in fit().

        Returns:
            Array of default probabilities in [0, 1].
        """
        self._check_fitted()
        xgb_probs = self.xgb_model.predict_proba(X)[:, 1]
        lgb_probs = self.lgb_model.predict_proba(X)[:, 1]
        meta_features = np.column_stack([xgb_probs, lgb_probs])
        return self.meta_model.predict_proba(meta_features)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Alias for predict_proba (risk scores are probabilities)."""
        return self.predict_proba(X)

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> ModelMetrics:
        """Compute evaluation metrics on a held-out set."""
        from sklearn.metrics import roc_auc_score

        probs = self.predict_proba(X)
        auc = float(roc_auc_score(y, probs))

        # KS statistic: max separation between cumulative distributions
        order = np.argsort(probs)
        y_sorted = np.asarray(y)[order]
        cum_pos = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
        cum_neg = np.cumsum(1 - y_sorted) / max((1 - y_sorted).sum(), 1)
        ks = float(np.max(np.abs(cum_pos - cum_neg)))

        # Precision in the riskiest decile
        threshold = np.quantile(probs, 0.9)
        top_decile = probs >= threshold
        precision_at_10 = (
            float(np.asarray(y)[top_decile].mean()) if top_decile.any() else 0.0
        )

        return ModelMetrics(
            auc_roc=auc,
            ks_statistic=ks,
            gini_coefficient=2 * auc - 1,
            precision_at_10=precision_at_10,
        )

    def save(
        self, version: str = "latest", model_dir: Path = DEFAULT_MODEL_DIR
    ) -> Path:
        """Persist the ensemble to disk with joblib."""
        self._check_fitted()
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / f"credit_risk_ensemble_{version}.joblib"
        joblib.dump(
            {
                "xgb_model": self.xgb_model,
                "lgb_model": self.lgb_model,
                "meta_model": self.meta_model,
                "feature_names": self.feature_names,
            },
            path,
        )
        logger.info("Model saved to %s", path)
        return path

    def load(
        self, version: str = "latest", model_dir: Path = DEFAULT_MODEL_DIR
    ) -> None:
        """
        Load a persisted ensemble from disk.

        Raises:
            FileNotFoundError: If no artefact exists for the given version.
        """
        path = model_dir / f"credit_risk_ensemble_{version}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"No model artefact found at {path}")
        artefact = joblib.load(path)
        self.xgb_model = artefact["xgb_model"]
        self.lgb_model = artefact["lgb_model"]
        self.meta_model = artefact["meta_model"]
        self.feature_names = artefact["feature_names"]
        self.is_fitted = True
        logger.info("Model loaded from %s", path)

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "Model is not fitted. Call fit() or load() before predicting."
            )
