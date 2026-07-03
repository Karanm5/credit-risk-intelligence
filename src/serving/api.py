"""
FastAPI application for real-time credit risk scoring.
Provides REST API endpoints for predictions with SHAP explanations.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import shap
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from src.config.settings import get_settings
from src.data.snowflake_connector import FeatureStore
from src.models.training import CreditRiskEnsemble

logger = logging.getLogger(__name__)
settings = get_settings()


# Prometheus metrics
PREDICTION_COUNTER = Counter(
    "credit_risk_predictions_total", "Total number of predictions made", ["risk_level"]
)

PREDICTION_LATENCY = Histogram(
    "credit_risk_prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0],
)

FEATURE_FETCH_LATENCY = Histogram(
    "credit_risk_feature_fetch_latency_seconds", "Feature retrieval latency in seconds"
)


# Request/Response schemas
class PredictionRequest(BaseModel):
    """Request schema for single prediction."""

    customer_id: str = Field(..., description="Customer identifier")
    include_explanation: bool = Field(
        default=True, description="Whether to include SHAP explanation"
    )


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""

    customer_ids: List[str] = Field(..., description="List of customer IDs")
    include_explanations: bool = Field(default=False)


class FeatureContribution(BaseModel):
    """SHAP feature contribution."""

    feature: str
    value: float
    contribution: float
    direction: str  # "increases_risk" or "decreases_risk"


class PredictionResponse(BaseModel):
    """Response schema for prediction."""

    customer_id: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: str  # "low", "medium", "high"
    confidence: float
    timestamp: datetime
    explanations: Optional[List[FeatureContribution]] = None
    model_version: str


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""

    predictions: List[PredictionResponse]
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    feature_store_connected: bool
    timestamp: datetime


# Global model and feature store instances
model: Optional[CreditRiskEnsemble] = None
feature_store: Optional[FeatureStore] = None
shap_explainer: Optional[shap.TreeExplainer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - handles startup and shutdown."""
    global model, feature_store, shap_explainer

    logger.info("Starting Credit Risk API...")

    # Load model. The service starts in a degraded state if no artefact
    # exists yet (e.g. fresh deployment, CI smoke test); /health reports it.
    try:
        model = CreditRiskEnsemble()
        model.load(settings.model.model_version)
        logger.info("Model loaded successfully")

        # Initialize SHAP explainer with XGBoost model
        shap_explainer = shap.TreeExplainer(model.xgb_model)
        logger.info("SHAP explainer initialized")
    except Exception as e:
        logger.error(f"Failed to load model, starting degraded: {e}")
        model = None
        shap_explainer = None

    # Connect to feature store
    try:
        feature_store = FeatureStore()
        if not feature_store.is_connected():
            logger.warning("Feature store not configured; running degraded")
    except Exception as e:
        logger.error(f"Feature store unavailable, starting degraded: {e}")
        feature_store = None

    yield

    # Cleanup
    logger.info("Shutting down Credit Risk API...")


# Create FastAPI app
app = FastAPI(
    title="Credit Risk Intelligence API",
    description="Real-time credit risk scoring with explainable AI",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_risk_level(score: float) -> str:
    """Convert risk score to risk level category."""
    if score < 0.3:
        return "low"
    elif score < 0.7:
        return "medium"
    else:
        return "high"


def compute_explanation(
    features: pd.DataFrame, feature_names: List[str], top_k: int = 5
) -> List[FeatureContribution]:
    """Compute SHAP explanations for a prediction."""

    if shap_explainer is None:
        return []

    # Get SHAP values
    shap_values = shap_explainer.shap_values(features)

    # For binary classification, shap_values might be a list
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Get positive class

    # Get top contributing features
    contributions = []
    feature_values = features.values[0]
    shap_vals = shap_values[0]

    # Sort by absolute contribution
    sorted_indices = np.argsort(np.abs(shap_vals))[::-1][:top_k]

    for idx in sorted_indices:
        contribution = float(shap_vals[idx])
        contributions.append(
            FeatureContribution(
                feature=feature_names[idx],
                value=float(feature_values[idx]),
                contribution=abs(contribution),
                direction="increases_risk" if contribution > 0 else "decreases_risk",
            )
        )

    return contributions


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        feature_store_connected=feature_store is not None,
        timestamp=datetime.now(timezone.utc),
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Generate credit risk prediction for a single customer.

    Returns risk score (0-1), risk level, and optional SHAP explanations.
    """
    import time

    start_time = time.time()

    if model is None or feature_store is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        # Fetch features
        with FEATURE_FETCH_LATENCY.time():
            features_dict = feature_store.get_features_for_customer(request.customer_id)

        if not features_dict:
            raise HTTPException(
                status_code=404, detail=f"Customer {request.customer_id} not found"
            )

        # Prepare features DataFrame
        features_df = pd.DataFrame([features_dict])
        features_df = features_df[model.feature_names]

        # Generate prediction
        with PREDICTION_LATENCY.time():
            risk_score = float(model.predict(features_df)[0])

        risk_level = get_risk_level(risk_score)

        # Update metrics
        PREDICTION_COUNTER.labels(risk_level=risk_level).inc()

        # Compute explanations if requested
        explanations = None
        if request.include_explanation:
            # Tree-based models require no feature scaling; SHAP explains
            # predictions directly on the raw feature values.
            explanations = compute_explanation(features_df, model.feature_names)

        processing_time = (time.time() - start_time) * 1000
        logger.info(
            f"Prediction for {request.customer_id}: "
            f"score={risk_score:.4f}, level={risk_level}, "
            f"time={processing_time:.2f}ms"
        )

        return PredictionResponse(
            customer_id=request.customer_id,
            risk_score=risk_score,
            risk_level=risk_level,
            confidence=1 - abs(0.5 - risk_score) * 2,  # Higher near 0 or 1
            timestamp=datetime.now(timezone.utc),
            explanations=explanations,
            model_version=settings.model.model_version,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Generate credit risk predictions for multiple customers.

    More efficient than multiple single predictions for bulk scoring.
    """
    import time

    start_time = time.time()

    if model is None or feature_store is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        # Fetch features for all customers
        features_df = feature_store.get_features_batch(request.customer_ids)

        if features_df.empty:
            raise HTTPException(status_code=404, detail="No customers found")

        # Ensure correct column order
        feature_cols = [c for c in model.feature_names if c in features_df.columns]
        features_for_prediction = features_df[feature_cols]

        # Generate predictions
        risk_scores = model.predict(features_for_prediction)

        # Build responses
        predictions = []
        for idx, row in features_df.iterrows():
            customer_id = row["customer_id"]
            risk_score = float(risk_scores[idx])
            risk_level = get_risk_level(risk_score)

            PREDICTION_COUNTER.labels(risk_level=risk_level).inc()

            predictions.append(
                PredictionResponse(
                    customer_id=customer_id,
                    risk_score=risk_score,
                    risk_level=risk_level,
                    confidence=1 - abs(0.5 - risk_score) * 2,
                    timestamp=datetime.now(timezone.utc),
                    explanations=None,  # Skip explanations for batch
                    model_version=settings.model.model_version,
                )
            )

        processing_time = (time.time() - start_time) * 1000

        logger.info(
            f"Batch prediction for {len(request.customer_ids)} customers: "
            f"time={processing_time:.2f}ms"
        )

        return BatchPredictionResponse(
            predictions=predictions, processing_time_ms=processing_time
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def model_info():
    """Get information about the deployed model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_version": settings.model.model_version,
        "feature_count": len(model.feature_names),
        "feature_names": model.feature_names,
        "base_models": ["XGBoost", "LightGBM", "Neural Network"],
        "meta_learner": "Calibrated Logistic Regression",
    }


@app.get("/model/feature-importance")
async def feature_importance():
    """Get feature importance from the model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    importance = model._get_feature_importance()

    # Sort by importance
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    return {
        "features": [
            {"name": name, "importance": imp} for name, imp in sorted_importance
        ]
    }


# Run with: uvicorn src.serving.api:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.serving.api:app",
        host=settings.api.host,
        port=settings.api.port,
        workers=settings.api.workers,
        reload=settings.api.debug,
    )
