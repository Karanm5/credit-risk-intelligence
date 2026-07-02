# Credit Risk Intelligence Platform

[![CI/CD Pipeline](https://github.com/Karanm5/credit-risk-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/Karanm5/credit-risk-intelligence/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A production-grade, real-time credit risk assessment platform using alternative data signals, graph-based features, and modern MLOps practices.**

## What Makes This Project Unique

Traditional credit scoring relies on limited historical data (payment history, credit utilisation). This platform takes a **fundamentally different approach**:

1. **Alternative Data Signals**: Transaction velocity patterns, merchant category diversity, temporal spending behaviours
2. **Graph-Based Features**: Models relationships between entities (accounts, merchants, geographies) to detect hidden risk patterns
3. **Real-Time Scoring**: Sub-100ms inference with streaming feature computation
4. **Explainable AI**: Every prediction comes with SHAP-based explanations for regulatory compliance

Note: The CI/CD pipeline issue will be rectify soon. The project will be live soon. Sorry for the inconvenience!!!
---

##  Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           CREDIT RISK INTELLIGENCE PLATFORM                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   External   │    │   Kafka      │    │  Delta Lake  │    │  Snowflake   │  │
│  │   Data APIs  │───▶│   Streams    │───▶│  (Bronze/    │───▶│  (Feature    │  │
│  │              │    │              │    │   Silver)    │    │   Store)     │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                 │                    │          │
│                                                 ▼                    ▼          │
│                                          ┌─────────────────────────────┐        │
│                                          │     Feature Engineering     │        │
│                                          │  ┌─────────┐ ┌───────────┐ │        │
│                                          │  │Temporal │ │  Graph    │ │        │
│                                          │  │Features │ │ Features  │ │        │
│                                          │  └─────────┘ └───────────┘ │        │
│                                          └─────────────────────────────┘        │
│                                                        │                        │
│                                                        ▼                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────────┐      │
│  │   MLflow     │◀───│   Model      │◀───│      Training Pipeline       │      │
│  │   Registry   │    │   Training   │    │  (XGBoost + LightGBM + NN)   │      │
│  └──────────────┘    └──────────────┘    └──────────────────────────────┘      │
│         │                                                                       │
│         ▼                                                                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                      │
│  │   FastAPI    │───▶│    SHAP      │───▶│  Monitoring  │                      │
│  │   Serving    │    │  Explainer   │    │  & Drift     │                      │
│  └──────────────┘    └──────────────┘    └──────────────┘                      │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Data Ingestion** | Apache Kafka, REST APIs | Real-time and batch data ingestion |
| **Data Lake** | Delta Lake (Databricks) | Bronze/Silver/Gold medallion architecture |
| **Data Warehouse** | Snowflake | Feature store and analytical queries |
| **Feature Engineering** | PySpark, NetworkX | Temporal and graph-based features |
| **ML Training** | XGBoost, LightGBM, PyTorch | Ensemble credit risk models |
| **Experiment Tracking** | MLflow | Model versioning and registry |
| **Model Serving** | FastAPI, Docker | Real-time inference API |
| **Explainability** | SHAP | Regulatory-compliant explanations |
| **Monitoring** | Evidently AI, Prometheus | Drift detection and metrics |
| **Orchestration** | Apache Airflow | Pipeline scheduling |
| **CI/CD** | GitHub Actions | Automated testing and deployment |
| **Infrastructure** | Docker, Kubernetes | Containerisation and orchestration |

---

## Data Architecture

### Medallion Architecture (Delta Lake)

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  BRONZE (Raw)           SILVER (Cleaned)        GOLD (Features) │
│  ─────────────          ───────────────         ─────────────── │
│                                                                  │
│  • Raw transactions     • Deduplicated          • Aggregated    │
│  • API responses        • Type-casted           • Feature store │
│  • Log data             • Validated             • ML-ready      │
│  • Schema-on-read       • Schema-enforced       • Optimised     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Feature Categories

| Category | Features | Description |
|----------|----------|-------------|
| **Temporal** | `txn_velocity_1h`, `txn_velocity_24h`, `spending_volatility_7d` | Time-windowed transaction patterns |
| **Behavioural** | `merchant_diversity_score`, `category_concentration`, `weekend_ratio` | Spending behaviour signals |
| **Graph** | `account_centrality`, `merchant_risk_propagation`, `community_risk_score` | Network-based risk indicators |
| **Economic** | `regional_unemployment_delta`, `sector_exposure_score` | Macro-economic overlays |

---

## Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Snowflake account (free trial works)
- Databricks account (community edition works)

### Installation

```bash
# Clone the repository
git clone https://github.com/Karanm5/credit-risk-intelligence.git
cd credit-risk-intelligence

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your Snowflake and Databricks credentials

# Run tests
make test

# Start local development environment
make dev
```

### Docker Deployment

```bash
# Build and run all services
docker-compose up --build

# Access the API
curl http://localhost:8000/health

# Score a customer
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "CUST_001"}'
```

---

## Project Structure

```
credit-risk-intelligence/
│
├── .github/
│   └── workflows/
│       ├── ci.yml                 # Continuous Integration
│       └── cd.yml                 # Continuous Deployment
│
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py            # Configuration management
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingestion.py           # Data ingestion from APIs
│   │   ├── delta_lake.py          # Delta Lake operations
│   │   └── snowflake_connector.py # Snowflake integration
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── temporal.py            # Time-based features
│   │   ├── behavioural.py         # Spending behaviour features
│   │   ├── graph.py               # Graph-based features
│   │   └── store.py               # Feature store operations
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── training.py            # Model training pipeline
│   │   ├── ensemble.py            # Ensemble model logic
│   │   ├── evaluation.py          # Model evaluation metrics
│   │   └── explainability.py      # SHAP explanations
│   │
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── api.py                 # FastAPI application
│   │   ├── schemas.py             # Pydantic schemas
│   │   └── monitoring.py          # Model monitoring
│   │
│   └── pipelines/
│       ├── __init__.py
│       ├── training_pipeline.py   # End-to-end training
│       └── inference_pipeline.py  # Real-time inference
│
├── tests/
│   ├── __init__.py
│   ├── test_features.py           # Feature engineering tests
│   ├── test_models.py             # Model tests
│   ├── test_api.py                # API endpoint tests
│   └── conftest.py                # Pytest fixtures
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_development.ipynb
│
├── configs/
│   ├── model_config.yaml          # Model hyperparameters
│   ├── feature_config.yaml        # Feature definitions
│   └── pipeline_config.yaml       # Pipeline settings
│
├── docker/
│   ├── Dockerfile                 # Application container
│   └── docker-compose.yml         # Multi-service setup
│
├── scripts/
│   ├── setup_snowflake.sql        # Snowflake schema setup
│   └── setup_delta_tables.py      # Delta Lake initialisation
│
├── requirements.txt
├── setup.py
├── Makefile
├── .env.example
└── README.md
```

---

## Feature Engineering Deep Dive

### Temporal Features

```python
# Transaction velocity captures spending intensity
def compute_transaction_velocity(df: DataFrame, windows: List[str]) -> DataFrame:
    """
    Compute transaction counts over multiple time windows.
    High velocity in short windows may indicate fraud or financial stress.
    """
    for window in windows:
        df = df.withColumn(
            f"txn_velocity_{window}",
            F.count("transaction_id").over(
                Window.partitionBy("customer_id")
                .orderBy(F.col("timestamp").cast("long"))
                .rangeBetween(-parse_window(window), 0)
            )
        )
    return df
```

### Graph Features

```python
# Build transaction graph and compute risk propagation
def compute_graph_features(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Model customer-merchant relationships as a graph.
    Risk propagates through the network - customers connected to
    high-risk merchants inherit elevated risk scores.
    """
    G = nx.from_pandas_edgelist(
        transactions_df,
        source='customer_id',
        target='merchant_id',
        edge_attr='amount',
        create_using=nx.DiGraph()
    )
    
    # PageRank identifies influential nodes
    pagerank = nx.pagerank(G, weight='amount')
    
    # Community detection for risk clustering
    communities = nx.community.louvain_communities(G.to_undirected())
    
    return compute_propagated_risk(G, pagerank, communities)
```

---

## Model Architecture

### Ensemble Strategy

The platform uses a **stacked ensemble** combining three base models:

```
┌─────────────────────────────────────────────────────────────┐
│                      Input Features                          │
│         (Temporal + Behavioural + Graph + Economic)          │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐    ┌──────────┐    ┌──────────┐
        │ XGBoost  │    │ LightGBM │    │  Neural  │
        │          │    │          │    │  Network │
        └──────────┘    └──────────┘    └──────────┘
              │               │               │
              └───────────────┼───────────────┘
                              ▼
                    ┌──────────────────┐
                    │   Meta-Learner   │
                    │ (Logistic Reg.)  │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Risk Score +    │
                    │  SHAP Explanation│
                    └──────────────────┘
```



## CI/CD Pipeline( not active yet- Have error)

### Continuous Integration

```yaml
# .github/workflows/ci.yml
on: [push, pull_request]

jobs:
  test:
    - Lint with flake8 and black
    - Type check with mypy
    - Unit tests with pytest
    - Integration tests
    - Coverage report (>85% required)
  
  security:
    - Dependency vulnerability scan
    - SAST with bandit
    - Secrets detection
```

### Continuous Deployment

```yaml
# .github/workflows/cd.yml
on:
  push:
    branches: [main]

jobs:
  deploy:
    - Build Docker image
    - Push to container registry
    - Deploy to staging
    - Run smoke tests
    - Deploy to production (manual approval)
    - Update MLflow model registry
```

---

## Monitoring & Observability

### Model Drift Detection

```python
# Evidently AI integration for drift monitoring
from evidently.report import Report
from evidently.metrics import DataDriftPreset, TargetDriftPreset

def monitor_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame):
    """
    Detect feature and prediction drift.
    Triggers retraining pipeline if drift exceeds threshold.
    """
    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset()
    ])
    report.run(reference_data=reference_data, current_data=current_data)
    
    if report.as_dict()['metrics'][0]['result']['drift_detected']:
        trigger_retraining_pipeline()
```

### Metrics Dashboard

- **Prediction latency** (p50, p95, p99)
- **Feature freshness** (lag from source)
- **Model accuracy** (rolling AUC)
- **Drift scores** (per feature)
- **API throughput** (requests/sec)

---

## Security & Compliance 

- **Audit logging**: All predictions logged with explanations
- **Model cards**: Documented bias analysis and limitations
- **GDPR compliance**: Right to explanation via SHAP

---

## Testing Strategy

```bash
# Run all tests
make test

# Run specific test categories
pytest tests/test_features.py -v      # Feature engineering
pytest tests/test_models.py -v        # Model training
pytest tests/test_api.py -v           # API endpoints

# Run with coverage
pytest --cov=src --cov-report=html
```

### Test Coverage Requirements

| Module | Minimum Coverage |
|--------|-----------------|
| `src/features/` | 90% |
| `src/models/` | 85% |
| `src/serving/` | 90% |
| `src/data/` | 80% |

* Note: Reason for the use of ensemble model will be uploaded soon.

---

## Further Reading

- [Delta Lake Documentation](https://docs.delta.io/)
- [Snowflake Feature Store Pattern](https://docs.snowflake.com/)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [SHAP for Model Explainability](https://shap.readthedocs.io/)
- [Evidently AI for Monitoring](https://docs.evidentlyai.com/)

---

## Author

**Karan**
- MSc Data Analytics (Distinction), Aston University
- [LinkedIn](https://linkedin.com/in/karan-th)
- [GitHub](https://github.com/Karanm5)


---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
