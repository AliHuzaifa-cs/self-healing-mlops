# Self-Healing MLOps Prototype

Local, Docker-based self-healing MLOps system for predicting LUCK (Lucky Cement, PSX)
next-day price direction. Demonstrates the full loop: monitoring -> detection ->
AI-driven diagnosis -> automated remediation -> validation -> rollback.

## Setup

1. `python -m venv venv` and activate it
2. `pip install -r requirements.txt`
3. `python src/training/fetch_data.py`
4. `python src/training/feature_engineering.py`
5. `python src/training/train_mlflow.py`
6. `docker compose up -d`
7. Install Ollama and pull `llama3.2:3b`

## Architecture

See project documentation for the full 13-stage build.

## Key components
- FastAPI model-serving API (Docker)
- Prometheus + Grafana + cAdvisor monitoring
- Evidently for data drift/quality
- MLflow experiment tracking + model registry
- SQLite-backed incident history
- Ollama-powered AI Ops Agent for root-cause diagnosis and remediation
