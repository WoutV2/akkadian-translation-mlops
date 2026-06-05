Generate guide
Generate todo
Decide what to do first
Local MVP:
    Create venv for packages
    add data
    add preprocessing
    create basic api
    create basic frontend
    install requirements
    train the model using simple settings
    use the trained model in the api
Recent actions:
    - Tested `training/ingest_feedback.py` end-to-end (appended feedback to CSV, handled)
    - Tested FastAPI endpoints `/feedback` and `/feedback/ingest` using TestClient
    - Added `docker-compose.yml` for local API + frontend integration
    - Created Kubernetes manifests under `infrastructure/k8s/` (api, frontend, postgres, ingress)
    - Fixed missing translation logic in FastAPI application (`inference/api/app.py`)
    - Added custom Nginx config and updated frontend Dockerfile to proxy requests in docker-compose
    - Simplified Kubernetes ingress routing paths to avoid complex rewrites
    - Created `training/conda.yml` and updated Azure ML job specification path

Next steps:
    - Test local integration using `docker-compose up --build`
    - Run retraining on Azure ML using `az ml job create --file training/aml_retrain_job.yaml`
    - Deploy manifests to AKS or local k3d using `kubectl apply -f infrastructure/k8s/`