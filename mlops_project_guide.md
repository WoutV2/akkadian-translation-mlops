# MLOps Project Guide — Akkadian→English Translation

This file documents the required tasks, an implementation plan, and practical code snippets to add a user feedback path that stores corrections in a database and enables automated retraining and redeployment.

1) Project goals (mapping to assignment tasks)
- Task 1: Cloud training (Azure ML) — run training/retraining via pipeline.
- Task 2: Kubernetes deployment — FastAPI backend, simple frontend, persistent DB, NGINX/Ingress.
- Task 3: CI/CD — GitHub Actions to trigger retrain and redeploy when data or code changes.

2) Recommended user-feedback workflow (high level)
- User queries model via frontend.
- If translation is poor, user clicks "Incorrect" and optionally submits corrected text.
- Frontend POSTs feedback to FastAPI `/feedback` endpoint.
- Backend stores feedback in DB (table `feedback_corrections`).
- A retrain orchestrator (periodic or triggered) fetches new corrections and appends them to training data, then starts a retrain job on Azure ML or locally.
- After successful retrain, CI/CD deploys the new model to Kubernetes (rolling update).

3) DB choice and schema
- Dev: SQLite (file-based) for quick setup; Prod: PostgreSQL (cloud or k8s StatefulSet + PVC)
- Minimal schema (SQLAlchemy):

```python
from sqlalchemy import Column, Integer, String, Text, DateTime, func
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class FeedbackCorrection(Base):
    __tablename__ = 'feedback_corrections'
    id = Column(Integer, primary_key=True)
    source_text = Column(Text, nullable=False)
    corrected_text = Column(Text, nullable=False)
    user_id = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    handled = Column(Integer, default=0)  # 0 = new, 1 = ingested into training
```

4) FastAPI endpoint example

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

app = FastAPI()

DATABASE_URL = "sqlite:///./feedback.db"  # switch to postgres in prod
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class FeedbackIn(BaseModel):
    source_text: str
    corrected_text: str
    user_id: str | None = None

@app.post("/feedback")
def submit_feedback(feedback: FeedbackIn):
    db = SessionLocal()
    try:
        row = FeedbackCorrection(
            source_text=feedback.source_text,
            corrected_text=feedback.corrected_text,
            user_id=feedback.user_id,
        )
        db.add(row)
        db.commit()
        return {"status": "ok", "id": row.id}
    finally:
        db.close()
```

5) Appending corrections to training data
- Write a small script `ingest_feedback.py` that reads unhandled rows from `feedback_corrections`, appends them to the cleaned training CSV (or saves a separate augmentation CSV), and marks rows as `handled=1`.

Example pseudo-logic:

```python
# ingest_feedback.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd

# read unhandled rows
# append as new rows to data/train_augmented.csv with columns `source_text,target_text`
# mark handled=1 in DB
```

6) Triggering retrain (design options)
- Option A (event-driven): When feedback is submitted, send a message to a queue (Azure Queue, Redis, RabbitMQ). A consumer batches changes and triggers an Azure ML pipeline run via CLI/SDK.
- Option B (scheduled): A cron job (Kubernetes CronJob or GitHub Actions scheduled run) runs `ingest_feedback.py` then triggers a retrain when `n` new examples arrive or after a time window.

7) Azure ML retrain invocation
- Use Azure ML CLI v2 or the Python SDK to submit a job that points to the training script and dataset path (including augmented CSVs).
- Example CLI snippet (simplified):

```bash
az ml job create --file aml_retrain_job.yaml
```

- Alternatively, use `azureml` Python SDK to programmatically submit and wait for the run.

8) CI/CD integration (GitHub Actions)
- Workflow A (on push to `main` for model code or datasets): build container images, push to container registry, update k8s manifests, and apply via `kubectl` or `helm` (use `k8s-deploy` action).
- Workflow B (on retrain complete): a job that pulls the newly produced model artifact and redeploys service (trigger from Azure ML via Webhook or poll for job completion). Use deployment strategies: `RollingUpdate` or `canary` (via `kubectl rollout` or Helm upgrades).

9) Kubernetes components
- `api` (FastAPI) Deployment + Service (ClusterIP)
- `frontend` Deployment + Service
- `db` StatefulSet (Postgres) + PVC or use managed DB
- `nginx` or Traefik Ingress for external access and TLS
- Optional: `retrainer` Job / CronJob that runs `ingest_feedback.py` and triggers Azure ML

10) Minimal frontend flow (HTML/JS)
- Provide a text input, a "Translate" button that calls `/translate` on API, and a button to mark incorrect that opens a small correction text box and submits to `/feedback`.
- Keep UI simple; use static files served by `frontend` container (React/Vue or plain JS + fetch).

11) Example `translate` FastAPI route (sketch)

```python
@app.post('/translate')
def translate(payload: dict):
    text = payload['text']
    # call model inference: either local in-memory model or call model-service
    prediction = model_infer(text)
    return {"translation": prediction}
```

12) Model serving options
- Option 1: Serve the fine-tuned HuggingFace model in the API container using `transformers` and `torch` (GPU node recommended). For k8s, consider node selectors/taints.
- Option 2: Use a dedicated model server (KFServing, TorchServe, or custom) and have the API call it.

13) Security and rate-limiting
- Add authentication for feedback submission if needed
- Add basic rate-limiting to avoid spam and accidental retrains

14) Local dev tips
- Start with SQLite + local FastAPI + simple HTML frontend. Verify feedback is persisted and `ingest_feedback.py` can append.
- Use k3d for local Kubernetes testing; use a local Postgres container or k8s PVC.
- Use `docker compose` for quick local integration (nginx + api + frontend + db).

15) Files to add to this repo (suggested)
- `mlops_project_guide.md`  <-- this file
- `inference/api/`:
  - `app.py` (FastAPI server with `/translate` and `/feedback`)
  - `requirements.txt`
  - `Dockerfile`
- `inference/frontend/`:
  - `index.html`, `app.js`, `Dockerfile`
- `infrastructure/k8s/`:
  - `api-deployment.yaml`, `frontend-deployment.yaml`, `db-statefulset.yaml`, `ingress.yaml`
- `ci/`:
  - `.github/workflows/retrain-and-deploy.yaml`
- `training/`:
  - `train.py`, `aml_retrain_job.yaml`, `ingest_feedback.py`, `requirements.txt`

16) Todo list to start in the right order
- Phase 1: local MVP
  - Make sure the current model artifact can be loaded and used for inference.
  - Build a local FastAPI app with `/translate` and `/feedback`.
  - Add SQLite and verify feedback is stored correctly.
  - Create a minimal frontend that can submit a translation and correction.
- Phase 2: training loop
  - Write `ingest_feedback.py` to append handled corrections to the training data.
  - Run one local retraining cycle from the augmented dataset.
  - Save the retrained model artifact in a predictable path.
- Phase 3: deployment
  - Containerize the API and frontend.
  - Move the API and DB into Kubernetes manifests.
  - Add a simple ingress or reverse proxy for external access.
- Phase 4: automation
  - Add Azure ML job submission for retraining.
  - Add GitHub Actions for build, retrain trigger, and deploy.
  - Add a smoke test after deployment.

17) Notes for the report and presentation
- Include screenshots of the frontend and API docs page
- Show the DB contents with `kubectl exec` (for k8s) or `sqlite3` output
- Show GitHub Actions logs for retrain and redeploy
- Explain design decisions concisely: why you chose SQLite vs Postgres, event-driven vs scheduled retrain, and deployment/update strategy

18) Extension: Deploy Different Versions of the Model
- Goal: serve multiple model variants (e.g., `small`, `medium`, `large`) simultaneously so you can route requests, compare performance, or provide low-latency fallback options.
- High-level approaches:
  - Parallel deployments: run each model version as its own Deployment + Service (e.g., `model-small`, `model-large`). Use an API or gateway to route traffic to a chosen version.
  - Single API, multiple backends: keep a single FastAPI front-end that forwards inference to the selected model-service (via internal service name).
  - Versioned artifacts: store each trained model artifact using a semantic tag or timestamp (e.g., `model:v1.0-small`, `model:v1.1-large`) in your artifact storage (Azure Blob / registry or model store).
- Kubernetes considerations:
  - Use separate Deployments for each model variant with resource requests/limits tuned per variant.
  - Use `HorizontalPodAutoscaler` or node selectors for resource segregation (GPU nodes for large models).
  - For traffic splitting, use Ingress + Traefik/NGINX with weighted routing or use a service mesh (e.g., Istio) for advanced canary/capacity control.
- CI/CD changes:
  - Tag training runs with a version string and publish artifacts to a predictable location (artifact storage path or registry tag).
  - Add a workflow that deploys a specific model version tag (e.g. `deploy-model-version` GitHub Action) which updates the Deployment image or config map referencing the artifact.
  - Support automated smoke-tests after deploy to verify the new version behaves.
- Experimentation and testing:
  - Implement a rollout strategy: start with a small percentage to the new model (canary) and increase on success.
  - Capture metrics (latency, error-rate, qualitative feedback) per-version in your logs or a monitoring stack (Prometheus + Grafana).
- Example manifest snippet (replace image / model path accordingly):

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-small
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: model
        image: myregistry.azurecr.io/animal-model:small-v1.0
        resources:
          requests:
            cpu: "500m"
            memory: "2Gi"
          limits:
            cpu: "1000m"
            memory: "4Gi"
        env:
        - name: MODEL_PATH
          value: "/models/small"
```

- Practical tips:
  - Keep a small, validated test dataset for quick smoke tests post-deploy.
  - Record which frontend users were routed to which version to enable A/B analysis.
  - Document the API option to select a model version (e.g., query param `?model=small`).
