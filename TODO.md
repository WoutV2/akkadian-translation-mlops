# Project TODO

Last updated: 2026-06-02

## Task 1 — Cloud Training (Azure ML)
- [x] Create local training script (`training/train.py`) using the Akkadian dataset
- [x] Add Azure ML job spec (`training/aml_retrain_job.yaml`) and test submission
- [x] Run a successful Azure ML pipeline job (verify artifacts stored)

## Task 2 — Kubernetes Deployment (k3d)
- [x] Implement FastAPI backend (`inference/api/app.py`) with `/translate` and `/feedback`
- [x] Add DB schema and local SQLite setup; verify writes to `feedback_corrections`
- [x] Create frontend (`inference/frontend/index.html`, `app.js`) to call API
- [x] Write k8s manifests (`infrastructure/k8s/`) for api, frontend, db (StatefulSet+PVC), ingress
- [x] Deploy and validate application in local k3d cluster
 - [x] Add `docker-compose.yml` for local integration (api + frontend)

## Task 3 — CI/CD Automation (GitHub Actions)
- [x] Workflow: trigger Azure ML retrain on dataset or training-code changes
- [x] Workflow: deploy new model artifact to cluster after successful retrain
- [x] Workflow: build and redeploy frontend/API on code changes

## Extensions
- [ ] Deploy different versions of the model (small / medium / large)
	- [ ] Add model-version tagging in training pipeline
	- [ ] Publish model artifacts with version tags
	- [ ] Add k8s manifests for each version (`model-small`, `model-medium`, `model-large`)
	- [ ] Add deployment workflow (canary / rolling) to promote versions
	- [ ] Add API/frontend option to select or route to a specific model version

## Notes
- Use `mlops_project_guide.md` as the implementation reference.
- Keep this checklist updated as tasks are completed.