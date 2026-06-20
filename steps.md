# Project Approach & Chronological Steps

This document summarizes the chronological approach, order of implementation, and the real-world engineering issues encountered and resolved during this project. Use this as a reference guide for your report and presentation.

---

## 1. Chronological Approach & Order of Implementation

### Phase 1: Local MVP
1. **Environment Setup**: Created a virtual environment (`.venv`) to isolate dependencies.
2. **Data Preprocessing**: Wrote `data/preprocess.py` to clean raw Akkadian transcripts and English translation text files (handling subscript normalization, determinatives, signs counting, and grouping lines into semantic blocks).
3. **Inference API**: Built a FastAPI application (`inference/api/app.py`) exposing `/translate` and `/feedback` endpoints.
4. **Local Database**: Integrated SQLite to store user translation corrections in the `feedback_corrections` table.
5. **Interactive UI**: Developed a lightweight frontend (`inference/frontend/index.html` and `app.js`) to allow typing Akkadian, reviewing translations, and submitting corrected translations.

### Phase 2: Ingestion & Local Training
1. **Feedback Loop Ingestion**: Implemented `training/ingest_feedback.py` to extract unhandled feedback entries, append them to the local training dataset (`train_cleaned.csv`), and mark them as handled in the database.
2. **Local Fine-Tuning**: Developed the training loop in `training/train.py` using HuggingFace's `Seq2SeqTrainer` to fine-tune `facebook/mbart-large-50-many-to-many-mmt`.
3. **Contamination Prevention**: Added logic to the `/feedback` endpoint that automatically deletes matching corrected records from the hold-out test set (`test_data` table) to prevent evaluation data leakage.

### Phase 3: Cloud Integration & CI/CD
1. **Azure ML Data Registration**: Configured `ingest_feedback.py` to automatically upload and register the augmented training, validation, and test datasets as versioned data assets in the Azure ML Studio workspace.
2. **Azure ML Model Registry**: Configured `train.py` to automatically register the fine-tuned model to Azure ML workspace as a custom model after training completes.
3. **GitHub Workflows**:
   - `build-deploy.yml`: Triggered on code pushes to compile/push API and Frontend images to ACR and deploy to AKS.
   - `daily-retrain.yml`: A scheduled nightly job running on a self-hosted Windows runner to ingest new feedback database records, run GPU retraining locally, and deploy the updated model version to AKS.

---

## 2. Issues Faced & Engineering Resolutions

### Issue 1: Missing Translation Logic in API
* **Problem**: Initially, the translation backend served mock/incomplete logic and did not properly run HuggingFace model inference.
* **Resolution**: Implemented the `TranslationService` class in `app.py` to load model weights and tokenizers, initialize device context (CUDA GPU vs CPU), and generate tokens with specific beam search parameters (`num_beams=2`) and ngram repetition limits to ensure stable translation outputs.

### Issue 2: HuggingFace Tokenizer Compatibility Crash
* **Problem**: When loading local fine-tuned tokenizers, newer versions of HuggingFace `transformers` crashed due to key incompatibilities (`extra_special_tokens` vs `additional_special_tokens`).
* **Resolution**: Created a custom `_sanitize_local_tokenizer` function in `app.py` that reads the tokenizer JSON configuration on start, patches older keys into newer, compatible configurations, and writes it back to prevent loading exceptions.

### Issue 3: Windows Console Unicode Encoding Crashes
* **Problem**: When running retraining dry-runs on Windows machines, printing sample Akkadian text to the console crashed Python with `UnicodeEncodeError`.
* **Resolution**: Added a character sanitizer to the stdout print statements in `train.py` (`src.encode('ascii', errors='replace').decode('ascii')`) to safely display character samples on standard terminal outputs without raising encoding exceptions.

### Issue 4: Redundant Optimizer States in Model Registry
* **Problem**: Initial model registrations pushed training checkpoint folders containing optimizer states (`optimizer.pt`, 4.8GB), wasting significant network bandwidth and storage in the Azure ML registry.
* **Resolution**: Added a clean registration directory creation step in `train.py` (`clean_registration/`) that copies only the model weights and tokenizer files (excluding checkpoint subfolders) before executing model registration.

### Issue 5: Extremely Large Docker Build Times
* **Problem**: Rebuilding the inference API docker container redownloaded massive libraries (PyTorch/Transformers) every time, leading to slow deployments.
* **Resolution**: Set up Docker Buildx with GitHub Actions runner caching (`cache-from: type=gha`, `cache-to: type=gha,mode=max`) to persist package cache directories and drastically reduce rebuild times.

### Issue 6: Ingress URL Rewriting Issues on AKS
* **Problem**: Complex NGINX URL rewrite annotations in Kubernetes ingress caused static assets (CSS, JS) to map incorrectly, breaking the frontend interface.
* **Resolution**: Restructured the API routing paths to mount the frontend static directory directly to `/static` inside FastAPI, and simplified `ingress.yaml` to route exact endpoints `/translate` and `/feedback` to the API service, mapping `/` directly to the frontend.