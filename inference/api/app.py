from __future__ import annotations

# Trigger rebuild for PyTorch thread optimization deployment
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
import logging

import torch
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
import json

# Setup directory paths relative to this file
_APP_PATH = Path(__file__).resolve()
PROJECT_DIR = _APP_PATH.parents[2] if len(_APP_PATH.parents) > 2 else _APP_PATH.parent
INFERENCE_DIR = PROJECT_DIR / "inference"
FRONTEND_DIR = INFERENCE_DIR / "frontend"

# Database & Model Config via Environment Variables (with SQLite and HF defaults)
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{PROJECT_DIR / 'feedback.db'}")
MODEL_NAME = os.getenv("MODEL_NAME", "facebook/mbart-large-50-many-to-many-mmt")
MODEL_DIR = os.getenv("MODEL_DIR", str(PROJECT_DIR / "mbart-finetuned")).strip()
SRC_LANG = os.getenv("SRC_LANG", "ar_AR")
TGT_LANG = os.getenv("TGT_LANG", "en_XX")
MAX_SOURCE_LENGTH = int(os.getenv("MAX_SOURCE_LENGTH", "192"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "128"))
NUM_BEAMS = int(os.getenv("NUM_BEAMS", "2"))

# Routing configuration for model sizes (small, medium, large)
SERVICE_MODEL_VERSION = os.getenv("SERVICE_MODEL_VERSION", "large").lower().strip()
ACTIVE_VERSIONS = os.getenv("ACTIVE_VERSIONS", "small,medium,large").strip()

# Database Schema Declarations imported from models.py
try:
    from inference.api.models import Base, TrainData, ValidationData, TestData, FeedbackCorrection
except ModuleNotFoundError:
    from models import Base, TrainData, ValidationData, TestData, FeedbackCorrection

# Setup SQLAlchemy engine and thread-safe session factories
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Pydantic schemas for request/response serialization and validation
class TranslateRequest(BaseModel):
    text: str = Field(min_length=1, description="Akkadian text to translate")
    model_version: str = Field(default="v2", description="Model version/tag to use (e.g. v1, v2, v3)")

class TranslateResponse(BaseModel):
    translation: str = Field(description="Generated English translation")
    model_source: str = Field(description="Source of the model loaded (e.g. HuggingFace repo or local path)")
    model_version: str = Field(default="v2", description="The version/tag of the model that served the translation")

class FeedbackRequest(BaseModel):
    source_text: str = Field(min_length=1, description="Original Akkadian text")
    corrected_text: str = Field(min_length=1, description="Corrected English translation")
    translated_text: str | None = None
    user_id: str | None = None

class FeedbackResponse(BaseModel):
    status: str
    id: int

class TranslationService:
    """
    A service class wrapping tokenizer initialization, HuggingFace model load,
    tokenizer sanitization patches, and thread-safe translation inference.
    """
    def __init__(self, model_source: str, model: MBartForConditionalGeneration, tokenizer: MBart50TokenizerFast):
        self.model_source = model_source
        self.model = model
        self.tokenizer = tokenizer
        self.lock = threading.Lock()

    @classmethod
    def load(cls) -> "TranslationService":
        """
        Initializes and returns the TranslationService. Decides whether to load the model
        from a local fine-tuned directory or pull it from the HuggingFace model registry.
        """
        model_source = MODEL_DIR if MODEL_DIR and Path(MODEL_DIR).exists() else MODEL_NAME
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("translation")
        logger.info("Using model source: %s", model_source)



        # Validate directory content if loading a local fine-tuned model
        if model_source != MODEL_NAME:
            ms_path = Path(model_source)
            has_weights = any((ms_path / fname).exists() for fname in (
                "model.safetensors",
                "pytorch_model.bin",
                "pytorch_model.bin.index.json",
            ))
            if not has_weights:
                raise RuntimeError(
                    f"Local model directory '{model_source}' does not contain model weights."
                    " Ensure you set `MODEL_DIR` to your trained model folder (contains model.safetensors)."
                )

        def _sanitize_local_tokenizer(dir_path: Path) -> None:
            """
            Sanitizes the configuration files of the local tokenizer.
            Older versions of transformers use extra_special_tokens which can lead to
            compatibility issues in newer environments. We map it to additional_special_tokens.
            """
            for fname in ("tokenizer_config.json", "tokenizer.json"):
                p = dir_path / fname
                if not p.exists():
                    continue
                try:
                    text = p.read_text(encoding="utf-8")
                    data = json.loads(text)
                except Exception:
                    logger.info("Could not read or parse %s; skipping sanitization", p)
                    continue
                changed = False
                if "extra_special_tokens" in data:
                    if isinstance(data["extra_special_tokens"], list):
                        data["additional_special_tokens"] = data.pop("extra_special_tokens")
                        changed = True
                    elif isinstance(data["extra_special_tokens"], dict) and "additional_special_tokens" in data["extra_special_tokens"].keys():
                        data["additional_special_tokens"] = data["extra_special_tokens"].pop("additional_special_tokens")
                        if not data["extra_special_tokens"]:
                            data.pop("extra_special_tokens")
                        changed = True
                if changed:
                    try:
                        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                        logger.info("Patched tokenizer file %s to compatible format", p)
                    except Exception as e:
                        logger.warning("Failed to write patched tokenizer file %s: %s", p, e)

        # Apply the sanitization patch before loading the tokenizer to avoid parsing crashes
        try:
            _sanitize_local_tokenizer(ms_path)
        except Exception:
            logger.info("Tokenizer sanitization failed or was skipped")

        # Load MBartForConditionalGeneration model weights
        try:
            model = MBartForConditionalGeneration.from_pretrained(model_source)
        except Exception as exc:
            logger.error("Failed to load model from %s: %s", model_source, exc)
            raise RuntimeError(
                f"Failed to load model from local model at '{model_source}'."
                " Fix model files or remove MODEL_DIR."
            )

        # Load the fast tokenizer wrapper
        try:
            tokenizer = MBart50TokenizerFast.from_pretrained(model_source)
        except Exception as exc:
            logger.warning("Failed to load tokenizer from %s: %s", model_source, exc)
            if model_source != MODEL_NAME:
                try:
                    _sanitize_local_tokenizer(Path(model_source))
                    tokenizer = MBart50TokenizerFast.from_pretrained(model_source)
                except Exception as exc2:
                    logger.error("Local tokenizer load failed after sanitization: %s", exc2)
                    raise RuntimeError(
                        f"Failed to load tokenizer from local model at '{model_source}'."
                        " Fix tokenizer files (tokenizer.json / tokenizer_config.json) or remove MODEL_DIR."
                    )
            else:
                raise

        # Setup source & target language tokens
        tokenizer.src_lang = SRC_LANG
        tokenizer.tgt_lang = TGT_LANG
        
        # Add Akkadian-specific special tokens for gaps (e.g. damaged/missing script pieces)
        try:
            tokenizer.add_special_tokens({"additional_special_tokens": ["<gap>", "<big_gap>"]})
        except Exception:
            logger.info("Could not add extra special tokens to tokenizer; continuing")

        # Resize model embeddings if new special tokens were added
        model.resize_token_embeddings(len(tokenizer))
        
        # Enforce target language tokens at generation start
        if hasattr(tokenizer, "lang_code_to_id") and TGT_LANG in tokenizer.lang_code_to_id:
            target_lang_id = tokenizer.lang_code_to_id[TGT_LANG]
            model.config.decoder_start_token_id = target_lang_id
            model.config.forced_bos_token_id = target_lang_id
            model.generation_config.decoder_start_token_id = target_lang_id
            model.generation_config.forced_bos_token_id = target_lang_id

        # Place model in evaluation mode and move to GPU if available
        model.eval()
        if torch.cuda.is_available():
            model.to("cuda")
        else:
            torch.set_num_threads(1)
            logger.info("Setting PyTorch CPU thread count to 1 for optimized container inference")

        return cls(model_source=model_source, model=model, tokenizer=tokenizer)

    def translate(self, text: str) -> str:
        """
        Executes tokenizer encoding, model generation, and decoding.
        Includes a repetition penalty and prevents repeating n-grams to stabilize output.

        For the small model running on a CPU-throttled pod, greedy decoding
        (num_beams=1) is used to avoid 504 gateway timeouts caused by beam search
        being too slow under a low CPU limit.
        """
        is_small = SERVICE_MODEL_VERSION == "small"

        # Cap tokens and use greedy decoding for the small variant to stay within timeout
        max_tokens = min(MAX_NEW_TOKENS, 32) if is_small else MAX_NEW_TOKENS
        num_beams = 1 if is_small else NUM_BEAMS

        with self.lock:
            encoded = self.tokenizer(text, return_tensors="pt", max_length=MAX_SOURCE_LENGTH, truncation=True)
            if torch.cuda.is_available():
                encoded = {k: v.to("cuda") for k, v in encoded.items()}
            with torch.no_grad():
                generate_kwargs = dict(
                    max_new_tokens=max_tokens,
                    num_beams=num_beams,
                    decoder_start_token_id=self.model.config.decoder_start_token_id,
                    forced_bos_token_id=self.model.config.forced_bos_token_id,
                )
                # Repetition penalty + no-repeat n-gram only make sense with beam search
                if num_beams > 1:
                    generate_kwargs["repetition_penalty"] = 2.0
                    generate_kwargs["no_repeat_ngram_size"] = 3
                generated_tokens = self.model.generate(**encoded, **generate_kwargs)
            translation = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            return translation

# Global translation service instance (lazy-loaded in lifespan)
translation_service: TranslationService | None = None

def init_db() -> None:
    """Helper to create all database tables if they do not exist."""
    Base.metadata.create_all(bind=engine)

def get_translation_service() -> TranslationService:
    """Retrieves translation service instance, failing if not initialized."""
    if translation_service is None:
        raise RuntimeError("Translation service is not ready")
    return translation_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager. Performs database setup and compiles
    the translation model at application startup, releasing memory at shutdown.
    """
    global translation_service
    init_db()
    if SERVICE_MODEL_VERSION == "router":
        logging.info("Running in ROUTER mode. No local model will be loaded.")
    else:
        translation_service = TranslationService.load()
    yield
    translation_service = None

# Initialize FastAPI application
app = FastAPI(title="Akkadian to English Translation Hub", lifespan=lifespan)

# Mount the static frontend directory if it exists
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def serve_frontend() -> FileResponse:
    """Serves the main frontend page."""
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_file)

@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint for Kubernetes probes and CI smoke tests."""
    if SERVICE_MODEL_VERSION == "router":
        versions = [v.strip().lower() for v in ACTIVE_VERSIONS.split(",") if v.strip()]
        default_version = versions[-1] if versions else "v2"
        target_url = os.getenv(f"{default_version.upper()}_API_URL", f"http://akk-api-{default_version}:8000")
        try:
            import httpx
            with httpx.Client(timeout=1.5) as client:
                resp = client.get(f"{target_url}/health")
                if resp.status_code != 200:
                    raise HTTPException(status_code=503, detail=f"Default model version ({default_version}) is loading")
        except Exception:
            raise HTTPException(status_code=503, detail="Model services are initializing or offline")
    else:
        if translation_service is None:
            raise HTTPException(status_code=503, detail="Local translation service is loading model weights")
    return {"status": "ok"}

@app.get("/versions")
def get_versions() -> dict:
    """Exposes the list of active/configured model versions."""
    versions = [v.strip().lower() for v in ACTIVE_VERSIONS.split(",") if v.strip()]
    return {"versions": versions, "default": "large"}

@app.get("/config")
def config() -> dict[str, str]:
    """Exposes current active runtime configurations for debugging."""
    return {
        "model_source": translation_service.model_source if translation_service else "loading",
        "database_url": DATABASE_URL,
    }

@app.post("/translate", response_model=TranslateResponse)
def translate(payload: TranslateRequest) -> TranslateResponse:
    """Translates the input Akkadian string using the active TranslationService or routes it."""
    requested_version = payload.model_version.lower().strip()
    
    # If this service is the requested model version, translate locally
    if SERVICE_MODEL_VERSION == requested_version:
        service = get_translation_service()
        return TranslateResponse(
            translation=service.translate(payload.text),
            model_source=service.model_source,
            model_version=SERVICE_MODEL_VERSION
        )
    
    # Otherwise, forward/route the request to the corresponding internal model service
    target_url = os.getenv(f"{requested_version.upper()}_API_URL", f"http://akk-api-{requested_version}:8000")
    
    if target_url:
        try:
            import httpx
            logging.info(f"Routing translation request to version {requested_version} service at {target_url}")
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(f"{target_url}/translate", json=payload.dict())
                if resp.status_code == 200:
                    data = resp.json()
                    return TranslateResponse(
                        translation=data.get("translation"),
                        model_source=data.get("model_source"),
                        model_version=data.get("model_version", requested_version)
                    )
                else:
                    raise HTTPException(status_code=resp.status_code, detail=resp.text)
        except Exception as e:
            logging.warning(f"Failed to route request to {requested_version} at {target_url}: {e}. Falling back to local model.")
            
    # Fallback to local model if available
    if translation_service is not None:
        service = get_translation_service()
        return TranslateResponse(
            translation=service.translate(payload.text),
            model_source=service.model_source + f" (local fallback from {requested_version})",
            model_version=SERVICE_MODEL_VERSION
        )
    else:
        raise HTTPException(
            status_code=503,
            detail=f"Model service for '{requested_version}' is unavailable and no local model fallback is loaded (running in router mode)."
        )

@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(payload: FeedbackRequest) -> FeedbackResponse:
    """
    Saves user-submitted translation feedback to the SQLite/PostgreSQL database.
    To prevent evaluation contamination, we remove the matching source text
    from the hold-out test set if it exists there.
    """
    session = SessionLocal()
    try:
        row = FeedbackCorrection(
            source_text=payload.source_text.strip(),
            corrected_text=payload.corrected_text.strip(),
            translated_text=payload.translated_text.strip() if payload.translated_text else None,
            user_id=payload.user_id.strip() if payload.user_id else None,
        )
        session.add(row)
        
        # Contamination prevention: remove matching row from test data set
        clean_source = payload.source_text.strip()
        deleted_count = session.query(TestData).filter(TestData.akkadian == clean_source).delete()
        if deleted_count > 0:
            logging.info(f"Removed {deleted_count} matching row(s) from test_data table.")
            
        session.commit()
        session.refresh(row)
        return FeedbackResponse(status="ok", id=row.id)
    except Exception as e:
        session.rollback()
        logging.error(f"Error in submit_feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@app.post("/feedback/ingest")
def ingest_feedback() -> dict:
    """
    Endpoint triggering immediate ingestion of unhandled feedback corrections.
    Fetches unhandled rows, updates database handled flags, and appends the 
    corrections directly to `train_cleaned.csv`.
    """
    session = SessionLocal()
    try:
        unhandled = session.query(FeedbackCorrection).filter_by(handled=0).all()
        if not unhandled:
            return {"status": "no new feedback to ingest"}
        new_data = []
        for row in unhandled:
            new_data.append({
                "akkadian": row.source_text,
                "english": row.corrected_text
            })
        for row in unhandled:
            row.handled = 1
        train_csv_path = PROJECT_DIR / "data" / "train_cleaned.csv"
        train_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_new = pd.DataFrame(new_data)
        if train_csv_path.exists():
            df_new.to_csv(train_csv_path, mode='a', header=False, index=False)
        else:
            df_new.to_csv(train_csv_path, mode='w', header=True, index=False)
        session.commit()
        return {"status": "success", "count": len(new_data)}
    finally:
        session.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("inference.api.app:app", host="0.0.0.0", port=8000, reload=True)