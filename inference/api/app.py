from __future__ import annotations

import os
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
from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
import json

_APP_PATH = Path(__file__).resolve()
PROJECT_DIR = _APP_PATH.parents[2] if len(_APP_PATH.parents) > 2 else _APP_PATH.parent
INFERENCE_DIR = PROJECT_DIR / "inference"
FRONTEND_DIR = INFERENCE_DIR / "frontend"

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{PROJECT_DIR / 'feedback.db'}")
MODEL_NAME = os.getenv("MODEL_NAME", "facebook/mbart-large-50-many-to-many-mmt")
MODEL_DIR = os.getenv("MODEL_DIR", str(PROJECT_DIR / "mbart-finetuned")).strip()
SRC_LANG = os.getenv("SRC_LANG", "ar_AR")
TGT_LANG = os.getenv("TGT_LANG", "en_XX")
MAX_SOURCE_LENGTH = int(os.getenv("MAX_SOURCE_LENGTH", "192"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "128"))
NUM_BEAMS = int(os.getenv("NUM_BEAMS", "2"))

Base = declarative_base()

class TrainData(Base):
    __tablename__ = "train_data"
    id = Column(Integer, primary_key=True, autoincrement=True)
    akkadian = Column(Text, nullable=False)
    english = Column(Text, nullable=False)

class ValidationData(Base):
    __tablename__ = "validation_data"
    id = Column(Integer, primary_key=True, autoincrement=True)
    akkadian = Column(Text, nullable=False)
    english = Column(Text, nullable=False)

class TestData(Base):
    __tablename__ = "test_data"
    id = Column(Integer, primary_key=True, autoincrement=True)
    akkadian = Column(Text, nullable=False)
    english = Column(Text, nullable=False)

class FeedbackCorrection(Base):
    __tablename__ = "feedback_corrections"
    id = Column(Integer, primary_key=True)
    source_text = Column(Text, nullable=False)
    corrected_text = Column(Text, nullable=False)
    translated_text = Column(Text, nullable=True)
    user_id = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    handled = Column(Integer, default=0, nullable=False)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class TranslateRequest(BaseModel):
    text: str = Field(min_length=1)

class TranslateResponse(BaseModel):
    translation: str
    model_source: str

class FeedbackRequest(BaseModel):
    source_text: str = Field(min_length=1)
    corrected_text: str = Field(min_length=1)
    translated_text: str | None = None
    user_id: str | None = None

class FeedbackResponse(BaseModel):
    status: str
    id: int

class TranslationService:
    def __init__(self, model_source: str, model: MBartForConditionalGeneration, tokenizer: MBart50TokenizerFast):
        self.model_source = model_source
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def load(cls) -> "TranslationService":
        model_source = MODEL_DIR if MODEL_DIR and Path(MODEL_DIR).exists() else MODEL_NAME
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("translation")
        logger.info("Using model source: %s", model_source)

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

        try:
            _sanitize_local_tokenizer(ms_path)
        except Exception:
            logger.info("Tokenizer sanitization failed or was skipped")

        try:
            model = MBartForConditionalGeneration.from_pretrained(model_source)
        except Exception as exc:
            logger.error("Failed to load model from %s: %s", model_source, exc)
            raise RuntimeError(
                f"Failed to load model from local model at '{model_source}'."
                " Fix model files or remove MODEL_DIR."
            )

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

        tokenizer.src_lang = SRC_LANG
        tokenizer.tgt_lang = TGT_LANG
        try:
            tokenizer.add_special_tokens({"additional_special_tokens": ["<gap>", "<big_gap>"]})
        except Exception:
            logger.info("Could not add extra special tokens to tokenizer; continuing")

        model.resize_token_embeddings(len(tokenizer))
        if hasattr(tokenizer, "lang_code_to_id") and TGT_LANG in tokenizer.lang_code_to_id:
            target_lang_id = tokenizer.lang_code_to_id[TGT_LANG]
            model.config.decoder_start_token_id = target_lang_id
            model.config.forced_bos_token_id = target_lang_id
            model.generation_config.decoder_start_token_id = target_lang_id
            model.generation_config.forced_bos_token_id = target_lang_id

        model.eval()
        if torch.cuda.is_available():
            model.to("cuda")

        return cls(model_source=model_source, model=model, tokenizer=tokenizer)

    def translate(self, text: str) -> str:
        encoded = self.tokenizer(text, return_tensors="pt", max_length=MAX_SOURCE_LENGTH, truncation=True)
        if torch.cuda.is_available():
            encoded = {k: v.to("cuda") for k, v in encoded.items()}
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **encoded,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=NUM_BEAMS,
                repetition_penalty=2.0,
                no_repeat_ngram_size=3,
                decoder_start_token_id=self.model.config.decoder_start_token_id,
                forced_bos_token_id=self.model.config.forced_bos_token_id,
            )
        translation = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translation

translation_service: TranslationService | None = None

def init_db() -> None:
    Base.metadata.create_all(bind=engine)

def get_translation_service() -> TranslationService:
    if translation_service is None:
        raise RuntimeError("Translation service is not ready")
    return translation_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    global translation_service
    init_db()
    translation_service = TranslationService.load()
    yield
    translation_service = None

app = FastAPI(title="Akkadian to English MVP", lifespan=lifespan)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def serve_frontend() -> FileResponse:
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_file)

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/config")
def config() -> dict[str, str]:
    return {
        "model_source": translation_service.model_source if translation_service else "loading",
        "database_url": DATABASE_URL,
    }

@app.post("/translate", response_model=TranslateResponse)
def translate(payload: TranslateRequest) -> TranslateResponse:
    service = get_translation_service()
    return TranslateResponse(translation=service.translate(payload.text), model_source=service.model_source)

@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(payload: FeedbackRequest) -> FeedbackResponse:
    session = SessionLocal()
    try:
        row = FeedbackCorrection(
            source_text=payload.source_text.strip(),
            corrected_text=payload.corrected_text.strip(),
            translated_text=payload.translated_text.strip() if payload.translated_text else None,
            user_id=payload.user_id.strip() if payload.user_id else None,
        )
        session.add(row)
        
        # Remove matching row from test data set
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