from __future__ import annotations

import os
from pathlib import Path
import logging

import pandas as pd

# Heavy ML imports are deferred to runtime so `--dry-run` can work without installing packages
torch = None
Dataset = None
MBartForConditionalGeneration = None
MBart50TokenizerFast = None
Seq2SeqTrainer = None
Seq2SeqTrainingArguments = None
DataCollatorForSeq2Seq = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Trigger cloud retraining pipeline
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

# Defaults inspired by mbart 5 notebook (fast-test settings)
TRAIN_CSV = os.getenv("TRAIN_CSV", str(DATA_DIR / "train_cleaned.csv"))
VAL_CSV = os.getenv("VAL_CSV", str(DATA_DIR / "validation_cleaned.csv"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", str(PROJECT_DIR / "mbart-finetuned"))
MODEL_NAME = os.getenv("MODEL_NAME", "facebook/mbart-large-50-many-to-many-mmt")
# Fast-test mode mirrors the notebook: use smaller sample caps and fewer epochs
FAST_TEST_RUN = os.getenv("FAST_TEST_RUN", "1") in ("1", "true", "True")
BLOCK_LENGTH = int(os.getenv("BLOCK_LENGTH", "192"))
MAX_SOURCE_LENGTH = int(os.getenv("MAX_SOURCE_LENGTH", str(BLOCK_LENGTH)))
MAX_TARGET_LENGTH = int(os.getenv("MAX_TARGET_LENGTH", str(BLOCK_LENGTH)))
MAX_TRAIN_SAMPLES = int(os.getenv("MAX_TRAIN_SAMPLES", "1200" if FAST_TEST_RUN else "-1"))
MAX_VAL_SAMPLES = int(os.getenv("MAX_VAL_SAMPLES", "200" if FAST_TEST_RUN else "-1"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "1" if FAST_TEST_RUN else "5"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))


def load_csv(path: str, max_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if max_rows and max_rows > 0 and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
    return df


def download_azure_data():
    from azure.identity import DefaultAzureCredential
    from azure.ai.ml import MLClient
    from azure.storage.blob import BlobClient
    import tempfile
    from pathlib import Path

    logger.info("Connecting to Azure ML workspace...")
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id="c282f4e7-0cf4-4c14-8e50-f6fecc19ce92",
        resource_group_name="azure-ai",
        workspace_name="verstraete-wout-ml"
    )

    tmp_dir = Path(tempfile.gettempdir()) / "azure_ml_data"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Get datastore details (including credentials/SAS token)
    datastore = ml_client.datastores.get("workspaceblobstore", include_secrets=True)
    storage_account_name = datastore.account_name
    container_name = datastore.container_name
    sas_token = datastore.credentials.get("sas_token") if datastore.credentials else None
    if sas_token and not sas_token.startswith("?"):
        sas_token = "?" + sas_token

    def download_asset(asset_name, filename):
        logger.info("Fetching metadata for %s...", asset_name)
        asset = ml_client.data.get(name=asset_name, label="latest")
        path_uri = asset.path
        
        # Extract the blob path relative to the datastore
        marker = "/paths/"
        idx = path_uri.find(marker)
        if idx == -1:
            raise ValueError(f"Could not parse blob path from asset URI: {path_uri}")
        blob_path = path_uri[idx + len(marker):]

        logger.info("Downloading %s from storage account %s...", filename, storage_account_name)
        blob_url = f"https://{storage_account_name}.blob.core.windows.net/{container_name}/{blob_path}{sas_token or ''}"
        blob_client = BlobClient.from_blob_url(blob_url)
        
        local_path = tmp_dir / filename
        with open(local_path, "wb") as f:
            download_stream = blob_client.download_blob()
            download_stream.readinto(f)
        
        logger.info("Downloaded %s to %s", asset_name, local_path)
        return str(local_path)

    train_path = download_asset("train_cleaned", "train_cleaned.csv")
    val_path = download_asset("validation_cleaned", "validation_cleaned.csv")

    logger.info("Data assets downloaded successfully.")
    return train_path, val_path


def main(dry_run: bool = False, use_azure_data: bool = False, train_csv: str | None = None, val_csv: str | None = None):
    train_csv = train_csv or TRAIN_CSV
    val_csv = val_csv or VAL_CSV

    if use_azure_data:
        try:
            train_csv, val_csv = download_azure_data()
        except Exception as exc:
            logger.error("Failed to download data assets from Azure ML: %s", exc)
            logger.info("Falling back to local data files.")

    logger.info("Reading CSV files: %s, %s", train_csv, val_csv)
    df_train = load_csv(train_csv, MAX_TRAIN_SAMPLES)
    df_val = load_csv(val_csv, MAX_VAL_SAMPLES)

    logger.info("Train samples: %s; Val samples: %s", len(df_train), len(df_val))

    if dry_run:
        # Lightweight checks without ML deps: sample and basic tokenization
        sample_train = df_train.sample(n=min(5, len(df_train)), random_state=42).reset_index(drop=True)
        for i, row in sample_train.iterrows():
            src = row.get("akkadian") or row.get("source_text") or ""
            tgt = row.get("english") or row.get("target_text") or ""
            # Prevent UnicodeEncodeError on Windows terminals by replacing unprintable characters
            clean_src = src[:200].encode('ascii', errors='replace').decode('ascii')
            clean_tgt = tgt[:200].encode('ascii', errors='replace').decode('ascii')
            print(f"\nSample {i+1} source: {clean_src}")
            print(f"Sample {i+1} target: {clean_tgt}")
            print(f"Source tokens (whitespace): {len(src.split())}; Target tokens: {len(tgt.split())}")
        print("\nDry run complete — preprocessing and basic checks passed.")
        return

    # Real training: import heavy libs now (so dry-run can skip them)
    global torch, Dataset, MBartForConditionalGeneration, MBart50TokenizerFast, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
    try:
        import torch as _torch
        from datasets import Dataset as _Dataset
        from transformers import (
            MBartForConditionalGeneration as _MBartForConditionalGeneration,
            MBart50TokenizerFast as _MBart50TokenizerFast,
            Seq2SeqTrainer as _Seq2SeqTrainer,
            Seq2SeqTrainingArguments as _Seq2SeqTrainingArguments,
            DataCollatorForSeq2Seq as _DataCollatorForSeq2Seq,
        )
    except Exception as exc:
        logger.error("Required ML packages are not installed: %s", exc)
        raise

    torch = _torch
    Dataset = _Dataset
    MBartForConditionalGeneration = _MBartForConditionalGeneration
    MBart50TokenizerFast = _MBart50TokenizerFast
    Seq2SeqTrainer = _Seq2SeqTrainer
    Seq2SeqTrainingArguments = _Seq2SeqTrainingArguments
    DataCollatorForSeq2Seq = _DataCollatorForSeq2Seq

    logger.info("Loading model and tokenizer: %s", MODEL_NAME)
    model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)
    tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.src_lang = os.getenv("SRC_LANG", "ar_AR")
    tokenizer.tgt_lang = os.getenv("TGT_LANG", "en_XX")
    tokenizer.add_special_tokens({"additional_special_tokens": ["<gap>", "<big_gap>"]})
    model.resize_token_embeddings(len(tokenizer))

    train_ds = Dataset.from_pandas(df_train)
    val_ds = Dataset.from_pandas(df_val)

    def preprocess(examples):
        inputs = tokenizer(examples["akkadian"], truncation=True, max_length=MAX_SOURCE_LENGTH)
        labels = tokenizer(examples["english"], truncation=True, max_length=MAX_TARGET_LENGTH)
        inputs["labels"] = labels["input_ids"]
        return inputs

    train_tok = train_ds.map(preprocess, batched=True)
    val_tok = val_ds.map(preprocess, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=50,
        save_total_limit=1,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        predict_with_generate=False,
        fp16=torch.cuda.is_available(),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    logger.info("Starting training for %s epochs", NUM_EPOCHS)
    trainer.train()

    logger.info("Saving final model to %s", OUTPUT_DIR)
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run a lightweight dry run without ML dependencies")
    parser.add_argument("--use-azure-data", action="store_true", help="Download and use data assets directly from Azure ML")
    parser.add_argument("--train-csv", type=str, default=None, help="Path to training CSV file")
    parser.add_argument("--val-csv", type=str, default=None, help="Path to validation CSV file")
    parsed = parser.parse_args()
    main(
        dry_run=parsed.dry_run,
        use_azure_data=parsed.use_azure_data,
        train_csv=parsed.train_csv,
        val_csv=parsed.val_csv,
    )
