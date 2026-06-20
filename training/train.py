from __future__ import annotations

import os
from pathlib import Path
import logging

import pandas as pd

# Heavy ML imports are deferred to runtime so `--dry-run` can work without installing heavy ML packages
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
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))


def load_csv(path: str, max_rows: int | None = None) -> pd.DataFrame:
    """Helper to load CSV and optionally sample a subset of rows to limit training runtime."""
    df = pd.read_csv(path)
    if max_rows and max_rows > 0 and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
    return df


def download_azure_data():
    """
    Downloads training and validation datasets from the Azure ML workspace blob storage.
    Uses DefaultAzureCredential for secure service principal authentication in production.
    """
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

    # Get workspace default datastore details (including secrets/SAS token)
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
        
        # Extract the blob path relative to the datastore root
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

    logger.info("Data assets downloaded successfully from Azure ML.")
    return train_path, val_path


def main(
    dry_run: bool = False,
    use_azure_data: bool = False,
    train_csv: str | None = None,
    val_csv: str | None = None,
    register: bool = True,
    max_steps: int | None = None,
    model_size: str = "large",
):
    global OUTPUT_DIR
    if model_size != "large":
        OUTPUT_DIR = str(Path(OUTPUT_DIR).parent / f"mbart-finetuned-{model_size}")

    train_csv = train_csv or TRAIN_CSV
    val_csv = val_csv or VAL_CSV
    max_steps = max_steps if max_steps is not None else MAX_STEPS

    # If enabled, fetch dataset assets from Azure ML Cloud datastore
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

    # Dry-run validation check without importing PyTorch or Transformers
    if dry_run:
        # Lightweight checks: sample and verify text formatting and whitespace token counts
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
        print("\nDry run complete — preprocessing and basic checks passed successfully.")
        return

    # Real training: import heavy ML libraries now (so dry-run can skip them entirely)
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

    # CUDA device capability logging
    cuda_available = torch.cuda.is_available()
    logger.info("========================================")
    logger.info("CUDA / GPU Verification:")
    logger.info("CUDA Available in PyTorch: %s", cuda_available)
    if cuda_available:
        logger.info("GPU Device Count: %d", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            logger.info("  GPU %d Name: %s", i, torch.cuda.get_device_name(i))
            logger.info("  GPU %d Capability: %s", i, torch.cuda.get_device_capability(i))
            logger.info("  GPU %d Memory Allocated: %.2f GB", i, torch.cuda.memory_allocated(i) / 1e9)
    else:
        logger.warning("CUDA/GPU is NOT available to PyTorch. Training will run on CPU.")
        logger.warning("If you are running in Docker, ensure Docker is configured with GPU support (e.g., --gpus all).")
    logger.info("========================================")

    # Initialize fine-tuned model and language tokenizer
    logger.info("Loading model and tokenizer: %s", MODEL_NAME)
    model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)
    tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
    
    # Prune model layers to create small/medium variants of your model
    if model_size == "small":
        model.model.encoder.layers = model.model.encoder.layers[:2]
        model.model.decoder.layers = model.model.decoder.layers[:2]
        model.config.encoder_layers = 2
        model.config.decoder_layers = 2
        logger.info("Pruned model layers to 2 encoder / 2 decoder layers (Small size)")
    elif model_size == "medium":
        model.model.encoder.layers = model.model.encoder.layers[:6]
        model.model.decoder.layers = model.model.decoder.layers[:6]
        model.config.encoder_layers = 6
        model.config.decoder_layers = 6
        logger.info("Pruned model layers to 6 encoder / 6 decoder layers (Medium size)")
    tokenizer.src_lang = os.getenv("SRC_LANG", "ar_AR")
    tokenizer.tgt_lang = os.getenv("TGT_LANG", "en_XX")
    
    # Register custom Akkadian structural tokens in the tokenizer
    tokenizer.add_special_tokens({"additional_special_tokens": ["<gap>", "<big_gap>"]})
    model.resize_token_embeddings(len(tokenizer))

    # Load dataframes into HuggingFace dataset format
    train_ds = Dataset.from_pandas(df_train)
    val_ds = Dataset.from_pandas(df_val)

    def preprocess(examples):
        """Tokenize Akkadian and English strings using maximum block boundary lengths."""
        inputs = tokenizer(examples["akkadian"], truncation=True, max_length=MAX_SOURCE_LENGTH)
        labels = tokenizer(examples["english"], truncation=True, max_length=MAX_TARGET_LENGTH)
        inputs["labels"] = labels["input_ids"]
        return inputs

    # Map preprocessing function to batch-tokenize the datasets
    train_tok = train_ds.map(preprocess, batched=True)
    val_tok = val_ds.map(preprocess, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)

    # Setup training configurations (epochs, batch sizes, steps)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        max_steps=max_steps,
        logging_steps=50,
        save_total_limit=1,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        predict_with_generate=False,
        fp16=torch.cuda.is_available(),
    )

    # Instantiate the standard HuggingFace Seq2SeqTrainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # Run the training loop
    logger.info("Starting training for %s epochs", NUM_EPOCHS)
    trainer.train()

    # Save fine-tuned model and tokenizer weights locally
    logger.info("Saving final model to %s", OUTPUT_DIR)
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Register the newly trained model version to Azure ML Model Registry
    if register:
        try:
            from azure.identity import DefaultAzureCredential
            from azure.ai.ml import MLClient
            from azure.ai.ml.entities import Model
            import shutil

            logger.info("Connecting to Azure ML workspace for model registration...")
            credential = DefaultAzureCredential()
            ml_client = MLClient(
                credential=credential,
                subscription_id="c282f4e7-0cf4-4c14-8e50-f6fecc19ce92",
                resource_group_name="azure-ai",
                workspace_name="verstraete-wout-ml"
            )

            # Create clean directory for registration to exclude heavy checkpoint folders
            clean_reg_dir = Path(OUTPUT_DIR) / "clean_registration"
            if clean_reg_dir.exists():
                shutil.rmtree(clean_reg_dir)
            clean_reg_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy only files (like model weights and configs) and exclude checkpoint directories
            for item in Path(OUTPUT_DIR).iterdir():
                if item.is_file():
                    shutil.copy(item, clean_reg_dir / item.name)

            logger.info("Registering model from clean path: %s", clean_reg_dir)
            model_name = f"akkadian-translation-model-{model_size}"
            model_asset = Model(
                path=str(clean_reg_dir),
                type="custom_model",
                name=model_name,
                description=f"Finetuned mBART {model_size} model for Akkadian to English translation",
                tags={"size": model_size}
            )
            
            registered_model = ml_client.models.create_or_update(model_asset)
            logger.info("Successfully registered model: %s version: %s", registered_model.name, registered_model.version)
            
            # Clean up the temporary clean registration directory
            shutil.rmtree(clean_reg_dir)
        except Exception as exc:
            logger.error("Failed to register model in Azure ML: %s", exc)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run a lightweight dry run without ML dependencies")
    parser.add_argument("--use-azure-data", action="store_true", help="Download and use data assets directly from Azure ML")
    parser.add_argument("--train-csv", type=str, default=None, help="Path to training CSV file")
    parser.add_argument("--val-csv", type=str, default=None, help="Path to validation CSV file")
    parser.add_argument("--no-register", action="store_true", help="Do not register the model in Azure ML after training")
    parser.add_argument("--max-steps", type=int, default=None, help="Limit training to a maximum number of steps (default: 200)")
    parser.add_argument("--model-size", type=str, default="large", choices=["small", "medium", "large"], help="The size of the model (small, medium, large)")
    parsed = parser.parse_args()
    
    # Do not register if dry run is selected
    register_model = not parsed.dry_run and not parsed.no_register
    
    main(
        dry_run=parsed.dry_run,
        use_azure_data=parsed.use_azure_data,
        train_csv=parsed.train_csv,
        val_csv=parsed.val_csv,
        register=register_model,
        max_steps=parsed.max_steps,
        model_size=parsed.model_size,
    )
