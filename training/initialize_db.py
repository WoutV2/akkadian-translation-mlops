import os
import sys
from pathlib import Path
import pandas as pd
from sqlalchemy import Column, Integer, Text, String, DateTime, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone

# Add project root directory to system path
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_DIR))

Base = declarative_base()

class TrainData(Base):
    """SQLAlchemy table for training datasets containing Akkadian and English pairs."""
    __tablename__ = "train_data"
    id = Column(Integer, primary_key=True, autoincrement=True)
    akkadian = Column(Text, nullable=False)
    english = Column(Text, nullable=False)

class ValidationData(Base):
    """SQLAlchemy table for validating the translations during model training."""
    __tablename__ = "validation_data"
    id = Column(Integer, primary_key=True, autoincrement=True)
    akkadian = Column(Text, nullable=False)
    english = Column(Text, nullable=False)

class TestData(Base):
    """SQLAlchemy table containing the hold-out test set to evaluate translator generalization."""
    __tablename__ = "test_data"
    id = Column(Integer, primary_key=True, autoincrement=True)
    akkadian = Column(Text, nullable=False)
    english = Column(Text, nullable=False)

class FeedbackCorrection(Base):
    """SQLAlchemy table that stores correction feedback submitted by users through the UI."""
    __tablename__ = "feedback_corrections"
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_text = Column(Text, nullable=False)
    corrected_text = Column(Text, nullable=False)
    translated_text = Column(Text, nullable=True)
    user_id = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    handled = Column(Integer, default=0, nullable=False)

def get_database_url():
    """Retrieves the DATABASE_URL environment variable, raising an error if it is not set."""
    url = os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL environment variable is not set!")
    return url

def load_and_map_df(path):
    """Loads a CSV file and normalizes headers to 'akkadian' and 'english'."""
    df = pd.read_csv(path)
    if "source_text" in df.columns:
        df = df.rename(columns={"source_text": "akkadian", "target_text": "english"})
    return df[["akkadian", "english"]]

def main():
    """
    Main database migration and initialization function. Creates DB tables if they don't exist,
    migrates/populates initial dataset entries from local clean CSV files, and registers files to Azure ML.
    """
    db_url = get_database_url()
    print(f"Connecting to database...")
    engine = create_engine(db_url)
    
    # 1. Create all SQL tables
    print("Creating tables if they do not exist...")
    Base.metadata.create_all(bind=engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # 2. Populate TrainData table if empty
        if session.query(TrainData).first() is None:
            train_csv = PROJECT_DIR / "data" / "train_cleaned.csv"
            if train_csv.exists():
                print(f"Loading train data from {train_csv}...")
                df = load_and_map_df(train_csv)
                print(f"Inserting {len(df)} train rows into DB...")
                # Write to database in chunks of 1000 to prevent buffer overflow/memory consumption
                df.to_sql("train_data", con=engine, if_exists="append", index=False, chunksize=1000)
                print("Train data loaded successfully.")
            else:
                print(f"Train CSV not found at {train_csv}")
        else:
            print("train_data table already contains data. Skipping migration.")

        # 3. Populate ValidationData table if empty
        if session.query(ValidationData).first() is None:
            val_csv = PROJECT_DIR / "data" / "validation_cleaned.csv"
            if val_csv.exists():
                print(f"Loading validation data from {val_csv}...")
                df = load_and_map_df(val_csv)
                print(f"Inserting {len(df)} validation rows into DB...")
                df.to_sql("validation_data", con=engine, if_exists="append", index=False, chunksize=1000)
                print("Validation data loaded successfully.")
            else:
                print(f"Validation CSV not found at {val_csv}")
        else:
            print("validation_data table already contains data. Skipping migration.")

        # 4. Populate TestData table if empty
        if session.query(TestData).first() is None:
            test_csv = PROJECT_DIR / "data" / "test_cleaned.csv"
            if test_csv.exists():
                print(f"Loading test data from {test_csv}...")
                df = load_and_map_df(test_csv)
                print(f"Inserting {len(df)} test rows into DB...")
                df.to_sql("test_data", con=engine, if_exists="append", index=False, chunksize=1000)
                print("Test data loaded successfully.")
            else:
                print(f"Test CSV not found at {test_csv}")
        else:
            print("test_data table already contains data. Skipping migration.")

    except Exception as e:
        print(f"Error during migration: {e}")
        session.rollback()
        raise e
    finally:
        session.close()
        
    print("Database initialization and migration completed.")

    # 5. Register initial/updated datasets to Azure ML Studio
    register_azure_datasets()

def register_azure_datasets():
    """
    Registers the initial datasets to Azure ML Studio workspace default storage container.
    """
    try:
        from azure.identity import DefaultAzureCredential
        from azure.ai.ml import MLClient
        from azure.ai.ml.entities import Data
        from azure.ai.ml.constants import AssetTypes
        
        print("Connecting to Azure ML workspace to register/update data assets...")
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id="c282f4e7-0cf4-4c14-8e50-f6fecc19ce92",
            resource_group_name="azure-ai",
            workspace_name="verstraete-wout-ml"
        )
        
        data_dir = PROJECT_DIR / "data"
        train_csv = data_dir / "train_cleaned.csv"
        val_csv = data_dir / "validation_cleaned.csv"
        test_csv = data_dir / "test_cleaned.csv"

        datasets = [
            ("train_cleaned", train_csv, "Cleaned Akkadian to English training dataset"),
            ("validation_cleaned", val_csv, "Cleaned Akkadian to English validation dataset"),
            ("test_cleaned", test_csv, "Cleaned Akkadian to English test dataset")
        ]
        
        for name, path, desc in datasets:
            if path.exists():
                print(f"Registering dataset '{name}' from {path}...")
                data_asset = Data(
                    path=str(path),
                    type=AssetTypes.URI_FILE,
                    description=desc,
                    name=name
                )
                registered = ml_client.data.create_or_update(data_asset)
                print(f"Successfully registered '{name}' version {registered.version}")
            else:
                print(f"Skipping registration for '{name}'; file not found at {path}")
    except Exception as e:
        print(f"Error during Azure ML dataset registration: {e}")

if __name__ == "__main__":
    main()
