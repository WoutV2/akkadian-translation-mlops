import sys
import os
from pathlib import Path

# Add project root to sys.path before importing local modules to enable absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from inference.api.app import FeedbackCorrection, TrainData, ValidationData, TestData

def main():
    """
    Main feedback ingestion entrypoint. Fetches unhandled user-submitted feedback,
    moves it to the training dataset table, updates the handled flag in the DB,
    exports database tables to local CSV files, and registers the datasets to Azure ML.
    """
    PROJECT_DIR = Path(__file__).resolve().parents[1]
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        print("DATABASE_URL environment variable is not set!")
        sys.exit(1)
    
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # 1. Fetch all unhandled feedback corrections
        unhandled = session.query(FeedbackCorrection).filter_by(handled=0).all()
        if not unhandled:
            print("No new feedback to ingest.")
            has_changes = False
        else:
            print(f"Found {len(unhandled)} unhandled feedback corrections.")
            
            # 2. Insert into train_data table and mark as handled (flag = 1)
            for row in unhandled:
                new_train_row = TrainData(
                    akkadian=row.source_text,
                    english=row.corrected_text
                )
                session.add(new_train_row)
                row.handled = 1
            session.commit()
            print(f"Successfully moved {len(unhandled)} corrections to train_data.")
            has_changes = True
            
        # 3. Export all data tables to local CSVs to prepare for training
        data_dir = PROJECT_DIR / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Export train_data table
        train_csv_path = data_dir / "train_cleaned.csv"
        print(f"Exporting train_data table to {train_csv_path}...")
        df_train = pd.read_sql_table("train_data", con=engine)
        # Keep only the required columns for training
        df_train = df_train[["akkadian", "english"]]
        df_train.to_csv(train_csv_path, index=False)
        print(f"Exported {len(df_train)} train rows.")
        
        # Export validation_data table
        val_csv_path = data_dir / "validation_cleaned.csv"
        print(f"Exporting validation_data table to {val_csv_path}...")
        df_val = pd.read_sql_table("validation_data", con=engine)
        df_val = df_val[["akkadian", "english"]]
        df_val.to_csv(val_csv_path, index=False)
        print(f"Exported {len(df_val)} validation rows.")

        # Export test_data table
        test_csv_path = data_dir / "test_cleaned.csv"
        print(f"Exporting test_data table to {test_csv_path}...")
        df_test = pd.read_sql_table("test_data", con=engine)
        df_test = df_test[["akkadian", "english"]]
        df_test.to_csv(test_csv_path, index=False)
        print(f"Exported {len(df_test)} test rows.")

        # 4. Upload/Register updated dataset files to Azure ML default datastore if changes occurred
        if has_changes:
            register_azure_datasets(PROJECT_DIR)

        # 5. Output changes indicator for GitHub Actions workflow variables
        github_output = os.getenv("GITHUB_OUTPUT")
        if github_output:
            with open(github_output, "a") as f:
                f.write(f"has_changes={'true' if has_changes else 'false'}\n")
        print(f"Ingestion completed. Has changes: {has_changes}")

    except Exception as e:
        session.rollback()
        print(f"Error during ingestion: {e}")
        raise e
    finally:
        session.close()

def register_azure_datasets(project_dir):
    """
    Registers updated training, validation, and test CSV datasets to Azure ML Studio.
    This enables versioned data tracking and allows Azure ML jobs to download the latest files.
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
        
        data_dir = Path(project_dir) / "data"
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
