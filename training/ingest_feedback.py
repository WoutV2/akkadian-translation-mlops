import sys
import os
from pathlib import Path

# Add project root to sys.path before importing local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from inference.api.app import FeedbackCorrection, TrainData, ValidationData, TestData

def main():
    PROJECT_DIR = Path(__file__).resolve().parents[1]
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        print("DATABASE_URL environment variable is not set!")
        sys.exit(1)
    DATABASE_URL = DATABASE_URL.strip()
    
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # 1. Fetch unhandled corrections
        unhandled = session.query(FeedbackCorrection).filter_by(handled=0).all()
        if not unhandled:
            print("No new feedback to ingest.")
            has_changes = False
        else:
            print(f"Found {len(unhandled)} unhandled feedback corrections.")
            
            # 2. Insert into train_data table and mark as handled
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
            
        # 3. Export all data tables to local CSVs
        data_dir = PROJECT_DIR / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Export train_data
        train_csv_path = data_dir / "train_cleaned.csv"
        print(f"Exporting train_data table to {train_csv_path}...")
        df_train = pd.read_sql_table("train_data", con=engine)
        # Keep only the required columns
        df_train = df_train[["akkadian", "english"]]
        df_train.to_csv(train_csv_path, index=False)
        print(f"Exported {len(df_train)} train rows.")
        
        # Export validation_data
        val_csv_path = data_dir / "validation_cleaned.csv"
        print(f"Exporting validation_data table to {val_csv_path}...")
        df_val = pd.read_sql_table("validation_data", con=engine)
        df_val = df_val[["akkadian", "english"]]
        df_val.to_csv(val_csv_path, index=False)
        print(f"Exported {len(df_val)} validation rows.")

        # Export test_data
        test_csv_path = data_dir / "test_cleaned.csv"
        print(f"Exporting test_data table to {test_csv_path}...")
        df_test = pd.read_sql_table("test_data", con=engine)
        df_test = df_test[["akkadian", "english"]]
        df_test.to_csv(test_csv_path, index=False)
        print(f"Exported {len(df_test)} test rows.")

        # Output changes indicator for GitHub actions
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

if __name__ == "__main__":
    main()

