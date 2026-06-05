import os
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from inference.api.app import FeedbackCorrection, Base

def main():
    PROJECT_DIR = Path(__file__).resolve().parents[1]
    DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{PROJECT_DIR / 'feedback.db'}")
    
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()

    unhandled = session.query(FeedbackCorrection).filter_by(handled=0).all()
    if not unhandled:
        print("No new feedback to ingest.")
        return

    print(f"Found {len(unhandled)} unhandled feedback corrections.")
    
    new_data = []
    for row in unhandled:
        new_data.append({
            "akkadian": row.source_text,
            "english": row.corrected_text
        })
        row.handled = 1

    train_csv_path = PROJECT_DIR / "data" / "train_cleaned.csv"
    
    df_new = pd.DataFrame(new_data)
    
    if train_csv_path.exists():
        df_new.to_csv(train_csv_path, mode='a', header=False, index=False)
    else:
        df_new.to_csv(train_csv_path, mode='w', header=True, index=False)
        
    session.commit()
    print(f"Successfully appended {len(new_data)} rows to {train_csv_path}")

if __name__ == "__main__":
    main()
