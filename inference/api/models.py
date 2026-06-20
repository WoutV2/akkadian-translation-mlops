from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class TrainData(Base):
    """
    SQLAlchemy table for training datasets containing original/augmented Akkadian and English pairs.
    """
    __tablename__ = "train_data"
    id = Column(Integer, primary_key=True, autoincrement=True)
    akkadian = Column(Text, nullable=False)
    english = Column(Text, nullable=False)

class ValidationData(Base):
    """
    SQLAlchemy table for validating the translations during model training.
    """
    __tablename__ = "validation_data"
    id = Column(Integer, primary_key=True, autoincrement=True)
    akkadian = Column(Text, nullable=False)
    english = Column(Text, nullable=False)

class TestData(Base):
    """
    SQLAlchemy table containing the hold-out test set to evaluate translator generalization.
    """
    __tablename__ = "test_data"
    id = Column(Integer, primary_key=True, autoincrement=True)
    akkadian = Column(Text, nullable=False)
    english = Column(Text, nullable=False)

class FeedbackCorrection(Base):
    """
    SQLAlchemy table that stores correction feedback submitted by users through the UI.
    """
    __tablename__ = "feedback_corrections"
    id = Column(Integer, primary_key=True)
    source_text = Column(Text, nullable=False)     # The original Akkadian query
    corrected_text = Column(Text, nullable=False)    # User's submitted correction
    translated_text = Column(Text, nullable=True)   # The model's original translation
    user_id = Column(String(64), nullable=True)     # Optional identifier for the user
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    handled = Column(Integer, default=0, nullable=False) # 0 = pending, 1 = ingested to training dataset
