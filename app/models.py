"""
Database models for the Video Upscaler application.
"""
from datetime import datetime
from enum import Enum
from sqlalchemy import Column, String, Integer, DateTime, Text, Float, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

Base = declarative_base()

class TaskStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ProcessingTask(Base):
    """Model for video processing tasks."""
    __tablename__ = 'processing_tasks'
    
    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    input_path = Column(String, nullable=False)
    output_path = Column(String, nullable=False)
    status = Column(String, default=TaskStatus.QUEUED.value)
    progress = Column(Float, default=0.0)
    scale = Column(Integer, default=4)
    model = Column(String, default="RealESRGAN_x4plus")
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    file_size = Column(Integer, nullable=True)
    
    def to_dict(self):
        """Convert task to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'filename': self.filename,
            'original_filename': self.original_filename,
            'status': self.status,
            'progress': self.progress,
            'scale': self.scale,
            'model': self.model,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'file_size': self.file_size
        }

# Database configuration
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///video_upscaler.db')

# Create engine and session factory
if DATABASE_URL.startswith('sqlite'):
    # Allow SQLite connections to be used across threads (required for background workers)
    engine = create_engine(
        DATABASE_URL,
        echo=False,
        connect_args={'check_same_thread': False}
    )
else:
    engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
