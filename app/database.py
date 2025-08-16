"""
Database service layer for task management.
"""
from typing import Optional, List
from sqlalchemy.orm import Session
from .models import ProcessingTask, TaskStatus, get_db
import uuid
from datetime import datetime

class TaskService:
    """Service class for managing processing tasks."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_task(
        self,
        filename: str,
        original_filename: str,
        input_path: str,
        output_path: str,
        scale: int = 4,
        model: str = "RealESRGAN_x4plus",
        file_size: Optional[int] = None
    ) -> ProcessingTask:
        """Create a new processing task."""
        task = ProcessingTask(
            id=str(uuid.uuid4()),
            filename=filename,
            original_filename=original_filename,
            input_path=input_path,
            output_path=output_path,
            scale=scale,
            model=model,
            file_size=file_size,
            status=TaskStatus.QUEUED.value,
            created_at=datetime.utcnow()
        )
        
        self.db.add(task)
        self.db.commit()
        self.db.refresh(task)
        return task
    
    def get_task(self, task_id: str) -> Optional[ProcessingTask]:
        """Get a task by ID."""
        return self.db.query(ProcessingTask).filter(ProcessingTask.id == task_id).first()
    
    def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        progress: Optional[float] = None,
        error_message: Optional[str] = None
    ) -> Optional[ProcessingTask]:
        """Update task status and progress."""
        task = self.get_task(task_id)
        if not task:
            return None
        
        task.status = status.value
        if progress is not None:
            task.progress = progress
        if error_message is not None:
            task.error_message = error_message
        
        # Update timestamps based on status
        if status == TaskStatus.PROCESSING and not task.started_at:
            task.started_at = datetime.utcnow()
        elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            task.completed_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(task)
        return task
    
    def get_active_tasks(self) -> List[ProcessingTask]:
        """Get all active (queued or processing) tasks."""
        return self.db.query(ProcessingTask).filter(
            ProcessingTask.status.in_([TaskStatus.QUEUED.value, TaskStatus.PROCESSING.value])
        ).all()
    
    def get_recent_tasks(self, limit: int = 50) -> List[ProcessingTask]:
        """Get recent tasks ordered by creation date."""
        return self.db.query(ProcessingTask).order_by(
            ProcessingTask.created_at.desc()
        ).limit(limit).all()
    
    def delete_task(self, task_id: str) -> bool:
        """Delete a task and its associated files."""
        task = self.get_task(task_id)
        if not task:
            return False
        
        # Clean up files
        import os
        try:
            if os.path.exists(task.input_path):
                os.remove(task.input_path)
            if os.path.exists(task.output_path):
                os.remove(task.output_path)
        except OSError:
            pass  # Files might already be deleted
        
        self.db.delete(task)
        self.db.commit()
        return True
