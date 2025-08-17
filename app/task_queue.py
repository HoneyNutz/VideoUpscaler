"""
Self-contained task queue system using SQLite and threading.
No external dependencies like Redis or Celery required.
"""
import threading
import time
import queue
import logging
from typing import Callable, Optional, Dict, Any
from datetime import datetime
from .models import SessionLocal, ProcessingTask, TaskStatus
from .video_processor import VideoProcessor

logger = logging.getLogger(__name__)

class TaskQueue:
    """SQLite-based task queue with worker threads."""
    
    def __init__(self, num_workers: int = 1, poll_interval: float = 1.0):
        self.num_workers = num_workers
        self.poll_interval = poll_interval
        self.workers = []
        self.running = False
        self.progress_callbacks: Dict[str, Callable] = {}
        self._lock = threading.Lock()
    
    def start(self):
        """Start the worker threads."""
        if self.running:
            return
        
        self.running = True
        logger.info(f"Starting {self.num_workers} worker threads")
        
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"TaskWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
    
    def stop(self):
        """Stop all worker threads."""
        logger.info("Stopping task queue workers")
        self.running = False
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        self.workers.clear()
    
    def add_task(self, task_id: str, progress_callback: Optional[Callable] = None):
        """Add a task to the queue."""
        with self._lock:
            if progress_callback:
                self.progress_callbacks[task_id] = progress_callback
        
        logger.info(f"Task {task_id} added to queue")
    
    def _worker_loop(self):
        """Main worker loop that processes tasks."""
        logger.info(f"Worker {threading.current_thread().name} started")
        
        while self.running:
            try:
                # Get next queued task from database
                task = self._get_next_task()
                
                if task:
                    logger.info(f"Processing task {task.id}")
                    self._process_task(task)
                else:
                    # No tasks available, sleep and check again
                    time.sleep(self.poll_interval)
                    
            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)
                time.sleep(self.poll_interval)
        
        logger.info(f"Worker {threading.current_thread().name} stopped")
    
    def _get_next_task(self) -> Optional[ProcessingTask]:
        """Get the next queued task from the database."""
        db = SessionLocal()
        try:
            # Get oldest queued task
            task = db.query(ProcessingTask).filter(
                ProcessingTask.status == TaskStatus.QUEUED.value
            ).order_by(ProcessingTask.created_at.asc()).first()
            
            if task:
                # Mark as processing to prevent other workers from picking it up
                task.status = TaskStatus.PROCESSING.value
                task.started_at = datetime.utcnow()
                db.commit()
                db.refresh(task)
                return task
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting next task: {e}")
            db.rollback()
            return None
        finally:
            db.close()
    
    def _process_task(self, task: ProcessingTask):
        """Process a single video task."""
        db = SessionLocal()
        
        try:
            # Get progress callback
            progress_callback = None
            with self._lock:
                progress_callback = self.progress_callbacks.get(task.id)
            
            def update_progress(progress: float, status: str):
                """Update task progress in database and notify via callback."""
                try:
                    # Update database
                    current_task = db.query(ProcessingTask).filter(
                        ProcessingTask.id == task.id
                    ).first()
                    
                    if current_task:
                        current_task.progress = progress
                        if status in [s.value for s in TaskStatus]:
                            current_task.status = status
                        db.commit()
                    
                    # Call progress callback if available
                    if progress_callback:
                        progress_callback(task.id, progress, status)
                        
                except Exception as e:
                    logger.error(f"Error updating progress: {e}")
            
            # Initialize video processor
            logger.info(f"Initializing VideoProcessor for task {task.id}: model={task.model}, scale={task.scale}")
            processor = VideoProcessor(
                model_name=task.model,
                scale=task.scale,
                progress_callback=update_progress
            )
            
            # Process the video
            logger.info(f"Starting video processing for task {task.id}: {task.input_path} -> {task.output_path}")
            processor.process_video(task.input_path, task.output_path)
            logger.info(f"Video processing completed for task {task.id}")
            
            # Mark as completed - refresh task from database first
            current_task = db.query(ProcessingTask).filter(
                ProcessingTask.id == task.id
            ).first()
            
            if current_task:
                current_task.status = TaskStatus.COMPLETED.value
                current_task.progress = 100.0
                current_task.completed_at = datetime.utcnow()
                db.commit()
                db.refresh(current_task)
                logger.info(f"Task {task.id} marked as completed in database")
            
            logger.info(f"Task {task.id} completed successfully")
            
            # Final progress update
            if progress_callback:
                progress_callback(task.id, 100.0, TaskStatus.COMPLETED.value)
            
        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}", exc_info=True)
            
            # Mark as failed - refresh task from database first
            current_task = db.query(ProcessingTask).filter(
                ProcessingTask.id == task.id
            ).first()
            
            if current_task:
                current_task.status = TaskStatus.FAILED.value
                current_task.error_message = str(e)
                current_task.completed_at = datetime.utcnow()
                db.commit()
                db.refresh(current_task)
                logger.info(f"Task {task.id} marked as failed in database")
            
            # Notify failure
            if progress_callback:
                progress_callback(task.id, current_task.progress if current_task else 0, TaskStatus.FAILED.value)
            
        finally:
            # Clean up progress callback
            with self._lock:
                self.progress_callbacks.pop(task.id, None)
            
            db.close()

# Global task queue instance
task_queue = TaskQueue(num_workers=1)

def start_task_queue():
    """Start the global task queue."""
    task_queue.start()

def stop_task_queue():
    """Stop the global task queue."""
    task_queue.stop()

def enqueue_task(task_id: str, progress_callback: Optional[Callable] = None):
    """Enqueue a task for processing."""
    task_queue.add_task(task_id, progress_callback)
