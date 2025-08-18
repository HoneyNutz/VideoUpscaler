import os
import shutil
from pathlib import Path
from .models import SessionLocal, ProcessingTask

def clear_uploads(upload_dir="uploads"):
    """Clear all uploaded files."""
    upload_path = Path(upload_dir)
    if upload_path.exists():
        file_count = len(list(upload_path.glob("*")))
        if file_count > 0:
            shutil.rmtree(upload_path)
            upload_path.mkdir(exist_ok=True)
            return f"Cleared {file_count} files from uploads directory"
        else:
            return "Uploads directory is already empty"
    return "Uploads directory doesn't exist"

def clear_processed(processed_dir="processed"):
    """Clear all processed files."""
    processed_path = Path(processed_dir)
    if processed_path.exists():
        files = list(processed_path.glob("*"))
        video_files = [f for f in files if f.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.3gp', '.3g2']]
        
        if video_files:
            for file in video_files:
                file.unlink()
            return f"Cleared {len(video_files)} processed video files"
        else:
            return "Processed directory has no video files"
    return "Processed directory doesn't exist"

def clear_database():
    """Clear all database records."""
    db = SessionLocal()
    try:
        task_count = db.query(ProcessingTask).count()
        if task_count > 0:
            db.query(ProcessingTask).delete()
            db.commit()
            return f"Cleared {task_count} database records"
        else:
            return "Database is already empty"
    except Exception as e:
        db.rollback()
        return f"Error clearing database: {e}"
    finally:
        db.close()

def get_storage_info():
    """Get current storage usage information."""
    upload_path = Path("uploads")
    processed_path = Path("processed")
    
    upload_size = 0
    upload_count = 0
    if upload_path.exists():
        for file in upload_path.rglob("*"):
            if file.is_file():
                upload_size += file.stat().st_size
                upload_count += 1
    
    processed_size = 0
    processed_count = 0
    if processed_path.exists():
        for file in processed_path.rglob("*"):
            if file.is_file() and file.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.3gp', '.3g2']:
                processed_size += file.stat().st_size
                processed_count += 1
    
    db = SessionLocal()
    try:
        db_count = db.query(ProcessingTask).count()
    except:
        db_count = 0
    finally:
        db.close()
    
    def format_size(size_bytes):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    return {
        'uploads': {'count': upload_count, 'size': format_size(upload_size)},
        'processed': {'count': processed_count, 'size': format_size(processed_size)},
        'database': {'count': db_count},
        'total': {'size': format_size(upload_size + processed_size)}
    }
