#!/usr/bin/env python3
"""
Cleanup utility for Video Upscaler.
Clears uploads, processed files, and database records.
"""
import os
import shutil
import argparse
from pathlib import Path
from app.models import SessionLocal, ProcessingTask

def clear_uploads(upload_dir="uploads"):
    """Clear all uploaded files."""
    upload_path = Path(upload_dir)
    if upload_path.exists():
        file_count = len(list(upload_path.glob("*")))
        if file_count > 0:
            shutil.rmtree(upload_path)
            upload_path.mkdir(exist_ok=True)
            print(f"🗑️  Cleared {file_count} files from uploads directory")
        else:
            print("📁 Uploads directory is already empty")
    else:
        print("📁 Uploads directory doesn't exist")

def clear_processed(processed_dir="processed"):
    """Clear all processed files."""
    processed_path = Path(processed_dir)
    if processed_path.exists():
        files = list(processed_path.glob("*"))
        # Filter out .DS_Store and other hidden files
        video_files = [f for f in files if f.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.3gp', '.3g2']]
        
        if video_files:
            for file in video_files:
                file.unlink()
            print(f"🗑️  Cleared {len(video_files)} processed video files")
        else:
            print("📁 Processed directory has no video files")
    else:
        print("📁 Processed directory doesn't exist")

def clear_database():
    """Clear all database records."""
    db = SessionLocal()
    try:
        # Count existing tasks
        task_count = db.query(ProcessingTask).count()
        
        if task_count > 0:
            # Delete all tasks
            db.query(ProcessingTask).delete()
            db.commit()
            print(f"🗑️  Cleared {task_count} database records")
        else:
            print("🗄️  Database is already empty")
    except Exception as e:
        print(f"❌ Error clearing database: {e}")
        db.rollback()
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
    
    # Database info
    db = SessionLocal()
    try:
        db_count = db.query(ProcessingTask).count()
    except:
        db_count = 0
    finally:
        db.close()
    
    def format_size(size_bytes):
        """Format bytes to human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    print("\n📊 Current Storage Usage:")
    print(f"   📁 Uploads: {upload_count} files ({format_size(upload_size)})")
    print(f"   🎬 Processed: {processed_count} files ({format_size(processed_size)})")
    print(f"   🗄️  Database: {db_count} records")
    print(f"   💾 Total: {format_size(upload_size + processed_size)}")

def main():
    parser = argparse.ArgumentParser(description="Cleanup Video Upscaler files and database")
    parser.add_argument("--uploads", action="store_true", help="Clear uploaded files")
    parser.add_argument("--processed", action="store_true", help="Clear processed files")
    parser.add_argument("--database", action="store_true", help="Clear database records")
    parser.add_argument("--all", action="store_true", help="Clear everything (uploads, processed, database)")
    parser.add_argument("--info", action="store_true", help="Show storage usage information")
    parser.add_argument("--confirm", action="store_true", help="Skip confirmation prompt")
    
    args = parser.parse_args()
    
    # Show info if requested or no other action specified
    if args.info or not any([args.uploads, args.processed, args.database, args.all]):
        get_storage_info()
        if not any([args.uploads, args.processed, args.database, args.all]):
            print("\nUse --help to see cleanup options")
            return
    
    # Determine what to clean
    clean_uploads = args.uploads or args.all
    clean_processed = args.processed or args.all
    clean_database = args.database or args.all
    
    if not any([clean_uploads, clean_processed, clean_database]):
        print("No cleanup actions specified. Use --help for options.")
        return
    
    # Confirmation prompt
    if not args.confirm:
        actions = []
        if clean_uploads:
            actions.append("uploaded files")
        if clean_processed:
            actions.append("processed files")
        if clean_database:
            actions.append("database records")
        
        print(f"\n⚠️  This will permanently delete: {', '.join(actions)}")
        response = input("Are you sure? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Cleanup cancelled")
            return
    
    print("\n🧹 Starting cleanup...")
    
    # Perform cleanup
    if clean_uploads:
        clear_uploads()
    
    if clean_processed:
        clear_processed()
    
    if clean_database:
        clear_database()
    
    print("✅ Cleanup completed!")
    
    # Show updated info
    if any([clean_uploads, clean_processed, clean_database]):
        get_storage_info()

if __name__ == "__main__":
    main()
