from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename
import os
import logging
from pathlib import Path
import uuid
import traceback
import shutil

from .models import create_tables, get_db, ProcessingTask, TaskStatus
from .database import TaskService
from .task_queue import start_task_queue, enqueue_task
from . import cleanup
from download_models import MODELS, download_file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize extensions
socketio = SocketIO()

def create_app(config=None):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Default configuration
    app.config.update(
        SECRET_KEY=os.environ.get('SECRET_KEY') or 'dev-key-please-change',
        UPLOAD_FOLDER=os.path.abspath('uploads'),
        PROCESSED_FOLDER=os.path.abspath('processed'),
        MODELS_DIR=os.path.abspath('models'),
        MAX_CONTENT_LENGTH=2 * 1024 * 1024 * 1024,  # 2GB max file size
    )
    
    # Override with custom config if provided
    if config is not None:
        app.config.update(config)
    
    # Ensure upload and processed directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    
    # Initialize database
    create_tables()
    
    # Start task queue
    start_task_queue()
    
    # Initialize extensions
    socketio.init_app(app, async_mode='threading', cors_allowed_origins="*", logger=True, engineio_logger=True)
    
    # Register blueprints or routes
    register_routes(app)
    
    return app

def register_routes(app):
    """Register all routes with the Flask application."""
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/upload', methods=['POST'])
    def upload_file():
        db = next(get_db())
        task_service = TaskService(db)
        
        try:
            if 'file' not in request.files:
                logger.error('No file part in the request')
                return jsonify({'error': 'No file part'}), 400
            
            file = request.files['file']
            if file.filename == '':
                logger.error('No file selected')
                return jsonify({'error': 'No selected file'}), 400
            
            # Validate file extension
            allowed_extensions = {'mp4', 'avi', 'mov', 'mkv', '3gp', '3g2'}
            file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
            if file_ext not in allowed_extensions:
                logger.error(f'Invalid file extension: {file.filename}')
                return jsonify({'error': 'Invalid file type. Allowed types: ' + ', '.join(sorted(allowed_extensions))}), 400
            
            # Generate a unique task ID
            task_id = str(uuid.uuid4())
            
            # Ensure upload directory exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
            
            # Save the uploaded file
            filename = secure_filename(f"{task_id}_{file.filename}")
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Handle 3GP to MP4 conversion for output path
            if file.filename.lower().endswith(('.3gp', '.3g2')):
                # Convert 3GP extension to MP4 for output
                base_name = os.path.splitext(filename)[0]
                output_filename = f"upscaled_{base_name}.mp4"
            else:
                output_filename = f"upscaled_{filename}"
            
            output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
            
            try:
                file.save(input_path)
                file_size = os.path.getsize(input_path)
                logger.info(f'Saved uploaded file to {input_path} ({file_size} bytes)')
            except Exception as e:
                logger.error(f'Error saving file: {str(e)}', exc_info=True)
                return jsonify({'error': f'Error saving file: {str(e)}'}), 500
            
            # Get processing parameters
            try:
                scale = int(request.form.get('scale', 4))
                model = request.form.get('model', 'RealESRGAN_x4plus')
                custom_save_path = request.form.get('save_path', '').strip()
                
                if scale not in [2, 4]:
                    raise ValueError('Scale must be either 2 or 4')
                    
                # Validate model exists in models directory
                model_path = os.path.join('models', f'{model}.pth')
                if not os.path.exists(model_path):
                    raise ValueError(f'Model {model} not found. Please download it first.')
                
                # Handle custom save path
                if custom_save_path:
                    # Validate and sanitize custom save path
                    custom_save_path = os.path.abspath(custom_save_path)
                    if not os.path.exists(os.path.dirname(custom_save_path)):
                        try:
                            os.makedirs(os.path.dirname(custom_save_path), exist_ok=True)
                        except Exception as e:
                            raise ValueError(f'Cannot create save directory: {str(e)}')
                    
                    # Use custom path but ensure proper filename
                    if os.path.isdir(custom_save_path):
                        output_path = os.path.join(custom_save_path, output_filename)
                    else:
                        output_path = custom_save_path
                else:
                    # Use default processed folder
                    output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
                
            except ValueError as e:
                logger.error(f'Invalid parameters: {str(e)}', exc_info=True)
                return jsonify({'error': f'Invalid parameters: {str(e)}'}), 400
            
            # Create task in database
            task = task_service.create_task(
                filename=filename,
                original_filename=file.filename,
                input_path=input_path,
                output_path=output_path,
                scale=scale,
                model=model,
                file_size=file_size
            )
            
            logger.info(f'Created task {task.id} for file {file.filename}')
            
            # Start processing with self-contained task queue
            try:
                def progress_callback(task_id, progress, status):
                    """Emit progress updates via WebSocket."""
                    socketio.emit('progress_update', {
                        'task_id': task_id,
                        'progress': progress,
                        'status': status
                    })
                
                enqueue_task(task.id, progress_callback)
                return jsonify({'task_id': task.id, 'status': 'queued'})
                
            except Exception as e:
                logger.error(f'Error starting video processing: {str(e)}', exc_info=True)
                task_service.update_task_status(
                    task.id, 
                    TaskStatus.FAILED, 
                    error_message=str(e)
                )
                return jsonify({
                    'error': 'Failed to start processing',
                    'details': str(e)
                }), 500
                
        except Exception as e:
            logger.error(f'Unexpected error in upload_file: {str(e)}', exc_info=True)
            return jsonify({
                'error': 'An unexpected error occurred',
                'details': str(e)
            }), 500
        finally:
            db.close()
        
    @app.route('/status/<task_id>')
    def get_status(task_id):
        db = next(get_db())
        task_service = TaskService(db)
        try:
            task = task_service.get_task(task_id)
            if not task:
                return jsonify({'error': 'Task not found'}), 404
            output_filename = os.path.basename(task.output_path) if task.output_path else None
            return jsonify({
                'status': task.status,
                'progress': task.progress,
                'filename': task.original_filename,
                'output_filename': output_filename,
                'error': task.error_message
            })
        finally:
            db.close()

    @app.route('/api/storage/info')
    def storage_info():
        try:
            info = cleanup.get_storage_info()
            return jsonify(info)
        except Exception as e:
            logger.error(f"Error getting storage info: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    # Removed duplicate /api/cleanup route (kept consolidated implementation below)

    @app.route('/api/models', methods=['GET'])
    def get_models():
        installed_models = [p.stem for p in Path(app.config['MODELS_DIR']).glob('*.pth')]
        models_status = []
        for name, url in MODELS.items():
            models_status.append({
                'name': name,
                'url': url,
                'installed': name in installed_models
            })
        return jsonify(models_status)

    @app.route('/api/models/download', methods=['POST'])
    def download_model_route():
        data = request.json
        model_name = data.get('model')
        if not model_name or model_name not in MODELS:
            return jsonify({'error': 'Invalid model name'}), 400
        
        url = MODELS[model_name]
        destination = os.path.join(app.config['MODELS_DIR'], f'{model_name}.pth')
        
        try:
            # Use the download function from download_models.py
            from download_models import download_file as download_model_file
            download_model_file(url, destination)
            return jsonify({'status': 'success', 'message': f'{model_name} downloaded successfully.'})
        except Exception as e:
            logger.error(f"Error downloading model {model_name}: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/benchmark/results')
    def benchmark_results():
        results_path = os.path.join(app.static_folder, 'data', 'benchmark_results.json')
        if os.path.exists(results_path):
            return send_from_directory(os.path.dirname(results_path), os.path.basename(results_path))
        else:
            return jsonify([])

    
    @app.route('/favicon.ico')
    def favicon():
        return send_from_directory(
            os.path.join(app.root_path, 'static'),
            'favicon.ico',
            mimetype='image/vnd.microsoft.icon'
        )
    
    @app.route('/download/<filename>')
    def download_file(filename):
        return send_from_directory(
            app.config['PROCESSED_FOLDER'],
            filename,
            as_attachment=True
        )

    @app.route('/api/save_to_path', methods=['POST'])
    def save_to_path_route():
        """Copy a processed file to a user-specified server path."""
        try:
            data = request.get_json() or {}
            filename = data.get('filename', '')
            save_path = (data.get('save_path') or '').strip()

            if not filename or not save_path:
                return jsonify({'error': 'filename and save_path are required'}), 400

            # Sanitize filename and locate source in processed folder
            safe_filename = secure_filename(os.path.basename(filename))
            src_path = os.path.join(app.config['PROCESSED_FOLDER'], safe_filename)

            if not os.path.exists(src_path) or not os.path.isfile(src_path):
                return jsonify({'error': 'Source file not found'}), 404

            # Resolve destination file path: decide directory vs file path robustly
            # Expand env vars and ~ for user convenience
            expanded = os.path.expandvars(os.path.expanduser(save_path))
            target_path = os.path.abspath(expanded)

            def resolve_dest_path(target_path: str, filename: str) -> str:
                # Existing directory -> save inside it
                if os.path.isdir(target_path):
                    return os.path.join(target_path, filename)
                # Path not existing yet: infer intent
                if not os.path.exists(target_path):
                    # Trailing separator indicates directory intent
                    if save_path.endswith(os.sep):
                        return os.path.join(target_path, filename)
                    base = os.path.basename(target_path)
                    name, ext = os.path.splitext(base)
                    # If there is an extension, assume user provided a file path
                    if ext:
                        return target_path
                    # Otherwise treat as a directory path
                    return os.path.join(target_path, filename)
                # Existing path and not a directory -> it's a file path
                return target_path

            dest_path = resolve_dest_path(target_path, safe_filename)

            # Ensure parent directory exists
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            # Avoid overwriting existing files by uniquifying only the file name
            def unique_file_path(path: str) -> str:
                directory, base = os.path.dirname(path), os.path.basename(path)
                name, ext = os.path.splitext(base)
                candidate = path
                i = 1
                while os.path.exists(candidate):
                    candidate = os.path.join(directory, f"{name} ({i}){ext}")
                    i += 1
                return candidate

            final_dest = unique_file_path(dest_path)
            shutil.copy2(src_path, final_dest)

            return jsonify({'status': 'success', 'saved_to': final_dest})
        except Exception as e:
            logger.error(f"Error saving to path: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500

    # Add a new route for task management
    @app.route('/api/tasks')
    def list_tasks():
        """List recent tasks."""
        db = next(get_db())
        task_service = TaskService(db)
        
        try:
            tasks = task_service.get_recent_tasks(limit=20)
            return jsonify([task.to_dict() for task in tasks])
        finally:
            db.close()
    
    @app.route('/api/tasks/<task_id>', methods=['DELETE'])
    def delete_task(task_id):
        """Delete a task and its files."""
        db = next(get_db())
        task_service = TaskService(db)
        try:
            # This is a placeholder. Actual deletion logic needs to be implemented.
            # For now, just return the task info before deletion.
            task = task_service.get_task(task_id)
            if not task:
                return jsonify({'error': 'Task not found'}), 404

            # Example of what would be returned. 
            # Deletion logic would go here.
            return jsonify({"status": "success", "message": f"Task {task_id} marked for deletion."})
        except Exception as e:
            logger.error(f"Error deleting task {task_id}: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
        finally:
            db.close()
    

    @app.route('/api/cleanup', methods=['POST'])
    def cleanup_storage():
        """Clean up storage based on request parameters."""
        data = request.get_json() or {}
        cleanup_type = data.get('type', 'all')
        messages = []

        try:
            if cleanup_type in ['all', 'uploads']:
                messages.append(cleanup.clear_uploads(app.config['UPLOAD_FOLDER']))
            if cleanup_type in ['all', 'processed']:
                messages.append(cleanup.clear_processed(app.config['PROCESSED_FOLDER']))
            if cleanup_type in ['all', 'database']:
                messages.append(cleanup.clear_database())
            
            if not messages:
                return jsonify({'error': 'Invalid cleanup type'}), 400

            return jsonify({'message': 'Cleanup completed successfully', 'details': messages})
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500

# Celery progress monitoring via WebSocket
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info('Client connected to WebSocket')

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info('Client disconnected from WebSocket')

# For development
app = create_app()

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
