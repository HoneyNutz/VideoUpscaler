from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename
import os
import logging
from pathlib import Path
import uuid
import traceback

from .models import create_tables, get_db, ProcessingTask, TaskStatus
from .database import TaskService
from .task_queue import start_task_queue, enqueue_task

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
                
                if scale not in [2, 4]:
                    raise ValueError('Scale must be either 2 or 4')
                    
                # Validate model exists in models directory
                model_path = os.path.join('models', f'{model}.pth')
                if not os.path.exists(model_path):
                    raise ValueError(f'Model {model} not found. Please download it first.')
                
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
            
            # Extract output filename from path
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
            success = task_service.delete_task(task_id)
            if success:
                return jsonify({'message': 'Task deleted successfully'})
            else:
                return jsonify({'error': 'Task not found'}), 404
        finally:
            db.close()

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
