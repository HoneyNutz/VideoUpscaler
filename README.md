# Video Upscaler with Real-ESRGAN

A self-contained Python application for upscaling videos using Real-ESRGAN, featuring both a web interface and command-line interface with persistent task management and real-time progress updates.

![Screenshot of the web interface](https://via.placeholder.com/800x500/1f2937/ffffff?text=Video+Upscaler+Web+Interface)

## Features

- **Web Interface**: Modern, dark-themed UI with drag-and-drop support
- **CLI**: Command-line interface for batch processing and automation
- **Multiple Models**: Support for different Real-ESRGAN models
- **Real-time Progress**: Track video processing progress in real-time
- **Self-Contained**: No external dependencies like Redis or Celery required
- **Persistent Tasks**: SQLite-based task queue with automatic recovery
- **GPU Acceleration**: Optimized for CUDA, MPS (Apple Silicon), and CPU
- **Cross-platform**: Works on Windows, macOS, and Linux

## Prerequisites

- Python 3.8 or higher
- FFmpeg (must be installed and added to system PATH)
- GPU: CUDA-compatible GPU, Apple Silicon (MPS), or CPU fallback

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/HoneyNutz/VideoUpscaler.git
cd VideoUpscaler
```

### 2. Simple Installation with UV (Recommended)

Since we've fixed the dependency issues, you can now install directly with UV:

```bash
# Install UV if you haven't already
curl -sSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with all dependencies (auto-detects GPU)
uv pip install -e ".[web]"

# Download models
python download_models.py

# Start the web interface
python run_web.py
```

This will:
- ✅ Install PyTorch with MPS/CUDA/CPU support automatically
- ✅ Install all AI dependencies with proper version constraints
- ✅ Download Real-ESRGAN models for video upscaling
- ✅ Start the web interface on http://localhost:5001

### 3. Alternative: GPU Detection Script

If you prefer the automatic GPU detection installer:

```bash
# After cloning and creating UV environment
python install.py
```

### 4. Manual Installation (Advanced)

#### Using UV

```bash
# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Run GPU-specific setup
python setup_gpu.py
```

#### Using pip (Legacy)

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e ".[web]"
```

### 3. Download Models

Download the Real-ESRGAN model(s) and place them in the `models` directory:

1. Download models from [Real-ESRGAN releases](https://github.com/xinntao/Real-ESRGAN/releases)
2. Create a `models` directory if it doesn't exist
3. Place the downloaded `.pth` files in the `models` directory

Supported models:
- `RealESRGAN_x4plus.pth` (general purpose, 4x upscaling)
- `realesr-animevideov3.pth` (optimized for anime videos, 4x upscaling)

## Usage

### Web Interface

Start the web server:

```bash
python run_web.py
```

Then open your browser to [http://localhost:5001](http://localhost:5001)

#### Web Interface Features
- Drag and drop video files
- Select upscale factor (2x or 4x)
- Choose between different models
- Real-time progress updates via WebSocket
- Persistent task management
- Download upscaled videos
- Task history and status tracking

### Command Line Interface

Basic usage:
```bash
python video_upscaler.py input.mp4 -o output.mp4
```

#### CLI Options

- `input`: Path to the input video file
- `-o, --output`: Output file path (default: adds `_x4` before the file extension)
- `--model`: Model to use (`RealESRGAN_x4plus` or `realesr-animevideov3`, default: `RealESRGAN_x4plus`)
- `--scale`: Upscaling factor (2 or 4, default: 4)
- `--tile`: Tile size for processing (0 for no tiling, useful for large videos, default: 0)
- `--fp32`: Use FP32 precision (default: FP16)

#### CLI Examples

1. Basic upscaling:
   ```bash
   python video_upscaler.py input.mp4 -o output_4k.mp4
   ```

2. Upscale with anime-optimized model:
   ```bash
   python video_upscaler.py anime.mp4 --model realesr-animevideov3 -o anime_upscaled.mp4
   ```

3. 2x upscaling with tiling for large videos:
   ```bash
   python video_upscaler.py large_video.mp4 --scale 2 --tile 400 -o large_upscaled.mp4
   ```

## Performance Tips

- **GPU Acceleration**: 
  - **CUDA**: Ensure CUDA is properly installed for NVIDIA GPUs
  - **MPS**: Automatic detection and usage on Apple Silicon Macs
  - **CPU**: Automatic fallback if no GPU acceleration available
- **Large Videos**: Use the `--tile` option to process in smaller chunks
- **Memory Usage**: Lower the tile size if you encounter out-of-memory errors
- **Batch Processing**: Use the CLI for batch processing multiple files
- **Concurrent Tasks**: Web interface supports multiple simultaneous video processing
- **Output Quality**: Higher scale factors require more processing time and memory

## Development

### Project Structure

```
video-upscaler/
├── app/                    # Application package
│   ├── __init__.py         # Flask application factory
│   ├── video_processor.py  # Core video processing logic
│   ├── models.py           # SQLAlchemy database models
│   ├── database.py         # Database service layer
│   ├── task_queue.py       # Self-contained task queue system
│   ├── static/             # Static assets (favicon, etc.)
│   └── templates/          # HTML templates
│       ├── base.html       # Base template
│       └── index.html      # Main interface
├── models/                 # Store Real-ESRGAN models here
├── uploads/                # Temporary storage for uploaded files
├── processed/              # Storage for processed videos
├── video_upscaler.py       # CLI interface
├── run_web.py              # Web interface entry point
├── pyproject.toml          # Project configuration and dependencies
├── uv.lock                 # UV dependency lock file (auto-generated)
├── video_upscaler.db       # SQLite database for task persistence
└── README.md               # This file
```

### Architecture

The application uses a **self-contained architecture** with no external dependencies:

- **Database**: SQLite for persistent task storage
- **Task Queue**: Python threading with SQLite polling
- **Real-time Updates**: WebSocket via Flask-SocketIO
- **Processing**: Background worker threads
- **GPU Support**: Automatic device detection (MPS > CUDA > CPU)

### Running Tests

```bash
# Install test dependencies
uv pip install -e ".[dev]"

# Run tests
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - The super-resolution model used
- [BasicSR](https://github.com/xinntao/BasicSR) - The training framework for Real-ESRGAN
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [Socket.IO](https://socket.io/) - Real-time communication
- [Tailwind CSS](https://tailwindcss.com/) - Utility-first CSS framework
