# Video Upscaler with Real-ESRGAN

A self-contained Python application for upscaling videos using Real-ESRGAN, featuring both a web interface and command-line interface with persistent task management and real-time progress updates.

![Screenshot of the web interface](https://via.placeholder.com/800x500/1f2937/ffffff?text=Video+Upscaler+Web+Interface)

## Features

- **🌐 Web Interface**: Modern, dark-themed UI with drag-and-drop support
- **⚡ CLI**: Command-line interface for batch processing and automation
- **🎯 6 AI Models**: Professional Real-ESRGAN models for different content types
- **📱 3GP Support**: Specialized mobile video upscaling with optimizations
- **📊 Real-time Progress**: Track video processing progress in real-time
- **🔧 Self-Contained**: No external dependencies like Redis or Celery required
- **💾 Persistent Tasks**: SQLite-based task queue with automatic recovery
- **🚀 GPU Acceleration**: Optimized for CUDA, MPS (Apple Silicon), and CPU
- **🌍 Cross-platform**: Works on Windows, macOS, and Linux
- **📦 Batch Processing**: Process entire directories of videos efficiently

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

**Supported Models** (6 total):
- `RealESRGAN_x4plus.pth` - General purpose, best for 3GP/photos (67MB)
- `realesr-general-x4v3.pth` - Balanced general purpose, recommended (32MB)
- `RealESRNet_x4plus.pth` - Enhanced detail preservation (67MB)
- `RealESRGAN_x2plus.pth` - 2x upscaling, faster processing (67MB)
- `realesr-animevideov3.pth` - Anime videos with temporal consistency (32MB)
- `RealESRGAN_x4plus_anime_6B.pth` - Compact anime model (17MB)

**Quick Download**:
```bash
# Download all models automatically
uv run python download_models.py

# Or download specific model
uv run python download_models.py --model RealESRGAN_x4plus
```

## Usage

### Web Interface

Start the web server:

```bash
# Option 1: Interactive main menu (recommended for beginners)
uv run python main_menu.py

# Option 2: Direct web interface
uv run python run_web.py
```

This provides a guided interface for all features including single file processing, batch operations, model downloads, and cleanup.

## CLI Mode

### Command Line Interface

Direct command-line usage for automation and power users:

```bash
# Single file upscaling
uv run python video_upscaler.py input.mp4 -o output.mp4

# 3GP mobile video (auto-converts to MP4)
uv run python video_upscaler.py old_phone_video.3gp

# Batch process directory
uv run python video_upscaler.py /path/to/videos --batch
```

### CLI Options

- `input`: Path to input video file or directory (for batch mode)
- `-o, --output`: Output file path (default: adds `_x4` suffix)
- `--model`: AI model to use (6 options, see `--list-models`)
- `--scale`: Upscaling factor (2 or 4, default: 4)
- `--batch`: Process all videos in input directory
- `--extensions`: File types to process (default: .mp4 .avi .mov .mkv .3gp .3g2)
- `--workers`: Parallel workers for batch mode (default: 1)
- `--tile`: Tile size for large videos (0 for auto, default: 0)
- `--fp32`: Use FP32 precision (default: FP16)
- `--test-3gp`: Test 3GP file support
- `--list-models`: Show all available models with descriptions
- `--cleanup`: Clean up uploads and database

### CLI Examples

**Single File Processing:**
```bash
# Basic 4x upscaling
uv run python video_upscaler.py input.mp4 -o output_4k.mp4

# 3GP mobile video (best model)
uv run python video_upscaler.py old_mobile.3gp --model RealESRGAN_x4plus

# Anime content with specialized model
uv run python video_upscaler.py anime.mp4 --model realesr-animevideov3

# Fast 2x upscaling for large files
uv run python video_upscaler.py large_video.mp4 --model RealESRGAN_x2plus --scale 2
```

**Batch Processing:**
```bash
# Process all videos in directory
uv run python video_upscaler.py ./videos --batch

# Process only 3GP files with anime model
uv run python video_upscaler.py ./mobile_archive --batch --extensions .3gp .3g2 --model realesr-animevideov3

# Custom output directory
uv run python video_upscaler.py ./input --batch -o ./upscaled_output
```

**Utility Commands:**
```bash
# Interactive main menu (easiest way to get started)
uv run python main_menu.py

# List all available models
uv run python video_upscaler.py --list-models

# Test 3GP support
uv run python video_upscaler.py --test-3gp

# Clean up uploads and database
uv run python video_upscaler.py --cleanup

# Download models
uv run python download_models.py --model realesr-general-x4v3
```

## Web UI/Flask Interface

### Starting the Web Interface

Launch the modern web interface with real-time progress tracking:

```bash
uv run python run_web.py
```

The web interface will be available at `http://localhost:5001`

### Web UI Features

**Modern Dark Theme Interface:**
- Professional glassmorphism design with dark theme
- Responsive layout optimized for desktop and mobile
- Sidebar navigation with intuitive menu system

**Video Processing:**
- Drag-and-drop file upload with visual feedback
- Support for all video formats: MP4, AVI, MOV, MKV, 3GP, 3G2
- Real-time progress tracking with WebSocket updates
- Cancel processing capability during operation
- Custom save location selector with directory picker

**Model Management:**
- Interactive model selection with descriptions
- 6 AI models optimized for different content types
- Automatic 3GP optimization suggestions
- Model performance indicators

**Storage Management:**
- Real-time storage usage monitoring
- One-click cleanup for uploads, processed files, and database
- Storage breakdown by category (uploads, processed, database)
- Automatic space calculation and reporting

**Menu System:**
- **Video Upscaler**: Main upload and processing interface
- **Batch Processing**: Multiple file processing (coming soon)
- **AI Models**: Model information and management
- **3GP Test**: Mobile video optimization testing
- **Download Models**: Model download interface
- **Storage Management**: Disk space and cleanup tools
- **Help & Documentation**: User guides and tips

### Web UI Advantages

- **User-Friendly**: No command-line knowledge required
- **Visual Feedback**: Real-time progress bars and status updates
- **File Management**: Custom save locations and organized storage
- **Error Handling**: Clear error messages and recovery options
- **Accessibility**: Works on any device with a web browser

## Cleanup & Maintenance

### Storage Management
The cleanup utility helps manage disk space and database records:

```bash
# Show current storage usage
uv run python cleanup.py --info

# Clean everything (uploads, processed files, database)
uv run python cleanup.py --all

# Clean specific components
uv run python cleanup.py --uploads      # Clear uploaded files
uv run python cleanup.py --processed    # Clear processed videos
uv run python cleanup.py --database     # Clear task records

# Skip confirmation prompt
uv run python cleanup.py --all --confirm
```

### Via CLI Integration
```bash
# Quick cleanup through main CLI
uv run python video_upscaler.py --cleanup
```

## Performance Tips

- **GPU Acceleration**: 
  - **CUDA**: Ensure CUDA is properly installed for NVIDIA GPUs
  - **MPS**: Automatic detection and usage on Apple Silicon Macs
  - **CPU**: Automatic fallback if no GPU acceleration available
- **Large Videos**: Use the `--tile` option to process in smaller chunks
- **Memory Usage**: Lower the tile size if you encounter out-of-memory errors
- **Batch Processing**: Use the CLI for batch processing multiple files
- **Storage Management**: Use `cleanup.py` to manage disk space and clear old files
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
