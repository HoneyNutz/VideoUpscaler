# 3GP Video Upscaling Guide

## Overview

Your video upscaler has comprehensive 3GP support integrated into the main application. 3GP files are perfect candidates for upscaling since they're typically low-resolution mobile videos.

## Quick Start

### Single File Processing

```bash
# Basic upscaling (3GP → MP4)
uv run python video_upscaler.py input.3gp

# With specific settings
uv run python video_upscaler.py input.3gp -o output.mp4 --scale 4 --model RealESRGAN_x4plus
```

### Web Interface

```bash
# Start web server
uv run python run_web.py

# Then upload 3GP files directly through the browser
# The interface explicitly supports .3gp and .3g2 files
```

### Batch Processing

```bash
# Process all video files (including 3GP) in a directory
uv run python video_upscaler.py /path/to/videos --batch

# With custom output directory
uv run python video_upscaler.py /path/to/videos --batch -o /path/to/output

# Using anime model for cartoon content
uv run python video_upscaler.py /path/to/videos --batch --model realesr-animevideov3

# Process only 3GP files
uv run python video_upscaler.py /path/to/videos --batch --extensions .3gp .3g2
```

### Testing 3GP Support

```bash
# Test if your system supports 3GP processing
uv run python video_upscaler.py --test-3gp
```

## 3GP-Specific Optimizations

When processing 3GP files, the upscaler automatically applies these optimizations:

### 🔧 **Automatic Detection**
- Detects 3GP/3G2 files by extension
- Identifies mobile resolution patterns (≤480x640)
- Recognizes low frame rates (≤20fps)

### 📈 **Performance Optimizations**
- **Larger Batch Sizes**: 3GP files are smaller, allowing more frames in memory
- **Optimized Tile Sizes**: Smaller tiles for low-resolution content
- **Memory Efficiency**: Better chunk sizing for mobile video dimensions

### 📱 **Mobile Video Handling**
- Automatic conversion to MP4 format (better compatibility)
- Optimized processing for common mobile resolutions:
  - 144p (176x144)
  - 240p (320x240) 
  - 360p (480x360)

## Typical Workflows

### Scenario 1: Old Mobile Videos
```bash
# Input: old_video.3gp (240x320, 15fps)
# Output: old_video_x4.mp4 (960x1280, 15fps)
uv run python video_upscaler.py old_video.3gp
```

### Scenario 2: Batch Processing Mobile Archive
```bash
# Process entire directory of 3GP files
uv run python batch_3gp_upscaler.py ./mobile_videos/
# Creates: ./mobile_videos/upscaled/ with all processed MP4s
```

### Scenario 3: Web Upload for Non-Technical Users
1. Start web server: `uv run python run_web.py`
2. Open browser to `http://localhost:5000`
3. Drag & drop 3GP files
4. Download upscaled MP4

## Expected Results

### Before Upscaling (3GP)
- **Resolution**: 240x320 pixels
- **Quality**: Compressed for mobile bandwidth
- **Format**: 3GP container
- **File Size**: ~1-5MB

### After Upscaling (MP4)
- **Resolution**: 960x1280 pixels (4x)
- **Quality**: Enhanced with AI upscaling
- **Format**: MP4 (universal compatibility)
- **File Size**: ~20-100MB (depending on content)

## Performance Tips

### 🚀 **For Best Performance**
1. **Use 4x scaling** for most 3GP files (they're typically very low resolution)
2. **RealESRGAN_x4plus model** works best for general mobile content
3. **Process one file at a time** to maximize GPU utilization
4. **Ensure sufficient disk space** (output files are much larger)

### 🎯 **Model Selection**
- **RealESRGAN_x4plus**: Best for photos, real-world mobile videos
- **realesr-animevideov3**: Better for animated/cartoon mobile content

### 💾 **System Requirements**
- **GPU Memory**: 4GB+ recommended for smooth processing
- **Storage**: ~20x input file size for output
- **RAM**: 8GB+ for optimal performance

## Troubleshooting

### Common Issues

**"Could not open video file"**
- Ensure file isn't corrupted
- Try with different 3GP file
- Check file permissions

**"Out of memory" errors**
- Reduce batch size in settings
- Close other GPU-intensive applications
- Use 2x scaling instead of 4x for very large files

**Slow processing**
- Ensure GPU acceleration is working
- Check that models are downloaded
- Monitor GPU utilization

### Getting Help

Test 3GP support directly:
```bash
uv run python video_upscaler.py --test-3gp
```

This will show:
- OpenCV version and backends
- 3GP format support status
- System compatibility information

## Advanced Usage

### Custom Batch Processing Script

```python
from app.video_processor import VideoProcessor

# Create processor with custom settings
processor = VideoProcessor(
    model_name='RealESRGAN_x4plus',
    scale=4,
    tile=256,  # Smaller tiles for mobile video
    fp32=False  # Use FP16 for speed
)

# Process with progress tracking
def progress_callback(progress, status):
    print(f"Progress: {progress}% - {status}")

processor.progress_callback = progress_callback
processor.process_video('input.3gp', 'output.mp4')
```

### Integration with Other Tools

The upscaler can be integrated into larger workflows:
- **Media servers**: Automatically upscale old mobile content
- **Archive processing**: Batch enhance historical mobile videos
- **Content creation**: Improve quality of mobile-shot footage

---

## Model Selection Guide

### **For 3GP Files (Recommended)**
```bash
# Best overall for 3GP mobile videos
uv run python video_upscaler.py old_mobile.3gp --model RealESRGAN_x4plus

# Faster processing, good quality
uv run python video_upscaler.py old_mobile.3gp --model realesr-general-x4v3

# Quick 2x upscaling for large 3GP files
uv run python video_upscaler.py large_mobile.3gp --model RealESRGAN_x2plus --scale 2
```

### **For Different Content Types**
```bash
# Anime/cartoon 3GP files
uv run python video_upscaler.py anime_mobile.3gp --model realesr-animevideov3

# High-detail preservation
uv run python video_upscaler.py detailed_video.3gp --model RealESRNet_x4plus

# Compact anime model (fastest)
uv run python video_upscaler.py cartoon.3gp --model RealESRGAN_x4plus_anime_6B
```

## Summary

Your video upscaler now provides comprehensive 3GP support with:
- ✅ **6 Real-ESRGAN models** for different content types
- ✅ **Automatic format detection** and mobile optimizations
- ✅ **Enhanced web interface** with intelligent model selection
- ✅ **Batch processing** capabilities with model choice
- ✅ **Smart 3GP→MP4 conversion** with quality preservation
- ✅ **Performance optimization** for mobile video content

Perfect for breathing new life into old mobile videos with professional-grade AI upscaling! 📱➡️🎬
