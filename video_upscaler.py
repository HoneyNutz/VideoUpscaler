#!/usr/bin/env python3
"""
Video Upscaler - A command-line interface for upscaling videos using Real-ESRGAN.

This module provides a CLI for the VideoUpscaler class, which handles the actual
video processing. For the web interface, see run_web.py.
"""
import os
import argparse
from pathlib import Path
from tqdm import tqdm

from app.video_processor import VideoProcessor

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Upscale videos using Real-ESRGAN')
    parser.add_argument('input', nargs='?', help='Path to the input video file or directory (for batch mode)')
    parser.add_argument('-o', '--output', help='Path to save the output video file', default=None)
    parser.add_argument('--model', default='RealESRGAN_x4plus',
                      choices=['RealESRGAN_x4plus', 'RealESRGAN_x4plus_anime_6B', 'RealESRNet_x4plus', 
                              'RealESRGAN_x2plus', 'realesr-animevideov3', 'realesr-general-x4v3'],
                      help='Model to use for upscaling (default: RealESRGAN_x4plus)')
    parser.add_argument('--scale', type=int, default=4, choices=[2, 4],
                      help='Upscaling factor (default: 4)')
    parser.add_argument('--fps', type=float, default=None,
                      help='FPS for the output video (default: same as input)')
    parser.add_argument('--tile', type=int, default=0,
                      help='Tile size, 0 for no tiling (default: 0)')
    parser.add_argument('--fp32', action='store_true',
                      help='Use fp32 precision (default: fp16)')
    parser.add_argument('--list-models', action='store_true', 
                        help='List all available models with descriptions')
    parser.add_argument('--test-3gp', nargs='?', const='test', metavar='INPUT',
                        help='Test 3GP file support (optionally specify input file)')
    parser.add_argument('--cleanup', action='store_true',
                        help='Clean up uploads, processed files, and database')
    parser.add_argument('--workers', type=int, default=1,
                      help='Number of parallel workers for batch mode (default: 1)')
    parser.add_argument('--extensions', nargs='+', 
                      default=['.mp4', '.avi', '.mov', '.mkv', '.3gp', '.3g2'],
                      help='File extensions to process in batch mode')
    parser.add_argument('--batch', action='store_true',
                      help='Process all video files in the input directory')
    
    args = parser.parse_args()
    
    # Handle list models mode
    if args.list_models:
        models = VideoProcessor.get_available_models()
        print("\n📋 Available Real-ESRGAN Models:")
        print("=" * 50)
        for model_name, info in models.items():
            print(f"\n🎯 {model_name}")
            print(f"   Description: {info['description']}")
            print(f"   Scale: {info['scale']}x")
            print(f"   Size: {info['size']}")
            print(f"   Best for: {info['best_for']}")
        print("\n💡 Usage: --model <model_name>")
        print("📚 Download models from: https://github.com/xinntao/Real-ESRGAN/releases")
        return 0
    
    # Handle test mode
    if args.test_3gp:
        VideoProcessor.test_3gp_support()
        return 0
    
    # Handle cleanup mode
    if args.cleanup:
        import subprocess
        import sys
        try:
            result = subprocess.run([sys.executable, 'cleanup.py', '--all', '--confirm'], 
                                  capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
            return result.returncode
        except Exception as e:
            print(f"❌ Error running cleanup: {e}")
            return 1
    
    # Validate input for non-test modes
    if not args.test_3gp and not args.list_models and not args.cleanup and not args.input:
        parser.error("Input file or directory is required unless using --test-3gp, --list-models, or --cleanup")
    
    # Handle batch mode
    if args.batch:
        return handle_batch_mode(args)
    
    # Set default output path if not provided (single file mode)
    if args.output is None:
        input_path = Path(args.input)
        # Convert 3GP to MP4 by default
        if input_path.suffix.lower() in ['.3gp', '.3g2']:
            output_path = input_path.parent / f"{input_path.stem}_x{args.scale}.mp4"
        else:
            output_path = input_path.parent / f"{input_path.stem}_x{args.scale}{input_path.suffix}"
        args.output = str(output_path)
    
    # Create the models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Check if the model exists
    model_path = os.path.join('models', f'{args.model}.pth')
    if not os.path.exists(model_path):
        print(f"Model {args.model} not found. Please download it first.")
        print(f"You can download it from: https://github.com/xinntao/Real-ESRGAN/releases")
        print(f"and place it in the 'models' directory.")
        return
    
    try:
        # Initialize progress callback
        progress_bar = tqdm(total=100, desc="Processing video")
        last_progress = 0
        
        def progress_callback(progress, status):
            nonlocal last_progress
            progress_delta = progress - last_progress
            if progress_delta > 0:
                progress_bar.update(progress_delta)
                last_progress = progress
            progress_bar.set_description(f"Status: {status}")
        
        # Initialize the video processor
        processor = VideoProcessor(
            model_name=args.model,
            scale=args.scale,
            tile=args.tile,
            fp32=args.fp32,
            progress_callback=progress_callback,
            batch_mode=False
        )
        
        # Process the video
        print(f"Starting video upscaling with {args.model} (scale: x{args.scale})")
        processor.process_video(args.input, args.output)
        
        progress_bar.close()
        print(f"\n✅ Video successfully upscaled to: {args.output}")
        
        # Show 3GP conversion message
        input_path = Path(args.input)
        if input_path.suffix.lower() in ['.3gp', '.3g2'] and args.output.lower().endswith('.mp4'):
            print(f"📱➡️🎬 3GP file converted to MP4 format")
        
    except KeyboardInterrupt:
        print("\n❌ Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        if os.path.exists(args.output):
            os.remove(args.output)
        return 1
    
    return 0

def handle_batch_mode(args):
    """Handle batch processing mode."""
    input_dir = Path(args.input)
    
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"❌ Input directory does not exist or is not a directory: {input_dir}")
        return 1
    
    # Find video files
    video_files = VideoProcessor.find_video_files(input_dir, args.extensions)
    
    if not video_files:
        print(f"❌ No video files found in {input_dir} with extensions {args.extensions}")
        return 1
    
    print(f"📁 Found {len(video_files)} video files to process")
    
    # Filter for 3GP files if any
    gp3_files = [f for f in video_files if f.suffix.lower() in ['.3gp', '.3g2']]
    if gp3_files:
        print(f"📱 Including {len(gp3_files)} 3GP files (will be converted to MP4)")
    
    try:
        # Initialize processor for batch mode
        processor = VideoProcessor(
            model_name=args.model,
            scale=args.scale,
            tile=args.tile,
            fp32=args.fp32,
            batch_mode=True
        )
        
        # Process batch
        results = processor.process_batch(
            input_paths=video_files,
            output_dir=args.output,
            max_workers=args.workers
        )
        
        # Show summary
        if results['failed'] > 0:
            print(f"\n⚠️  Some files failed to process. Check the output above for details.")
            return 1
        else:
            print(f"\n🎉 All files processed successfully!")
            return 0
            
    except KeyboardInterrupt:
        print("\n⏹️  Batch processing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Batch processing error: {str(e)}")
        return 1

if __name__ == '__main__':
    import sys
    sys.exit(main())
