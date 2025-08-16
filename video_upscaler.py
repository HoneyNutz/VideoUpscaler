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
    parser.add_argument('input', help='Path to the input video file')
    parser.add_argument('-o', '--output', help='Path to save the output video file', default=None)
    parser.add_argument('--model', default='RealESRGAN_x4plus',
                      help='Model to use for upscaling (default: RealESRGAN_x4plus)')
    parser.add_argument('--scale', type=int, default=4, choices=[2, 4],
                      help='Upscaling factor (default: 4)')
    parser.add_argument('--fps', type=float, default=None,
                      help='FPS for the output video (default: same as input)')
    parser.add_argument('--tile', type=int, default=0,
                      help='Tile size, 0 for no tiling (default: 0)')
    parser.add_argument('--fp32', action='store_true',
                      help='Use fp32 precision (default: fp16)')
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output is None:
        input_path = Path(args.input)
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
            progress_callback=progress_callback
        )
        
        # Process the video
        print(f"Starting video upscaling with {args.model} (scale: x{args.scale})")
        processor.process_video(args.input, args.output)
        
        progress_bar.close()
        print(f"\n✅ Video successfully upscaled to: {args.output}")
        
    except KeyboardInterrupt:
        print("\n❌ Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        if os.path.exists(args.output):
            os.remove(args.output)
        return 1
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
