import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
import tempfile
import subprocess
from pathlib import Path
from .gpu_detector import gpu_detector

class VideoProcessor:
    def __init__(self, model_name='RealESRGAN_x4plus', scale=4, tile=0, fp32=False, progress_callback=None, max_memory_gb=4):
        """
        Initialize the VideoProcessor.
        
        Args:
            model_name (str): Name of the Real-ESRGAN model to use
            scale (int): Upscaling factor (2 or 4)
            tile (int): Tile size for processing (0 for no tiling)
            fp32 (bool): Use FP32 precision if True, otherwise FP16
            progress_callback (callable): Callback function for progress updates
            max_memory_gb (float): Maximum memory to use for frame buffering
        """
        self.scale = scale
        self.tile = tile
        self.fp32 = fp32
        self.max_memory_gb = max_memory_gb
        self.progress_callback = progress_callback or (lambda p, s: None)
        self.upsampler = self._initialize_upsampler(model_name)
    
    def _initialize_upsampler(self, model_name):
        """Initialize the Real-ESRGAN upsampler with the specified model."""
        model_path = os.path.join('models', f'{model_name}.pth')
        
        # Use GPU detector for optimal device selection
        device = gpu_detector.get_device()
        gpu_info = gpu_detector.get_device_info()
        
        print(f"Using device: {device} ({gpu_info['gpu_type'].upper()})")
        if 'gpu_name' in gpu_info:
            print(f"GPU: {gpu_info['gpu_name']} ({gpu_info['gpu_memory_gb']}GB)")
        
        # Apply GPU-specific optimizations
        gpu_detector.optimize_for_gpu()
        
        # Set optimal tile size if not specified
        if self.tile == 0:
            self.tile = gpu_detector.get_optimal_tile_size()
            print(f"Using optimal tile size: {self.tile}")
        
        # Determine if half precision should be used
        if not self.fp32:
            self.fp32 = not gpu_detector.should_use_half_precision()
        
        # Determine model parameters based on model name
        if model_name == 'RealESRGAN_x4plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                          num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
        elif model_name == 'realesr-animevideov3':
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, 
                                  num_conv=16, upscale=4, act_type='prelu')
            netscale = 4
        else:
            raise ValueError(f'Model {model_name} is not supported.')
        
        # Initialize the upsampler with GPU device
        # Use FP16 for MPS acceleration unless explicitly disabled
        use_half = not self.fp32 and device.type in ['mps', 'cuda']
        
        upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=model,
            tile=self.tile,
            tile_pad=10,
            pre_pad=0,
            half=use_half,
            device=device
        )
        
        # Warm up the model for better performance
        if device.type in ['mps', 'cuda']:
            dummy_input = torch.randn(1, 3, 64, 64).to(device)
            with torch.no_grad():
                try:
                    _ = upsampler.model(dummy_input)
                    # Synchronize based on device type
                    if device.type == 'mps' and hasattr(torch, 'mps') and hasattr(torch.mps, 'synchronize'):
                        torch.mps.synchronize()
                    elif device.type == 'cuda':
                        torch.cuda.synchronize()
                except Exception:
                    pass  # Ignore warmup errors
        
        return upsampler
    
    def process_frame(self, frame):
        """Process a single frame with the upsampler."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Upscale the frame with MPS optimizations
        with torch.no_grad():  # Disable gradient computation for inference
            output, _ = self.upsampler.enhance(
                frame_rgb,
                outscale=self.scale
            )
            
            # Synchronize GPU operations for better performance
            device = gpu_detector.get_device()
            if device.type == 'mps' and hasattr(torch, 'mps') and hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()
        
        # Convert back to BGR
        return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    
    def process_frames_batch(self, frames, batch_size=4):
        """Process multiple frames in batches for better GPU utilization."""
        processed_frames = []
        device = gpu_detector.get_device()
        
        # Process frames in batches
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            batch_results = []
            
            with torch.no_grad():
                for frame in batch:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Upscale the frame
                    output, _ = self.upsampler.enhance(
                        frame_rgb,
                        outscale=self.scale
                    )
                    
                    # Convert back to BGR and store
                    batch_results.append(cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
                
                # Synchronize once per batch for better performance
                if device.type == 'mps' and hasattr(torch, 'mps') and hasattr(torch.mps, 'synchronize'):
                    torch.mps.synchronize()
                elif device.type == 'cuda':
                    torch.cuda.synchronize()
            
            processed_frames.extend(batch_results)
        
        return processed_frames
    
    def _calculate_memory_usage(self, width, height, frames_count):
        """Calculate memory usage for frame buffer in GB."""
        # Estimate: 3 bytes per pixel (RGB) * 2 (input + output) * frames
        bytes_per_frame = width * height * 3 * 2
        total_bytes = bytes_per_frame * frames_count
        return total_bytes / (1024 ** 3)  # Convert to GB
    
    def _get_optimal_chunk_size(self, width, height, total_frames):
        """Calculate optimal chunk size based on memory constraints."""
        # Start with desired batch size and adjust based on memory
        gpu_info = gpu_detector.get_device_info()
        if gpu_info['gpu_memory_gb'] >= 16:
            base_chunk = 100
        elif gpu_info['gpu_memory_gb'] >= 8:
            base_chunk = 50
        else:
            base_chunk = 25
        
        # Adjust based on memory usage
        while base_chunk > 1:
            memory_usage = self._calculate_memory_usage(width, height, base_chunk)
            if memory_usage <= self.max_memory_gb:
                break
            base_chunk //= 2
        
        return max(1, min(base_chunk, total_frames))
    
    def process_video(self, input_path, output_path):
        """
        Process a video file and save the upscaled version.
        
        Args:
            input_path (str): Path to the input video file
            output_path (str): Path to save the output video
        """
        # Validate input file exists
        if not os.path.exists(input_path):
            raise ValueError(f"Input video file does not exist: {input_path}")
        
        # Open the video file
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}. This may be due to an unsupported codec or corrupted file.")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Validate video properties
        if width <= 0 or height <= 0:
            cap.release()
            raise ValueError(f"Invalid video dimensions: {width}x{height}")
        
        if fps <= 0:
            print(f"Warning: Invalid FPS ({fps}), defaulting to 30 FPS")
            fps = 30.0
        
        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Could not determine frame count: {total_frames}")
        
        print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Calculate output dimensions
        out_width = width * self.scale
        out_height = height * self.scale
        
        # Update progress
        self.progress_callback(0, 'Processing video...')
        
        # Create a temporary directory for frame processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_frames_dir = os.path.join(temp_dir, 'frames')
            os.makedirs(temp_frames_dir, exist_ok=True)
            
            # Process frames with batch optimization
            frame_paths = []
            frame_count = 0
            
            # Clear GPU cache before processing
            device = gpu_detector.get_device()
            if device.type == 'mps' and hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            elif device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Calculate optimal chunk size for memory streaming
            chunk_size = self._get_optimal_chunk_size(width, height, total_frames)
            
            # Determine batch size for GPU processing
            gpu_info = gpu_detector.get_device_info()
            if gpu_info['gpu_memory_gb'] >= 16:
                batch_size = 8
            elif gpu_info['gpu_memory_gb'] >= 8:
                batch_size = 4
            else:
                batch_size = 2
            
            print(f"Processing {total_frames} frames in chunks of {chunk_size}, batch size {batch_size}")
            
            # Process video in chunks to manage memory
            with tqdm(total=total_frames, desc="Processing frames") as pbar:
                for chunk_start in range(0, total_frames, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, total_frames)
                    chunk_frames = []
                    
                    # Read chunk of frames
                    cap.set(cv2.CAP_PROP_POS_FRAMES, chunk_start)
                    for i in range(chunk_start, chunk_end):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        chunk_frames.append(frame)
                    
                    if not chunk_frames:
                        break
                    
                    # Process chunk in batches
                    processed_chunk = self.process_frames_batch(chunk_frames, batch_size)
                    
                    # Save processed frames
                    for i, processed_frame in enumerate(processed_chunk):
                        frame_idx = chunk_start + i
                        frame_path = os.path.join(temp_frames_dir, f'frame_{frame_idx:06d}.png')
                        cv2.imwrite(frame_path, processed_frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                        frame_paths.append(frame_path)
                        
                        progress = int((frame_idx + 1) / total_frames * 90)
                        self.progress_callback(progress, f'Processing frame {frame_idx + 1}/{total_frames}')
                        pbar.update(1)
                    
                    # Clear memory after each chunk
                    del chunk_frames, processed_chunk
                    if device.type == 'mps' and hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                    elif device.type == 'cuda':
                        torch.cuda.empty_cache()
            
            # Release the video capture
            cap.release()
            
            # Update status
            self.progress_callback(95, 'Creating output video...')
            
            # Create video from processed frames using ffmpeg
            self._create_video_from_frames(
                frame_paths, 
                output_path, 
                fps, 
                out_width, 
                out_height
            )
            
            # Update status
            self.progress_callback(100, 'Processing complete!')
    
    def _create_video_from_frames(self, frame_paths, output_path, fps, width, height):
        """Create a video from processed frames using ffmpeg."""
        # Create a temporary file with the list of frames
        list_file = os.path.join(os.path.dirname(frame_paths[0]), 'frame_list.txt')
        with open(list_file, 'w') as f:
            for frame_path in frame_paths:
                f.write(f"file '{frame_path}'\n")
        
        # Use ffmpeg with hardware acceleration based on platform
        device = gpu_detector.get_device()
        gpu_info = gpu_detector.get_device_info()
        
        # Base ffmpeg command
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-f', 'concat',
            '-safe', '0',
            '-r', str(fps),
            '-i', list_file,
        ]
        
        # Add hardware acceleration based on platform
        if gpu_info['gpu_type'] == 'mps':
            # Use VideoToolbox hardware encoder on macOS
            ffmpeg_cmd.extend([
                '-c:v', 'h264_videotoolbox',
                '-b:v', '10M',  # Higher bitrate for quality
                '-pix_fmt', 'yuv420p',
            ])
        elif gpu_info['gpu_type'] == 'cuda':
            # Use NVENC hardware encoder on NVIDIA GPUs
            ffmpeg_cmd.extend([
                '-c:v', 'h264_nvenc',
                '-preset', 'fast',
                '-crf', '20',
                '-pix_fmt', 'yuv420p',
            ])
        else:
            # Fallback to software encoding
            ffmpeg_cmd.extend([
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '20',
                '-pix_fmt', 'yuv420p',
                '-threads', '0',  # Use all available CPU cores
            ])
        
        # Add scaling and output path
        ffmpeg_cmd.extend([
            '-vf', f'scale={width}:{height}:force_original_aspect_ratio=decrease',
            output_path
        ])
        
        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            error_msg = f"Error creating video: {e}"
            if e.stderr:
                error_msg += f"\nffmpeg stderr: {e.stderr}"
            raise RuntimeError(error_msg)
