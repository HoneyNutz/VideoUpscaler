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
import concurrent.futures
from threading import Lock

class VideoProcessor:
    @staticmethod
    def get_available_models():
        """
        Get list of available Real-ESRGAN models with descriptions.
        
        Returns:
            dict: Model names and their descriptions
        """
        return {
            'RealESRGAN_x4plus': {
                'description': 'General purpose 4x upscaling (best for photos/real content)',
                'scale': 4,
                'size': 'Large (~67MB)',
                'best_for': 'Photos, real-world content, 3GP videos'
            },
            'RealESRGAN_x4plus_anime_6B': {
                'description': 'Optimized 4x anime upscaling (smaller, faster)',
                'scale': 4,
                'size': 'Small (~17MB)',
                'best_for': 'Anime images, cartoon content'
            },
            'RealESRNet_x4plus': {
                'description': 'Enhanced 4x upscaling with better detail preservation',
                'scale': 4,
                'size': 'Large (~67MB)',
                'best_for': 'High-quality photos, detailed images'
            },
            'RealESRGAN_x2plus': {
                'description': 'General purpose 2x upscaling (faster processing)',
                'scale': 2,
                'size': 'Large (~67MB)',
                'best_for': 'Quick upscaling, large videos'
            },
            'realesr-animevideov3': {
                'description': 'Specialized for anime videos (temporal consistency)',
                'scale': 4,
                'size': 'Medium (~32MB)',
                'best_for': 'Anime videos, animated content'
            },
            'realesr-general-x4v3': {
                'description': 'General purpose with good balance (recommended)',
                'scale': 4,
                'size': 'Medium (~32MB)',
                'best_for': 'Mixed content, general use'
            }
        }
    
    @staticmethod
    def test_3gp_support():
        """
        Test if the system supports 3GP file processing.
        
        Returns:
            dict: Support status and system information
        """
        print("Testing 3GP support in OpenCV...")
        
        info = {
            'opencv_version': cv2.__version__,
            'backends': [cv2.videoio_registry.getBackendName(i) for i in cv2.videoio_registry.getBackends()],
            'ffmpeg_available': False,
            'supported': False
        }
        
        # Check if FFMPEG backend is available (required for 3GP)
        info['ffmpeg_available'] = any('FFMPEG' in backend for backend in info['backends'])
        info['supported'] = info['ffmpeg_available']
        
        print(f"OpenCV version: {info['opencv_version']}")
        print(f"Available backends: {info['backends']}")
        print(f"FFMPEG backend available: {info['ffmpeg_available']}")
        
        if info['supported']:
            print("✅ 3GP files are supported through FFMPEG backend")
        else:
            print("❌ FFMPEG backend not available - 3GP support may be limited")
            
        return info
    
    @staticmethod
    def find_video_files(directory, extensions=None):
        """
        Find all video files in a directory.
        
        Args:
            directory (str): Directory to search
            extensions (list): File extensions to search for
            
        Returns:
            list: List of video file paths
        """
        if extensions is None:
            extensions = ['.mp4', '.avi', '.mov', '.mkv', '.3gp', '.3g2', '.flv', '.webm', '.wmv']
            
        directory = Path(directory)
        files = []
        
        for ext in extensions:
            files.extend(directory.glob(f'**/*{ext}'))
            files.extend(directory.glob(f'**/*{ext.upper()}'))
            
        return sorted(files)
    def __init__(self, model_name='RealESRGAN_x4plus', scale=4, tile=0, fp32=False, progress_callback=None, max_memory_gb=4, batch_mode=False):
        """
        Initialize the VideoProcessor.
        
        Args:
            model_name (str): Name of the Real-ESRGAN model to use
            scale (int): Upscaling factor (2 or 4)
            tile (int): Tile size for processing (0 for no tiling)
            fp32 (bool): Use FP32 precision if True, otherwise FP16
            progress_callback (callable): Callback function for progress updates
            max_memory_gb (float): Maximum memory to use for frame buffering
            batch_mode (bool): Enable batch processing mode
        """
        self.scale = scale
        self.tile = tile
        self.fp32 = fp32
        self.max_memory_gb = max_memory_gb
        self.batch_mode = batch_mode
        self.progress_callback = progress_callback or (lambda p, s: None)
        self.upsampler = self._initialize_upsampler(model_name)
        
        # Batch processing state
        self.progress_lock = Lock() if batch_mode else None
        self.total_files = 0
        self.completed_files = 0
    
    def _get_3gp_optimized_tile_size(self, width, height):
        """
        Get optimized tile size for 3GP/mobile videos.
        
        Args:
            width (int): Video width
            height (int): Video height
            
        Returns:
            int: Optimized tile size
        """
        # For small mobile videos, use smaller tiles for better memory efficiency
        if width <= 320 and height <= 240:
            return min(self._base_tile_size, 128)
        elif width <= 480 and height <= 360:
            return min(self._base_tile_size, 256)
        else:
            return self._base_tile_size
    
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
        
        # Store for later 3GP optimization use
        self._base_tile_size = self.tile
        
        # Determine if half precision should be used
        if not self.fp32:
            self.fp32 = not gpu_detector.should_use_half_precision()
        
        # Determine model parameters based on model name
        if model_name == 'RealESRGAN_x4plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                          num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
        elif model_name == 'RealESRGAN_x4plus_anime_6B':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                          num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
        elif model_name == 'RealESRNet_x4plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                          num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
        elif model_name == 'RealESRGAN_x2plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                          num_block=23, num_grow_ch=32, scale=2)
            netscale = 2
        elif model_name == 'realesr-animevideov3':
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, 
                                  num_conv=16, upscale=4, act_type='prelu')
            netscale = 4
        elif model_name == 'realesr-general-x4v3':
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, 
                                  num_conv=32, upscale=4, act_type='prelu')
            netscale = 4
        else:
            raise ValueError(f'Model {model_name} is not supported. Available models: RealESRGAN_x4plus, RealESRGAN_x4plus_anime_6B, RealESRNet_x4plus, RealESRGAN_x2plus, realesr-animevideov3, realesr-general-x4v3')
        
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
    
    def _detect_3gp_optimizations(self, input_path, width, height, fps):
        """
        Detect if 3GP-specific optimizations should be applied.
        
        Args:
            input_path (str): Path to the input video file
            width (int): Video width
            height (int): Video height
            fps (float): Video frame rate
            
        Returns:
            dict: Optimization settings for 3GP files
        """
        is_3gp = input_path.lower().endswith(('.3gp', '.3g2'))
        is_mobile_res = width <= 480 and height <= 640  # Common mobile resolutions
        is_low_fps = fps <= 20  # Common for older mobile videos
        
        optimizations = {
            'is_3gp': is_3gp,
            'use_smaller_tiles': is_3gp or is_mobile_res,
            'use_higher_batch_size': is_3gp or is_mobile_res,
            'suggested_output_format': 'mp4' if is_3gp else None
        }
        
        if is_3gp:
                print(f"📱 3GP file detected: {width}x{height} @ {fps}fps")
                print("🔧 Applying mobile video optimizations...")
            
        return optimizations
    
    def process_batch(self, input_paths, output_dir=None, max_workers=1, batch_size=4):
        """
{{ ... }}
        Process multiple video files in batch.
        
        Args:
            input_paths (list): List of input video file paths
            output_dir (str): Output directory (default: same as input with _upscaled suffix)
            max_workers (int): Number of parallel workers (1 recommended for GPU)
            
        Returns:
            dict: Processing results summary
        """
        if not input_paths:
            return {'successful': 0, 'failed': 0, 'results': []}
            
        self.total_files = len(input_paths)
        self.completed_files = 0
        
        print(f"📦 Starting batch processing of {self.total_files} files")
        print(f"🎯 Using model: {self.upsampler.model.__class__.__name__} (scale: x{self.scale})")
        
        results = []
        successful = 0
        failed = 0
        
        def process_single_batch_file(input_path):
            """Process a single file in batch mode."""
            input_path = Path(input_path)
            
            # Determine output path
            if output_dir:
                output_dir_path = Path(output_dir)
                output_dir_path.mkdir(exist_ok=True)
            else:
                output_dir_path = input_path.parent / f"{input_path.parent.name}_upscaled"
                output_dir_path.mkdir(exist_ok=True)
            
            # Generate output filename (convert 3GP to MP4)
            if input_path.suffix.lower() in ['.3gp', '.3g2']:
                output_filename = f"{input_path.stem}_x{self.scale}.mp4"
            else:
                output_filename = f"{input_path.stem}_x{self.scale}{input_path.suffix}"
                
            output_path = output_dir_path / output_filename
            
            # Skip if already exists
            if output_path.exists():
                if not self.batch_mode:
                    print(f"⏭️  Skipping {input_path.name} (already exists)")
                with self.progress_lock:
                    self.completed_files += 1
                return {'input': str(input_path), 'output': str(output_path), 'status': 'skipped'}
            
            try:
                # Silent progress for batch mode
                original_callback = self.progress_callback
                if self.batch_mode:
                    self.progress_callback = lambda p, s: None
                
                if not self.batch_mode:
                    print(f"🔄 Processing: {input_path.name}")
                    
                self.process_video(str(input_path), str(output_path))
                
                # Restore original callback
                self.progress_callback = original_callback
                
                with self.progress_lock:
                    self.completed_files += 1
                    if not self.batch_mode:
                        print(f"✅ Completed: {output_filename} ({self.completed_files}/{self.total_files})")
                
                return {'input': str(input_path), 'output': str(output_path), 'status': 'success'}
                
            except Exception as e:
                if not self.batch_mode:
                    print(f"❌ Error processing {input_path.name}: {str(e)}")
                return {'input': str(input_path), 'output': None, 'status': 'failed', 'error': str(e)}
        
        # Process files
        if max_workers == 1:
            # Sequential processing (safer for GPU)
            for input_path in input_paths:
                result = process_single_batch_file(input_path)
                results.append(result)
                if result['status'] == 'success':
                    successful += 1
                elif result['status'] == 'failed':
                    failed += 1
        else:
            # Parallel processing (experimental)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_single_batch_file, path) for path in input_paths]
                
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    results.append(result)
                    if result['status'] == 'success':
                        successful += 1
                    elif result['status'] == 'failed':
                        failed += 1
        
        summary = {
            'successful': successful,
            'failed': failed,
            'skipped': len([r for r in results if r['status'] == 'skipped']),
            'total': self.total_files,
            'results': results
        }
        
        print(f"\n📊 Batch processing complete!")
        print(f"✅ Successful: {successful}")
        print(f"⏭️  Skipped: {summary['skipped']}")
        print(f"❌ Failed: {failed}")
        
        return summary
    
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
        
        # Detect 3GP-specific optimizations
        optimizations = self._detect_3gp_optimizations(input_path, width, height, fps)
        
        # Handle 3GP to MP4 conversion for output path
        if optimizations['is_3gp'] and not output_path.lower().endswith('.mp4'):
            # Convert output extension from .3gp to .mp4
            output_path = os.path.splitext(output_path)[0] + '.mp4'
            print(f"🔄 Converting output format: 3GP → MP4")
        
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
            
            # Determine batch size for GPU processing with 3GP optimizations
            gpu_info = gpu_detector.get_device_info()
            if optimizations['use_higher_batch_size'] and gpu_info['gpu_memory_gb'] >= 4:
                # 3GP files are typically smaller, can use larger batches
                if gpu_info['gpu_memory_gb'] >= 16:
                    batch_size = 12
                elif gpu_info['gpu_memory_gb'] >= 8:
                    batch_size = 8
                else:
                    batch_size = 6
                print("📈 Using larger batch size for mobile video")
            else:
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
            
            # Show 3GP conversion message
            if optimizations['is_3gp'] and output_path.lower().endswith('.mp4'):
                print(f"📱➡️🎬 3GP successfully converted to MP4: {Path(output_path).name}")
    
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
