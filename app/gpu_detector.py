"""
Runtime GPU Detection Module
Dynamically detects and configures GPU acceleration at runtime.
"""

import torch
import platform
import subprocess
import logging

logger = logging.getLogger(__name__)


class GPUDetector:
    """Detects available GPU acceleration and configures PyTorch accordingly."""
    
    def __init__(self):
        self.gpu_type = None
        self.device = None
        self._detect_gpu()
    
    def _detect_gpu(self):
        """Detect the best available GPU acceleration."""
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        # Check for Apple Silicon MPS
        if system == "darwin" and machine == "arm64":
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.gpu_type = "mps"
                self.device = torch.device('mps')
                logger.info("✅ Apple Silicon MPS acceleration detected")
                return
        
        # Check for NVIDIA CUDA
        if torch.cuda.is_available():
            self.gpu_type = "cuda"
            self.device = torch.device('cuda')
            cuda_version = torch.version.cuda
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown"
            logger.info(f"✅ NVIDIA CUDA acceleration detected: {gpu_name} (CUDA {cuda_version})")
            return
        
        # Check for AMD ROCm (if available)
        try:
            if hasattr(torch.backends, 'hip') and torch.backends.hip.is_available():
                self.gpu_type = "rocm"
                self.device = torch.device('hip')
                logger.info("✅ AMD ROCm acceleration detected")
                return
        except AttributeError:
            pass
        
        # Fallback to CPU
        self.gpu_type = "cpu"
        self.device = torch.device('cpu')
        logger.info("ℹ️  Using CPU acceleration (no GPU detected)")
    
    def get_device(self):
        """Get the optimal PyTorch device."""
        return self.device
    
    def get_gpu_type(self):
        """Get the GPU type string."""
        return self.gpu_type
    
    def optimize_for_gpu(self):
        """Apply GPU-specific optimizations."""
        if self.gpu_type == "mps":
            # MPS optimizations
            if hasattr(torch.backends.mps, 'enable_fallback'):
                torch.backends.mps.enable_fallback(True)
            if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            logger.info("Applied MPS optimizations")
            
        elif self.gpu_type == "cuda":
            # CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Applied CUDA optimizations")
            
        elif self.gpu_type == "rocm":
            # ROCm optimizations
            logger.info("Applied ROCm optimizations")
    
    def get_optimal_tile_size(self):
        """Get optimal tile size based on GPU type and memory."""
        if self.gpu_type == "mps":
            # Apple Silicon - conservative tile size
            return 512
        elif self.gpu_type == "cuda":
            # NVIDIA - check VRAM
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if vram_gb >= 8:
                    return 1024
                elif vram_gb >= 4:
                    return 512
                else:
                    return 256
            return 512
        elif self.gpu_type == "rocm":
            # AMD - moderate tile size
            return 512
        else:
            # CPU - smaller tile size
            return 256
    
    def should_use_half_precision(self):
        """Determine if half precision should be used."""
        if self.gpu_type in ["mps", "cuda"]:
            return True
        return False
    
    def get_device_info(self):
        """Get detailed device information."""
        info = {
            "gpu_type": self.gpu_type,
            "device": str(self.device),
            "tile_size": self.get_optimal_tile_size(),
            "half_precision": self.should_use_half_precision()
        }
        
        if self.gpu_type == "cuda" and torch.cuda.is_available():
            info.update({
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
            })
        elif self.gpu_type == "mps":
            info.update({
                "pytorch_version": torch.__version__,
                "system": platform.system(),
                "machine": platform.machine(),
                "gpu_memory_gb": 8.0  # Conservative estimate for Apple Silicon
            })
        
        # Ensure gpu_memory_gb is always present
        if "gpu_memory_gb" not in info:
            info["gpu_memory_gb"] = 4.0  # Default fallback for CPU or unknown GPU types
        
        return info


# Global GPU detector instance
gpu_detector = GPUDetector()


def get_optimal_device():
    """Get the optimal PyTorch device for the current system."""
    return gpu_detector.get_device()


def get_gpu_info():
    """Get GPU information for the current system."""
    return gpu_detector.get_device_info()


def optimize_pytorch():
    """Apply PyTorch optimizations for the detected GPU."""
    gpu_detector.optimize_for_gpu()
