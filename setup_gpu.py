#!/usr/bin/env python3
"""
GPU Detection and PyTorch Installation Script
Automatically detects GPU capabilities and installs appropriate PyTorch version.
"""

import subprocess
import sys
import platform
import os
from pathlib import Path


def run_command(cmd, capture_output=True):
    """Run a command and return the result."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return False, "", str(e)


def detect_gpu_type():
    """Detect the type of GPU available on the system."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    print(f"Detecting GPU on {system} ({machine})...")
    
    # Check for Apple Silicon (M1/M2/M3/M4)
    if system == "darwin" and machine == "arm64":
        print("✅ Apple Silicon detected - MPS acceleration available")
        return "mps"
    
    # Check for NVIDIA GPU on any platform
    nvidia_check_commands = [
        "nvidia-smi",
        "nvcc --version",
        "which nvidia-smi"
    ]
    
    for cmd in nvidia_check_commands:
        success, output, _ = run_command(cmd)
        if success and output:
            print("✅ NVIDIA GPU detected - CUDA acceleration available")
            return "cuda"
    
    # Check for AMD GPU on Linux
    if system == "linux":
        amd_check_commands = [
            "rocm-smi",
            "which rocm-smi",
            "ls /opt/rocm"
        ]
        
        for cmd in amd_check_commands:
            success, output, _ = run_command(cmd)
            if success:
                print("✅ AMD GPU detected - ROCm acceleration available")
                return "rocm"
    
    print("ℹ️  No GPU acceleration detected - using CPU")
    return "cpu"


def get_torch_install_command(gpu_type):
    """Get the appropriate PyTorch installation command based on GPU type."""
    base_packages = [
        "numpy>=1.24.0",
        "opencv-python>=4.8.0", 
        "tqdm>=4.65.0",
        "basicsr>=1.4.2",
        "realesrgan>=0.3.0",
        "facexlib>=0.3.0",
        "gfpgan>=1.3.8",
        "ffmpeg-python>=0.2.0",
        "requests>=2.31.0",
        "python-multipart>=0.0.6",
        "python-dotenv>=1.0.0",
        "sqlalchemy>=2.0.0"
    ]
    
    if gpu_type == "mps":
        # Apple Silicon - use compatible PyTorch version
        torch_packages = [
            "torch==2.0.1",
            "torchvision==0.15.2"
        ]
    elif gpu_type == "cuda":
        # NVIDIA GPU - use CUDA-enabled PyTorch
        torch_packages = [
            "torch>=2.0.0+cu118",
            "torchvision>=0.15.0+cu118"
        ]
        # Add CUDA index
        base_packages.append("--index-url https://download.pytorch.org/whl/cu118")
    elif gpu_type == "rocm":
        # AMD GPU - use ROCm-enabled PyTorch
        torch_packages = [
            "torch>=2.0.0+rocm5.4.2",
            "torchvision>=0.15.0+rocm5.4.2"
        ]
        base_packages.append("--index-url https://download.pytorch.org/whl/rocm5.4.2")
    else:
        # CPU only
        torch_packages = [
            "torch>=2.0.0+cpu",
            "torchvision>=0.15.0+cpu"
        ]
        base_packages.append("--index-url https://download.pytorch.org/whl/cpu")
    
    return torch_packages + base_packages


def update_pyproject_toml(gpu_type):
    """Update pyproject.toml with appropriate dependencies."""
    pyproject_path = Path("pyproject.toml")
    
    if not pyproject_path.exists():
        print("❌ pyproject.toml not found")
        return False
    
    # Read current content
    with open(pyproject_path, 'r') as f:
        content = f.read()
    
    # Define GPU-specific dependency sets
    if gpu_type == "mps":
        torch_deps = '''    "torch==2.0.1",
    "torchvision==0.15.2",'''
    elif gpu_type == "cuda":
        torch_deps = '''    "torch>=2.0.0",
    "torchvision>=0.15.0",'''
    elif gpu_type == "rocm":
        torch_deps = '''    "torch>=2.0.0",
    "torchvision>=0.15.0",'''
    else:  # cpu
        torch_deps = '''    "torch>=2.0.0",
    "torchvision>=0.15.0",'''
    
    # Update dependencies section
    base_deps = '''    "numpy>=1.24.0",
    "opencv-python>=4.8.0",
    "tqdm>=4.65.0",
    "basicsr>=1.4.2",
    "realesrgan>=0.3.0",
    "facexlib>=0.3.0",
    "gfpgan>=1.3.8",
    "ffmpeg-python>=0.2.0",
    "requests>=2.31.0",
    "python-multipart>=0.0.6",
    "python-dotenv>=1.0.0",
    "sqlalchemy>=2.0.0",'''
    
    new_deps = f'''dependencies = [
{torch_deps}
{base_deps}
]'''
    
    # Replace dependencies section
    import re
    pattern = r'dependencies = \[.*?\]'
    content = re.sub(pattern, new_deps, content, flags=re.DOTALL)
    
    # Write updated content
    with open(pyproject_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Updated pyproject.toml for {gpu_type.upper()} acceleration")
    return True


def install_dependencies(gpu_type):
    """Install dependencies using uv."""
    print(f"Installing dependencies for {gpu_type.upper()} acceleration...")
    
    # Install PyTorch with appropriate backend
    torch_packages = get_torch_install_command(gpu_type)
    
    if gpu_type == "cuda":
        # Install CUDA PyTorch
        cmd = f"uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
    elif gpu_type == "rocm":
        # Install ROCm PyTorch  
        cmd = f"uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.4.2"
    elif gpu_type == "cpu":
        # Install CPU-only PyTorch
        cmd = f"uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
    else:  # mps
        # Install standard PyTorch (works with MPS)
        cmd = "uv pip install torch==2.0.1 torchvision==0.15.2"
    
    success, output, error = run_command(cmd)
    if not success:
        print(f"❌ Failed to install PyTorch: {error}")
        return False
    
    # Install other dependencies
    other_deps = [
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "tqdm>=4.65.0", 
        "basicsr>=1.4.2",
        "realesrgan>=0.3.0",
        "facexlib>=0.3.0",
        "gfpgan>=1.3.8",
        "ffmpeg-python>=0.2.0",
        "requests>=2.31.0",
        "python-multipart>=0.0.6",
        "python-dotenv>=1.0.0",
        "sqlalchemy>=2.0.0",
        "flask>=2.0.0",
        "flask-socketio>=5.0.0"
    ]
    
    for dep in other_deps:
        cmd = f"uv pip install '{dep}'"
        success, _, error = run_command(cmd)
        if not success:
            print(f"⚠️  Warning: Failed to install {dep}: {error}")
    
    print("✅ Dependencies installed successfully")
    return True


def verify_installation(gpu_type):
    """Verify that PyTorch is working with the detected GPU."""
    print("Verifying installation...")
    
    test_script = f'''
import torch
print(f"PyTorch version: {{torch.__version__}}")
print(f"CUDA available: {{torch.cuda.is_available()}}")

if hasattr(torch.backends, "mps"):
    print(f"MPS available: {{torch.backends.mps.is_available()}}")
else:
    print("MPS not available (PyTorch < 1.12)")

# Test tensor creation
device = "cpu"
if "{gpu_type}" == "cuda" and torch.cuda.is_available():
    device = "cuda"
elif "{gpu_type}" == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

print(f"Using device: {{device}}")
x = torch.randn(3, 3).to(device)
print(f"Test tensor created successfully on {{device}}")

# Test Real-ESRGAN import
try:
    import realesrgan
    print("✅ Real-ESRGAN imported successfully")
except ImportError as e:
    print(f"❌ Real-ESRGAN import failed: {{e}}")
'''
    
    success, output, error = run_command(f'python -c "{test_script}"')
    if success:
        print("✅ Installation verification passed")
        print(output)
        return True
    else:
        print(f"❌ Installation verification failed: {error}")
        return False


def main():
    """Main setup function."""
    print("🚀 VideoUpscaler GPU Setup")
    print("=" * 50)
    
    # Detect GPU type
    gpu_type = detect_gpu_type()
    
    # Update pyproject.toml
    if not update_pyproject_toml(gpu_type):
        print("❌ Failed to update pyproject.toml")
        return 1
    
    # Install dependencies
    if not install_dependencies(gpu_type):
        print("❌ Failed to install dependencies")
        return 1
    
    # Verify installation
    if not verify_installation(gpu_type):
        print("❌ Installation verification failed")
        return 1
    
    print("\n🎉 Setup completed successfully!")
    print(f"GPU acceleration: {gpu_type.upper()}")
    print("You can now run: python run_web.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
