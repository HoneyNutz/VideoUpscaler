#!/usr/bin/env python3
"""
Cross-Platform VideoUpscaler Installation Script
Automatically detects system capabilities and installs appropriate dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_setup():
    """Run the GPU setup script."""
    setup_script = Path(__file__).parent / "setup_gpu.py"
    
    if not setup_script.exists():
        print("❌ setup_gpu.py not found")
        return False
    
    try:
        result = subprocess.run([sys.executable, str(setup_script)], check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Setup failed: {e}")
        return False


def main():
    """Main installation function."""
    print("🚀 VideoUpscaler Cross-Platform Installer")
    print("=" * 50)
    print("This will automatically detect your GPU and install optimal dependencies.")
    print()
    
    # Check if uv is available
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ UV package manager not found. Please install UV first:")
        print("   curl -sSf https://astral.sh/uv/install.sh | sh")
        return 1
    
    # Run GPU setup
    if not run_setup():
        print("❌ Installation failed")
        return 1
    
    print("\n🎉 Installation completed successfully!")
    print("Run: python run_web.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
