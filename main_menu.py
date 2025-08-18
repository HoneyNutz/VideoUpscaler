#!/usr/bin/env python3
"""
Interactive main menu for Video Upscaler.
Provides a user-friendly interface for all video upscaling operations.
"""
import os
import sys
import subprocess
from pathlib import Path

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def show_banner():
    """Display the application banner."""
    print("=" * 60)
    print("🎬 VIDEO UPSCALER WITH REAL-ESRGAN")
    print("=" * 60)
    print("Professional AI video upscaling with 6 models")
    print("Supports: MP4, AVI, MOV, MKV, 3GP, 3G2")
    print("=" * 60)

def show_menu():
    """Display the main menu options."""
    print("\n📋 MAIN MENU:")
    print("1. 🎯 Upscale Single Video")
    print("2. 📦 Batch Process Videos")
    print("3. 🌐 Start Web Interface")
    print("4. 📋 List Available Models")
    print("5. 📱 Test 3GP Support")
    print("6. 📥 Download Models")
    print("7. 🧹 Cleanup Storage")
    print("8. 📊 Show Storage Info")
    print("9. ❓ Help & Documentation")
    print("0. 🚪 Exit")
    print("-" * 40)

def run_command(cmd, description=""):
    """Run a command and handle errors."""
    if description:
        print(f"\n🔄 {description}...")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(f"❌ Error: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False

def get_file_input(prompt="Enter file path: "):
    """Get and validate file input from user."""
    while True:
        file_path = input(prompt).strip()
        if not file_path:
            print("❌ Please enter a file path")
            continue
        
        path = Path(file_path)
        if not path.exists():
            print(f"❌ File not found: {file_path}")
            continue
        
        if not path.is_file():
            print(f"❌ Not a file: {file_path}")
            continue
            
        return str(path)

def get_directory_input(prompt="Enter directory path: "):
    """Get and validate directory input from user."""
    while True:
        dir_path = input(prompt).strip()
        if not dir_path:
            print("❌ Please enter a directory path")
            continue
        
        path = Path(dir_path)
        if not path.exists():
            print(f"❌ Directory not found: {dir_path}")
            continue
        
        if not path.is_dir():
            print(f"❌ Not a directory: {dir_path}")
            continue
            
        return str(path)

def single_video_upscale():
    """Handle single video upscaling."""
    print("\n🎯 SINGLE VIDEO UPSCALING")
    print("-" * 30)
    
    # Get input file
    input_file = get_file_input("Enter video file path: ")
    
    # Get model choice
    print("\nSelect model (or press Enter for default):")
    run_command("uv run python video_upscaler.py --list-models")
    model = input("\nModel name (default: RealESRGAN_x4plus): ").strip()
    if not model:
        model = "RealESRGAN_x4plus"
    
    # Get scale
    while True:
        scale = input("Scale factor (2 or 4, default: 4): ").strip()
        if not scale:
            scale = "4"
            break
        if scale in ["2", "4"]:
            break
        print("❌ Scale must be 2 or 4")
    
    # Build command
    cmd = f'uv run python video_upscaler.py "{input_file}" --model {model} --scale {scale}'
    
    # Run upscaling
    success = run_command(cmd, f"Upscaling {Path(input_file).name}")
    
    if success:
        print("✅ Video upscaling completed!")
    else:
        print("❌ Video upscaling failed!")

def batch_process():
    """Handle batch processing."""
    print("\n📦 BATCH PROCESSING")
    print("-" * 20)
    
    # Get input directory
    input_dir = get_directory_input("Enter directory with videos: ")
    
    # Get model choice
    print("\nSelect model (or press Enter for default):")
    run_command("uv run python video_upscaler.py --list-models")
    model = input("\nModel name (default: RealESRGAN_x4plus): ").strip()
    if not model:
        model = "RealESRGAN_x4plus"
    
    # Get workers
    workers = input("Number of parallel workers (default: 1): ").strip()
    if not workers:
        workers = "1"
    
    # Build command
    cmd = f'uv run python video_upscaler.py "{input_dir}" --batch --model {model} --workers {workers}'
    
    # Run batch processing
    success = run_command(cmd, f"Batch processing videos in {Path(input_dir).name}")
    
    if success:
        print("✅ Batch processing completed!")
    else:
        print("❌ Batch processing failed!")

def start_web_interface():
    """Start the web interface."""
    print("\n🌐 STARTING WEB INTERFACE")
    print("-" * 30)
    print("Web interface will start at: http://localhost:5001")
    print("Press Ctrl+C to stop the server")
    input("\nPress Enter to continue...")
    
    try:
        subprocess.run("uv run python run_web.py", shell=True)
    except KeyboardInterrupt:
        print("\n🛑 Web server stopped")

def download_models():
    """Handle model downloads."""
    print("\n📥 DOWNLOAD MODELS")
    print("-" * 20)
    
    print("Available models:")
    run_command("uv run python video_upscaler.py --list-models")
    
    print("\nOptions:")
    print("1. Download all models")
    print("2. Download specific model")
    
    choice = input("\nChoice (1-2): ").strip()
    
    if choice == "1":
        cmd = "uv run python download_models.py"
        run_command(cmd, "Downloading all models")
    elif choice == "2":
        model = input("Enter model name: ").strip()
        if model:
            cmd = f"uv run python download_models.py --model {model}"
            run_command(cmd, f"Downloading {model}")
    else:
        print("❌ Invalid choice")

def cleanup_storage():
    """Handle storage cleanup."""
    print("\n🧹 CLEANUP STORAGE")
    print("-" * 20)
    
    # Show current usage first
    run_command("uv run python cleanup.py --info")
    
    print("\nCleanup options:")
    print("1. Clean everything (uploads + processed + database)")
    print("2. Clean uploads only")
    print("3. Clean processed files only") 
    print("4. Clean database only")
    print("5. Cancel")
    
    choice = input("\nChoice (1-5): ").strip()
    
    commands = {
        "1": "uv run python cleanup.py --all",
        "2": "uv run python cleanup.py --uploads", 
        "3": "uv run python cleanup.py --processed",
        "4": "uv run python cleanup.py --database"
    }
    
    if choice in commands:
        success = run_command(commands[choice], "Cleaning up storage")
        if success:
            print("✅ Cleanup completed!")
    elif choice == "5":
        print("Cleanup cancelled")
    else:
        print("❌ Invalid choice")

def show_help():
    """Show help and documentation."""
    print("\n❓ HELP & DOCUMENTATION")
    print("-" * 25)
    print("📚 README: Check README.md for complete documentation")
    print("🌐 Web Interface: Most user-friendly option")
    print("⚡ CLI: Best for batch processing and automation")
    print("📱 3GP Support: Automatic mobile video optimizations")
    print("🎯 Models: 6 specialized AI models for different content")
    print("\n💡 Quick Tips:")
    print("• Use RealESRGAN_x4plus for 3GP/mobile videos")
    print("• Use realesr-animevideov3 for anime content")
    print("• Use realesr-general-x4v3 for balanced quality")
    print("• Web interface supports drag & drop")
    print("• Cleanup regularly to manage disk space")

def wait_for_enter():
    """Wait for user to press Enter."""
    input("\nPress Enter to continue...")

def main():
    """Main menu loop."""
    while True:
        clear_screen()
        show_banner()
        show_menu()
        
        choice = input("Enter your choice (0-9): ").strip()
        
        if choice == "1":
            single_video_upscale()
            wait_for_enter()
        elif choice == "2":
            batch_process()
            wait_for_enter()
        elif choice == "3":
            start_web_interface()
            wait_for_enter()
        elif choice == "4":
            print("\n📋 AVAILABLE MODELS")
            print("-" * 20)
            run_command("uv run python video_upscaler.py --list-models")
            wait_for_enter()
        elif choice == "5":
            print("\n📱 TESTING 3GP SUPPORT")
            print("-" * 25)
            run_command("uv run python video_upscaler.py --test-3gp", "Testing 3GP support")
            wait_for_enter()
        elif choice == "6":
            download_models()
            wait_for_enter()
        elif choice == "7":
            cleanup_storage()
            wait_for_enter()
        elif choice == "8":
            print("\n📊 STORAGE INFORMATION")
            print("-" * 25)
            run_command("uv run python cleanup.py --info")
            wait_for_enter()
        elif choice == "9":
            show_help()
            wait_for_enter()
        elif choice == "0":
            print("\n👋 Thanks for using Video Upscaler!")
            break
        else:
            print("❌ Invalid choice. Please try again.")
            wait_for_enter()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
