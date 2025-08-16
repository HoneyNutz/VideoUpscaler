import os
import requests
import argparse
from pathlib import Path
from tqdm import tqdm

def download_file(url, destination):
    """Download a file with progress bar."""
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Check if the file already exists
    if os.path.exists(destination):
        print(f"File {destination} already exists. Skipping download.")
        return
    
    print(f"Downloading {url} to {destination}")
    
    # Stream the download to handle large files
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get the total file size
    total_size = int(response.headers.get('content-length', 0))
    
    # Download the file with progress bar
    with open(destination, 'wb') as f, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

MODELS = {
    'RealESRGAN_x4plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    'realesr-animevideov3': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth',
}

def main():
    parser = argparse.ArgumentParser(description='Download Real-ESRGAN models')
    parser.add_argument('--model', choices=['all', 'RealESRGAN_x4plus', 'realesr-animevideov3'], 
                       default='all', help='Model to download (default: all)')
    parser.add_argument('--output-dir', default='models', help='Directory to save models (default: models/)')
    
    args = parser.parse_args()
    
    # Determine which models to download
    models_to_download = [args.model] if args.model != 'all' else MODELS.keys()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download each model
    for model_name in models_to_download:
        if model_name in MODELS:
            url = MODELS[model_name]
            output_path = os.path.join(args.output_dir, f'{model_name}.pth')
            try:
                download_file(url, output_path)
                print(f"Successfully downloaded {model_name}")
            except Exception as e:
                print(f"Error downloading {model_name}: {str(e)}")
        else:
            print(f"Unknown model: {model_name}")
    
    print("\nModel download complete!")
    print("You can now use the video upscaler with the downloaded models.")

if __name__ == '__main__':
    main()
