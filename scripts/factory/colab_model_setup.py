#!/usr/bin/env python3
"""
🚀 Colab Model Setup for MeeTARA Lab
Downloads models once and saves to Google Drive for future Colab sessions
This eliminates the need to download models every time you run Colab
"""

import os
import sys
import time
from pathlib import Path

def setup_colab_environment():
    """Setup Colab environment and mount Google Drive"""
    print("🚀 Setting up Colab environment...")
    
    # Mount Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted successfully")
        return True
    except ImportError:
        print("❌ Not running in Colab - Google Drive not available")
        return False

def check_drive_models():
    """Check if models are already in Google Drive"""
    drive_path = Path("/content/drive/MyDrive/meetara-lab/models/base_models")
    
    if not drive_path.exists():
        print("📁 No models found in Google Drive")
        return False
    
    # Check for downloaded models
    model_dirs = list(drive_path.glob("*"))
    if not model_dirs:
        print("📁 No model directories found in Drive")
        return False
    
    print(f"📁 Found {len(model_dirs)} model directories in Drive:")
    for model_dir in model_dirs:
        if model_dir.is_dir():
            size_gb = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file()) / (1024**3)
            print(f"   - {model_dir.name}: {size_gb:.2f}GB")
    
    return True

def sync_models_from_drive():
    """Sync models from Google Drive to local Colab environment"""
    print("📁 Syncing models from Google Drive...")
    
    drive_path = Path("/content/drive/MyDrive/meetara-lab/models/base_models")
    local_path = Path("/content/meetara-lab/models/base_models")
    
    if not drive_path.exists():
        print("❌ No models found in Google Drive")
        return False
    
    local_path.mkdir(parents=True, exist_ok=True)
    
    # Copy models from Drive to local
    import shutil
    copied_count = 0
    
    for model_dir in drive_path.iterdir():
        if model_dir.is_dir():
            local_model_dir = local_path / model_dir.name
            if local_model_dir.exists():
                shutil.rmtree(local_model_dir)
            shutil.copytree(model_dir, local_model_dir)
            copied_count += 1
            print(f"   ✅ Synced {model_dir.name}")
    
    print(f"📊 Synced {copied_count} models from Drive")
    return copied_count > 0

def download_models_to_drive():
    """Download models and save to Google Drive"""
    print("📥 Downloading models to Google Drive...")
    
    # Import the downloader
    sys.path.append(str(Path.cwd() / "scripts" / "factory"))
    from download_base_models import BaseModelDownloader
    
    # Initialize downloader with Drive path
    drive_path = "/content/drive/MyDrive/meetara-lab/models/base_models"
    downloader = BaseModelDownloader(drive_path)
    
    # Download all models
    results = downloader.download_all_base_models(sync_to_drive=True)
    
    success_count = sum(results.values())
    print(f"📊 Downloaded {success_count}/{len(results)} models")
    
    return success_count == len(results)

def main():
    """Main Colab setup function"""
    print("=" * 60)
    print("🚀 MeeTARA Lab - Colab Model Setup")
    print("=" * 60)
    
    # Setup Colab environment
    if not setup_colab_environment():
        print("❌ Colab setup failed")
        return False
    
    # Check if models are already in Drive
    if check_drive_models():
        print("\n📁 Models found in Google Drive!")
        choice = input("Do you want to sync from Drive to local? (y/n): ").lower()
        if choice == 'y':
            if sync_models_from_drive():
                print("✅ Models synced successfully!")
                print("💡 You can now run training without downloading models again")
                return True
            else:
                print("❌ Failed to sync models")
                return False
        else:
            print("⏭️ Skipping sync")
            return True
    
    # Download models to Drive
    print("\n📥 No models found in Drive - downloading now...")
    print("⏰ This will take 30-60 minutes for all models")
    print("💾 Models will be saved to Google Drive for future use")
    
    choice = input("Continue with download? (y/n): ").lower()
    if choice != 'y':
        print("⏭️ Download cancelled")
        return False
    
    if download_models_to_drive():
        print("✅ All models downloaded and saved to Drive!")
        print("💡 Future Colab sessions will be much faster")
        return True
    else:
        print("❌ Download failed")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Colab setup complete!")
        print("💡 You can now run training with: python cloud-training/production_launcher.py --category healthcare")
    else:
        print("\n❌ Colab setup failed") 