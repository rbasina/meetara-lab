#!/usr/bin/env python3
"""
🚀 Enhanced Colab Model Setup for MeeTARA Lab
Complete setup including llama.cpp + CUDA compilation, model downloads, and Google Drive sync
This provides a one-stop setup for production-ready Colab environments
"""

import os
import sys
import time
import subprocess
from pathlib import Path
import shutil

def fix_numpy_jax_compatibility():
    """Fix NumPy/JAX compatibility issues in Colab"""
    print("🔧 Fixing NumPy/JAX compatibility...")
    
    try:
        # First, uninstall problematic packages
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "jax", "jaxlib"], 
                      capture_output=True, text=True)
        print("✅ Uninstalled JAX packages")
        
        # Install compatible NumPy version
        subprocess.run([sys.executable, "-m", "pip", "install", "numpy==1.24.3"], 
                      check=True, capture_output=True, text=True)
        print("✅ Installed compatible NumPy version")
        
        # Reinstall JAX with compatible versions
        subprocess.run([sys.executable, "-m", "pip", "install", "jax==0.4.20", "jaxlib==0.4.20"], 
                      check=True, capture_output=True, text=True)
        print("✅ Reinstalled JAX with compatible versions")
        
        return True
    except Exception as e:
        print(f"❌ Failed to fix NumPy/JAX compatibility: {e}")
        return False

def setup_colab_environment():
    """Setup Colab environment and mount Google Drive"""
    print("🚀 Setting up Colab environment...")
    
    # Check if we're in Colab
    try:
        import google.colab
        print("✅ Running in Google Colab environment")
    except ImportError:
        print("❌ Not running in Colab - Google Drive not available")
        return False
    
    # Mount Google Drive with better error handling
    try:
        from google.colab import drive
        print("📁 Attempting to mount Google Drive...")
        drive.mount('/content/drive')
        print("✅ Google Drive mounted successfully")
        return True
    except Exception as e:
        print(f"⚠️ Google Drive mounting failed: {e}")
        print("💡 Continuing without Google Drive - models will be downloaded locally only")
        print("💡 You can manually mount Drive later if needed")
        return False

def install_requirements():
    """Install required packages for MeeTARA Lab with Colab compatibility fixes"""
    print("📦 Installing required packages...")
    
    try:
        # Fix NumPy/JAX compatibility first
        if not fix_numpy_jax_compatibility():
            print("⚠️ NumPy/JAX compatibility fix failed, continuing...")
        
        # Install core requirements with specific versions for Colab
        colab_compatible_packages = [
            "torch==2.1.0",
            "torchvision==0.16.0", 
            "torchaudio==2.1.0",
            "transformers==4.36.2",
            "accelerate==0.25.0",
            "datasets==2.16.1",
            "peft==0.7.1",
            "tokenizers==0.15.0",
            "bitsandbytes==0.41.3",
            "triton==2.1.0",
            "llama-cpp-python==0.2.23",
            "gguf==0.1.0",
            "speechbrain==0.5.16",
            "librosa==0.10.1",
            "soundfile==0.12.1",
            "opencv-python==4.8.1.78",
            "Pillow==10.1.0",
            "numpy==1.24.3",
            "pandas==2.1.4",
            "scipy==1.11.4",
            "pyyaml==6.0.1",
            "tqdm==4.66.1",
            "rich==13.7.0",
            "huggingface_hub==0.19.4",
            "wandb==0.16.1",
            "tensorboard==2.15.1",
            "requests==2.31.0",
            "urllib3==2.0.7"
        ]
        
        for package in colab_compatible_packages:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", package], 
                              check=True, capture_output=True, text=True)
                print(f"✅ {package} installed")
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Failed to install {package}: {e}")
        
        # Install additional packages for llama.cpp
        additional_packages = [
            "cmake",
            "ninja"
        ]
        
        for package in additional_packages:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", package], 
                              check=True, capture_output=True, text=True)
                print(f"✅ {package} installed")
            except subprocess.CalledProcessError:
                print(f"⚠️ Failed to install {package} (may already be installed)")
        
        print("✅ All packages installed successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def ensure_llama_cpp():
    """Ensure llama.cpp is present and valid. Auto-clone if missing or broken."""
    llama_path = Path("llama.cpp")
    cmake_file = llama_path / "CMakeLists.txt"
    if not cmake_file.exists():
        print("⚠️ llama.cpp/CMakeLists.txt not found. Cloning fresh repo...")
        if llama_path.exists():
            print("🧹 Removing broken or partial llama.cpp directory...")
            shutil.rmtree(llama_path)
        # Clone official repo
        import subprocess
        result = subprocess.run([
            "git", "clone", "https://github.com/ggerganov/llama.cpp.git", str(llama_path)
        ], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ llama.cpp cloned successfully!")
        else:
            print(f"❌ Failed to clone llama.cpp: {result.stderr}")
            return False
    else:
        print("✅ llama.cpp/CMakeLists.txt found.")
    return True

def setup_llama_cpp():
    """Setup llama.cpp with CUDA support"""
    print("🔧 Setting up llama.cpp with CUDA support...")
    
    # Ensure llama.cpp is present and valid
    if not ensure_llama_cpp():
        print("❌ Could not ensure llama.cpp is present and valid.")
        return False
    
    try:
        # Navigate to llama.cpp directory
        llama_path = Path("llama.cpp")
        if not llama_path.exists():
            print("❌ llama.cpp directory not found")
            return False
        
        os.chdir(llama_path)
        print(f"📁 Working in: {os.getcwd()}")
        
        # Check CUDA availability
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                cuda_version = torch.version.cuda
                print(f"✅ CUDA available: {cuda_version}")
            else:
                print("⚠️ CUDA not available, will compile CPU-only version")
        except ImportError:
            print("⚠️ PyTorch not available, proceeding with CPU compilation")
            cuda_available = False
        
        # Clean the build directory to avoid stale CMake cache
        build_dir = Path("build")
        if build_dir.exists():
            shutil.rmtree(build_dir)

        # Configure CMake with CUDA support
        cmake_cmd = ["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"]
        if cuda_available:
            cmake_cmd.append("-DGGML_CUDA=ON")
            print("🔧 Configuring with CUDA support...")
        else:
            print("🔧 Configuring CPU-only version...")
        
        result = subprocess.run(cmake_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ CMake configuration failed: {result.stderr}")
            return False
        
        print("✅ CMake configuration successful")
        
        # Build llama.cpp with all essential tools
        print("🔨 Building llama.cpp (essential tools)...")
        build_cmd = ["cmake", "--build", "build", "--config", "Release", "-j"]
        result = subprocess.run(build_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Build failed: {result.stderr}")
            return False
        
        print("✅ llama.cpp build successful")
        
        # Check for quantize tool in different possible locations
        quantize_paths = [
            "build/bin/quantize",
            "build/bin/llama-quantize",
            "build/quantize",
            "build/llama-quantize"
        ]
        
        quantize_found = False
        for quantize_path in quantize_paths:
            if Path(quantize_path).exists():
                print(f"✅ Found quantize tool: {quantize_path}")
                quantize_found = True
                # Create symlink for consistent access
                try:
                    os.symlink(str(Path(quantize_path).absolute()), "build/bin/quantize")
                    print("✅ Created symlink: build/bin/quantize")
                except FileExistsError:
                    print("✅ Symlink already exists")
                except Exception as e:
                    print(f"⚠️ Could not create symlink: {e}")
                break
        
        if not quantize_found:
            print("❌ Quantize tool not found in any expected location")
            print("🔍 Checking build directory contents:")
            if Path("build/bin").exists():
                for file in Path("build/bin").iterdir():
                    print(f"   - {file.name}")
            return False
        
        # Verify key tools are available
        tools_to_check = [
            "build/bin/quantize",
            "convert_hf_to_gguf.py"
        ]
        
        for tool in tools_to_check:
            if Path(tool).exists():
                print(f"✅ {tool} available")
            else:
                print(f"❌ {tool} not found")
                if tool == "build/bin/quantize":
                    print("💡 This will cause simulated quantization")
        
        # Return to project root
        os.chdir("..")
        print(f"📁 Returned to: {os.getcwd()}")
        
        return True
        
    except Exception as e:
        print(f"❌ llama.cpp setup failed: {e}")
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

def verify_setup():
    """Verify that the complete setup is working"""
    print("🔍 Verifying setup...")
    
    checks = []
    
    # Check llama.cpp tools
    llama_tools = [
        "llama.cpp/build/bin/llama-quantize",
        "llama.cpp/convert_hf_to_gguf.py"
    ]
    
    for tool in llama_tools:
        if Path(tool).exists():
            checks.append(f"✅ {tool}")
        else:
            checks.append(f"❌ {tool}")
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            checks.append(f"✅ CUDA available: {torch.version.cuda}")
        else:
            checks.append("⚠️ CUDA not available")
    except ImportError:
        checks.append("❌ PyTorch not available")
    
    # Check model directories
    model_path = Path("models/base_models")
    if model_path.exists():
        model_count = len(list(model_path.iterdir()))
        checks.append(f"✅ Models directory: {model_count} models")
    else:
        checks.append("❌ Models directory not found")
    
    print("\n📊 Setup Verification:")
    for check in checks:
        print(f"   {check}")
    
    return all("✅" in check for check in checks)

def main():
    """Main Colab setup function"""
    print("=" * 60)
    print("🚀 MeeTARA Lab - Enhanced Colab Setup")
    print("=" * 60)
    
    # Step 1: Setup Colab environment (Google Drive is optional)
    drive_available = setup_colab_environment()
    if not drive_available:
        print("⚠️ Continuing without Google Drive - models will be stored locally only")
    
    # Step 2: Install requirements
    if not install_requirements():
        print("❌ Requirements installation failed")
        return False
    
    # Step 3: Setup llama.cpp with CUDA
    if not setup_llama_cpp():
        print("❌ llama.cpp setup failed")
        return False
    
    # Step 4: Handle model downloads/sync (only if Drive is available)
    if drive_available:
        if check_drive_models():
            print("\n📁 Models found in Google Drive!")
            choice = input("Do you want to sync from Drive to local? (y/n): ").lower()
            if choice == 'y':
                if sync_models_from_drive():
                    print("✅ Models synced successfully!")
                else:
                    print("❌ Failed to sync models")
                    return False
            else:
                print("⏭️ Skipping sync")
        else:
            print("\n📥 No models found in Drive - downloading now...")
            print("⏰ This will take 30-60 minutes for all models")
            print("💾 Models will be saved to Google Drive for future use")
            
            choice = input("Continue with download? (y/n): ").lower()
            if choice == 'y':
                if download_models_to_drive():
                    print("✅ All models downloaded and saved to Drive!")
                else:
                    print("❌ Download failed")
                    return False
            else:
                print("⏭️ Download cancelled")
    else:
        print("\n📥 Google Drive not available - downloading models locally...")
        print("⏰ This will take 30-60 minutes for all models")
        print("💾 Models will be stored locally only")
        
        choice = input("Continue with local download? (y/n): ").lower()
        if choice == 'y':
            # Download models locally
            try:
                sys.path.append(str(Path.cwd() / "scripts" / "factory"))
                from download_base_models import BaseModelDownloader
                
                downloader = BaseModelDownloader("models/base_models")
                results = downloader.download_all_base_models(sync_to_drive=False)
                
                success_count = sum(results.values())
                print(f"📊 Downloaded {success_count}/{len(results)} models locally")
                
                if success_count == len(results):
                    print("✅ All models downloaded locally!")
                else:
                    print("⚠️ Some models failed to download")
            except Exception as e:
                print(f"❌ Local download failed: {e}")
                return False
        else:
            print("⏭️ Download cancelled")
    
    # Step 5: Verify complete setup
    if verify_setup():
        print("\n🎉 Complete setup successful!")
        print("💡 You can now run training with: python cloud-training/production_launcher.py --category healthcare")
        return True
    else:
        print("\n⚠️ Setup completed with warnings - some components may not work optimally")
        return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Enhanced Colab setup complete!")
        print("🚀 Ready for production training!")
    else:
        print("\n❌ Enhanced Colab setup failed") 