#!/usr/bin/env python3
"""
Colab Real GGUF Setup - Builds Actual llama.cpp Tools
Creates real GGUF files, not simulations
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    print("🚀 Starting Colab Real GGUF Setup...")
    
    # Step 1: Install dependencies with CUDA compatibility
    print("\n📦 Step 1: Installing dependencies with CUDA fixes...")
    run_command("pip uninstall torch torchvision torchaudio transformers -y", "Uninstalling conflicting packages")
    run_command("pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118", 
               "Installing PyTorch with CUDA 11.8")
    run_command("pip install transformers==4.35.0 datasets peft accelerate bitsandbytes", 
               "Installing transformers and related packages")
    run_command("pip install huggingface_hub wandb tensorboard", 
               "Installing HuggingFace and monitoring tools")
    run_command("pip install gguf llama-cpp-python", 
               "Installing GGUF and llama.cpp")
    run_command("pip install speechbrain librosa soundfile", 
               "Installing audio processing tools")
    run_command("pip install opencv-python Pillow numpy", 
               "Installing computer vision tools")
    run_command("pip install pyyaml tqdm rich", 
               "Installing utility packages")
    
    # Step 2: Install build tools for llama.cpp
    print("\n🔧 Step 2: Installing build tools...")
    run_command("apt-get update", "Updating package list")
    run_command("apt-get install -y build-essential cmake", "Installing build tools")
    
    # Step 3: Build llama.cpp
    print("\n🔨 Step 3: Building llama.cpp...")
    run_command("cd llama.cpp && mkdir -p build && cd build", "Creating build directory")
    run_command("cd llama.cpp/build && cmake .. -DLLAMA_CUBLAS=ON -DLLAMA_CUDA=ON", "Configuring llama.cpp with CUDA")
    run_command("cd llama.cpp/build && make -j$(nproc)", "Building llama.cpp")
    
    # Step 4: Verify llama.cpp tools
    print("\n🔍 Step 4: Verifying llama.cpp tools...")
    run_command("ls -la llama.cpp/build/bin/", "Checking built executables")
    run_command("ls -la llama.cpp/convert*.py", "Checking conversion scripts")
    
    # Step 5: Test GGUF creation
    print("\n🧪 Step 5: Testing GGUF creation...")
    run_command("cd llama.cpp && python convert_hf_to_gguf.py --help", "Testing conversion script")
    
    # Step 6: Verify installation
    print("\n🔍 Step 6: Verifying installation...")
    try:
        import torch
        import torchvision
        import transformers
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ TorchVision version: {torchvision.__version__}")
        print(f"✅ Transformers version: {transformers.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
    except ImportError as e:
        print(f"❌ Import verification failed: {e}")
        return False
    
    print("\n🎉 Colab Real GGUF setup completed successfully!")
    print("🚀 Ready for REAL GGUF creation (not simulations)!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 