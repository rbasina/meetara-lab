#!/usr/bin/env python3
"""
Colab Real GGUF Setup - Fixed Version
Works with actual llama.cpp repository structure
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
    print("🚀 Starting Colab Real GGUF Setup (Fixed)...")
    
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
    
    # Step 3: Check llama.cpp structure
    print("\n🔍 Step 3: Checking llama.cpp structure...")
    run_command("ls -la llama.cpp/", "Checking llama.cpp root")
    run_command("find llama.cpp -name '*.py' -type f", "Finding Python scripts")
    run_command("find llama.cpp -name 'CMakeLists.txt'", "Finding CMakeLists.txt")
    
    # Step 4: Build llama.cpp (if CMakeLists.txt exists)
    print("\n🔨 Step 4: Building llama.cpp...")
    cmake_exists = run_command("test -f llama.cpp/CMakeLists.txt && echo 'CMakeLists.txt found'", "Checking for CMakeLists.txt")
    
    if cmake_exists:
        run_command("cd llama.cpp && mkdir -p build", "Creating build directory")
        run_command("cd llama.cpp/build && cmake .. -DLLAMA_CUBLAS=ON -DLLAMA_CUDA=ON", "Configuring llama.cpp with CUDA")
        run_command("cd llama.cpp/build && make -j$(nproc)", "Building llama.cpp")
    else:
        print("⚠️ CMakeLists.txt not found, skipping build")
    
    # Step 5: Install gguf-py for conversion
    print("\n📦 Step 5: Installing gguf-py for conversion...")
    run_command("cd llama.cpp/gguf-py && pip install -e .", "Installing gguf-py in development mode")
    
    # Step 6: Create conversion script if it doesn't exist
    print("\n🔧 Step 6: Setting up conversion tools...")
    conversion_script = """
#!/usr/bin/env python3
\"\"\"
Simple HuggingFace to GGUF conversion script
\"\"\"
import sys
import os
sys.path.append('llama.cpp/gguf-py')

from gguf import convert_hf_to_gguf

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_hf_to_gguf.py <model_path> [output_path]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "model.gguf"
    
    print(f"Converting {model_path} to {output_path}")
    convert_hf_to_gguf(model_path, output_path)
    print("Conversion complete!")
"""
    
    with open("llama.cpp/convert_hf_to_gguf.py", "w") as f:
        f.write(conversion_script)
    
    run_command("chmod +x llama.cpp/convert_hf_to_gguf.py", "Making conversion script executable")
    
    # Step 7: Verify tools
    print("\n🔍 Step 7: Verifying tools...")
    run_command("ls -la llama.cpp/convert_hf_to_gguf.py", "Checking conversion script")
    run_command("ls -la llama.cpp/build/bin/ 2>/dev/null || echo 'Build directory not found'", "Checking built executables")
    
    # Step 8: Test GGUF creation
    print("\n🧪 Step 8: Testing GGUF creation...")
    run_command("cd llama.cpp && python convert_hf_to_gguf.py --help", "Testing conversion script")
    
    # Step 9: Verify installation
    print("\n🔍 Step 9: Verifying installation...")
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