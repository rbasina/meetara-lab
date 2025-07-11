#!/usr/bin/env python3
"""
MeeTARA Lab - llama.cpp Setup Script
Installs llama.cpp and llama-cpp-python for GGUF testing
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import shutil

# Try to import yaml, provide guidance if not found
try:
    import yaml
except ImportError:
    print("❌ PyYAML is not installed. Please run 'pip install pyyaml' and run the script again.")
    sys.exit(1)

def install_llama_cpp_python():
    """Install llama-cpp-python package"""
    print("🔧 Installing llama-cpp-python...")
    
    try:
        # Install with CUDA support if available
        if platform.system() == "Windows":
            # Windows with CUDA
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "llama-cpp-python[cuda]", "--upgrade"
            ], check=True)
        else:
            # Linux/Mac
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "llama-cpp-python", "--upgrade"
            ], check=True)
        
        print("✅ llama-cpp-python installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install llama-cpp-python: {e}")
        return False

def clone_llama_cpp():
    """Clone llama.cpp repository"""
    print("📥 Cloning llama.cpp repository...")
    
    try:
        llama_cpp_dir = Path("llama.cpp")
        
        if llama_cpp_dir.exists():
            print("✅ llama.cpp directory already exists")
            return True
        
        subprocess.run([
            "git", "clone", "https://github.com/ggerganov/llama.cpp.git"
        ], check=True)
        
        print("✅ llama.cpp cloned successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to clone llama.cpp: {e}")
        return False

def build_llama_cpp():
    """Builds llama.cpp using CMake."""
    print("🔨 Building llama.cpp...")
    try:
        # Force a clean build by removing the build directory first
        build_dir = Path('llama.cpp/build')
        if build_dir.exists():
            print("🧹 Cleaning previous build...")
            shutil.rmtree(build_dir)
            
        # Build with CMake, disabling CURL
        subprocess.run(['cmake', '-B', 'build', '-DLLAMA_CURL=OFF'], cwd='llama.cpp', check=True)
        subprocess.run(['cmake', '--build', 'build', '--config', 'Release'], cwd='llama.cpp', check=True)
        print("✅ llama.cpp built successfully")
        
        # Verify if convert_hf_to_gguf.py exists after build
        converter_script_path = Path('llama.cpp') / "convert_hf_to_gguf.py"
        if not converter_script_path.exists():
            print(f"❌ Critical: convert_hf_to_gguf.py not found after llama.cpp build at {converter_script_path}")
            return False
        print(f"✅ Verified: convert_hf_to_gguf.py found at {converter_script_path}")
        
        # Check for quantize executable in build directory
        import platform
        if platform.system() == "Windows":
            quantize_path = Path('llama.cpp') / "build" / "bin" / "test-quantize-stats.exe"
            if not quantize_path.exists():
                print(f"❌ Critical: test-quantize-stats.exe not found after llama.cpp build at {quantize_path}")
                return False
            print(f"✅ Verified: test-quantize-stats.exe found at {quantize_path}")
        else:
            # For Linux/Mac, check for the standard quantize executable
            quantize_path = Path('llama.cpp') / "quantize"
            if not quantize_path.exists():
                print(f"❌ Critical: quantize executable not found after llama.cpp build at {quantize_path}")
                return False
            print(f"✅ Verified: quantize executable found at {quantize_path}")

        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to build llama.cpp: {e}")
        return False

def test_installation():
    """Test llama.cpp installation"""
    print("🧪 Testing llama.cpp installation...")
    
    try:
        # Test Python package
        import llama_cpp
        print("✅ llama-cpp-python import successful")
        
        # Test executable
        llama_cpp_dir = Path("llama.cpp")
        if platform.system() == "Windows":
            main_exe = llama_cpp_dir / "build" / "bin" / "Release" / "main.exe"
        else:
            main_exe = llama_cpp_dir / "main"
        
        if main_exe.exists():
            print(f"✅ llama.cpp executable found: {main_exe}")
        else:
            print("⚠️ llama.cpp executable not found")
        
        return True
        
    except ImportError:
        print("❌ llama-cpp-python not properly installed")
        return False

def update_config():
    """Update trinity_config.yaml with the absolute path to the llama.cpp directory."""
    print("⚙️ Updating unified configuration with llama.cpp path...")

    try:
        config_path = Path("config/trinity_config.yaml")

        if not config_path.exists():
            print(f"⚠️ Unified configuration file not found at '{config_path}'. Skipping update.")
            return False

        # Read current config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Update llama.cpp path at the top level
        llama_cpp_dir = Path("llama.cpp").resolve()
        config["llama_cpp_path"] = str(llama_cpp_dir)

        # Write updated config
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"✅ Unified configuration updated with llama_cpp_path: {llama_cpp_dir}")
        return True

    except yaml.YAMLError as e:
        print(f"❌ Failed to read or parse YAML configuration: {e}")
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred while updating configuration: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 MeeTARA Lab - llama.cpp Setup")
    print("=" * 50)
    
    success_count = 0
    total_steps = 5
    
    # Step 1: Install Python package
    if install_llama_cpp_python():
        success_count += 1
    
    # Step 2: Clone repository
    if clone_llama_cpp():
        success_count += 1
    
    # Step 3: Build executables
    if build_llama_cpp():
        success_count += 1
    
    # Step 4: Test installation
    if test_installation():
        success_count += 1
    
    # Step 5: Update configuration
    if update_config():
        success_count += 1
    
    print("\n" + "=" * 50)
    print(f"🎯 Setup completed: {success_count}/{total_steps} steps successful")
    
    if success_count == total_steps:
        print("✅ llama.cpp setup completed successfully!")
        print("🧪 You can now run real GGUF testing in MeeTARA Lab")
    else:
        print("⚠️ Some steps failed. Check the output above for details.")
        print("💡 You can still use simulated testing if needed.")

if __name__ == "__main__":
    main() 