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
    """Build llama.cpp executables"""
    print("🔨 Building llama.cpp...")
    
    try:
        llama_cpp_dir = Path("llama.cpp")
        
        if not llama_cpp_dir.exists():
            print("❌ llama.cpp directory not found")
            return False
        
        # Change to llama.cpp directory
        os.chdir(llama_cpp_dir)
        
        if platform.system() == "Windows":
            # Windows build
            subprocess.run(["cmake", "-B", "build"], check=True)
            subprocess.run(["cmake", "--build", "build", "--config", "Release"], check=True)
        else:
            # Linux/Mac build
            subprocess.run(["make", "-j4"], check=True)
        
        # Change back to original directory
        os.chdir("..")
        
        print("✅ llama.cpp built successfully")
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
    """Update trinity-config.json with llama.cpp path"""
    print("⚙️ Updating configuration...")
    
    try:
        import json
        
        config_path = Path("config/trinity-config.json")
        
        if not config_path.exists():
            print("⚠️ trinity-config.json not found")
            return False
        
        # Read current config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Update llama.cpp path
        llama_cpp_dir = Path("llama.cpp").resolve()
        config["llama_cpp_path"] = str(llama_cpp_dir)
        
        # Write updated config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Configuration updated with llama.cpp path: {llama_cpp_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to update configuration: {e}")
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