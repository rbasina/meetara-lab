#!/usr/bin/env python3
"""
🚀 MeeTARA Lab - Real Model Comparison Launcher
Quick launcher for the real model comparison UI
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        "flask",
        "flask-cors"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} - OK")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - Missing")
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install"
            ] + missing_packages, check=True)
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            return False
    
    return True

def check_llama_cpp():
    """Check if llama-cpp-python is available"""
    print("🦙 Checking llama-cpp-python...")
    
    try:
        import llama_cpp
        print("✅ llama-cpp-python - Available (Real model loading)")
        return True
    except ImportError:
        print("⚠️ llama-cpp-python - Not available (Simulation mode)")
        print("💡 To enable real model loading, run: pip install llama-cpp-python")
        return False

def check_models():
    """Check if GGUF models are available"""
    print("🤖 Checking GGUF models...")
    
    base_dir = Path(__file__).parent.parent
    models_dir = base_dir / "models"
    
    models_to_check = [
        "A_universal_full/meetara_a_universal_full.gguf",
        "B_universal_lite/meetara_b_universal_lite.gguf", 
        "C_category_specific/meetara_c_category_specific.gguf"
    ]
    
    available_models = []
    
    for model_path in models_to_check:
        full_path = models_dir / model_path
        if full_path.exists():
            size_mb = full_path.stat().st_size / (1024 * 1024)
            print(f"✅ {model_path} - {size_mb:.1f}MB")
            available_models.append(model_path)
        else:
            print(f"❌ {model_path} - Not found")
    
    if not available_models:
        print("⚠️ No GGUF models found - running in simulation mode")
        print("💡 Run the GGUF factory to create models first")
    
    return len(available_models) > 0

def launch_server():
    """Launch the Flask server"""
    print("🚀 Starting MeeTARA Real Model Comparison Server...")
    
    # Set up paths
    ui_dir = Path(__file__).parent
    backend_script = ui_dir / "meetara_real_model_comparison.py"
    
    if not backend_script.exists():
        print(f"❌ Backend script not found: {backend_script}")
        return False
    
    # Change to UI directory
    os.chdir(ui_dir)
    
    try:
        # Start the Flask server
        print("🌐 Server starting at http://localhost:5001")
        print("🔧 Press Ctrl+C to stop the server")
        
        # Wait a moment then open browser
        def open_browser():
            time.sleep(2)
            webbrowser.open("http://localhost:5001")
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Run the server
        subprocess.run([sys.executable, "meetara_real_model_comparison.py"], check=True)
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed to start: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main launcher function"""
    print("🤖 MeeTARA Real Model Comparison Launcher")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed")
        return
    
    # Check llama-cpp-python
    llama_available = check_llama_cpp()
    
    # Check models
    models_available = check_models()
    
    print("\n" + "=" * 50)
    print("🎯 System Status:")
    print(f"   🔧 Dependencies: ✅ Ready")
    print(f"   🦙 llama-cpp-python: {'✅ Available' if llama_available else '⚠️ Simulation mode'}")
    print(f"   🤖 GGUF Models: {'✅ Available' if models_available else '⚠️ Simulation mode'}")
    
    if llama_available and models_available:
        print("🚀 Ready for real model comparison!")
    else:
        print("🎭 Running in simulation mode")
    
    print("\n🌐 Starting web interface...")
    
    # Launch server
    launch_server()

if __name__ == "__main__":
    main() 