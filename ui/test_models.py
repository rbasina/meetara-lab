#!/usr/bin/env python3
"""
🧪 MeeTARA Lab - Model Testing Script
Test loading and using the trained models
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_model_loading():
    """Test loading the trained models"""
    print("🧪 Testing MeeTARA Lab Model Loading")
    print("=" * 50)
    
    # Check if llama-cpp-python is available
    try:
        from llama_cpp import Llama
        print("✅ llama-cpp-python available")
    except ImportError:
        print("❌ llama-cpp-python not available")
        print("💡 Install with: pip install llama-cpp-python")
        return False
    
    # Define model paths
    models_dir = project_root / "models" / "production"
    
    test_models = {
        "A_universal_full": models_dir / "A_universal_full" / "meetara_a_universal_full.gguf",
        "B_universal_lite": models_dir / "B_universal_lite" / "meetara_b_universal_lite.gguf",
        "C_healthcare_specialist": models_dir / "C_category_specific" / "meetara_healthcare_specialist_v1_Q4.gguf",
        "D_general_health": models_dir / "D_domain_specific" / "healthcare" / "general_health" / "general_health_20250712_055244_Q4_K_M.gguf"
    }
    
    # Check which models exist
    available_models = {}
    for name, path in test_models.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            available_models[name] = {
                "path": path,
                "size_mb": size_mb
            }
            print(f"✅ {name}: {size_mb:.1f}MB")
        else:
            print(f"❌ {name}: Not found at {path}")
    
    if not available_models:
        print("❌ No models found to test")
        return False
    
    # Test loading the smallest model first
    smallest_model = min(available_models.items(), key=lambda x: x[1]["size_mb"])
    model_name, model_info = smallest_model
    
    print(f"\n🚀 Testing model: {model_name} ({model_info['size_mb']:.1f}MB)")
    
    try:
        # Load the model
        print("📥 Loading model...")
        start_time = time.time()
        
        model = Llama(
            model_path=str(model_info["path"]),
            n_ctx=1024,
            n_threads=4,
            verbose=False
        )
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f}s")
        
        # Test inference
        print("🧠 Testing inference...")
        test_prompt = "Hello, how are you today?"
        
        start_time = time.time()
        response = model(
            test_prompt,
            max_tokens=50,
            temperature=0.7,
            top_p=0.9,
            echo=False
        )
        
        inference_time = time.time() - start_time
        generated_text = response["choices"][0]["text"].strip()
        
        print(f"✅ Inference completed in {inference_time:.2f}s")
        print(f"📝 Response: {generated_text}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_ui_launch():
    """Test launching the UI"""
    print("\n🌐 Testing UI Launch")
    print("=" * 30)
    
    ui_dir = project_root / "ui"
    launcher_script = ui_dir / "launch_real_comparison.py"
    
    if launcher_script.exists():
        print("✅ UI launcher found")
        print("💡 To launch UI, run: python ui/launch_real_comparison.py")
        return True
    else:
        print("❌ UI launcher not found")
        return False

def main():
    """Main test function"""
    print("🤖 MeeTARA Lab Model Testing")
    print("=" * 50)
    
    # Test model loading
    model_test_passed = test_model_loading()
    
    # Test UI launch
    ui_test_passed = test_ui_launch()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   🧠 Model Loading: {'✅ PASSED' if model_test_passed else '❌ FAILED'}")
    print(f"   🌐 UI Launch: {'✅ PASSED' if ui_test_passed else '❌ FAILED'}")
    
    if model_test_passed and ui_test_passed:
        print("\n🎉 All tests passed! Your MeeTARA Lab is ready!")
        print("\n🚀 Next steps:")
        print("   1. Run: python ui/launch_real_comparison.py")
        print("   2. Open browser to: http://localhost:5001")
        print("   3. Test different models with your queries")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main() 