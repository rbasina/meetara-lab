#!/usr/bin/env python3
"""Simple validation of MeeTARA Lab Trinity Architecture"""

import sys
import os
from pathlib import Path

def main():
    print("🚀 MeeTARA Lab - Simple Validation Check")
    print("=" * 50)
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"✅ Python Version: {python_version}")
    
    # Check conda environment
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "unknown")
    print(f"✅ Conda Environment: {conda_env}")
    
    # Check directory structure (go up one level since we're in tests folder now)
    project_root = Path.cwd().parent if Path.cwd().name == "tests" else Path.cwd()
    dirs_to_check = [
        "trinity-core", "intelligence-hub", "model-factory",
        "cloud-training", "notebooks", "memory-bank", "tests"
    ]
    
    print("\n📁 Directory Structure:")
    for dir_name in dirs_to_check:
        dir_path = project_root / dir_name
        status = "✅" if dir_path.exists() else "❌"
        print(f"   {status} {dir_name}")
    
    # Check key files
    key_files = [
        "notebooks/colab_gpu_training_template.ipynb",
        "OPEN_IN_COLAB.md",
        "requirements.txt",
        ".gitignore"
    ]
    
    print("\n📄 Key Files:")
    for file_name in key_files:
        file_path = project_root / file_name
        status = "✅" if file_path.exists() else "❌"
        print(f"   {status} {file_name}")
    
    # Check Trinity components
    trinity_files = [
        "trinity-core/tts_manager.py",
        "trinity-core/emotion_detector.py",
        "trinity-core/intelligent_router.py",
        "intelligence-hub/domain_experts.py",
        "model-factory/gguf_factory.py"
    ]
    
    print("\n🧠 Trinity Components (sample):")
    for file_name in trinity_files:
        file_path = project_root / file_name
        status = "✅" if file_path.exists() else "❌"
        size = file_path.stat().st_size if file_path.exists() else 0
        print(f"   {status} {file_name} ({size} bytes)")
    
    # Count total Trinity components
    trinity_core = project_root / "trinity-core"
    if trinity_core.exists():
        py_files = list(trinity_core.glob("*.py"))
        print(f"\n🎯 Trinity Core: {len(py_files)} Python files")
    
    print("\n" + "=" * 50)
    print("🎉 Basic validation complete!")
    print("💡 Ready for Trinity notebook testing!")

if __name__ == "__main__":
    main() 