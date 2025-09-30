#!/usr/bin/env python3
"""
MeeTARA Service Bundle Auto-Installer
Automatically integrates service bundle with MeeTARA frontend
"""

import os
import json
import shutil
from pathlib import Path

def install_meetara_services():
    """Install MeeTARA services to frontend repository"""
    print("🚀 Installing MeeTARA Service Bundle...")
    
    # Detect MeeTARA frontend directory
    current_dir = Path(__file__).parent.parent
    meetara_frontend = current_dir.parent / "meetara"  # Assume sibling directory
    
    if not meetara_frontend.exists():
        print("❌ MeeTARA frontend repository not found!")
        print(f"Expected location: {meetara_frontend}")
        return False
    
    # Create services directory in frontend
    frontend_services = meetara_frontend / "services"
    frontend_models = meetara_frontend / "models"
    
    frontend_services.mkdir(exist_ok=True)
    frontend_models.mkdir(exist_ok=True)
    
    # Copy models
    bundle_dir = Path(__file__).parent.parent
    models_dir = bundle_dir / "models"
    services_dir = bundle_dir / "services"
    config_dir = bundle_dir / "config"
    
    # Copy all files
    if models_dir.exists():
        shutil.copytree(models_dir, frontend_models, dirs_exist_ok=True)
        print("✅ Models copied to frontend")
    
    if services_dir.exists():
        shutil.copytree(services_dir, frontend_services, dirs_exist_ok=True) 
        print("✅ Services copied to frontend")
    
    if config_dir.exists():
        frontend_config = meetara_frontend / "config" / "services"
        frontend_config.mkdir(parents=True, exist_ok=True)
        shutil.copytree(config_dir, frontend_config, dirs_exist_ok=True)
        print("✅ Configuration copied to frontend")
    
    print("🎉 MeeTARA Service Bundle installed successfully!")
    return True

if __name__ == "__main__":
    install_meetara_services()
