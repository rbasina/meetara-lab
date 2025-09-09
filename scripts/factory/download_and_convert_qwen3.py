#!/usr/bin/env python3
"""
MeeTARA Lab - Qwen3 Model Downloader and Converter
Downloads Qwen3 base models and converts them to GGUF format for mobile/desktop

This script:
1. Downloads Qwen3-4B and Qwen3-8B models (Thinking and Instruct variants)
2. Converts them to GGUF format using llama.cpp
3. Organizes them into mobile/desktop structure

Author: MeeTARA Lab Trinity Architecture
Date: September 8, 2025
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Qwen3ModelDownloader:
    """Downloads and converts Qwen3 models to GGUF format"""
    
    def __init__(self):
        """Initialize the downloader"""
        self.base_dir = Path("models/production")
        self.mobile_dir = self.base_dir / "mobile"
        self.desktop_dir = self.base_dir / "desktop"
        
        # Model specifications
        self.mobile_models = {
            "Qwen3-4B-Thinking-2507": "Qwen/Qwen3-4B-Thinking-2507",
            "Qwen3-4B-Instruct-2507": "Qwen/Qwen3-4B-Instruct-2507"
        }
        
        self.desktop_models = {
            "Qwen3-8B-Thinking-2507": "Qwen/Qwen3-8B-Thinking-2507", 
            "Qwen3-8B-Instruct-2507": "Qwen/Qwen3-8B-Instruct-2507"
        }
        
        # Ensure directories exist
        self.mobile_dir.mkdir(parents=True, exist_ok=True)
        self.desktop_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Qwen3ModelDownloader initialized")
    
    def download_model(self, model_name: str, model_id: str) -> Path:
        """Download a model from Hugging Face"""
        logger.info(f"📥 Downloading {model_name}...")
        
        try:
            # Use huggingface-hub to download
            from huggingface_hub import snapshot_download
            
            # Download to a temporary directory
            temp_dir = Path(f"temp_models/{model_name}")
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Download the model
            snapshot_download(
                repo_id=model_id,
                local_dir=str(temp_dir),
                local_dir_use_symlinks=False
            )
            
            logger.info(f"✅ Downloaded {model_name}")
            return temp_dir
            
        except Exception as e:
            logger.error(f"❌ Failed to download {model_name}: {str(e)}")
            return None
    
    def convert_to_gguf(self, model_dir: Path, model_name: str, output_dir: Path) -> Path:
        """Convert a model to GGUF format using llama.cpp"""
        logger.info(f"🔄 Converting {model_name} to GGUF...")
        
        try:
            # Generate output filename
            timestamp = datetime.now().strftime("%Y%m%d")
            output_filename = f"meetara-{model_name}-Q4_K_M-{timestamp}.gguf"
            output_path = output_dir / output_filename
            
            # Find the convert script
            convert_script = Path("llama.cpp/convert_hf_to_gguf.py")
            if not convert_script.exists():
                logger.error(f"❌ Convert script not found: {convert_script}")
                return None
            
            # Run the conversion
            cmd = [
                "python", str(convert_script),
                str(model_dir),
                "--outfile", str(output_path),
                "--outtype", "q4_k_m"
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd="llama.cpp")
            
            if result.returncode == 0:
                logger.info(f"✅ Converted {model_name} to GGUF")
                return output_path
            else:
                logger.error(f"❌ Conversion failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error converting {model_name}: {str(e)}")
            return None
    
    def create_mobile_models(self):
        """Create mobile models (4B parameters)"""
        logger.info("📱 Creating mobile models...")
        created_models = {}
        
        for model_name, model_id in self.mobile_models.items():
            try:
                # Download model
                model_dir = self.download_model(model_name, model_id)
                if not model_dir:
                    continue
                
                # Convert to GGUF
                gguf_path = self.convert_to_gguf(model_dir, model_name, self.mobile_dir)
                if gguf_path:
                    created_models[model_name] = str(gguf_path)
                
                # Cleanup temp directory
                shutil.rmtree(model_dir, ignore_errors=True)
                
            except Exception as e:
                logger.error(f"❌ Error creating mobile model {model_name}: {str(e)}")
        
        return created_models
    
    def create_desktop_models(self):
        """Create desktop models (8B parameters)"""
        logger.info("🖥️ Creating desktop models...")
        created_models = {}
        
        for model_name, model_id in self.desktop_models.items():
            try:
                # Download model
                model_dir = self.download_model(model_name, model_id)
                if not model_dir:
                    continue
                
                # Convert to GGUF
                gguf_path = self.convert_to_gguf(model_dir, model_name, self.desktop_dir)
                if gguf_path:
                    created_models[model_name] = str(gguf_path)
                
                # Cleanup temp directory
                shutil.rmtree(model_dir, ignore_errors=True)
                
            except Exception as e:
                logger.error(f"❌ Error creating desktop model {model_name}: {str(e)}")
        
        return created_models
    
    def create_universal_models(self, mobile_models: dict, desktop_models: dict):
        """Create universal model files (placeholders for now)"""
        logger.info("🌐 Creating universal model files...")
        
        timestamp = datetime.now().strftime("%Y%m%d")
        
        # Mobile universal
        mobile_universal = self.mobile_dir / f"meetara_mobile_universal-model-Q4_K_M-{timestamp}.gguf"
        if mobile_models:
            # Copy the first mobile model as universal
            first_model = list(mobile_models.values())[0]
            shutil.copy2(first_model, mobile_universal)
            logger.info(f"✅ Created mobile universal model: {mobile_universal.name}")
        
        # Desktop universal
        desktop_universal = self.desktop_dir / f"meetara_desktop_universal-model-Q4_K_M-{timestamp}.gguf"
        if desktop_models:
            # Copy the first desktop model as universal
            first_model = list(desktop_models.values())[0]
            shutil.copy2(first_model, desktop_universal)
            logger.info(f"✅ Created desktop universal model: {desktop_universal.name}")
    
    def create_manifest(self, mobile_models: dict, desktop_models: dict):
        """Create a manifest file"""
        manifest = {
            "created_at": datetime.now().isoformat(),
            "description": "MeeTARA Lab Mobile & Desktop Models - Qwen3 Base Models",
            "models": {
                "mobile": {
                    "description": "4B parameter models for mobile devices",
                    "models": mobile_models
                },
                "desktop": {
                    "description": "8B parameter models for desktop applications",
                    "models": desktop_models
                }
            }
        }
        
        manifest_path = self.base_dir / "model_manifest.json"
        import json
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Manifest created: {manifest_path}")

def main():
    """Main execution function"""
    logger.info("🚀 Starting Qwen3 Model Download and Conversion")
    
    try:
        # Initialize downloader
        downloader = Qwen3ModelDownloader()
        
        # Create mobile models
        logger.info("📱 Creating mobile models (4B parameters)...")
        mobile_models = downloader.create_mobile_models()
        
        # Create desktop models
        logger.info("🖥️ Creating desktop models (8B parameters)...")
        desktop_models = downloader.create_desktop_models()
        
        # Create universal models
        downloader.create_universal_models(mobile_models, desktop_models)
        
        # Create manifest
        downloader.create_manifest(mobile_models, desktop_models)
        
        # Report results
        logger.info("📊 Model Creation Summary:")
        logger.info(f"Mobile models created: {len(mobile_models)}")
        logger.info(f"Desktop models created: {len(desktop_models)}")
        
        if mobile_models or desktop_models:
            logger.info("🎉 Model creation completed successfully!")
            logger.info("\n📁 Model Structure:")
            logger.info("models/production/")
            logger.info("├── mobile/")
            for model_name, model_path in mobile_models.items():
                logger.info(f"│   ├── {Path(model_path).name}")
            logger.info("├── desktop/")
            for model_name, model_path in desktop_models.items():
                logger.info(f"│   ├── {Path(model_path).name}")
            logger.info("└── speech_models/ (already exists)")
        else:
            logger.warning("⚠️ No models were created successfully")
            
    except Exception as e:
        logger.error(f"❌ Model creation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
