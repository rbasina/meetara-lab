#!/usr/bin/env python3
"""
MeeTARA Lab - Convert Downloaded Models to GGUF
Converts already downloaded Qwen3 models to GGUF format

This script:
1. Uses the already downloaded models in temp_models/
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

class ConvertDownloadedModels:
    """Converts already downloaded models to GGUF format"""
    
    def __init__(self):
        """Initialize the converter"""
        self.base_dir = Path("models/production")
        self.mobile_dir = self.base_dir / "mobile"
        self.desktop_dir = self.base_dir / "desktop"
        self.temp_dir = Path("temp_models")
        
        # Model mappings
        self.models = {
            "mobile": {
                "Qwen3-4B-Base": "Qwen3-4B-Base",
                "Qwen3-4B-Thinking-2507": "Qwen3-4B-Thinking-2507"
            },
            "desktop": {
                "Qwen3-8B-Thinking-2507": "Qwen3-8B-Thinking-2507"
            }
        }
        
        # Ensure directories exist
        self.mobile_dir.mkdir(parents=True, exist_ok=True)
        self.desktop_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("ConvertDownloadedModels initialized")
    
    def convert_to_gguf(self, model_dir: Path, model_name: str, output_dir: Path) -> Path:
        """Convert a model to GGUF format using llama.cpp"""
        logger.info(f"🔄 Converting {model_name} to GGUF...")
        
        try:
            # Generate output filename
            timestamp = datetime.now().strftime("%Y%m%d")
            output_filename = f"meetara-{model_name}-Q8_0-{timestamp}.gguf"
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
                "--outtype", "q8_0"
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                logger.info(f"✅ Converted {model_name} to GGUF")
                return output_path
            else:
                logger.error(f"❌ Conversion failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error converting {model_name}: {str(e)}")
            return None
    
    def convert_models_for_platform(self, platform: str, models: dict):
        """Convert models for a specific platform (mobile/desktop)"""
        logger.info(f"📱 Converting {platform} models...")
        created_models = {}
        output_dir = self.mobile_dir if platform == "mobile" else self.desktop_dir
        
        for model_name, temp_name in models.items():
            try:
                # Check if temp model exists
                model_dir = self.temp_dir / temp_name
                if not model_dir.exists():
                    logger.warning(f"⚠️ Model directory not found: {model_dir}")
                    continue
                
                # Convert to GGUF
                gguf_path = self.convert_to_gguf(model_dir, model_name, output_dir)
                if gguf_path:
                    created_models[model_name] = str(gguf_path)
                
            except Exception as e:
                logger.error(f"❌ Error converting {platform} model {model_name}: {str(e)}")
        
        return created_models
    
    def create_universal_models(self, mobile_models: dict, desktop_models: dict):
        """Create universal model files"""
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
            "description": "MeeTARA Lab Mobile & Desktop Models - Qwen3 Models",
            "models": {
                "mobile": {
                    "description": "4B parameter models for mobile devices",
                    "model_count": len(mobile_models),
                    "models": mobile_models
                },
                "desktop": {
                    "description": "8B parameter models for desktop applications", 
                    "model_count": len(desktop_models),
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
    logger.info("🚀 Starting Conversion of Downloaded Models to GGUF")
    
    try:
        # Initialize converter
        converter = ConvertDownloadedModels()
        
        # Convert mobile models
        logger.info("📱 Converting mobile models (4B parameters)...")
        mobile_models = converter.convert_models_for_platform("mobile", converter.models["mobile"])
        
        # Convert desktop models
        logger.info("🖥️ Converting desktop models (8B parameters)...")
        desktop_models = converter.convert_models_for_platform("desktop", converter.models["desktop"])
        
        # Create universal models
        converter.create_universal_models(mobile_models, desktop_models)
        
        # Create manifest
        converter.create_manifest(mobile_models, desktop_models)
        
        # Report results
        logger.info("📊 Model Conversion Summary:")
        logger.info(f"Mobile models created: {len(mobile_models)}")
        logger.info(f"Desktop models created: {len(desktop_models)}")
        
        if mobile_models or desktop_models:
            logger.info("🎉 Model conversion completed successfully!")
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
            logger.warning("⚠️ No models were converted successfully")
            
    except Exception as e:
        logger.error(f"❌ Model conversion failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
