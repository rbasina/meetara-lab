#!/usr/bin/env python3
"""
MeeTARA Lab - Qwen3 Model Downloader and Converter
Downloads Qwen3 base models and converts them to GGUF format for mobile/desktop

This script:
1. Downloads Qwen3-4B and Qwen3-8B models (Thinking and Instruct variants)
2. Converts them to GGUF format using llama.cpp with optimized intermediate formats:
   - Mobile models (4B): f16 → IQ4_XS (smaller intermediate, no disk space issues)
   - Desktop models (8B): q8_0 → IQ4_XS (avoids f16 disk space issues for large models)
3. Organizes them into mobile/desktop structure

Author: MeeTARA Lab Trinity Architecture
Date: September 11, 2025
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
        
        # Resolve Hugging Face cache directory
        hf_cache_env = os.getenv("HF_HUB_CACHE")
        if hf_cache_env:
            self.hf_cache_dir = Path(hf_cache_env)
        else:
            hf_home = os.getenv("HF_HOME")
            if hf_home:
                self.hf_cache_dir = Path(hf_home) / "hub"
            else:
                self.hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        
        # Model specifications
        self.mobile_models = {
            "Qwen3-4B-Thinking-2507": "Qwen/Qwen3-4B-Thinking-2507",
            "Qwen3-4B-Instruct-2507": "Qwen/Qwen3-4B-Instruct-2507"
        }
        
        self.desktop_models = {
            "Qwen3-8B": "Qwen/Qwen3-8B"  # Instruction-tuned version - perfect for emotional intelligence
        }
        
        # Ensure directories exist
        self.mobile_dir.mkdir(parents=True, exist_ok=True)
        self.desktop_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Qwen3ModelDownloader initialized")
    
    def download_model(self, model_name: str, model_id: str) -> Path:
        """Download a model to the HF cache and return local snapshot path"""
        logger.info(f"📥 Downloading {model_name}...")
        
        try:
            # Use huggingface-hub to download
            from huggingface_hub import snapshot_download
            
            # Ensure cache directory exists
            self.hf_cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Download to cache; returns snapshot path
            snapshot_path = snapshot_download(
                repo_id=model_id,
                cache_dir=str(self.hf_cache_dir),
                local_dir=None,
                local_dir_use_symlinks=False
            )
            
            logger.info(f"✅ Downloaded {model_name} to cache: {snapshot_path}")
            return Path(snapshot_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to download {model_name}: {str(e)}")
            return None
    
    def convert_to_gguf(self, model_dir: Path, model_name: str, output_dir: Path) -> Path:
        """Convert a model to GGUF format using llama.cpp"""
        logger.info(f"🔄 Converting {model_name} to GGUF...")
        
        try:
            # Determine if this is a desktop model (8B) or mobile model (4B)
            is_desktop_model = "8B" in model_name
            
            # Generate output filename (will be updated after quantization)
            timestamp = datetime.now().strftime("%Y%m%d")
            if is_desktop_model:
                # Use q8_0 intermediate for desktop models (8B) to avoid disk space issues
                intermediate_type = "q8_0"
                output_filename = f"meetara-{model_name}-{intermediate_type}-{timestamp}.gguf"
            else:
                # Use f16 intermediate for mobile models (4B) - smaller, no disk space issues
                intermediate_type = "f16"
                output_filename = f"meetara-{model_name}-{intermediate_type}-{timestamp}.gguf"
            
            output_path = output_dir / output_filename
            
            # Find the convert script (use absolute path, no cwd to avoid double path issues)
            convert_script = Path("llama.cpp") / "convert_hf_to_gguf.py"
            if not convert_script.exists():
                logger.error(f"❌ Convert script not found: {convert_script}")
                return None
            
            # Run the conversion
            cmd = [
                "python", str(convert_script),
                str(model_dir),
                "--outfile", str(output_path),
                "--outtype", intermediate_type  # q8_0 for desktop, f16 for mobile
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✅ Converted {model_name} to GGUF ({intermediate_type})")
                
                # Now quantize to IQ4_XS using llama.cpp quantize tool
                quantized_path = self._quantize_to_iq4_xs(output_path, model_name)
                if quantized_path:
                    # Remove the intermediate file
                    output_path.unlink()
                    return quantized_path
                else:
                    logger.warning(f"⚠️ Quantization to IQ4_XS failed, keeping {intermediate_type} version")
                    return output_path
            else:
                logger.error(f"❌ Conversion failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error converting {model_name}: {str(e)}")
            return None
    
    def _quantize_to_iq4_xs(self, input_path: Path, model_name: str) -> Path:
        """Quantize GGUF file to IQ4_XS using llama.cpp quantize tool
        Supports both f16 (mobile) and q8_0 (desktop) intermediate formats"""
        try:
            # Create quantized filename
            timestamp = datetime.now().strftime("%Y%m%d")
            quantized_filename = f"meetara-{model_name}-IQ4_XS-{timestamp}.gguf"
            quantized_path = input_path.parent / quantized_filename
            
            # Find the quantize tool (Windows executable)
            quantize_tool = Path("llama.cpp") / "build" / "bin" / "quantize.exe"
            if not quantize_tool.exists():
                logger.error(f"❌ Quantize tool not found: {quantize_tool}")
                return None
            
            # Run quantization
            cmd = [
                str(quantize_tool),
                "--allow-requantize",
                str(input_path),
                str(quantized_path),
                "IQ4_XS"
            ]
            
            logger.info(f"🔄 Quantizing to IQ4_XS: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✅ Quantized to IQ4_XS: {quantized_path.name}")
                return quantized_path
            else:
                logger.error(f"❌ Quantization failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error quantizing {model_name}: {str(e)}")
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
                
                # Keep HF cache; do not delete
                
            except Exception as e:
                logger.error(f"❌ Error creating mobile model {model_name}: {str(e)}")
        
        return created_models
    
    def _get_fallback_desktop_model(self) -> str:
        """Get fallback desktop model id from config or use Llama-3-8B-Instruct."""
        try:
            import yaml  # already a dependency above
            config_path = Path("config/trinity_config.yaml")
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    cfg = yaml.safe_load(f) or {}
                names = (cfg.get('model_names') or {})
                return names.get('llama3_8b_instruct', 'meta-llama/Llama-3-8B-Instruct')
        except Exception as e:
            logger.warning(f"⚠️ Could not load fallback from config: {e}")
        return 'meta-llama/Llama-3-8B-Instruct'

    def create_desktop_models(self):
        """Create desktop models (8B parameters)"""
        logger.info("🖥️ Creating desktop models...")
        created_models = {}
        
        for model_name, model_id in self.desktop_models.items():
            try:
                # Download model
                model_dir = self.download_model(model_name, model_id)
                if not model_dir:
                    # Fallback to a compatible desktop model if Qwen3 8B repo not found
                    fallback_id = self._get_fallback_desktop_model()
                    fallback_name = fallback_id.split('/')[-1]
                    logger.warning(f"⚠️ Falling back to {fallback_id} for {model_name}")
                    model_dir = self.download_model(fallback_name, fallback_id)
                    if not model_dir:
                        continue
                    # Override output model_name to reflect the actual downloaded model
                    model_name = fallback_name
                
                # Convert to GGUF
                gguf_path = self.convert_to_gguf(model_dir, model_name, self.desktop_dir)
                if gguf_path:
                    created_models[model_name] = str(gguf_path)
                
                # Keep HF cache; do not delete
                
            except Exception as e:
                logger.error(f"❌ Error creating desktop model {model_name}: {str(e)}")
        
        return created_models
    
    def create_universal_models(self, mobile_models: dict, desktop_models: dict):
        """Create universal model files with intelligent model selection"""
        logger.info("🌐 Creating universal model files...")
        
        timestamp = datetime.now().strftime("%Y%m%d")
        
        # Mobile universal models - create separate universals for each model type
        if mobile_models:
            for model_name, model_path in mobile_models.items():
                if "Thinking" in model_name:
                    mobile_universal = self.mobile_dir / f"meetara_mobile_thinking_universal-IQ4_XS-{timestamp}.gguf"
                    shutil.copy2(model_path, mobile_universal)
                    logger.info(f"✅ Created mobile Thinking universal model: {mobile_universal.name}")
                elif "Instruct" in model_name:
                    mobile_universal = self.mobile_dir / f"meetara_mobile_instruct_universal-IQ4_XS-{timestamp}.gguf"
                    shutil.copy2(model_path, mobile_universal)
                    logger.info(f"✅ Created mobile Instruct universal model: {mobile_universal.name}")
                else:
                    # Generic mobile universal for any other models
                    mobile_universal = self.mobile_dir / f"meetara_mobile_universal-{model_name}-IQ4_XS-{timestamp}.gguf"
                    shutil.copy2(model_path, mobile_universal)
                    logger.info(f"✅ Created mobile universal model: {mobile_universal.name}")
        
        # Desktop universal
        desktop_universal = self.desktop_dir / f"meetara_desktop_universal-model-IQ4_XS-{timestamp}.gguf"
        if desktop_models:
            # Copy the first desktop model as universal
            first_model = list(desktop_models.values())[0]
            shutil.copy2(first_model, desktop_universal)
            logger.info(f"✅ Created desktop universal model: {desktop_universal.name}")
    
    def create_manifest(self, mobile_models: dict, desktop_models: dict):
        """Create a manifest file"""
        timestamp = datetime.now().strftime("%Y%m%d")
        
        # Find universal models
        mobile_universals = {}
        for file in self.mobile_dir.glob(f"*universal*{timestamp}*"):
            if "thinking" in file.name.lower():
                mobile_universals["thinking_universal"] = str(file)
            elif "instruct" in file.name.lower():
                mobile_universals["instruct_universal"] = str(file)
            else:
                mobile_universals["universal"] = str(file)
        
        desktop_universals = {}
        for file in self.desktop_dir.glob(f"*universal*{timestamp}*"):
            desktop_universals["universal"] = str(file)
        
        manifest = {
            "created_at": datetime.now().isoformat(),
            "description": "MeeTARA Lab Mobile & Desktop Models - Qwen3 Base Models",
            "models": {
                "mobile": {
                    "description": "4B parameter models for mobile devices",
                    "individual_models": mobile_models,
                    "universal_models": mobile_universals,
                    "usage": {
                        "thinking_universal": "Use for complex reasoning, problem-solving, and analytical tasks",
                        "instruct_universal": "Use for instruction-following, conversation, and general chat tasks"
                    }
                },
                "desktop": {
                    "description": "8B parameter models for desktop applications",
                    "individual_models": desktop_models,
                    "universal_models": desktop_universals,
                    "usage": {
                        "universal": "High-performance model for all desktop tasks"
                    }
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
