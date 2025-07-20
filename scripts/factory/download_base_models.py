#!/usr/bin/env python3
"""
🏗️ Base Models Downloader for MeeTARA Lab
Downloads real models from HuggingFace and saves to Google Drive for Colab reuse
Enhanced with resume capability, progress tracking, and Drive integration
"""

import logging
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Load base models from config instead of hardcoding
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "trinity_core"))

from trinity_core.core_components.config_manager import SmartTrinityConfigManager

config_manager = SmartTrinityConfigManager()
model_names = config_manager.get_config_dict().get('model_names', {})
if not model_names:
    raise ValueError("❌ No model_names configured")

BASE_MODELS = list(model_names.values())
logger.info(f"✅ Loading {len(BASE_MODELS)} models from config")

# Removed Phi-3 models due to LoRA compatibility issues and Colab import errors
# Removed: "microsoft/Phi-3-medium-4k-instruct",
# Removed: "microsoft/Phi-3-mini-4k-instruct", 
# Removed: "microsoft/Phi-3.5-mini-instruct",


class BaseModelDownloader:
    """Downloads and manages base models with real HuggingFace downloads"""
    
    def __init__(self, drive_path: str = None):
        self.base_dir = Path(__file__).parent.parent.parent
        self.base_models_dir = self.base_dir / "models" / "base_models"
        self.base_models_dir.mkdir(parents=True, exist_ok=True)
        
        # Google Drive path for Colab
        self.drive_path = drive_path or "/content/drive/MyDrive/meetara-lab/models/base_models"
        if drive_path:
            self.drive_models_dir = Path(drive_path)
            self.drive_models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("🏗️ Base Model Downloader initialized (Real Downloads)")
        logger.info(f"📁 Local models directory: {self.base_models_dir}")
        if drive_path:
            logger.info(f"📁 Drive models directory: {self.drive_models_dir}")
    
    def check_model_status(self, model_name: str) -> Tuple[bool, str]:
        """Check if a model is already downloaded and complete"""
        model_dir = self.base_models_dir / model_name.replace('/', '_')
        tokenizer_path = model_dir / "tokenizer.json"
        model_path = model_dir / "pytorch_model.bin"
        
        if not model_dir.exists():
            return False, "not_found"
        
        if not tokenizer_path.exists() or not model_path.exists():
            return False, "incomplete"
        
        return True, "complete"
    
    def download_base_model(self, model_name: str, force_redownload: bool = False) -> bool:
        """Download a single base model from HuggingFace"""
        
        # Check if already downloaded
        is_complete, status = self.check_model_status(model_name)
        if is_complete and not force_redownload:
            logger.info(f"✅ {model_name} already downloaded and complete")
            return True
        
        logger.info(f"📥 Downloading {model_name}...")
        start_time = time.time()
        
        try:
            # Create model directory
            model_dir = self.base_models_dir / model_name.replace('/', '_')
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Download tokenizer
            logger.info(f"   🔧 Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.base_models_dir)
            tokenizer.save_pretrained(model_dir)
            
            # Download model
            logger.info(f"   🧠 Downloading model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                cache_dir=self.base_models_dir,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else None
            )
            model.save_pretrained(model_dir)
            
            # Calculate size
            total_size = 0
            for file_path in model_dir.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            
            download_time = time.time() - start_time
            size_gb = total_size / (1024**3)
            
            logger.info(f"   ✅ Downloaded {model_name}: {size_gb:.2f}GB in {download_time:.1f}s")
            
            # Create metadata
            metadata = {
                "model_name": model_name,
                "download_time": download_time,
                "size_gb": size_gb,
                "status": "complete",
                "download_date": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download {model_name}: {e}")
            return False
    
    def sync_to_drive(self, model_name: str) -> bool:
        """Sync downloaded model to Google Drive"""
        if not hasattr(self, 'drive_models_dir'):
            logger.warning("Drive path not set, skipping sync")
            return False
        
        try:
            local_dir = self.base_models_dir / model_name.replace('/', '_')
            drive_dir = self.drive_models_dir / model_name.replace('/', '_')
            
            if not local_dir.exists():
                logger.error(f"Local model {model_name} not found")
                return False
            
            # Copy to drive
            import shutil
            if drive_dir.exists():
                shutil.rmtree(drive_dir)
            shutil.copytree(local_dir, drive_dir)
            
            logger.info(f"📁 Synced {model_name} to Drive")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to sync {model_name} to Drive: {e}")
            return False
    
    def sync_from_drive(self, model_name: str) -> bool:
        """Sync model from Google Drive to local"""
        if not hasattr(self, 'drive_models_dir'):
            logger.warning("Drive path not set, skipping sync")
            return False
        
        try:
            local_dir = self.base_models_dir / model_name.replace('/', '_')
            drive_dir = self.drive_models_dir / model_name.replace('/', '_')
            
            if not drive_dir.exists():
                logger.error(f"Drive model {model_name} not found")
                return False
            
            # Copy from drive
            import shutil
            if local_dir.exists():
                shutil.rmtree(local_dir)
            shutil.copytree(drive_dir, local_dir)
            
            logger.info(f"📁 Synced {model_name} from Drive")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to sync {model_name} from Drive: {e}")
            return False
    
    def get_download_status(self) -> Dict[str, str]:
        """Get current download status for all models"""
        status = {}
        for model_name in BASE_MODELS:
            is_complete, model_status = self.check_model_status(model_name)
            status[model_name] = "complete" if is_complete else model_status
        return status
    
    def download_all_base_models(self, force_redownload: bool = False, sync_to_drive: bool = True) -> Dict[str, bool]:
        """Download all base models with resume capability"""
        logger.info(f"🚀 Downloading {len(BASE_MODELS)} base models...")
        
        # Show current status
        current_status = self.get_download_status()
        logger.info("📊 Current download status:")
        for model, status in current_status.items():
            logger.info(f"   {model}: {status}")
        
        results = {}
        success_count = 0
        
        for model_name in BASE_MODELS:
            success = self.download_base_model(model_name, force_redownload)
            results[model_name] = success
            if success:
                success_count += 1
                if sync_to_drive:
                    self.sync_to_drive(model_name)
        
        logger.info(f"📊 Downloaded {success_count}/{len(BASE_MODELS)} base models")
        return results
    
    def create_download_manifest(self) -> None:
        """Create manifest for all downloaded models"""
        manifest = {
            "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_models": len(BASE_MODELS),
            "models": {}
        }
        
        total_size_gb = 0
        
        for model_name in BASE_MODELS:
            model_dir = self.base_models_dir / model_name.replace('/', '_')
            metadata_path = model_dir / "metadata.json"
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                manifest["models"][model_name] = metadata
                total_size_gb += metadata.get("size_gb", 0)
            else:
                manifest["models"][model_name] = {"status": "not_downloaded"}
        
        manifest["total_size_gb"] = total_size_gb
        
        manifest_path = self.base_models_dir / "download_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"📋 Created manifest: {total_size_gb:.2f}GB total")

def main():
    """Main function with Colab integration"""
    logger.info("🚀 Starting Base Model Download (Colab Optimized)...")
    logger.info("=" * 50)
    
    try:
        # Check if running in Colab
        drive_path = None
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            drive_path = "/content/drive/MyDrive/meetara-lab/models/base_models"
            logger.info("📁 Google Drive mounted for Colab")
        except ImportError:
            logger.info("🖥️ Running locally (no Google Drive)")
        
        downloader = BaseModelDownloader(drive_path)
        
        # Show what needs to be downloaded
        status = downloader.get_download_status()
        incomplete_models = [model for model, stat in status.items() if stat != "complete"]
        
        if not incomplete_models:
            logger.info("🎉 All base models already downloaded!")
        else:
            logger.info(f"📋 Need to download/resume {len(incomplete_models)} models:")
            for model in incomplete_models:
                logger.info(f"   - {model} ({status[model]})")
        
        # Download all base models (will skip complete ones)
        results = downloader.download_all_base_models(sync_to_drive=bool(drive_path))
        
        # Create manifest
        downloader.create_download_manifest()
        
        success_count = sum(results.values())
        logger.info(f"✅ Download complete: {success_count}/{len(BASE_MODELS)} models")
        
        if drive_path:
            logger.info("💡 Tip: Models are synced to Google Drive for future Colab sessions")
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        raise

if __name__ == "__main__":
    main() 