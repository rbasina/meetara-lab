#!/usr/bin/env python3
"""
MeeTARA Lab - Mobile & Desktop Model Generator
Generates mobile (4B) and desktop (8B) models based on domain configuration

This script:
1. Reads the mobile/desktop domain mappings from trinity_config.yaml
2. Downloads the appropriate Qwen3 base models
3. Converts them to GGUF format
4. Organizes them into mobile/desktop structure

Author: MeeTARA Lab Trinity Architecture
Date: September 8, 2025
"""

import os
import sys
import yaml
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

class MobileDesktopModelGenerator:
    """Generates mobile and desktop models based on domain configuration"""
    
    def __init__(self, config_path="config/trinity_config.yaml"):
        """Initialize the generator with configuration"""
        self.config_path = config_path
        self.config = self.load_config()
        
        # Setup directories
        self.base_dir = Path("models/production")
        self.mobile_dir = self.base_dir / "mobile"
        self.desktop_dir = self.base_dir / "desktop"
        
        # Ensure directories exist
        self.mobile_dir.mkdir(parents=True, exist_ok=True)
        self.desktop_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("MobileDesktopModelGenerator initialized")
    
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"❌ Failed to load configuration: {str(e)}")
            return None
    
    def get_domain_mappings(self):
        """Extract mobile and desktop domain mappings from config"""
        mobile_domains = {}
        desktop_domains = {}
        
        if not self.config or 'domain_config' not in self.config:
            logger.error("❌ No domain_config found in configuration")
            return mobile_domains, desktop_domains
        
        for category, category_config in self.config['domain_config'].items():
            if 'mobile_models' in category_config:
                for domain, model in category_config['mobile_models'].items():
                    mobile_domains[domain] = {
                        'model': model,
                        'category': category,
                        'tier': category_config.get('category_tier', 'quality')
                    }
            
            if 'desktop_models' in category_config:
                for domain, model in category_config['desktop_models'].items():
                    desktop_domains[domain] = {
                        'model': model,
                        'category': category,
                        'tier': category_config.get('category_tier', 'quality')
                    }
        
        logger.info(f"📱 Found {len(mobile_domains)} mobile domains")
        logger.info(f"🖥️ Found {len(desktop_domains)} desktop domains")
        
        return mobile_domains, desktop_domains
    
    def download_model(self, model_name: str, model_id: str) -> Path:
        """Download a model from Hugging Face"""
        logger.info(f"📥 Downloading {model_name}...")
        
        try:
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
    
    def generate_mobile_models(self, mobile_domains: dict):
        """Generate mobile models for all mobile domains"""
        logger.info("📱 Generating mobile models...")
        
        # Group domains by model to avoid duplicate downloads
        model_groups = {}
        for domain, config in mobile_domains.items():
            model = config['model']
            if model not in model_groups:
                model_groups[model] = []
            model_groups[model].append(domain)
        
        created_models = {}
        
        for model_id, domains in model_groups.items():
            try:
                # Extract model name from model_id
                model_name = model_id.split('/')[-1]
                
                # Download model
                model_dir = self.download_model(model_name, model_id)
                if not model_dir:
                    continue
                
                # Convert to GGUF for each domain
                for domain in domains:
                    domain_config = mobile_domains[domain]
                    gguf_path = self.convert_to_gguf(model_dir, f"{domain}-{model_name}", self.mobile_dir)
                    if gguf_path:
                        created_models[domain] = str(gguf_path)
                
                # Cleanup temp directory
                shutil.rmtree(model_dir, ignore_errors=True)
                
            except Exception as e:
                logger.error(f"❌ Error creating mobile models for {model_id}: {str(e)}")
        
        return created_models
    
    def generate_desktop_models(self, desktop_domains: dict):
        """Generate desktop models for all desktop domains"""
        logger.info("🖥️ Generating desktop models...")
        
        # Group domains by model to avoid duplicate downloads
        model_groups = {}
        for domain, config in desktop_domains.items():
            model = config['model']
            if model not in model_groups:
                model_groups[model] = []
            model_groups[model].append(domain)
        
        created_models = {}
        
        for model_id, domains in model_groups.items():
            try:
                # Extract model name from model_id
                model_name = model_id.split('/')[-1]
                
                # Download model
                model_dir = self.download_model(model_name, model_id)
                if not model_dir:
                    continue
                
                # Convert to GGUF for each domain
                for domain in domains:
                    domain_config = desktop_domains[domain]
                    gguf_path = self.convert_to_gguf(model_dir, f"{domain}-{model_name}", self.desktop_dir)
                    if gguf_path:
                        created_models[domain] = str(gguf_path)
                
                # Cleanup temp directory
                shutil.rmtree(model_dir, ignore_errors=True)
                
            except Exception as e:
                logger.error(f"❌ Error creating desktop models for {model_id}: {str(e)}")
        
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
            "description": "MeeTARA Lab Mobile & Desktop Models - Domain-Specific Qwen3 Models",
            "strategy": "Intelligent model allocation based on complexity and device constraints",
            "models": {
                "mobile": {
                    "description": "4B parameter models for mobile devices - fast, efficient, battery-friendly",
                    "model_count": len(mobile_models),
                    "models": mobile_models
                },
                "desktop": {
                    "description": "8B parameter models for desktop applications - complex reasoning, detailed analysis",
                    "model_count": len(desktop_models),
                    "models": desktop_models
                }
            }
        }
        
        manifest_path = self.base_dir / "mobile_desktop_manifest.json"
        import json
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Manifest created: {manifest_path}")

def main():
    """Main execution function"""
    logger.info("🚀 Starting Mobile & Desktop Model Generation")
    
    try:
        # Initialize generator
        generator = MobileDesktopModelGenerator()
        
        if not generator.config:
            logger.error("❌ Failed to load configuration")
            sys.exit(1)
        
        # Get domain mappings
        mobile_domains, desktop_domains = generator.get_domain_mappings()
        
        if not mobile_domains and not desktop_domains:
            logger.warning("⚠️ No mobile or desktop domains found in configuration")
            return
        
        # Generate mobile models
        mobile_models = {}
        if mobile_domains:
            logger.info("📱 Generating mobile models (4B parameters)...")
            mobile_models = generator.generate_mobile_models(mobile_domains)
        
        # Generate desktop models
        desktop_models = {}
        if desktop_domains:
            logger.info("🖥️ Generating desktop models (8B parameters)...")
            desktop_models = generator.generate_desktop_models(desktop_domains)
        
        # Create universal models
        generator.create_universal_models(mobile_models, desktop_models)
        
        # Create manifest
        generator.create_manifest(mobile_models, desktop_models)
        
        # Report results
        logger.info("📊 Model Generation Summary:")
        logger.info(f"Mobile models created: {len(mobile_models)}")
        logger.info(f"Desktop models created: {len(desktop_models)}")
        
        if mobile_models or desktop_models:
            logger.info("🎉 Model generation completed successfully!")
            logger.info("\n📁 Model Structure:")
            logger.info("models/production/")
            logger.info("├── mobile/")
            for domain, model_path in mobile_models.items():
                logger.info(f"│   ├── {Path(model_path).name}")
            logger.info("├── desktop/")
            for domain, model_path in desktop_models.items():
                logger.info(f"│   ├── {Path(model_path).name}")
            logger.info("└── speech_models/ (already exists)")
        else:
            logger.warning("⚠️ No models were created successfully")
            
    except Exception as e:
        logger.error(f"❌ Model generation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
