#!/usr/bin/env python3
"""
🎯 MeeTARA Service Bundle Generator
Enhanced approach for backend-to-frontend integration

PURPOSE:
- Creates unified service bundle for MeeTARA frontend repository
- Packages GGUF models + Speech + Translation + Routing in single deployment
- Generates frontend-ready configuration and API endpoints

ARCHITECTURE:
Backend (meetara-lab) → Service Bundle → Frontend (meetara repo)

Author: MeeTARA Lab Trinity Architecture
Date: September 9, 2025
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MeeTARAServiceBundleGenerator:
    """
    Generates unified service bundle for MeeTARA frontend integration
    
    Bundle Structure:
    meetara_service_bundle_YYYYMMDD/
    ├── models/
    │   ├── meetara_mobile_universal.gguf      # Mobile GGUF
    │   └── meetara_desktop_universal.gguf     # Desktop GGUF
    ├── services/
    │   ├── emotion/                           # Emotion detection
    │   ├── speech/                            # Voice synthesis
    │   ├── translation/                       # Multi-language
    │   └── routing/                           # Intelligent routing
    ├── config/
    │   ├── service_config.json               # Service configuration
    │   ├── model_mapping.json                # Model-domain mapping
    │   └── api_endpoints.json                # Frontend API endpoints
    ├── deployment/
    │   ├── install.py                        # Auto-installation script
    │   └── README.md                         # Integration guide
    └── bundle_manifest.json                  # Complete bundle info
    """
    
    def __init__(self, target_meetara_repo: Optional[str] = None):
        """
        Initialize the bundle generator
        
        Args:
            target_meetara_repo: Path to MeeTARA frontend repository (optional)
        """
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.bundle_name = f"meetara_service_bundle_{self.timestamp}"
        
        # Source paths (current meetara-lab structure)
        self.source_models = Path("models/production")
        self.source_services = Path("services")
        self.source_config = Path("config")
        
        # Target bundle structure
        self.bundle_root = Path("deployment/bundles") / self.bundle_name
        self.bundle_models = self.bundle_root / "models"
        self.bundle_services = self.bundle_root / "services"
        self.bundle_config = self.bundle_root / "config"
        self.bundle_deployment = self.bundle_root / "deployment"
        
        # Optional target MeeTARA repository
        self.target_meetara_repo = Path(target_meetara_repo) if target_meetara_repo else None
        
        logger.info(f"🎯 Initialized MeeTARA Service Bundle Generator: {self.bundle_name}")
    
    def _load_trinity_config(self) -> dict:
        """Load trinity configuration from YAML file"""
        try:
            import yaml
            config_path = Path("config/trinity_config.yaml")
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            else:
                logger.warning("⚠️ trinity_config.yaml not found, using fallback configuration")
                return self._get_fallback_config()
        except Exception as e:
            logger.error(f"❌ Failed to load trinity_config.yaml: {e}")
            return self._get_fallback_config()
    
    def _get_fallback_config(self) -> dict:
        """Fallback configuration if trinity_config.yaml is not available"""
        return {
            "domain_config": {
                "healthcare": {
                    "category_tier": "premium",
                    "domains": {
                        "general_health": "Qwen/Qwen3-4B-Thinking-2507",
                        "mental_health": "Qwen/Qwen3-4B-Instruct-2507",
                        "nutrition": "Qwen/Qwen3-4B-Thinking-2507",
                        "sleep": "Qwen/Qwen3-4B-Instruct-2507",
                        "stress_management": "Qwen/Qwen3-4B-Thinking-2507"
                    }
                },
                "business": {
                    "category_tier": "expert",
                    "domains": {
                        "entrepreneurship": "Qwen/Qwen3-4B-Thinking-2507",
                        "marketing": "Qwen/Qwen3-4B-Thinking-2507",
                        "sales": "Qwen/Qwen3-4B-Thinking-2507",
                        "customer_service": "Qwen/Qwen3-4B-Thinking-2507",
                        "project_management": "Qwen/Qwen3-4B-Thinking-2507"
                    }
                }
            }
        }
    
    def generate_bundle(self) -> Path:
        """Generate complete service bundle"""
        logger.info("🚀 Generating MeeTARA Service Bundle...")
        
        # Create bundle structure
        self._create_bundle_structure()
        
        # Copy models
        self._bundle_models()
        
        # Bundle services
        self._bundle_services()
        
        # Generate configurations
        self._generate_configurations()
        
        # Create deployment scripts
        self._create_deployment_scripts()
        
        # Generate manifest
        self._generate_bundle_manifest()
        
        # Optional: Deploy to MeeTARA repo
        if self.target_meetara_repo:
            self._deploy_to_meetara_repo()
        
        logger.info(f"✅ Service bundle generated: {self.bundle_root}")
        return self.bundle_root
    
    def _create_bundle_structure(self):
        """Create bundle directory structure"""
        logger.info("📁 Creating bundle structure...")
        
        directories = [
            self.bundle_models,
            self.bundle_services / "emotion",
            self.bundle_services / "voice",
            self.bundle_services / "translation",
            self.bundle_services / "routing",
            self.bundle_config,
            self.bundle_deployment
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ Bundle structure created")
    
    def _bundle_models(self):
        """Bundle GGUF models for frontend"""
        logger.info("🤖 Bundling GGUF models...")
        
        # Find latest universal models
        mobile_models = list(self.source_models.glob("mobile/meetara_mobile_universal*.gguf"))
        desktop_models = list(self.source_models.glob("desktop/meetara_desktop_universal*.gguf"))
        
        if mobile_models:
            latest_mobile = max(mobile_models, key=lambda p: p.stat().st_mtime)
            shutil.copy2(latest_mobile, self.bundle_models / "meetara_mobile_universal.gguf")
            logger.info(f"✅ Bundled mobile model: {latest_mobile.name}")
        
        if desktop_models:
            latest_desktop = max(desktop_models, key=lambda p: p.stat().st_mtime)
            shutil.copy2(latest_desktop, self.bundle_models / "meetara_desktop_universal.gguf")
            logger.info(f"✅ Bundled desktop model: {latest_desktop.name}")
    
    def _bundle_services(self):
        """Bundle all service models"""
        logger.info("🎤 Bundling service models...")
        
        # Copy all service directories with correct structure
        service_mappings = {
            "emotion": "speech/emotion",
            "voice": "speech/voice", 
            "routing": "speech/routing",
            "translation": "translation"
        }
        
        for service_name, source_path in service_mappings.items():
            full_source_path = self.source_services / source_path
            if full_source_path.exists():
                target_path = self.bundle_services / service_name
                if full_source_path.is_dir():
                    shutil.copytree(full_source_path, target_path, dirs_exist_ok=True)
                else:
                    shutil.copy2(full_source_path, target_path)
                logger.info(f"✅ Bundled service: {service_name}")
        
        # Copy service configuration files from speech directory
        config_files = ["speech_config.json", "trinity_enhancements.json"]
        for config_file in config_files:
            source_file = self.source_services / "speech" / config_file
            if source_file.exists():
                shutil.copy2(source_file, self.bundle_services / config_file)
                logger.info(f"✅ Bundled config: {config_file}")
    
    def _generate_configurations(self):
        """Generate frontend-ready configurations"""
        logger.info("⚙️ Generating configurations...")
        
        # Load trinity configuration to get actual domains and categories
        trinity_config = self._load_trinity_config()
        
        # Extract all domains and categories from trinity_config
        all_domains = {}
        all_categories = []
        category_tiers = {"premium": [], "expert": [], "quality": [], "specialized": []}
        
        for category, config in trinity_config.get("domain_config", {}).items():
            all_categories.append(category)
            domains = config.get("domains", {})
            domain_list = list(domains.keys())
            all_domains[category] = domain_list
            
            # Determine tier based on category_tier
            tier = config.get("category_tier", "quality")
            if tier in category_tiers:
                category_tiers[tier].append(category)
        
        # Count total domains
        total_domains = sum(len(domains) for domains in all_domains.values())
        
        # Service configuration for frontend
        service_config = {
            "version": "1.0.0",
            "bundle_id": self.bundle_name,
            "created_at": datetime.now().isoformat(),
            "models": {
                "mobile": {
                    "path": "models/meetara_mobile_universal.gguf",
                    "type": "GGUF",
                    "size_mb": self._get_file_size_mb("models/meetara_mobile_universal.gguf"),
                    "domains": f"{total_domains}+ emotional intelligence domains"
                },
                "desktop": {
                    "path": "models/meetara_desktop_universal.gguf", 
                    "type": "GGUF",
                    "size_mb": self._get_file_size_mb("models/meetara_desktop_universal.gguf"),
                    "domains": f"{total_domains}+ emotional intelligence domains"
                }
            },
            "services": {
                "emotion_detection": {
                    "models": ["services/emotion/rms_model.pkl", "services/emotion/ser_model.pkl"],
                    "api_endpoint": "/api/v1/emotion/detect"
                },
                "voice_synthesis": {
                    "models": "services/voice/",
                    "categories": all_categories,
                    "api_endpoint": "/api/v1/voice/synthesize"
                },
                "translation": {
                    "supported_languages": ["hi", "te", "ta", "gu", "bn", "ml", "mr", "pa", "as", "si", "kn", "ur", "ar", "de", "es", "fr", "ja", "ko", "zh"],
                    "models": "services/translation/",
                    "shared_nllb_model": "services/translation/shared_nllb_model/",
                    "api_endpoint": "/api/v1/translate",
                    "optimization": "99.1% storage reduction with shared NLLB model"
                },
                "intelligent_routing": {
                    "models": ["services/routing/domain_router.pkl", "services/routing/emotion_router.pkl"],
                    "api_endpoint": "/api/v1/route"
                }
            }
        }
        
        # Model-domain mapping with complete domain coverage
        model_mapping = {
            "domains": all_domains,
            "routing_logic": {
                "primary": "domain_detection",
                "fallback": "emotion_based_routing",
                "confidence_threshold": 0.85
            },
            "category_tiers": category_tiers,
            "total_domains": total_domains,
            "total_categories": len(all_categories)
        }
        
        # API endpoints for frontend integration
        api_endpoints = {
            "base_url": "/api/v1",
            "endpoints": {
                "chat": "/chat",
                "emotion": "/emotion/detect",
                "voice": "/voice/synthesize", 
                "translate": "/translate",
                "route": "/route",
                "health": "/health"
            },
            "authentication": {
                "required": False,
                "type": "local_only"
            }
        }
        
        # Save configurations
        configs = {
            "service_config.json": service_config,
            "model_mapping.json": model_mapping,
            "api_endpoints.json": api_endpoints
        }
        
        for filename, config_data in configs.items():
            config_path = self.bundle_config / filename
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Generated: {filename}")
    
    def _create_deployment_scripts(self):
        """Create deployment scripts for MeeTARA frontend"""
        logger.info("🚀 Creating deployment scripts...")
        
        # Auto-installation script
        install_script = '''#!/usr/bin/env python3
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
'''
        
        # README for integration
        readme_content = f'''# MeeTARA Service Bundle Integration Guide

## 📦 Bundle: {self.bundle_name}

This bundle contains all necessary components for MeeTARA frontend integration:

### 🤖 Models Included:
- **Mobile Universal Model**: Optimized for mobile devices (4B parameters)
- **Desktop Universal Model**: Full-featured for desktop applications (8B parameters)
- **62+ Emotional Intelligence Domains**: Complete domain coverage

### 🎤 Services Included:
- **Emotion Detection**: Real-time RMS and SER emotion analysis models
- **Voice Synthesis**: 15+ category-specific voice profiles (healthcare, business, creative, etc.)
- **Translation**: 19 languages with shared NLLB model (99.1% storage optimization)
  - **Indian Languages**: Hindi, Telugu, Tamil, Gujarati, Bengali, Malayalam, Marathi, Punjabi, Assamese, Sinhala, Kannada, Urdu
  - **International**: Arabic, German, Spanish, French, Japanese, Korean, Chinese
- **Intelligent Routing**: Domain and emotion-based routing with confidence scoring

### 🚀 Quick Installation:

1. **Automatic Installation**:
```bash
python deployment/install.py
```

2. **Manual Installation**:
```bash
# Copy to MeeTARA frontend repository
cp -r models/ /path/to/meetara/models/
cp -r services/ /path/to/meetara/services/
cp -r config/ /path/to/meetara/config/services/
```

### 🔧 Configuration:
- Service config: `config/service_config.json`
- Model mapping: `config/model_mapping.json`
- API endpoints: `config/api_endpoints.json`

### 📱 Frontend Integration:
1. Import service configurations
2. Initialize model loaders
3. Set up API endpoints
4. Configure routing logic

### 🎯 Usage in MeeTARA Frontend:
```javascript
// Initialize MeeTARA services
import {{ MeeTARAServices }} from './services/meetara-services';

const meetara = new MeeTARAServices({{
    modelPath: './models/meetara_mobile_universal.gguf',
    servicesPath: './services/',
    configPath: './config/services/'
}});

// Use emotional intelligence
const response = await meetara.chat("I'm feeling stressed about work");
const emotion = await meetara.detectEmotion(response);
const voice = await meetara.synthesizeVoice(response, 'healthcare');
```

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
'''
        
        # Save deployment files
        install_path = self.bundle_deployment / "install.py"
        readme_path = self.bundle_deployment / "README.md"
        
        with open(install_path, 'w', encoding='utf-8') as f:
            f.write(install_script)
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        # Make install script executable
        install_path.chmod(0o755)
        
        logger.info("✅ Deployment scripts created")
    
    def _generate_bundle_manifest(self):
        """Generate complete bundle manifest"""
        logger.info("📋 Generating bundle manifest...")
        
        manifest = {
            "bundle_info": {
                "name": self.bundle_name,
                "version": "1.0.0",
                "created_at": datetime.now().isoformat(),
                "description": "Complete MeeTARA Service Bundle for Frontend Integration"
            },
            "contents": {
                "models": self._get_directory_contents("models"),
                "services": self._get_directory_contents("services"),
                "config": self._get_directory_contents("config"),
                "deployment": self._get_directory_contents("deployment")
            },
            "integration": {
                "target": "MeeTARA Frontend Repository",
                "method": "Auto-installation or manual copy",
                "requirements": ["Python 3.8+", "Node.js 16+", "MeeTARA Frontend"]
            },
            "capabilities": {
                "emotional_intelligence": "62+ domains",
                "voice_synthesis": "15+ category-specific voice profiles",
                "translation": "19 languages (11 Indian + 8 international) with shared NLLB model",
                "emotion_detection": "Real-time RMS and SER models",
                "intelligent_routing": "Domain + emotion based routing",
                "storage_optimization": "99.1% reduction with shared NLLB model"
            }
        }
        
        manifest_path = self.bundle_root / "bundle_manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Bundle manifest generated: bundle_manifest.json")
    
    def _deploy_to_meetara_repo(self):
        """Deploy bundle directly to MeeTARA repository"""
        logger.info(f"🎯 Deploying to MeeTARA repository: {self.target_meetara_repo}")
        
        if not self.target_meetara_repo.exists():
            logger.error(f"❌ MeeTARA repository not found: {self.target_meetara_repo}")
            return
        
        # Create services and models directories in frontend
        frontend_services = self.target_meetara_repo / "services"
        frontend_models = self.target_meetara_repo / "models"
        frontend_config = self.target_meetara_repo / "config" / "services"
        
        frontend_services.mkdir(exist_ok=True)
        frontend_models.mkdir(exist_ok=True)
        frontend_config.mkdir(parents=True, exist_ok=True)
        
        # Copy bundle contents
        shutil.copytree(self.bundle_models, frontend_models, dirs_exist_ok=True)
        shutil.copytree(self.bundle_services, frontend_services, dirs_exist_ok=True)
        shutil.copytree(self.bundle_config, frontend_config, dirs_exist_ok=True)
        
        logger.info("✅ Bundle deployed to MeeTARA repository")
    
    def _get_file_size_mb(self, relative_path: str) -> float:
        """Get file size in MB"""
        file_path = self.bundle_root / relative_path
        if file_path.exists():
            return round(file_path.stat().st_size / (1024 * 1024), 2)
        return 0.0
    
    def _get_directory_contents(self, dir_name: str) -> List[str]:
        """Get directory contents for manifest"""
        dir_path = self.bundle_root / dir_name
        if not dir_path.exists():
            return []
        
        contents = []
        for item in dir_path.rglob("*"):
            if item.is_file():
                relative_path = item.relative_to(dir_path)
                contents.append(str(relative_path))
        
        return sorted(contents)

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate MeeTARA Service Bundle")
    parser.add_argument("--target-repo", help="Path to MeeTARA frontend repository")
    parser.add_argument("--deploy", action="store_true", help="Deploy directly to frontend repo")
    
    args = parser.parse_args()
    
    target_repo = args.target_repo if args.deploy else None
    
    generator = MeeTARAServiceBundleGenerator(target_repo)
    bundle_path = generator.generate_bundle()
    
    print(f"\n🎉 MeeTARA Service Bundle Generated Successfully!")
    print(f"📦 Bundle Location: {bundle_path}")
    print(f"📋 See deployment/README.md for integration instructions")

if __name__ == "__main__":
    main()
