#!/usr/bin/env python3
"""
MeeTARA Lab - Mobile & Desktop Model Launcher
Quick launcher for creating mobile and desktop models

Usage:
    python scripts/launch_mobile_desktop_models.py [--mobile-only] [--desktop-only] [--all]

Author: MeeTARA Lab Trinity Architecture
Date: September 8, 2025
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.factory.mobile_desktop_model_factory import MobileDesktopModelFactory
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description='MeeTARA Lab Mobile & Desktop Model Factory')
    parser.add_argument('--mobile-only', action='store_true', help='Create only mobile models')
    parser.add_argument('--desktop-only', action='store_true', help='Create only desktop models')
    parser.add_argument('--all', action='store_true', help='Create all models (default)')
    
    args = parser.parse_args()
    
    # Default to all if no specific option is chosen
    if not any([args.mobile_only, args.desktop_only]):
        args.all = True
    
    logger.info("🚀 Starting MeeTARA Lab Mobile & Desktop Model Factory")
    logger.info(f"Mode: {'Mobile Only' if args.mobile_only else 'Desktop Only' if args.desktop_only else 'All Models'}")
    
    try:
        # Initialize factory
        factory = MobileDesktopModelFactory()
        
        mobile_models = {}
        desktop_models = {}
        
        # Create mobile models
        if args.mobile_only or args.all:
            logger.info("📱 Creating mobile models...")
            mobile_models = factory.create_mobile_models()
        
        # Create desktop models
        if args.desktop_only or args.all:
            logger.info("🖥️ Creating desktop models...")
            desktop_models = factory.create_desktop_models()
        
        # Validate all models
        logger.info("🔍 Validating models...")
        all_models = {**mobile_models, **desktop_models}
        validation_results = factory.validate_models(all_models)
        
        # Create manifest
        logger.info("📋 Creating model manifest...")
        manifest_path = factory.create_model_manifest(mobile_models, desktop_models)
        
        # Report results
        logger.info("📊 Model Creation Summary:")
        logger.info(f"Mobile models created: {len(mobile_models)}")
        logger.info(f"Desktop models created: {len(desktop_models)}")
        logger.info(f"Total models: {len(all_models)}")
        logger.info(f"Validation success rate: {sum(validation_results.values())}/{len(validation_results)}")
        logger.info(f"Manifest created: {manifest_path}")
        
        if all(validation_results.values()):
            logger.info("🎉 All models created and validated successfully!")
            logger.info("\n📁 Model Structure Created:")
            logger.info("models/production/")
            logger.info("├── mobile/")
            for model_name, model_path in mobile_models.items():
                logger.info(f"│   ├── {Path(model_path).name}")
            logger.info("├── desktop/")
            for model_name, model_path in desktop_models.items():
                logger.info(f"│   ├── {Path(model_path).name}")
            logger.info("└── speech_models/")
            logger.info("    ├── emotion/")
            logger.info("    ├── voice/")
            logger.info("    ├── routing/")
            logger.info("    └── translation/")
            logger.info("        ├── hi_model/")
            logger.info("        └── te_model/")
        else:
            logger.warning("⚠️ Some models failed validation - check logs for details")
            
    except Exception as e:
        logger.error(f"❌ Factory execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
