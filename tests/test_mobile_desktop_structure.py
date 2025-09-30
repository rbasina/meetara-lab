#!/usr/bin/env python3
"""
MeeTARA Lab - Mobile & Desktop Model Structure Validator
Validates the complete model structure and organization

Author: MeeTARA Lab Trinity Architecture
Date: September 8, 2025
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelStructureValidator:
    """Validates the mobile and desktop model structure"""
    
    def __init__(self):
        """Initialize the validator"""
        self.base_path = Path("models/production")
        self.validation_results = {}
        
    def validate_directory_structure(self) -> Dict[str, bool]:
        """Validate that all required directories exist"""
        logger.info("🔍 Validating directory structure...")
        
        required_dirs = [
            "mobile",
            "desktop"
        ]
        
        # Check services directory structure separately
        services_dirs = [
            "services/routing",
            "services/translation/hi_model", 
            "services/translation/te_model"
        ]
        
        results = {}
        for dir_path in required_dirs:
            full_path = self.base_path / dir_path
            exists = full_path.exists() and full_path.is_dir()
            results[dir_path] = exists
            
            if exists:
                logger.info(f"✅ Directory exists: {dir_path}")
            else:
                logger.error(f"❌ Directory missing: {dir_path}")
        
        self.validation_results["directory_structure"] = results
        return results
    
    def validate_speech_models(self) -> Dict[str, bool]:
        """Validate speech model files"""
        logger.info("🎤 Validating speech models...")
        
        speech_models = {
            "routing/domain_router.pkl": "services/routing/domain_router.pkl",
            "routing/emotion_router.pkl": "services/routing/emotion_router.pkl"
        }
        
        results = {}
        for model_name, file_path in speech_models.items():
            path = Path(file_path)
            exists = path.exists() and path.stat().st_size > 0
            results[model_name] = exists
            
            if exists:
                size_mb = path.stat().st_size / (1024 * 1024)
                logger.info(f"✅ Speech model: {model_name} ({size_mb:.2f} MB)")
            else:
                logger.error(f"❌ Speech model missing: {model_name}")
        
        self.validation_results["speech_models"] = results
        return results
    
    def validate_translation_models(self) -> Dict[str, bool]:
        """Validate translation model files"""
        logger.info("🌐 Validating translation models...")
        
        translation_models = {
            "hi_model/model.pt": "services/translation/hi_model/model.pt",
            "hi_model/tokenizer": "services/translation/hi_model/tokenizer",
            "te_model/model.pt": "services/translation/te_model/model.pt",
            "te_model/tokenizer": "services/translation/te_model/tokenizer"
        }
        
        results = {}
        for model_name, file_path in translation_models.items():
            path = Path(file_path)
            exists = path.exists()
            results[model_name] = exists
            
            if exists:
                if path.is_file():
                    size_mb = path.stat().st_size / (1024 * 1024)
                    logger.info(f"✅ Translation model: {model_name} ({size_mb:.2f} MB)")
                else:
                    logger.info(f"✅ Translation directory: {model_name}")
            else:
                logger.error(f"❌ Translation model missing: {model_name}")
        
        self.validation_results["translation_models"] = results
        return results
    
    def validate_configuration(self) -> Dict[str, bool]:
        """Validate configuration files"""
        logger.info("⚙️ Validating configuration...")
        
        config_files = {
            "trinity_config.yaml": "config/trinity_config.yaml",
            "model_manifest.json": "models/production/model_manifest.json"
        }
        
        results = {}
        for config_name, file_path in config_files.items():
            path = Path(file_path)
            exists = path.exists()
            results[config_name] = exists
            
            if exists:
                logger.info(f"✅ Configuration: {config_name}")
            else:
                logger.warning(f"⚠️ Configuration missing: {config_name}")
        
        self.validation_results["configuration"] = results
        return results
    
    def validate_factory_scripts(self) -> Dict[str, bool]:
        """Validate factory scripts"""
        logger.info("🏭 Validating factory scripts...")
        
        scripts = {
            "download_and_convert_qwen3.py": "factory/download_and_convert_qwen3.py"
        }
        
        results = {}
        for script_name, file_path in scripts.items():
            path = Path(file_path)
            exists = path.exists()
            results[script_name] = exists
            
            if exists:
                logger.info(f"✅ Factory script: {script_name}")
            else:
                logger.error(f"❌ Factory script missing: {script_name}")
        
        self.validation_results["factory_scripts"] = results
        return results
    
    def generate_validation_report(self) -> str:
        """Generate a comprehensive validation report"""
        logger.info("📋 Generating validation report...")
        
        report = {
            "validation_timestamp": str(Path().cwd()),
            "overall_status": "PASS" if all(
                all(category.values()) for category in self.validation_results.values()
            ) else "FAIL",
            "validation_results": self.validation_results,
            "summary": {
                "total_checks": sum(len(category) for category in self.validation_results.values()),
                "passed_checks": sum(
                    sum(1 for result in category.values() if result) 
                    for category in self.validation_results.values()
                ),
                "failed_checks": sum(
                    sum(1 for result in category.values() if not result) 
                    for category in self.validation_results.values()
                )
            }
        }
        
        # Save report
        report_path = Path("validation_report_mobile_desktop.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Validation report saved: {report_path}")
        return str(report_path)
    
    def print_summary(self):
        """Print validation summary"""
        logger.info("📊 Validation Summary:")
        
        total_checks = sum(len(category) for category in self.validation_results.values())
        passed_checks = sum(
            sum(1 for result in category.values() if result) 
            for category in self.validation_results.values()
        )
        failed_checks = total_checks - passed_checks
        
        logger.info(f"Total checks: {total_checks}")
        logger.info(f"Passed: {passed_checks}")
        logger.info(f"Failed: {failed_checks}")
        logger.info(f"Success rate: {(passed_checks/total_checks)*100:.1f}%")
        
        if failed_checks == 0:
            logger.info("🎉 All validations passed! Model structure is complete.")
        else:
            logger.warning(f"⚠️ {failed_checks} validations failed. Check logs for details.")

def main():
    """Main validation function"""
    logger.info("🚀 Starting MeeTARA Lab Mobile & Desktop Model Structure Validation")
    
    try:
        # Initialize validator
        validator = ModelStructureValidator()
        
        # Run all validations
        validator.validate_directory_structure()
        validator.validate_speech_models()
        validator.validate_translation_models()
        validator.validate_configuration()
        validator.validate_factory_scripts()
        
        # Generate report
        report_path = validator.generate_validation_report()
        
        # Print summary
        validator.print_summary()
        
        logger.info(f"✅ Validation complete. Report: {report_path}")
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
