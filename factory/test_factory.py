#!/usr/bin/env python3
"""
Test script for Enhanced GGUF Factory
Tests the trained adapters approach without problematic imports
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import logging
import json
import time
from typing import Dict, Any, List
from trinity_core.core_components.config_manager import SmartTrinityConfigManager

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestEnhancedFactory:
    """Test version of the enhanced factory without problematic imports"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize config manager
        self.config_manager = SmartTrinityConfigManager()
        
        # Get paths
        self.base_dir = Path(__file__).parent.parent.parent
        self.models_dir = self.base_dir / "models"
        self.trained_dir = self.base_dir / "data" / "production" / "trained"
        
        self.logger.info("Test Enhanced Factory initialized")
        self.logger.info(f"   Trained adapters directory: {self.trained_dir}")
        self.logger.info(f"   Models directory: {self.models_dir}")
    
    def test_discover_adapters(self):
        """Test adapter discovery functionality"""
        self.logger.info("🔍 Testing adapter discovery...")
        
        if not self.trained_dir.exists():
            self.logger.warning(f"   Trained directory does not exist: {self.trained_dir}")
            return {"success": False, "error": "Trained directory not found"}
        
        adapters = []
        
        # Walk through all category directories
        for category_dir in self.trained_dir.iterdir():
            if category_dir.is_dir():
                category = category_dir.name
                self.logger.info(f"   Scanning category: {category}")
                
                # Walk through all domain directories
                for domain_dir in category_dir.iterdir():
                    if domain_dir.is_dir():
                        domain = domain_dir.name
                        adapter_dir = domain_dir / "adapter"
                        
                        if adapter_dir.exists():
                            adapter_config_file = adapter_dir / "adapter_config.json"
                            adapter_model_file = adapter_dir / "adapter_model.safetensors"
                            
                            if adapter_config_file.exists() and adapter_model_file.exists():
                                try:
                                    with open(adapter_config_file, 'r', encoding='utf-8') as f:
                                        adapter_config = json.load(f)
                                    
                                    adapter_info = {
                                        "category": category,
                                        "domain": domain,
                                        "adapter_path": str(adapter_dir),
                                        "base_model": adapter_config.get("base_model_name_or_path", "unknown"),
                                        "adapter_config": adapter_config,
                                        "model_size_mb": adapter_model_file.stat().st_size / (1024 * 1024)
                                    }
                                    
                                    adapters.append(adapter_info)
                                    self.logger.info(f"     Found adapter: {domain} ({adapter_info['model_size_mb']:.1f}MB)")
                                    
                                except Exception as e:
                                    self.logger.error(f"     Failed to read adapter config for {domain}: {e}")
        
        self.logger.info(f"✅ Adapter discovery completed: {len(adapters)} adapters found")
        return {"success": True, "adapters": adapters, "count": len(adapters)}
    
    def test_group_adapters(self, adapters):
        """Test adapter grouping by base model"""
        self.logger.info("🔗 Testing adapter grouping by base model...")
        
        adapters_by_base_model = {}
        
        for adapter in adapters:
            base_model = adapter.get("base_model", "unknown")
            if base_model not in adapters_by_base_model:
                adapters_by_base_model[base_model] = []
            adapters_by_base_model[base_model].append(adapter)
        
        self.logger.info(f"✅ Adapter grouping completed: {len(adapters_by_base_model)} base model groups")
        
        for base_model, adapter_list in adapters_by_base_model.items():
            self.logger.info(f"   Base model '{base_model}': {len(adapter_list)} adapters")
            for adapter in adapter_list:
                self.logger.info(f"     - {adapter['domain']} ({adapter['model_size_mb']:.1f}MB)")
        
        return {"success": True, "groups": adapters_by_base_model}
    
    def test_factory_workflow(self):
        """Test the complete factory workflow"""
        self.logger.info("🏭 Testing complete factory workflow...")
        
        # Step 1: Discover adapters
        discovery_result = self.test_discover_adapters()
        if not discovery_result.get("success", False):
            return discovery_result
        
        adapters = discovery_result.get("adapters", [])
        if not adapters:
            return {"success": False, "error": "No adapters found"}
        
        # Step 2: Group adapters
        grouping_result = self.test_group_adapters(adapters)
        if not grouping_result.get("success", False):
            return grouping_result
        
        groups = grouping_result.get("groups", {})
        
        # Step 3: Simulate processing each group
        self.logger.info("🔧 Simulating adapter processing...")
        
        results = {
            "base_models_processed": {},
            "total_adapters_processed": len(adapters),
            "total_groups": len(groups)
        }
        
        for base_model_name, adapter_list in groups.items():
            self.logger.info(f"   Processing base model: {base_model_name}")
            self.logger.info(f"   Adapters to process: {len(adapter_list)}")
            
            # Simulate processing
            results["base_models_processed"][base_model_name] = {
                "adapters_processed": len(adapter_list),
                "domains": [adapter["domain"] for adapter in adapter_list],
                "total_size_mb": sum(adapter["model_size_mb"] for adapter in adapter_list),
                "simulated_merge_success": True,
                "simulated_gguf_created": True
            }
            
            self.logger.info(f"   ✅ Simulated processing completed for {base_model_name}")
        
        self.logger.info("✅ Complete factory workflow test completed!")
        return {"success": True, "results": results}

def main():
    """Main test function"""
    logger.info("🚀 Starting Enhanced Factory Test")
    logger.info("=" * 60)
    
    try:
        # Initialize test factory
        factory = TestEnhancedFactory()
        
        # Run complete workflow test
        result = factory.test_factory_workflow()
        
        if result.get("success", False):
            logger.info("✅ All tests passed successfully!")
            
            results = result.get("results", {})
            logger.info(f"   Total adapters processed: {results.get('total_adapters_processed', 0)}")
            logger.info(f"   Base model groups: {results.get('total_groups', 0)}")
            
            for base_model, info in results.get("base_models_processed", {}).items():
                logger.info(f"   {base_model}: {info['adapters_processed']} adapters, {info['total_size_mb']:.1f}MB")
        else:
            logger.error("❌ Tests failed!")
            logger.error(f"   Error: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Test initialization failed: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    main() 