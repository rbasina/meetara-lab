#!/usr/bin/env python3
"""
🚀 Enhanced GGUF Factory - Current Trained Adapters Approach
Reads trained adapters from data/production/trained/
Groups by base model and merges adapters with base model
Creates GGUF files from merged models
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import logging
import json
import gc
import os
import tempfile
import time
import random
from typing import Dict, Any, List
import asyncio
from trinity_core.agents.model_factory import IntelligentModelFactory
from trinity_core.agents.speech_models_factory import SpeechModelsFactory
from trinity_core.agents.translation_factory import TranslationFactory
from trinity_core.core_components.config_manager import SmartTrinityConfigManager

# Setup detailed logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('gguf_factory.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class EnhancedGGUFFactory:
    """Current approach: Merge trained adapters with base model to create GGUF files"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize config manager (source of all configuration)
        self.config_manager = SmartTrinityConfigManager()
        
        # Initialize Trinity Super Agents
        self.model_factory = IntelligentModelFactory()
        self.speech_factory = SpeechModelsFactory()
        self.translation_factory = TranslationFactory()
        
        # Get configuration-driven paths
        self.base_dir = Path(__file__).parent.parent.parent
        self.models_dir = self.base_dir / "models"
        self.trained_dir = self.base_dir / "data" / "production" / "trained"
        self.temp_dir = Path(tempfile.mkdtemp(prefix="gguf_factory_"))
        
        # Load all configuration from config files
        self._load_orchestration_config()
        
        self.logger.info("Enhanced GGUF Factory initialized (Current Trained Adapters Approach)")
        self.logger.info(f"   Models directory: {self.models_dir}")
        self.logger.info(f"   Trained adapters directory: {self.trained_dir}")
        self.logger.info(f"   Temporary directory: {self.temp_dir}")
        self.logger.info(f"   Configuration-driven: All values from config files")
        self.logger.info(f"   Agent delegation: Model Factory, Speech Factory, Translation Factory")
    
    def _load_orchestration_config(self):
        """Load all orchestration configuration from config files"""
        try:
            # Load orchestration-specific config
            orchestration_config_path = self.base_dir / "config" / "orchestration-config.json"
            
            if orchestration_config_path.exists():
                with open(orchestration_config_path, 'r', encoding='utf-8') as f:
                    orchestration_config = json.load(f)
                
                # Extract orchestration settings
                self.orchestration_config = orchestration_config.get("orchestration", {})
                self.speech_models_config = orchestration_config.get("speech_models", {})
                self.garbage_collection_config = orchestration_config.get("garbage_collection", {})
                self.manifest_config = orchestration_config.get("manifest", {})
                self.logging_config = orchestration_config.get("logging", {})
                self.paths_config = orchestration_config.get("paths", {})
                self.performance_config = orchestration_config.get("performance", {})
                self.error_handling_config = orchestration_config.get("error_handling", {})
                self.config_references = orchestration_config.get("config_references", {})
                
                # Load referenced configurations
                self._load_referenced_configs()
                
                self.logger.info(f"   Loaded orchestration config: {orchestration_config_path}")
            else:
                # Fallback to trinity-config.json
                json_config = self.config_manager.json_config
                
                self.orchestration_config = {
                    "enabled": True,
                    "agent_delegation": {
                        "model_factory": {"enabled": True},
                        "speech_factory": {"enabled": True},
                        "translation_factory": {"enabled": True}
                    }
                }
                self.speech_models_config = {"enabled": True}
                self.garbage_collection_config = {"enabled": True}
                self.manifest_config = {"enabled": True}
                self.logging_config = {"level": "INFO"}
                self.paths_config = {"models_dir": "models"}
                self.performance_config = {"timeout_seconds": 7200}
                self.error_handling_config = {"continue_on_error": False}
                self.config_references = {}
                
                self.logger.info(f"   Using fallback config from trinity-config.json")
            
            # Load domain categories for model creation
            self.domain_categories = self.config_manager.get_all_domain_categories()
            
            self.logger.info(f"   Orchestration enabled: {self.orchestration_config.get('enabled', True)}")
            self.logger.info(f"   Domain categories: {len(self.domain_categories)} categories")
            
        except Exception as e:
            self.logger.error(f"Failed to load orchestration config: {e}")
            # Use minimal fallback
            self.orchestration_config = self._get_minimal_orchestration_config()
            self.domain_categories = {"healthcare": ["general_health"]}
    
    def _load_referenced_configs(self):
        """Load configurations referenced by orchestration config"""
        try:
            # Load model variants from trinity-config.json
            json_config = self.config_manager.json_config
            self.model_variants_config = json_config.get("universal_model_architecture", {})
            
            # Load translation config from translation_config.json
            translation_config_path = self.base_dir / "config" / "translation_config.json"
            if translation_config_path.exists():
                with open(translation_config_path, 'r', encoding='utf-8') as f:
                    translation_config = json.load(f)
                self.translation_config = translation_config.get("translation_config", {})
            else:
                self.translation_config = {"enabled": True, "supported_languages": ["hi", "te"]}
            
            self.logger.info(f"   Loaded referenced configs: model_variants, translation")
            
        except Exception as e:
            self.logger.error(f"Failed to load referenced configs: {e}")
            # Fallback configurations
            self.model_variants_config = {}
            self.translation_config = {"enabled": True, "supported_languages": ["hi", "te"]}
    
    def _get_minimal_orchestration_config(self) -> Dict[str, Any]:
        """Minimal fallback configuration"""
        return {
            "model_variants": {
                "A_universal_full": {"enabled": True},
                "B_universal_lite": {"enabled": True},
                "C_category_specific": {"enabled": True}
            },
            "speech_models": {"enabled": True},
            "translation": {"enabled": True},
            "garbage_collection": {"enabled": True},
            "manifest": {"enabled": True}
        }
    
    def __del__(self):
        """Cleanup temporary directory"""
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            (getattr(self, "logger", logger)).info(f"Cleaned up temporary directory: {self.temp_dir}")
    
    def create_all_models(self) -> Dict[str, Any]:
        """Orchestrate creation of all models using trained adapters approach"""
        self.logger.info("Starting comprehensive model creation (Trained Adapters Approach)...")
        self.logger.info("=" * 80)
        
        results = {
            "model_variants": {},
            "speech_models": {},
            "translation": {},
            "garbage_collection": {},
            "success": False
        }
        
        try:
            # Step 1: Process trained adapters and create GGUF files
            self.logger.info("STEP 1: Processing trained adapters and creating GGUF files...")
            results["model_variants"] = self._process_trained_adapters()
            self.logger.info("STEP 1: Trained adapters processing completed")
            
            # Step 2: Create speech models (delegate to SpeechModelsFactory)
            self.logger.info("STEP 2: Creating speech models (SpeechModelsFactory)...")
            results["speech_models"] = self._orchestrate_speech_models()
            self.logger.info("STEP 2: Speech models completed")
            
            # Step 3: Create translation models (delegate to TranslationFactory)
            self.logger.info("STEP 3: Creating translation models (TranslationFactory)...")
            results["translation"] = self._orchestrate_translation_models()
            self.logger.info("STEP 3: Translation models completed")
            
            # Step 4: Garbage collection (orchestrate cleanup)
            self.logger.info("STEP 4: Running garbage collection...")
            results["garbage_collection"] = self._orchestrate_garbage_collection()
            self.logger.info("STEP 4: Garbage collection completed")
            
            # Step 5: Create comprehensive manifest (orchestrate documentation)
            self.logger.info("STEP 5: Creating comprehensive manifest...")
            self._orchestrate_comprehensive_manifest(results)
            self.logger.info("STEP 5: Comprehensive manifest created")
            
            results["success"] = True
            self.logger.info("ALL STEPS COMPLETED SUCCESSFULLY!")
            
        except Exception as e:
            self.logger.error(f"Model creation failed at step: {e}")
            results["error"] = str(e)
        
        return results
    
    def _process_trained_adapters(self) -> Dict[str, Any]:
        """Process trained adapters and create GGUF files from merged models"""
        self.logger.info("🏭 Processing trained adapters and creating GGUF files...")
        
        results = {
            "base_models_processed": {},
            "gguf_files_created": {},
            "total_adapters_processed": 0,
            "total_gguf_files_created": 0
        }
        
        try:
            # Step 1: Discover all trained adapters
            trained_adapters = self._discover_trained_adapters()
            self.logger.info(f"   Discovered {len(trained_adapters)} trained adapters")
            
            # Step 2: Group adapters by base model
            adapters_by_base_model = self._group_adapters_by_base_model(trained_adapters)
            self.logger.info(f"   Grouped into {len(adapters_by_base_model)} base model groups")
            
            # Step 3: Process each base model group
            for base_model_name, adapters in adapters_by_base_model.items():
                self.logger.info(f"   Processing base model: {base_model_name}")
                self.logger.info(f"   Adapters to merge: {len(adapters)}")
                
                # Create request for merging adapters with base model
                request = self._create_adapter_merge_request(base_model_name, adapters)
                
                # Delegate to IntelligentModelFactory for merging
                merge_result = self.model_factory.merge_adapters_with_base_model(request)
                
                if merge_result.get("success", False):
                    # Create GGUF file from merged model
                    gguf_result = self._create_gguf_from_merged_model(base_model_name, merge_result)
                    
                    results["base_models_processed"][base_model_name] = {
                        "adapters_merged": len(adapters),
                        "merge_success": True,
                        "gguf_created": gguf_result.get("success", False),
                        "gguf_file": gguf_result.get("gguf_file", None),
                        "gguf_size_mb": gguf_result.get("size_mb", 0)
                    }
                    
                    if gguf_result.get("success", False):
                        results["gguf_files_created"][base_model_name] = gguf_result
                        results["total_gguf_files_created"] += 1
                    
                    results["total_adapters_processed"] += len(adapters)
                else:
                    self.logger.error(f"   Failed to merge adapters for base model: {base_model_name}")
                    results["base_models_processed"][base_model_name] = {
                        "adapters_merged": 0,
                        "merge_success": False,
                        "error": merge_result.get("error", "Unknown error")
                    }
            
            self.logger.info(f"✅ Trained adapters processing completed:")
            self.logger.info(f"   Total adapters processed: {results['total_adapters_processed']}")
            self.logger.info(f"   Total GGUF files created: {results['total_gguf_files_created']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Trained adapters processing failed: {e}")
            return {"error": str(e)}
    
    def _discover_trained_adapters(self) -> List[Dict[str, Any]]:
        """Discover all trained adapters in data/production/trained/"""
        adapters = []
        
        if not self.trained_dir.exists():
            self.logger.warning(f"   Trained directory does not exist: {self.trained_dir}")
            return adapters
        
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
                                # Read adapter config to get base model info
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
        
        return adapters
    
    def _group_adapters_by_base_model(self, adapters: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group adapters by their base model"""
        adapters_by_base_model = {}
        
        for adapter in adapters:
            base_model = adapter.get("base_model", "unknown")
            if base_model not in adapters_by_base_model:
                adapters_by_base_model[base_model] = []
            adapters_by_base_model[base_model].append(adapter)
        
        return adapters_by_base_model
    
    def _create_adapter_merge_request(self, base_model_name: str, adapters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create request for merging adapters with base model"""
        
        # Get base model configuration from config
        base_model_config = self.config_manager._global_params.get('fallback_base_model', base_model_name)
        
        request = {
            "base_model_name": base_model_name,
            "base_model_path": base_model_config,
            "adapters": adapters,
            "merge_strategy": "weighted_average",  # or "sequential", "parallel"
            "quantization_type": "Q4_K_M",
            "target_size_mb": 8.3,
            "quality_targets": {
                "minimum_quality": 0.70,
                "target_accuracy": 99.99,
                "validation_required": True,
                "llama_cpp_compatibility": True
            },
            "trinity_enhancements": {
                "contextual_intelligence": True,
                "emotion_context_learning": True,
                "crisis_intervention": True,
                "professional_boundaries": True,
                "dynamic_ratio_optimization": True
            }
        }
        
        return request
    
    def _create_gguf_from_merged_model(self, base_model_name: str, merge_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create GGUF file from merged model"""
        
        try:
            # Get merged model path from merge result
            merged_model_path = merge_result.get("merged_model_path")
            if not merged_model_path:
                return {"success": False, "error": "No merged model path provided"}
            
            # Create GGUF creation request
            request = {
                "model_path": merged_model_path,
                "output_path": str(self.models_dir / "production" / f"{base_model_name}_merged.gguf"),
                "quantization_type": "Q4_K_M",
                "target_size_mb": 8.3,
                "llama_cpp_compatibility": True
            }
            
            # Delegate to IntelligentModelFactory for GGUF creation
            gguf_result = self.model_factory.create_gguf_from_model(request)
            
            if gguf_result.get("success", False):
                gguf_file = gguf_result.get("gguf_file")
                size_mb = gguf_result.get("size_mb", 0)
                
                self.logger.info(f"   ✅ GGUF created: {gguf_file} ({size_mb:.1f}MB)")
                
                return {
                    "success": True,
                    "gguf_file": gguf_file,
                    "size_mb": size_mb,
                    "base_model": base_model_name
                }
            else:
                self.logger.error(f"   ❌ GGUF creation failed: {gguf_result.get('error', 'Unknown error')}")
                return {"success": False, "error": gguf_result.get("error", "Unknown error")}
                
        except Exception as e:
            self.logger.error(f"   ❌ GGUF creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _orchestrate_speech_models(self) -> Dict[str, Any]:
        """Orchestrate speech models creation using SpeechModelsFactory"""
        speech_config = self.speech_models_config
        
        if not speech_config.get("enabled", True):
            self.logger.info("   Speech models disabled in config")
            return {"success": False, "reason": "disabled_in_config"}
        
        self.logger.info("   Creating speech models...")
        
        # Create request for speech models
        request = {
            "domain": "universal",
            "category": "all",
            "create_all_voices": True,
            "trinity_enhanced": True,
            "tara_compatible": True,
            "config": speech_config
        }
        
        # Delegate to SpeechModelsFactory
        try:
            result = asyncio.run(self.speech_factory.create_speech_models(request))
            self.logger.info(f"   Speech models completed: {result.get('success', False)}")
            return result
        except Exception as e:
            self.logger.error(f"   Speech models failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _orchestrate_translation_models(self) -> Dict[str, Any]:
        """Orchestrate translation models creation using TranslationFactory"""
        translation_config = self.translation_config
        
        if not translation_config.get("enabled", True):
            self.logger.info("   Translation models disabled in config")
            return {"success": False, "reason": "disabled_in_config"}
        
        self.logger.info("   Creating translation models...")
        
        # Get supported languages from config
        supported_languages = translation_config.get("supported_languages", ["hi", "te"])
        # Filter out languages not present in the translation factory's supported_languages
        valid_languages = [lang for lang in supported_languages if lang in self.translation_factory.supported_languages]

        # Delegate to TranslationFactory
        try:
            result = self.translation_factory.create_translation_bundle(
                languages=valid_languages,
                quantization_type=translation_config.get("quantization_type", "Q4_K_M")
            )
            self.logger.info(f"   Translation models completed: {result.get('success', False)}")
            return result
        except Exception as e:
            self.logger.error(f"   Translation models failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _orchestrate_garbage_collection(self) -> Dict[str, Any]:
        """Orchestrate garbage collection"""
        gc_config = self.garbage_collection_config
        
        if not gc_config.get("enabled", True):
            self.logger.info("   Garbage collection disabled in config")
            return {"success": False, "reason": "disabled_in_config"}
        
        self.logger.info("   Running garbage collection...")
        
        try:
            # Force garbage collection
            collected = gc.collect()
            
            # Get memory stats
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            result = {
                "success": True,
                "collected_objects": collected,
                "memory_usage_mb": memory_info.rss / (1024 * 1024),
                "memory_percent": process.memory_percent()
            }
            
            self.logger.info(f"   Garbage collection completed: {collected} objects collected")
            return result
            
        except Exception as e:
            self.logger.error(f"   Garbage collection failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _orchestrate_comprehensive_manifest(self, results: Dict[str, Any]) -> None:
        """Orchestrate comprehensive manifest creation"""
        manifest_config = self.manifest_config
        
        if not manifest_config.get("enabled", True):
            self.logger.info("   Manifest creation disabled in config")
            return
        
        self.logger.info("   Creating comprehensive manifest...")
        
        try:
            # Create manifest data
            manifest_data = {
                "creation_timestamp": time.time(),
                "creation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "orchestration_config": self.orchestration_config,
                "results": results,
                "domain_categories": self.domain_categories,
                "agent_contributions": {
                    "model_factory": "IntelligentModelFactory",
                    "speech_factory": "SpeechModelsFactory", 
                    "translation_factory": "TranslationFactory"
                },
                "approach": "trained_adapters_merge",
                "trained_adapters_source": str(self.trained_dir)
            }
            
            # Save manifest using paths config
            manifest_file = self.paths_config.get("manifest_file", "models/comprehensive_manifest.json")
            manifest_path = self.base_dir / manifest_file
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            
            output_format = manifest_config.get("output_format", "json")
            output_encoding = manifest_config.get("output_encoding", "utf-8")
            
            with open(manifest_path, 'w', encoding=output_encoding) as f:
                json.dump(manifest_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"   Comprehensive manifest created: {manifest_path}")
            
        except Exception as e:
            self.logger.error(f"   Manifest creation failed: {e}")

class WorkingEnhancedFactory:
    """
    An older factory script for testing.
    Refactored to work with the modern config manager.
    """
    def __init__(self):
        self.config_manager = SmartTrinityConfigManager()
        self.all_domains = self.config_manager.get_all_domains_flat()
        self.domains_by_category = self._get_domains_by_category()

    def _get_domains_by_category(self):
        """Gets a dictionary of available domains grouped by category."""
        domains_by_category = {}
        for domain, details in self.all_domains.items():
            category = details.get('category', 'unknown')
            if category not in domains_by_category:
                domains_by_category[category] = []
            domains_by_category[category].append(domain)
        return domains_by_category

    def run_factory_simulation(self):
        """Simulates the factory process."""
        print("🏭 Running Enhanced Factory Simulation...")
        for category, domains in self.domains_by_category.items():
            print(f"\nProcessing category: {category.upper()}")
            for domain in domains:
                params = self.config_manager.get_tara_proven_params(domain)
                print(f"  -> Simulating build for '{domain}' with model {params.get('base_model')}...")
                time.sleep(0.05)
        print("\n✅ Factory simulation complete.")

def main():
    """Main orchestration function"""
    logger.info("🚀 Starting Enhanced GGUF Factory (Trained Adapters Approach)")
    logger.info("=" * 80)
    
    try:
        # Initialize factory
        factory = EnhancedGGUFFactory()
        
        # Create all models
        results = factory.create_all_models()
        
        # Report results
        if results.get("success", False):
            logger.info("✅ All models created successfully!")
            logger.info(f"   Base models processed: {len(results.get('model_variants', {}).get('base_models_processed', {}))}")
            logger.info(f"   GGUF files created: {results.get('model_variants', {}).get('total_gguf_files_created', 0)}")
            logger.info(f"   Speech models: {results.get('speech_models', {}).get('success', False)}")
            logger.info(f"   Translation models: {results.get('translation', {}).get('success', False)}")
            logger.info(f"   Garbage collection: {results.get('garbage_collection', {}).get('success', False)}")
        else:
            logger.error("❌ Model creation failed!")
            logger.error(f"   Error: {results.get('error', 'Unknown error')}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Factory initialization failed: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    main()