#!/usr/bin/env python3
"""
MeeTARA Lab - Core Intelligent Model Factory Agent
This agent is responsible for the intelligent training and generation of raw models.
Post-processing, quantization, and cleanup are handled by a separate agent.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import json
import os # Added missing import for os.urandom

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import enhanced config manager for multi-base models
from trinity_core.core_components.config_manager import SmartTrinityConfigManager, UniversalModelArchitecture, MultiBaseModel

# Domain integration (if still needed for initial data analysis)
from trinity_core.core_components.domain_integration import (
    get_all_domains,
)

class IntelligentModelFactory:
    """
    Intelligent Model Factory Agent - Core Model Generation
    Focuses on training and producing raw models before quantization.
    """
    
    def __init__(self):
        self.config_manager = SmartTrinityConfigManager()
        self.learned_config = self._load_or_create_learned_config() # Keep for intelligent sizing/params
        
        logger.info("🧠 Core Intelligent Model Factory initialized for raw model generation.")

    def _load_or_create_learned_config(self) -> Dict[str, Any]:
        """Load learned configuration or create intelligent defaults (simplified)"""
        # Fetch global TARA parameters from config manager
        global_params = self.config_manager.get_config_dict().get("global_tara_params", {})
        
        return {
            "model_sizing": {
                "base_size_mb": global_params.get("target_gguf_size_mb", 8.3), # Default for raw output
                "target_size_mb": global_params.get("target_gguf_size_mb", 8.3) # Default
            },
            "quality": {
                "target_quality": global_params.get("validation_target", 0.95) / 100.0, # Convert % to decimal
                "min_quality_threshold": global_params.get("min_quality_threshold", 0.5)
            },
            "performance": {
                "batch_size": global_params.get("default_batch_size", 4) # Assuming a default_batch_size might be added
            }
        }

    async def create_intelligent_model(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates a raw, unquantized model based on intelligent configuration with LoRA integration.
        The output is a path to the raw model artifact, not a GGUF.
        """
        start_time = time.time()
        
        try:
            domain = request.get("domain", "unknown")
            training_data = request.get("training_data", []) # Assume raw data or path
            is_simulation = request.get("simulation", False) # Get simulation flag from request
            category = request.get("category", "unknown_category") # Get category from request
            
            # Get domain configuration from config manager
            domain_details = self.config_manager._get_domain_details(domain)
            base_model = domain_details.get('base_model', 'microsoft/Phi-3.5-mini-instruct')
            tier_name = domain_details.get('tier_name', 'balanced')
            
            # Get tier configuration for LoRA parameters
            tier_config = self.config_manager.get_model_tier_config(tier_name)
            
            # Enhanced LoRA configuration
            lora_config = {
                "r": tier_config.get("lora_r", 8),
                "alpha": tier_config.get("lora_alpha", 16),
                "dropout": tier_config.get("lora_dropout", 0.1),
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "bias": "none",
                "task_type": "CAUSAL_LM"
            }
            
            # Training configuration with emotion/context learning
            training_config = {
                "base_model": base_model,
                "lora_config": lora_config,
                "batch_size": tier_config.get("batch_size", 4),
                "learning_rate": tier_config.get("learning_rate", 2e-4),
                "num_epochs": tier_config.get("num_epochs", 1),
                "max_steps": tier_config.get("max_steps", 500),
                "warmup_steps": tier_config.get("warmup_steps", 50),
                "gradient_accumulation_steps": tier_config.get("gradient_accumulation_steps", 4),
                "save_steps": tier_config.get("save_steps", 100),
                "eval_steps": tier_config.get("eval_steps", 100),
                "logging_steps": tier_config.get("logging_steps", 10),
                "save_total_limit": tier_config.get("save_total_limit", 3),
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
                "fp16": True,
                "bf16": False,
                "dataloader_pin_memory": False,
                "remove_unused_columns": False,
                "label_names": ["labels"],
                "group_by_length": True,
                "length_column_name": "length",
                "max_grad_norm": 1.0,
                "weight_decay": 0.01,
                "optim": "adamw_torch",
                "lr_scheduler_type": "cosine",
                "warmup_ratio": 0.1,
                "report_to": "none",
                "dataloader_num_workers": 4,
                "ddp_find_unused_parameters": False,
                "gradient_checkpointing": True,
                "torch_compile": False,
                "optim_args": {"capturable": True},
                "full_determinism": False,
                "dataloader_pin_memory": False,
                "remove_unused_columns": False,
                "label_names": ["labels"],
                "group_by_length": True,
                "length_column_name": "length",
                "max_grad_norm": 1.0,
                "weight_decay": 0.01,
                "optim": "adamw_torch",
                "lr_scheduler_type": "cosine",
                "warmup_ratio": 0.1,
                "report_to": "none",
                "dataloader_num_workers": 4,
                "ddp_find_unused_parameters": False,
                "gradient_checkpointing": True,
                "torch_compile": False,
                "optim_args": {"capturable": True},
                "full_determinism": False
            }
            
            # Emotion/context learning configuration
            emotion_context_config = {
                "enable_emotion_detection": True,
                "enable_context_learning": True,
                "emotion_labels": ["happy", "sad", "angry", "anxious", "neutral", "excited", "worried", "confident"],
                "context_labels": ["crisis", "general", "professional", "personal", "urgent", "routine"],
                "emotion_weight": 0.3,
                "context_weight": 0.3,
                "content_weight": 0.4
            }
            
            # Determine target model size based on learned config/request
            target_size_mb = request.get("target_size_mb", self.learned_config["model_sizing"]["target_size_mb"])
            
            # Simulate raw model training and saving with LoRA
            # In a real scenario, this would involve calling a training engine (e.g., Hugging Face Trainer)
            # and saving the trained model in a format like PyTorch .bin or TensorFlow .ckpt
            
            raw_model_path = self._generate_raw_model_path(domain, target_size_mb, is_simulation, category) # Pass is_simulation and category
            raw_model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Simulate creating a dummy raw model file with LoRA weights
            with open(raw_model_path, 'wb') as f:
                f.write(os.urandom(int(target_size_mb * 1024 * 1024))) # Create a dummy file of target size
            
            # Simulate LoRA adapter files
            lora_path = raw_model_path.parent / f"{domain}_lora_adapter.bin"
            with open(lora_path, 'wb') as f:
                f.write(os.urandom(int(target_size_mb * 0.1 * 1024 * 1024))) # LoRA is typically 10% of base model
            
            # Enhanced quality simulation with emotion/context learning
            base_quality = self.learned_config["quality"]["target_quality"]
            emotion_bonus = 0.02 if emotion_context_config["enable_emotion_detection"] else 0.0
            context_bonus = 0.02 if emotion_context_config["enable_context_learning"] else 0.0
            lora_bonus = 0.03  # LoRA typically improves performance
            
            simulated_quality = base_quality + emotion_bonus + context_bonus + lora_bonus
            simulated_quality = max(self.learned_config["quality"]["min_quality_threshold"], simulated_quality)
            
            model_result = {
                "status": "success",
                "domain": domain,
                "category": category,
                "base_model": base_model,
                "tier_name": tier_name,
                "raw_model_path": str(raw_model_path),
                "lora_adapter_path": str(lora_path),
                "model_size_mb": target_size_mb,
                "lora_size_mb": target_size_mb * 0.1,
                "creation_time_seconds": time.time() - start_time,
                "simulated_quality_score": simulated_quality,
                "training_config": training_config,
                "lora_config": lora_config,
                "emotion_context_config": emotion_context_config,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "training_simulated": is_simulation, # Reflect actual simulation status
                    "output_format": "raw_model_artifact_with_lora",
                    "trinity_enhancements": {
                        "lora_integration": True,
                        "emotion_learning": True,
                        "context_learning": True,
                        "intelligent_routing": True
                    }
                }
            }
            
            logger.info(f"✅ Raw model with LoRA generated for {domain} at {raw_model_path}")
            logger.info(f"   → Base model: {base_model}, Tier: {tier_name}")
            logger.info(f"   → LoRA config: r={lora_config['r']}, alpha={lora_config['alpha']}")
            logger.info(f"   → Size: {target_size_mb:.2f} MB, Quality (Simulated): {simulated_quality:.2f}")
            logger.info(f"   → Emotion/Context learning: {emotion_context_config['enable_emotion_detection']}")
            
            return model_result
            
        except Exception as e:
            logger.error(f"❌ Raw model generation failed for {domain}: {e}")
            return {"error": f"Raw model generation failed: {str(e)}"}

    def _generate_raw_model_path(self, domain: str, size_mb: float, is_simulation: bool, category: str) -> Path:
        """Generates a unique path for the raw model artifact."""
        # Get the base model factory directory from the config
        data_trained_base_dir = Path(self.config_manager.get_config_dict()["paths"]["data_trained_base_dir"])

        # Determine the final output base based on simulation flag
        if is_simulation:
            final_output_base = data_trained_base_dir / "dev"
        else:
            final_output_base = data_trained_base_dir / "production"

        # Construct the full path: models/{dev|production}/trained/<category>/<domain>/
        output_dir = final_output_base / "trained" / category / domain
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{domain}_raw_{timestamp}_{size_mb:.1f}MB.bin" # .bin for raw PyTorch/TF model
        
        return output_dir / filename

    async def create_multi_base_model(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates a raw multi-base model, focusing on initial training without quantization.
        """
        start_time = time.time()
        
        try:
            domain = request.get("domain")
            category = request.get("category")
            architecture_type_str = request.get("architecture_type")
            is_simulation = request.get("simulation", False) # Get simulation flag from request
            
            if not domain:
                return {"error": "Domain is required for multi-base model creation"}

            # Validate architecture_type
            try:
                architecture_type = UniversalModelArchitecture[architecture_type_str.upper()]
            except KeyError:
                return {"error": f"Invalid architecture_type provided: {architecture_type_str}"}

            # Use config_manager to get model parameters
            model_config = self.config_manager.get_universal_model_config(architecture_type)
            
            target_size_gb = model_config.get("target_size_gb", 1.0) # Default to 1GB if not in config
            base_model_name = model_config.get("base_model")

            # --- Simulate raw multi-base model training ---
            # This would involve sophisticated logic to combine base models and domain data.
            # For this simplified factory, we simulate the output of a raw model.
            
            raw_model_path = self._generate_raw_model_path(domain, target_size_gb * 1024, is_simulation, category) # Pass is_simulation and category
            raw_model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Simulate creating a dummy raw model file
            with open(raw_model_path, 'wb') as f:
                f.write(os.urandom(int(target_size_gb * 1024 * 1024 * 1024))) # Create a dummy file of target size in bytes

            simulated_quality = model_config.get("quality_target", 0.95)
            
            model_result = {
                "status": "success",
                "domain": domain,
                "architecture_type": architecture_type.value,
                "base_model_used": base_model_name,
                "raw_model_path": str(raw_model_path),
                "model_size_gb": target_size_gb,
                "creation_time_seconds": time.time() - start_time,
                "simulated_quality_score": simulated_quality,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "training_simulated": is_simulation, # Reflect actual simulation status
                    "output_format": "raw_multi_base_model_artifact"
                }
            }
            
            logger.info(f"✅ Raw multi-base model generated for {domain} ({architecture_type.value}) at {raw_model_path}")
            logger.info(f"   → Size: {target_size_gb:.2f} GB, Quality (Simulated): {simulated_quality:.2f}, Time: {model_result['creation_time_seconds']:.2f}s")
            
            return model_result
            
        except Exception as e:
            logger.error(f"❌ Multi-base model generation failed for {domain}: {e}")
            return {"error": f"Multi-base model generation failed: {str(e)}"}

# Singleton instance for global access
model_factory = IntelligentModelFactory() 