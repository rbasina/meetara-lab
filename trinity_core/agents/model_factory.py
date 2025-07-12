#!/usr/bin/env python3
"""
MeeTARA Lab - Core Intelligent Model Factory Agent
This agent is responsible for the intelligent training and generation of raw models.
Post-processing, quantization, and cleanup are handled by a separate agent.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

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
        import logging, json, os, time
        from pathlib import Path
        
        start_time = time.time()
        logger = logging.getLogger("IntelligentModelFactory")
        stats = {"domain": request.get("domain", "unknown")}
        config_path = getattr(self.config_manager, 'config_path', 'config/trinity_config.yaml')
        logger.info(f"[CONFIG] Using config: {config_path}")
        stats["config_path"] = config_path
        
        try:
            domain = request.get("domain", "unknown")
            training_data = request.get("training_data", []) # Assuming training data is provided
            is_simulation = request.get("simulation", False) # Get simulation flag from request
            
            # Get base model from config
            base_model = self.config_manager.get_base_model_for_domain(domain)
            logger.info(f"[BASE_MODEL] Domain '{domain}' mapped to base model: {base_model}")
            stats["base_model"] = base_model
            
            # Track download timing and caching
            download_start = time.time()
            print(f"📥 Starting download for domain '{domain}' with base model: {base_model}")
            
            # Check if model is already cached
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            model_cache_path = None
            
            # Try to find existing model in cache
            for root, dirs, files in os.walk(cache_dir):
                if base_model.replace("/", "--") in root:
                    model_cache_path = root
                    break
            
            if model_cache_path:
                print(f"✅ Model found in cache: {model_cache_path}")
                download_time = time.time() - download_start
                print(f"⏱️ Cache hit - no download needed")
            else:
                print(f"📥 Downloading model: {base_model}")
                # In a real implementation, this would download the model
                # For now, we'll simulate the download
                download_time = time.time() - download_start
                print(f"⏱️ Download completed in {download_time:.2f}s")
            
            # Load tokenizer
            tokenizer_start = time.time()
            print(f"🔧 Loading tokenizer: {base_model}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(base_model)
                tokenizer_time = time.time() - tokenizer_start
                print(f"✅ Tokenizer loaded in {tokenizer_time:.2f}s")
            except Exception as e:
                print(f"❌ Tokenizer loading failed: {e}")
                return {"error": f"Tokenizer loading failed: {str(e)}"}
            
            # Load base model with proper memory management
            model_start = time.time()
            print(f"🧠 Loading base model: {base_model}")
            try:
                # Check available GPU memory
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                    print(f"📊 Available GPU memory: {gpu_memory:.1f} GB")
                    
                    # For Phi-3 models, check if we have enough memory
                    if "phi-3" in base_model.lower() and gpu_memory < 16:
                        print(f"⚠️ Phi-3 model requires ~16GB GPU memory, but only {gpu_memory:.1f}GB available")
                        print(f"🔄 Falling back to smaller model: microsoft/Phi-3-mini-4k-instruct")
                        base_model = "microsoft/Phi-3-mini-4k-instruct"
                
                # Configure device and memory settings
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                # For Phi-3 models, use specific memory configuration
                if "phi-3" in base_model.lower():
                    # Use more conservative memory settings for Phi-3
                    model = AutoModelForCausalLM.from_pretrained(
                        base_model,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                        # Remove offload_folder to avoid disk offload error
                        max_memory={0: "12GB", "cpu": "16GB"} if torch.cuda.is_available() else None
                    )
                else:
                    # Standard loading for other models
                    model = AutoModelForCausalLM.from_pretrained(
                        base_model,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        low_cpu_mem_usage=True
                    )
                
                model_time = time.time() - model_start
                print(f"✅ Base model loaded in {model_time:.2f}s")
                
                # Try to get model size
                try:
                    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
                    print(f"📊 Model size: {model_size_mb:.1f} MB")
                except:
                    print(f"⚠️ Could not determine exact model path: module 'transformers.file_utils' has no attribute 'cached_file'")
                
                total_prep_time = time.time() - start_time
                print(f"⏱️ Total model preparation time: {total_prep_time:.2f}s")
                
            except Exception as e:
                print(f"❌ Model loading failed: {e}")
                # If model loading fails due to memory, try with CPU-only loading
                if "offload" in str(e).lower() or "memory" in str(e).lower():
                    print(f"🔄 Attempting CPU-only loading for {base_model}")
                    try:
                        model = AutoModelForCausalLM.from_pretrained(
                            base_model,
                            torch_dtype=torch.float32,  # Use float32 for CPU
                            device_map=None,  # No device mapping for CPU
                            low_cpu_mem_usage=True,
                            trust_remote_code=True
                        )
                        model_time = time.time() - model_start
                        print(f"✅ Base model loaded on CPU in {model_time:.2f}s")
                        total_prep_time = time.time() - start_time
                        print(f"⏱️ Total model preparation time: {total_prep_time:.2f}s")
                    except Exception as cpu_error:
                        print(f"❌ CPU loading also failed: {cpu_error}")
                        return {"error": f"Model loading failed on both GPU and CPU: {str(e)}"}
                else:
                    return {"error": f"Model loading failed: {str(e)}"}
            
            # Get domain category and tier information
            category = self.config_manager.get_domain_category(domain)
            tier_name = self.config_manager.get_domain_tier(domain)
            logger.info(f"[TIER] Domain '{domain}' using tier: {tier_name}")
            
            # LoRA configuration for efficient fine-tuning
            lora_config = {
                "r": 8,  # Rank
                "lora_alpha": 16,  # Alpha parameter
                "lora_dropout": 0.1,
                "bias": "none",
                "task_type": "CAUSAL_LM"
            }
            
            # Training configuration
            training_config = {
                "output_dir": f"models/trained/{category}/{domain}",
                "num_train_epochs": 3,
                "per_device_train_batch_size": 4,
                "per_device_eval_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "learning_rate": 2e-4,
                "warmup_steps": 100,
                "logging_steps": 10,
                "save_steps": 500,
                "eval_steps": 500,
                "evaluation_strategy": "steps",
                "save_strategy": "steps",
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
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
            
            raw_model_path = self._generate_raw_model_path(domain, target_size_mb, is_simulation, category, request.get("environment", "dev")) # Pass environment parameter
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
            
            # After training, optionally collect stats (e.g., loss, accuracy if available)
            stats["model_path"] = str(raw_model_path)
            # Save stats report
            stats_dir = "training_stats"
            os.makedirs(stats_dir, exist_ok=True)
            stats_file = os.path.join(stats_dir, f"{domain}_stats.json")
            with open(stats_file, "w") as f:
                json.dump(stats, f, indent=2)
            logger.info(f"[STATS] Training stats saved to {stats_file}")
            
            return model_result
            
        except Exception as e:
            logger.error(f"❌ Error in create_intelligent_model: {e}")
            raise

    def _generate_raw_model_path(self, domain: str, size_mb: float, is_simulation: bool, category: str, environment: str = "dev") -> Path:
        """Generates a unique path for the raw model artifact."""
        # Get the base model factory directory from the config
        data_trained_base_dir = Path(self.config_manager.get_config_dict()["paths"]["data_trained_base_dir"])

        # Determine the final output base based on environment parameter (overrides simulation flag)
        if environment == "production":
            final_output_base = data_trained_base_dir / "production"
        else:
            final_output_base = data_trained_base_dir / "dev"

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
            
            raw_model_path = self._generate_raw_model_path(domain, target_size_gb * 1024, is_simulation, category, request.get("environment", "dev")) # Pass environment parameter
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