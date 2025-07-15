#!/usr/bin/env python3
"""
🚀 QLoRA & LoRA Manager for MeeTARA Lab
Centralized management of QLoRA and LoRA integration with configuration-driven approach
"""

import logging
import torch
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class QLoRAManager:
    """Centralized QLoRA and LoRA management for MeeTARA Lab"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.qlora_config = config_manager.get_config_dict().get('qlora_config', {})
        self.model_type_map = config_manager.get_config_dict().get('model_type_map', {})
        
        logger.info("🚀 QLoRA Manager initialized")
    
    def detect_gpu_capabilities(self) -> Dict[str, Any]:
        """Detect GPU capabilities for QLoRA/LoRA"""
        capabilities = {
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "qlora_supported": False,
            "lora_supported": False,
            "recommended_method": "none"
        }
        
        if not capabilities["cuda_available"]:
            logger.info("⚠️ CUDA not available - LoRA/QLoRA not supported")
            return capabilities
        
        # Check GPU memory
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        capabilities["gpu_memory_gb"] = gpu_memory_gb
        
        min_memory = self.qlora_config.get('min_gpu_memory_gb', 8)
        recommended_memory = self.qlora_config.get('recommended_gpu_memory_gb', 16)
        
        if gpu_memory_gb >= recommended_memory:
            capabilities["qlora_supported"] = True
            capabilities["lora_supported"] = True
            capabilities["recommended_method"] = "qlora"
            logger.info(f"✅ GPU memory {gpu_memory_gb:.1f}GB - QLoRA recommended")
        elif gpu_memory_gb >= min_memory:
            capabilities["lora_supported"] = True
            capabilities["recommended_method"] = "lora"
            logger.info(f"✅ GPU memory {gpu_memory_gb:.1f}GB - LoRA supported")
        else:
            logger.warning(f"⚠️ GPU memory {gpu_memory_gb:.1f}GB insufficient for LoRA")
        
        return capabilities
    
    def get_model_specific_settings(self, base_model: str) -> Dict[str, Any]:
        """Get model-specific QLoRA/LoRA settings"""
        # Determine model type
        model_type = None
        for model_name, config in self.model_type_map.items():
            if model_name.lower() in base_model.lower():
                model_type = config.get('model_type', 'unknown')
                break
        
        if not model_type:
            # Default to generic settings
            return {
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.1
            }
        
        # Get model-specific settings
        model_settings = self.qlora_config.get('model_specific_settings', {}).get(model_type, {})
        
        return {
            "target_modules": model_settings.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
            "lora_r": model_settings.get('lora_r', 8),
            "lora_alpha": model_settings.get('lora_alpha', 16),
            "lora_dropout": model_settings.get('lora_dropout', 0.1)
        }
    
    def create_qlora_config(self, base_model: str, lora_config: Dict[str, Any]) -> Optional[Any]:
        """Create QLoRA configuration"""
        try:
            from peft import LoraConfig, TaskType
            
            model_settings = self.get_model_specific_settings(base_model)
            
            qlora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=model_settings['lora_r'],
                lora_alpha=model_settings['lora_alpha'],
                lora_dropout=model_settings['lora_dropout'],
                target_modules=model_settings['target_modules']
            )
            
            logger.info(f"✅ QLoRA config created for {base_model}")
            logger.info(f"   Target modules: {model_settings['target_modules']}")
            logger.info(f"   LoRA r: {model_settings['lora_r']}, alpha: {model_settings['lora_alpha']}")
            
            return qlora_config
            
        except ImportError:
            logger.error("❌ PEFT not available for QLoRA")
            return None
        except Exception as e:
            logger.error(f"❌ Failed to create QLoRA config: {e}")
            return None
    
    def create_lora_config(self, base_model: str, lora_config: Dict[str, Any]) -> Optional[Any]:
        """Create standard LoRA configuration"""
        try:
            from peft import LoraConfig, TaskType
            
            model_settings = self.get_model_specific_settings(base_model)
            
            lora_config_peft = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=model_settings['lora_r'],
                lora_alpha=model_settings['lora_alpha'],
                lora_dropout=model_settings['lora_dropout'],
                target_modules=model_settings['target_modules']
            )
            
            logger.info(f"✅ LoRA config created for {base_model}")
            logger.info(f"   Target modules: {model_settings['target_modules']}")
            
            return lora_config_peft
            
        except ImportError:
            logger.error("❌ PEFT not available for LoRA")
            return None
        except Exception as e:
            logger.error(f"❌ Failed to create LoRA config: {e}")
            return None
    
    def apply_qlora(self, model: Any, base_model: str, lora_config: Dict[str, Any]) -> Tuple[Any, bool]:
        """Apply QLoRA to model"""
        try:
            from transformers import BitsAndBytesConfig
            from peft import get_peft_model
            
            # Create QLoRA config
            qlora_config = self.create_qlora_config(base_model, lora_config)
            if not qlora_config:
                return model, False
            
            # Create 4-bit quantization config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            
            # Reload model with 4-bit quantization
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Apply QLoRA
            model = get_peft_model(model, qlora_config)
            
            logger.info(f"✅ QLoRA successfully applied to {base_model}")
            return model, True
            
        except Exception as e:
            logger.warning(f"⚠️ QLoRA failed: {e}")
            return model, False
    
    def apply_lora(self, model: Any, base_model: str, lora_config: Dict[str, Any]) -> Tuple[Any, bool]:
        """Apply standard LoRA to model"""
        try:
            from peft import get_peft_model
            
            # Create LoRA config
            lora_config_peft = self.create_lora_config(base_model, lora_config)
            if not lora_config_peft:
                return model, False
            
            # Apply LoRA
            model = get_peft_model(model, lora_config_peft)
            
            logger.info(f"✅ LoRA successfully applied to {base_model}")
            return model, True
            
        except Exception as e:
            logger.warning(f"⚠️ LoRA failed: {e}")
            return model, False
    
    def setup_optimization(self, training_args: Any) -> Any:
        """Setup training optimization based on QLoRA config"""
        optimization_config = self.qlora_config.get('optimization', {})
        
        # Apply optimization settings
        if optimization_config.get('gradient_checkpointing', True):
            training_args.gradient_checkpointing = True
        
        if optimization_config.get('fp16', True):
            training_args.fp16 = True
        
        if optimization_config.get('dataloader_pin_memory', True):
            training_args.dataloader_pin_memory = True
        
        if 'dataloader_num_workers' in optimization_config:
            training_args.dataloader_num_workers = optimization_config['dataloader_num_workers']
        
        logger.info("✅ Training optimization applied")
        return training_args
    
    def get_memory_management_config(self) -> Dict[str, Any]:
        """Get memory management configuration"""
        memory_config = self.qlora_config.get('memory_management', {})
        
        return {
            "max_memory": memory_config.get('max_memory', 'auto'),
            "device_map": memory_config.get('device_map', 'auto'),
            "low_cpu_mem_usage": memory_config.get('low_cpu_mem_usage', True),
            "torch_dtype": memory_config.get('torch_dtype', 'float16')
        }
    
    def validate_model_compatibility(self, base_model: str) -> Dict[str, Any]:
        """Validate model compatibility with QLoRA/LoRA"""
        compatibility = {
            "model": base_model,
            "qlora_compatible": False,
            "lora_compatible": False,
            "issues": []
        }
        
        # Check if model is in our supported list
        model_found = False
        for model_name in self.model_type_map.keys():
            if model_name.lower() in base_model.lower():
                model_found = True
                break
        
        if not model_found:
            compatibility["issues"].append("Model not in supported model list")
            return compatibility
        
        # Check for known compatibility issues
        if "phi" in base_model.lower():
            compatibility["issues"].append("Phi models have known LoRA compatibility issues")
        elif "dialogpt" in base_model.lower():
            compatibility["issues"].append("DialoGPT models have GGUF conversion issues")
        else:
            compatibility["lora_compatible"] = True
            compatibility["qlora_compatible"] = True
        
        return compatibility
    
    def get_recommended_method(self, base_model: str, gpu_capabilities: Dict[str, Any]) -> str:
        """Get recommended LoRA method based on model and GPU capabilities"""
        # Check model compatibility
        compatibility = self.validate_model_compatibility(base_model)
        if not compatibility["lora_compatible"]:
            return "none"
        
        # Check GPU capabilities
        if gpu_capabilities["qlora_supported"]:
            return "qlora"
        elif gpu_capabilities["lora_supported"]:
            return "lora"
        else:
            return "none"
    
    def log_integration_status(self, base_model: str, method: str, success: bool):
        """Log integration status for monitoring"""
        status = {
            "timestamp": torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else "CPU",
            "model": base_model,
            "method": method,
            "success": success,
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
        }
        
        logger.info(f"📊 LoRA Integration Status: {json.dumps(status, indent=2)}")
        return status 