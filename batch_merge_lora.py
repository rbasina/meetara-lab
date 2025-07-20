#!/usr/bin/env python3
"""
Batch merge script for multiple LoRA adapters with same base model
Efficiently processes all domains at once
"""

import os
import logging
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_all_lora_models(base_path: str) -> List[Dict]:
    """Find all LoRA models in the trained directory"""
    lora_models = []
    base_path = Path(base_path)
    
    # Look for directories containing adapter_model.safetensors
    for domain_dir in base_path.rglob("adapter_model.safetensors"):
        domain_path = domain_dir.parent
        category = domain_path.parent.name
        domain_name = domain_path.name
        
        lora_models.append({
            "path": str(domain_path),
            "category": category,
            "domain": domain_name,
            "full_name": f"{category}/{domain_name}"
        })
    
    return lora_models

def batch_merge_lora_models(base_model_name: str, trained_path: str, output_base_path: str):
    """Batch merge all LoRA models with the same base model"""
    
    logger.info("🔍 Finding all LoRA models...")
    lora_models = find_all_lora_models(trained_path)
    
    if not lora_models:
        logger.warning("⚠️ No LoRA models found!")
        return False
    
    logger.info(f"📊 Found {len(lora_models)} LoRA models:")
    for model in lora_models:
        logger.info(f"   → {model['full_name']}")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        # Load base model once (shared across all domains)
        logger.info(f"🔄 Loading base model: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load tokenizer once (shared across all domains)
        logger.info("🔄 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        
        success_count = 0
        
        for i, model_info in enumerate(lora_models, 1):
            logger.info(f"\n🔄 Processing {i}/{len(lora_models)}: {model_info['full_name']}")
            
            try:
                # Load LoRA adapters
                logger.info(f"   📁 Loading LoRA from: {model_info['path']}")
                lora_model = PeftModel.from_pretrained(base_model, model_info['path'])
                
                # Merge adapters with base model
                logger.info("   🔄 Merging LoRA adapters...")
                merged_model = lora_model.merge_and_unload()
                
                # Create output directory
                output_path = os.path.join(output_base_path, model_info['category'], model_info['domain'])
                os.makedirs(output_path, exist_ok=True)
                
                # Save merged model
                logger.info(f"   💾 Saving to: {output_path}")
                merged_model.save_pretrained(output_path)
                tokenizer.save_pretrained(output_path)
                
                success_count += 1
                logger.info(f"   ✅ Successfully merged {model_info['full_name']}")
                
            except Exception as e:
                logger.error(f"   ❌ Failed to merge {model_info['full_name']}: {e}")
                continue
        
        logger.info(f"\n🎉 Batch merge completed!")
        logger.info(f"✅ Successfully merged: {success_count}/{len(lora_models)} models")
        
        if success_count == len(lora_models):
            logger.info("🎯 100% success rate!")
        else:
            logger.warning(f"⚠️ {len(lora_models) - success_count} models failed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Batch merge failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    # Configuration
    # Load base model name from config instead of hardcoding
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent / "trinity_core"))
    from core_components.config_manager import SmartTrinityConfigManager
    
    config_manager = SmartTrinityConfigManager()
    global_params = config_manager.get_config_dict().get('global_tara_params', {})
    base_model_name = global_params.get('fallback_base_model')
    if not base_model_name:
        model_names = config_manager.get_config_dict().get('model_names', {})
        if model_names:
            base_model_name = list(model_names.values())[0]
        else:
            raise ValueError("❌ No base model configured")
    print(f"✅ Using config-driven base model: {base_model_name}")
    trained_path = "G:/My Drive/meetara-lab/data/production/trained"
    output_base_path = "G:/My Drive/meetara-lab/data/production/trained_merged"
    
    logger.info("🚀 Starting batch LoRA merge...")
    logger.info(f"🔧 Base model: {base_model_name}")
    logger.info(f"📁 Trained models: {trained_path}")
    logger.info(f"📤 Output base: {output_base_path}")
    
    success = batch_merge_lora_models(base_model_name, trained_path, output_base_path)
    
    if success:
        logger.info("🎉 Batch merge completed successfully!")
    else:
        logger.error("❌ Batch merge failed!") 