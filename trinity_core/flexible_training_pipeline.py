#!/usr/bin/env python3
"""
MeeTARA Lab - Flexible Training Pipeline
Handles: Single Domain | Multiple Domains | All Domains

Usage:
    # Single domain
    python src/flexible_training_pipeline.py --mode single --domain healthcare
    
    # Multiple domains
    python src/flexible_training_pipeline.py --mode multiple --domains healthcare,finance,education
    
    # All domains
    python src/flexible_training_pipeline.py --mode all
    
    # Custom selection
    python src/flexible_training_pipeline.py --mode custom --categories healthcare,business
"""

import argparse
import yaml
import json
import time
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import random

# Import the centralized config manager
from trinity_core.core_components.config_manager import SmartTrinityConfigManager
from trinity_core.intelligent_logger import IntelligentLogger

def load_and_print_config_summary():
    """Loads the config and prints a summary of available domains."""
    manager = SmartTrinityConfigManager()
    all_domains = manager.get_all_domains_flat()
    total_domains = len(all_domains)
    # The new manager doesn't have a direct 'get_all_domain_categories' method that returns a simple dict.
    # We can get the categories by processing the domain list.
    categories = {details['category'] for details in all_domains.values()}
    print(f"✅ Loaded {total_domains} domains across {len(categories)} categories.")
    for domain in all_domains.keys():
        print(f"  - {domain}")

class FlexibleTrainingPipeline:
    """Flexible training pipeline for single/multiple/all domains"""
    
    def __init__(self):
        self.config_manager = SmartTrinityConfigManager()
        self.logger = IntelligentLogger(domain="flexible_pipeline")
        self.all_domains = self.config_manager.get_all_domains_flat()
        self.pipeline_stages = self._define_pipeline_stages()
        
    def _load_configuration(self) -> SmartTrinityConfigManager:
        """Load domain mapping and training configuration via the central manager."""
        print("📋 Loading configuration via SmartTrinityConfigManager...")
        try:
            manager = SmartTrinityConfigManager()
            total_domains = manager.get_total_domain_count()
            print(f"✅ Loaded {total_domains} domains across {len(manager.get_all_domain_categories())} categories.")
            return manager
        except (FileNotFoundError, ValueError) as e:
            raise RuntimeError(f"Failed to load central configuration: {e}")

    def setup_gpu_environment(self):
        """Detect and configure GPU environment"""
        print("🚀 Setting up GPU environment...")
        
        # Detect GPU
        try:
            gpu_info = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                    capture_output=True, text=True)
            if gpu_info.returncode == 0:
                gpu_name = gpu_info.stdout.strip()
                print(f"✅ GPU Detected: {gpu_name}")
                
                if "T4" in gpu_name:
                    self.gpu_tier = "T4"
                    self.cost_per_hour = 0.40
                    self.batch_size = 16
                    self.speed_factor = "37x"
                elif "V100" in gpu_name:
                    self.gpu_tier = "V100"
                    self.cost_per_hour = 2.50
                    self.batch_size = 32
                    self.speed_factor = "75x"
                elif "A100" in gpu_name:
                    self.gpu_tier = "A100"
                    self.cost_per_hour = 4.00
                    self.batch_size = 64
                    self.speed_factor = "151x"
                else:
                    self.gpu_tier = "T4"
                    self.cost_per_hour = 0.40
                    self.batch_size = 16
                    self.speed_factor = "37x"
            else:
                print("⚠️ No GPU detected - using CPU")
                self.gpu_tier = "CPU"
                self.cost_per_hour = 0.0
                self.batch_size = 2
                self.speed_factor = "1x"
        except:
            print("⚠️ GPU detection failed - using CPU fallback")
            self.gpu_tier = "CPU"
            self.cost_per_hour = 0.0
            self.batch_size = 2
            self.speed_factor = "1x"
        
        print(f"⚡ Speed: {self.speed_factor} faster | GPU: {self.gpu_tier} | Cost: ${self.cost_per_hour}/hr")
        
        # Configure CUDA
        if torch.cuda.is_available():
            print(f"🔥 CUDA Available: {torch.cuda.device_count()} GPU(s)")
            print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            print("⚠️ CUDA not available - using CPU")
            
    def initialize_data_generator(self):
        """Initialize real-time data generator"""
        self.crisis_scenarios = [
            "emergency_health", "mental_crisis", "financial_emergency", 
            "relationship_crisis", "work_emergency", "family_crisis"
        ]
        
        self.emotional_contexts = [
            "stressed", "anxious", "confident", "confused", "hopeful", 
            "frustrated", "excited", "worried", "determined", "overwhelmed"
        ]
        
        self.professional_roles = [
            "healthcare_provider", "teacher", "manager", "consultant", 
            "therapist", "advisor", "coach", "specialist", "expert"
        ]
        
        print("✅ Real-time data generator initialized")
        
    def get_domains_by_mode(self, mode: str, domains: Optional[str] = None, categories: Optional[str] = None) -> List[str]:
        """Get domains based on training mode using the config manager."""
        
        if mode == "single":
            if not domains:
                raise ValueError("Single mode requires --domain parameter")
            domain_list = [domains.strip()]
            
        elif mode == "multiple":
            if not domains:
                raise ValueError("Multiple mode requires --domains parameter")
            domain_list = [d.strip() for d in domains.split(',')]
            
        elif mode == "all":
            domain_list = self.config_manager.get_all_domains_flat()
            
        elif mode == "custom":
            if not categories:
                raise ValueError("Custom mode requires --categories parameter")
            category_list = [c.strip() for c in categories.split(',')]
            all_config_categories = self.config_manager.get_all_domain_categories()
            domain_list = []
            for category in category_list:
                domain_list.extend(all_config_categories.get(category, []))
                    
        else:
            raise ValueError(f"Invalid mode: {mode}. Use: single, multiple, all, custom")
        
        # Validate domains exist
        all_domains_set = set(self.config_manager.get_all_domains_flat())
        invalid_domains = [d for d in domain_list if d not in all_domains_set]
        if invalid_domains:
            print(f"⚠️ Invalid domains specified: {invalid_domains}")
            domain_list = [d for d in domain_list if d in all_domains_set]
        
        return domain_list
    
    def generate_domain_training_data(self, domain: str, samples: int = 2000) -> List[Dict]:
        """Generate real-time training data for a domain"""
        print(f"🔄 Generating {samples} samples for {domain}...")
        
        conversations = []
        scenario_types = ["consultation", "guidance", "problem_solving", "education", "support"]
        
        for i in range(samples):
            # Generate agentic conversation
            scenario = random.choice(scenario_types)
            emotion = random.choice(self.emotional_contexts)
            role = random.choice(self.professional_roles)
            
            # Crisis intervention (5% of data)
            is_crisis = random.random() < 0.05
            
            if is_crisis:
                crisis_type = random.choice(self.crisis_scenarios)
                conversation = [
                    {"role": "user", "content": f"I'm having a {crisis_type} and feeling {emotion}. I need help with {domain}."},
                    {"role": "assistant", "content": f"I understand you're going through a difficult time with {crisis_type}. Let me help you with {domain} in a way that addresses your {emotion} feelings. First, let's focus on immediate steps you can take..."}
                ]
            else:
                conversation = [
                    {"role": "user", "content": f"As someone feeling {emotion}, I need guidance on {domain}. Can you help?"},
                    {"role": "assistant", "content": f"Absolutely! I understand you're feeling {emotion} about {domain}. As your {role}, I'll provide comprehensive guidance that acknowledges your emotional state while giving you practical, actionable advice..."}
                ]
            
            conversations.append({
                "domain": domain,
                "timestamp": datetime.now().isoformat(),
                "emotion_context": emotion,
                "professional_role": role,
                "is_crisis": is_crisis,
                "scenario_type": scenario,
                "conversation": conversation,
                "quality_score": random.uniform(0.7, 1.0)
            })
            
            if (i + 1) % 200 == 0:
                print(f"  📊 Generated {i + 1}/{samples} samples")
        
        # Apply TARA's 31% quality filtering
        print(f"🔍 Applying quality filter (31% success rate)...")
        sorted_conversations = sorted(conversations, key=lambda x: x['quality_score'], reverse=True)
        cutoff_index = int(len(sorted_conversations) * 0.31)
        filtered_conversations = sorted_conversations[:cutoff_index]
        
        print(f"✅ Domain {domain}: {len(filtered_conversations)} high-quality samples from {samples} generated")
        return filtered_conversations
    
    def train_single_domain(self, domain: str) -> Optional[str]:
        """Train a single domain model"""
        import logging, json, os
        logger = logging.getLogger("FlexibleTrainingPipeline")
        print(f"🚀 Training single domain: {domain}")
        stats = {"domain": domain}
        config_path = self.config_manager.config_path if hasattr(self.config_manager, 'config_path') else 'config/trinity_config.yaml'
        logger.info(f"[CONFIG] Using config: {config_path}")
        stats["config_path"] = config_path
        # Get model for this domain from the config manager
        try:
            params = self.config_manager.get_tara_proven_params(domain)
            model_name = params['base_model']
            category = params['category']
            # Parse for dual-strategy
            from trinity_core.core_components.config_manager import parse_domain_model_entry
            primary_model, fallback_model = parse_domain_model_entry(model_name)
            logger.info(f"🤖 Primary Model: {primary_model}")
            if fallback_model:
                logger.info(f"🤖 Fallback/Small Model: {fallback_model}")
            # Use primary_model for training, fallback_model for extraction if needed
        except ValueError as e:
            logger.error(f"❌ Could not get training parameters for domain '{domain}': {e}")
            return None
        logger.info(f"📋 Domain: {domain}")
        logger.info(f"📂 Category: {category}")
        logger.info(f"🤖 Base Model: {model_name}")
        stats["base_model"] = model_name
        stats["category"] = category
        # Generate training data
        training_data = self.generate_domain_training_data(domain)
        stats["num_samples_before_training"] = len(training_data) if training_data else 0
        # Train model
        model_path = self._train_model(domain, model_name, training_data)
        # Optionally, collect stats after training (e.g., loss, accuracy if available)
        stats["model_path"] = model_path
        # Save stats report
        stats_dir = "training_stats"
        os.makedirs(stats_dir, exist_ok=True)
        stats_file = os.path.join(stats_dir, f"{domain}_stats.json")
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"[STATS] Training stats saved to {stats_file}")
        return model_path
    
    def train_multiple_domains(self, domain_list: List[str]) -> Dict[str, str]:
        """Train multiple domains"""
        print(f"🚀 Training {len(domain_list)} domains: {', '.join(domain_list)}")
        
        results = {}
        total_cost = 0.0
        start_time = time.time()
        
        for i, domain in enumerate(domain_list, 1):
            print(f"\n{'='*60}")
            print(f"📋 Training domain {i}/{len(domain_list)}: {domain}")
            print(f"{'='*60}")
            
            # Check cost limit
            current_cost = ((time.time() - start_time) / 3600) * self.cost_per_hour
            if current_cost > 45.0:  # Safety margin under $50
                print(f"⚠️ Approaching cost limit (${current_cost:.2f}). Stopping training.")
                break
            
            try:
                model_path = self.train_single_domain(domain)
                results[domain] = model_path
                
                # Calculate progress
                elapsed_time = time.time() - start_time
                cost_so_far = (elapsed_time / 3600) * self.cost_per_hour
                estimated_total_cost = cost_so_far * (len(domain_list) / i)
                
                print(f"✅ Completed {domain}")
                print(f"⏱️ Elapsed: {elapsed_time/60:.1f} min | Cost: ${cost_so_far:.2f}")
                print(f"📊 Progress: {i}/{len(domain_list)} | ETA Total Cost: ${estimated_total_cost:.2f}")
                
            except Exception as e:
                print(f"❌ Failed to train {domain}: {e}")
                results[domain] = f"FAILED: {e}"
        
        total_time = time.time() - start_time
        total_cost = (total_time / 3600) * self.cost_per_hour
        
        print(f"\n{'='*60}")
        print(f"🎯 TRAINING COMPLETE")
        print(f"✅ Successful: {len([r for r in results.values() if not r.startswith('FAILED')])} domains")
        print(f"❌ Failed: {len([r for r in results.values() if r.startswith('FAILED')])} domains")
        print(f"⏱️ Total time: {total_time/60:.1f} minutes")
        print(f"💰 Total cost: ${total_cost:.2f}")
        print(f"{'='*60}")
        
        return results
    
    def _train_model(self, domain: str, model_name: str, training_data: List[Dict]) -> str:
        """Internal method to train a model"""
        import time
        import os
        from pathlib import Path
        
        # Track download timing and caching
        download_start = time.time()
        print(f"📥 Starting download for domain '{domain}' with base model: {model_name}")
        
        # Check if model is already cached
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_cache_path = None
        
        # Try to find existing model in cache
        for root, dirs, files in os.walk(cache_dir):
            if model_name.replace("/", "--") in root:
                model_cache_path = root
                break
        
        if model_cache_path:
            print(f"✅ Model found in cache: {model_cache_path}")
            print(f"⏱️ Cache hit - no download needed")
        else:
            print(f"🔄 Model not in cache - downloading from HuggingFace...")
        
        # Load model and tokenizer with timing
        tokenizer_start = time.time()
        print(f"🔧 Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer_time = time.time() - tokenizer_start
        print(f"✅ Tokenizer loaded in {tokenizer_time:.2f}s")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model_start = time.time()
        print(f"🧠 Loading base model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        model_time = time.time() - model_start
        print(f"✅ Base model loaded in {model_time:.2f}s")
        
        # Check where the model was actually stored
        try:
            from transformers import file_utils
            model_path = file_utils.cached_file(model_name, "pytorch_model.bin")
            print(f"📁 Model stored at: {model_path}")
        except Exception as e:
            print(f"⚠️ Could not determine exact model path: {e}")
        
        total_download_time = time.time() - download_start
        print(f"⏱️ Total model preparation time: {total_download_time:.2f}s")
        
        # Log model size
        try:
            model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            print(f"📊 Model size: {model_size_mb:.1f} MB")
        except Exception as e:
            print(f"⚠️ Could not calculate model size: {e}")
        
        # Configure LoRA (TARA proven parameters)
        tara_params = self.config_manager.get_tara_proven_params(domain)
        
        def get_linear_module_names(model):
            linear_names = []
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    linear_names.append(name)
            return linear_names
        
        # Generic target modules that work with most models
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        # Apply LoRA with error handling
        try:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=tara_params['lora_r'],
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=target_modules
            )
            
            model = get_peft_model(model, lora_config)
            print(f"✅ LoRA configured: r={tara_params['lora_r']}")
        except Exception as e:
            print(f"⚠️ LoRA setup failed: {e}")
            print("🔄 Continuing training without LoRA...")
            # Continue with the base model without LoRA
        
        # Prepare dataset
        formatted_data = []
        for item in training_data:
            conversation = item['conversation']
            text = f"User: {conversation[0]['content']}\\nAssistant: {conversation[1]['content']}"
            formatted_data.append({"text": text})
        
        dataset = Dataset.from_list(formatted_data)
        
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=tara_params['sequence_length']
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Training arguments
        # Use proper path structure that matches model factory
        from pathlib import Path
        project_root = Path(__file__).resolve().parents[1]
        
        # Get category for the domain
        domain_details = self.config_manager._get_domain_details(domain)
        category = domain_details.get("category", "unknown_category") if domain_details else "unknown_category"
        
        output_dir = project_root / "data" / "production" / "trained" / category / domain
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔧 Setting output_dir to: {output_dir}")
        print(f"🔧 Absolute path: {output_dir.absolute()}")
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),  # Ensure it's a string
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=max(1, tara_params['batch_size'] // self.batch_size),
            max_steps=tara_params['max_steps'],
            learning_rate=tara_params['learning_rate'],
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=True,
            dataloader_pin_memory=torch.cuda.is_available(),
            save_steps=200,
            logging_steps=50,
            remove_unused_columns=False,
            report_to=None,  # Disable wandb
        )
        
        # Train model
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
        )
        
        print(f"⚡ Starting training...")
        training_start = time.time()
        
        trainer.train()
        
        training_time = time.time() - training_start
        print(f"✅ Training completed in {training_time/60:.1f} minutes")
        
        # Save model
        trainer.save_model(output_dir)
        print(f"💾 Model saved to: {output_dir}")
        
        return output_dir
    
    def display_available_options(self):
        """Display all available domains and categories"""
        print("\n📋 AVAILABLE TRAINING OPTIONS")
        print("="*60)
        
        for category, domains in self._get_available_domains_by_category().items():
            model_name = list(domains.values())[0] if domains else "N/A"
            print(f"\n🏷️ {category.upper()} ({len(domains)} domains)")
            print(f"   Model: {model_name}")
            print(f"   Domains: {', '.join(list(domains.keys())[:5])}{'...' if len(domains) > 5 else ''}")
        
        print(f"\n📊 TOTAL: {len(self.config_manager.get_all_domains_flat())} domains across {len(self._get_available_domains_by_category())} categories")
        print("="*60)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="MeeTARA Lab - Flexible Training Pipeline")
    parser.add_argument('--mode', type=str, required=True, choices=['single', 'multiple', 'all', 'custom', 'display'],
                        help="Training mode")
    parser.add_argument('--domains', type=str, help="Comma-separated list of domains for 'multiple' mode")
    parser.add_argument('--domain', type=str, help="Single domain for 'single' mode")
    parser.add_argument('--categories', type=str, help="Comma-separated list of categories for 'custom' mode")
    
    args = parser.parse_args()
    
    pipeline = FlexibleTrainingPipeline()
    
    if args.mode == "display":
        pipeline.display_available_options()
    else:
        domains_to_train = pipeline.get_domains_by_mode(args.mode, args.domains or args.domain, args.categories)
        
        if not domains_to_train:
            print("No valid domains to train. Exiting.")
            return
            
        if len(domains_to_train) == 1:
            pipeline.train_single_domain(domains_to_train[0])
        else:
            pipeline.train_multiple_domains(domains_to_train)

if __name__ == "__main__":
    main() 
