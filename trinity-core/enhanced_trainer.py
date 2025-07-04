"""
Enhanced Trinity Trainer with Production Validation
Ensures models work in MeeTARA during training - prevents training success ≠ production reliability gap.
Adapted from TARA Universal Model's proven enhanced training system.
"""

import os
import json
import logging
import asyncio
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from .production_validator import ProductionValidator

logger = logging.getLogger(__name__)

class EnhancedTrinityTrainer:
    """
    Enhanced Trinity Trainer with integrated production validation.
    Tests models during training to ensure MeeTARA compatibility.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        domain: str = "parenting",
        output_dir: str = "models",
        batch_size: int = 6,
        seq_length: int = 512,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        learning_rate: float = 2e-4,
        num_epochs: int = 3,
        max_steps: int = 846,
        gradient_accumulation_steps: int = 4,
        fp16: bool = True,
        gradient_checkpointing: bool = False
    ):
        self.model_name = model_name
        self.domain = domain
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.fp16 = fp16
        self.gradient_checkpointing = gradient_checkpointing
        
        # Enhanced training components
        self.validator = ProductionValidator()
        self.training_history = []
        self.validation_checkpoints = []
        
        os.makedirs(output_dir, exist_ok=True)
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def load_model_and_tokenizer(self):
        """Load model and tokenizer with LoRA configuration."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.fp16 else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        self.model = get_peft_model(base_model, lora_config)
        
        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def prepare_dataset(self, data: list) -> list:
        """Prepare dataset for training."""
        processed = []
        for item in data:
            if isinstance(item, dict):
                # Handle MeeTARA training data format
                if 'input' in item and 'output' in item:
                    text = f"User: {item['input']}\nAssistant: {item['output']}"
                    processed.append({'text': text})
                elif 'text' in item:
                    processed.append({'text': item['text']})
            elif isinstance(item, str):
                processed.append({'text': item})
        return processed

    def tokenize_function(self, examples):
        """Tokenize examples for training."""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=self.seq_length,
        )

    async def train_with_validation(self, train_data: list, resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """
        Enhanced training with production validation at checkpoints.
        
        Args:
            train_data: Training data list
            resume_from_checkpoint: Path to checkpoint to resume from, or None
            
        Returns:
            Training results with validation scores
        """
        if resume_from_checkpoint:
            logger.info(f"🔄 Resuming training from checkpoint: {resume_from_checkpoint}")
        else:
            logger.info(f"🚀 Enhanced training starting for {self.domain}")
            
        logger.info(f"📊 Production validation enabled")
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Prepare dataset
        processed_data = self.prepare_dataset(train_data)
        
        import datasets
        ds = datasets.Dataset.from_list(processed_data)
        ds = ds.map(self.tokenize_function, batched=True, remove_columns=["text"])
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )
        
        # Enhanced training arguments with validation checkpoints
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            warmup_steps=100,
            save_steps=200,  # Save every 200 steps for validation
            eval_steps=200,
            logging_steps=50,
            max_steps=self.max_steps,
            fp16=self.fp16,
            gradient_checkpointing=self.gradient_checkpointing,
            save_total_limit=3,
            report_to=None,
            run_name=f"{self.domain}_enhanced_training",
            dataloader_pin_memory=False,
            remove_unused_columns=False
        )
        
        # Custom trainer with production validation callbacks
        self.trainer = ProductionValidatedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=ds,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            validator=self.validator,
            domain=self.domain,
            enhanced_trainer=self
        )
        
        # Start training with validation
        start_time = time.time()
        
        training_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        training_duration = time.time() - start_time
        logger.info(f"✅ Training completed in {training_duration:.2f} seconds")
        
        # Final production validation
        logger.info("🔍 Final production validation...")
        final_validation = await self.validator.validate_model_production_ready(
            self.domain, self.output_dir
        )
        
        # Save enhanced training results
        enhanced_results = {
            "domain": self.domain,
            "training_duration": training_duration,
            "training_result": {
                "train_runtime": training_result.metrics.get("train_runtime", 0),
                "train_loss": training_result.metrics.get("train_loss", 0),
                "train_samples": len(ds)
            },
            "final_validation": final_validation,
            "production_ready": final_validation.get("production_ready", False),
            "validation_history": self.validation_checkpoints,
            "timestamp": datetime.now().isoformat(),
            "resumed_from_checkpoint": resume_from_checkpoint is not None
        }
        
        # Save results
        results_path = os.path.join(self.output_dir, "enhanced_training_results.json")
        with open(results_path, 'w') as f:
            json.dump(enhanced_results, f, indent=2)
        
        # Log final status
        if enhanced_results["production_ready"]:
            logger.info(f"🎉 {self.domain} model is PRODUCTION READY!")
            logger.info(f"📊 Validation Score: {final_validation.get('overall_score', 0):.2f}")
        else:
            logger.warning(f"⚠️ {self.domain} model needs improvement for production")
            logger.warning(f"📊 Validation Score: {final_validation.get('overall_score', 0):.2f}")
        
        return enhanced_results

class ProductionValidatedTrainer(Trainer):
    """
    Custom Trainer that validates models during training.
    """
    
    def __init__(self, validator, domain, enhanced_trainer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validator = validator
        self.domain = domain
        self.enhanced_trainer = enhanced_trainer
        self.last_validation_step = 0
        
    def on_save(self, args, state, control, **kwargs):
        """Perform production validation when model is saved."""
        if state.global_step - self.last_validation_step >= 200:  # Validate every 200 steps
            logger.info(f"🔍 Production validation at step {state.global_step}")
            
            # Run async validation in sync context
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                validation_result = loop.run_until_complete(
                    self.validator.validate_model_production_ready(
                        self.domain, args.output_dir
                    )
                )
                
                # Store validation result
                checkpoint_result = {
                    "step": state.global_step,
                    "validation": validation_result,
                    "timestamp": datetime.now().isoformat()
                }
                self.enhanced_trainer.validation_checkpoints.append(checkpoint_result)
                
                # Log validation result
                production_ready = validation_result.get("production_ready", False)
                score = validation_result.get("overall_score", 0)
                if production_ready:
                    logger.info(f"✅ Step {state.global_step}: Model is production ready (Score: {score:.2f})")
                else:
                    logger.warning(f"⚠️ Step {state.global_step}: Model needs improvement (Score: {score:.2f})")
                
                self.last_validation_step = state.global_step
                
            except Exception as e:
                logger.error(f"❌ Production validation failed: {e}")
        
        return super().on_save(args, state, control, **kwargs)

class TrainingOrchestrator:
    """
    Orchestrates training across all domains with enhanced validation.
    """
    
    def __init__(self, domains: List[str] = None):
        self.domains = domains or ["parenting", "communication", "healthcare", "business", "education"]
        self.training_progress = {}
        
    async def train_all_domains_enhanced(self, training_data: Dict[str, List], resume_from_checkpoint: bool = False):
        """
        Train all domains with enhanced validation.
        
        Args:
            training_data: Dictionary of domain -> training data
            resume_from_checkpoint: Whether to resume from latest checkpoints
        """
        logger.info("🚀 Starting enhanced training for all domains")
        logger.info("📊 Production validation enabled for all models")
        
        start_time = time.time()
        
        for domain in self.domains:
            if domain not in training_data:
                logger.warning(f"⚠️ No training data for {domain}, skipping")
                continue
                
            logger.info(f"\n🎯 Training {domain} domain with production validation")
            
            domain_start = time.time()
            
            try:
                # Initialize enhanced trainer
                trainer = EnhancedTrinityTrainer(
                    domain=domain,
                    output_dir=f"models/{domain}/enhanced_training"
                )
                
                # Check for checkpoint if resuming
                checkpoint_path = None
                if resume_from_checkpoint:
                    checkpoint_dir = f"models/{domain}/enhanced_training"
                    if os.path.exists(checkpoint_dir):
                        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
                        if checkpoints:
                            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
                            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
                            logger.info(f"Found checkpoint for {domain}: {checkpoint_path}")
                
                # Train with validation
                results = await trainer.train_with_validation(
                    train_data=training_data[domain],
                    resume_from_checkpoint=checkpoint_path
                )
                
                domain_duration = time.time() - domain_start
                
                # Record progress
                self.training_progress[domain] = {
                    "status": "completed",
                    "duration": domain_duration,
                    "results": results,
                    "resumed_from": checkpoint_path
                }
                
                logger.info(f"✅ {domain} training completed in {domain_duration:.2f} seconds")
                
            except Exception as e:
                logger.error(f"❌ {domain} training failed: {e}")
                self.training_progress[domain] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        total_duration = time.time() - start_time
        logger.info(f"\n🎉 All domain training completed in {total_duration:.2f} seconds")
        
        # Save overall results
        results_path = "training_results/enhanced_training_summary.json"
        os.makedirs("training_results", exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump({
                "domains": self.training_progress,
                "total_duration": total_duration,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        
        return self.training_progress 