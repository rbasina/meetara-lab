#!/usr/bin/env python3
"""
Intelligent Logging System for MeeTARA Lab
Tracks model selection, parameter generation, and training decisions with detailed logs
"""

import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import os

class IntelligentLogger:
    """Comprehensive logging system for tracking all training decisions and processes"""
    
    def __init__(self, domain: str = "general", log_level: str = "INFO"):
        self.domain = domain
        self.session_id = f"{domain}_{int(time.time())}"
        self.start_time = datetime.now()
        
        # Create logs directory
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Setup different log files
        self.setup_loggers(log_level)
        
        # Track session data
        self.session_data = {
            "session_id": self.session_id,
            "domain": domain,
            "start_time": self.start_time.isoformat(),
            "decisions": [],
            "parameters": {},
            "model_selection": {},
            "sample_generation": {},
            "training_progress": [],
            "quality_metrics": {}
        }
        
        self.log_session_start()
    
    def setup_loggers(self, log_level: str):
        """Setup multiple loggers for different purposes"""
        
        # Main logger
        self.main_logger = logging.getLogger(f"meetara_main_{self.session_id}")
        self.main_logger.setLevel(getattr(logging, log_level))
        
        # Model selection logger
        self.model_logger = logging.getLogger(f"meetara_model_{self.session_id}")
        self.model_logger.setLevel(logging.INFO)
        
        # Parameter logger
        self.param_logger = logging.getLogger(f"meetara_params_{self.session_id}")
        self.param_logger.setLevel(logging.DEBUG)
        
        # Training logger
        self.training_logger = logging.getLogger(f"meetara_training_{self.session_id}")
        self.training_logger.setLevel(logging.INFO)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        )
        
        # File handlers with UTF-8 encoding
        main_handler = logging.FileHandler(self.logs_dir / f"meetara_main_{self.domain}_{self.session_id}.log", encoding='utf-8')
        main_handler.setFormatter(detailed_formatter)
        self.main_logger.addHandler(main_handler)
        
        model_handler = logging.FileHandler(self.logs_dir / f"model_selection_{self.domain}_{self.session_id}.log", encoding='utf-8')
        model_handler.setFormatter(detailed_formatter)
        self.model_logger.addHandler(model_handler)
        
        param_handler = logging.FileHandler(self.logs_dir / f"parameters_{self.domain}_{self.session_id}.log", encoding='utf-8')
        param_handler.setFormatter(detailed_formatter)
        self.param_logger.addHandler(param_handler)
        
        training_handler = logging.FileHandler(self.logs_dir / f"training_{self.domain}_{self.session_id}.log", encoding='utf-8')
        training_handler.setFormatter(simple_formatter)
        self.training_logger.addHandler(training_handler)
        
        # Console handler for main logger
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(simple_formatter)
        self.main_logger.addHandler(console_handler)
        
        # Prevent duplicate logs
        for logger in [self.main_logger, self.model_logger, self.param_logger, self.training_logger]:
            logger.propagate = False
    
    def log_session_start(self):
        """Log session initialization"""
        self.main_logger.info("="*80)
        self.main_logger.info(f"🚀 MEETARA LAB TRAINING SESSION STARTED")
        self.main_logger.info(f"📋 Session ID: {self.session_id}")
        self.main_logger.info(f"🎯 Domain: {self.domain}")
        self.main_logger.info(f"⏰ Start Time: {self.start_time}")
        self.main_logger.info("="*80)
    
    def log_config_loading(self, yaml_loaded: bool, json_loaded: bool, total_domains: int):
        """Log configuration loading details"""
        self.main_logger.info("📊 CONFIGURATION LOADING")
        self.main_logger.info(f"   YAML Config: {'✅ Loaded' if yaml_loaded else '❌ Failed'}")
        self.main_logger.info(f"   JSON Config: {'✅ Loaded' if json_loaded else '❌ Failed'}")
        self.main_logger.info(f"   Total Domains: {total_domains}")
        
        self.session_data["config_loading"] = {
            "yaml_loaded": yaml_loaded,
            "json_loaded": json_loaded,
            "total_domains": total_domains,
            "timestamp": datetime.now().isoformat()
        }
    
    def log_domain_validation(self, domain: str, is_valid: bool, category: str = None, suggestions: List[str] = None):
        """Log domain validation process"""
        if is_valid:
            self.main_logger.info(f"✅ Domain '{domain}' validated successfully")
            self.main_logger.info(f"   Category: {category}")
        else:
            self.main_logger.warning(f"❌ Domain '{domain}' validation failed")
            if suggestions:
                self.main_logger.info(f"   💡 Suggestions: {', '.join(suggestions[:3])}")
        
        self.session_data["domain_validation"] = {
            "domain": domain,
            "valid": is_valid,
            "category": category,
            "suggestions": suggestions,
            "timestamp": datetime.now().isoformat()
        }
    
    def log_model_selection(self, domain: str, base_model: str, model_tier: str, selection_reason: str):
        """Log detailed model selection process"""
        self.model_logger.info("🤖 MODEL SELECTION PROCESS")
        self.model_logger.info(f"   Domain: {domain}")
        self.model_logger.info(f"   Selected Model: {base_model}")
        self.model_logger.info(f"   Model Tier: {model_tier}")
        self.model_logger.info(f"   Selection Reason: {selection_reason}")
        
        self.main_logger.info(f"🤖 Model Selected: {base_model} ({model_tier}) for {domain}")
        
        self.session_data["model_selection"] = {
            "domain": domain,
            "base_model": base_model,
            "model_tier": model_tier,
            "selection_reason": selection_reason,
            "timestamp": datetime.now().isoformat()
        }
    
    def log_parameter_generation(self, domain: str, model_tier: str, parameters: Dict[str, Any], source: str = "YAML"):
        """Log detailed parameter generation process"""
        self.param_logger.info("⚙️ PARAMETER GENERATION")
        self.param_logger.info(f"   Domain: {domain}")
        self.param_logger.info(f"   Model Tier: {model_tier}")
        self.param_logger.info(f"   Parameter Source: {source}")
        self.param_logger.info("   Generated Parameters:")
        
        for param, value in parameters.items():
            self.param_logger.info(f"     {param}: {value}")
            
            # Explain parameter choice
            explanation = self._explain_parameter_choice(param, value, model_tier)
            if explanation:
                self.param_logger.info(f"       → {explanation}")
        
        self.main_logger.info(f"⚙️ Parameters generated for {domain} ({model_tier})")
        self.main_logger.info(f"   Batch size: {parameters.get('batch_size', 'N/A')}")
        self.main_logger.info(f"   LoRA rank: {parameters.get('lora_r', 'N/A')}")
        self.main_logger.info(f"   Max steps: {parameters.get('max_steps', 'N/A')}")
        self.main_logger.info(f"   Learning rate: {parameters.get('learning_rate', 'N/A')}")
        
        self.session_data["parameters"] = {
            "domain": domain,
            "model_tier": model_tier,
            "parameters": parameters,
            "source": source,
            "timestamp": datetime.now().isoformat()
        }
    
    def log_sample_generation(self, domain: str, target_samples: int, generated_samples: int, quality_score: float, generation_time: float):
        """Log sample generation process"""
        self.main_logger.info("📝 SAMPLE GENERATION")
        self.main_logger.info(f"   Domain: {domain}")
        self.main_logger.info(f"   Target Samples: {target_samples:,}")
        self.main_logger.info(f"   Generated Samples: {generated_samples:,}")
        self.main_logger.info(f"   Quality Score: {quality_score:.2%}")
        self.main_logger.info(f"   Generation Time: {generation_time:.2f}s")
        
        efficiency = (generated_samples / target_samples) * 100 if target_samples > 0 else 0
        self.main_logger.info(f"   Generation Efficiency: {efficiency:.1f}%")
        
        self.session_data["sample_generation"] = {
            "domain": domain,
            "target_samples": target_samples,
            "generated_samples": generated_samples,
            "quality_score": quality_score,
            "generation_time": generation_time,
            "efficiency": efficiency,
            "timestamp": datetime.now().isoformat()
        }
    
    def log_training_step(self, step: int, loss: float, accuracy: float = None, learning_rate: float = None):
        """Log individual training steps"""
        if step % 100 == 0:  # Log every 100 steps
            self.training_logger.info(f"Step {step:4d} | Loss: {loss:.4f}" + 
                                    (f" | Acc: {accuracy:.2%}" if accuracy else "") +
                                    (f" | LR: {learning_rate:.2e}" if learning_rate else ""))
        
        self.session_data["training_progress"].append({
            "step": step,
            "loss": loss,
            "accuracy": accuracy,
            "learning_rate": learning_rate,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_gguf_creation(self, domain: str, gguf_info: Dict[str, Any]):
        """Log GGUF file creation process"""
        self.main_logger.info("📦 GGUF CREATION")
        self.main_logger.info(f"   Domain: {domain}")
        self.main_logger.info(f"   Format: {gguf_info.get('format', 'N/A')}")
        self.main_logger.info(f"   Size: {gguf_info.get('size', 'N/A')}MB")
        self.main_logger.info(f"   Compression: {gguf_info.get('compression', 'N/A')}")
        self.main_logger.info(f"   Quality: {gguf_info.get('quality', 'N/A')}%")
        self.main_logger.info(f"   Filename: {gguf_info.get('filename', 'N/A')}")
        
        self.session_data["gguf_creation"] = {
            "domain": domain,
            "gguf_info": gguf_info,
            "timestamp": datetime.now().isoformat()
        }
    
    def log_quality_validation(self, domain: str, quality_score: float, quality_target: float, passed: bool):
        """Log quality validation results"""
        status = "✅ PASSED" if passed else "❌ FAILED"
        self.main_logger.info(f"🔍 QUALITY VALIDATION: {status}")
        self.main_logger.info(f"   Domain: {domain}")
        self.main_logger.info(f"   Quality Score: {quality_score:.1f}%")
        self.main_logger.info(f"   Quality Target: {quality_target:.1f}%")
        self.main_logger.info(f"   Difference: {quality_score - quality_target:+.1f}%")
        
        self.session_data["quality_metrics"] = {
            "domain": domain,
            "quality_score": quality_score,
            "quality_target": quality_target,
            "passed": passed,
            "difference": quality_score - quality_target,
            "timestamp": datetime.now().isoformat()
        }
    
    def log_session_summary(self):
        """Log comprehensive session summary"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        self.main_logger.info("="*80)
        self.main_logger.info("TRAINING SESSION COMPLETED")
        self.main_logger.info(f"Session ID: {self.session_id}")
        self.main_logger.info(f"Domain: {self.domain}")
        self.main_logger.info(f"Duration: {duration}")
        self.main_logger.info(f"End Time: {end_time}")
        
        # Summary statistics
        if self.session_data.get("sample_generation"):
            samples = self.session_data["sample_generation"]
            self.main_logger.info(f"Samples Generated: {samples.get('generated_samples', 0):,}")
        
        if self.session_data.get("quality_metrics"):
            quality = self.session_data["quality_metrics"]
            self.main_logger.info(f"Final Quality: {quality.get('quality_score', 0):.1f}%")
        
        self.main_logger.info("="*80)
        
        # Save session data to JSON
        self.session_data["end_time"] = end_time.isoformat()
        self.session_data["duration_seconds"] = duration.total_seconds()
        
        session_file = self.logs_dir / f"session_summary_{self.domain}_{self.session_id}.json"
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(self.session_data, f, indent=2, default=str)
        
        self.main_logger.info(f"Session data saved to: {session_file}")
    
    def _explain_parameter_choice(self, param: str, value: Any, model_tier: str) -> str:
        """Explain why a specific parameter value was chosen"""
        explanations = {
            "batch_size": {
                8: "Higher batch size for small models (memory efficient)",
                4: "Balanced batch size for medium models",
                2: "Standard TARA proven batch size",
                1: "Memory-optimized batch size for large models"
            },
            "lora_r": {
                4: "Lower rank for efficiency in small models",
                6: "Moderate rank for balanced performance",
                8: "Standard TARA proven rank",
                12: "Higher rank for quality models",
                16: "Expert-level rank for maximum capability",
                20: "Premium rank for highest quality"
            },
            "max_steps": {
                500: "Fewer steps for fast convergence in small models",
                650: "Moderate steps for balanced training",
                846: "TARA proven optimal steps",
                1000: "Extended steps for quality training",
                1200: "Expert-level steps for maximum learning",
                1500: "Premium steps for highest quality"
            }
        }
        
        return explanations.get(param, {}).get(value, f"Optimized for {model_tier} tier")
    
    def log_decision(self, decision_type: str, decision: str, reasoning: str):
        """Log any important decision made during training"""
        self.main_logger.info(f"🧠 DECISION: {decision_type}")
        self.main_logger.info(f"   Decision: {decision}")
        self.main_logger.info(f"   Reasoning: {reasoning}")
        
        self.session_data["decisions"].append({
            "type": decision_type,
            "decision": decision,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat()
        })

# Global logger instance
_current_logger: Optional[IntelligentLogger] = None

def get_logger(domain: str = "general") -> IntelligentLogger:
    """Get or create logger for domain"""
    global _current_logger
    if _current_logger is None or _current_logger.domain != domain:
        _current_logger = IntelligentLogger(domain)
    return _current_logger

def log_info(message: str):
    """Quick logging function"""
    if _current_logger:
        _current_logger.main_logger.info(message)

def log_warning(message: str):
    """Quick warning logging"""
    if _current_logger:
        _current_logger.main_logger.warning(message) 