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
    
    def __init__(self, domain: str = "general", log_level: str = "INFO", **kwargs):
        self.domain = domain
        self.session_id = kwargs.get("session_id", f"{domain}_{int(time.time())}")
        self.start_time = kwargs.get("start_time", datetime.now().isoformat())
        self.is_simulation = kwargs.get("is_simulation", False)
        
        # Determine the base logs directory based on simulation flag
        base_logs_dir = Path("logs")
        if self.is_simulation:
            final_logs_base = base_logs_dir / "dev"
        else:
            final_logs_base = base_logs_dir / "production"

        # Create logs directory for the specific domain within the dev/production structure
        self.logs_dir = final_logs_base / self.domain
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup different log files
        self.setup_loggers(log_level)
        
        # Track session data from kwargs
        self.session_data = {
            "session_id": self.session_id,
            "domain": domain,
            "start_time": self.start_time,
            "is_simulation": self.is_simulation,
            "decisions": [],
            "parameters": {},
            "model_selection": {},
            "sample_generation": {},
            "training_progress": [],
            "quality_metrics": {},
            "config_summary": kwargs # Store all incoming config for full context
        }
        
        self.log_session_start()
        # Log config details immediately after session start
        # self.log_config_loading(
        #     yaml_loaded=True, 
        #     total_domains=kwargs.get("total_domains_in_config", 0)
        # )
        # self.log_domain_validation(
        #     domain=self.domain,
        #     is_valid=kwargs.get("is_valid", False),
        #     category=kwargs.get("category", "N/A")
        # )
    
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
    
    def log_config_loading(self, yaml_loaded: bool, total_domains: int):
        """Log configuration loading details"""
        self.main_logger.info("📊 CONFIGURATION LOADING")
        self.main_logger.info(f"   YAML Config: {'✅ Loaded' if yaml_loaded else '❌ Failed'}")
        self.main_logger.info(f"   Total Domains: {total_domains}")
        
        self.session_data["config_loading"] = {
            "yaml_loaded": yaml_loaded,
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
    
    def log_system_initialized(self, system_name: str = "MeeTARA Lab System"):
        """Log that the overall system has been initialized."""
        self.main_logger.info("✨ SYSTEM INITIALIZATION COMPLETE")
        self.main_logger.info(f"   {system_name} has been successfully initialized.")
        self.main_logger.info("--------------------------------------------------------------------------------")
        self.session_data["system_initialized"] = {
            "system_name": system_name,
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
    
    def log_training_start(self, domains_to_process: List[str]):
        """Log the start of the overall training process."""
        self.main_logger.info("")
        self.main_logger.info("🚀 TRAINING ORCHESTRATION INITIATED")
        self.main_logger.info(f"   Domains to process: {', '.join(domains_to_process)}")
        self.main_logger.info("--------------------------------------------------------------------------------")
        self.session_data["training_start"] = {
            "domains_to_process": domains_to_process,
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
        """Log quality validation of generated data or trained models."""
        if passed:
            self.main_logger.info(f"✅ Quality validation passed for {domain}. Score: {quality_score:.2%}")
        else:
            self.main_logger.warning(f"❌ Quality validation failed for {domain}. Score: {quality_score:.2%}")
        
        self.session_data["quality_metrics"][domain] = {
            "quality_score": quality_score,
            "quality_target": quality_target,
            "passed": passed,
            "timestamp": datetime.now().isoformat()
        }

    def log_training_completed(self, overall_results: Dict[str, Any]):
        """Log the completion of the overall training process."""
        self.main_logger.info("")
        self.main_logger.info("✅ TRAINING ORCHESTRATION COMPLETED")
        self.main_logger.info(f"   Overall Success: {overall_results.get('overall_success', False)}")
        self.main_logger.info(f"   Total Domains Processed: {overall_results.get('total_domains_processed', 0)}")
        self.main_logger.info(f"   Successful Domains: {overall_results.get('successful_domains_count', 0)}")
        self.main_logger.info(f"   Failed Domains: {overall_results.get('failed_domains_count', 0)}")
        self.main_logger.info(f"   Total Processing Time: {overall_results.get('total_processing_time_seconds', 0.0):.2f}s")
        self.main_logger.info("--------------------------------------------------------------------------------")
        self.session_data["training_completion"] = {
            "overall_success": overall_results.get('overall_success'),
            "total_domains_processed": overall_results.get('total_domains_processed'),
            "successful_domains_count": overall_results.get('successful_domains_count'),
            "failed_domains_count": overall_results.get('failed_domains_count'),
            "total_processing_time_seconds": overall_results.get('total_processing_time_seconds'),
            "timestamp": datetime.now().isoformat()
        }

    def log_comprehensive_summary(self, overall_results: Dict[str, Any]):
        """Log a comprehensive summary of the entire training session."""
        self.main_logger.info("")
        self.main_logger.info("📊 COMPREHENSIVE SESSION SUMMARY")
        self.main_logger.info("--------------------------------------------------------------------------------")

        # Performance Metrics Summary
        optimization_gains = overall_results.get("optimization_gains", {})
        self.main_logger.info("🚀 Performance Metrics:")
        self.main_logger.info(f"   Speed Improvement: {optimization_gains.get('speed_improvement', 'N/A')}")
        self.main_logger.info(f"   Success Rate: {optimization_gains.get('success_rate', 0.0):.2%}")
        self.main_logger.info(f"   Baseline Time: {optimization_gains.get('baseline_time', 'N/A')}s")
        optimized_time = optimization_gains.get('optimized_time', 'N/A')
        if isinstance(optimized_time, (int, float)):
            self.main_logger.info(f"   Optimized Time: {optimized_time:.2f}s")
        else:
            self.main_logger.info(f"   Optimized Time: {optimized_time}")
        time_saved = optimization_gains.get('time_saved', 'N/A')
        if isinstance(time_saved, (int, float)):
            self.main_logger.info(f"   Time Saved: {time_saved:.2f}s")
        else:
            self.main_logger.info(f"   Time Saved: {time_saved}")

        # Quality Validation Summary
        quality_validation = overall_results.get("overall_quality_validation", {})
        self.main_logger.info("\n⭐ Quality Validation:")
        self.main_logger.info(f"   Total Domains: {quality_validation.get('total_domains', 0)}")
        self.main_logger.info(f"   Successful Domains: {quality_validation.get('successful_domains', 0)}")
        self.main_logger.info(f"   Failed Domains: {quality_validation.get('failed_domains', 0)}")
        
        self.main_logger.info("\n💡 Optimization Applied:")
        for opt in quality_validation.get("optimization_applied", []):
            self.main_logger.info(f"   - {opt}")
        
        self.main_logger.info("\n🎯 Recommendations:")
        for rec in quality_validation.get("recommendations", []):
            self.main_logger.info(f"   - {rec}")

        self.main_logger.info("\n📈 Domain Breakdown (Success/Failure):")
        for domain_name in overall_results.get("domain_breakdown", {}).get("successful_domains", []):
            self.main_logger.info(f"   ✅ {domain_name}")
        for domain_name in overall_results.get("domain_breakdown", {}).get("failed_domains", []):
            self.main_logger.info(f"   ❌ {domain_name}")
        
        self.main_logger.info("--------------------------------------------------------------------------------")
        self.session_data["comprehensive_summary"] = overall_results

    def log_session_summary(self):
        """Log overall session summary from stored session_data"""
        self.main_logger.info("\n" + "="*80)
        self.main_logger.info(f"✨ MEETARA LAB SESSION SUMMARY - ID: {self.session_id}")
        self.main_logger.info(f"⏰ End Time: {datetime.now().isoformat()}")
        
        total_processing_time = 0.0
        if "training_completion" in self.session_data and "total_processing_time_seconds" in self.session_data["training_completion"]:
            total_processing_time = self.session_data["training_completion"]["total_processing_time_seconds"]
        self.main_logger.info(f"⏱️ Total Session Duration: {total_processing_time:.2f} seconds")
        
        total_domains_in_config = self.session_data.get("config_loading", {}).get("total_domains", "N/A")
        self.main_logger.info(f"📊 Total Domains in Config: {total_domains_in_config}")
        
        successful_domains = self.session_data.get("training_completion", {}).get("successful_domains_count", "N/A")
        self.main_logger.info(f"✅ Successful Domains: {successful_domains}")
        
        failed_domains = self.session_data.get("training_completion", {}).get("failed_domains_count", "N/A")
        self.main_logger.info(f"❌ Failed Domains: {failed_domains}")
        
        overall_success = self.session_data.get("training_completion", {}).get("overall_success", "N/A")
        self.main_logger.info(f"🚀 Overall Success: {overall_success}")
        
        self.main_logger.info("\nDetailed domain breakdown:")
        domain_breakdown = self.session_data.get("comprehensive_summary", {}).get("domain_breakdown", {})
        if domain_breakdown:
            for domain_type, domains_list in domain_breakdown.items():
                if domains_list:
                    self.main_logger.info(f"  - {domain_type.replace('_', ' ').title()}:")
                    for domain_name in domains_list:
                        self.main_logger.info(f"    • {domain_name}")
        else:
            self.main_logger.info("  No detailed domain breakdown available.")

        self.main_logger.info("\nOptimization Summary:")
        optimization_applied = self.session_data.get("comprehensive_summary", {}).get("overall_quality_validation", {}).get("optimization_applied", [])
        if optimization_applied:
            for opt in optimization_applied:
                self.main_logger.info(f"  - {opt}")
        else:
            self.main_logger.info("  No specific optimizations logged.")

        recommendations = self.session_data.get("comprehensive_summary", {}).get("overall_quality_validation", {}).get("recommendations", [])
        if recommendations:
            self.main_logger.info("\nRecommendations:")
            for rec in recommendations:
                self.main_logger.info(f"  - {rec}")
        else:
            self.main_logger.info("  No specific recommendations logged.")

        self.main_logger.info("\nEnd of Session Summary" + "\n" + "="*80)
    
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