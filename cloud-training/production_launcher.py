#!/usr/bin/env python3
"""
MeeTARA Lab - Trinity Production Launcher
This is the single entry point for launching REAL training sessions.
It uses the complete Trinity Super-Agent ecosystem.
WITH COMPREHENSIVE INTELLIGENT LOGGING
"""
import os
import sys
import asyncio
import argparse
from pathlib import Path
from dataclasses import dataclass
import time
from datetime import datetime

# -- Set up the Python path --
# This is the crucial part. It adds the project's root directory to the Python path.
# This allows modules like 'trinity_core' to be found from within this script.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from trinity_core.agents.system_integration.complete_agent_ecosystem import CompleteAgentEcosystem
from trinity_core.core_components.config_manager import SmartTrinityConfigManager
from trinity_core.intelligent_logger import IntelligentLogger

# -- Set up the Python path --
# This ensures that all project modules can be imported correctly from this entry point.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "model-factory"))
sys.path.insert(0, str(project_root / "trinity_core"))
sys.path.insert(0, str(project_root / "scripts" / "training"))

# Import Intelligent Logging System
try:
    from trinity_core.intelligent_logger import get_logger
    LOGGING_AVAILABLE = True
    print("✅ Intelligent Logging System imported successfully")
except ImportError as e:
    print(f"⚠️ Intelligent Logging import failed: {e}")
    LOGGING_AVAILABLE = False

INTELLIGENT_LOGGING_AVAILABLE = True

@dataclass
class TrainingResult:
    """Class to hold training result information"""
    status: str
    message: str
    results: dict = None

class RealTrinityProductionLauncher:
    """The main class for launching and managing the Trinity training ecosystem."""
    
    def __init__(self):
        """Initialize the launcher, including the complete agent ecosystem."""
        self.trinity_ecosystem = CompleteAgentEcosystem()
        self.simulation = False  # Ensure attribute is initialized
        self.logger = None
        self.config_manager = None
        self.session_start_time = time.time()
        
        # Initialize Trinity ecosystem
        self.trinity_ecosystem = CompleteAgentEcosystem()
        
        try:
            # Initialize intelligent logging and config
            self.config_manager = SmartTrinityConfigManager()
            self.logger = IntelligentLogger(domain="trinity_session") # Initialize main session logger
            print("📊 Intelligent Logging System ready")
        except Exception as e:
            print(f"⚠️ Intelligent logging not available: {e}")
            
    def _initialize_domain_logging(self, domain):
        """Initializes the intelligent logger for a specific domain."""
        try:
            domain_config = self.config_manager.get_tara_proven_params(domain)
            session_id = f"{domain}_{int(time.time())}"

            logger = IntelligentLogger(
                domain=domain,
                session_id=session_id,
                start_time=datetime.now().isoformat(),
                is_simulation=self.simulation,
                total_domains_in_config=len(self.config_manager.get_all_domains_flat()),
                is_valid=True,  # If we got this far, it's valid
                category=domain_config.get("category", "N/A"),
                model_tier=domain_config.get("model_tier", "N/A"),
                base_model=domain_config.get("base_model", "N/A"),
                sample_count=domain_config["sample_count"],
                max_steps=domain_config["max_steps"],
                num_epochs=domain_config.get("num_epochs", "N/A"),
                learning_rate=domain_config["learning_rate"],
            )
            
            # Log model selection
            logger.log_model_selection(
                domain=domain,
                base_model=domain_config["base_model"],
                model_tier=domain_config["model_tier"],
                selection_reason=f"{domain_config['model_tier'].title()} tier model selected for {domain_config.get('category', 'unknown')} category - optimized for domain requirements"
            )
            
            # Log parameter generation
            logger.log_parameter_generation(
                domain=domain,
                model_tier=domain_config["model_tier"],
                parameters={
                    "batch_size": domain_config["batch_size"],
                    "lora_r": domain_config["lora_r"],
                    "max_steps": domain_config["max_steps"],
                    "learning_rate": domain_config["learning_rate"],
                    "sample_count": domain_config["sample_count"],
                    "quality_target": domain_config["validation_target"],
                    "gradient_accumulation": domain_config.get("gradient_accumulation", 4),
                    "warmup_steps": domain_config.get("warmup_steps", 84)
                },
                source="YAML_TIER_SPECIFIC"
            )
            
            # Log training decisions
            logger.log_decision(
                decision_type="Model Selection",
                decision=f"Selected {domain_config['base_model']} for {domain}",
                reasoning=f"{domain_config['model_tier'].title()} tier model provides optimal performance for {domain_config.get('category', 'unknown')} domain requirements"
            )
            
            logger.log_decision(
                decision_type="Parameter Optimization",
                decision=f"Using tier-specific parameters: batch_size={domain_config['batch_size']}, max_steps={domain_config['max_steps']}, lora_r={domain_config['lora_r']}",
                reasoning=f"Parameters optimized for {domain_config['model_tier']} tier based on model size and domain complexity"
            )
        
            return logger
            
        except (ValueError, KeyError) as e:
            print(f"❌ Error initializing logger for domain '{domain}': {e}")
            return None
        
    def _log_training_progress(self, logger, domain: str, step: int, total_steps: int, loss: float, speed: float):
        """Log training progress"""
        if not logger:
            return
            
        # Calculate accuracy estimate (simplified)
        accuracy = max(0.0, min(1.0, 1.0 - (loss / 10.0)))
        
        # Log training step
        logger.log_training_step(step, loss, accuracy, None)
        
        # Log progress decisions
        if step % 200 == 0:
            progress = (step / total_steps) * 100
            logger.log_decision(
                decision_type="Training Progress",
                decision=f"Step {step}/{total_steps} ({progress:.1f}%)",
                reasoning=f"Loss: {loss:.4f}, Speed: {speed:.1f}x, Accuracy: {accuracy:.2%}"
            )
            
    def _log_training_completion(self, logger, domain: str, training_result: dict):
        """Log training completion"""
        if not logger:
            return
            
        # Log sample generation (from training result)
        samples_generated = training_result.get('samples_generated', 0)
        generation_time = training_result.get('total_time_seconds', 0) * 0.1  # Estimate 10% for data gen
        
        logger.log_sample_generation(
            domain=domain,
            target_samples=samples_generated,
            generated_samples=samples_generated,
            quality_score=0.95,  # This could also be dynamic if available
            generation_time=generation_time
        )
        
        # Log GGUF creation for each variant produced
        model_variants = training_result.get("model_variants_created", [])
        if not model_variants:
            logger.warning(f"No model variants found in the result for domain {domain}.")

        for variant in model_variants:
            if variant.get("status") == "success":
                # Extract real data from the variant result
                gguf_info = {
                    "format": variant.get("quantization_used", "UNKNOWN"),
                    "size": variant.get("model_size_mb", 0),
                    "compression": variant.get("quantization_used", "UNKNOWN"),
                    "quality": variant.get("configuration_used", {}).get("metadata", {}).get("data_analysis", {}).get("quality_score", 0) * 100,
                    "filename": Path(variant.get("output_path", "")).name,
                    "model_path": variant.get("output_path", "N/A"),
                    "model_size_mb": variant.get("model_size_mb", 0)
                }
                logger.log_gguf_creation(domain=domain, gguf_info=gguf_info)

        # Log quality validation - this could be made more sophisticated
        # For now, we'll use the quality from the first successful variant
        successful_variants = [v for v in model_variants if v.get("status") == "success"]
        if successful_variants:
            first_variant = successful_variants[0]
            quality_score = first_variant.get("configuration_used", {}).get("metadata", {}).get("data_analysis", {}).get("quality_score", 0) * 100
            target_quality = 95.0 # This could also be dynamic from config
            
            logger.log_quality_validation(
                domain=domain,
                quality_score=quality_score,
                quality_target=target_quality,
                passed=quality_score >= target_quality
            )
            
            # Log final decisions
            logger.log_decision(
                decision_type="Training Completion",
                decision=f"Training completed for {domain}",
                reasoning=f"Final quality: {quality_score:.1f}%, Time: {training_result.get('total_time_seconds', 0):.1f}s, Variants: {len(successful_variants)}"
            )

        # Complete session
        logger.log_session_summary()
            
    async def launch_trinity_training(self, category: str = None, domains: list = None, simulation: bool = False):
        """Launch real Trinity training with Intelligence Hub and Comprehensive Logging"""
        self.simulation = simulation  # Set simulation status for the instance

        if not LOGGING_AVAILABLE or not self.logger:
            print("❌ Intelligent logging not available")
            return TrainingResult(status="error", message="Intelligent logging not available")
            
        print(f"\n🚀 LAUNCHING TRINITY SUPER-AGENT TRAINING WITH INTELLIGENT LOGGING")
        print("="*70)
        print("🧠 Intelligence Hub: Analyzing domain patterns...")
        print("🏭 Data Generator: Preparing real-time data generation...")
        print("🎯 Training Orchestrator: Coordinating multi-domain training...")
        print("🔍 Quality Assurance: Setting up validation pipelines...")
        print("📊 Intelligent Logging: Capturing all decisions and processes...")
        print("="*70)
        
        # Add simulation status to the log
        if simulation:
            print("🚦 RUNNING IN SIMULATION MODE - NO REAL TRAINING WILL OCCUR")
        
        # Determine domains to train
        if category:
            # Get domains for specific category
            category_domains = self._get_domains_for_category(category)
            domains_to_train = category_domains
            print(f"🎯 Training category: {category.upper()}")
            print(f"   → Domains: {domains_to_train}")
        elif domains:
            domains_to_train = domains
            print(f"🎯 Training specific domains: {domains_to_train}")
        else:
            # Train all 62 domains
            domains_to_train = None  # Will train all
            print("🌍 Training ALL 62 domains across 7 categories")
        
        # Initialize logging for each domain
        domain_loggers = {}
        if domains_to_train:
            for domain in domains_to_train:
                print(f"\n📊 Initializing intelligent logging for {domain}...")
                domain_loggers[domain] = self._initialize_domain_logging(domain)
        
        # Launch Trinity ecosystem training
        print(f"\n🚀 Launching Trinity ecosystem training...")
        result = await self.trinity_ecosystem.coordinate_complete_training(
            domains_to_train=domains_to_train,
            simulation=simulation
        )
        
        # Log training results
        if result and result.get('results'):
            for domain_result in result['results']:
                domain = domain_result['domain']
                logger = domain_loggers.get(domain)
                
                if logger and domain_result.get('training_result'):
                    print(f"📊 Logging training completion for {domain}...")
                    self._log_training_completion(logger, domain, domain_result['training_result'])
        
        # Log overall session summary
        if LOGGING_AVAILABLE:
            total_time = time.time() - self.session_start_time
            print(f"\n📊 COMPREHENSIVE LOGGING SUMMARY")
            print(f"   → Total session time: {total_time:.1f}s")
            print(f"   → Domains processed: {len(domains_to_train) if domains_to_train else 'ALL'}")
            print(f"   → Log files created in: logs/ directory")
            print(f"   → Each domain has detailed logs for:")
            print(f"     • Model selection reasoning")
            print(f"     • Parameter generation explanations")
            print(f"     • Training progress and decisions")
            print(f"     • Quality validation results")
            print(f"     • Complete session summaries")
        
        return result
        
    def _get_domains_for_category(self, category: str) -> list:
        """Get domains for a specific category"""
        domain_mapping = {
            "healthcare": [
                "general_health", "mental_health", "nutrition", "fitness", "sleep",
                "stress_management", "preventive_care", "chronic_conditions", 
                "medication_management", "emergency_care", "women_health", "senior_health"
            ],
            "daily_life": [
                "parenting", "relationships", "personal_assistant", "communication",
                "home_management", "shopping", "planning", "transportation",
                "time_management", "decision_making", "conflict_resolution", "work_life_balance"
            ],
            "business": [
                "entrepreneurship", "marketing", "sales", "customer_service",
                "project_management", "team_leadership", "financial_planning", "operations",
                "hr_management", "strategy", "consulting", "legal_business"
            ],
            "education": [
                "academic_tutoring", "skill_development", "career_guidance", 
                "exam_preparation", "language_learning", "research_assistance",
                "study_techniques", "educational_technology"
            ],
            "creative": [
                "writing", "storytelling", "content_creation", "social_media",
                "design_thinking", "photography", "music", "art_appreciation"
            ],
            "technology": [
                "programming", "ai_ml", "cybersecurity", "data_analysis",
                "tech_support", "software_development"
            ],
            "specialized": [
                "legal", "financial", "scientific_research", "engineering"
            ]
        }
        
        return domain_mapping.get(category, [])

async def main():
    """Main function to run the Trinity Production Launcher"""
    parser = argparse.ArgumentParser(
        description="Launch real Trinity training sessions with the Super-Agent ecosystem.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Group for domain selection
    domain_group = parser.add_mutually_exclusive_group(required=True)
    domain_group.add_argument(
        '--category', 
        type=str, 
        help="Train all domains in a specific category (e.g., 'healthcare')"
    )
    domain_group.add_argument(
        '--domains', 
        nargs='+', 
        help="Train a specific list of domains (e.g., 'symptom_checker' 'financial_advisor')"
    )
    domain_group.add_argument(
        '--all', 
        action='store_true', 
        help="Train all 62 domains"
    )
    
    # Simulation flag
    parser.add_argument(
        '--simulation', 
        action='store_true', 
        help="Run in simulation mode (generates artifacts but no real training)"
    )

    args = parser.parse_args()
    
    launcher = RealTrinityProductionLauncher()
    
    # Determine domains to run
    if args.category:
        domains_to_run = launcher._get_domains_for_category(args.category)
        print(f"🎯 Training category: {args.category.upper()}")
        print(f"   → Domains: {domains_to_run}")
    elif args.domains:
        domains_to_run = args.domains
        print(f"🎯 Training specific domains: {domains_to_run}")
    elif args.all:
        domains_to_run = launcher.config_manager.get_all_domains_flat() if launcher.config_manager else []
        print("🌍 Training ALL 62 domains across 7 categories")
    else:
        # This case should not be reached due to the mutually exclusive group
        print("❌ Please specify --category, --domains, or --all.")
        return

    await launcher.launch_trinity_training(
        category=args.category, 
        domains=domains_to_run,
        simulation=args.simulation
    )

if __name__ == "__main__":
    asyncio.run(main())
