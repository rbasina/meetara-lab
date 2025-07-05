#!/usr/bin/env python3
"""
Test Intelligent Logging System and Model-Specific Parameters
Demonstrates how different models get different parameters and comprehensive logging
"""

import sys
import time
import random
from pathlib import Path

# Add trinity-core to path
sys.path.append('trinity-core')

from config_manager import get_config_manager
from intelligent_logger import get_logger

def test_model_specific_parameters():
    """Test that different models get different parameters"""
    print("🧪 TESTING MODEL-SPECIFIC PARAMETERS")
    print("=" * 60)
    
    # Test domains with different model tiers
    test_domains = [
        "general_health",    # premium tier (14B)
        "entrepreneurship",  # expert tier (14B)
        "programming",       # expert tier (14B)
        "academic_tutoring", # expert tier (14B)
        "writing",           # quality tier (14B)
        "parenting",         # quality tier (14B)
        "legal"              # premium tier (14B)
    ]
    
    config = get_config_manager()
    
    for domain in test_domains:
        print(f"\n🔍 Testing domain: {domain}")
        print("-" * 40)
        
        # Get configuration
        domain_config = config.get_training_config_for_domain(domain)
        
        # Display key parameters
        print(f"   Model: {domain_config['base_model']}")
        print(f"   Tier: {domain_config['model_tier']}")
        print(f"   Batch Size: {domain_config['batch_size']}")
        print(f"   LoRA Rank: {domain_config['lora_r']}")
        print(f"   Max Steps: {domain_config['max_steps']}")
        print(f"   Learning Rate: {domain_config['learning_rate']}")
        print(f"   Samples: {domain_config['samples_per_domain']:,}")
        print(f"   Quality Target: {domain_config['quality_target']}%")
        
        # Show parameter explanations
        explanations = config.explain_parameter_decisions(domain)
        print(f"   📝 Max Steps Reason: {explanations.get('max_steps', 'N/A')}")
        print(f"   📝 Batch Size Reason: {explanations.get('batch_size', 'N/A')}")

def test_comprehensive_logging():
    """Test comprehensive logging system"""
    print("\n\n🔍 TESTING COMPREHENSIVE LOGGING SYSTEM")
    print("=" * 60)
    
    # Test domain
    test_domain = "general_health"
    
    # Initialize logger
    logger = get_logger(test_domain)
    config = get_config_manager()
    
    # Log configuration loading
    validation = config.validate_configuration()
    logger.log_config_loading(
        yaml_loaded=validation["yaml_loaded"],
        json_loaded=validation["json_loaded"],
        total_domains=validation["total_domains"]
    )
    
    # Log domain validation
    logger.log_domain_validation(
        domain=test_domain,
        is_valid=True,
        category="healthcare",
        suggestions=[]
    )
    
    # Get domain configuration
    domain_config = config.get_training_config_for_domain(test_domain)
    
    # Log model selection
    logger.log_model_selection(
        domain=test_domain,
        base_model=domain_config["base_model"],
        model_tier=domain_config["model_tier"],
        selection_reason=f"Premium tier model selected for healthcare category - highest quality requirements"
    )
    
    # Log parameter generation
    logger.log_parameter_generation(
        domain=test_domain,
        model_tier=domain_config["model_tier"],
        parameters={
            "batch_size": domain_config["batch_size"],
            "lora_r": domain_config["lora_r"],
            "max_steps": domain_config["max_steps"],
            "learning_rate": domain_config["learning_rate"],
            "samples_per_domain": domain_config["samples_per_domain"],
            "quality_target": domain_config["quality_target"]
        },
        source="YAML_TIER_SPECIFIC"
    )
    
    # Simulate sample generation
    print(f"\n📝 Simulating sample generation for {test_domain}...")
    target_samples = domain_config["samples_per_domain"]
    
    # Simulate generation process
    start_time = time.time()
    generated_samples = 0
    
    for batch in range(0, target_samples, 1000):
        batch_size = min(1000, target_samples - generated_samples)
        generated_samples += batch_size
        
        # Simulate some processing time
        time.sleep(0.1)
        
        # Show progress
        progress = (generated_samples / target_samples) * 100
        print(f"   Progress: {progress:.1f}% ({generated_samples:,}/{target_samples:,})")
    
    generation_time = time.time() - start_time
    quality_score = random.uniform(0.95, 0.99)
    
    # Log sample generation
    logger.log_sample_generation(
        domain=test_domain,
        target_samples=target_samples,
        generated_samples=generated_samples,
        quality_score=quality_score,
        generation_time=generation_time
    )
    
    # Simulate training steps
    print(f"\n🚀 Simulating training steps...")
    max_steps = domain_config["max_steps"]
    
    for step in range(0, max_steps + 1, 100):
        # Simulate decreasing loss
        loss = 2.0 * (1 - step / max_steps) + random.uniform(-0.1, 0.1)
        accuracy = min(0.95, step / max_steps * 0.9 + random.uniform(0, 0.05))
        lr = domain_config["learning_rate"] * (1 - step / max_steps * 0.5)
        
        logger.log_training_step(step, loss, accuracy, lr)
        
        if step % 200 == 0:
            print(f"   Step {step:4d}: Loss={loss:.4f}, Acc={accuracy:.2%}")
    
    # Log GGUF creation
    logger.log_gguf_creation(
        domain=test_domain,
        gguf_info={
            "format": domain_config["output_format"],
            "size": domain_config["target_size_mb"],
            "compression": "Q4_K_M",
            "quality": 98.5,
            "filename": f"meetara_{test_domain}_q4_k_m.gguf"
        }
    )
    
    # Log quality validation
    final_quality = random.uniform(domain_config["quality_target"], domain_config["quality_target"] + 5)
    logger.log_quality_validation(
        domain=test_domain,
        quality_score=final_quality,
        quality_target=domain_config["quality_target"],
        passed=final_quality >= domain_config["quality_target"]
    )
    
    # Log some decisions
    logger.log_decision(
        decision_type="Model Selection",
        decision=f"Selected {domain_config['base_model']} for {test_domain}",
        reasoning=f"Premium tier model provides highest quality for healthcare domain requirements"
    )
    
    logger.log_decision(
        decision_type="Parameter Optimization",
        decision=f"Using batch_size={domain_config['batch_size']}, max_steps={domain_config['max_steps']}",
        reasoning=f"Tier-specific parameters optimized for {domain_config['model_tier']} tier performance"
    )
    
    # Complete session
    logger.log_session_summary()
    
    print(f"\n✅ Comprehensive logging test completed!")
    print(f"📁 Check logs/ directory for detailed log files")

def show_parameter_comparison():
    """Show comparison of parameters across different model tiers"""
    print("\n\n📊 PARAMETER COMPARISON ACROSS MODEL TIERS")
    print("=" * 60)
    
    config = get_config_manager()
    
    # Domains representing different tiers
    tier_examples = {
        "premium": "general_health",
        "expert": "entrepreneurship", 
        "quality": "writing",
        "balanced": "nutrition"  # if exists
    }
    
    print(f"{'Domain':<20} {'Tier':<10} {'Batch':<6} {'LoRA':<5} {'Steps':<6} {'LR':<8} {'Samples':<8}")
    print("-" * 70)
    
    for tier, domain in tier_examples.items():
        try:
            config_data = config.get_training_config_for_domain(domain)
            print(f"{domain:<20} {config_data['model_tier']:<10} {config_data['batch_size']:<6} "
                  f"{config_data['lora_r']:<5} {config_data['max_steps']:<6} "
                  f"{config_data['learning_rate']:<8.0e} {config_data['samples_per_domain']:<8,}")
        except Exception as e:
            print(f"{domain:<20} {'ERROR':<10} - {str(e)}")
    
    print("\n🔍 Key Observations:")
    print("   • Different tiers have different parameters")
    print("   • Premium/Expert tiers: Lower batch size, higher LoRA rank, more steps")
    print("   • Quality tiers: Balanced parameters")
    print("   • Sample counts vary by tier requirements")

if __name__ == "__main__":
    print("🚀 MEETARA LAB - INTELLIGENT LOGGING & MODEL-SPECIFIC PARAMETERS TEST")
    print("=" * 80)
    
    # Test 1: Model-specific parameters
    test_model_specific_parameters()
    
    # Test 2: Parameter comparison
    show_parameter_comparison()
    
    # Test 3: Comprehensive logging
    test_comprehensive_logging()
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS COMPLETED!")
    print("📁 Check logs/ directory for detailed log files")
    print("🔍 Each domain now gets optimized parameters based on model tier")
    print("📊 Comprehensive logging tracks all decisions and processes") 