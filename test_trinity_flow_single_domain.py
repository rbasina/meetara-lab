#!/usr/bin/env python3
"""
Trinity Flow Test - Single Domain
Test the complete Trinity flow for one domain to identify missing components
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root / "trinity-core"))
sys.path.append(str(project_root / "model-factory"))
sys.path.append(str(project_root / "cloud-training"))

def test_trinity_flow_single_domain():
    """Test the complete Trinity flow for chronic_conditions domain"""
    
    print("🧪 TRINITY FLOW TEST - SINGLE DOMAIN")
    print("=" * 50)
    print("Testing: chronic_conditions (healthcare)")
    print("=" * 50)
    
    # Step 1: Test Trinity Data Generation
    print("\n1️⃣ TESTING TRINITY DATA GENERATION")
    print("-" * 30)
    
    try:
        # Import Trinity Agent Ecosystem
        from trinity_core.agents.complete_agent_ecosystem import TrinityAgentEcosystem
        
        # Initialize ecosystem
        ecosystem = TrinityAgentEcosystem()
        print("✅ Trinity Agent Ecosystem imported successfully")
        
        # Generate training data for chronic_conditions
        domain = "chronic_conditions"
        category = "healthcare"
        
        print(f"🏭 Generating training data for {domain}...")
        training_data = ecosystem.generate_training_data(domain, category, samples=100)
        
        if training_data:
            print(f"✅ Generated {len(training_data)} training samples")
            print(f"   Sample: {training_data[0][:100]}...")
        else:
            print("❌ No training data generated")
            
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("   Trinity Agent Ecosystem not found or misconfigured")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Step 2: Test GPU Training Engine
    print("\n2️⃣ TESTING GPU TRAINING ENGINE")
    print("-" * 30)
    
    try:
        from gpu_training_engine import GPUTrainingEngine, GPUTrainingConfig
        
        # Create training config
        config = GPUTrainingConfig(
            domain="chronic_conditions",
            base_model="microsoft/DialoGPT-medium",
            batch_size=2,
            max_steps=100,
            target_model_size_mb=8.3
        )
        
        # Initialize engine
        engine = GPUTrainingEngine(config)
        print("✅ GPU Training Engine imported successfully")
        print(f"   Device: {engine.device}")
        
        # Test training with sample data
        sample_texts = [
            "How to manage chronic diabetes symptoms?",
            "What are the best treatments for hypertension?",
            "Managing arthritis pain naturally"
        ]
        
        print("🧠 Testing model training...")
        training_result = engine.train_model(sample_texts)
        
        if training_result.get("training_completed"):
            print("✅ Model training completed")
            print(f"   Speed improvement: {training_result.get('speed_improvement', 0):.1f}x")
        else:
            print("❌ Model training failed")
            
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("   GPU Training Engine not found or misconfigured")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Step 3: Test GGUF Factory
    print("\n3️⃣ TESTING GGUF FACTORY")
    print("-" * 30)
    
    try:
        from universal_gguf_factory import UniversalGGUFFactory
        
        # Initialize factory
        factory = UniversalGGUFFactory()
        print("✅ Universal GGUF Factory imported successfully")
        
        # Test GGUF creation
        print("🏭 Testing GGUF creation...")
        
        # Sample training result
        training_result = {
            "domain": "chronic_conditions",
            "base_model": "microsoft/DialoGPT-medium",
            "training_completed": True,
            "speed_improvement": 37.0,
            "final_loss": 0.15,
            "training_samples": 100
        }
        
        gguf_result = factory.create_gguf_model("chronic_conditions", training_result)
        
        if gguf_result.get("status") == "success":
            print("✅ GGUF creation completed")
            print(f"   File size: {gguf_result.get('final_size_mb', 0):.1f}MB")
            print(f"   Output path: {gguf_result.get('output_path', 'N/A')}")
        else:
            print("❌ GGUF creation failed")
            
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("   Universal GGUF Factory not found or misconfigured")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Step 4: Test Complete Integration
    print("\n4️⃣ TESTING COMPLETE INTEGRATION")
    print("-" * 30)
    
    try:
        from production_launcher import ProductionLauncher
        
        # Initialize launcher
        launcher = ProductionLauncher()
        print("✅ Production Launcher imported successfully")
        
        # Test single domain training
        print("🚀 Testing complete pipeline...")
        
        result = launcher.train_single_domain("chronic_conditions", "healthcare")
        
        if result.get("success"):
            print("✅ Complete pipeline successful")
            print(f"   Training: {result.get('training_status', 'Unknown')}")
            print(f"   GGUF: {result.get('gguf_status', 'Unknown')}")
        else:
            print("❌ Complete pipeline failed")
            
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("   Production Launcher not found or misconfigured")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Summary
    print("\n🎯 TRINITY FLOW DIAGNOSIS COMPLETE")
    print("=" * 50)
    print("Check the results above to identify missing components")
    print("=" * 50)

if __name__ == "__main__":
    test_trinity_flow_single_domain() 