#!/usr/bin/env python3
"""
Trinity Flow Test - Fixed Version
Test the complete Trinity flow with correct import paths
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

# Add correct project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root / "trinity-core"))
sys.path.append(str(project_root / "trinity-core" / "agents"))
sys.path.append(str(project_root / "model-factory"))
sys.path.append(str(project_root / "cloud-training"))

def test_trinity_flow_fixed():
    """Test the complete Trinity flow with correct paths"""
    
    print("🧪 TRINITY FLOW TEST - FIXED VERSION")
    print("=" * 50)
    print("Testing: chronic_conditions (healthcare)")
    print("=" * 50)
    
    # Step 1: Test Trinity Data Generation (Fixed Paths)
    print("\n1️⃣ TESTING TRINITY DATA GENERATION")
    print("-" * 30)
    
    try:
        # Try different import paths for Trinity agents
        import_paths = [
            "trinity-core.agents.complete_agent_ecosystem",
            "complete_agent_ecosystem", 
            "04_system_integration.02_complete_agent_ecosystem",
            "system_integration.complete_agent_ecosystem"
        ]
        
        ecosystem = None
        for import_path in import_paths:
            try:
                if "." in import_path:
                    module_parts = import_path.split(".")
                    module = __import__(import_path, fromlist=[module_parts[-1]])
                    if hasattr(module, 'TrinityAgentEcosystem'):
                        ecosystem = module.TrinityAgentEcosystem()
                        print(f"✅ Trinity Agent Ecosystem imported from: {import_path}")
                        break
                else:
                    module = __import__(import_path)
                    if hasattr(module, 'TrinityAgentEcosystem'):
                        ecosystem = module.TrinityAgentEcosystem()
                        print(f"✅ Trinity Agent Ecosystem imported from: {import_path}")
                        break
            except ImportError:
                continue
        
        if ecosystem:
            print("🏭 Testing training data generation...")
            # Test with simple domain
            domain = "chronic_conditions"
            category = "healthcare"
            
            # Check if ecosystem has the right methods
            if hasattr(ecosystem, 'generate_training_data'):
                training_data = ecosystem.generate_training_data(domain, category, samples=10)
                print(f"✅ Generated {len(training_data)} training samples")
            elif hasattr(ecosystem, 'train_domains'):
                result = ecosystem.train_domains([domain], category)
                print(f"✅ Domain training result: {result.get('status', 'Unknown')}")
            else:
                print("❌ No suitable training method found in ecosystem")
        else:
            print("❌ Trinity Agent Ecosystem not found in any import path")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Step 2: Test GPU Training Engine (Fixed Paths)
    print("\n2️⃣ TESTING GPU TRAINING ENGINE")
    print("-" * 30)
    
    try:
        from gpu_training_engine import GPUTrainingEngine, GPUTrainingConfig
        
        config = GPUTrainingConfig(
            domain="chronic_conditions",
            base_model="microsoft/DialoGPT-medium",
            batch_size=2,
            max_steps=50,
            target_model_size_mb=8.3
        )
        
        engine = GPUTrainingEngine(config)
        print("✅ GPU Training Engine imported successfully")
        print(f"   Device: {engine.device}")
        
        # Test with minimal training data
        sample_texts = ["Managing chronic conditions effectively"]
        
        print("🧠 Testing model training...")
        training_result = engine.train_model(sample_texts)
        
        if training_result.get("training_completed"):
            print("✅ Model training completed")
            print(f"   Speed improvement: {training_result.get('speed_improvement', 0):.1f}x")
        else:
            print("❌ Model training failed")
            
    except ImportError as e:
        print(f"❌ Import Error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Step 3: Test GGUF Factory (Fixed Paths)
    print("\n3️⃣ TESTING GGUF FACTORY")
    print("-" * 30)
    
    try:
        # Try different GGUF factory imports
        gguf_imports = [
            "universal_gguf_factory.UniversalGGUFFactory",
            "model-factory.universal_gguf_factory.UniversalGGUFFactory",
            "scripts.gguf_factory.universal_gguf_factory.UniversalGGUFFactory"
        ]
        
        factory = None
        for import_path in gguf_imports:
            try:
                module_path, class_name = import_path.rsplit(".", 1)
                module = __import__(module_path, fromlist=[class_name])
                factory_class = getattr(module, class_name)
                factory = factory_class()
                print(f"✅ GGUF Factory imported from: {import_path}")
                break
            except (ImportError, AttributeError):
                continue
        
        if factory:
            print("🏭 Testing GGUF creation...")
            
            training_result = {
                "domain": "chronic_conditions",
                "base_model": "microsoft/DialoGPT-medium",
                "training_completed": True,
                "speed_improvement": 37.0,
                "final_loss": 0.15,
                "training_samples": 10
            }
            
            # Check available methods
            if hasattr(factory, 'create_gguf_model'):
                gguf_result = factory.create_gguf_model("chronic_conditions", training_result)
                print(f"✅ GGUF creation result: {gguf_result.get('status', 'Unknown')}")
            elif hasattr(factory, 'create_domain_gguf'):
                gguf_result = factory.create_domain_gguf("chronic_conditions", training_result)
                print(f"✅ GGUF creation result: {gguf_result.get('status', 'Unknown')}")
            else:
                print("❌ No suitable GGUF creation method found")
        else:
            print("❌ GGUF Factory not found in any import path")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Step 4: Test Production Launcher (Fixed Paths)
    print("\n4️⃣ TESTING PRODUCTION LAUNCHER")
    print("-" * 30)
    
    try:
        from production_launcher import ProductionLauncher
        
        launcher = ProductionLauncher()
        print("✅ Production Launcher imported successfully")
        
        # Check available methods
        if hasattr(launcher, 'train_single_domain'):
            result = launcher.train_single_domain("chronic_conditions", "healthcare")
            print(f"✅ Single domain training: {result.get('success', False)}")
        elif hasattr(launcher, 'launch_training'):
            result = launcher.launch_training(domains=["chronic_conditions"], category="healthcare")
            print(f"✅ Training launched: {result.get('status', 'Unknown')}")
        else:
            print("❌ No suitable training method found in launcher")
            
    except ImportError as e:
        print(f"❌ Import Error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Summary and Recommendations
    print("\n🎯 TRINITY FLOW DIAGNOSIS COMPLETE")
    print("=" * 50)
    print("RECOMMENDATIONS:")
    print("1. Fix import paths in all Trinity components")
    print("2. Ensure agent ecosystem is properly connected")
    print("3. Verify GGUF factory integration")
    print("4. Test end-to-end pipeline with corrected paths")
    print("=" * 50)

if __name__ == "__main__":
    test_trinity_flow_fixed() 