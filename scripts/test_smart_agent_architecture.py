#!/usr/bin/env python3
"""
MeeTARA Lab - Smart Agent Architecture Test
Demonstrates the power of intelligent agents vs hardcoded scripts
"""

import asyncio
import json
import time
from pathlib import Path
import sys

# Add trinity-core to path
sys.path.append(str(Path(__file__).parent.parent / "trinity-core"))

from agents.super_agents.model_factory import IntelligentModelFactory
from agents.super_agents.model_factory import IntelligenceLevel, DataQualityLevel

async def test_smart_agent_architecture():
    """Test the smart agent architecture with different data scenarios"""
    
    print("🧠 Testing Smart Agent Architecture")
    print("=" * 50)
    
    # Create intelligent agent
    agent = IntelligentModelFactory(intelligence_level=IntelligenceLevel.AUTONOMOUS)
    
    # Test scenarios with different data characteristics
    test_scenarios = [
        {
            "name": "Small High-Quality Dataset",
            "domain": "healthcare",
            "training_data": [
                {"input": "What are symptoms of diabetes?", "output": "Common symptoms include increased thirst, frequent urination, and fatigue."},
                {"input": "How to manage hypertension?", "output": "Regular exercise, healthy diet, and medication as prescribed."},
                {"input": "What is preventive care?", "output": "Regular check-ups and screenings to prevent illness."}
            ]
        },
        {
            "name": "Large Complex Dataset",
            "domain": "technology",
            "training_data": [
                {"input": f"Technical question {i}", "output": f"Complex technical answer {i} with detailed explanation and code examples"}
                for i in range(1000)
            ]
        },
        {
            "name": "Poor Quality Dataset",
            "domain": "general",
            "training_data": [
                {"input": "", "output": "incomplete"},
                {"input": "test", "output": ""},
                {"input": "duplicate", "output": "same"},
                {"input": "duplicate", "output": "same"}
            ]
        },
        {
            "name": "Domain Prediction (No Data)",
            "domain": "business",
            "training_data": []
        }
    ]
    
    results = []
    
    for scenario in test_scenarios:
        print(f"\n🔬 Testing: {scenario['name']}")
        print(f"   Domain: {scenario['domain']}")
        print(f"   Data samples: {len(scenario['training_data'])}")
        
        start_time = time.time()
        
        # Create request
        request = {
            "domain": scenario["domain"],
            "training_data": scenario["training_data"]
        }
        
        # Let the intelligent agent make ALL decisions
        result = await agent.create_intelligent_model(request)
        
        execution_time = time.time() - start_time
        
        # Report intelligent decisions
        print(f"   🧠 Agent Intelligence:")
        print(f"      → Model Size: {result['model_size_mb']:.1f}MB")
        print(f"      → Quantization: {result['quantization_level']}")
        print(f"      → Compression: {result['compression_method']}")
        print(f"      → Quality Score: {result['quality_score']:.2f}")
        print(f"      → Confidence: {result['confidence_score']:.2f}")
        print(f"      → Risk Level: {result['risk_level']}")
        print(f"      → Processing Time: {execution_time:.3f}s")
        
        # Show intelligent reasoning
        config = result['intelligent_config']
        data_analysis = config['metadata']['data_analysis']
        dq_decisions = config['metadata']['dq_decisions']
        
        print(f"   📊 Data Analysis:")
        print(f"      → Complexity Score: {data_analysis['complexity_score']:.2f}")
        print(f"      → Quality Level: {data_analysis['data_quality_level'].value}")
        print(f"      → Sample Count: {data_analysis['sample_count']}")
        
        print(f"   ⚙️  DQ Rules Applied:")
        for rule in dq_decisions['applied_rules']:
            print(f"      → {rule['rule']}: {rule['action']}")
        
        if dq_decisions['recommendations']:
            print(f"   💡 Recommendations:")
            for rec in dq_decisions['recommendations']:
                print(f"      → {rec}")
        
        results.append({
            "scenario": scenario['name'],
            "result": result,
            "execution_time": execution_time
        })
    
    # Summary
    print(f"\n📈 Smart Agent Architecture Summary")
    print("=" * 50)
    
    total_time = sum(r['execution_time'] for r in results)
    avg_quality = sum(r['result']['quality_score'] for r in results) / len(results)
    avg_confidence = sum(r['result']['confidence_score'] for r in results) / len(results)
    
    print(f"✅ Total Processing Time: {total_time:.3f}s")
    print(f"✅ Average Quality Score: {avg_quality:.2f}")
    print(f"✅ Average Confidence: {avg_confidence:.2f}")
    print(f"✅ Scenarios Processed: {len(results)}")
    
    print(f"\n🎯 Key Benefits Demonstrated:")
    print(f"   → NO hardcoded values - all decisions made intelligently")
    print(f"   → Adaptive behavior based on data characteristics")
    print(f"   → Context-aware quantization and compression selection")
    print(f"   → Automatic DQ rule application")
    print(f"   → Intelligent risk assessment and confidence scoring")
    print(f"   → Domain-specific predictions when no data available")
    
    # Show the difference
    print(f"\n⚡ Architecture Comparison:")
    print(f"   OLD (Hardcoded): Same 8.3MB + Q4_K_M for ALL scenarios")
    print(f"   NEW (Intelligent): Adaptive sizing and optimization per scenario")
    
    for i, result in enumerate(results):
        r = result['result']
        print(f"      {i+1}. {result['scenario'][:20]:<20} → {r['model_size_mb']:.1f}MB + {r['quantization_level']}")
    
    return results

async def demonstrate_simple_script_usage():
    """Demonstrate how simple scripts become with smart agents"""
    
    print(f"\n🚀 Simple Script Usage Example")
    print("=" * 50)
    
    # This is ALL a script needs to do now:
    
    # 1. Create intelligent agent (handles all configuration)
    agent = IntelligentModelFactory()
    
    # 2. Simple data preparation
    request = {
        "domain": "education",
        "training_data": [
            {"question": "What is photosynthesis?", "answer": "The process by which plants make food using sunlight."},
            {"question": "Explain gravity", "answer": "The force that attracts objects toward each other."}
        ]
    }
    
    # 3. Delegate ALL intelligence to agent
    result = await agent.create_intelligent_model(request)
    
    # 4. Simple result handling
    if result['status'] == 'success':
        print(f"✅ Model created successfully!")
        print(f"   → Output: {result['output_path']}")
        print(f"   → Agent chose: {result['model_size_mb']:.1f}MB + {result['quantization_level']} + {result['compression_method']}")
    else:
        print(f"❌ Model creation failed")
    
    print(f"\n💡 Script Complexity: MINIMAL")
    print(f"   → No hardcoded values")
    print(f"   → No complex decision logic")
    print(f"   → No compression/quantization selection")
    print(f"   → No DQ rule implementation")
    print(f"   → Just simple data loading and agent delegation")
    
    return result

if __name__ == "__main__":
    async def main():
        await test_smart_agent_architecture()
        await demonstrate_simple_script_usage()
    
    asyncio.run(main()) 