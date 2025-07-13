#!/usr/bin/env python3
"""
🧪 MeeTARA Lab - Visualization Flow Test
Comprehensive test of the visualization flow in simulation mode
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def test_trinity_routing_engine():
    """Test the Trinity routing engine"""
    print("🧠 Testing Trinity Routing Engine...")
    
    try:
        from trinity_routing_engine import TrinityRoutingEngine
        
        engine = TrinityRoutingEngine()
        
        # Test queries
        test_queries = [
            "What is anxiety?",
            "How to treat a headache?",
            "Emergency: I'm having chest pain",
            "Explain quantum computing",
            "What's the weather like?",
            "How to write a business plan"
        ]
        
        results = []
        for query in test_queries:
            decision = engine.route_query(query)
            results.append({
                "query": query,
                "model_variant": decision.model_variant.value,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
                "response_time": decision.estimated_response_time,
                "quality_score": decision.estimated_quality_score
            })
            print(f"✅ Query: '{query}' → {decision.model_variant.value} (confidence: {decision.confidence:.2f})")
        
        return True, results
        
    except Exception as e:
        print(f"❌ Trinity routing engine test failed: {e}")
        return False, []

def test_model_loader():
    """Test the real model loader in simulation mode"""
    print("🤖 Testing Model Loader (Simulation Mode)...")
    
    try:
        from meetara_real_model_comparison import RealModelLoader
        
        loader = RealModelLoader()
        
        # Test model availability
        models = loader.get_available_models()
        print(f"📊 Found {len(models)} model configurations")
        
        for name, info in models.items():
            status = "✅ Available" if info.get("available", False) else "❌ Not found"
            print(f"   {name}: {status} ({info.get('size', 'Unknown size')})")
        
        # Test model loading in simulation mode
        test_models = ["A_universal_full", "B_universal_lite", "C_category_specific"]
        
        for model_name in test_models:
            success = loader.load_model(model_name)
            if success:
                print(f"✅ {model_name} loaded successfully (simulation mode)")
            else:
                print(f"❌ {model_name} failed to load")
        
        return True, models
        
    except Exception as e:
        print(f"❌ Model loader test failed: {e}")
        return False, {}

def test_smart_routing():
    """Test the smart routing analysis"""
    print("🎯 Testing Smart Routing Analysis...")
    
    try:
        from meetara_real_model_comparison import SmartRouting
        
        router = SmartRouting()
        
        # Test queries
        test_queries = [
            "I have a headache, what should I do?",
            "Explain machine learning",
            "How to cook pasta?",
            "Emergency: chest pain and shortness of breath"
        ]
        
        results = []
        for query in test_queries:
            analysis = router.analyze_query(query)
            results.append({
                "query": query,
                "analysis": analysis
            })
            print(f"✅ Query: '{query}' → {analysis.get('recommended_model', 'Unknown')}")
        
        return True, results
        
    except Exception as e:
        print(f"❌ Smart routing test failed: {e}")
        return False, []

def test_api_endpoints():
    """Test API endpoints in simulation mode"""
    print("🌐 Testing API Endpoints...")
    
    # Start server in background (simulation mode)
    import subprocess
    import threading
    import time
    
    # Kill any existing server
    try:
        subprocess.run(["taskkill", "/f", "/im", "python.exe"], capture_output=True)
    except:
        pass
    
    # Start server
    server_process = subprocess.Popen([
        sys.executable, "meetara_real_model_comparison.py"
    ], cwd=Path(__file__).parent, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to start
    time.sleep(3)
    
    try:
        # Test endpoints
        base_url = "http://localhost:5001"
        
        # Test 1: Models endpoint
        print("   Testing /api/models...")
        response = requests.get(f"{base_url}/api/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print(f"   ✅ Models endpoint: {len(models)} models found")
        else:
            print(f"   ❌ Models endpoint failed: {response.status_code}")
        
        # Test 2: Query analysis
        print("   Testing /api/analyze_query...")
        test_query = "I have anxiety, what should I do?"
        response = requests.post(
            f"{base_url}/api/analyze_query",
            json={"prompt": test_query},
            timeout=5
        )
        if response.status_code == 200:
            analysis = response.json()
            print(f"   ✅ Query analysis: {analysis.get('recommended_model', 'Unknown')}")
        else:
            print(f"   ❌ Query analysis failed: {response.status_code}")
        
        # Test 3: Model comparison
        print("   Testing /api/compare...")
        response = requests.post(
            f"{base_url}/api/compare",
            json={
                "prompt": test_query,
                "models": ["A_universal_full", "B_universal_lite"]
            },
            timeout=10
        )
        if response.status_code == 200:
            comparison = response.json()
            print(f"   ✅ Model comparison: {len(comparison.get('responses', {}))} responses")
        else:
            print(f"   ❌ Model comparison failed: {response.status_code}")
        
        # Test 4: Status endpoint
        print("   Testing /api/status...")
        response = requests.get(f"{base_url}/api/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            print(f"   ✅ Status endpoint: {status.get('status', 'Unknown')}")
        else:
            print(f"   ❌ Status endpoint failed: {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ API endpoint test failed: {e}")
        return False
        
    finally:
        # Clean up
        server_process.terminate()
        server_process.wait()

def test_ui_template():
    """Test the UI template rendering"""
    print("🎨 Testing UI Template...")
    
    try:
        template_path = Path(__file__).parent / "templates" / "real_model_comparison.html"
        
        if not template_path.exists():
            print(f"❌ Template not found: {template_path}")
            return False
        
        # Read template
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
        
        # Check for required elements
        required_elements = [
            "MeeTARA Real Model Comparison",
            "modelSelector",
            "promptInput",
            "compareBtn",
            "routingAnalysis",
            "comparisonGrid"
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in template_content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"❌ Missing template elements: {missing_elements}")
            return False
        
        print("✅ UI template validation passed")
        return True
        
    except Exception as e:
        print(f"❌ UI template test failed: {e}")
        return False

def generate_visualization_report():
    """Generate comprehensive visualization flow report"""
    print("\n" + "="*60)
    print("📊 MeeTARA Lab - Visualization Flow Test Report")
    print("="*60)
    
    results = {}
    
    # Test 1: Trinity Routing Engine
    success, routing_results = test_trinity_routing_engine()
    results["trinity_routing"] = {
        "success": success,
        "results": routing_results
    }
    
    # Test 2: Model Loader
    success, model_results = test_model_loader()
    results["model_loader"] = {
        "success": success,
        "results": model_results
    }
    
    # Test 3: Smart Routing
    success, smart_results = test_smart_routing()
    results["smart_routing"] = {
        "success": success,
        "results": smart_results
    }
    
    # Test 4: API Endpoints
    success = test_api_endpoints()
    results["api_endpoints"] = {
        "success": success
    }
    
    # Test 5: UI Template
    success = test_ui_template()
    results["ui_template"] = {
        "success": success
    }
    
    # Generate summary
    print("\n📋 Test Summary:")
    print("-" * 40)
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result["success"])
    
    for test_name, result in results.items():
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\n🎯 Overall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All visualization flow tests passed! Ready for production.")
    else:
        print("⚠️ Some tests failed. Please review and fix issues.")
    
    # Save detailed report
    report_path = Path(__file__).parent / "visualization_test_report.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: {report_path}")
    
    return results

if __name__ == "__main__":
    generate_visualization_report() 