#!/usr/bin/env python3
"""
MeeTARA Lab - Centralized Domain Mapping Test
Tests domain integration across all environments (local, Colab, different devices)
"""

import sys
import os
from pathlib import Path

def test_environment_detection():
    """Test environment detection and path resolution"""
    print("🔍 Environment Detection Test")
    print("=" * 50)
    
    # Check current environment
    is_colab = False
    try:
        import google.colab
        is_colab = True
        print("✅ Environment: Google Colab")
    except ImportError:
        print("✅ Environment: Local/Other")
    
    print(f"   → Python version: {sys.version}")
    print(f"   → Current working directory: {Path.cwd()}")
    print(f"   → Script location: {Path(__file__).parent}")
    print(f"   → Home directory: {Path.home()}")
    
    return is_colab

def test_domain_integration_import():
    """Test importing centralized domain integration"""
    print("\n🔧 Domain Integration Import Test")
    print("=" * 50)
    
    # Try multiple import paths
    import_success = False
    
    # Method 1: Direct import
    try:
        sys.path.append('trinity-core')
        from domain_integration import get_domain_stats, get_all_domains, get_domain_categories
        print("✅ Import Method 1: Direct import successful")
        import_success = True
    except ImportError as e:
        print(f"⚠️ Import Method 1 failed: {e}")
    
    # Method 2: Relative import
    if not import_success:
        try:
            sys.path.append(str(Path(__file__).parent / "trinity-core"))
            from domain_integration import get_domain_stats, get_all_domains, get_domain_categories
            print("✅ Import Method 2: Relative import successful")
            import_success = True
        except ImportError as e:
            print(f"⚠️ Import Method 2 failed: {e}")
    
    # Method 3: Package import
    if not import_success:
        try:
            from trinity_core.domain_integration import get_domain_stats, get_all_domains, get_domain_categories
            print("✅ Import Method 3: Package import successful")
            import_success = True
        except ImportError as e:
            print(f"⚠️ Import Method 3 failed: {e}")
    
    if not import_success:
        print("❌ All import methods failed!")
        return False, None, None, None
    
    return True, get_domain_stats, get_all_domains, get_domain_categories

def test_config_file_loading():
    """Test config file loading across different paths"""
    print("\n📁 Config File Loading Test")
    print("=" * 50)
    
    # Test different config paths
    possible_paths = [
        "config/trinity_domain_model_mapping_config.yaml",
        Path(__file__).parent / "config" / "trinity_domain_model_mapping_config.yaml",
        Path.cwd() / "config" / "trinity_domain_model_mapping_config.yaml",
        "/content/meetara-lab/config/trinity_domain_model_mapping_config.yaml",  # Colab
        "/content/drive/MyDrive/meetara-lab/config/trinity_domain_model_mapping_config.yaml",  # Colab Drive
        Path.home() / "Documents" / "meetara-lab" / "config" / "trinity_domain_model_mapping_config.yaml",
    ]
    
    config_found = False
    for path in possible_paths:
        if Path(path).exists():
            print(f"✅ Config found: {path}")
            config_found = True
            break
        else:
            print(f"⚠️ Config not found: {path}")
    
    if not config_found:
        print("❌ Config file not found in any expected location!")
        print("\n💡 Expected config file locations:")
        for path in possible_paths:
            print(f"   - {path}")
    
    return config_found

def test_domain_functionality(get_domain_stats, get_all_domains, get_domain_categories):
    """Test domain functionality"""
    print("\n🎯 Domain Functionality Test")
    print("=" * 50)
    
    try:
        # Test domain stats
        stats = get_domain_stats()
        print(f"✅ Domain Stats Retrieved:")
        print(f"   → Total domains: {stats['total_domains']}")
        print(f"   → Total categories: {stats['total_categories']}")
        print(f"   → Config loaded: {stats['config_loaded']}")
        print(f"   → Config path: {stats.get('config_path', 'Unknown')}")
        
        # Test all domains
        all_domains = get_all_domains()
        print(f"✅ All Domains Retrieved: {len(all_domains)} domains")
        
        # Test domain categories
        categories = get_domain_categories()
        print(f"✅ Domain Categories Retrieved:")
        for category, domains in categories.items():
            print(f"   → {category}: {len(domains)} domains")
        
        # Verify total count matches
        total_from_categories = sum(len(domains) for domains in categories.values())
        if total_from_categories == len(all_domains) == stats['total_domains']:
            print("✅ Domain count consistency verified")
        else:
            print(f"⚠️ Domain count mismatch: stats={stats['total_domains']}, all_domains={len(all_domains)}, categories={total_from_categories}")
        
        return True
        
    except Exception as e:
        print(f"❌ Domain functionality test failed: {e}")
        return False

def test_cloud_training_integration():
    """Test cloud training script integration"""
    print("\n☁️ Cloud Training Integration Test")
    print("=" * 50)
    
    try:
        # Test training orchestrator import
        sys.path.append('cloud-training')
        from training_orchestrator import TrainingOrchestrator
        print("✅ Training Orchestrator import successful")
        
        # Test creating instance (without starting)
        orchestrator = TrainingOrchestrator()
        print("✅ Training Orchestrator instance created")
        
        # Check if it has centralized domain mapping
        if hasattr(orchestrator, 'domain_mapping') and orchestrator.domain_mapping.get('centralized'):
            print("✅ Training Orchestrator using centralized domain mapping")
        else:
            print("⚠️ Training Orchestrator may not be using centralized mapping")
        
        return True
        
    except Exception as e:
        print(f"❌ Cloud training integration test failed: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive test suite"""
    print("🚀 MeeTARA Lab - Centralized Domain Mapping Test Suite")
    print("=" * 70)
    
    # Test results
    results = {
        "environment_detection": False,
        "domain_integration_import": False,
        "config_file_loading": False,
        "domain_functionality": False,
        "cloud_training_integration": False
    }
    
    # Run tests
    is_colab = test_environment_detection()
    results["environment_detection"] = True
    
    import_success, get_domain_stats, get_all_domains, get_domain_categories = test_domain_integration_import()
    results["domain_integration_import"] = import_success
    
    config_found = test_config_file_loading()
    results["config_file_loading"] = config_found
    
    if import_success:
        domain_func_success = test_domain_functionality(get_domain_stats, get_all_domains, get_domain_categories)
        results["domain_functionality"] = domain_func_success
    
    cloud_success = test_cloud_training_integration()
    results["cloud_training_integration"] = cloud_success
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Centralized domain mapping is working correctly.")
    elif passed >= total - 1:
        print("⚠️ Most tests passed. Minor issues may exist but system should work.")
    else:
        print("❌ Multiple test failures. System may not work correctly.")
    
    # Environment-specific recommendations
    print("\n💡 Environment-Specific Notes:")
    if is_colab:
        print("   → Running on Google Colab")
        print("   → Ensure repository is cloned to /content/meetara-lab/")
        print("   → Or mount Google Drive and use /content/drive/MyDrive/meetara-lab/")
    else:
        print("   → Running on local/other environment")
        print("   → Ensure you're in the project root directory")
        print("   → Config file should be at config/trinity_domain_model_mapping_config.yaml")
    
    return results

if __name__ == "__main__":
    run_comprehensive_test() 
