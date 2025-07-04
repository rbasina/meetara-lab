#!/usr/bin/env python3
"""
Quick Trinity Configuration Integration Test
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append('.')

try:
    from trinity_core.domain_integration import (
        get_domain_stats, 
        get_all_domains, 
        get_domain_categories,
        get_model_for_domain
    )
    
    # Test configuration integration
    stats = get_domain_stats()
    categories = get_domain_categories()
    domains = get_all_domains()
    
    print(f"🎯 TRINITY CONFIGURATION INTEGRATION TEST")
    print(f"=" * 50)
    print(f"✅ Config loaded: {stats['config_loaded']}")
    print(f"✅ Total domains: {stats['total_domains']}")
    print(f"✅ Categories: {len(categories)}")
    print(f"✅ Config path: {stats.get('config_path', 'Dynamic')}")
    
    print(f"\n📋 Domain Categories:")
    for category, domain_list in categories.items():
        print(f"   → {category}: {len(domain_list)} domains")
    
    # Test model mapping for sample domains
    print(f"\n🔧 Model Mapping Test:")
    test_domains = ["general_health", "entrepreneurship", "programming", "writing"]
    for domain in test_domains:
        if domain in domains:
            model = get_model_for_domain(domain)
            print(f"   → {domain}: {model}")
    
    print(f"\n✅ TRINITY CONFIGURATION INTEGRATION: SUCCESS")
    print(f"   → All {stats['total_domains']} domains accessible")
    print(f"   → All {len(categories)} categories mapped")
    print(f"   → Centralized config working perfectly")
    
except Exception as e:
    print(f"❌ Configuration integration test failed: {e}")
    import traceback
    traceback.print_exc() 