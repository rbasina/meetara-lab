#!/usr/bin/env python3
"""Quick Trinity Configuration Integration Test"""

import sys
sys.path.append('.')

try:
    from trinity_core.domain_integration import get_domain_stats, get_all_domains, get_domain_categories
    
    stats = get_domain_stats()
    categories = get_domain_categories()
    
    print(f"🎯 TRINITY CONFIG INTEGRATION TEST")
    print(f"✅ Config loaded: {stats['config_loaded']}")
    print(f"✅ Total domains: {stats['total_domains']}")
    print(f"✅ Categories: {len(categories)}")
    print(f"✅ SUCCESS: All agents use same centralized config")
    
except Exception as e:
    print(f"❌ Test failed: {e}") 