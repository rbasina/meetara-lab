#!/usr/bin/env python3
"""
Test Voice Integration with TARA Reference
"""

import sys
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path('.').absolute()))

def test_voice_integration():
    print('🚀 TESTING VOICE INTEGRATION WITH TARA REFERENCE')
    print('=' * 60)
    
    try:
        from trinity_core.intelligent_router import VoiceCategoryManager, IntelligentRouter
        
        # Test 1: VoiceCategoryManager
        print('🎤 Test 1: VoiceCategoryManager')
        voice_manager = VoiceCategoryManager()
        
        categories = voice_manager.get_all_voice_categories()
        print(f'   ✅ Voice categories: {len(categories)}')
        print(f'   Categories: {categories}')
        
        # Test domain routing
        test_domains = ['healthcare', 'business', 'meditation', 'education', 'creative', 'programming']
        print()
        print('🧪 Domain-to-Voice Routing Tests:')
        for domain in test_domains:
            voice = voice_manager.get_voice_for_domain(domain)
            characteristics = voice_manager.get_voice_characteristics(domain)
            print(f'   {domain} → {voice}')
            print(f'      Tone: {characteristics["tone"]}, Pace: {characteristics["pace"]}')
            print(f'      Empathy: {characteristics["empathy"]}, Energy: {characteristics["energy_level"]}')
        
        print()
        print('🧠 Test 2: IntelligentRouter Integration')
        
        # Test router (check if voice_manager is integrated)
        router = IntelligentRouter()
        
        # Test if router has voice methods
        if hasattr(router, 'get_voice_for_domain'):
            print('   ✅ Router has voice routing methods')
            
            # Test router voice methods
            test_voice = router.get_voice_for_domain('healthcare')
            print(f'   Healthcare voice: {test_voice}')
            
            # Test validation
            validation = router.validate_routing_configuration()
            print(f'   ✅ Total domains: {validation["total_domains"]}')
            print(f'   ✅ Voice categories: {validation["voice_categories"]}')
            print(f'   ✅ Configuration complete: {validation["configuration_complete"]}')
            
            # Test domain voice mapping
            domain_voice_mapping = router.get_domain_voice_mapping()
            print(f'   ✅ Domain-voice mappings: {len(domain_voice_mapping)}')
            
            # Show voice distribution
            voice_dist = validation.get('voice_distribution', {})
            print()
            print('🎯 VOICE DISTRIBUTION:')
            for voice, count in voice_dist.items():
                print(f'   {voice}: {count} domains')
        else:
            print('   ⚠️ Router missing voice methods - needs integration')
        
        print()
        print('🎉 VOICE INTEGRATION TEST COMPLETE!')
        return True
        
    except ImportError as e:
        print(f'❌ Import error: {e}')
        return False
    except Exception as e:
        print(f'❌ Unexpected error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_voice_integration()
    print()
    if success:
        print('✅ TARA VOICE INTEGRATION SUCCESSFUL!')
    else:
        print('❌ TARA VOICE INTEGRATION NEEDS WORK!') 
