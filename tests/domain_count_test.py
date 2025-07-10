#!/usr/bin/env python3
"""
Simple test to count and list all available domains in the data generator.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trinity_core.agents.data_generator import TrinityDataGenerator
from trinity_core.core_components.config_manager import SmartTrinityConfigManager

def main():
    # Create mock hub
    class MockHub:
        def __init__(self, config):
            self.config_manager = config
            self.mcp = None
            self.intelligence = None
    
    config = SmartTrinityConfigManager()
    hub = MockHub(config)
    
    # Initialize data generator
    dg = TrinityDataGenerator(hub=hub)
    
    # Count and list domains
    total_domains = len(dg.domain_templates)
    print(f"✅ Total domains available: {total_domains}")
    print("\n📋 Available domains:")
    
    # Group domains by category
    healthcare_domains = []
    daily_life_domains = []
    business_domains = []
    education_domains = []
    technology_domains = []
    space_domains = []
    creative_domains = []
    specialized_domains = []
    
    for domain in sorted(dg.domain_templates.keys()):
        if domain in ['healthcare', 'mental_health', 'nutrition', 'fitness', 'sleep', 'stress_management', 
                     'preventive_care', 'chronic_conditions', 'medication_management', 'emergency_care', 
                     'women_health', 'senior_health']:
            healthcare_domains.append(domain)
        elif domain in ['parenting', 'relationships', 'personal_assistant', 'communication', 'home_management',
                       'shopping', 'planning', 'transportation', 'time_management', 'decision_making',
                       'conflict_resolution', 'work_life_balance']:
            daily_life_domains.append(domain)
        elif domain in ['entrepreneurship', 'marketing', 'sales', 'customer_service', 'project_management',
                       'team_leadership', 'financial_planning', 'operations', 'hr_management', 'strategy',
                       'consulting', 'legal_business']:
            business_domains.append(domain)
        elif domain in ['teaching', 'language_learning', 'research', 'study_skills', 'career_planning',
                       'skill_development', 'test_preparation']:
            education_domains.append(domain)
        elif domain in ['programming', 'web_development', 'mobile_apps', 'ai_ml', 'cybersecurity']:
            technology_domains.append(domain)
        elif domain in ['space_research', 'aerospace_engineering', 'satellite_systems', 'space_mission_planning']:
            space_domains.append(domain)
        elif domain in ['creative_writing', 'visual_arts', 'music', 'content_creation', 'design',
                       'innovation', 'photography', 'film_making']:
            creative_domains.append(domain)
        elif domain in ['legal_assistance', 'financial_planning']:
            specialized_domains.append(domain)
        else:
            print(f"  - {domain} (uncategorized)")
    
    print(f"\n🏥 Healthcare Domains ({len(healthcare_domains)}):")
    for domain in healthcare_domains:
        print(f"  - {domain}")
    
    print(f"\n🏠 Daily Life Domains ({len(daily_life_domains)}):")
    for domain in daily_life_domains:
        print(f"  - {domain}")
    
    print(f"\n💼 Business Domains ({len(business_domains)}):")
    for domain in business_domains:
        print(f"  - {domain}")
    
    print(f"\n🎓 Education Domains ({len(education_domains)}):")
    for domain in education_domains:
        print(f"  - {domain}")
    
    print(f"\n💻 Technology Domains ({len(technology_domains)}):")
    for domain in technology_domains:
        print(f"  - {domain}")
    
    print(f"\n🚀 Space Technology Domains ({len(space_domains)}):")
    for domain in space_domains:
        print(f"  - {domain}")
    
    print(f"\n🎨 Creative Domains ({len(creative_domains)}):")
    for domain in creative_domains:
        print(f"  - {domain}")
    
    print(f"\n⚖️ Specialized Domains ({len(specialized_domains)}):")
    for domain in specialized_domains:
        print(f"  - {domain}")
    
    # Check for specific domains you mentioned
    print(f"\n🔍 Specific Domain Check:")
    print(f"  - Programming: {'✅' if 'programming' in dg.domain_templates else '❌'}")
    print(f"  - Space Research: {'✅' if 'space_research' in dg.domain_templates else '❌'}")
    print(f"  - Aerospace Engineering: {'✅' if 'aerospace_engineering' in dg.domain_templates else '❌'}")
    print(f"  - Satellite Systems: {'✅' if 'satellite_systems' in dg.domain_templates else '❌'}")
    print(f"  - Space Mission Planning: {'✅' if 'space_mission_planning' in dg.domain_templates else '❌'}")
    
    return total_domains

if __name__ == "__main__":
    main() 