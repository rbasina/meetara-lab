#!/usr/bin/env python3
"""
Test script to verify domain coverage between config and data generator.
"""

from trinity_core.agents.data_generator import TrinityDataGenerator

class MockConfigManager:
    """Mock config manager for testing."""
    def __init__(self):
        self.config = {
            "global_tara_params": {
                "sequence_length": 64,
                "validation_target": 101.0,
                "output_format": "Q4_K_M",
                "target_gguf_size_mb": 8.3
            },
            "domain_config": {}
        }
    
    def get_config_dict(self):
        return self.config

class MockHub:
    """Mock hub for testing TrinityDataGenerator."""
    def __init__(self):
        self.config_manager = MockConfigManager()
        self.emotion_detector = None
        self.tts_manager = None
        self.intelligent_router = None
        self.mcp = None
        self.intelligence = None  # Added to satisfy TrinityDataGenerator

def main():
    print("🔍 Checking domain coverage...")
    
    # Create mock hub and load domains from generator
    mock_hub = MockHub()
    dg = TrinityDataGenerator(mock_hub)
    generator_domains = list(dg.domain_templates.keys())
    
    print(f"\n📊 DOMAIN ANALYSIS:")
    print(f"Total domains in generator: {len(generator_domains)}")
    
    # Categorize domains based on config structure
    healthcare_domains = [
        "general_health", "mental_health", "nutrition", "fitness", "sleep",
        "stress_management", "preventive_care", "chronic_conditions", 
        "medication_management", "emergency_care", "women_health", "senior_health"
    ]
    
    daily_life_domains = [
        "parenting", "relationships", "personal_assistant", "communication",
        "home_management", "shopping", "planning", "transportation",
        "time_management", "decision_making", "conflict_resolution", "work_life_balance"
    ]
    
    business_domains = [
        "entrepreneurship", "marketing", "sales", "customer_service",
        "project_management", "team_leadership", "financial_planning", "operations",
        "hr_management", "strategy", "consulting", "legal_business"
    ]
    
    education_domains = [
        "academic_tutoring", "skill_development", "career_guidance", "exam_preparation",
        "language_learning", "research_assistance", "study_techniques", "educational_technology"
    ]
    
    creative_domains = [
        "writing", "storytelling", "content_creation", "social_media",
        "design_thinking", "photography", "music", "art_appreciation",
        "mythology", "spiritual"
    ]
    
    psychology_wellness_domains = [
        "psychology", "yoga", "life_coaching", "social_support"
    ]
    
    sports_recreation_domains = [
        "sports_recreation", "fitness"
    ]
    
    business_professional_domains = [
        "remote_work", "social_media_management", "digital_literacy", "language_learning"
    ]
    
    research_academic_domains = [
        "research", "academic_tutoring"
    ]
    
    legal_financial_domains = [
        "legal_assistance", "insurance", "real_estate"
    ]
    
    emergency_crisis_domains = [
        "crisis_management", "disaster_preparedness", "emergency_response", "safety_security"
    ]
    
    aerospace_transportation_domains = [
        "aeronautics", "automobile", "space_technology"
    ]
    
    industrial_manufacturing_domains = [
        "agriculture", "manufacturing"
    ]
    
    travel_tourism_domains = [
        "travel_tourism"
    ]
    
    technology_domains = [
        "programming", "ai_ml", "cybersecurity", "data_analysis", "tech_support", "software_development"
    ]
    
    specialized_domains = [
        "legal", "financial", "scientific_research", "engineering"
    ]
    
    # All expected domains
    all_expected_domains = (
        healthcare_domains + daily_life_domains + business_domains + 
        education_domains + creative_domains + psychology_wellness_domains +
        sports_recreation_domains + business_professional_domains + 
        research_academic_domains + legal_financial_domains + 
        emergency_crisis_domains + aerospace_transportation_domains +
        industrial_manufacturing_domains + travel_tourism_domains +
        technology_domains + specialized_domains
    )
    
    # Check coverage
    expected_set = set(all_expected_domains)
    generator_set = set(generator_domains)
    
    missing_domains = expected_set - generator_set
    extra_domains = generator_set - expected_set
    
    print(f"\n✅ COVERAGE ANALYSIS:")
    print(f"Expected domains: {len(expected_set)}")
    print(f"Generator domains: {len(generator_set)}")
    print(f"Coverage: {len(expected_set & generator_set)}/{len(expected_set)}")
    
    if missing_domains:
        print(f"\n❌ MISSING DOMAINS ({len(missing_domains)}):")
        for domain in sorted(missing_domains):
            print(f"  - {domain}")
    
    if extra_domains:
        print(f"\n➕ EXTRA DOMAINS ({len(extra_domains)}):")
        for domain in sorted(extra_domains):
            print(f"  - {domain}")
    
    # Category breakdown
    print(f"\n📋 CATEGORY BREAKDOWN:")
    categories = {
        "Healthcare": healthcare_domains,
        "Daily Life": daily_life_domains,
        "Business": business_domains,
        "Education": education_domains,
        "Creative": creative_domains,
        "Psychology & Wellness": psychology_wellness_domains,
        "Sports & Recreation": sports_recreation_domains,
        "Business & Professional": business_professional_domains,
        "Research & Academic": research_academic_domains,
        "Legal & Financial": legal_financial_domains,
        "Emergency & Crisis": emergency_crisis_domains,
        "Aerospace & Transportation": aerospace_transportation_domains,
        "Industrial & Manufacturing": industrial_manufacturing_domains,
        "Travel & Tourism": travel_tourism_domains,
        "Technology": technology_domains,
        "Specialized": specialized_domains
    }
    
    for category, domains in categories.items():
        covered = len([d for d in domains if d in generator_set])
        total = len(domains)
        print(f"  {category}: {covered}/{total} ({covered/total*100:.1f}%)")
    
    coverage_percentage = len(expected_set & generator_set) / len(expected_set) * 100
    print(f"\n📈 OVERALL COVERAGE: {coverage_percentage:.1f}%")
    
    if missing_domains:
        print(f"\n⚠️  WARNING: {len(missing_domains)} domains missing!")
    else:
        print(f"\n🎉 SUCCESS: All expected domains are covered!")
    
    # Test domain template structure
    print(f"\n🔧 TESTING DOMAIN TEMPLATE STRUCTURE:")
    test_domain = list(generator_domains)[0] if generator_domains else None
    if test_domain:
        template = dg.domain_templates[test_domain]
        required_keys = ["scenarios", "user_intents", "conversation_starters", "response_patterns", "trinity_phase"]
        missing_keys = [key for key in required_keys if key not in template]
        
        if missing_keys:
            print(f"❌ Template structure issue in '{test_domain}': missing {missing_keys}")
        else:
            print(f"✅ Template structure is correct for '{test_domain}'")
            print(f"   - Scenarios: {len(template['scenarios'])}")
            print(f"   - User intents: {len(template['user_intents'])}")
            print(f"   - Conversation starters: {len(template['conversation_starters'])}")
            print(f"   - Response patterns: {len(template['response_patterns'])}")
            print(f"   - Trinity phase: {template['trinity_phase']}")
    
    # Test a few sample domains
    print(f"\n🧪 TESTING SAMPLE DOMAINS:")
    sample_domains = ["general_health", "entrepreneurship", "writing", "psychology"]
    for domain in sample_domains:
        if domain in dg.domain_templates:
            template = dg.domain_templates[domain]
            print(f"✅ {domain}: {len(template['conversation_starters'])} starters, {template['trinity_phase']} phase")
        else:
            print(f"❌ {domain}: Missing from templates")

if __name__ == "__main__":
    main() 