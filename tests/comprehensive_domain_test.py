#!/usr/bin/env python3
"""
Comprehensive Domain Test for MeeTARA Lab
Tests all 65 domains from original TARA Universal Model with rich multi-scenario format
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
from typing import Dict, List, Any

# Trinity Architecture imports
from trinity_core.agents.comprehensive_data_generator import ComprehensiveDataGenerator
from trinity_core.agents.comprehensive_domain_templates import COMPREHENSIVE_DOMAIN_TEMPLATES

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveDomainTester:
    """Test comprehensive domain templates and data generation."""
    
    def __init__(self):
        self.test_results = {}
        self.passed_tests = 0
        self.total_tests = 0
    
    def test_comprehensive_templates(self) -> bool:
        """Test comprehensive domain templates."""
        logger.info("🧪 Testing comprehensive domain templates...")
        
        # Test 1: Check if all templates are loaded
        self.total_tests += 1
        if len(COMPREHENSIVE_DOMAIN_TEMPLATES) >= 60:  # Should have 65+ domains
            logger.info(f"✅ Template count: {len(COMPREHENSIVE_DOMAIN_TEMPLATES)} domains")
            self.passed_tests += 1
        else:
            logger.error(f"❌ Insufficient templates: {len(COMPREHENSIVE_DOMAIN_TEMPLATES)} domains")
            return False
        
        # Test 2: Check template structure
        self.total_tests += 1
        template_structure_valid = True
        required_fields = ["scenarios", "user_intents", "conversation_starters", "response_patterns"]
        
        for domain, template in COMPREHENSIVE_DOMAIN_TEMPLATES.items():
            for field in required_fields:
                if field not in template:
                    logger.error(f"❌ Missing field '{field}' in {domain}")
                    template_structure_valid = False
                    break
                if not isinstance(template[field], list) or len(template[field]) < 5:
                    logger.error(f"❌ Insufficient {field} in {domain}: {len(template[field])}")
                    template_structure_valid = False
                    break
        
        if template_structure_valid:
            logger.info("✅ All templates have proper structure")
            self.passed_tests += 1
        else:
            return False
        
        # Test 3: Check domain categories
        self.total_tests += 1
        categories = self._categorize_domains()
        total_domains = sum(len(domains) for domains in categories.values())
        
        if total_domains >= 60:
            logger.info(f"✅ Domain categorization: {total_domains} domains across categories")
            for category, domains in categories.items():
                if domains:
                    logger.info(f"  {category}: {len(domains)} domains")
            self.passed_tests += 1
        else:
            logger.error(f"❌ Insufficient categorized domains: {total_domains}")
            return False
        
        return True
    
    def test_data_generator(self) -> bool:
        """Test comprehensive data generator."""
        logger.info("🧪 Testing comprehensive data generator...")
        
        # Create mock hub
        class MockHub:
            def __init__(self):
                self.config_manager = MockConfigManager()
                self.mcp = None
                self.intelligence = None
        
        class MockConfigManager:
            def get_config_dict(self):
                return {"data_generation": {"samples_per_domain": 1000}}
        
        hub = MockHub()
        
        try:
            # Test 1: Initialize generator
            self.total_tests += 1
            generator = ComprehensiveDataGenerator(hub)
            logger.info(f"✅ Generator initialized with {len(generator.domain_templates)} domains")
            self.passed_tests += 1
        except Exception as e:
            logger.error(f"❌ Failed to initialize generator: {e}")
            return False
        
        # Test 2: Test domain expert creation
        self.total_tests += 1
        test_domains = ["healthcare", "mental_health", "entrepreneurship", "programming"]
        domain_expert_valid = True
        
        for domain in test_domains:
            if domain in generator.domain_templates:
                expert = generator._create_domain_expert_agent(domain)
                if not expert or "domain" not in expert:
                    logger.error(f"❌ Invalid domain expert for {domain}")
                    domain_expert_valid = False
                    break
            else:
                logger.error(f"❌ Domain not found: {domain}")
                domain_expert_valid = False
                break
        
        if domain_expert_valid:
            logger.info("✅ Domain expert creation working")
            self.passed_tests += 1
        else:
            return False
        
        # Test 3: Test urgency analysis
        self.total_tests += 1
        test_starters = [
            "I'm having a medical emergency!",
            "Can you help me with a general question?",
            "I'm in crisis and need immediate help!",
            "What's the weather like today?"
        ]
        
        urgency_score = generator._analyze_urgency_patterns(test_starters)
        if 0 <= urgency_score <= 1:
            logger.info(f"✅ Urgency analysis working: {urgency_score:.3f}")
            self.passed_tests += 1
        else:
            logger.error(f"❌ Invalid urgency score: {urgency_score}")
            return False
        
        # Test 4: Test domain criticality
        self.total_tests += 1
        test_domains = ["healthcare", "business", "creative", "shopping"]
        criticality_valid = True
        
        for domain in test_domains:
            criticality = generator._detect_domain_criticality(domain)
            if not (0 <= criticality <= 1):
                logger.error(f"❌ Invalid criticality for {domain}: {criticality}")
                criticality_valid = False
                break
        
        if criticality_valid:
            logger.info("✅ Domain criticality detection working")
            self.passed_tests += 1
        else:
            return False
        
        # Test 5: Test conversation generation
        self.total_tests += 1
        try:
            conversation = generator._generate_blended_conversation("healthcare", generator.domain_templates["healthcare"])
            if conversation and "conversation" in conversation:
                logger.info("✅ Conversation generation working")
                self.passed_tests += 1
            else:
                logger.error("❌ Invalid conversation generated")
                return False
        except Exception as e:
            logger.error(f"❌ Conversation generation failed: {e}")
            return False
        
        return True
    
    def test_domain_coverage(self) -> bool:
        """Test domain coverage from original TARA Universal Model."""
        logger.info("🧪 Testing domain coverage...")
        
        # Expected domains from original TARA Universal Model
        expected_domains = {
            "Healthcare": [
                "general_health", "mental_health", "nutrition", "fitness", "sleep", 
                "stress_management", "preventive_care", "chronic_conditions", 
                "medication_management", "emergency_care", "women_health", "senior_health"
            ],
            "Daily Life": [
                "parenting", "relationships", "personal_assistant", "communication",
                "home_management", "shopping", "planning", "transportation",
                "time_management", "decision_making", "conflict_resolution", "work_life_balance"
            ],
            "Business": [
                "entrepreneurship", "marketing", "sales", "customer_service",
                "project_management", "team_leadership", "financial_planning", "operations",
                "hr_management", "strategy", "consulting", "legal_business"
            ],
            "Education": [
                "academic_tutoring", "skill_development", "career_guidance", "exam_preparation",
                "language_learning", "research_assistance", "study_techniques", "educational_technology"
            ],
            "Technology": [
                "programming", "ai_ml", "cybersecurity", "data_analysis", "tech_support", "software_development"
            ],
            "Space Technology": [
                "space_research", "aerospace_engineering", "satellite_systems", "space_mission_planning"
            ],
            "Creative": [
                "creative_writing", "visual_arts", "music", "content_creation", "design",
                "innovation", "photography", "film_making"
            ],
            "Specialized": [
                "legal_assistance", "financial_planning", "scientific_research", "engineering"
            ]
        }
        
        # Test domain coverage
        self.total_tests += 1
        coverage_valid = True
        missing_domains = []
        
        for category, expected_domains_list in expected_domains.items():
            for domain in expected_domains_list:
                if domain not in COMPREHENSIVE_DOMAIN_TEMPLATES:
                    missing_domains.append(domain)
                    coverage_valid = False
        
        if coverage_valid:
            logger.info("✅ All expected domains covered")
            self.passed_tests += 1
        else:
            logger.error(f"❌ Missing domains: {missing_domains}")
            return False
        
        # Test rich format
        self.total_tests += 1
        rich_format_valid = True
        
        for domain, template in COMPREHENSIVE_DOMAIN_TEMPLATES.items():
            # Check for rich multi-scenario format
            if len(template.get("scenarios", [])) < 10:
                logger.error(f"❌ Insufficient scenarios in {domain}: {len(template.get('scenarios', []))}")
                rich_format_valid = False
            if len(template.get("conversation_starters", [])) < 10:
                logger.error(f"❌ Insufficient conversation starters in {domain}: {len(template.get('conversation_starters', []))}")
                rich_format_valid = False
            if len(template.get("response_patterns", [])) < 10:
                logger.error(f"❌ Insufficient response patterns in {domain}: {len(template.get('response_patterns', []))}")
                rich_format_valid = False
        
        if rich_format_valid:
            logger.info("✅ All domains have rich multi-scenario format")
            self.passed_tests += 1
        else:
            return False
        
        return True
    
    def _categorize_domains(self) -> Dict[str, List[str]]:
        """Categorize domains for testing."""
        categories = {
            "Healthcare": [],
            "Daily Life": [],
            "Business": [],
            "Education": [],
            "Technology": [],
            "Space Technology": [],
            "Creative": [],
            "Specialized": []
        }
        
        for domain in COMPREHENSIVE_DOMAIN_TEMPLATES.keys():
            if any(h in domain for h in ["health", "mental", "nutrition", "fitness", "sleep", "stress", "preventive", "chronic", "medication", "emergency", "women", "senior"]):
                categories["Healthcare"].append(domain)
            elif any(dl in domain for dl in ["parenting", "relationships", "personal", "communication", "home", "shopping", "planning", "transportation", "time", "decision", "conflict", "work_life"]):
                categories["Daily Life"].append(domain)
            elif any(b in domain for b in ["entrepreneurship", "marketing", "sales", "customer", "project", "team", "financial", "operations", "hr", "strategy", "consulting", "legal_business"]):
                categories["Business"].append(domain)
            elif any(e in domain for e in ["academic", "skill", "career", "exam", "language", "research", "study", "educational"]):
                categories["Education"].append(domain)
            elif any(t in domain for t in ["programming", "ai_ml", "cybersecurity", "data", "tech", "software"]):
                categories["Technology"].append(domain)
            elif any(s in domain for s in ["space", "aerospace", "satellite"]):
                categories["Space Technology"].append(domain)
            elif any(c in domain for c in ["creative", "visual", "music", "content", "design", "innovation", "photography", "film"]):
                categories["Creative"].append(domain)
            elif any(sp in domain for sp in ["legal", "financial", "scientific", "engineering"]):
                categories["Specialized"].append(domain)
        
        return categories
    
    def run_all_tests(self) -> bool:
        """Run all comprehensive tests."""
        logger.info("🚀 Starting comprehensive domain tests...")
        
        tests = [
            ("Comprehensive Templates", self.test_comprehensive_templates),
            ("Data Generator", self.test_data_generator),
            ("Domain Coverage", self.test_domain_coverage)
        ]
        
        all_passed = True
        
        for test_name, test_func in tests:
            logger.info(f"\n📋 Running {test_name} test...")
            try:
                if test_func():
                    logger.info(f"✅ {test_name} test passed")
                else:
                    logger.error(f"❌ {test_name} test failed")
                    all_passed = False
            except Exception as e:
                logger.error(f"❌ {test_name} test error: {e}")
                all_passed = False
        
        # Print summary
        logger.info(f"\n📊 Test Summary:")
        logger.info(f"  Passed: {self.passed_tests}/{self.total_tests}")
        logger.info(f"  Success Rate: {(self.passed_tests/self.total_tests)*100:.1f}%")
        
        if all_passed:
            logger.info("🎉 All comprehensive domain tests passed!")
        else:
            logger.error("❌ Some comprehensive domain tests failed!")
        
        return all_passed

def main():
    """Run comprehensive domain tests."""
    tester = ComprehensiveDomainTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ Comprehensive domain validation successful!")
        print("🎯 All 65+ domains from original TARA Universal Model are properly implemented")
        print("📊 Rich multi-scenario format confirmed for all domains")
        print("🚀 MeeTARA Lab now matches original TARA Universal Model scope and quality")
    else:
        print("\n❌ Comprehensive domain validation failed!")
        print("🔧 Please check the implementation and fix any issues")

if __name__ == "__main__":
    main() 