#!/usr/bin/env python3
"""
MeeTARA Lab - Comprehensive Domain Validation
Tests all 61 domains for data generation, accuracy, and Trinity Architecture integration.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trinity_core.agents.data_generator import TrinityDataGenerator
from trinity_core.core_components.config_manager import SmartTrinityConfigManager

class ComprehensiveDomainValidator:
    """
    Comprehensive validator for all 61 domains.
    Tests data generation, accuracy, and Trinity Architecture integration.
    """
    
    def __init__(self):
        self.config_manager = SmartTrinityConfigManager()
        
        # Create a mock hub object for the data generator
        class MockHub:
            def __init__(self, config_manager):
                self.config_manager = config_manager
                self.mcp = None
                self.intelligence = None
        
        mock_hub = MockHub(self.config_manager)
        self.data_generator = TrinityDataGenerator(hub=mock_hub)
        
        self.test_results = {
            "total_domains": 0,
            "passed_domains": 0,
            "failed_domains": 0,
            "domain_details": {},
            "overall_accuracy": 0.0,
            "trinity_phases": {},
            "crisis_intervention": 0,
            "emotional_intelligence": 0,
            "professional_boundaries": 0
        }
    
    def validate_domain(self, domain: str) -> Dict[str, Any]:
        """Validate a single domain for data generation capabilities."""
        result = {
            "domain": domain,
            "template_available": False,
            "domain_expert_created": False,
            "trinity_phase_correct": False,
            "crisis_intervention": False,
            "emotional_intelligence": False,
            "professional_boundaries": False,
            "sample_generation": False,
            "errors": []
        }
        
        try:
            # Check if domain template exists
            if domain in self.data_generator.domain_templates:
                result["template_available"] = True
                
                # Check domain expert creation
                domain_expert = self.data_generator._create_domain_expert_agent(domain)
                if domain_expert and "domain" in domain_expert:
                    result["domain_expert_created"] = True
                
                # Check Trinity phase
                if domain_expert and "trinity_phase" in domain_expert:
                    result["trinity_phase_correct"] = True
                    self.test_results["trinity_phases"][domain_expert["trinity_phase"]] = \
                        self.test_results["trinity_phases"].get(domain_expert["trinity_phase"], 0) + 1
                
                # Check crisis intervention
                if domain_expert and domain_expert.get("crisis_intervention", False):
                    result["crisis_intervention"] = True
                    self.test_results["crisis_intervention"] += 1
                
                # Check emotional intelligence
                if domain_expert and domain_expert.get("emotional_intelligence", False):
                    result["emotional_intelligence"] = True
                    self.test_results["emotional_intelligence"] += 1
                
                # Check professional boundaries
                if domain_expert and domain_expert.get("capabilities"):
                    result["professional_boundaries"] = True
                    self.test_results["professional_boundaries"] += 1
                
                # Test sample generation
                try:
                    # Generate a small sample to test functionality
                    sample_data = self.data_generator.generate_domain_data(
                        domain=domain,
                        num_samples=1,
                        quality_threshold=0.8
                    )
                    if sample_data:
                        result["sample_generation"] = True
                except Exception as e:
                    result["errors"].append(f"Sample generation failed: {str(e)}")
            
        except Exception as e:
            result["errors"].append(f"Domain validation failed: {str(e)}")
        
        return result
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation on all domains."""
        print("🚀 Starting Comprehensive Domain Validation...")
        print("=" * 60)
        
        # Get all available domains
        all_domains = list(self.data_generator.domain_templates.keys())
        self.test_results["total_domains"] = len(all_domains)
        
        print(f"📊 Testing {len(all_domains)} domains...")
        print()
        
        # Test each domain
        for i, domain in enumerate(all_domains, 1):
            print(f"🔍 Testing domain {i}/{len(all_domains)}: {domain}")
            
            result = self.validate_domain(domain)
            self.test_results["domain_details"][domain] = result
            
            # Count passed/failed
            passed_checks = sum([
                result["template_available"],
                result["domain_expert_created"],
                result["trinity_phase_correct"],
                result["professional_boundaries"]
            ])
            
            if passed_checks >= 3:  # At least 3 out of 4 core checks must pass
                self.test_results["passed_domains"] += 1
                status = "✅ PASSED"
            else:
                self.test_results["failed_domains"] += 1
                status = "❌ FAILED"
            
            print(f"   {status} - Template: {result['template_available']}, Expert: {result['domain_expert_created']}, Trinity: {result['trinity_phase_correct']}, Boundaries: {result['professional_boundaries']}")
            
            if result["errors"]:
                print(f"   ⚠️  Errors: {', '.join(result['errors'])}")
        
        # Calculate overall accuracy
        if self.test_results["total_domains"] > 0:
            self.test_results["overall_accuracy"] = (
                self.test_results["passed_domains"] / self.test_results["total_domains"]
            ) * 100
        
        return self.test_results
    
    def print_summary_report(self):
        """Print comprehensive summary report."""
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE DOMAIN VALIDATION SUMMARY")
        print("=" * 60)
        
        print(f"🎯 Total Domains Tested: {self.test_results['total_domains']}")
        print(f"✅ Passed Domains: {self.test_results['passed_domains']}")
        print(f"❌ Failed Domains: {self.test_results['failed_domains']}")
        print(f"📈 Overall Accuracy: {self.test_results['overall_accuracy']:.1f}%")
        
        print(f"\n🧠 Trinity Architecture Phases:")
        for phase, count in self.test_results["trinity_phases"].items():
            print(f"   - {phase}: {count} domains")
        
        print(f"\n🚨 Crisis Intervention Domains: {self.test_results['crisis_intervention']}")
        print(f"💝 Emotional Intelligence Domains: {self.test_results['emotional_intelligence']}")
        print(f"🛡️ Professional Boundaries Domains: {self.test_results['professional_boundaries']}")
        
        # Domain categories summary
        healthcare_domains = [d for d in self.test_results["domain_details"].keys() 
                           if d in ['healthcare', 'mental_health', 'nutrition', 'fitness', 'sleep', 
                                   'stress_management', 'preventive_care', 'chronic_conditions', 
                                   'medication_management', 'emergency_care', 'women_health', 'senior_health']]
        
        space_domains = [d for d in self.test_results["domain_details"].keys() 
                        if d in ['space_research', 'aerospace_engineering', 'satellite_systems', 'space_mission_planning']]
        
        tech_domains = [d for d in self.test_results["domain_details"].keys() 
                       if d in ['programming', 'web_development', 'mobile_apps', 'ai_ml', 'cybersecurity']]
        
        print(f"\n🏥 Healthcare Domains: {len(healthcare_domains)} available")
        print(f"🚀 Space Technology Domains: {len(space_domains)} available")
        print(f"💻 Technology Domains: {len(tech_domains)} available")
        
        # Check specific domains you mentioned
        print(f"\n🔍 Specific Domain Status:")
        specific_domains = ['programming', 'space_research', 'aerospace_engineering', 'satellite_systems', 'space_mission_planning']
        for domain in specific_domains:
            if domain in self.test_results["domain_details"]:
                result = self.test_results["domain_details"][domain]
                status = "✅" if result["template_available"] and result["domain_expert_created"] else "❌"
                print(f"   - {domain}: {status}")
            else:
                print(f"   - {domain}: ❌ (not found)")
        
        # Failed domains details
        failed_domains = [domain for domain, result in self.test_results["domain_details"].items()
                         if sum([result["template_available"], result["domain_expert_created"], 
                                result["trinity_phase_correct"], result["professional_boundaries"]]) < 3]
        
        if failed_domains:
            print(f"\n⚠️ Failed Domains:")
            for domain in failed_domains:
                result = self.test_results["domain_details"][domain]
                print(f"   - {domain}: {result['errors']}")
        
        print("\n" + "=" * 60)
        
        # Save detailed report
        report_path = f"validation_reports/comprehensive_domain_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs("validation_reports", exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"📄 Detailed report saved to: {report_path}")

def main():
    """Main validation function."""
    validator = ComprehensiveDomainValidator()
    
    try:
        # Run comprehensive validation
        results = validator.run_comprehensive_validation()
        
        # Print summary report
        validator.print_summary_report()
        
        # Return success/failure
        if results["overall_accuracy"] >= 80.0:
            print("🎉 Comprehensive validation completed successfully!")
            return 0
        else:
            print("⚠️ Comprehensive validation completed with issues.")
            return 1
            
    except Exception as e:
        print(f"❌ Validation failed with error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main()) 