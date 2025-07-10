#!/usr/bin/env python3
"""
MeeTARA Lab - Comprehensive Healthcare Domain Validation
Tests all 12 healthcare domains for data generation, accuracy, and Trinity Architecture integration.
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

class HealthcareComprehensiveValidator:
    """
    Comprehensive validator for all 12 healthcare domains.
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
            "domain_results": {},
            "overall_accuracy": 0.0,
            "trinity_integration": {},
            "crisis_intervention": {},
            "emotional_intelligence": {},
            "professional_boundaries": {}
        }
        
        # All 12 healthcare domains from configuration (updated to general_health)
        self.healthcare_domains = [
            "general_health", "mental_health", "nutrition", "fitness", "sleep",
            "stress_management", "preventive_care", "chronic_conditions",
            "medication_management", "emergency_care", "women_health", "senior_health"
        ]
        
        # Expected Trinity phases for each domain
        self.expected_trinity_phases = {
            "general_health": "einstein_fusion",
            "mental_health": "einstein_fusion", 
            "nutrition": "perplexity_intelligence",
            "fitness": "arc_reactor_foundation",
            "sleep": "perplexity_intelligence",
            "stress_management": "einstein_fusion",
            "preventive_care": "perplexity_intelligence",
            "chronic_conditions": "perplexity_intelligence",
            "medication_management": "einstein_fusion",
            "emergency_care": "einstein_fusion",
            "women_health": "perplexity_intelligence",
            "senior_health": "perplexity_intelligence"
        }
        
        # Crisis intervention domains
        self.crisis_domains = [
            "general_health", "mental_health", "stress_management", 
            "medication_management", "emergency_care"
        ]
        
        # Safety-critical domains
        self.safety_critical_domains = [
            "general_health", "mental_health", "medication_management", "emergency_care"
        ]
    
    def validate_healthcare_domains(self) -> Dict[str, Any]:
        """
        Validate all 12 healthcare domains comprehensively.
        """
        print("🏥 Healthcare Comprehensive Domain Validation")
        print("=" * 60)
        print(f"Testing {len(self.healthcare_domains)} healthcare domains...")
        print()
        
        self.test_results["total_domains"] = len(self.healthcare_domains)
        
        for domain in self.healthcare_domains:
            print(f"📋 Testing Domain: {domain}")
            domain_result = self._validate_single_domain(domain)
            self.test_results["domain_results"][domain] = domain_result
            
            if domain_result["overall_status"] == "PASSED":
                self.test_results["passed_domains"] += 1
            else:
                self.test_results["failed_domains"] += 1
            
            print(f"   Status: {domain_result['overall_status']}")
            print(f"   Trinity Phase: {domain_result['trinity_phase']}")
            print(f"   Crisis Intervention: {domain_result['crisis_intervention']}")
            print(f"   Emotional Intelligence: {domain_result['emotional_intelligence']}")
            print(f"   Professional Boundaries: {domain_result['professional_boundaries']}")
            print()
        
        # Calculate overall accuracy
        total_score = sum(result["accuracy_score"] for result in self.test_results["domain_results"].values())
        self.test_results["overall_accuracy"] = total_score / len(self.healthcare_domains)
        
        # Generate comprehensive report
        self._generate_comprehensive_report()
        
        return self.test_results
    
    def _validate_single_domain(self, domain: str) -> Dict[str, Any]:
        """
        Validate a single healthcare domain.
        """
        result = {
            "domain": domain,
            "overall_status": "FAILED",
            "accuracy_score": 0.0,
            "trinity_phase": "unknown",
            "crisis_intervention": "FAILED",
            "emotional_intelligence": "FAILED", 
            "professional_boundaries": "FAILED",
            "data_generation": "FAILED",
            "template_availability": "FAILED",
            "domain_expert": "FAILED",
            "errors": []
        }
        
        try:
            # Test 1: Template Availability
            if domain in self.data_generator.domain_templates:
                result["template_availability"] = "PASSED"
                templates = self.data_generator.domain_templates[domain]
                if len(templates) >= 2:  # At least 2 scenarios
                    result["accuracy_score"] += 0.2
            else:
                result["errors"].append(f"No templates found for domain: {domain}")
            
            # Test 2: Domain Expert Creation
            domain_expert = self.data_generator._create_domain_expert_agent(domain)
            if domain_expert and domain_expert.get("domain") == domain:
                result["domain_expert"] = "PASSED"
                result["accuracy_score"] += 0.2
                
                # Check Trinity phase
                expected_phase = self.expected_trinity_phases.get(domain, "arc_reactor_foundation")
                actual_phase = domain_expert.get("trinity_phase", "unknown")
                result["trinity_phase"] = actual_phase
                
                if actual_phase == expected_phase:
                    result["accuracy_score"] += 0.2
                else:
                    result["errors"].append(f"Expected Trinity phase {expected_phase}, got {actual_phase}")
                
                # Check crisis intervention
                if domain in self.crisis_domains:
                    if domain_expert.get("crisis_intervention"):
                        result["crisis_intervention"] = "PASSED"
                        result["accuracy_score"] += 0.2
                    else:
                        result["errors"].append(f"Domain {domain} should have crisis intervention")
                else:
                    result["crisis_intervention"] = "NOT APPLICABLE"
                
                # Check emotional intelligence
                if domain_expert.get("emotional_intelligence"):
                    result["emotional_intelligence"] = "PASSED"
                    result["accuracy_score"] += 0.1
                else:
                    result["errors"].append(f"Domain {domain} should have emotional intelligence")
                
                # Check professional boundaries
                safety_level = domain_expert.get("safety_level", "unknown")
                if domain in self.safety_critical_domains:
                    if safety_level in ["maximum", "high"]:
                        result["professional_boundaries"] = "PASSED"
                        result["accuracy_score"] += 0.1
                    else:
                        result["errors"].append(f"Domain {domain} should have high safety level")
                else:
                    result["professional_boundaries"] = "PASSED"
                    result["accuracy_score"] += 0.1
            else:
                result["errors"].append(f"Failed to create domain expert for {domain}")
            
            # Test 3: Data Generation
            try:
                # Generate a sample conversation
                domain_config = {
                    "conversation_starters": [f"I need help with {domain}"],
                    "user_intents": ["general_inquiry"]
                }
                
                conversation = self.data_generator._generate_blended_conversation(domain, domain_config)
                
                if conversation and "conversation" in conversation:
                    result["data_generation"] = "PASSED"
                    result["accuracy_score"] += 0.2
                    
                    # Check for Trinity Architecture metadata
                    if "trinity_phase" in conversation:
                        result["accuracy_score"] += 0.1
                    
                    # Check for crisis intervention if applicable
                    if domain in self.crisis_domains:
                        if "crisis_intervention" in str(conversation).lower():
                            result["accuracy_score"] += 0.1
                else:
                    result["errors"].append(f"Failed to generate conversation for {domain}")
                    
            except Exception as e:
                result["errors"].append(f"Data generation error for {domain}: {str(e)}")
            
            # Determine overall status
            if result["accuracy_score"] >= 0.8:  # 80% threshold
                result["overall_status"] = "PASSED"
            elif result["accuracy_score"] >= 0.6:
                result["overall_status"] = "PARTIAL"
            else:
                result["overall_status"] = "FAILED"
                
        except Exception as e:
            result["errors"].append(f"Validation error for {domain}: {str(e)}")
        
        return result
    
    def _generate_comprehensive_report(self):
        """
        Generate comprehensive validation report.
        """
        print("=" * 60)
        print("🏥 HEALTHCARE COMPREHENSIVE VALIDATION REPORT")
        print("=" * 60)
        
        # Summary statistics
        total_domains = self.test_results["total_domains"]
        passed_domains = self.test_results["passed_domains"]
        failed_domains = self.test_results["failed_domains"]
        overall_accuracy = self.test_results["overall_accuracy"]
        
        print(f"📊 VALIDATION SUMMARY:")
        print(f"   Total Domains: {total_domains}")
        print(f"   Passed Domains: {passed_domains}")
        print(f"   Failed Domains: {failed_domains}")
        print(f"   Success Rate: {(passed_domains/total_domains)*100:.1f}%")
        print(f"   Overall Accuracy: {overall_accuracy*100:.1f}%")
        print()
        
        # Trinity Architecture summary
        trinity_phases = {}
        for domain, result in self.test_results["domain_results"].items():
            phase = result["trinity_phase"]
            trinity_phases[phase] = trinity_phases.get(phase, 0) + 1
        
        print("⚡ TRINITY ARCHITECTURE:")
        for phase, count in trinity_phases.items():
            print(f"   {phase}: {count} domains")
        print()
        
        # Crisis intervention summary
        crisis_count = sum(1 for result in self.test_results["domain_results"].values() 
                         if result["crisis_intervention"] == "PASSED")
        print(f"🏥 HEALTHCARE CAPABILITIES:")
        print(f"   Crisis Intervention: {crisis_count} domains")
        print(f"   Emotional Intelligence: {sum(1 for r in self.test_results['domain_results'].values() if r['emotional_intelligence'] == 'PASSED')} domains")
        print(f"   Professional Boundaries: {sum(1 for r in self.test_results['domain_results'].values() if r['professional_boundaries'] == 'PASSED')} domains")
        print()
        
        # Domain-specific results
        print("📋 DOMAIN-SPECIFIC RESULTS:")
        for domain, result in self.test_results["domain_results"].items():
            status_icon = "✅" if result["overall_status"] == "PASSED" else "❌"
            print(f"   {status_icon} {domain}: {result['accuracy_score']*100:.1f}% accuracy")
            if result["errors"]:
                for error in result["errors"][:2]:  # Show first 2 errors
                    print(f"      ⚠️  {error}")
        print()
        
        # Quality metrics
        print("📈 QUALITY METRICS:")
        print(f"   Template Availability: {sum(1 for r in self.test_results['domain_results'].values() if r['template_availability'] == 'PASSED')}/{total_domains}")
        print(f"   Domain Expert Creation: {sum(1 for r in self.test_results['domain_results'].values() if r['domain_expert'] == 'PASSED')}/{total_domains}")
        print(f"   Data Generation: {sum(1 for r in self.test_results['domain_results'].values() if r['data_generation'] == 'PASSED')}/{total_domains}")
        print()
        
        # Overall assessment
        if overall_accuracy >= 0.9:
            assessment = "✅ EXCELLENT"
        elif overall_accuracy >= 0.8:
            assessment = "✅ GOOD"
        elif overall_accuracy >= 0.7:
            assessment = "⚠️  NEEDS IMPROVEMENT"
        else:
            assessment = "❌ NEEDS MAJOR IMPROVEMENT"
        
        print(f"🎯 OVERALL ASSESSMENT: {assessment}")
        print(f"   Trinity Architecture: {'FULLY INTEGRATED' if overall_accuracy >= 0.8 else 'PARTIALLY INTEGRATED'}")
        print(f"   Crisis Intervention: {'ADVANCED CAPABILITIES' if crisis_count >= 5 else 'BASIC CAPABILITIES'}")
        print(f"   Emotional Intelligence: {'REAL-TIME DETECTION' if overall_accuracy >= 0.8 else 'LIMITED DETECTION'}")
        print(f"   Professional Boundaries: {'COMPREHENSIVE COMPLIANCE' if overall_accuracy >= 0.8 else 'BASIC COMPLIANCE'}")
        print()
        
        # Save detailed report
        report_path = f"tests/reports/healthcare_comprehensive_validation_report.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Detailed report saved to: {report_path}")
        print(f"⏱️  Validation completed in: {time.time():.2f} seconds")

def main():
    """Main validation function."""
    print("🚀 Starting Healthcare Comprehensive Domain Validation...")
    
    validator = HealthcareComprehensiveValidator()
    results = validator.validate_healthcare_domains()
    
    # Return exit code based on success rate
    success_rate = results["passed_domains"] / results["total_domains"]
    if success_rate >= 0.8:
        print("✅ Healthcare validation completed successfully!")
        return 0
    else:
        print("❌ Healthcare validation needs improvement!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 