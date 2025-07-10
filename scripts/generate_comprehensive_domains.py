#!/usr/bin/env python3
"""
Comprehensive Domain Template Generator for MeeTARA Lab
Generates rich multi-scenario templates for all 61+ domains from original TARA Universal Model
"""

import yaml
import json
from typing import Dict, List, Any
from pathlib import Path

class ComprehensiveDomainGenerator:
    """Generate comprehensive domain templates with rich multi-scenario format"""
    
    def __init__(self):
        self.domain_templates = {}
        
    def generate_all_domains(self):
        """Generate comprehensive templates for all 61+ domains"""
        
        # ===== HEALTHCARE DOMAINS (12 domains) =====
        self._generate_healthcare_domains()
        
        # ===== DAILY LIFE DOMAINS (12 domains) =====
        self._generate_daily_life_domains()
        
        # ===== BUSINESS DOMAINS (12 domains) =====
        self._generate_business_domains()
        
        # ===== EDUCATION DOMAINS (8 domains) =====
        self._generate_education_domains()
        
        # ===== TECHNOLOGY DOMAINS (6 domains) =====
        self._generate_technology_domains()
        
        # ===== SPACE TECHNOLOGY DOMAINS (4 domains) =====
        self._generate_space_technology_domains()
        
        # ===== CREATIVE DOMAINS (8 domains) =====
        self._generate_creative_domains()
        
        # ===== SPECIALIZED DOMAINS (4 domains) =====
        self._generate_specialized_domains()
        
        return self.domain_templates
    
    def _generate_healthcare_domains(self):
        """Generate comprehensive healthcare domain templates"""
        
        # General Health
        self.domain_templates["general_health"] = {
            "scenarios": [
                "emergency_crisis_intervention", "medical_guidance", "wellness_support",
                "preventive_care", "chronic_condition_management", "medication_safety",
                "health_screening", "lifestyle_guidance", "mental_health_referral",
                "crisis_intervention", "professional_consultation", "health_education"
            ],
            "user_intents": [
                "emergency_support", "medical_inquiry", "wellness_guidance",
                "preventive_care", "condition_management", "medication_help",
                "screening_inquiry", "lifestyle_advice", "mental_health_support",
                "crisis_help", "professional_consultation", "health_education"
            ],
            "conversation_starters": [
                "I'm feeling really sick and don't know what to do. Can you help me understand my symptoms?",
                "What are the warning signs I should look out for with this condition?",
                "How can I improve my overall health and wellness?",
                "What preventive measures should I take for my age group?",
                "I'm managing a chronic condition. What lifestyle changes would help?",
                "I'm worried about medication interactions. Can you help me understand?",
                "What health screenings should I get at my age?",
                "How can I maintain a healthy lifestyle with my busy schedule?",
                "I think I might need mental health support. What should I do?",
                "I'm in a health crisis and need immediate guidance.",
                "Should I see a specialist for this issue?",
                "How can I learn more about maintaining good health?"
            ],
            "response_patterns": [
                "emergency_guidance", "medical_education", "wellness_advice",
                "preventive_guidance", "condition_support", "medication_education",
                "screening_guidance", "lifestyle_advice", "mental_health_referral",
                "crisis_support", "professional_referral", "health_education"
            ],
            "trinity_phase": "einstein_fusion",
            "crisis_intervention": True,
            "emotional_intelligence": True,
            "professional_boundaries": True
        }
        
        # Mental Health
        self.domain_templates["mental_health"] = {
            "scenarios": [
                "crisis_intervention", "therapeutic_support", "emotional_guidance",
                "stress_management", "anxiety_support", "depression_help",
                "trauma_support", "relationship_issues", "self_care_guidance",
                "professional_referral", "coping_strategies", "wellness_planning"
            ],
            "user_intents": [
                "crisis_help", "therapeutic_guidance", "emotional_support",
                "stress_relief", "anxiety_management", "depression_support",
                "trauma_help", "relationship_advice", "self_care_guidance",
                "professional_help", "coping_help", "wellness_planning"
            ],
            "conversation_starters": [
                "I'm having a mental health crisis and need immediate help.",
                "I've been feeling really anxious lately. What can I do?",
                "I think I might be depressed. How do I know for sure?",
                "I'm dealing with trauma and don't know how to cope.",
                "My relationships are suffering because of my mental health.",
                "I need help developing better self-care practices.",
                "Should I see a therapist for these issues?",
                "What coping strategies work for anxiety?",
                "I'm feeling overwhelmed and stressed all the time.",
                "How can I support someone with mental health issues?",
                "I need help creating a wellness plan for myself.",
                "What are the signs that I need professional help?"
            ],
            "response_patterns": [
                "crisis_intervention", "therapeutic_guidance", "emotional_support",
                "stress_management", "anxiety_help", "depression_support",
                "trauma_guidance", "relationship_advice", "self_care_guidance",
                "professional_referral", "coping_strategies", "wellness_planning"
            ],
            "trinity_phase": "einstein_fusion",
            "crisis_intervention": True,
            "emotional_intelligence": True,
            "professional_boundaries": True
        }
        
        # Continue with other healthcare domains...
        healthcare_domains = [
            "nutrition", "fitness", "sleep", "stress_management", "preventive_care",
            "chronic_conditions", "medication_management", "emergency_care", 
            "women_health", "senior_health"
        ]
        
        for domain in healthcare_domains:
            self.domain_templates[domain] = self._create_healthcare_template(domain)
    
    def _create_healthcare_template(self, domain: str) -> Dict[str, Any]:
        """Create healthcare domain template with rich scenarios"""
        base_template = {
            "scenarios": [
                "emergency_crisis_intervention", "medical_guidance", "wellness_support",
                "preventive_care", "condition_management", "safety_guidance",
                "lifestyle_guidance", "professional_referral", "education_support",
                "crisis_intervention", "specialized_care", "health_planning"
            ],
            "user_intents": [
                "emergency_support", "medical_inquiry", "wellness_guidance",
                "preventive_care", "condition_help", "safety_concerns",
                "lifestyle_advice", "professional_help", "education_need",
                "crisis_help", "specialized_care", "health_planning"
            ],
            "conversation_starters": [
                f"I need help with {domain.replace('_', ' ')}. Can you guide me?",
                f"What should I know about {domain.replace('_', ' ')}?",
                f"I'm concerned about my {domain.replace('_', ' ')}. What can I do?",
                f"How can I improve my {domain.replace('_', ' ')}?",
                f"What are the best practices for {domain.replace('_', ' ')}?",
                f"I need professional help with {domain.replace('_', ' ')}.",
                f"What resources are available for {domain.replace('_', ' ')}?",
                f"How do I know if I need help with {domain.replace('_', ' ')}?",
                f"What are the warning signs for {domain.replace('_', ' ')} issues?",
                f"I want to learn more about {domain.replace('_', ' ')}.",
                f"How can I support someone with {domain.replace('_', ' ')} needs?",
                f"What should I expect with {domain.replace('_', ' ')} care?"
            ],
            "response_patterns": [
                "emergency_guidance", "medical_education", "wellness_advice",
                "preventive_guidance", "condition_support", "safety_advice",
                "lifestyle_guidance", "professional_referral", "education_support",
                "crisis_support", "specialized_guidance", "health_planning"
            ],
            "trinity_phase": "einstein_fusion",
            "crisis_intervention": True,
            "emotional_intelligence": True,
            "professional_boundaries": True
        }
        return base_template
    
    def _generate_daily_life_domains(self):
        """Generate comprehensive daily life domain templates"""
        daily_life_domains = [
            "parenting", "relationships", "personal_assistant", "communication",
            "home_management", "shopping", "planning", "transportation",
            "time_management", "decision_making", "conflict_resolution", "work_life_balance"
        ]
        
        for domain in daily_life_domains:
            self.domain_templates[domain] = self._create_daily_life_template(domain)
    
    def _create_daily_life_template(self, domain: str) -> Dict[str, Any]:
        """Create daily life domain template with rich scenarios"""
        return {
            "scenarios": [
                "daily_challenges", "life_guidance", "personal_support",
                "skill_development", "problem_solving", "relationship_help",
                "organization_help", "decision_support", "conflict_resolution",
                "work_life_balance", "personal_growth", "life_planning"
            ],
            "user_intents": [
                "daily_help", "life_guidance", "personal_support",
                "skill_development", "problem_solving", "relationship_help",
                "organization_help", "decision_support", "conflict_resolution",
                "work_life_balance", "personal_growth", "life_planning"
            ],
            "conversation_starters": [
                f"I need help with {domain.replace('_', ' ')}. Can you guide me?",
                f"How can I improve my {domain.replace('_', ' ')} skills?",
                f"I'm struggling with {domain.replace('_', ' ')}. What should I do?",
                f"What are the best practices for {domain.replace('_', ' ')}?",
                f"I want to learn more about {domain.replace('_', ' ')}.",
                f"How do I handle {domain.replace('_', ' ')} challenges?",
                f"What resources can help with {domain.replace('_', ' ')}?",
                f"I need advice on {domain.replace('_', ' ')}.",
                f"How can I be better at {domain.replace('_', ' ')}?",
                f"What should I know about {domain.replace('_', ' ')}?",
                f"I'm looking for {domain.replace('_', ' ')} guidance.",
                f"How do I approach {domain.replace('_', ' ')} effectively?"
            ],
            "response_patterns": [
                "daily_guidance", "life_advice", "personal_support",
                "skill_development", "problem_solving", "relationship_help",
                "organization_advice", "decision_support", "conflict_resolution",
                "work_life_balance", "personal_growth", "life_planning"
            ],
            "trinity_phase": "perplexity_intelligence",
            "crisis_intervention": False,
            "emotional_intelligence": True,
            "professional_boundaries": False
        }
    
    def _generate_business_domains(self):
        """Generate comprehensive business domain templates"""
        business_domains = [
            "entrepreneurship", "marketing", "sales", "customer_service",
            "project_management", "team_leadership", "financial_planning", "operations",
            "hr_management", "strategy", "consulting", "legal_business"
        ]
        
        for domain in business_domains:
            self.domain_templates[domain] = self._create_business_template(domain)
    
    def _create_business_template(self, domain: str) -> Dict[str, Any]:
        """Create business domain template with rich scenarios"""
        return {
            "scenarios": [
                "business_planning", "market_research", "strategy_development",
                "team_management", "financial_planning", "operations_optimization",
                "customer_engagement", "growth_strategies", "risk_management",
                "professional_development", "industry_analysis", "competitive_advantage"
            ],
            "user_intents": [
                "business_planning_help", "market_research_guidance", "strategy_development",
                "team_management_help", "financial_planning_guidance", "operations_optimization",
                "customer_engagement_help", "growth_strategies_guidance", "risk_management",
                "professional_development_help", "industry_analysis", "competitive_advantage"
            ],
            "conversation_starters": [
                f"I need help with {domain.replace('_', ' ')}. Can you guide me?",
                f"What are the best practices for {domain.replace('_', ' ')}?",
                f"How can I improve my {domain.replace('_', ' ')} skills?",
                f"I'm starting a new {domain.replace('_', ' ')} project. What should I know?",
                f"What strategies work best for {domain.replace('_', ' ')}?",
                f"I need advice on {domain.replace('_', ' ')} challenges.",
                f"How do I approach {domain.replace('_', ' ')} effectively?",
                f"What resources can help with {domain.replace('_', ' ')}?",
                f"I want to learn more about {domain.replace('_', ' ')}.",
                f"How can I be more successful in {domain.replace('_', ' ')}?",
                f"What should I focus on for {domain.replace('_', ' ')}?",
                f"I'm looking for {domain.replace('_', ' ')} guidance."
            ],
            "response_patterns": [
                "business_guidance", "strategy_advice", "professional_development",
                "team_management", "financial_planning", "operations_optimization",
                "customer_engagement", "growth_strategies", "risk_management",
                "professional_development", "industry_analysis", "competitive_advantage"
            ],
            "trinity_phase": "perplexity_intelligence",
            "crisis_intervention": False,
            "emotional_intelligence": True,
            "professional_boundaries": True
        }
    
    def _generate_education_domains(self):
        """Generate comprehensive education domain templates"""
        education_domains = [
            "academic_tutoring", "skill_development", "career_guidance", "exam_preparation",
            "language_learning", "research_assistance", "study_techniques", "educational_technology"
        ]
        
        for domain in education_domains:
            self.domain_templates[domain] = self._create_education_template(domain)
    
    def _create_education_template(self, domain: str) -> Dict[str, Any]:
        """Create education domain template with rich scenarios"""
        return {
            "scenarios": [
                "learning_support", "skill_development", "academic_guidance",
                "study_strategies", "career_planning", "research_methods",
                "technology_integration", "assessment_preparation", "learning_optimization",
                "educational_resources", "academic_advice", "learning_planning"
            ],
            "user_intents": [
                "learning_help", "skill_development", "academic_guidance",
                "study_strategies", "career_planning", "research_methods",
                "technology_integration", "assessment_preparation", "learning_optimization",
                "educational_resources", "academic_advice", "learning_planning"
            ],
            "conversation_starters": [
                f"I need help with {domain.replace('_', ' ')}. Can you guide me?",
                f"How can I improve my {domain.replace('_', ' ')} skills?",
                f"What are the best strategies for {domain.replace('_', ' ')}?",
                f"I'm struggling with {domain.replace('_', ' ')}. What should I do?",
                f"How do I approach {domain.replace('_', ' ')} effectively?",
                f"What resources can help with {domain.replace('_', ' ')}?",
                f"I want to learn more about {domain.replace('_', ' ')}.",
                f"How can I be more successful in {domain.replace('_', ' ')}?",
                f"What should I focus on for {domain.replace('_', ' ')}?",
                f"I'm looking for {domain.replace('_', ' ')} guidance.",
                f"How do I prepare for {domain.replace('_', ' ')} challenges?",
                f"What are the key principles of {domain.replace('_', ' ')}?"
            ],
            "response_patterns": [
                "learning_guidance", "skill_development", "academic_support",
                "study_strategies", "career_guidance", "research_methods",
                "technology_integration", "assessment_preparation", "learning_optimization",
                "educational_resources", "academic_advice", "learning_planning"
            ],
            "trinity_phase": "arc_reactor_foundation",
            "crisis_intervention": False,
            "emotional_intelligence": True,
            "professional_boundaries": True
        }
    
    def _generate_technology_domains(self):
        """Generate comprehensive technology domain templates"""
        tech_domains = [
            "programming", "ai_ml", "cybersecurity", "data_analysis", "tech_support", "software_development"
        ]
        
        for domain in tech_domains:
            self.domain_templates[domain] = self._create_technology_template(domain)
    
    def _create_technology_template(self, domain: str) -> Dict[str, Any]:
        """Create technology domain template with rich scenarios"""
        return {
            "scenarios": [
                "technical_guidance", "problem_solving", "skill_development",
                "best_practices", "troubleshooting", "innovation_support",
                "learning_guidance", "project_help", "technology_integration",
                "security_guidance", "optimization_help", "development_support"
            ],
            "user_intents": [
                "technical_help", "problem_solving", "skill_development",
                "best_practices", "troubleshooting", "innovation_support",
                "learning_guidance", "project_help", "technology_integration",
                "security_guidance", "optimization_help", "development_support"
            ],
            "conversation_starters": [
                f"I need help with {domain.replace('_', ' ')}. Can you guide me?",
                f"How can I improve my {domain.replace('_', ' ')} skills?",
                f"What are the best practices for {domain.replace('_', ' ')}?",
                f"I'm having trouble with {domain.replace('_', ' ')}. What should I do?",
                f"How do I approach {domain.replace('_', ' ')} problems?",
                f"What resources can help with {domain.replace('_', ' ')}?",
                f"I want to learn more about {domain.replace('_', ' ')}.",
                f"How can I be more effective in {domain.replace('_', ' ')}?",
                f"What should I focus on for {domain.replace('_', ' ')}?",
                f"I'm looking for {domain.replace('_', ' ')} guidance.",
                f"How do I solve {domain.replace('_', ' ')} challenges?",
                f"What are the key principles of {domain.replace('_', ' ')}?"
            ],
            "response_patterns": [
                "technical_guidance", "problem_solving", "skill_development",
                "best_practices", "troubleshooting", "innovation_support",
                "learning_guidance", "project_help", "technology_integration",
                "security_guidance", "optimization_help", "development_support"
            ],
            "trinity_phase": "perplexity_intelligence",
            "crisis_intervention": False,
            "emotional_intelligence": True,
            "professional_boundaries": True
        }
    
    def _generate_space_technology_domains(self):
        """Generate comprehensive space technology domain templates"""
        space_domains = [
            "space_research", "aerospace_engineering", "satellite_systems", "space_mission_planning"
        ]
        
        for domain in space_domains:
            self.domain_templates[domain] = self._create_space_technology_template(domain)
    
    def _create_space_technology_template(self, domain: str) -> Dict[str, Any]:
        """Create space technology domain template with rich scenarios"""
        return {
            "scenarios": [
                "research_guidance", "engineering_support", "mission_planning",
                "technical_analysis", "innovation_support", "project_management",
                "safety_guidance", "optimization_help", "development_support",
                "scientific_methods", "engineering_principles", "mission_design"
            ],
            "user_intents": [
                "research_help", "engineering_support", "mission_planning",
                "technical_analysis", "innovation_support", "project_management",
                "safety_guidance", "optimization_help", "development_support",
                "scientific_methods", "engineering_principles", "mission_design"
            ],
            "conversation_starters": [
                f"I need help with {domain.replace('_', ' ')}. Can you guide me?",
                f"How can I improve my {domain.replace('_', ' ')} skills?",
                f"What are the best practices for {domain.replace('_', ' ')}?",
                f"I'm working on a {domain.replace('_', ' ')} project. What should I know?",
                f"How do I approach {domain.replace('_', ' ')} challenges?",
                f"What resources can help with {domain.replace('_', ' ')}?",
                f"I want to learn more about {domain.replace('_', ' ')}.",
                f"How can I be more effective in {domain.replace('_', ' ')}?",
                f"What should I focus on for {domain.replace('_', ' ')}?",
                f"I'm looking for {domain.replace('_', ' ')} guidance.",
                f"How do I solve {domain.replace('_', ' ')} problems?",
                f"What are the key principles of {domain.replace('_', ' ')}?"
            ],
            "response_patterns": [
                "research_guidance", "engineering_support", "mission_planning",
                "technical_analysis", "innovation_support", "project_management",
                "safety_guidance", "optimization_help", "development_support",
                "scientific_methods", "engineering_principles", "mission_design"
            ],
            "trinity_phase": "perplexity_intelligence",
            "crisis_intervention": False,
            "emotional_intelligence": True,
            "professional_boundaries": True
        }
    
    def _generate_creative_domains(self):
        """Generate comprehensive creative domain templates"""
        creative_domains = [
            "creative_writing", "visual_arts", "music", "content_creation", "design",
            "innovation", "photography", "film_making"
        ]
        
        for domain in creative_domains:
            self.domain_templates[domain] = self._create_creative_template(domain)
    
    def _create_creative_template(self, domain: str) -> Dict[str, Any]:
        """Create creative domain template with rich scenarios"""
        return {
            "scenarios": [
                "creative_guidance", "skill_development", "inspiration_support",
                "technique_help", "project_guidance", "artistic_development",
                "innovation_support", "creative_process", "artistic_expression",
                "skill_improvement", "creative_problem_solving", "artistic_planning"
            ],
            "user_intents": [
                "creative_help", "skill_development", "inspiration_support",
                "technique_help", "project_guidance", "artistic_development",
                "innovation_support", "creative_process", "artistic_expression",
                "skill_improvement", "creative_problem_solving", "artistic_planning"
            ],
            "conversation_starters": [
                f"I need help with {domain.replace('_', ' ')}. Can you guide me?",
                f"How can I improve my {domain.replace('_', ' ')} skills?",
                f"What are the best techniques for {domain.replace('_', ' ')}?",
                f"I'm working on a {domain.replace('_', ' ')} project. What should I know?",
                f"How do I approach {domain.replace('_', ' ')} challenges?",
                f"What resources can help with {domain.replace('_', ' ')}?",
                f"I want to learn more about {domain.replace('_', ' ')}.",
                f"How can I be more creative in {domain.replace('_', ' ')}?",
                f"What should I focus on for {domain.replace('_', ' ')}?",
                f"I'm looking for {domain.replace('_', ' ')} inspiration.",
                f"How do I develop my {domain.replace('_', ' ')} style?",
                f"What are the key principles of {domain.replace('_', ' ')}?"
            ],
            "response_patterns": [
                "creative_guidance", "skill_development", "inspiration_support",
                "technique_help", "project_guidance", "artistic_development",
                "innovation_support", "creative_process", "artistic_expression",
                "skill_improvement", "creative_problem_solving", "artistic_planning"
            ],
            "trinity_phase": "arc_reactor_foundation",
            "crisis_intervention": False,
            "emotional_intelligence": True,
            "professional_boundaries": False
        }
    
    def _generate_specialized_domains(self):
        """Generate comprehensive specialized domain templates"""
        specialized_domains = [
            "legal_assistance", "financial_planning", "scientific_research", "engineering"
        ]
        
        for domain in specialized_domains:
            self.domain_templates[domain] = self._create_specialized_template(domain)
    
    def _create_specialized_template(self, domain: str) -> Dict[str, Any]:
        """Create specialized domain template with rich scenarios"""
        return {
            "scenarios": [
                "professional_guidance", "expert_advice", "specialized_support",
                "technical_analysis", "best_practices", "industry_insights",
                "regulatory_guidance", "quality_assurance", "innovation_support",
                "risk_management", "compliance_help", "professional_development"
            ],
            "user_intents": [
                "professional_help", "expert_advice", "specialized_support",
                "technical_analysis", "best_practices", "industry_insights",
                "regulatory_guidance", "quality_assurance", "innovation_support",
                "risk_management", "compliance_help", "professional_development"
            ],
            "conversation_starters": [
                f"I need help with {domain.replace('_', ' ')}. Can you guide me?",
                f"What are the best practices for {domain.replace('_', ' ')}?",
                f"How can I improve my {domain.replace('_', ' ')} skills?",
                f"I'm working on a {domain.replace('_', ' ')} project. What should I know?",
                f"How do I approach {domain.replace('_', ' ')} challenges?",
                f"What resources can help with {domain.replace('_', ' ')}?",
                f"I want to learn more about {domain.replace('_', ' ')}.",
                f"How can I be more effective in {domain.replace('_', ' ')}?",
                f"What should I focus on for {domain.replace('_', ' ')}?",
                f"I'm looking for {domain.replace('_', ' ')} guidance.",
                f"How do I solve {domain.replace('_', ' ')} problems?",
                f"What are the key principles of {domain.replace('_', ' ')}?"
            ],
            "response_patterns": [
                "professional_guidance", "expert_advice", "specialized_support",
                "technical_analysis", "best_practices", "industry_insights",
                "regulatory_guidance", "quality_assurance", "innovation_support",
                "risk_management", "compliance_help", "professional_development"
            ],
            "trinity_phase": "perplexity_intelligence",
            "crisis_intervention": False,
            "emotional_intelligence": True,
            "professional_boundaries": True
        }
    
    def save_templates(self, output_path: str = "trinity_core/agents/comprehensive_domain_templates.py"):
        """Save comprehensive domain templates to file"""
        with open(output_path, 'w') as f:
            f.write('"""\n')
            f.write('Comprehensive Domain Templates for MeeTARA Lab\n')
            f.write('Generated from original TARA Universal Model with rich multi-scenario format\n')
            f.write('"""\n\n')
            f.write('from typing import Dict, Any\n\n')
            f.write('COMPREHENSIVE_DOMAIN_TEMPLATES = {\n')
            
            for domain, template in self.domain_templates.items():
                f.write(f'    "{domain}": {json.dumps(template, indent=8)},\n')
            
            f.write('}\n')
        
        print(f"✅ Comprehensive domain templates saved to {output_path}")
        print(f"📊 Total domains: {len(self.domain_templates)}")

def main():
    """Generate comprehensive domain templates"""
    generator = ComprehensiveDomainGenerator()
    templates = generator.generate_all_domains()
    
    print(f"🚀 Generated comprehensive templates for {len(templates)} domains")
    print("\n📋 Domain Categories:")
    
    categories = {
        "Healthcare": [d for d in templates.keys() if any(h in d for h in ["health", "mental", "nutrition", "fitness", "sleep", "stress", "preventive", "chronic", "medication", "emergency", "women", "senior"])],
        "Daily Life": [d for d in templates.keys() if any(dl in d for dl in ["parenting", "relationships", "personal", "communication", "home", "shopping", "planning", "transportation", "time", "decision", "conflict", "work_life"])],
        "Business": [d for d in templates.keys() if any(b in d for b in ["entrepreneurship", "marketing", "sales", "customer", "project", "team", "financial", "operations", "hr", "strategy", "consulting", "legal_business"])],
        "Education": [d for d in templates.keys() if any(e in d for e in ["academic", "skill", "career", "exam", "language", "research", "study", "educational"])],
        "Technology": [d for d in templates.keys() if any(t in d for t in ["programming", "ai_ml", "cybersecurity", "data", "tech", "software"])],
        "Space Technology": [d for d in templates.keys() if any(s in d for s in ["space", "aerospace", "satellite"])],
        "Creative": [d for d in templates.keys() if any(c in d for c in ["creative", "visual", "music", "content", "design", "innovation", "photography", "film"])],
        "Specialized": [d for d in templates.keys() if any(sp in d for sp in ["legal", "financial", "scientific", "engineering"])]
    }
    
    for category, domains in categories.items():
        if domains:
            print(f"  {category}: {len(domains)} domains")
    
    generator.save_templates()
    
    print(f"\n✅ Successfully generated comprehensive templates for all {len(templates)} domains!")
    print("🎯 All domains now have rich multi-scenario format matching original TARA Universal Model")

if __name__ == "__main__":
    main() 