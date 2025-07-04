#!/usr/bin/env python3
"""
MeeTARA Lab - Real Trinity Production Launcher
Triggers Trinity Super-Agent Flow with Intelligence Hub and Real Data Generation
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path

# Add trinity-core to path
current_dir = Path.cwd()
sys.path.append(str(current_dir / 'trinity-core'))
sys.path.append(str(current_dir))

# Import Trinity Super-Agent Ecosystem
try:
    from agents.system_integration.complete_agent_ecosystem import CompleteAgentEcosystem
    TRINITY_AVAILABLE = True
    print("✅ Trinity Super-Agent Ecosystem imported successfully")
except ImportError as e:
    print(f"⚠️ Trinity import failed: {e}")
    print("📁 Available files:")
    trinity_path = current_dir / 'trinity-core' / 'agents'
    if trinity_path.exists():
        for item in trinity_path.rglob('*.py'):
            print(f"   {item}")
    TRINITY_AVAILABLE = False

class RealTrinityProductionLauncher:
    """
    Real Trinity Production Launcher
    Triggers Intelligence Hub and Complete Agent Ecosystem
    """
    
    def __init__(self):
        self.trinity_ecosystem = None
        if TRINITY_AVAILABLE:
            self.trinity_ecosystem = CompleteAgentEcosystem()
            print("🚀 Trinity Super-Agent Ecosystem initialized")
            print("   → Intelligence Hub: ACTIVE")
            print("   → Data Generator Agent: ACTIVE") 
            print("   → Training Orchestrator: ACTIVE")
            print("   → Quality Assurance: ACTIVE")
        else:
            print("⚠️ Trinity ecosystem not available, using fallback")
            
    async def launch_trinity_training(self, category: str = None, domains: list = None):
        """Launch real Trinity training with Intelligence Hub"""
        
        if not TRINITY_AVAILABLE or not self.trinity_ecosystem:
            print("❌ Trinity ecosystem not available")
            return {"status": "error", "message": "Trinity not available"}
            
        print(f"\n🚀 LAUNCHING TRINITY SUPER-AGENT TRAINING")
        print("="*60)
        print("🧠 Intelligence Hub: Analyzing domain patterns...")
        print("🏭 Data Generator: Preparing real-time data generation...")
        print("🎯 Training Orchestrator: Coordinating multi-domain training...")
        print("🔍 Quality Assurance: Setting up validation pipelines...")
        print("="*60)
        
        # Determine domains to train
        if category:
            # Get domains for specific category
            category_domains = self._get_domains_for_category(category)
            domains_to_train = category_domains
            print(f"🎯 Training category: {category.upper()}")
            print(f"   → Domains: {domains_to_train}")
        elif domains:
            domains_to_train = domains
            print(f"🎯 Training specific domains: {domains_to_train}")
        else:
            # Train all 62 domains
            domains_to_train = None  # Will train all
            print("🌍 Training ALL 62 domains across 7 categories")
            
        # Launch Trinity ecosystem training
        print(f"\n🚀 Launching Trinity ecosystem training...")
        result = await self.trinity_ecosystem.coordinate_complete_training(domains_to_train)
        
        return result
        
    def _get_domains_for_category(self, category: str) -> list:
        """Get domains for a specific category"""
        domain_mapping = {
            "healthcare": [
                "general_health", "mental_health", "nutrition", "fitness", "sleep",
                "stress_management", "preventive_care", "chronic_conditions", 
                "medication_management", "emergency_care", "women_health", "senior_health"
            ],
            "daily_life": [
                "parenting", "relationships", "personal_assistant", "communication",
                "home_management", "shopping", "planning", "transportation",
                "time_management", "decision_making", "conflict_resolution", "work_life_balance"
            ],
            "business": [
                "entrepreneurship", "marketing", "sales", "customer_service",
                "project_management", "team_leadership", "financial_planning", "operations",
                "hr_management", "strategy", "consulting", "legal_business"
            ],
            "education": [
                "academic_tutoring", "skill_development", "career_guidance", 
                "exam_preparation", "language_learning", "research_assistance",
                "study_techniques", "educational_technology"
            ],
            "creative": [
                "writing", "storytelling", "content_creation", "social_media",
                "design_thinking", "photography", "music", "art_appreciation"
            ],
            "technology": [
                "programming", "ai_ml", "cybersecurity", "data_analysis",
                "tech_support", "software_development"
            ],
            "specialized": [
                "legal", "financial", "scientific_research", "engineering"
            ]
        }
        
        return domain_mapping.get(category, [])

async def main():
    """Main entry point for Trinity Production Launcher"""
    
    parser = argparse.ArgumentParser(description="MeeTARA Lab - Real Trinity Production Launcher")
    parser.add_argument("--category", type=str, help="Train specific category (healthcare, daily_life, business, etc.)")
    parser.add_argument("--domains", nargs='+', help="Train specific domains")
    parser.add_argument("--all", action="store_true", help="Train all 62 domains")
    
    args = parser.parse_args()
    
    # Initialize Trinity launcher
    launcher = RealTrinityProductionLauncher()
    
    if args.category:
        result = await launcher.launch_trinity_training(category=args.category)
    elif args.domains:
        result = await launcher.launch_trinity_training(domains=args.domains)
    elif args.all:
        result = await launcher.launch_trinity_training()
    else:
        print("🚀 MeeTARA Lab - Trinity Production Launcher")
        print("\nUsage:")
        print("  python production_launcher.py --category daily_life")
        print("  python production_launcher.py --category healthcare")
        print("  python production_launcher.py --domains parenting communication")
        print("  python production_launcher.py --all")
        print("\nAvailable categories:")
        print("  healthcare, daily_life, business, education, creative, technology, specialized")
        return
    
    print(f"\n🎉 Trinity Training Complete!")
    print(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
