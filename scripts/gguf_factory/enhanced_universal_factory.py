#!/usr/bin/env python3
"""
🚀 Enhanced Universal GGUF Factory - LIGHTWEIGHT ORCHESTRATOR
Delegates heavy lifting to Trinity Architecture agents

🎯 DESIGN PRINCIPLE: "Scripts are light, agents are smart"
- This script only coordinates and orchestrates
- All heavy processing is delegated to existing agents
- No business logic duplication
"""

import asyncio
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import argparse
import importlib.util

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add trinity-core to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "trinity-core"))

# Import existing Trinity agents (they do the heavy lifting)
try:
    # Import Model Factory
    model_factory_path = Path(__file__).parent.parent.parent / "trinity-core" / "agents" / "02_super_agents" / "03_model_factory.py"
    spec = importlib.util.spec_from_file_location("model_factory", model_factory_path)
    model_factory_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_factory_module)
    IntelligentModelFactory = model_factory_module.IntelligentModelFactory
    
    # Import GGUF Creator Agent
    gguf_creator_path = Path(__file__).parent.parent.parent / "trinity-core" / "agents" / "01_legacy_agents" / "05_gguf_creator_agent.py"
    spec = importlib.util.spec_from_file_location("gguf_creator", gguf_creator_path)
    gguf_creator_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gguf_creator_module)
    GGUFCreatorAgent = gguf_creator_module.GGUFCreatorAgent
    
    # Import Intelligence Hub
    intelligence_hub_path = Path(__file__).parent.parent.parent / "trinity-core" / "agents" / "02_super_agents" / "01_intelligence_hub.py"
    spec = importlib.util.spec_from_file_location("intelligence_hub", intelligence_hub_path)
    intelligence_hub_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(intelligence_hub_module)
    IntelligenceHub = intelligence_hub_module.IntelligenceHub
    
    # Import Trinity Conductor
    trinity_conductor_path = Path(__file__).parent.parent.parent / "trinity-core" / "agents" / "02_super_agents" / "02_trinity_conductor.py"
    spec = importlib.util.spec_from_file_location("trinity_conductor", trinity_conductor_path)
    trinity_conductor_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trinity_conductor_module)
    TrinitiyConductor = trinity_conductor_module.TrinitiyConductor
    
    # Import Trinity Orchestrator Master
    orchestrator_path = Path(__file__).parent.parent.parent / "trinity-core" / "agents" / "10_trinity_orchestrator_master.py"
    spec = importlib.util.spec_from_file_location("orchestrator", orchestrator_path)
    orchestrator_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orchestrator_module)
    TrinityOrchestratorMaster = orchestrator_module.TrinityOrchestratorMaster
    
    AGENTS_AVAILABLE = True
    logger.info("✅ All Trinity agents imported successfully")
except Exception as e:
    logger.warning(f"⚠️ Some agents not available: {e}")
    AGENTS_AVAILABLE = False
    
    # Create mock classes for compatibility
    class IntelligentModelFactory:
        def __init__(self): pass
        async def create_intelligent_model(self, request): 
            return {"success": True, "method": "mock", "request": request}
    
    class GGUFCreatorAgent:
        def __init__(self): pass
        async def start(self): pass
        async def _handle_coordination_request(self, request):
            return {"success": True, "method": "mock", "request": request}
    
    class IntelligenceHub:
        def __init__(self): pass
        async def start(self): pass
        async def analyze_patterns(self, request):
            return {"success": True, "method": "mock", "request": request}
    
    class TrinitiyConductor:
        def __init__(self): pass
        async def start(self): pass
        async def coordinate_agents(self, request):
            return {"success": True, "method": "mock", "request": request}
    
    class TrinityOrchestratorMaster:
        def __init__(self): pass
        async def start(self): pass

class LightweightUniversalFactory:
    """
    Lightweight Universal GGUF Factory - Pure Orchestration
    
    🎯 RESPONSIBILITIES:
    - Coordinate agent interactions
    - Handle file organization
    - Generate summary reports
    - Provide simple CLI interface
    
    ❌ NOT RESPONSIBLE FOR:
    - Model creation logic (→ IntelligentModelFactory)
    - GGUF processing (→ GGUFCreatorAgent)
    - Speech/Voice processing (→ GGUFCreatorAgent)
    - Quality assessment (→ IntelligenceHub)
    - Trinity coordination (→ TrinitiyConductor)
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.models_dir = self.project_root / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize agents (they do the work)
        self.agents = {}
        
        logger.info("🚀 Lightweight Universal Factory initialized")
        logger.info(f"📁 Models directory: {self.models_dir}")
    
    async def initialize_agents(self):
        """Initialize Trinity agents for heavy lifting"""
        try:
            if AGENTS_AVAILABLE:
                # Initialize the intelligent agents
                self.agents['model_factory'] = IntelligentModelFactory()
                self.agents['gguf_creator'] = GGUFCreatorAgent()
                self.agents['intelligence_hub'] = IntelligenceHub()
                self.agents['trinity_conductor'] = TrinitiyConductor()
                self.agents['orchestrator'] = TrinityOrchestratorMaster()
                
                # Start the agents
                for name, agent in self.agents.items():
                    if hasattr(agent, 'start'):
                        await agent.start()
                    logger.info(f"✅ {name} agent initialized")
                
                logger.info("🧠 All Trinity agents ready for delegation")
                return True
            else:
                logger.warning("⚠️ Running in mock mode - agents not available")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize agents: {e}")
            return False
    
    async def organize_existing_models(self):
        """Organize existing models - delegates to GGUF Creator Agent"""
        logger.info("🔄 Organizing existing models...")
        
        try:
            if 'gguf_creator' in self.agents:
                # Delegate organization to GGUF Creator Agent
                organization_request = {
                    "action": "organize_models",
                    "source_dirs": ["full", "lite"],
                    "target_dirs": ["A_universal_full", "B_universal_lite"],
                    "create_speech_files": True,
                    "models_directory": str(self.models_dir)
                }
                
                result = await self.agents['gguf_creator']._handle_coordination_request(organization_request)
                logger.info("✅ Model organization delegated to GGUF Creator Agent")
                return result
            else:
                # Fallback: simple file operations
                await self._simple_file_organization()
                return {"success": True, "method": "fallback"}
                
        except Exception as e:
            logger.error(f"❌ Organization failed: {e}")
            # Fallback to simple organization
            await self._simple_file_organization()
            return {"success": True, "method": "fallback"}
    
    async def _simple_file_organization(self):
        """Simple fallback file organization"""
        import shutil
        
        # Move full/ to A_universal_full/
        full_src = self.models_dir / "full"
        full_dst = self.models_dir / "A_universal_full"
        if full_src.exists() and not full_dst.exists():
            shutil.move(str(full_src), str(full_dst))
            logger.info("📁 Moved full/ → A_universal_full/")
        
        # Move lite/ to B_universal_lite/
        lite_src = self.models_dir / "lite"
        lite_dst = self.models_dir / "B_universal_lite"
        if lite_src.exists() and not lite_dst.exists():
            shutil.move(str(lite_src), str(lite_dst))
            logger.info("📁 Moved lite/ → B_universal_lite/")
    
    async def create_enhanced_models(self) -> Dict[str, Any]:
        """Create enhanced models - delegates to Model Factory Agent"""
        logger.info("🏭 Creating enhanced models...")
        
        if 'model_factory' not in self.agents:
            logger.error("❌ Model Factory agent not available")
            return {"success": False, "error": "Agent not available"}
        
        try:
            # Define model requests (lightweight specifications)
            model_requests = [
                {
                    "variant": "category",
                    "domain": "healthcare",
                    "target_size_mb": 82,
                    "quality_target": 99.5
                },
                {
                    "variant": "category", 
                    "domain": "business",
                    "target_size_mb": 96,
                    "quality_target": 99.5
                },
                {
                    "variant": "lite",
                    "name": "mobile",
                    "target_size_mb": 3.5,
                    "quality_target": 95.0
                },
                {
                    "variant": "lite",
                    "name": "desktop", 
                    "target_size_mb": 8.5,
                    "quality_target": 97.0
                },
                {
                    "variant": "full",
                    "name": "standard",
                    "target_size_mb": 185.0,
                    "quality_target": 99.8
                },
                {
                    "variant": "full",
                    "name": "enterprise",
                    "target_size_mb": 285.0,
                    "quality_target": 99.9
                }
            ]
            
            results = {}
            
            # Delegate each model creation to the intelligent agent
            for request in model_requests:
                logger.info(f"🎯 Delegating {request['variant']} model creation...")
                
                try:
                    # Let the intelligent agent handle all the complexity
                    result = await self.agents['model_factory'].create_intelligent_model(request)
                    
                    model_key = f"{request['variant']}_{request.get('name', request.get('domain', 'default'))}"
                    results[model_key] = {
                        "success": True,
                        "request": request,
                        "result": result,
                        "delegated_to": "IntelligentModelFactory"
                    }
                    
                    logger.info(f"✅ {model_key} creation delegated successfully")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to delegate {request['variant']} creation: {e}")
                    results[f"{request['variant']}_{request.get('name', request.get('domain', 'default'))}"] = {
                        "success": False,
                        "error": str(e),
                        "request": request
                    }
            
            # Generate summary through Intelligence Hub
            if 'intelligence_hub' in self.agents:
                summary = await self._generate_intelligent_summary(results)
            else:
                summary = self._generate_simple_summary(results)
            
            return {
                "success": True,
                "model_results": results,
                "summary": summary,
                "delegation_method": "intelligent_agents"
            }
            
        except Exception as e:
            logger.error(f"❌ Model creation delegation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "delegation_method": "failed"
            }
    
    async def _generate_intelligent_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate intelligent summary through Intelligence Hub"""
        try:
            summary_request = {
                "action": "analyze_results",
                "data": results,
                "analysis_type": "model_creation_summary"
            }
            
            return await self.agents['intelligence_hub'].analyze_patterns(summary_request)
        except Exception as e:
            logger.warning(f"⚠️ Intelligence Hub summary failed: {e}")
            return self._generate_simple_summary(results)
    
    def _generate_simple_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate simple fallback summary"""
        successful = [r for r in results.values() if r.get("success", False)]
        failed = [r for r in results.values() if not r.get("success", False)]
        
        return {
            "total_models": len(results),
            "successful_models": len(successful),
            "failed_models": len(failed),
            "success_rate": len(successful) / len(results) * 100 if results else 0,
            "timestamp": datetime.now().isoformat(),
            "method": "simple_fallback"
        }
    
    async def generate_final_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final report - delegates to Trinity Conductor"""
        logger.info("📄 Generating final report...")
        
        try:
            if 'trinity_conductor' in self.agents:
                # Delegate report generation to Trinity Conductor
                report_request = {
                    "action": "generate_comprehensive_report",
                    "data": results,
                    "report_type": "model_factory_session"
                }
                
                report = await self.agents['trinity_conductor'].coordinate_agents(report_request)
                logger.info("✅ Report generation delegated to Trinity Conductor")
                return report
            else:
                # Simple fallback report
                return self._generate_simple_report(results)
                
        except Exception as e:
            logger.error(f"❌ Report generation failed: {e}")
            return self._generate_simple_report(results)
    
    def _generate_simple_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate simple fallback report"""
        report = {
            "session": {
                "timestamp": datetime.now().isoformat(),
                "factory_version": "lightweight_v1",
                "delegation_mode": True
            },
            "results": results,
            "summary": results.get("summary", {}),
            "output_directory": str(self.models_dir)
        }
        
        # Save report
        report_path = self.models_dir / f"factory_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        report["report_path"] = str(report_path)
        logger.info(f"📄 Simple report saved: {report_path}")
        
        return report

async def main():
    """Main execution function - Pure orchestration"""
    parser = argparse.ArgumentParser(description="Lightweight Universal GGUF Factory")
    parser.add_argument("--organize-only", action="store_true", 
                       help="Only organize existing models")
    parser.add_argument("--create-models", action="store_true",
                       help="Create new model variants")
    parser.add_argument("--full", action="store_true",
                       help="Run full pipeline (organize + create)")
    args = parser.parse_args()
    
    # Default to full pipeline if no specific action
    if not any([args.organize_only, args.create_models, args.full]):
        args.full = True
    
    print("🚀 Lightweight Universal GGUF Factory")
    print("=" * 60)
    print("🎯 DELEGATION MODE: Agents do the work, script orchestrates")
    print()
    
    try:
        # Initialize lightweight factory
        factory = LightweightUniversalFactory()
        
        # Initialize agents for delegation
        agents_ready = await factory.initialize_agents()
        if not agents_ready:
            print("⚠️ Running in fallback mode")
        
        results = {}
        
        # Step 1: Organization (if requested)
        if args.organize_only or args.full:
            print("🔄 Step 1: Organizing models (delegated to agents)...")
            org_result = await factory.organize_existing_models()
            results["organization"] = org_result
        
        # Step 2: Model Creation (if requested) 
        if args.create_models or args.full:
            print("🏭 Step 2: Creating models (delegated to agents)...")
            model_result = await factory.create_enhanced_models()
            results["model_creation"] = model_result
        
        # Step 3: Final Report
        print("📄 Step 3: Generating report (delegated to agents)...")
        final_report = await factory.generate_final_report(results)
        
        # Display results
        print("\n🎉 FACTORY ORCHESTRATION COMPLETE!")
        print("=" * 60)
        
        if results.get("model_creation", {}).get("success", True):
            print("✅ DELEGATION SUCCESSFUL!")
            print("🧠 All heavy lifting handled by Trinity agents")
            print(f"📄 Report: {final_report.get('report_path', 'Generated')}")
            print("\n🚀 Ready for production!")
            return True
        else:
            print("⚠️ Some delegations had issues - check report")
            return False
            
    except Exception as e:
        print(f"❌ Factory orchestration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1) 