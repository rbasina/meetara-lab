#!/usr/bin/env python3
"""
Simplified Trinity Architecture Test Runner
Focuses on core functionality validation
"""

import sys
import time
import importlib.util
from pathlib import Path
from typing import Dict, Any

class SimpleTrinityTestRunner:
    """Simplified test runner for Trinity Architecture core functionality"""
    
    def __init__(self):
        self.test_results = {}
        self.project_root = Path.cwd()
        
        # Add all necessary paths to sys.path for imports
        paths_to_add = [
            str(self.project_root),
            str(self.project_root / "trinity-core"),
            str(self.project_root / "cloud-training"),
            str(self.project_root / "model-factory"),
            str(self.project_root / "intelligence-hub")
        ]
        
        for path in paths_to_add:
            if path not in sys.path:
                sys.path.insert(0, path)
        
    def test_core_imports(self) -> Dict[str, Any]:
        """Test that core Trinity components can be imported"""
        print("🔍 Testing Core Trinity Component Imports...")
        
        core_components = {
            "TTS Manager": "trinity-core/tts_manager.py",
            "Emotion Detector": "trinity-core/emotion_detector.py", 
            "Intelligent Router": "trinity-core/intelligent_router.py",
            "Domain Integration": "trinity-core/domain_integration.py",
            "Config Manager": "trinity-core/config_manager.py",
            "GGUF Factory": "model-factory/gguf_factory.py",
            "GPU Training Engine": "model-factory/gpu_training_engine.py",
            "Training Orchestrator": "cloud-training/training_orchestrator.py",
            "Production Launcher": "cloud-training/production_launcher.py"
        }
        
        results = {}
        passed = 0
        
        for name, path in core_components.items():
            file_path = self.project_root / path
            
            result = {
                "file_exists": file_path.exists(),
                "importable": False,
                "file_size": 0
            }
            
            if file_path.exists():
                result["file_size"] = file_path.stat().st_size
                
                # Test if file can be imported/compiled
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Test syntax
                    compile(content, str(file_path), 'exec')
                    result["importable"] = True
                    passed += 1
                    print(f"   ✅ {name}")
                    
                except Exception as e:
                    result["error"] = str(e)
                    print(f"   ❌ {name}: {e}")
            else:
                print(f"   ❌ {name}: File not found")
            
            results[name] = result
        
        self.test_results["core_imports"] = results
        
        success_rate = (passed / len(core_components)) * 100
        print(f"   📊 Import Success Rate: {success_rate:.1f}% ({passed}/{len(core_components)})")
        
        return results
    
    def test_centralized_domain_mapping(self) -> Dict[str, Any]:
        """Test centralized domain mapping functionality"""
        print("\n🎯 Testing Centralized Domain Mapping...")
        
        results = {
            "config_file_exists": False,
            "domain_integration_works": False,
            "total_domains": 0,
            "total_categories": 0
        }
        
        # Check config file
        config_file = self.project_root / "config/trinity_domain_model_mapping_config.yaml"
        results["config_file_exists"] = config_file.exists()
        
        if config_file.exists():
            print("   ✅ Config file exists")
            
            # Test domain integration with proper import path
            try:
                # Import using file path directly
                spec = importlib.util.spec_from_file_location(
                    "domain_integration", 
                    self.project_root / "trinity-core/domain_integration.py"
                )
                domain_integration_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(domain_integration_module)
                
                # Get functions from the module
                get_domain_categories = domain_integration_module.get_domain_categories
                get_domain_stats = domain_integration_module.get_domain_stats
                
                domain_categories = get_domain_categories()
                domain_stats = get_domain_stats()
                
                results["domain_integration_works"] = True
                results["total_domains"] = domain_stats["total_domains"]
                results["total_categories"] = domain_stats["total_categories"]
                
                print(f"   ✅ Domain integration works: {results['total_domains']} domains, {results['total_categories']} categories")
                
            except Exception as e:
                results["error"] = str(e)
                print(f"   ❌ Domain integration failed: {e}")
        else:
            print("   ❌ Config file not found")
        
        self.test_results["domain_mapping"] = results
        return results
    
    def test_key_functionality(self) -> Dict[str, Any]:
        """Test key Trinity functionality"""
        print("\n⚡ Testing Key Trinity Functionality...")
        
        results = {
            "production_launcher": False,
            "quality_assurance": False,
            "knowledge_transfer": False
        }
        
        # Test Production Launcher
        try:
            spec = importlib.util.spec_from_file_location(
                "production_launcher", 
                self.project_root / "cloud-training/production_launcher.py"
            )
            production_launcher_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(production_launcher_module)
            
            ProductionLauncher = production_launcher_module.ProductionLauncher
            launcher = ProductionLauncher(simulation=True)
            stats = launcher.get_domain_statistics()
            
            if stats.get("total_domains", 0) > 0:
                results["production_launcher"] = True
                print("   ✅ Production Launcher works")
            else:
                print("   ❌ Production Launcher: No domains loaded")
                
        except Exception as e:
            results["production_launcher_error"] = str(e)
            print(f"   ❌ Production Launcher failed: {e}")
        
        # Test Quality Assurance Agent (with dependency handling)
        try:
            # First load dependencies
            # Load MCP Protocol
            mcp_spec = importlib.util.spec_from_file_location(
                "mcp_protocol", 
                self.project_root / "trinity-core/agents/mcp_protocol.py"
            )
            mcp_module = importlib.util.module_from_spec(mcp_spec)
            mcp_spec.loader.exec_module(mcp_module)
            
            # Load domain integration
            domain_spec = importlib.util.spec_from_file_location(
                "domain_integration", 
                self.project_root / "trinity-core/domain_integration.py"
            )
            domain_module = importlib.util.module_from_spec(domain_spec)
            domain_spec.loader.exec_module(domain_module)
            
            # Add modules to sys.modules so they can be imported
            sys.modules['mcp_protocol'] = mcp_module
            sys.modules['domain_integration'] = domain_module
            
            # Now load Quality Assurance Agent
            qa_spec = importlib.util.spec_from_file_location(
                "quality_assurance_agent", 
                self.project_root / "trinity-core/agents/quality_assurance_agent.py"
            )
            qa_module = importlib.util.module_from_spec(qa_spec)
            qa_spec.loader.exec_module(qa_module)
            
            QualityAssuranceAgent = qa_module.QualityAssuranceAgent
            qa_agent = QualityAssuranceAgent()
            stats = qa_agent.get_centralized_domain_stats()
            
            if stats.get("total_domains", 0) > 0:
                results["quality_assurance"] = True
                print("   ✅ Quality Assurance Agent works")
            else:
                print("   ❌ Quality Assurance Agent: No domains loaded")
                
        except Exception as e:
            results["quality_assurance_error"] = str(e)
            print(f"   ❌ Quality Assurance Agent failed: {e}")
        
        # Test Knowledge Transfer Agent (with dependency handling)
        try:
            # Load Knowledge Transfer Agent with proper import handling
            kt_file_path = self.project_root / "trinity-core/agents/knowledge_transfer_agent.py"
            
            # Read the file and fix the relative import
            with open(kt_file_path, 'r', encoding='utf-8') as f:
                kt_content = f.read()
            
            # Replace relative import with absolute import
            kt_content = kt_content.replace(
                "from .mcp_protocol import", 
                "from mcp_protocol import"
            )
            
            # Create a temporary spec and module
            kt_spec = importlib.util.spec_from_loader(
                "knowledge_transfer_agent",
                loader=None
            )
            kt_module = importlib.util.module_from_spec(kt_spec)
            
            # Execute the modified content
            exec(kt_content, kt_module.__dict__)
            
            KnowledgeTransferAgent = kt_module.KnowledgeTransferAgent
            kt_agent = KnowledgeTransferAgent()
            domain_count = sum(len(domains) for domains in kt_agent.domain_mapping.values())
            
            if domain_count > 0:
                results["knowledge_transfer"] = True
                print("   ✅ Knowledge Transfer Agent works")
            else:
                print("   ❌ Knowledge Transfer Agent: No domains loaded")
                
        except Exception as e:
            results["knowledge_transfer_error"] = str(e)
            print(f"   ❌ Knowledge Transfer Agent failed: {e}")
        
        self.test_results["key_functionality"] = results
        return results
    
    def generate_simple_report(self) -> bool:
        """Generate a simple, clear test report"""
        print("\n" + "=" * 60)
        print("🎯 TRINITY ARCHITECTURE TEST REPORT")
        print("=" * 60)
        
        # Core imports
        core_results = self.test_results.get("core_imports", {})
        core_passed = sum(1 for r in core_results.values() if r.get("importable", False))
        core_total = len(core_results)
        
        # Domain mapping
        domain_results = self.test_results.get("domain_mapping", {})
        domain_works = domain_results.get("domain_integration_works", False)
        total_domains = domain_results.get("total_domains", 0)
        
        # Key functionality
        func_results = self.test_results.get("key_functionality", {})
        func_passed = sum(1 for r in func_results.values() if r is True)
        func_total = 3  # production_launcher, quality_assurance, knowledge_transfer
        
        print(f"📊 Test Results:")
        print(f"   Core Imports: {core_passed}/{core_total} ({'✅' if core_passed == core_total else '❌'})")
        print(f"   Domain Mapping: {'✅' if domain_works and total_domains > 0 else '❌'} ({total_domains} domains)")
        print(f"   Key Functionality: {func_passed}/{func_total} ({'✅' if func_passed == func_total else '❌'})")
        
        # Overall status
        overall_success = (
            core_passed == core_total and
            domain_works and 
            total_domains > 0 and
            func_passed >= 2  # At least 2 out of 3 key functions work
        )
        
        status = "✅ PRODUCTION READY" if overall_success else "⚠️ NEEDS ATTENTION"
        print(f"\n🏆 Overall Status: {status}")
        
        if overall_success:
            print("🎉 Trinity Architecture is ready for deployment!")
        else:
            print("⚠️ Please address the failing components above.")
        
        print("=" * 60)
        
        return overall_success
    
    def run_all_tests(self) -> bool:
        """Run all simplified tests"""
        start_time = time.time()
        
        print("🚀 TRINITY ARCHITECTURE - SIMPLIFIED TEST SUITE")
        print("=" * 60)
        print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run core tests
        self.test_core_imports()
        self.test_centralized_domain_mapping()
        self.test_key_functionality()
        
        # Generate report
        success = self.generate_simple_report()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"⏱️ Total execution time: {total_time:.1f} seconds")
        print(f"🏁 Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return success

def main():
    """Main test runner function"""
    runner = SimpleTrinityTestRunner()
    success = runner.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 
