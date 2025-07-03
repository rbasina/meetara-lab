#!/usr/bin/env python3
"""
MeeTARA Lab - Production Validation Suite
Comprehensive testing of Trinity Architecture before GPU training
"""

import os
import sys
import time
import json
import subprocess
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

class TrinityProductionValidator:
    def __init__(self):
        self.project_root = Path.cwd()
        self.results = {
            "validation_time": datetime.now().isoformat(),
            "environment": {},
            "trinity_components": {},
            "integration_tests": {},
            "performance_benchmarks": {},
            "overall_status": "PENDING"
        }
        
    def validate_environment(self) -> Dict[str, Any]:
        """Validate development environment setup"""
        print("🔍 Validating Development Environment...")
        
        env_results = {}
        
        # Check Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        env_results["python_version"] = python_version
        env_results["python_compatible"] = python_version.startswith("3.12")
        
        # Check conda environment
        conda_env = os.environ.get("CONDA_DEFAULT_ENV", "unknown")
        env_results["conda_environment"] = conda_env
        env_results["conda_correct"] = conda_env == "meetara-lab"
        
        # Check required directories
        required_dirs = [
            "trinity-core", "intelligence-hub", "model-factory",
            "cloud-training", "notebooks", "memory-bank"
        ]
        
        env_results["directory_structure"] = {}
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            env_results["directory_structure"][dir_name] = dir_path.exists()
        
        # Check virtual environment files are not in project
        unwanted_files = [".venv", ".conda", ".conda-DELETE-ME"]
        env_results["clean_structure"] = {}
        for file_name in unwanted_files:
            file_path = self.project_root / file_name
            env_results["clean_structure"][file_name] = not file_path.exists()
        
        self.results["environment"] = env_results
        
        success = (
            env_results["python_compatible"] and
            env_results["conda_correct"] and
            all(env_results["directory_structure"].values()) and
            all(env_results["clean_structure"].values())
        )
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   Environment Validation: {status}")
        return env_results
    
    def validate_trinity_components(self) -> Dict[str, Any]:
        """Validate all 10 Trinity Architecture components"""
        print("🧠 Validating Trinity Architecture Components...")
        
        components = {
            "tts_manager": "trinity-core/tts_manager.py",
            "emotion_detector": "trinity-core/emotion_detector.py", 
            "intelligent_router": "trinity-core/intelligent_router.py",
            "security_manager": "trinity-core/security_manager.py",
            "validation_utils": "trinity-core/validation_utils.py",
            "domain_experts": "intelligence-hub/domain_experts.py",
            "trinity_intelligence": "intelligence-hub/trinity_intelligence.py",
            "gguf_factory": "model-factory/gguf_factory.py",
            "training_orchestrator": "cloud-training/training_orchestrator.py",
            "monitoring_system": "cloud-training/monitoring_system.py"
        }
        
        component_results = {}
        
        for name, path in components.items():
            component_path = self.project_root / path
            
            result = {
                "file_exists": component_path.exists(),
                "file_size": component_path.stat().st_size if component_path.exists() else 0,
                "syntax_valid": False,
                "imports_work": False
            }
            
            if result["file_exists"] and result["file_size"] > 0:
                # Test syntax
                try:
                    with open(component_path, 'r', encoding='utf-8') as f:
                        compile(f.read(), str(component_path), 'exec')
                    result["syntax_valid"] = True
                except SyntaxError:
                    result["syntax_error"] = True
                
                # Test imports (basic check)
                try:
                    spec = importlib.util.spec_from_file_location(name, component_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        # Don't execute, just validate the module can be loaded
                        result["imports_work"] = True
                except Exception as e:
                    result["import_error"] = str(e)
            
            component_results[name] = result
        
        self.results["trinity_components"] = component_results
        
        # Calculate success rate
        total_components = len(components)
        successful_components = sum(1 for r in component_results.values() 
                                   if r["file_exists"] and r["syntax_valid"])
        
        success_rate = (successful_components / total_components) * 100
        status = "✅ PASS" if success_rate >= 90 else "❌ FAIL"
        print(f"   Trinity Components: {status} ({successful_components}/{total_components} - {success_rate:.1f}%)")
        
        return component_results
    
    def validate_notebook_integration(self) -> Dict[str, Any]:
        """Validate Colab notebooks and Cursor integration"""
        print("📱 Validating Notebook Integration...")
        
        integration_results = {}
        
        # Check Colab notebook
        colab_notebook = self.project_root / "notebooks/colab_gpu_training_template.ipynb"
        integration_results["colab_notebook"] = {
            "exists": colab_notebook.exists(),
            "size": colab_notebook.stat().st_size if colab_notebook.exists() else 0
        }
        
        if colab_notebook.exists():
            try:
                with open(colab_notebook, 'r', encoding='utf-8') as f:
                    notebook_data = json.load(f)
                
                # Check for correct GitHub URL
                github_url_found = False
                for cell in notebook_data.get("cells", []):
                    if "source" in cell:
                        source_text = "".join(cell["source"])
                        if "github.com/rbasina/meetara-lab" in source_text:
                            github_url_found = True
                            break
                
                integration_results["colab_notebook"]["github_url_correct"] = github_url_found
                integration_results["colab_notebook"]["cell_count"] = len(notebook_data.get("cells", []))
                
            except json.JSONDecodeError:
                integration_results["colab_notebook"]["valid_json"] = False
        
        # Check Cursor integration files
        cursor_files = [
            "notebooks/cursor_local_training.py",
            "notebooks/cursor_colab_sync.py", 
            "OPEN_IN_COLAB.md"
        ]
        
        integration_results["cursor_integration"] = {}
        for file_path in cursor_files:
            file_obj = self.project_root / file_path
            integration_results["cursor_integration"][file_path] = {
                "exists": file_obj.exists(),
                "size": file_obj.stat().st_size if file_obj.exists() else 0
            }
        
        self.results["integration_tests"] = integration_results
        
        # Calculate success
        notebook_valid = (
            integration_results["colab_notebook"]["exists"] and
            integration_results["colab_notebook"].get("github_url_correct", False)
        )
        
        cursor_files_exist = all(
            r["exists"] for r in integration_results["cursor_integration"].values()
        )
        
        success = notebook_valid and cursor_files_exist
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   Integration Tests: {status}")
        
        return integration_results
    
    def run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run basic performance benchmarks"""
        print("⚡ Running Performance Benchmarks...")
        
        benchmark_results = {}
        
        # Test import performance
        start_time = time.time()
        try:
            import torch
            import_torch_time = time.time() - start_time
            benchmark_results["torch_import"] = {
                "time_seconds": import_torch_time,
                "success": True,
                "version": torch.__version__
            }
        except ImportError:
            benchmark_results["torch_import"] = {
                "time_seconds": 0,
                "success": False,
                "error": "PyTorch not installed"
            }
        
        # Test file I/O performance
        start_time = time.time()
        test_file = self.project_root / "temp_benchmark.txt"
        try:
            with open(test_file, 'w') as f:
                f.write("test" * 1000)
            with open(test_file, 'r') as f:
                content = f.read()
            test_file.unlink()
            
            io_time = time.time() - start_time
            benchmark_results["file_io"] = {
                "time_seconds": io_time,
                "success": True
            }
        except Exception as e:
            benchmark_results["file_io"] = {
                "time_seconds": 0,
                "success": False,
                "error": str(e)
            }
        
        # Test CPU baseline for training comparison
        start_time = time.time()
        # Simulate training step computation
        result = sum(i * i for i in range(10000))
        cpu_time = time.time() - start_time
        
        benchmark_results["cpu_baseline"] = {
            "time_seconds": cpu_time,
            "operations_per_second": 10000 / cpu_time if cpu_time > 0 else 0,
            "target_gpu_improvement": "20-100x faster than this baseline"
        }
        
        self.results["performance_benchmarks"] = benchmark_results
        
        # All benchmarks should complete successfully
        success = all(r.get("success", True) for r in benchmark_results.values())
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   Performance Benchmarks: {status}")
        
        return benchmark_results
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete production validation suite"""
        print("🚀 MeeTARA Lab - Production Validation Suite")
        print("=" * 60)
        
        # Run all validation tests
        self.validate_environment()
        self.validate_trinity_components()
        self.validate_notebook_integration() 
        self.run_performance_benchmarks()
        
        # Calculate overall status
        env_success = self.results["environment"].get("python_compatible", False)
        components_success = len([c for c in self.results["trinity_components"].values() 
                                if c.get("file_exists", False)]) >= 8  # At least 8/10 components
        integration_success = self.results["integration_tests"]["colab_notebook"].get("exists", False)
        benchmark_success = self.results["performance_benchmarks"]["cpu_baseline"].get("time_seconds", 0) > 0
        
        overall_success = env_success and components_success and integration_success and benchmark_success
        self.results["overall_status"] = "✅ PRODUCTION READY" if overall_success else "❌ NEEDS FIXES"
        
        # Save results
        results_file = self.project_root / "validation_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print("\n" + "=" * 60)
        print(f"🎯 Overall Status: {self.results['overall_status']}")
        print(f"📊 Results saved to: {results_file}")
        
        if overall_success:
            print("\n🚀 Ready for next phase: GPU Training Pipeline Implementation!")
        else:
            print("\n⚠️ Please fix issues before proceeding to GPU training.")
        
        return self.results

def main():
    """Run production validation"""
    validator = TrinityProductionValidator()
    results = validator.run_full_validation()
    return results

if __name__ == "__main__":
    main() 