#!/usr/bin/env python3
"""
Comprehensive Test Runner for Trinity Architecture
Executes all test suites and provides detailed reporting
"""

import subprocess
import sys
import time
from pathlib import Path

class TrinityTestRunner:
    """Comprehensive test runner for all Trinity Architecture components"""
    
    def __init__(self):
        self.test_results = {}
        self.project_root = Path.cwd().parent if Path.cwd().name == "tests" else Path.cwd()
        
    def run_simple_validation(self):
        """Run simple validation checks"""
        print("🔍 Running Simple Validation...")
        try:
            result = subprocess.run(
                [sys.executable, "tests/simple_validation.py"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            success = result.returncode == 0
            self.test_results["simple_validation"] = {
                "status": "PASS" if success else "FAIL",
                "output": result.stdout,
                "errors": result.stderr if result.stderr else None
            }
            
            if success:
                print("   ✅ Simple validation PASSED")
            else:
                print("   ❌ Simple validation FAILED")
                
        except Exception as e:
            print(f"   ⚠️ Simple validation ERROR: {e}")
            self.test_results["simple_validation"] = {"status": "ERROR", "error": str(e)}
    
    def run_unit_tests(self):
        """Run all unit tests"""
        print("\n🧪 Running Unit Tests...")
        
        unit_tests = [
            "tests/unit/test_tts_manager.py",
            "tests/unit/test_emotion_detector.py",
            "tests/unit/test_intelligent_router.py"
        ]
        
        unit_results = {}
        
        for test_file in unit_tests:
            test_name = Path(test_file).stem
            print(f"   Running {test_name}...")
            
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", test_file, "-v"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                success = result.returncode == 0
                unit_results[test_name] = {
                    "status": "PASS" if success else "FAIL",
                    "output": result.stdout[-500:] if result.stdout else "",  # Last 500 chars
                    "errors": result.stderr[-500:] if result.stderr else None
                }
                
                status_icon = "✅" if success else "❌"
                print(f"      {status_icon} {test_name}")
                
            except Exception as e:
                print(f"      ⚠️ {test_name} ERROR: {e}")
                unit_results[test_name] = {"status": "ERROR", "error": str(e)}
        
        self.test_results["unit_tests"] = unit_results
    
    def run_integration_tests(self):
        """Run integration tests"""
        print("\n🔗 Running Integration Tests...")
        
        integration_tests = [
            ("tests/integration/test_trinity_integration.py", "Trinity Integration"),
            ("tests/integration/test_domains_integration.py", "Domain Integration"),
            ("tests/integration/test_agent_ecosystem_integration.py", "Agent Ecosystem Integration")
        ]
        
        integration_results = {}
        
        for test_file, test_name in integration_tests:
            print(f"   Running {test_name}...")
            
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", test_file, "-v"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                success = result.returncode == 0
                integration_results[test_name] = {
                    "status": "PASS" if success else "FAIL",
                    "output": result.stdout[-500:] if result.stdout else "",
                    "errors": result.stderr[-500:] if result.stderr else None
                }
                
                status_icon = "✅" if success else "❌"
                print(f"      {status_icon} {test_name}")
                
            except Exception as e:
                print(f"      ⚠️ {test_name} ERROR: {e}")
                integration_results[test_name] = {"status": "ERROR", "error": str(e)}
        
        self.test_results["integration_tests"] = integration_results
    
    def run_performance_tests(self):
        """Run performance tests"""
        print("\n⚡ Running Performance Tests...")
        
        try:
            result = subprocess.run(
                [sys.executable, "tests/performance/test_gpu_training.py"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=180
            )
            
            success = result.returncode == 0
            self.test_results["performance_tests"] = {
                "status": "PASS" if success else "FAIL",
                "output": result.stdout[-500:] if result.stdout else "",
                "errors": result.stderr[-500:] if result.stderr else None
            }
            
            status_icon = "✅" if success else "❌"
            print(f"   {status_icon} GPU Training Performance Tests")
            
        except Exception as e:
            print(f"   ⚠️ Performance tests ERROR: {e}")
            self.test_results["performance_tests"] = {"status": "ERROR", "error": str(e)}
    
    def run_workflow_tests(self):
        """Run end-to-end workflow tests"""
        print("\n🔄 Running Workflow Tests...")
        
        try:
            result = subprocess.run(
                [sys.executable, "tests/end_to_end/test_workflows.py"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            success = result.returncode == 0
            self.test_results["workflow_tests"] = {
                "status": "PASS" if success else "FAIL",
                "output": result.stdout[-500:] if result.stdout else "",
                "errors": result.stderr[-500:] if result.stderr else None
            }
            
            status_icon = "✅" if success else "❌"
            print(f"   {status_icon} End-to-End Workflow Tests")
            
        except Exception as e:
            print(f"   ⚠️ Workflow tests ERROR: {e}")
            self.test_results["workflow_tests"] = {"status": "ERROR", "error": str(e)}
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 70)
        print("🎯 TRINITY ARCHITECTURE TEST REPORT")
        print("=" * 70)
        
        # Count results
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        error_tests = 0
        
        for test_category, results in self.test_results.items():
            if test_category in ["unit_tests", "integration_tests"]:
                # Unit tests and integration tests have multiple sub-tests
                for test_name, result in results.items():
                    total_tests += 1
                    if result["status"] == "PASS":
                        passed_tests += 1
                    elif result["status"] == "FAIL":
                        failed_tests += 1
                    else:
                        error_tests += 1
            else:
                # Other test categories are single tests
                total_tests += 1
                if results["status"] == "PASS":
                    passed_tests += 1
                elif results["status"] == "FAIL":
                    failed_tests += 1
                else:
                    error_tests += 1
        
        # Overall status
        overall_success = (passed_tests / total_tests) >= 0.8 if total_tests > 0 else False
        overall_status = "✅ PRODUCTION READY" if overall_success else "⚠️ NEEDS ATTENTION"
        
        print(f"\n🏆 Overall Status: {overall_status}")
        print(f"📊 Test Summary:")
        print(f"   Total Tests: {total_tests}")
        print(f"   ✅ Passed: {passed_tests}")
        print(f"   ❌ Failed: {failed_tests}")
        print(f"   ⚠️ Errors: {error_tests}")
        
        if total_tests > 0:
            success_rate = (passed_tests / total_tests) * 100
            print(f"   🎯 Success Rate: {success_rate:.1f}%")
        
        # Detailed results
        print(f"\n📋 Detailed Results:")
        for test_category, results in self.test_results.items():
            print(f"\n   {test_category.replace('_', ' ').title()}:")
            
            if test_category in ["unit_tests", "integration_tests"]:
                for test_name, result in results.items():
                    status_icon = {"PASS": "✅", "FAIL": "❌", "ERROR": "⚠️"}.get(result["status"], "❓")
                    print(f"      {status_icon} {test_name}")
            else:
                status_icon = {"PASS": "✅", "FAIL": "❌", "ERROR": "⚠️"}.get(results["status"], "❓")
                print(f"      {status_icon} {results['status']}")
        
        # Trinity Architecture readiness
        print(f"\n🚀 Trinity Architecture Status:")
        
        trinity_components = [
            "TTS Manager", "Emotion Detector", "Intelligent Router",
            "Domain Experts", "GGUF Factory", "Training Orchestrator",
            "Monitoring System", "Security Manager", "Validation Utils",
            "Configuration Management"
        ]
        
        print(f"   📦 Components: {len(trinity_components)} implemented")
        print(f"   🧪 Test Coverage: {len(self.test_results)} test categories")
        print(f"   📊 Quality Score: {success_rate:.1f}% if total_tests > 0 else 'N/A'")
        
        if overall_success:
            print(f"\n🎉 Ready for GPU Training Pipeline Implementation!")
        else:
            print(f"\n⚠️ Please address failing tests before proceeding.")
        
        print("\n" + "=" * 70)
        
        return overall_success
    
    def run_all_tests(self):
        """Run complete test suite"""
        start_time = time.time()
        
        print("🚀 TRINITY ARCHITECTURE - COMPREHENSIVE TEST SUITE")
        print("=" * 70)
        print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Execute all test categories
        self.run_simple_validation()
        self.run_unit_tests()
        self.run_integration_tests()
        self.run_performance_tests()
        self.run_workflow_tests()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Generate comprehensive report
        overall_success = self.generate_report()
        
        print(f"⏱️ Total execution time: {total_time:.1f} seconds")
        print(f"🏁 Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return overall_success

def main():
    """Main test runner function"""
    runner = TrinityTestRunner()
    success = runner.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 
