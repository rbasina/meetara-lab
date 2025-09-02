#!/usr/bin/env python3
"""
Test script to verify AI services configuration in MeeTARA Lab.
This script tests the AI service initialization without making actual API calls.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
    print("✅ .env file loaded in test script")
except ImportError:
    print("⚠️  python-dotenv not available")
except Exception as e:
    print(f"⚠️  Error loading .env: {e}")

def test_environment_variables():
    """Test if environment variables are properly set."""
    print("🔍 Testing Environment Variables...")
    
    required_vars = ["OPENAI_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mask the API key for security
            masked_value = f"{value[:10]}...{value[-4:]}" if len(value) > 14 else "***"
            print(f"  ✅ {var}: {masked_value}")
        else:
            print(f"  ❌ {var}: Not set")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("   Please set these in your .env file or system environment")
        return False
    
    print("  ✅ All required environment variables are set")
    return True

def test_dotenv_loading():
    """Test if python-dotenv is working."""
    print("\n🔍 Testing .env File Loading...")
    
    try:
        from dotenv import load_dotenv
        print("  ✅ python-dotenv is installed")
        
        # Check if .env file exists
        env_file = project_root / ".env"
        if env_file.exists():
            print("  ✅ .env file found")
            
            # Load and check if variables are accessible
            load_dotenv()
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                print("  ✅ .env file loaded successfully")
                return True
            else:
                print("  ❌ .env file loaded but variables not accessible")
                return False
        else:
            print("  ⚠️  .env file not found - using system environment variables")
            return True
            
    except ImportError:
        print("  ⚠️  python-dotenv not installed - using system environment variables")
        return True

def test_configuration_files():
    """Test if configuration files are properly set up."""
    print("\n🔍 Testing Configuration Files...")
    
    # Check AI services config
    ai_config_file = project_root / "config" / "ai_services_config.yaml"
    if ai_config_file.exists():
        print("  ✅ AI services configuration file found")
        
        # Check that it doesn't contain API keys
        with open(ai_config_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if "api_key" in content.lower():
                print("  ⚠️  Configuration file contains API key references")
                return False
            else:
                print("  ✅ Configuration file is clean (no API keys)")
                return True
    else:
        print("  ❌ AI services configuration file not found")
        return False

def test_imports():
    """Test if required modules can be imported."""
    print("\n🔍 Testing Module Imports...")
    
    try:
        from trinity_core.agents.data_generator import TrinityDataGenerator
        print("  ✅ TrinityDataGenerator imported successfully")
        return True
    except ImportError as e:
        print(f"  ❌ Failed to import TrinityDataGenerator: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 MeeTARA Lab AI Services Setup Test")
    print("=" * 50)
    
    tests = [
        test_environment_variables,
        test_dotenv_loading,
        test_configuration_files,
        test_imports
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ❌ Test failed with error: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! AI services are properly configured.")
        print("\nNext steps:")
        print("1. Run your MeeTARA Lab training pipeline")
        print("2. Check logs for AI service initialization messages")
        print("3. Monitor API usage and costs")
    else:
        print("⚠️  Some tests failed. Please check the configuration.")
        print("\nTroubleshooting:")
        print("1. Ensure your .env file contains valid API keys")
        print("2. Check that python-dotenv is installed")
        print("3. Verify configuration files are in the correct locations")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
