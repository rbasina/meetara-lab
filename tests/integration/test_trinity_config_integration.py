#!/usr/bin/env python3
"""
Trinity Architecture - Unified Configuration Integration Test
Verifies that the SmartTrinityConfigManager and its consumers
correctly load and interpret the unified 'trinity_config.yaml'.
"""

import os
import sys
import unittest
from unittest.mock import patch, mock_open
from pathlib import Path

# Adjust the path to import the config manager
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trinity_core.core_components.config_manager import SmartTrinityConfigManager

class TestUnifiedTrinityConfigIntegration(unittest.TestCase):
    """
    This test suite validates the integration of the SmartTrinityConfigManager
    with the new, unified 'trinity_config.yaml'.
    """

    def setUp(self):
        """Set up a mock YAML config for testing."""
        self.mock_yaml_content = """
global_tara_params:
  output_format: Q4_K_M
  validation_target: 101.0

model_tiers:
  premium:
    sample_count: 1000
    num_epochs: 2
    batch_size: 4
    lora_r: 16
    learning_rate: 0.0001
  developer:
    sample_count: 100
    num_epochs: 1
    batch_size: 2
    lora_r: 8
    learning_rate: 0.0002

domain_config:
  healthcare:
    base_model: "phi-3-large"
    category_tier: "premium"
    domains:
      symptom_checker:
      medical_translator:
        category_tier: "premium" # Override
  business:
    base_model: "gpt-4-mini"
    category_tier: "developer"
    domains:
      financial_advisor:
      market_analyst:
        base_model: "llama-3-expert" # Override
        category_tier: "premium"
"""
        # Patch 'open' to return our mock YAML content
        self.mock_open = mock_open(read_data=self.mock_yaml_content)
        patcher = patch('builtins.open', self.mock_open)
        patcher.start()
        self.addCleanup(patcher.stop)

        # Patch os.path.exists to always return True for the config file
        self.path_exists_patcher = patch('os.path.exists', return_value=True)
        self.path_exists_patcher.start()
        self.addCleanup(self.path_exists_patcher.stop)
        
        # We need to reset the singleton instance to ensure our mock is used
        if SmartTrinityConfigManager._instance:
            SmartTrinityConfigManager._instance = None
            SmartTrinityConfigManager._config = None

        self.config_manager = SmartTrinityConfigManager()

    def test_singleton_instance(self):
        """Test that SmartTrinityConfigManager is a singleton."""
        instance1 = SmartTrinityConfigManager()
        instance2 = SmartTrinityConfigManager()
        self.assertIs(instance1, instance2)

    def test_config_loading_success(self):
        """Test that the configuration is loaded successfully."""
        self.assertIsNotNone(self.config_manager._config)
        self.assertIn('model_tiers', self.config_manager._config)

    def test_get_all_domains_flat(self):
        """Test retrieval of a flat list of all domains."""
        domains = self.config_manager.get_all_domains_flat()
        self.assertEqual(len(domains), 4)
        self.assertIn('symptom_checker', domains)
        self.assertIn('market_analyst', domains)

    def test_get_base_model_for_domain(self):
        """Tests that the correct base model is returned for a specific domain."""
        base_model = self.config_manager.get_base_model_for_domain("shopping")
        self.assertIsNotNone(base_model)
        # This assertion depends on the test config
        self.assertEqual(base_model, "HuggingFaceTB/SmolLM2-1.7B") 

    def test_get_all_categories_and_domains(self):
        """Tests retrieving all domains and implicitly tests category presence."""
        all_domains = self.config_manager.get_all_domains_flat()
        self.assertTrue(len(all_domains) > 50) # Should be 62 in the test config

        # Check if categories are present in the details of the domains
        all_categories = {details.get('category') for details in all_domains.values() if details.get('category')}
        self.assertTrue(len(all_categories) > 5) # Should be 7 in the test config
        
        # Verify a known category is present
        self.assertIn("daily_life", all_categories)
        self.assertIn("healthcare", all_categories)

    def test_get_tara_proven_params_simple(self):
        """Test getting parameters for a domain with inherited settings."""
        params = self.config_manager.get_tara_proven_params('financial_advisor')
        self.assertEqual(params['base_model'], 'gpt-4-mini')
        self.assertEqual(params['lora_r'], 8)
        self.assertEqual(params['max_steps'], 50) # (100 samples / 2 batch_size) * 1 epoch

    def test_get_tara_proven_params_override(self):
        """Test getting parameters for a domain with overridden settings."""
        params = self.config_manager.get_tara_proven_params('market_analyst')
        self.assertEqual(params['base_model'], 'llama-3-expert')
        self.assertEqual(params['lora_r'], 16) # From premium tier
        self.assertEqual(params['max_steps'], 500) # (1000 samples / 4 batch_size) * 2 epochs

    def test_get_tara_proven_params_invalid_domain(self):
        """Test that getting parameters for a non-existent domain raises an error."""
        with self.assertRaises(ValueError):
            self.config_manager.get_tara_proven_params('non_existent_domain')

    def test_config_validation_failure_missing_section(self):
        """Test that config validation fails if a critical section is missing."""
        # Reset singleton to force re-initialization
        SmartTrinityConfigManager._instance = None
        SmartTrinityConfigManager._config = None
        
        # Provide broken config data
        broken_yaml_content = "global_tara_params: {}"
        with patch('builtins.open', mock_open(read_data=broken_yaml_content)):
             with self.assertRaises(ValueError):
                SmartTrinityConfigManager()

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False) 