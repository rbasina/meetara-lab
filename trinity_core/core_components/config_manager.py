#!/usr/bin/env python3
"""
Trinity Configuration Manager - SMART YAML-Based Configuration
Loads all configuration from the unified 'trinity_config.yaml'.
"""

import yaml
import logging
import os
from pathlib import Path
from enum import Enum

class UniversalModelArchitecture(Enum):
    """
    Defines the supported universal model architectures.
    The string value should match the key in trinity_config.yaml.
    """
    QWEN_14B = "qwen_14b_universal"
    PHI_3_MINI = "phi_3_mini_universal"
    MISTRAL_7B = "mistral_7b_universal"

class MultiBaseModel(Enum):
    """
    Maps specific domain models to their required base model architecture.
    """
    PHI_3_MINI_INSTRUCT = "microsoft/Phi-3.5-mini-instruct"  # Fixed: Use correct model name
    QWEN2_1_5B_INSTRUCT = "Qwen/Qwen2-1.5B-Instruct"
    MISTRAL_7B_INSTRUCT = "mistralai/Mistral-7B-Instruct-v0.2"
    
    @classmethod
    def from_str(cls, model_string: str):
        for member in cls:
            if member.value == model_string:
                return member
        raise ValueError(f"'{model_string}' is not a valid MultiBaseModel.")

class SmartTrinityConfigManager:
    _instance = None
    _config = None
    _model_tiers = None
    _global_params = None
    _domain_config = None  # Added to store the domain-specific part of the config
    _domain_cache = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SmartTrinityConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_path=None):
        if self._config is None:
            if config_path is None:
                # Default to the new unified config file
                config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'trinity_config.yaml')
            
            if not os.path.exists(config_path):
                logging.error(f"Configuration file not found at {config_path}")
                raise FileNotFoundError(f"Configuration file not found at {config_path}")

            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self._config = yaml.safe_load(f)
                
                # Pre-load and validate key configuration sections
                self._model_tiers = self._config.get('model_tiers')
                if not self._model_tiers:
                    raise ValueError("Config validation failed: 'model_tiers' section is missing or empty.")

                self._global_params = self._config.get('global_tara_params')
                if not self._global_params:
                    raise ValueError("Config validation failed: 'global_tara_params' section is missing or empty.")

                self._domain_config = self._config.get('domain_config')
                if not self._domain_config:
                    raise ValueError("Config validation failed: 'domain_config' section is missing or empty.")

                logging.info(f"Successfully loaded and validated unified configuration from {config_path}")

            except (yaml.YAMLError, ValueError) as e:
                logging.error(f"Error processing configuration file {config_path}: {e}")
                raise

    def get_universal_model_config(self, architecture: UniversalModelArchitecture):
        """
        Retrieves the configuration for a specific universal model architecture.
        """
        if not isinstance(architecture, UniversalModelArchitecture):
            raise TypeError("architecture must be a UniversalModelArchitecture Enum member.")
            
        config = self._config.get(architecture.value)
        if not config:
            raise ValueError(f"Configuration for '{architecture.value}' not found in trinity_config.yaml.")
            
        # Basic validation
        required_keys = ['base_model', 'max_steps', 'sample_count', 'batch_size', 'lora_r', 'learning_rate']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Configuration for '{architecture.value}' is missing required key: '{key}'.")
        
        # Add the architecture name for clarity
        config['architecture'] = architecture.value
        return config

    def get_llama_cpp_path(self) -> Path:
        """
        Retrieves the path to the llama.cpp executable from the config.
        """
        path_str = self._config.get("llama_cpp_path")
        if not path_str:
            raise ValueError("Config validation failed: 'llama_cpp_path' is missing or empty.")
        
        llama_path = Path(path_str)
        
        if not llama_path.exists():
            raise FileNotFoundError(f"The specified llama.cpp path does not exist: {llama_path}")
            
        return llama_path

    def _get_domain_details(self, domain_name):
        """
        Retrieves the configuration for a specific domain by correctly parsing the nested YAML structure.
        """
        if domain_name in self._domain_cache:
            return self._domain_cache[domain_name]

        for category_name, category_config in self._domain_config.items():
            logging.debug(f"ConfigManager: Checking category '{category_name}' for domain '{domain_name}'")
            domains = category_config.get('domains', {})
            if domain_name in domains:
                domain_entry = domains[domain_name]
                
                # Handle both string and dict domain entries
                if isinstance(domain_entry, str):
                    # If domain entry is a string (base model), use it directly
                    base_model = domain_entry
                elif isinstance(domain_entry, dict):
                    # If domain entry is a dict, get base_model from it
                    base_model = domain_entry.get('base_model', category_config.get('base_model'))
                else:
                    # Fallback to category base model
                    base_model = category_config.get('base_model')
                
                if not base_model:
                    raise ValueError(f"Configuration error: No base model found for domain '{domain_name}' in category '{category_name}'.")
                
                logging.debug(f"ConfigManager: Found domain '{domain_name}' in category '{category_name}'. Base model: {base_model}")

                # Determine tier_name from category_config
                tier_name = category_config.get('category_tier')

                if not tier_name:
                    raise ValueError(f"Configuration error: Model tier not specified for category '{category_name}'.")

                # Retrieve generate_synthetic_data with fallback hierarchy
                if isinstance(domain_entry, dict):
                    generate_synthetic = domain_entry.get('generate_synthetic_data', 
                        category_config.get('generate_synthetic_data', 
                            self._global_params.get('generate_synthetic_data', False)))
                else:
                    generate_synthetic = category_config.get('generate_synthetic_data', 
                        self._global_params.get('generate_synthetic_data', False))

                self._domain_cache[domain_name] = {
                    'base_model': base_model,
                    'tier_name': tier_name,
                    'category': category_name,
                    'generate_synthetic_data': generate_synthetic # Add the new flag
                }
                return self._domain_cache[domain_name]

        raise ValueError(f"Domain '{domain_name}' not found in any category in the configuration.")

    def get_model_tier_config(self, tier_name):
        """
        Retrieves the complete configuration for a given model tier.
        This no longer calculates max_steps.
        """
        tier_params = self._config.get('model_tiers', {}).get(tier_name)
        if not tier_params:
            raise ValueError(f"Parameters for model tier '{tier_name}' not found in 'model_tiers'.")

        # Create a mutable copy to work with
        tier_config = tier_params.copy()
            
        required_keys = ['sample_count', 'num_epochs', 'batch_size', 'lora_r', 'learning_rate']
        for key in required_keys:
            if key not in tier_config:
                raise ValueError(f"Configuration error in tier '{tier_name}': missing required key '{key}'.")
        
        return tier_config

    def get_tara_proven_params(self, domain_name):
        """
        Constructs the final training parameters for a domain by combining global,
        tier-specific, and domain-specific settings, including dynamic max_steps calculation.
        """
        domain_details = self._get_domain_details(domain_name)

        if domain_details is None:
            raise ValueError(f"Domain details for '{domain_name}' could not be retrieved from configuration. Received None.")

        try:
            tier_name = domain_details['tier_name']
        except KeyError as e:
            raise ValueError(f"Configuration error: 'tier_name' missing in domain details for '{domain_name}'. Error: {e}")

        tier_config = self.get_model_tier_config(tier_name)

        # Start with global defaults
        params = self._global_params.copy()

        # Layer tier-specific parameters on top
        params.update(tier_config)
        
        # Calculate max_steps dynamically based on tier params, ensuring it's an integer
        try:
            sample_count = int(params['sample_count'])
            batch_size = int(params['batch_size'])
            num_epochs = int(params.get('num_epochs', 1))  # Default to 1 epoch if not specified
            
            if batch_size <= 0:
                raise ValueError("batch_size must be greater than 0.")
                
            # Use ceiling division to ensure all samples are seen
            params['max_steps'] = (sample_count + batch_size - 1) // batch_size * num_epochs
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Could not calculate max_steps for tier '{tier_name}'. Check sample_count, batch_size, and num_epochs. Error: {e}")

        # Add domain-specific and tier name info
        params['base_model'] = domain_details['base_model']
        params['domain'] = domain_name
        params['category'] = domain_details['category']
        params['model_tier'] = tier_name
        
        # Final validation
        required_keys = ['output_format', 'validation_target', 'max_steps', 'lora_r', 'batch_size', 'learning_rate', 'sample_count']
        for key in required_keys:
            if key not in params:
                raise ValueError(f"Failed to construct parameters for domain '{domain_name}'. Final config is missing key: '{key}'.")

        return params

    def get_all_domains_flat(self):
        """
        Returns a flat list of all domain names from all categories.
        """
        all_domains = []
        for category_name, category_config in self._domain_config.items():
            domains = category_config.get('domains', {})
            all_domains.extend(domains.keys())
        return all_domains

    def get_base_model_for_domain(self, domain_name: str) -> str:
        """
        Gets the base model for a specific domain.
        This is the method that the model factory calls.
        """
        domain_details = self._get_domain_details(domain_name)
        return domain_details['base_model']

    def get_tier_config(self, domain_name: str) -> dict:
        """
        Gets the tier configuration for a specific domain.
        """
        domain_details = self._get_domain_details(domain_name)
        tier_name = domain_details['tier_name']
        return self.get_model_tier_config(tier_name)

    def get_config_dict(self):
        """
        Returns a copy of the entire loaded configuration dictionary.
        """
        return self._config.copy() if self._config else {}