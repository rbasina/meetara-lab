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
from typing import Dict

class UniversalModelArchitecture(Enum):
    """
    Defines the supported universal model architectures.
    The string value should match the key in trinity_config.yaml.
    """
    QWEN_7B = "qwen_7b_universal"
    QWEN_14B = "qwen_14b_universal"
    MISTRAL_7B = "mistral_7b_universal"

class MultiBaseModel(Enum):
    """
    Maps specific domain models to their required base model architecture.
    Model names are read from configuration file, not hardcoded.
    """
    QWEN2_7B_INSTRUCT = "qwen2_7b_instruct"
    QWEN2_14B_INSTRUCT = "qwen2_14b_instruct"
    MISTRAL_7B_INSTRUCT = "mistral_7b_instruct"
    
    @classmethod
    def from_str(cls, model_string: str):
        for member in cls:
            if member.value == model_string:
                return member
        raise ValueError(f"'{model_string}' is not a valid MultiBaseModel.")
    
    def get_model_name(self, config_manager):
        """Get the actual model name from configuration file."""
        model_mapping = config_manager._config.get('model_names', {})
        return model_mapping.get(self.value, self.value)

def parse_domain_model_entry(model_entry: str):
    """
    Parses a model entry from config.
    Returns (primary_model, fallback_model) tuple.
    If only one model is present, fallback_model is None.
    """
    if ',' in model_entry:
        models = [m.strip() for m in model_entry.split(',')]
        return models[0], models[1] if len(models) > 1 else None
    elif '|' in model_entry:
        models = [m.strip() for m in model_entry.split('|')]
        return models[0], models[1] if len(models) > 1 else None
    else:
        return model_entry.strip(), None

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

    def get_config(self):
        """
        Fallback method for backward compatibility.
        Returns the same as get_config_dict().
        """
        return self.get_config_dict()

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

        # Ensure _domain_config is properly loaded
        if not self._domain_config:
            raise ValueError("Domain configuration not loaded. Please ensure config is properly initialized.")

        # Debug logging to understand the structure
        logging.debug(f"ConfigManager: Searching for domain '{domain_name}'")
        logging.debug(f"ConfigManager: Available categories: {list(self._domain_config.keys())}")

        for category_name, category_config in self._domain_config.items():
            logging.debug(f"ConfigManager: Checking category '{category_name}' for domain '{domain_name}'")
            
            # Ensure category_config is a dictionary
            if not isinstance(category_config, dict):
                logging.warning(f"ConfigManager: Category '{category_name}' is not a dictionary: {type(category_config)}")
                continue
                
            domains = category_config.get('domains', {})
            
            # Ensure domains is a dictionary
            if not isinstance(domains, dict):
                logging.warning(f"ConfigManager: Domains in category '{category_name}' is not a dictionary: {type(domains)}")
                continue
                
            if domain_name in domains:
                domain_entry = domains[domain_name]
                logging.debug(f"ConfigManager: Found domain '{domain_name}' in category '{category_name}'")
                logging.debug(f"ConfigManager: Domain entry type: {type(domain_entry)}, value: {domain_entry}")
                
                # Handle both string and dict domain entries
                if isinstance(domain_entry, str):
                    # If domain entry is a string (base model), use it directly
                    base_model_entry = domain_entry
                    primary_model, fallback_model = parse_domain_model_entry(base_model_entry)
                    base_model = primary_model  # Use primary for backward compatibility
                    logging.debug(f"ConfigManager: Using string base model: {base_model}")
                    if fallback_model:
                        logging.debug(f"ConfigManager: Fallback model available: {fallback_model}")
                elif isinstance(domain_entry, dict):
                    # If domain entry is a dict, get base_model from it
                    base_model_entry = domain_entry.get('base_model', category_config.get('base_model'))
                    primary_model, fallback_model = parse_domain_model_entry(base_model_entry)
                    base_model = primary_model  # Use primary for backward compatibility
                    logging.debug(f"ConfigManager: Using dict base model: {base_model}")
                    if fallback_model:
                        logging.debug(f"ConfigManager: Fallback model available: {fallback_model}")
                else:
                    # Fallback to category base model
                    base_model_entry = category_config.get('base_model')
                    primary_model, fallback_model = parse_domain_model_entry(base_model_entry)
                    base_model = primary_model  # Use primary for backward compatibility
                    logging.debug(f"ConfigManager: Using category fallback base model: {base_model}")
                    if fallback_model:
                        logging.debug(f"ConfigManager: Fallback model available: {fallback_model}")
                
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

                # Add both models to the returned dict for downstream use
                domain_details = {
                    'base_model': base_model,
                    'primary_model': primary_model,
                    'fallback_model': fallback_model,
                    'category': category_name,
                    'tier_name': tier_name
                }
                
                # Cache the result
                self._domain_cache[domain_name] = domain_details
                logging.debug(f"ConfigManager: Cached domain details for '{domain_name}': {domain_details}")
                return domain_details

        # If we get here, domain was not found
        available_domains = []
        for category_name, category_config in self._domain_config.items():
            if isinstance(category_config, dict):
                domains = category_config.get('domains', {})
                if isinstance(domains, dict):
                    available_domains.extend(domains.keys())
        
        raise ValueError(f"Domain '{domain_name}' not found in any category in the configuration. Available domains: {available_domains}")

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

    def get_models_for_domain(self, domain_name: str) -> Dict[str, str]:
        """
        Gets both primary and fallback models for a specific domain.
        Returns a dictionary with 'primary_model' and 'fallback_model' keys.
        """
        domain_details = self._get_domain_details(domain_name)
        return {
            'primary_model': domain_details['primary_model'],
            'fallback_model': domain_details['fallback_model']
        }

    def get_base_model_for_domain(self, domain_name: str, check_memory: bool = True) -> str:
        """
        Gets the base model for a specific domain.
        Always returns the primary model for training, regardless of memory.
        Fails with an error if the primary model cannot be loaded due to memory constraints.
        Only fallback to the secondary model if explicitly configured for that domain.
        """
        domain_details = self._get_domain_details(domain_name)
        primary_model = domain_details['primary_model']
        # Fallback logic is not used for training; always return primary_model
        return primary_model

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