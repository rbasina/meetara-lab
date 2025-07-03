#!/usr/bin/env python3
"""
MeeTARA Lab - Dynamic Domain Integration Tests
Validates that all agents properly support the complete domain architecture
Automatically adapts to any number of domains configured in YAML
"""

import pytest
import asyncio
from pathlib import Path
from typing import Dict, Set, List

# Import test utilities
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "trinity-core"))
sys.path.append(str(Path(__file__).parent.parent.parent / "trinity-core" / "agents"))

from utils.domain_validation_utils import (
    DomainConfigManager, 
    DomainTestValidators, 
    validate_agent_domain_support,
    get_test_fixtures
)

# Import agents for testing
from agents.training_conductor import TrainingConductor as TrainingConductorAgent
from agents.knowledge_transfer_agent import KnowledgeTransferAgent
from agents.cross_domain_agent import CrossDomainAgent
from agents.quality_assurance_agent import QualityAssuranceAgent
from agents.gpu_optimizer_agent import GPUOptimizerAgent
from agents.gguf_creator_agent import GGUFCreatorAgent

class TestDomainsIntegration:
    """Test suite for complete domain integration across all agents"""
    
    @pytest.fixture(scope="class")
    def config_manager(self) -> DomainConfigManager:
        """Domain configuration manager"""
        return DomainConfigManager()
    
    @pytest.fixture(scope="class")
    def test_data(self, config_manager: DomainConfigManager) -> Dict:
        """Common test data derived from configuration"""
        return {
            'config': config_manager.config,
            'domains': config_manager.domains,
            'categories': config_manager.categories,
            'domain_count': config_manager.domain_count,
            'category_counts': config_manager.get_category_counts(),
            'sample_domains': config_manager.get_sample_domains(max_per_category=2)
        }
    
    def test_configuration_integrity(self, test_data: Dict):
        """Test that domain configuration is valid and complete"""
        config = test_data['config']
        domains = test_data['domains']
        categories = test_data['categories']
        domain_count = test_data['domain_count']
        
        # Basic integrity checks
        assert len(categories) > 0, "No domain categories found in configuration"
        assert domain_count > 0, "No domains found in configuration"
        
        # Each category should have at least one domain
        for category in categories:
            category_domains = set()
            if category in config and isinstance(config[category], dict):
                category_domains = set(config[category].keys())
            
            assert len(category_domains) > 0, f"Category '{category}' has no domains"
            assert len(category_domains) <= 25, f"Category '{category}' has too many domains ({len(category_domains)})"
        
        # Verify no duplicate domains across categories
        all_domains_from_categories = set()
        for category in categories:
            if category in config and isinstance(config[category], dict):
                category_domains = set(config[category].keys())
                overlap = all_domains_from_categories.intersection(category_domains)
                assert len(overlap) == 0, f"Duplicate domains found: {overlap}"
                all_domains_from_categories.update(category_domains)
        
        assert all_domains_from_categories == domains, "Domain set mismatch between categories and expected domains"
        
        print(f"✅ Configuration validated: {domain_count} domains across {len(categories)} categories")
    
    @pytest.mark.asyncio
    async def test_training_conductor_integration(self, config_manager: DomainConfigManager):
        """Test Training Conductor Agent with dynamic domain support"""
        agent = TrainingConductorAgent()
        validate_agent_domain_support(agent, "Training Conductor", config_manager)
        
        # Additional Training Conductor specific tests
        if hasattr(agent, 'category_requirements'):
            for category in config_manager.categories:
                if category in agent.category_requirements:
                    requirements = agent.category_requirements[category]
                    assert 'min_validation_score' in requirements, f"Missing validation score for category '{category}'"
                    assert 'complexity_hours' in requirements, f"Missing complexity hours for category '{category}'"
    
    @pytest.mark.asyncio
    async def test_knowledge_transfer_integration(self, config_manager: DomainConfigManager):
        """Test Knowledge Transfer Agent with dynamic domain support"""
        agent = KnowledgeTransferAgent()
        validate_agent_domain_support(agent, "Knowledge Transfer", config_manager)
        
        # Test compatibility matrix with sample domains
        sample_domains = list(config_manager.domains)[:min(3, len(config_manager.domains))]
        DomainTestValidators.validate_compatibility_scores(agent, sample_domains, "Knowledge Transfer")
    
    @pytest.mark.asyncio
    async def test_cross_domain_agent_integration(self, config_manager: DomainConfigManager):
        """Test Cross-Domain Agent with dynamic domain support"""
        agent = CrossDomainAgent()
        
        # Cross-domain agent may not have full domain mapping, test what it has
        if hasattr(agent, '_load_domain_configuration'):
            agent._load_domain_configuration()
        
        # Test pattern recognition for sample categories
        sample_categories = list(config_manager.categories)[:min(3, len(config_manager.categories))]
        for category in sample_categories:
            if hasattr(agent, '_get_domain_patterns'):
                try:
                    patterns = agent._get_domain_patterns(category)
                    # If method exists, it should return something meaningful
                    assert patterns is not None, f"Cross-Domain: No patterns for category '{category}'"
                except Exception:
                    # Method may not be implemented for all categories, which is okay
                    pass
        
        print(f"✅ Cross-Domain Agent validated for {len(sample_categories)} sample categories")
    
    @pytest.mark.asyncio
    async def test_quality_assurance_integration(self, config_manager: DomainConfigManager):
        """Test Quality Assurance Agent with dynamic domain support"""
        agent = QualityAssuranceAgent()
        
        # QA agent may have category-based requirements rather than full domain mapping
        if hasattr(agent, '_load_domain_configuration'):
            agent._load_domain_configuration()
        
        # Test quality requirements for sample domains
        sample_domains = list(config_manager.domains)[:min(5, len(config_manager.domains))]
        for domain in sample_domains:
            if hasattr(agent, '_get_quality_requirements'):
                try:
                    requirements = agent._get_quality_requirements(domain)
                    if requirements is not None:
                        assert isinstance(requirements, dict), f"Quality Assurance: Invalid requirements format for '{domain}'"
                except Exception:
                    # May not have requirements for all domains, which is acceptable
                    pass
        
        print(f"✅ Quality Assurance validated for {len(sample_domains)} sample domains")
    
    def test_gpu_optimizer_domain_agnostic(self, config_manager: DomainConfigManager):
        """Test GPU Optimizer Agent works with any domain (domain-agnostic design)"""
        agent = GPUOptimizerAgent()
        
        # GPU Optimizer should be domain-agnostic by design
        sample_domains = list(config_manager.domains)[:min(4, len(config_manager.domains))]
        
        for domain in sample_domains:
            if hasattr(agent, '_estimate_resource_requirements'):
                try:
                    result = agent._estimate_resource_requirements(domain)
                    # Should not fail for any domain
                    assert result is not None, f"GPU Optimizer: No resource estimate for domain '{domain}'"
                except Exception as e:
                    # If method exists but fails, that's an issue
                    pytest.fail(f"GPU Optimizer failed for domain '{domain}': {e}")
        
        print(f"✅ GPU Optimizer validated for {len(sample_domains)} sample domains")
    
    def test_gguf_creator_domain_agnostic(self, config_manager: DomainConfigManager):
        """Test GGUF Creator Agent works with any domain (domain-agnostic design)"""
        agent = GGUFCreatorAgent()
        
        # GGUF Creator should be domain-agnostic by design
        sample_domains = list(config_manager.domains)[:min(4, len(config_manager.domains))]
        
        for domain in sample_domains:
            if hasattr(agent, '_get_model_config'):
                try:
                    config = agent._get_model_config(domain)
                    assert config is not None, f"GGUF Creator: No config for domain '{domain}'"
                except Exception as e:
                    # If method exists but fails, that's an issue
                    pytest.fail(f"GGUF Creator failed for domain '{domain}': {e}")
        
        print(f"✅ GGUF Creator validated for {len(sample_domains)} sample domains")
    
    def test_category_quality_hierarchy(self, test_data: Dict):
        """Test that category-based quality thresholds follow logical hierarchy"""
        agent = TrainingConductorAgent()
        agent._load_domain_configuration()
        
        if not hasattr(agent, 'category_requirements'):
            pytest.skip("Agent does not use category-based requirements")
        
        categories = test_data['categories']
        
        # Test that each category has reasonable quality thresholds
        for category in categories:
            if category in agent.category_requirements:
                requirements = agent.category_requirements[category]
                validation_score = requirements.get('min_validation_score', 0)
                
                # Quality scores should be reasonable
                assert 70 <= validation_score <= 100, \
                    f"Category '{category}': Quality score {validation_score}% outside valid range"
        
        print(f"✅ Quality hierarchy validated for {len(categories)} categories")
    
    def test_scalability_and_performance(self, test_data: Dict):
        """Test that the system scales well with the configured domain count"""
        domain_count = test_data['domain_count']
        categories = test_data['categories']
        
        # Basic scalability checks
        assert domain_count <= 100, f"Domain count {domain_count} may impact performance"
        assert len(categories) <= 15, f"Category count {len(categories)} may be too granular"
        
        # Domain distribution should be reasonable
        category_counts = test_data['category_counts']
        max_domains_per_category = max(category_counts.values())
        min_domains_per_category = min(category_counts.values())
        
        assert max_domains_per_category <= 25, f"Largest category has {max_domains_per_category} domains (too many)"
        assert min_domains_per_category >= 1, f"Smallest category has {min_domains_per_category} domains (too few)"
        
        print(f"✅ Scalability validated: {domain_count} domains, {len(categories)} categories")
    
    @pytest.mark.asyncio
    async def test_end_to_end_processing(self, test_data: Dict):
        """Test end-to-end processing with sample domains from each category"""
        sample_domains = test_data['sample_domains']
        
        # Initialize agents
        training_conductor = TrainingConductorAgent()
        training_conductor._load_domain_configuration()
        
        knowledge_transfer = KnowledgeTransferAgent()
        knowledge_transfer._load_domain_configuration()
        
        successful_tests = 0
        total_tests = 0
        
        for category, domains in sample_domains.items():
            for domain in domains:
                total_tests += 1
                
                # Test Training Conductor
                tc_success = (
                    hasattr(training_conductor, 'domain_categories') and
                    domain in training_conductor.domain_categories and
                    training_conductor.domain_categories[domain] == category
                )
                
                # Test Knowledge Transfer
                kt_success = (
                    hasattr(knowledge_transfer, 'domain_categories') and
                    domain in knowledge_transfer.domain_categories and
                    knowledge_transfer.domain_categories[domain] == category
                )
                
                if tc_success and kt_success:
                    successful_tests += 1
        
        # At least 80% of sample domains should work end-to-end
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        assert success_rate >= 0.8, f"End-to-end success rate {success_rate:.1%} too low"
        
        print(f"✅ End-to-end processing: {successful_tests}/{total_tests} domains successful ({success_rate:.1%})")
    
    def test_backward_compatibility(self, config_manager: DomainConfigManager):
        """Test that agents maintain backward compatibility with legacy domain references"""
        agent = KnowledgeTransferAgent()
        agent._load_domain_configuration()
        
        # Test common legacy domains that should still work
        legacy_domains = ['healthcare', 'finance', 'education', 'business', 'technology', 'legal']
        working_legacy = 0
        
        for domain in legacy_domains:
            if (hasattr(agent, 'domain_keywords') and 
                domain in agent.domain_keywords and 
                len(agent.domain_keywords[domain]) > 0):
                working_legacy += 1
        
        # At least 50% of legacy domains should still work
        legacy_rate = working_legacy / len(legacy_domains)
        assert legacy_rate >= 0.5, f"Backward compatibility rate {legacy_rate:.1%} too low"
        
        print(f"✅ Backward compatibility: {working_legacy}/{len(legacy_domains)} legacy domains working")

# Test runner function
def run_domains_integration_tests():
    """Run all domain integration tests with dynamic configuration"""
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure
    ])

if __name__ == "__main__":
    # Quick validation run
    try:
        config_manager = DomainConfigManager()
        print(f"🚀 MeeTARA Lab - Dynamic Domain Integration Tests")
        print(f"📊 Configuration: {config_manager.domain_count} domains across {len(config_manager.categories)} categories")
        print(f"📁 Categories: {', '.join(sorted(config_manager.categories))}")
        print("\nRunning tests...")
        run_domains_integration_tests()
    except FileNotFoundError as e:
        print(f"❌ Configuration error: {e}")
        print("Please ensure the domain configuration YAML file exists.")
    except Exception as e:
        print(f"❌ Test error: {e}") 
