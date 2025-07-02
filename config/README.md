# Configuration Files
*MeeTARA Lab Configuration Management*

## 📁 Configuration Overview

This directory contains configuration files for MeeTARA Lab system components.

## 📋 Configuration Files

### `cloud-optimized-domain-mapping.yaml`
**Purpose**: Defines the mapping and optimization settings for cloud GPU training across 60+ domains.

**Key Sections**:
- **Domain Definitions**: Complete list of AI domains (Business, Technical, Creative, etc.)
- **GPU Optimization**: Provider-specific configurations (Google Colab, Lambda Labs, etc.)
- **Cost Parameters**: Budget allocation and cost optimization rules
- **Performance Targets**: Speed and quality benchmarks per domain
- **Training Parameters**: Proven TARA Universal Model settings

**Usage**:
- Referenced by `cloud-training/training_orchestrator.py`
- Used by `cost-optimization/cost_monitor.py` 
- Integrated with `model-factory/gguf_factory.py`

## 🛠️ Configuration Standards

### File Format
- **YAML**: Primary format for configuration files
- **JSON**: Alternative format for API configurations
- **Environment Variables**: Sensitive data (API keys, secrets)

### Naming Convention
- Use kebab-case: `cloud-optimized-domain-mapping.yaml`
- Include purpose in name: `training-parameters.yaml`
- Version if needed: `model-config-v2.yaml`

### Documentation
- Include inline comments for complex settings
- Reference documentation in this README
- Link to relevant component documentation

## 🔧 Configuration Usage

### Loading Configuration
```python
# Example: Loading domain mapping
import yaml

with open('config/cloud-optimized-domain-mapping.yaml', 'r') as f:
    domain_config = yaml.safe_load(f)
```

### Environment Variables
```bash
# Set environment variables for sensitive data
export OPENAI_API_KEY="your-key-here"
export PERPLEXITY_API_KEY="your-key-here"
```

### Configuration Validation
- Use schema validation for YAML files
- Implement configuration tests in `tests/` directory
- Document required vs optional parameters

## 📊 Current Configuration Status

- ✅ **Domain Mapping**: Complete 60+ domains configured
- ✅ **GPU Optimization**: Multi-cloud provider support
- ✅ **Cost Parameters**: Budget optimization rules defined
- ⏳ **Security Config**: API key management setup pending
- ⏳ **Deployment Config**: Production settings pending

## 🔗 Related Documentation

- **[Cloud Training Guide](../docs/cloud-training/README.md)** - GPU orchestration setup
- **[Cost Optimization](../docs/cost-optimization/README.md)** - Budget management
- **[Model Factory](../docs/model-factory/README.md)** - GGUF creation configuration
- **[Performance Benchmarks](../docs/performance/README.md)** - Configuration tuning

## 🚀 Next Steps

1. **Security Configuration**: Implement secure API key management
2. **Environment Configs**: Create dev/staging/production configs
3. **Validation Schema**: Add YAML schema validation
4. **Configuration Tests**: Comprehensive config testing
5. **Documentation**: Detailed parameter documentation

---

*Configuration files are version-controlled and documented. Last updated: July 2, 2025* 