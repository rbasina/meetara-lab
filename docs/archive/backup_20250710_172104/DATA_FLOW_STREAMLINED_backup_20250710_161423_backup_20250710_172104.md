# MeeTARA Lab - Final Data Flow Architecture

## Overview
This document outlines the final data flow architecture for MeeTARA Lab, using the existing `models/` directory for GGUF files and `data/` directory for training data and trained models.

## Final Data Flow Structure

```
data/
├── training/                # Generated training data (JSON files)
│   ├── healthcare/          # Training data by category
│   │   ├── general_health_training_data.json
│   │   ├── mental_health_training_data.json
│   │   └── [other healthcare domains]
│   ├── business/           
│   │   ├── entrepreneurship_training_data.json
│   │   ├── marketing_training_data.json
│   │   └── [other business domains]
│   └── [other categories]/
├── models/                  # Trained model files (.pt)
│   └── trained/             
│       ├── healthcare/
│       │   ├── general_health_trained_model.pt
│       │   ├── mental_health_trained_model.pt
│       │   └── [other healthcare models]
│       ├── business/
│       │   ├── entrepreneurship_trained_model.pt
│       │   ├── marketing_trained_model.pt
│       │   └── [other business models]
│       └── [other categories]/
└── logs/                   # Training and processing logs

models/                     # GGUF files by test type
├── A_universal_full/       # Universal full models (4.6GB)
│   └── meetara_universal_full_Q4_K_M.gguf
├── B_universal_lite/       # Universal lite models (1.2GB)
│   └── meetara_universal_lite_Q4_K_M.gguf
├── C_category_specific/    # Category-specific models (150MB each)
│   ├── healthcare/
│   │   └── meetara_healthcare_category_Q4_K_M.gguf
│   ├── business/
│   │   └── meetara_business_category_Q4_K_M.gguf
│   └── [other categories]/
├── D_domain_specific/      # Domain-specific models (8.3MB each)
│   ├── healthcare/
│   │   ├── meetara_general_health_Q4_K_M.gguf
│   │   ├── meetara_mental_health_Q4_K_M.gguf
│   │   ├── general_health_metadata.json
│   │   └── [other healthcare GGUF]
│   ├── business/
│   │   ├── meetara_entrepreneurship_Q4_K_M.gguf
│   │   ├── meetara_marketing_Q4_K_M.gguf
│   │   ├── entrepreneurship_metadata.json
│   │   └── [other business GGUF]
│   └── [other categories]/
└── gguf/                   # GGUF files by environment
    ├── production/         # Production-ready GGUF
    ├── development/        # Development GGUF
    └── legacy/            # Legacy backups
```

## Test-Type Based GGUF Destinations

### **Domain Testing** (Individual domains)
- **Destination**: `models/D_domain_specific/{category}/`
- **Example**: `models/D_domain_specific/healthcare/meetara_general_health_Q4_K_M.gguf`
- **Size**: 8.3MB per domain
- **Use Case**: Testing specific domain performance

### **Category Testing** (Entire categories)
- **Destination**: `models/C_category_specific/{category}/`
- **Example**: `models/C_category_specific/business/meetara_business_category_Q4_K_M.gguf`
- **Size**: 150MB per category
- **Use Case**: Testing category-wide intelligence

### **Universal Testing** (All domains)
- **Destination**: `models/A_universal_full/` or `models/B_universal_lite/`
- **Example**: `models/A_universal_full/meetara_universal_full_Q4_K_M.gguf`
- **Size**: 4.6GB (full) or 1.2GB (lite)
- **Use Case**: Testing complete system capability

### **Development Testing**
- **Destination**: `models/gguf/development/`
- **Example**: `models/gguf/development/meetara_test_domain_Q4_K_M.gguf`
- **Use Case**: Development and testing iterations

## Data Flow Process

### 1. Training Data Generation
- **Input**: Domain specifications from YAML config
- **Process**: Intelligence Hub generates high-quality samples
- **Output**: `data/training/{category}/{domain}_training_data.json`
- **Size**: 5,000-15,000 samples per domain (tier-specific)

### 2. Model Training
- **Input**: Training data from `data/training/`
- **Process**: GPU Training Engine with tier-specific parameters
- **Output**: `data/models/trained/{category}/{domain}_trained_model.pt`
- **Size**: 80-100MB per trained model

### 3. GGUF Creation (Based on Test Type)
- **Input**: Trained models from `data/models/trained/`
- **Process**: GGUF Factory with multiple quantization levels
- **Output**: Varies by test type:
  - Domain: `models/D_domain_specific/{category}/`
  - Category: `models/C_category_specific/{category}/`
  - Universal: `models/A_universal_full/` or `models/B_universal_lite/`
  - Development: `models/gguf/development/`

### 4. Metadata Generation
- **Input**: Training and GGUF results
- **Process**: Comprehensive metadata creation
- **Output**: `models/D_domain_specific/{category}/{domain}_metadata.json`
- **Size**: ~1KB per metadata file

## Key Benefits

### 1. **Test-Type Organization**
- Clear separation by test scope (domain vs category vs universal)
- Easy to find models based on testing needs
- Scalable for different testing scenarios

### 2. **Existing Structure Preserved**
- Leverages well-designed `models/` directory
- Maintains compatibility with existing documentation
- No need to migrate existing files

### 3. **Logical Data Flow**
- Training data → Trained models → GGUF files
- Clear progression from input to final output
- Metadata co-located with GGUF files

### 4. **Production Ready**
- Development vs production separation
- Clear deployment paths
- Easy backup and versioning

## File Naming Conventions

### Training Data
- Format: `{domain}_training_data.json`
- Example: `general_health_training_data.json`

### Trained Models
- Format: `{domain}_trained_model.pt`
- Example: `general_health_trained_model.pt`

### GGUF Models
- **Domain**: `meetara_{domain}_{quantization}.gguf`
- **Category**: `meetara_{category}_category_{quantization}.gguf`
- **Universal**: `meetara_universal_{type}_{quantization}.gguf`

### Metadata
- Format: `{domain}_metadata.json`
- Example: `general_health_metadata.json`

## Implementation Status

### ✅ **Updated for Final Structure**
1. **Complete Agent Ecosystem**: GGUF creation uses `models/D_domain_specific/`
2. **Integrated GPU Pipeline**: Development GGUF uses `models/gguf/development/`
3. **GPU Training Engine**: Trained models saved to `data/models/trained/`
4. **Intelligent Logger**: Logs saved to `logs/` directory

### 🎯 **Ready for Different Test Types**
- **Domain Testing**: `python cloud-training/production_launcher.py --domains general_health`
- **Category Testing**: `python cloud-training/production_launcher.py --category healthcare`
- **Universal Testing**: `python cloud-training/production_launcher.py --universal`

## Summary

The final data flow architecture provides:
- **Test-Type Based Organization**: GGUF files organized by scope (A, B, C, D)
- **Logical Data Progression**: Training → Models → GGUF → Metadata
- **Existing Structure Preserved**: Leverages well-designed `models/` directory
- **Production Ready**: Clear separation and deployment paths

This architecture perfectly supports your requirement for GGUF files to go to `G:\My Drive\meetara-lab\models` based on the type of test being performed (domain, category, or universal). 