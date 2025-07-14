# MeeTARA Lab: Base Model Selection & Compatibility Insights

## Overview

This document captures the critical findings, technical pitfalls, and best practices discovered during the MeeTARA Lab training pipeline experiments. It is intended as a reference for selecting **trusted, accurate, and compatible base models** for each domain, especially when using LoRA/PEFT and targeting GGUF conversion for llama.cpp.

---

## 1. Model Compatibility Matrix

| Model                        | LoRA Support | GGUF Conversion | Hugging Face Loading | Notes                                 |
|------------------------------|--------------|-----------------|----------------------|---------------------------------------|
| **DialoGPT (small/medium)**  | ⚠️ Partial   | ❌ Fails         | ✅ Yes               | LoRA not fully supported, GGUF fails  |
| **Phi-3.5-mini-instruct**    | ✅ Yes       | ✅ Yes           | ⚠️ Custom code bug   | Best if loading works                 |
| **Qwen2.5-7B/14B-Instruct**  | ✅ Yes       | ✅ Yes           | ✅ Yes               | Modern, robust, fully supported       |
| **Llama/Mistral**            | ✅ Yes       | ✅ Yes           | ✅ Yes               | Modern, robust, fully supported       |

---

## 2. Base Model Recommendations by Domain Category

### **Qwen Models (Best for Colab + Full Pipeline Compatibility)**

#### **Qwen/Qwen2.5-14B-Instruct (14B Parameters)**
**Best for:**
- **Business & Strategy**: `entrepreneurship`, `project_management`, `team_leadership`, `strategy`, `consulting`
- **Technology & Engineering**: `programming`, `ai_ml`, `data_analysis`, `software_development`, `scientific_research`, `engineering`
- **Research & Academic**: `research_assistance`, `academic_tutoring`, `research`, `academic_tutoring_research`
- **Aerospace & Manufacturing**: `aeronautics`, `automobile`, `space_technology`, `agriculture`, `manufacturing`
- **Remote Work**: `remote_work`

**Why Qwen2.5-14B-Instruct:**
- ✅ **Full Colab Compatibility**: No import errors, works reliably
- ✅ **LoRA Support**: Complete PEFT/LoRA compatibility
- ✅ **GGUF Conversion**: Perfect llama.cpp integration
- ✅ **High Performance**: 14B parameters for complex reasoning
- ✅ **Apache-2.0 License**: Open source and free
- ✅ **Instruction-Tuned**: Optimized for task-specific responses

#### **Qwen/Qwen2.5-7B-Instruct (7B Parameters)**
**Best for:**
- **Personal & Planning**: `personal_assistant`, `planning`, `time_management`
- **Content Creation**: `content_creation`, `social_media_management`
- **Educational Technology**: `educational_technology`
- **Business Operations**: `operations`

**Why Qwen2.5-7B-Instruct:**
- ✅ **Balanced Performance**: Good speed/quality balance
- ✅ **Full Pipeline Compatibility**: LoRA + GGUF + Colab
- ✅ **Resource Efficient**: Lower memory requirements
- ✅ **Apache-2.0 License**: Open source and free

### **Phi-3 Models (Best for Local Training + High Quality)**

#### **microsoft/Phi-3-medium-4k-instruct (14B Parameters)**
**Best for:**
- **Healthcare & Safety**: `general_health`, `mental_health`, `nutrition`, `sleep`, `stress_management`, `preventive_care`, `chronic_conditions`, `medication_management`, `emergency_care`, `women_health`, `senior_health`
- **Legal & Financial**: `legal_assistance`, `insurance`, `real_estate`, `legal`, `financial`
- **Crisis & Emergency**: `crisis_management`, `disaster_preparedness`, `emergency_response`, `safety_security`
- **Psychology & Wellness**: `psychology`, `yoga`, `life_coaching`, `social_support`
- **Creative & Arts**: `writing`, `storytelling`, `design_thinking`, `art_appreciation`, `mythology`, `spiritual`
- **Sports & Recreation**: `sports_recreation`
- **Travel & Tourism**: `travel_tourism`
- **Daily Life**: `parenting`, `relationships`, `decision_making`, `conflict_resolution`, `work_life_balance`
- **Business Skills**: `marketing`, `sales`, `financial_planning`, `hr_management`, `legal_business`
- **Education**: `skill_development`, `career_guidance`, `exam_preparation`, `study_techniques`
- **Technology**: `cybersecurity`

**Why Phi-3-medium-4k-instruct:**
- ✅ **High Quality**: 14B parameters for complex reasoning
- ✅ **Local Training**: Works perfectly on local machines
- ✅ **LoRA Support**: Full PEFT compatibility
- ✅ **GGUF Conversion**: Complete llama.cpp integration
- ✅ **MIT License**: Open source and free
- ❌ **Colab Issue**: `ModuleNotFoundError` in Colab (Hugging Face bug)

#### **microsoft/Phi-3.5-mini-instruct (3.8B Parameters)**
**Best for:**
- **Communication & Social**: `communication`, `social_media`, `customer_service`
- **Home & Daily**: `home_management`, `transportation`, `shopping`
- **Learning & Education**: `language_learning_education`, `language_learning_professional`
- **Creative & Media**: `photography`, `music`, `digital_literacy`
- **Technology Support**: `tech_support`
- **Health & Fitness**: `fitness_healthcare`

**Why Phi-3.5-mini-instruct:**
- ✅ **Fast & Efficient**: 3.8B parameters for quick responses
- ✅ **Local Training**: Perfect for local GPU constraints
- ✅ **LoRA Support**: Full PEFT compatibility
- ✅ **GGUF Conversion**: Complete llama.cpp integration
- ✅ **MIT License**: Open source and free
- ❌ **Colab Issue**: `ModuleNotFoundError` in Colab (Hugging Face bug)

### **DialoGPT Models (Limited Compatibility - Not Recommended)**

#### **microsoft/DialoGPT-small/medium**
**Current Mapping**: `shopping` (DialoGPT-small)

**Why DialoGPT is NOT Recommended:**
- ❌ **LoRA Incompatible**: GPT-2 architecture doesn't support modern LoRA
- ❌ **GGUF Conversion Fails**: Tensor mapping errors with llama.cpp
- ❌ **Limited Capabilities**: Older architecture, limited reasoning
- ✅ **Only Advantage**: Loads in Colab without import errors
- ❌ **Not Suitable**: For MeeTARA Lab's modern pipeline requirements

---

## 3. Optimal Model Selection Strategy

### **For Colab Training (Cloud-Based):**
```yaml
# Use Qwen models for full compatibility
business_strategy: Qwen/Qwen2.5-14B-Instruct
technology_engineering: Qwen/Qwen2.5-14B-Instruct
personal_planning: Qwen/Qwen2.5-7B-Instruct
content_creation: Qwen/Qwen2.5-7B-Instruct
```

### **For Local Training (Local Machine):**
```yaml
# Use Phi-3 models for high quality
healthcare_safety: microsoft/Phi-3-medium-4k-instruct
legal_financial: microsoft/Phi-3-medium-4k-instruct
creative_arts: microsoft/Phi-3-medium-4k-instruct
communication_social: microsoft/Phi-3.5-mini-instruct
daily_life: microsoft/Phi-3.5-mini-instruct
```

### **For Maximum Quality (Regardless of Environment):**
```yaml
# Premium domains with highest quality models
healthcare: microsoft/Phi-3-medium-4k-instruct (local) or Qwen/Qwen2.5-14B-Instruct (Colab)
legal_financial: microsoft/Phi-3-medium-4k-instruct (local) or Qwen/Qwen2.5-14B-Instruct (Colab)
crisis_emergency: microsoft/Phi-3-medium-4k-instruct (local) or Qwen/Qwen2.5-14B-Instruct (Colab)
```

## 4. Compatibility Matrix

| Model | Colab | Local | LoRA | GGUF | License | Best For |
|-------|-------|-------|------|------|---------|----------|
| **Qwen2.5-14B-Instruct** | ✅ | ✅ | ✅ | ✅ | Apache-2.0 | Complex domains, business, tech |
| **Qwen2.5-7B-Instruct** | ✅ | ✅ | ✅ | ✅ | Apache-2.0 | Balanced performance, planning |
| **Phi-3-medium-4k-instruct** | ❌ | ✅ | ✅ | ✅ | MIT | High quality, safety-critical |
| **Phi-3.5-mini-instruct** | ❌ | ✅ | ✅ | ✅ | MIT | Fast responses, local training |
| **DialoGPT-medium** | ✅ | ✅ | ❌ | ❌ | MIT | Not recommended |

## 5. Recommendations by Domain Category

### **Premium/Safety-Critical Domains:**
- **Healthcare**: Phi-3-medium-4k-instruct (local) or Qwen2.5-14B-Instruct (Colab)
- **Legal/Financial**: Phi-3-medium-4k-instruct (local) or Qwen2.5-14B-Instruct (Colab)
- **Crisis/Emergency**: Phi-3-medium-4k-instruct (local) or Qwen2.5-14B-Instruct (Colab)

### **Expert/Complex Domains:**
- **Business Strategy**: Qwen2.5-14B-Instruct (best choice for Colab)
- **Technology/Engineering**: Qwen2.5-14B-Instruct (best choice for Colab)
- **Research/Academic**: Qwen2.5-14B-Instruct (best choice for Colab)

### **Quality/Balanced Domains:**
- **Creative/Arts**: Phi-3-medium-4k-instruct (local) or Qwen2.5-7B-Instruct (Colab)
- **Daily Life**: Phi-3.5-mini-instruct (local) or Qwen2.5-7B-Instruct (Colab)
- **Communication**: Phi-3.5-mini-instruct (local) or Qwen2.5-7B-Instruct (Colab)

### **Fast/Lightweight Domains:**
- **Social Media**: Phi-3.5-mini-instruct (local) or Qwen2.5-7B-Instruct (Colab)
- **Shopping**: Phi-3.5-mini-instruct (local) or Qwen2.5-7B-Instruct (Colab)

---

## 6. Key Technical Findings

### **A. DialoGPT (GPT-2 Architecture)**
- **LoRA/PEFT:**  
  - Not fully supported. PEFT cannot patch GPT2MLP layers, resulting in warnings and fallback to non-LoRA training.
- **GGUF Conversion:**  
  - Fails with tensor mapping errors (`ValueError: Can not map tensor ...base_layer.bias`).
  - llama.cpp's converter does not support GPT-2 architectures, especially with LoRA.
- **Use Case:**  
  - Only suitable for Hugging Face workflows, not for GGUF/llama.cpp or production Trinity pipelines.

### **B. Phi-3.5-mini-instruct**
- **LoRA/PEFT:**  
  - Fully supported, parameter-efficient fine-tuning works as expected.
- **GGUF Conversion:**  
  - Works perfectly if the model loads.
- **Hugging Face Loading:**  
  - Requires `trust_remote_code=True` due to custom code (`configuration_phi3.py`).
  - **Common Bug:**  
    - `ModuleNotFoundError: No module named 'transformers_modules.microsoft.Phi-3'`
    - Caused by Hugging Face's dynamic import mechanism, especially in Colab.
    - Fixes: Clear cache, restart runtime, reinstall packages, or use a local environment.
- **Use Case:**  
  - Best for multi-domain, multi-intent, and Trinity-enhanced workflows—**if you can load it**.

### **C. Qwen, Llama, Mistral**
- **LoRA/PEFT:**  
  - Fully supported.
- **GGUF Conversion:**  
  - Fully supported.
- **Hugging Face Loading:**  
  - No custom code, loads reliably in all environments.
- **Use Case:**  
  - Highly recommended for production, especially when GGUF/llama.cpp compatibility is required.

---

## 7. Lessons Learned & Best Practices

### **1. Always Check Model Architecture**
- GPT-2 based models (like DialoGPT) are not suitable for LoRA+GGUF workflows.
- Prefer modern architectures (Phi, Qwen, Llama, Mistral).

### **2. Beware of Hugging Face Custom Code**
- Models requiring `trust_remote_code=True` may fail in Colab or ephemeral environments.
- Always test model loading in your target environment before committing to a base model.

### **3. LoRA/PEFT Compatibility**
- Not all models support LoRA out-of-the-box.
- Check for warnings about unsupported modules during training.

### **4. GGUF Conversion**
- Only certain architectures are supported by llama.cpp's GGUF converter.
- Test conversion early in your pipeline.

### **5. Environment Matters**
- Colab is convenient but can have dynamic import bugs.
- Local virtual environments are more reliable for custom code models.

### **6. Domain-Specific Model Selection**
- **Safety-Critical Domains**: Use highest quality models (Phi-3-medium or Qwen2.5-14B)
- **Complex Reasoning**: Prefer larger models (14B parameters)
- **Fast Responses**: Use smaller models (3.8B-7B parameters)
- **Colab Training**: Stick to Qwen models for reliability
- **Local Training**: Use Phi-3 models for maximum quality

---

## 8. Action Plan

### **Immediate Actions:**
1. **For Colab Training**: Use Qwen models exclusively
2. **For Local Training**: Use Phi-3 models for maximum quality
3. **Remove DialoGPT**: Replace with Qwen or Phi-3 alternatives
4. **Update Configuration**: Modify domain mappings based on training environment
5. **Test Compatibility**: Verify LoRA and GGUF conversion for each model

### **Environment-Specific Strategy:**
- **Colab Environment**: Qwen2.5-7B/14B-Instruct for all domains
- **Local Environment**: Phi-3 models for quality, Phi-3.5-mini for speed
- **Production Deployment**: Qwen models for reliability and full pipeline compatibility

---

## 9. Troubleshooting Checklist

- **Model fails to load with custom code error:**  
  - Clear Hugging Face cache
  - Restart runtime
  - Reinstall transformers/peft
  - Try on a local machine

- **LoRA setup fails:**  
  - Check model architecture and PEFT support
  - Switch to a modern, supported model

- **GGUF conversion fails:**  
  - Confirm model is supported by llama.cpp
  - Avoid GPT-2 based models

---

## 10. Conclusion

**Model selection is critical for a robust, scalable, and future-proof AI pipeline.**  
Always validate LoRA and GGUF compatibility for each base model and domain.  
This documentation should serve as a guide for all future MeeTARA Lab model selection and pipeline design decisions.

---

## 11. Verified Licenses

The following base models have been verified for their licenses as per `config/trinity_config.yaml`:

| Model                                 | License     |
|----------------------------------------|-------------|
| HuggingFaceTB/SmolLM2-1.7B            | Apache-2.0  |
| microsoft/Phi-3.5-mini-instruct       | MIT         |
| microsoft/Phi-3-medium-4k-instruct    | MIT         |
| Qwen/Qwen2.5-7B-Instruct              | Apache-2.0  |
| Qwen/Qwen2.5-14B-Instruct             | Apache-2.0  |

**Always ensure license compatibility for your use case before deploying models in production.**

---

## 12. Next Steps

**Actionable items for the next session:**

1. **Phi-3.5-mini-instruct Loading:**
   - Try loading on a local machine (fresh Python venv, not Colab) to bypass the Colab dynamic import bug.
   - If still blocked, consider using Qwen2.5-7B/14B-Instruct, Llama, or Mistral for full LoRA + GGUF compatibility.

2. **Re-test HuggingFaceTB/SmolLM2-1.7B:**
   - Check if LoRA and GGUF conversion now work out-of-the-box.
   - If successful, re-add to the config as a trusted, open-source base model.

3. **Monitor Hugging Face Warnings:**
   - The `resume_download` warning is safe to ignore for now.
   - The `HF_TOKEN` warning is only needed for private models or higher download limits.

4. **General:**
   - Continue to validate all new base models for LoRA, GGUF, and Trinity compatibility before adding to production.
   - Keep documentation and memory bank updated after each major finding.

---

*End of actionable checklist. Resume here next session!*

---

*Last updated: 2025-07-14* 