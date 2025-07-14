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

## 2. Key Technical Findings

### **A. DialoGPT (GPT-2 Architecture)**
- **LoRA/PEFT:**  
  - Not fully supported. PEFT cannot patch GPT2MLP layers, resulting in warnings and fallback to non-LoRA training.
- **GGUF Conversion:**  
  - Fails with tensor mapping errors (`ValueError: Can not map tensor ...base_layer.bias`).
  - llama.cpp’s converter does not support GPT-2 architectures, especially with LoRA.
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
    - Caused by Hugging Face’s dynamic import mechanism, especially in Colab.
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

## 3. Lessons Learned & Best Practices

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
- Only certain architectures are supported by llama.cpp’s GGUF converter.
- Test conversion early in your pipeline.

### **5. Environment Matters**
- Colab is convenient but can have dynamic import bugs.
- Local virtual environments are more reliable for custom code models.

---

## 4. Recommendations for Model Selection

- **For GGUF/llama.cpp and LoRA:**  
  - Use Phi-3.5-mini-instruct, Qwen2.5, Llama, or Mistral.
- **For Hugging Face-only workflows:**  
  - DialoGPT is acceptable, but you lose LoRA and GGUF benefits.
- **For maximum reliability:**  
  - Avoid models that require `trust_remote_code=True` unless you have a robust local setup.

---

## 5. Troubleshooting Checklist

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

## 6. Conclusion

**Model selection is critical for a robust, scalable, and future-proof AI pipeline.**  
Always validate LoRA and GGUF compatibility for each base model and domain.  
This documentation should serve as a guide for all future MeeTARA Lab model selection and pipeline design decisions.

---

## 7. Verified Licenses

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

## 8. Next Steps

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