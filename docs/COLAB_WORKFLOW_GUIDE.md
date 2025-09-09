# MeeTARA Lab - Colab Workflow Guide
*Two-Phase Approach for Secure and Efficient Training*

## 🎯 **OVERVIEW**

This guide explains how to use the enhanced `production_launcher.py` with the new `--skip-training` parameter to separate data generation (local) from model training (Colab).

## 🚀 **TWO-PHASE WORKFLOW**

### **Phase 1: Local Data Generation (Cursor)**
Generate training data locally with AI services and commit to GitHub.

### **Phase 2: Colab Model Training**
Clone repository in Colab and train models with pre-generated data.

---

## 📋 **PHASE 1: LOCAL DATA GENERATION**

### **Step 1: Generate Data for Single Domain**
```bash
# Generate data for entrepreneurship domain
python cloud-training/production_launcher.py \
    --skip-training \
    --domain entrepreneurship \
    --commit

# This will:
# ✅ Generate 6000 samples using AI services
# ✅ Save data to data/production/training/
# ✅ Commit and push to GitHub automatically
```

### **Step 2: Generate Data for Category**
```bash
# Generate data for all healthcare domains
python cloud-training/production_launcher.py \
    --skip-training \
    --category healthcare \
    --commit

# This will generate data for all domains in healthcare category
```

### **Step 3: Generate Data for All Domains**
```bash
# Generate data for all 62+ domains
python cloud-training/production_launcher.py \
    --skip-training \
    --all \
    --commit

# WARNING: This will take several hours and use significant API credits
```

### **Available Options:**
- `--skip-training`: Skip model training, only generate data
- `--commit`: Automatically commit and push data to GitHub
- `--environment dev|production`: Choose data path (default: dev)
- `--domain <name>`: Single domain
- `--category <name>`: All domains in category
- `--all`: All domains

---

## 🚀 **PHASE 2: COLAB MODEL TRAINING**

### **Step 1: Open Colab Notebook**
1. Go to [Google Colab](https://colab.research.google.com/)
2. Upload `notebooks/colab_training_phase.ipynb`
3. Or create new notebook and copy the cells

### **Step 2: Clone Repository**
```python
# Replace with your actual GitHub URL
!git clone https://github.com/your-username/meetara-lab.git
!cd meetara-lab
```

### **Step 3: Install Dependencies**
```python
%pip install torch transformers peft datasets accelerate bitsandbytes llama-cpp-python -q
%pip install -r meetara-lab/requirements.txt -q
```

### **Step 4: Train Single Domain**
```bash
!cd meetara-lab && python cloud-training/production_launcher.py \
    --domains entrepreneurship \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --skip-quantization \
    --environment production
```

### **Step 5: Train Category**
```bash
!cd meetara-lab && python cloud-training/production_launcher.py \
    --category healthcare \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --skip-quantization \
    --environment production
```

### **Step 6: Train All Domains**
```bash
!cd meetara-lab && python cloud-training/production_launcher.py \
    --all \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --skip-quantization \
    --environment production
```

---

## 🔐 **SECURITY BENEFITS**

### **✅ API Keys Stay Local**
- OpenAI, Gemini, DeepSeek keys never leave your machine
- No sensitive credentials in Colab environment
- Full control over API usage and costs

### **✅ Cost Control**
- Generate data locally with controlled API usage
- Use Colab's free/cheap GPU time for training
- No unexpected API charges in Colab

### **✅ Quality Control**
- Test and validate data generation locally
- Debug issues before expensive Colab training
- Full control over data quality

---

## 💰 **COST ANALYSIS**

### **Local Data Generation:**
- **API Costs**: ~$5-15 per domain (6000 samples)
- **Time**: 2-3 hours per domain
- **Control**: Full control over costs and quality

### **Colab Training:**
- **GPU Costs**: Free (T4) or ~$0.50/hour (V100/A100)
- **Time**: 30-60 minutes per domain
- **Scalability**: Easy to scale across domains

---

## 🎯 **RECOMMENDED WORKFLOW**

### **For Single Domain Testing:**
1. **Local**: `python cloud-training/production_launcher.py --skip-training --domain entrepreneurship --commit`
2. **Colab**: Train with single domain command
3. **Local**: Test trained model

### **For Category Training:**
1. **Local**: `python cloud-training/production_launcher.py --skip-training --category healthcare --commit`
2. **Colab**: Train with category command
3. **Local**: Validate all domain models

### **For Full Production:**
1. **Local**: `python cloud-training/production_launcher.py --skip-training --all --commit`
2. **Colab**: Train with all domains command
3. **Local**: Deploy to MeeTARA frontend

---

## 🛠️ **TROUBLESHOOTING**

### **Data Generation Issues:**
- Check API keys in `.env` file
- Verify domain exists in config
- Check internet connection for AI services

### **Colab Training Issues:**
- Ensure data was committed to GitHub
- Check GPU availability in Colab
- Verify base model is accessible

### **Git Issues:**
- Ensure you have push permissions
- Check if repository is up to date
- Verify GitHub credentials

---

## 📊 **MONITORING PROGRESS**

### **Local Data Generation:**
- Monitor API usage and costs
- Check data quality scores
- Verify file generation

### **Colab Training:**
- Monitor GPU usage
- Check training progress
- Verify safetensor file generation

---

## 🎉 **SUCCESS INDICATORS**

### **Phase 1 Complete:**
- ✅ Data files generated in `data/production/training/`
- ✅ Files committed to GitHub
- ✅ Quality scores > 0.95

### **Phase 2 Complete:**
- ✅ Safetensor files generated
- ✅ Training completed without errors
- ✅ Models ready for deployment

---

## 🚀 **NEXT STEPS**

After successful training:
1. **Download Models**: Download safetensor files from Colab
2. **Test Locally**: Validate models work correctly
3. **Deploy**: Integrate with MeeTARA frontend
4. **Monitor**: Track performance and usage

---

*This workflow ensures security, cost control, and quality while leveraging Colab's powerful GPUs for efficient training.*
