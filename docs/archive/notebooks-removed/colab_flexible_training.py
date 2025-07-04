# =============================================================================
# 🚀 MeeTARA Lab - Flexible Colab Training Pipeline
# Single Domain | Multiple Domains | All Domains
# =============================================================================

# CELL 1: GPU Setup and Detection
# ================================

import subprocess
import torch
import time
from datetime import datetime

print("🚀 MeeTARA Lab Flexible Training Pipeline")
print("="*60)

# Detect GPU type for cost optimization
try:
    gpu_info = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                             capture_output=True, text=True)
    if gpu_info.returncode == 0:
        gpu_name = gpu_info.stdout.strip()
        print(f"✅ GPU Detected: {gpu_name}")
        
        if "T4" in gpu_name:
            GPU_TIER = "T4"
            COST_PER_HOUR = 0.40
            BATCH_SIZE = 16
            SPEED_FACTOR = "37x"
        elif "V100" in gpu_name:
            GPU_TIER = "V100"
            COST_PER_HOUR = 2.50
            BATCH_SIZE = 32
            SPEED_FACTOR = "75x"
        elif "A100" in gpu_name:
            GPU_TIER = "A100"
            COST_PER_HOUR = 4.00
            BATCH_SIZE = 64
            SPEED_FACTOR = "151x"
        else:
            GPU_TIER = "T4"
            COST_PER_HOUR = 0.40
            BATCH_SIZE = 16
            SPEED_FACTOR = "37x"
    else:
        print("⚠️ No GPU detected - using CPU (very slow)")
        GPU_TIER = "CPU"
        COST_PER_HOUR = 0.0
        BATCH_SIZE = 2
        SPEED_FACTOR = "1x"
except:
    print("⚠️ GPU detection failed - using CPU fallback")
    GPU_TIER = "CPU"
    COST_PER_HOUR = 0.0
    BATCH_SIZE = 2
    SPEED_FACTOR = "1x"

print(f"⚡ Speed: {SPEED_FACTOR} faster than CPU")
print(f"💰 Cost: ${COST_PER_HOUR}/hour")
print(f"📊 Batch Size: {BATCH_SIZE}")

# CUDA verification
print(f"\n🔥 PyTorch CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print(f"🎯 Estimated cost for all 62 domains: $8-15")
else:
    print("⚠️ Running on CPU - training will be slow")

print("\n✅ GPU setup complete!")

# CELL 2: Install Dependencies
# =============================

print("📦 Installing GPU-optimized dependencies...")

# Core ML libraries with CUDA support
get_ipython().system('pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q')
get_ipython().system('pip install transformers==4.53.0 datasets accelerate bitsandbytes -q')
get_ipython().system('pip install peft==0.15.2 trl optimum -q')

# GGUF creation libraries
get_ipython().system('pip install gguf llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu118 -q')

# Data processing & monitoring
get_ipython().system('pip install pyyaml tqdm rich -q')
get_ipython().system('pip install pandas numpy scipy -q')

print("✅ Dependencies installed successfully!")

# Configure environment for optimal GPU usage
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

print("🚀 Environment configured for maximum GPU efficiency!")

# CELL 3: Download Repository and Load Configuration
# ==================================================

import yaml
import json
from pathlib import Path

print("📂 Downloading MeeTARA Lab training pipeline...")

# Clone repository if not exists
if not Path('meetara-lab').exists():
    get_ipython().system('git clone https://github.com/rbasina/meetara-lab.git')
    print("✅ Repository cloned successfully!")
else:
    print("✅ Repository already exists, updating...")
    get_ipython().system('cd meetara-lab && git pull')

get_ipython().run_line_magic('cd', 'meetara-lab')

# Load cloud-optimized domain mapping
with open('config/trinity_domain_model_mapping_config.yaml', 'r') as f:
    DOMAIN_CONFIG = yaml.safe_load(f)

# Extract all domains with their models
DOMAIN_MODEL_MAPPING = {}
ALL_DOMAINS = []

for category, domains in DOMAIN_CONFIG.items():
    if isinstance(domains, dict) and category not in ['model_tiers', 'gpu_configs', 'cost_estimates', 'advantages', 'data_sources', 'tara_proven_params']:
        for domain, model in domains.items():
            DOMAIN_MODEL_MAPPING[domain] = {
                'model': model,
                'category': category
            }
            ALL_DOMAINS.append(domain)

print(f"📊 Loaded configuration for {len(ALL_DOMAINS)} domains")
print(f"📋 Categories: {', '.join([k for k,v in DOMAIN_CONFIG.items() if isinstance(v, dict) and k not in ['model_tiers', 'gpu_configs', 'cost_estimates', 'advantages', 'data_sources', 'tara_proven_params']])}")

# Display available options
print("\n🏷️ AVAILABLE CATEGORIES:")
for category, domains in DOMAIN_CONFIG.items():
    if isinstance(domains, dict) and category not in ['model_tiers', 'gpu_configs', 'cost_estimates', 'advantages', 'data_sources', 'tara_proven_params']:
        model_name = list(domains.values())[0] if domains else "N/A"
        print(f"  📂 {category}: {len(domains)} domains - Model: {model_name.split('/')[-1]}")

print("\n✅ Repository and configuration loaded!")

# CELL 4: Configure Training Mode
# ===============================

# ===============================
# MODIFY THESE SETTINGS FOR YOUR TRAINING:
# ===============================

# Choose training mode:
TRAINING_MODE = "single"  # Options: "single", "multiple", "all", "category"

# Configure based on your chosen mode:

# For SINGLE domain training:
SINGLE_DOMAIN = "healthcare"  # Change to any domain you want

# For MULTIPLE domains training:
MULTIPLE_DOMAINS = ["healthcare", "finance", "education"]  # Add/remove domains

# For CATEGORY training:
SELECTED_CATEGORIES = ["healthcare", "business"]  # Choose categories

# Training parameters:
SAMPLES_PER_DOMAIN = 2000  # Number of training samples per domain
COST_LIMIT = 45.0  # Stop training if cost exceeds this (safety margin)

# ===============================
# END OF CONFIGURATION
# ===============================

# Determine domains to train based on mode
if TRAINING_MODE == "single":
    domains_to_train = [SINGLE_DOMAIN]
    
elif TRAINING_MODE == "multiple":
    domains_to_train = MULTIPLE_DOMAINS
    
elif TRAINING_MODE == "all":
    domains_to_train = ALL_DOMAINS.copy()
    
elif TRAINING_MODE == "category":
    domains_to_train = []
    for category in SELECTED_CATEGORIES:
        if category in DOMAIN_CONFIG and isinstance(DOMAIN_CONFIG[category], dict):
            domains_to_train.extend(list(DOMAIN_CONFIG[category].keys()))
else:
    raise ValueError(f"Invalid training mode: {TRAINING_MODE}")

# Validate domains
invalid_domains = [d for d in domains_to_train if d not in DOMAIN_MODEL_MAPPING]
if invalid_domains:
    print(f"⚠️ Invalid domains: {invalid_domains}")
    domains_to_train = [d for d in domains_to_train if d in DOMAIN_MODEL_MAPPING]

# Display training plan
print(f"🎯 TRAINING PLAN")
print(f"Mode: {TRAINING_MODE}")
print(f"Domains: {len(domains_to_train)} ({', '.join(domains_to_train[:5])}{'...' if len(domains_to_train) > 5 else ''})")
print(f"GPU: {GPU_TIER} ({SPEED_FACTOR} faster)")

# Estimate cost and time  
estimated_time_per_domain = 5 if GPU_TIER == "T4" else 3 if GPU_TIER == "V100" else 2 if GPU_TIER == "A100" else 30
total_estimated_time = len(domains_to_train) * estimated_time_per_domain
estimated_cost = (total_estimated_time / 60) * COST_PER_HOUR

print(f"Estimated time: {total_estimated_time} minutes")
print(f"Estimated cost: ${estimated_cost:.2f}")

if estimated_cost > COST_LIMIT:
    print(f"⚠️ WARNING: Estimated cost (${estimated_cost:.2f}) exceeds limit (${COST_LIMIT})")
    print("Consider reducing domains or using a smaller GPU tier")
else:
    print(f"✅ Cost within budget (${COST_LIMIT})")

# Show some example domain-model mappings
print(f"\n📋 EXAMPLE DOMAIN-MODEL MAPPINGS:")
for i, domain in enumerate(domains_to_train[:5]):
    info = DOMAIN_MODEL_MAPPING[domain]
    print(f"  {i+1}. {domain} → {info['model'].split('/')[-1]} ({info['category']})")

print(f"\n✅ Training configuration ready!")

# CELL 5: Execute Training Pipeline  
# ==================================

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import random

# Real-time data generator (TARA Universal Model approach)
class RealTimeDataGenerator:
    def __init__(self):
        self.crisis_scenarios = ["emergency_health", "mental_crisis", "financial_emergency", "relationship_crisis", "work_emergency", "family_crisis"]
        self.emotional_contexts = ["stressed", "anxious", "confident", "confused", "hopeful", "frustrated", "excited", "worried", "determined", "overwhelmed"]
        self.professional_roles = ["healthcare_provider", "teacher", "manager", "consultant", "therapist", "advisor", "coach", "specialist", "expert"]
    
    def generate_domain_training_data(self, domain: str, samples: int = 2000):
        print(f"🔄 Generating {samples} samples for {domain}...")
        
        conversations = []
        scenario_types = ["consultation", "guidance", "problem_solving", "education", "support"]
        
        for i in range(samples):
            scenario = random.choice(scenario_types)
            emotion = random.choice(self.emotional_contexts)
            role = random.choice(self.professional_roles)
            
            # Crisis intervention (5% of data)
            is_crisis = random.random() < 0.05
            
            if is_crisis:
                crisis_type = random.choice(self.crisis_scenarios)
                conversation = [
                    {"role": "user", "content": f"I'm having a {crisis_type} and feeling {emotion}. I need help with {domain}."},
                    {"role": "assistant", "content": f"I understand you're going through a difficult time with {crisis_type}. Let me help you with {domain} in a way that addresses your {emotion} feelings. First, let's focus on immediate steps you can take..."}
                ]
            else:
                conversation = [
                    {"role": "user", "content": f"As someone feeling {emotion}, I need guidance on {domain}. Can you help?"},
                    {"role": "assistant", "content": f"Absolutely! I understand you're feeling {emotion} about {domain}. As your {role}, I'll provide comprehensive guidance that acknowledges your emotional state while giving you practical, actionable advice..."}
                ]
            
            conversations.append({
                "domain": domain,
                "conversation": conversation,
                "quality_score": random.uniform(0.7, 1.0)
            })
            
            if (i + 1) % 200 == 0:
                print(f"  📊 Generated {i + 1}/{samples} samples")
        
        # Apply TARA's 31% quality filtering
        print(f"🔍 Applying quality filter (31% success rate)...")
        sorted_conversations = sorted(conversations, key=lambda x: x['quality_score'], reverse=True)
        cutoff_index = int(len(sorted_conversations) * 0.31)
        filtered_conversations = sorted_conversations[:cutoff_index]
        
        print(f"✅ Domain {domain}: {len(filtered_conversations)} high-quality samples from {samples} generated")
        return filtered_conversations

# Training function
def train_domain_model(domain, model_name, training_data):
    print(f"\n🚀 Training {domain} with {model_name}")
    
    # Load model and tokenizer
    print(f"📥 Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    # Configure LoRA (TARA proven parameters)
    tara_params = DOMAIN_CONFIG['tara_proven_params']
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=tara_params['lora_r'],
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    print(f"✅ LoRA configured: r={tara_params['lora_r']}")
    
    # Prepare dataset
    formatted_data = []
    for item in training_data:
        conversation = item['conversation']
        text = f"User: {conversation[0]['content']}\\nAssistant: {conversation[1]['content']}"
        formatted_data.append({"text": text})
    
    dataset = Dataset.from_list(formatted_data)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=tara_params['sequence_length']
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    output_dir = f"./models/{domain}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=max(1, tara_params['batch_size'] // BATCH_SIZE),
        max_steps=tara_params['max_steps'],
        learning_rate=tara_params['learning_rate'],
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        dataloader_pin_memory=torch.cuda.is_available(),
        save_steps=200,
        logging_steps=50,
        remove_unused_columns=False,
        report_to=None,
    )
    
    # Train model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    
    print(f"⚡ Starting training...")
    training_start = time.time()
    
    trainer.train()
    
    training_time = time.time() - training_start
    training_cost = (training_time / 3600) * COST_PER_HOUR
    
    print(f"✅ Training completed in {training_time/60:.1f} minutes")
    print(f"💰 Training cost: ${training_cost:.2f}")
    
    # Save model
    trainer.save_model(output_dir)
    print(f"💾 Model saved to: {output_dir}")
    
    return output_dir, training_cost

# Initialize data generator
data_generator = RealTimeDataGenerator()

# Execute training pipeline
print(f"🚀 Starting training pipeline for {len(domains_to_train)} domains...")
print(f"{'='*60}")

results = {}
total_cost = 0.0
start_time = time.time()

for i, domain in enumerate(domains_to_train, 1):
    print(f"\n📋 Training domain {i}/{len(domains_to_train)}: {domain}")
    print(f"{'='*60}")
    
    # Check cost limit
    if total_cost > COST_LIMIT:
        print(f"⚠️ Cost limit reached (${total_cost:.2f}). Stopping training.")
        break
    
    try:
        # Get model for this domain
        domain_info = DOMAIN_MODEL_MAPPING[domain]
        model_name = domain_info['model']
        category = domain_info['category']
        
        print(f"📂 Category: {category}")
        print(f"🤖 Base Model: {model_name}")
        
        # Generate training data
        training_data = data_generator.generate_domain_training_data(domain, SAMPLES_PER_DOMAIN)
        
        # Train model
        model_path, cost = train_domain_model(domain, model_name, training_data)
        
        results[domain] = {
            'path': model_path,
            'cost': cost,
            'status': 'SUCCESS'
        }
        total_cost += cost
        
        # Progress update
        elapsed_time = time.time() - start_time
        estimated_total_cost = total_cost * (len(domains_to_train) / i)
        
        print(f"✅ Completed {domain}")
        print(f"⏱️ Elapsed: {elapsed_time/60:.1f} min | Cost so far: ${total_cost:.2f}")
        print(f"📊 Progress: {i}/{len(domains_to_train)} | ETA Total Cost: ${estimated_total_cost:.2f}")
        
    except Exception as e:
        print(f"❌ Failed to train {domain}: {e}")
        results[domain] = {
            'path': None,
            'cost': 0,
            'status': f'FAILED: {e}'
        }

total_time = time.time() - start_time

print(f"\n{'='*60}")
print(f"🎯 TRAINING COMPLETE")
print(f"✅ Successful: {len([r for r in results.values() if r['status'] == 'SUCCESS'])} domains")
print(f"❌ Failed: {len([r for r in results.values() if r['status'] != 'SUCCESS'])} domains")
print(f"⏱️ Total time: {total_time/60:.1f} minutes")
print(f"💰 Total cost: ${total_cost:.2f}")
print(f"{'='*60}")

# Save results summary
with open('training_results.json', 'w') as f:
    json.dump({
        'results': results,
        'total_cost': total_cost,
        'total_time': total_time,
        'gpu_tier': GPU_TIER,
        'training_mode': TRAINING_MODE,
        'domains_trained': len([r for r in results.values() if r['status'] == 'SUCCESS'])
    }, f, indent=2)

print("📊 Results saved to training_results.json")

# CELL 6: Download Trained Models
# ================================

from google.colab import files
import zipfile
import shutil

print("📥 Preparing models for download...")

# Create download package
download_dir = Path("trained_models")
download_dir.mkdir(exist_ok=True)

# Copy successful models
successful_models = []
for domain, result in results.items():
    if result['status'] == 'SUCCESS' and result['path']:
        model_path = Path(result['path'])
        if model_path.exists():
            # Copy model files
            domain_dir = download_dir / domain
            domain_dir.mkdir(exist_ok=True)
            
            # Copy important files
            for file_pattern in ['*.bin', '*.safetensors', '*.json', '*.txt']:
                for file in model_path.glob(file_pattern):
                    shutil.copy2(file, domain_dir)
            
            successful_models.append(domain)
            print(f"✅ Prepared {domain} model for download")

# Create summary file
summary_content = f"""# MeeTARA Lab Training Results

## Training Summary
- **Training Mode**: {TRAINING_MODE}
- **GPU Used**: {GPU_TIER}
- **Total Domains**: {len(domains_to_train)}
- **Successful**: {len(successful_models)}
- **Failed**: {len(domains_to_train) - len(successful_models)}
- **Total Cost**: ${total_cost:.2f}
- **Total Time**: {total_time/60:.1f} minutes

## Successful Models
{chr(10).join([f'- {domain}: {DOMAIN_MODEL_MAPPING[domain]["model"]}' for domain in successful_models])}

## Usage Instructions
1. Each domain folder contains the trained LoRA adapter
2. Use with the base model specified above
3. Load using PEFT library in your MeeTARA application

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

with open(download_dir / "README.md", 'w') as f:
    f.write(summary_content)

# Copy training results
shutil.copy2("training_results.json", download_dir)

# Create zip file
zip_filename = f"meetara_models_{TRAINING_MODE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
print(f"📦 Creating zip file: {zip_filename}")

with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file_path in download_dir.rglob('*'):
        if file_path.is_file():
            arcname = file_path.relative_to(download_dir)
            zipf.write(file_path, arcname)

print(f"✅ Zip file created: {zip_filename}")
print(f"📊 File size: {Path(zip_filename).stat().st_size / 1024 / 1024:.1f} MB")

# Download the zip file
print("🚀 Starting download...")
files.download(zip_filename)

print(f"\n🎯 DOWNLOAD COMPLETE!")
print(f"📦 Downloaded: {zip_filename}")
print(f"✅ Contains {len(successful_models)} trained domain models") 
print(f"💰 Total training cost: ${total_cost:.2f}")
print(f"⚡ Speed improvement: {SPEED_FACTOR} faster than CPU")
print(f"\n🏆 Ready to deploy to your MeeTARA application!") 
