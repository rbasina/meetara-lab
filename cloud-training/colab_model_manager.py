#!/usr/bin/env python3
"""
?? Colab Model Download & Mapping Manager - Quality Focused
Handles intelligent model downloading, caching, and domain mapping for Google Colab
"""

import os
import json
import yaml
import time
import torch
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import subprocess

class ColabModelManager:
    """Manages model downloading and domain mapping for Google Colab"""
    
    def __init__(self, cache_dir: str = "/content/model_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load domain mapping
        self.domain_mapping = self._load_domain_mapping()
        
        # Model download status
        self.downloaded_models = {}
        self.model_sizes = {}
        self.download_times = {}
        
        # Colab-specific optimizations
        self.colab_optimizations = {
            "use_drive_cache": True,      # Cache to Google Drive
            "parallel_downloads": False,   # Sequential to avoid memory issues
            "low_memory_mode": True,      # Memory-efficient loading
            "auto_cleanup": True,         # Clean unused models
            "compression": True           # Use compressed models when available
        }
        
        # GPU memory management
        self.gpu_memory_limit = self._get_gpu_memory_limit()
        self.memory_buffer = 2048  # 2GB buffer for safety
        
        print(f"?? Colab Model Manager initialized")
        print(f"?? Cache directory: {self.cache_dir}")
        print(f"??? GPU memory: {self.gpu_memory_limit:.1f}GB available")
    
    def _load_domain_mapping(self) -> Dict[str, Any]:
        """Load quality-focused domain mapping"""
        possible_paths = [
            "config/trinity_domain_model_mapping_config.yaml",
            "../config/trinity_domain_model_mapping_config.yaml",
            "/content/meetara-lab/config/trinity_domain_model_mapping_config.yaml"
        ]
        
        for path in possible_paths:
            try:
                with open(path, 'r') as f:
                    mapping = yaml.safe_load(f)
                    print(f"? Domain mapping loaded from: {path}")
                    return mapping
            except FileNotFoundError:
                continue
        
        print("?? Domain mapping not found, using minimal configuration")
        return self._get_minimal_mapping()
    
    def _get_minimal_mapping(self) -> Dict[str, Any]:
        """Minimal mapping for testing"""
        return {
            "model_tiers": {
                "fast": "microsoft/Phi-3.5-mini-instruct",
                "quality": "microsoft/Phi-3-medium-4k-instruct",
                "expert": "Qwen/Qwen2.5-14B-Instruct",
                "premium": "microsoft/Phi-3-medium-14B-instruct"
            },
            "healthcare": {"general_health": "microsoft/Phi-3-medium-14B-instruct"},
            "business": {"entrepreneurship": "Qwen/Qwen2.5-14B-Instruct"},
            "education": {"academic_tutoring": "Qwen/Qwen2.5-14B-Instruct"}
        }
    
    def _get_gpu_memory_limit(self) -> float:
        """Get available GPU memory in GB"""
        try:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                return gpu_memory / (1024**3)  # Convert to GB
            else:
                return 0.0
        except Exception:
            return 12.0  # Default assumption for Colab
    
    def get_domain_model_mapping(self) -> Dict[str, str]:
        """Get complete domain to model mapping"""
        domain_model_mapping = {}
        
        # Process each category dynamically from domain mapping
        categories = list(self.domain_mapping.keys())
        
        for category in categories:
            category_domains = self.domain_mapping.get(category, {})
            for domain, model in category_domains.items():
                domain_model_mapping[domain] = model
        
        return domain_model_mapping
    
    def get_unique_models(self) -> List[str]:
        """Get list of unique models needed for all domains"""
        domain_model_mapping = self.get_domain_model_mapping()
        unique_models = list(set(domain_model_mapping.values()))
        
        print(f"?? UNIQUE MODELS NEEDED FOR ALL DOMAINS:")
        for i, model in enumerate(unique_models, 1):
            domains_using = [d for d, m in domain_model_mapping.items() if m == model]
            print(f"   {i}. {model}")
            print(f"      ? Used by {len(domains_using)} domains: {domains_using[:3]}{'...' if len(domains_using) > 3 else ''}")
        
        return unique_models
    
    def estimate_download_requirements(self) -> Dict[str, Any]:
        """Estimate download time and storage requirements"""
        unique_models = self.get_unique_models()
        
        # Model size estimates (GB)
        model_size_estimates = {
            "HuggingFaceTB/SmolLM2-1.7B": 3.5,
            "microsoft/Phi-3.5-mini-instruct": 7.5,
            "microsoft/Phi-3-medium-4k-instruct": 28.0,
            "microsoft/Phi-3-medium-14B-instruct": 28.0,
            "Qwen/Qwen2.5-7B-Instruct": 14.5,
            "Qwen/Qwen2.5-14B-Instruct": 29.0
        }
        
        # Download speed estimates (MB/s)
        colab_download_speed = 50  # Conservative estimate for Colab
        
        total_size_gb = 0
        total_download_time = 0
        
        download_plan = []
        
        for model in unique_models:
            size_gb = model_size_estimates.get(model, 15.0)  # Default 15GB
            download_time_minutes = (size_gb * 1024) / (colab_download_speed * 60)
            
            total_size_gb += size_gb
            total_download_time += download_time_minutes
            
            download_plan.append({
                "model": model,
                "size_gb": size_gb,
                "download_time_minutes": download_time_minutes,
                "memory_fit": size_gb < (self.gpu_memory_limit - self.memory_buffer/1024)
            })
        
        return {
            "unique_models": len(unique_models),
            "total_size_gb": total_size_gb,
            "total_download_time_minutes": total_download_time,
            "total_download_time_hours": total_download_time / 60,
            "download_plan": download_plan,
            "storage_recommendation": self._get_storage_recommendation(total_size_gb),
            "memory_strategy": self._get_memory_strategy(download_plan)
        }
    
    def _get_storage_recommendation(self, total_size_gb: float) -> str:
        """Get storage recommendation"""
        if total_size_gb < 15:
            return "? Colab default storage sufficient"
        elif total_size_gb < 50:
            return "?? Consider using Google Drive for caching"
        else:
            return "?? Requires Google Drive Pro for large model cache"
    
    def _get_memory_strategy(self, download_plan: List[Dict]) -> str:
        """Get memory management strategy"""
        large_models = [p for p in download_plan if not p["memory_fit"]]
        
        if not large_models:
            return "? All models fit in GPU memory"
        elif len(large_models) <= 2:
            return "?? Sequential loading required for large models"
        else:
            return "?? Memory optimization required - consider model quantization"
    
    async def download_model_for_domain(self, domain: str) -> Tuple[str, bool]:
        """Download specific model for a domain"""
        domain_model_mapping = self.get_domain_model_mapping()
        
        if domain not in domain_model_mapping:
            print(f"? Domain '{domain}' not found in mapping")
            return "", False
        
        model_name = domain_model_mapping[domain]
        
        print(f"?? Downloading model for domain '{domain}': {model_name}")
        
        # Check if already downloaded
        if model_name in self.downloaded_models:
            print(f"? Model already cached: {model_name}")
            return model_name, True
        
        try:
            # Download model using transformers
            download_start = time.time()
            
            print(f"? Starting download: {model_name}")
            
            # Use HuggingFace hub for downloading
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Download tokenizer first (smaller)
            print("   ?? Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir / "tokenizers"),
                trust_remote_code=True
            )
            
            # Download model
            print("   ?? Downloading model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir / "models"),
                torch_dtype=torch.float16,  # Use FP16 for memory efficiency
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            download_time = time.time() - download_start
            
            # Track download
            self.downloaded_models[model_name] = True
            self.download_times[model_name] = download_time
            
            print(f"? Model downloaded successfully: {model_name}")
            print(f"   ?? Download time: {download_time/60:.1f} minutes")
            
            return model_name, True
            
        except Exception as e:
            print(f"? Failed to download model {model_name}: {e}")
            return model_name, False
    
    def create_colab_training_sequence(self, target_domains: List[str] = None) -> Dict[str, Any]:
        """Create optimized training sequence for Colab"""
        
        if target_domains is None:
            # Get all domains
            domain_model_mapping = self.get_domain_model_mapping()
            target_domains = list(domain_model_mapping.keys())
        
        # Group domains by model for efficient batch processing
        model_domain_groups = {}
        domain_model_mapping = self.get_domain_model_mapping()
        
        for domain in target_domains:
            if domain in domain_model_mapping:
                model = domain_model_mapping[domain]
                if model not in model_domain_groups:
                    model_domain_groups[model] = []
                model_domain_groups[model].append(domain)
        
        # Create training sequence (one model at a time for memory efficiency)
        training_sequence = []
        
        for model, domains in model_domain_groups.items():
            training_sequence.append({
                "model": model,
                "domains": domains,
                "domain_count": len(domains),
                "estimated_download_time": self._estimate_model_download_time(model),
                "estimated_training_time": len(domains) * 10,  # 10 min per domain
                "memory_strategy": "sequential_unload"
            })
        
        # Sort by model size (smallest first for faster initial results)
        training_sequence.sort(key=lambda x: self._get_model_size_estimate(x["model"]))
        
        return {
            "total_models": len(model_domain_groups),
            "total_domains": len(target_domains),
            "training_sequence": training_sequence,
            "estimated_total_time_hours": sum(seq["estimated_download_time"] + seq["estimated_training_time"] for seq in training_sequence) / 60,
            "colab_optimization": "? Optimized for Colab memory and session limits"
        }
    
    def _estimate_model_download_time(self, model_name: str) -> float:
        """Estimate download time in minutes"""
        size_gb = self._get_model_size_estimate(model_name)
        return (size_gb * 1024) / (50 * 60)  # 50 MB/s download speed
    
    def _get_model_size_estimate(self, model_name: str) -> float:
        """Get model size estimate in GB"""
        size_estimates = {
            "HuggingFaceTB/SmolLM2-1.7B": 3.5,
            "microsoft/Phi-3.5-mini-instruct": 7.5,
            "microsoft/Phi-3-medium-4k-instruct": 28.0,
            "microsoft/Phi-3-medium-14B-instruct": 28.0,
            "Qwen/Qwen2.5-7B-Instruct": 14.5,
            "Qwen/Qwen2.5-14B-Instruct": 29.0
        }
        return size_estimates.get(model_name, 15.0)
    
    def print_colab_strategy(self):
        """Print complete Colab download and training strategy"""
        print("?? COLAB MODEL DOWNLOAD & TRAINING STRATEGY")
        print("=" * 70)
        
        # Model requirements
        requirements = self.estimate_download_requirements()
        print(f"?? MODEL REQUIREMENTS:")
        print(f"   Unique models needed: {requirements['unique_models']}")
        print(f"   Total download size: {requirements['total_size_gb']:.1f}GB")
        print(f"   Total download time: {requirements['total_download_time_hours']:.1f} hours")
        print(f"   Storage: {requirements['storage_recommendation']}")
        print(f"   Memory: {requirements['memory_strategy']}")
        print()
        
        # Training sequence
        sequence = self.create_colab_training_sequence()
        print(f"? OPTIMIZED TRAINING SEQUENCE:")
        print(f"   Total models to process: {sequence['total_models']}")
        print(f"   Total domains to train: {sequence['total_domains']}")
        print(f"   Estimated total time: {sequence['estimated_total_time_hours']:.1f} hours")
        print()
        
        for i, step in enumerate(sequence['training_sequence'], 1):
            print(f"   Step {i}: {step['model']}")
            print(f"      ? Domains: {step['domain_count']} ({step['domains'][:2]}{'...' if len(step['domains']) > 2 else ''})")
            print(f"      ? Download: {step['estimated_download_time']:.1f}min + Training: {step['estimated_training_time']:.1f}min")
        print()
        
        # Colab-specific recommendations
        print("?? COLAB OPTIMIZATION RECOMMENDATIONS:")
        print("   1. ? Mount Google Drive for model caching")
        print("   2. ? Use Colab Pro+ for A100 GPU access")
        print("   3. ? Enable background execution to prevent timeouts")
        print("   4. ? Process one model at a time to avoid memory issues")
        print("   5. ? Save checkpoints frequently")
        print("   6. ? Use gradient checkpointing for large models")

# Demo usage
async def demo_colab_strategy():
    """Demo the Colab model management strategy"""
    manager = ColabModelManager()
    
    # Show complete strategy
    manager.print_colab_strategy()
    
    # Test domain-specific download
    print("\n?? TESTING DOMAIN-SPECIFIC DOWNLOAD:")
    test_domains = ['general_health', 'entrepreneurship', 'academic_tutoring']
    
    for domain in test_domains:
        print(f"\n?? Testing download for domain: {domain}")
        # Note: Actual download commented out for demo
        # model, success = await manager.download_model_for_domain(domain)
        domain_mapping = manager.get_domain_model_mapping()
        model = domain_mapping.get(domain, "Unknown")
        print(f"   Model required: {model}")
        print(f"   Size estimate: {manager._get_model_size_estimate(model):.1f}GB")

if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_colab_strategy())
