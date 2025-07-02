"""
MeeTARA Lab - GPU Orchestrator
Multi-cloud GPU coordination with cost optimization
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import requests
import os

# Import trinity-core agents
import sys
sys.path.append('../trinity-core')
from agents.mcp_protocol import BaseAgent, AgentType, MessageType, MCPMessage, mcp_protocol

class CloudProvider(Enum):
    GOOGLE_COLAB = "google_colab"
    LAMBDA_LABS = "lambda_labs"
    RUNPOD = "runpod"
    VAST_AI = "vast_ai"
    PAPERSPACE = "paperspace"

class GPUType(Enum):
    T4 = "T4"
    V100 = "V100"
    A100 = "A100"
    RTX_4090 = "RTX_4090"
    H100 = "H100"

@dataclass
class GPUInstance:
    provider: CloudProvider
    instance_id: str
    gpu_type: GPUType
    gpu_count: int
    cost_per_hour: float
    region: str
    status: str
    utilization: float = 0.0
    memory_used: float = 0.0
    memory_total: float = 0.0

@dataclass
class TrainingJob:
    domain: str
    estimated_duration: timedelta
    gpu_requirements: GPUType
    priority: int
    cost_limit: float
    assigned_instance: Optional[str] = None
    start_time: Optional[datetime] = None
    status: str = "pending"

class GPUOrchestratorAgent(BaseAgent):
    """GPU Orchestrator for multi-cloud training coordination"""
    
    def __init__(self, mcp=None):
        super().__init__(AgentType.GPU_OPTIMIZER, mcp or mcp_protocol)
        self.active_instances: Dict[str, GPUInstance] = {}
        self.training_jobs: Dict[str, TrainingJob] = {}
        self.cost_tracking: Dict[str, float] = {
            "daily_spend": 0.0,
            "weekly_spend": 0.0,
            "monthly_spend": 0.0
        }
        self.provider_configs: Dict[CloudProvider, Dict[str, Any]] = {}
        self.optimization_strategies: Dict[str, bool] = {
            "use_spot_instances": True,
            "auto_scaling": True,
            "cost_optimization": True,
            "performance_priority": False,
            "regional_optimization": True
        }
        
    async def start(self):
        """Start the GPU Orchestrator Agent"""
        await super().start()
        
        # Initialize provider configurations
        await self._initialize_providers()
        
        # Start monitoring loops
        asyncio.create_task(self._monitor_instances())
        asyncio.create_task(self._optimize_costs())
        asyncio.create_task(self._manage_training_jobs())
        
        print("⚡ GPU Orchestrator ready for multi-cloud coordination")
        
    async def handle_mcp_message(self, message: MCPMessage):
        """Handle incoming MCP messages"""
        if message.message_type == MessageType.COORDINATION_REQUEST:
            await self._handle_coordination_request(message.data)
        elif message.message_type == MessageType.OPTIMIZATION_REQUEST:
            await self._handle_optimization_request(message.data)
        elif message.message_type == MessageType.STATUS_UPDATE:
            await self._handle_status_update(message.data)
            
    async def _initialize_providers(self):
        """Initialize cloud provider configurations"""
        self.provider_configs = {
            CloudProvider.GOOGLE_COLAB: {
                "api_endpoint": "https://colab.googleapis.com/api/v1",
                "gpu_types": [GPUType.T4, GPUType.V100, GPUType.A100],
                "cost_per_hour": {
                    GPUType.T4: 0.35,
                    GPUType.V100: 0.74,
                    GPUType.A100: 1.28
                },
                "max_session_hours": 12,
                "spot_available": False,
                "regions": ["us-central1", "europe-west1"]
            },
            CloudProvider.LAMBDA_LABS: {
                "api_endpoint": "https://cloud.lambdalabs.com/api/v1",
                "gpu_types": [GPUType.RTX_4090, GPUType.A100, GPUType.H100],
                "cost_per_hour": {
                    GPUType.RTX_4090: 0.50,
                    GPUType.A100: 1.10,
                    GPUType.H100: 1.99
                },
                "max_session_hours": 24,
                "spot_available": True,
                "regions": ["us-west-1", "us-east-1", "europe-west-1"]
            },
            CloudProvider.RUNPOD: {
                "api_endpoint": "https://api.runpod.io/graphql",
                "gpu_types": [GPUType.RTX_4090, GPUType.A100],
                "cost_per_hour": {
                    GPUType.RTX_4090: 0.39,
                    GPUType.A100: 0.79
                },
                "max_session_hours": 24,
                "spot_available": True,
                "regions": ["us-east", "us-west", "eu-central"]
            },
            CloudProvider.VAST_AI: {
                "api_endpoint": "https://vast.ai/api/v0",
                "gpu_types": [GPUType.RTX_4090, GPUType.A100],
                "cost_per_hour": {
                    GPUType.RTX_4090: 0.29,
                    GPUType.A100: 0.59
                },
                "max_session_hours": 48,
                "spot_available": True,
                "regions": ["worldwide"]
            }
        }
        
        print(f"☁️ Initialized {len(self.provider_configs)} cloud providers")
        
    async def allocate_gpu_for_training(self, domain: str, requirements: Dict[str, Any]) -> Optional[str]:
        """Allocate optimal GPU instance for training"""
        try:
            # Create training job
            job = TrainingJob(
                domain=domain,
                estimated_duration=timedelta(hours=requirements.get("estimated_hours", 2)),
                gpu_requirements=GPUType(requirements.get("gpu_type", "T4")),
                priority=requirements.get("priority", 1),
                cost_limit=requirements.get("cost_limit", 50.0)
            )
            
            # Find optimal instance
            best_instance = await self._find_optimal_instance(job)
            
            if best_instance:
                # Launch instance
                instance_id = await self._launch_instance(best_instance, job)
                
                if instance_id:
                    job.assigned_instance = instance_id
                    job.start_time = datetime.now()
                    job.status = "running"
                    self.training_jobs[domain] = job
                    
                    # Update context
                    self.update_context({
                        "gpu_utilization": self._get_gpu_utilization(),
                        "cost_tracking": self.cost_tracking
                    })
                    
                    print(f"🚀 GPU allocated for {domain}: {instance_id}")
                    return instance_id
                    
        except Exception as e:
            print(f"❌ Failed to allocate GPU for {domain}: {e}")
            
        return None
        
    async def _find_optimal_instance(self, job: TrainingJob) -> Optional[Dict[str, Any]]:
        """Find the optimal GPU instance for a training job"""
        candidates = []
        
        for provider, config in self.provider_configs.items():
            if job.gpu_requirements in config["gpu_types"]:
                cost_per_hour = config["cost_per_hour"][job.gpu_requirements]
                total_cost = cost_per_hour * job.estimated_duration.total_seconds() / 3600
                
                if total_cost <= job.cost_limit:
                    candidates.append({
                        "provider": provider,
                        "gpu_type": job.gpu_requirements,
                        "cost_per_hour": cost_per_hour,
                        "total_cost": total_cost,
                        "config": config,
                        "score": self._calculate_instance_score(provider, job, cost_per_hour)
                    })
                    
        # Sort by optimization score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        return candidates[0] if candidates else None
        
    def _calculate_instance_score(self, provider: CloudProvider, job: TrainingJob, cost_per_hour: float) -> float:
        """Calculate optimization score for instance selection"""
        score = 100.0
        
        # Cost optimization (40% weight)
        cost_factor = 1.0 - (cost_per_hour / 2.0)  # Normalize against $2/hour max
        score += cost_factor * 40
        
        # Performance optimization (30% weight)
        if job.gpu_requirements == GPUType.A100:
            score += 30
        elif job.gpu_requirements == GPUType.V100:
            score += 25
        elif job.gpu_requirements == GPUType.T4:
            score += 20
            
        # Provider reliability (20% weight)
        reliability_scores = {
            CloudProvider.GOOGLE_COLAB: 25,
            CloudProvider.LAMBDA_LABS: 22,
            CloudProvider.RUNPOD: 20,
            CloudProvider.VAST_AI: 18
        }
        score += reliability_scores.get(provider, 15)
        
        # Priority adjustment (10% weight)
        if job.priority >= 2:
            score += 10
            
        return score
        
    async def _launch_instance(self, instance_config: Dict[str, Any], job: TrainingJob) -> Optional[str]:
        """Launch a GPU instance"""
        provider = instance_config["provider"]
        
        try:
            if provider == CloudProvider.GOOGLE_COLAB:
                return await self._launch_colab_instance(instance_config, job)
            elif provider == CloudProvider.LAMBDA_LABS:
                return await self._launch_lambda_instance(instance_config, job)
            elif provider == CloudProvider.RUNPOD:
                return await self._launch_runpod_instance(instance_config, job)
            elif provider == CloudProvider.VAST_AI:
                return await self._launch_vast_instance(instance_config, job)
                
        except Exception as e:
            print(f"❌ Failed to launch {provider.value} instance: {e}")
            
        return None
        
    async def _launch_colab_instance(self, config: Dict[str, Any], job: TrainingJob) -> Optional[str]:
        """Launch Google Colab Pro+ instance"""
        # For Colab, we'll use the notebook-based approach
        instance_id = f"colab_{job.domain}_{int(time.time())}"
        
        instance = GPUInstance(
            provider=CloudProvider.GOOGLE_COLAB,
            instance_id=instance_id,
            gpu_type=job.gpu_requirements,
            gpu_count=1,
            cost_per_hour=config["cost_per_hour"],
            region="us-central1",
            status="launching"
        )
        
        self.active_instances[instance_id] = instance
        
        # Create Colab notebook for this training job
        await self._create_colab_notebook(instance_id, job)
        
        instance.status = "running"
        return instance_id
        
    async def _launch_lambda_instance(self, config: Dict[str, Any], job: TrainingJob) -> Optional[str]:
        """Launch Lambda Labs instance"""
        # Simulate Lambda Labs API call
        instance_id = f"lambda_{job.domain}_{int(time.time())}"
        
        instance = GPUInstance(
            provider=CloudProvider.LAMBDA_LABS,
            instance_id=instance_id,
            gpu_type=job.gpu_requirements,
            gpu_count=1,
            cost_per_hour=config["cost_per_hour"],
            region="us-west-1",
            status="running"
        )
        
        self.active_instances[instance_id] = instance
        return instance_id
        
    async def _launch_runpod_instance(self, config: Dict[str, Any], job: TrainingJob) -> Optional[str]:
        """Launch RunPod instance"""
        # Simulate RunPod API call
        instance_id = f"runpod_{job.domain}_{int(time.time())}"
        
        instance = GPUInstance(
            provider=CloudProvider.RUNPOD,
            instance_id=instance_id,
            gpu_type=job.gpu_requirements,
            gpu_count=1,
            cost_per_hour=config["cost_per_hour"],
            region="us-east",
            status="running"
        )
        
        self.active_instances[instance_id] = instance
        return instance_id
        
    async def _launch_vast_instance(self, config: Dict[str, Any], job: TrainingJob) -> Optional[str]:
        """Launch Vast.ai instance"""
        # Simulate Vast.ai API call
        instance_id = f"vast_{job.domain}_{int(time.time())}"
        
        instance = GPUInstance(
            provider=CloudProvider.VAST_AI,
            instance_id=instance_id,
            gpu_type=job.gpu_requirements,
            gpu_count=1,
            cost_per_hour=config["cost_per_hour"],
            region="worldwide",
            status="running"
        )
        
        self.active_instances[instance_id] = instance
        return instance_id
        
    async def _create_colab_notebook(self, instance_id: str, job: TrainingJob):
        """Create optimized Colab notebook for training"""
        notebook_content = self._generate_training_notebook(job)
        
        # Save notebook for the specific training job
        notebook_path = f"../notebooks/colab_{job.domain}_training.ipynb"
        
        with open(notebook_path, 'w') as f:
            json.dump(notebook_content, f, indent=2)
            
        print(f"📓 Created Colab notebook: {notebook_path}")
        
    def _generate_training_notebook(self, job: TrainingJob) -> Dict[str, Any]:
        """Generate optimized training notebook"""
        return {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        f"# MeeTARA Lab - {job.domain.title()} Domain Training\\n",
                        f"**GPU Optimized Training Pipeline**\\n",
                        f"- Domain: {job.domain}\\n",
                        f"- GPU Type: {job.gpu_requirements.value}\\n",
                        f"- Estimated Duration: {job.estimated_duration}\\n",
                        f"- Cost Limit: ${job.cost_limit}"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Install optimized dependencies\\n",
                        "!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\\n",
                        "!pip install transformers datasets accelerate\\n",
                        "!pip install peft bitsandbytes\\n",
                        "!pip install wandb tensorboard\\n",
                        "\\n",
                        "# Enable GPU optimization\\n",
                        "import torch\\n",
                        "print(f'GPU Available: {torch.cuda.is_available()}')\\n",
                        "print(f'GPU Count: {torch.cuda.device_count()}')\\n",
                        "print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Clone MeeTARA Lab training pipeline\\n",
                        "!git clone https://github.com/meetara-ai/meetara-lab.git\\n",
                        "%cd meetara-lab\\n",
                        "\\n",
                        "# Set up environment\\n",
                        "import os\\n",
                        "os.environ['CUDA_VISIBLE_DEVICES'] = '0'\\n",
                        "os.environ['TOKENIZERS_PARALLELISM'] = 'false'"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        f"# Start optimized training for {job.domain}\\n",
                        f"!python trinity-core/training/gpu_optimized_pipeline.py \\\\\\n",
                        f"  --domain {job.domain} \\\\\\n",
                        f"  --gpu_type {job.gpu_requirements.value} \\\\\\n",
                        f"  --batch_size_auto \\\\\\n",
                        f"  --mixed_precision \\\\\\n",
                        f"  --gradient_checkpointing \\\\\\n",
                        f"  --cost_limit {job.cost_limit} \\\\\\n",
                        f"  --auto_shutdown"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.10.0"
                },
                "accelerator": "GPU",
                "gpuClass": job.gpu_requirements.value
            },
            "nbformat": 4,
            "nbformat_minor": 0
        }
        
    async def _monitor_instances(self):
        """Monitor active GPU instances"""
        while self.running:
            try:
                for instance_id, instance in self.active_instances.items():
                    # Simulate monitoring
                    instance.utilization = await self._get_instance_utilization(instance_id)
                    instance.memory_used, instance.memory_total = await self._get_instance_memory(instance_id)
                    
                    # Update cost tracking
                    if instance.status == "running":
                        hourly_cost = instance.cost_per_hour / 60  # Per minute
                        self.cost_tracking["daily_spend"] += hourly_cost
                        
                # Send status update
                self.broadcast_message(
                    MessageType.RESOURCE_STATUS,
                    {
                        "gpu_utilization": self._get_gpu_utilization(),
                        "memory_usage": self._get_memory_usage(),
                        "cost_tracking": self.cost_tracking,
                        "active_instances": len(self.active_instances)
                    }
                )
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                print(f"❌ Instance monitoring error: {e}")
                await asyncio.sleep(60)
                
    async def _optimize_costs(self):
        """Optimize costs across all instances"""
        while self.running:
            try:
                # Check daily spending limit
                if self.cost_tracking["daily_spend"] > 45:  # $45 warning threshold
                    await self._implement_cost_reduction()
                    
                # Optimize instance allocation
                await self._optimize_instance_allocation()
                
                # Check for idle instances
                await self._check_idle_instances()
                
                await asyncio.sleep(300)  # Optimize every 5 minutes
                
            except Exception as e:
                print(f"❌ Cost optimization error: {e}")
                await asyncio.sleep(300)
                
    async def _manage_training_jobs(self):
        """Manage training job lifecycle"""
        while self.running:
            try:
                for domain, job in list(self.training_jobs.items()):
                    if job.status == "running" and job.start_time:
                        # Check if training should complete
                        elapsed = datetime.now() - job.start_time
                        if elapsed >= job.estimated_duration * 1.2:  # 20% buffer
                            await self._check_training_completion(domain, job)
                            
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                print(f"❌ Training job management error: {e}")
                await asyncio.sleep(60)
                
    async def _get_instance_utilization(self, instance_id: str) -> float:
        """Get GPU utilization for instance"""
        # Simulate GPU utilization monitoring
        import random
        return random.uniform(70, 95)  # Simulate high utilization
        
    async def _get_instance_memory(self, instance_id: str) -> Tuple[float, float]:
        """Get memory usage for instance"""
        # Simulate memory monitoring
        import random
        total = random.uniform(15, 80)  # GB
        used = total * random.uniform(0.6, 0.9)
        return used, total
        
    def _get_gpu_utilization(self) -> Dict[str, float]:
        """Get GPU utilization across all instances"""
        return {
            instance_id: instance.utilization 
            for instance_id, instance in self.active_instances.items()
        }
        
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage across all instances"""
        return {
            instance_id: instance.memory_used / instance.memory_total * 100
            for instance_id, instance in self.active_instances.items()
            if instance.memory_total > 0
        }
        
    async def _implement_cost_reduction(self):
        """Implement cost reduction strategies"""
        print("💰 Implementing cost reduction strategies")
        
        # Switch to cheaper instances
        for instance_id, instance in list(self.active_instances.items()):
            if instance.cost_per_hour > 1.0:  # Expensive instances
                await self._migrate_to_cheaper_instance(instance_id)
                
        # Enable more aggressive spot instance usage
        self.optimization_strategies["use_spot_instances"] = True
        
    async def _optimize_instance_allocation(self):
        """Optimize instance allocation across providers"""
        # Consolidate workloads where possible
        underutilized = [
            instance_id for instance_id, instance in self.active_instances.items()
            if instance.utilization < 50
        ]
        
        if len(underutilized) > 1:
            print(f"🔄 Consolidating {len(underutilized)} underutilized instances")
            # Implement consolidation logic
            
    async def _check_idle_instances(self):
        """Check for and terminate idle instances"""
        for instance_id, instance in list(self.active_instances.items()):
            if instance.utilization < 10:  # Very low utilization
                print(f"🛑 Terminating idle instance: {instance_id}")
                await self._terminate_instance(instance_id)
                
    async def _terminate_instance(self, instance_id: str):
        """Terminate a GPU instance"""
        if instance_id in self.active_instances:
            instance = self.active_instances[instance_id]
            instance.status = "terminating"
            
            # Provider-specific termination
            if instance.provider == CloudProvider.GOOGLE_COLAB:
                # Colab sessions auto-terminate
                pass
            # Add other provider termination logic
            
            del self.active_instances[instance_id]
            print(f"✅ Instance terminated: {instance_id}")
            
    async def _check_training_completion(self, domain: str, job: TrainingJob):
        """Check if training has completed"""
        # Send completion check to conductor
        self.send_message(
            AgentType.CONDUCTOR,
            MessageType.STATUS_UPDATE,
            {
                "action": "check_training_completion",
                "domain": domain,
                "elapsed_time": datetime.now() - job.start_time if job.start_time else None
            }
        )
        
    async def _migrate_to_cheaper_instance(self, instance_id: str):
        """Migrate training to a cheaper instance"""
        print(f"💸 Migrating to cheaper instance: {instance_id}")
        # Implementation would depend on training checkpoint capabilities
        
    async def _handle_coordination_request(self, data: Dict[str, Any]):
        """Handle coordination requests"""
        action = data.get("action", "")
        
        if action == "allocate_resources":
            domain = data.get("domain")
            estimated_duration = data.get("estimated_duration")
            
            # Allocate GPU for training
            instance_id = await self.allocate_gpu_for_training(domain, {
                "estimated_hours": estimated_duration.total_seconds() / 3600 if estimated_duration else 2,
                "gpu_type": "A100" if domain in ["healthcare", "mental_health"] else "T4",
                "priority": 2 if domain in ["healthcare", "mental_health"] else 1,
                "cost_limit": 50.0
            })
            
            if instance_id:
                self.send_message(
                    AgentType.CONDUCTOR,
                    MessageType.STATUS_UPDATE,
                    {
                        "action": "resource_allocated",
                        "domain": domain,
                        "instance_id": instance_id
                    }
                )
                
        elif action == "emergency_scaling":
            # Implement emergency scaling
            await self._emergency_scale_resources(data)
            
    async def _handle_optimization_request(self, data: Dict[str, Any]):
        """Handle optimization requests"""
        strategy = data.get("strategy", "")
        
        if strategy == "scale_up_resources":
            await self._scale_up_resources(data)
        elif strategy == "optimize_costs":
            await self._implement_cost_reduction()
        elif strategy == "reduce_memory_usage":
            await self._optimize_memory_usage(data)
            
    async def _handle_status_update(self, data: Dict[str, Any]):
        """Handle status updates"""
        # Process any relevant status updates
        pass
        
    async def _emergency_scale_resources(self, data: Dict[str, Any]):
        """Emergency resource scaling"""
        domain = data.get("domain")
        print(f"🚨 Emergency scaling for {domain}")
        
        # Launch additional instances immediately
        await self.allocate_gpu_for_training(domain, {
            "estimated_hours": 4,
            "gpu_type": "A100",
            "priority": 3,
            "cost_limit": 100.0  # Higher limit for emergency
        })
        
    async def _scale_up_resources(self, data: Dict[str, Any]):
        """Scale up resources"""
        current_utilization = data.get("current_utilization", {})
        
        # Launch additional instances if utilization is high
        if any(util > 90 for util in current_utilization.values()):
            print("📈 Scaling up resources due to high utilization")
            # Implementation would launch additional instances
            
    async def _optimize_memory_usage(self, data: Dict[str, Any]):
        """Optimize memory usage"""
        domain = data.get("domain")
        suggested_batch_size = data.get("suggested_batch_size", 4)
        
        print(f"🧠 Optimizing memory usage for {domain}: batch_size={suggested_batch_size}")
        
        # Send optimization parameters to training
        self.send_message(
            AgentType.CONDUCTOR,
            MessageType.OPTIMIZATION_REQUEST,
            {
                "action": "update_training_parameters",
                "domain": domain,
                "batch_size": suggested_batch_size,
                "gradient_accumulation": True,
                "memory_efficient": True
            }
        )

# Global GPU orchestrator instance
gpu_orchestrator = GPUOrchestratorAgent() 