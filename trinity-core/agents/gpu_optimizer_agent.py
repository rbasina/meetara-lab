"""
MeeTARA Lab - GPU Optimizer Agent
Intelligent GPU resource allocation and performance optimization
"""

import asyncio
import time
import psutil
import torch
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from .mcp_protocol import BaseAgent, AgentType, MessageType, MCPMessage, mcp_protocol

class GPUOptimizerAgent(BaseAgent):
    """Intelligent GPU resource allocation and performance optimization"""
    
    def __init__(self, mcp=None):
        super().__init__(AgentType.GPU_OPTIMIZER, mcp or mcp_protocol)
        self.gpu_configs = {}
        self.current_allocations = {}
        self.performance_history = []
        
    async def start(self):
        """Start the GPU Optimizer Agent"""
        await super().start()
        await self._initialize_gpu_environment()
        asyncio.create_task(self._resource_monitoring_loop())
        print("⚡ GPU Optimizer Agent started")
        
    async def handle_mcp_message(self, message: MCPMessage):
        """Handle incoming MCP messages"""
        if message.message_type == MessageType.COORDINATION_REQUEST:
            await self._handle_coordination_request(message.data)
            
    async def _initialize_gpu_environment(self):
        """Initialize GPU detection and configuration"""
        print("🔍 Detecting GPU environment...")
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ {gpu_count} CUDA GPU(s) detected")
            
            for i in range(gpu_count):
                gpu_props = torch.cuda.get_device_properties(i)
                gpu_name = gpu_props.name
                gpu_memory = gpu_props.total_memory / 1e9
                
                # Classify GPU tier
                if "T4" in gpu_name:
                    tier, cost, batch_size, speed = "T4", 0.40, 16, 37
                elif "V100" in gpu_name:
                    tier, cost, batch_size, speed = "V100", 2.50, 32, 75
                elif "A100" in gpu_name:
                    tier, cost, batch_size, speed = "A100", 4.00, 64, 151
                else:
                    tier, cost, batch_size, speed = "Unknown", 1.00, 8, 10
                
                self.gpu_configs[i] = {
                    "name": gpu_name,
                    "memory_gb": gpu_memory,
                    "tier": tier,
                    "cost_per_hour": cost,
                    "optimal_batch_size": batch_size,
                    "speed_factor": speed,
                    "available": True
                }
                
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB) - {tier} tier")
        else:
            print("⚠️ No CUDA GPUs available - CPU fallback mode")
            self.gpu_configs[0] = {
                "name": "CPU",
                "tier": "CPU",
                "cost_per_hour": 0.0,
                "optimal_batch_size": 2,
                "speed_factor": 1,
                "available": True
            }
            
    async def _resource_monitoring_loop(self):
        """Monitor GPU resources continuously"""
        while self.running:
            try:
                # Broadcast resource status every 15 seconds
                self.broadcast_message(
                    MessageType.RESOURCE_STATUS,
                    {
                        "gpu_configs": self.gpu_configs,
                        "allocations": self.current_allocations,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                await asyncio.sleep(15)
            except Exception as e:
                print(f"❌ GPU monitoring error: {e}")
                await asyncio.sleep(30)
                
    async def _handle_coordination_request(self, data: Dict[str, Any]):
        """Handle resource allocation requests from Training Conductor"""
        action = data.get("action")
        
        if action == "allocate_resources":
            await self._allocate_training_resources(data)
        elif action == "release_resources":
            await self._release_training_resources(data)
            
    async def _allocate_training_resources(self, data: Dict[str, Any]):
        """Allocate GPU resources for domain training"""
        domain = data.get("domain")
        estimated_duration = data.get("estimated_duration", timedelta(hours=1))
        
        print(f"🎯 Allocating resources for domain: {domain}")
        
        # Find best available GPU
        best_gpu = None
        best_score = -1
        
        for gpu_id, config in self.gpu_configs.items():
            if config["available"]:
                score = config["speed_factor"] * (1.0 / max(0.1, config["cost_per_hour"]))
                if score > best_score:
                    best_score = score
                    best_gpu = gpu_id
                    
        if best_gpu is not None:
            # Reserve resources
            self.current_allocations[domain] = {
                "gpu_id": best_gpu,
                "allocated_at": datetime.now(),
                "estimated_duration": estimated_duration,
                "batch_size": self.gpu_configs[best_gpu]["optimal_batch_size"]
            }
            
            self.gpu_configs[best_gpu]["available"] = False
            
            print(f"✅ Resources allocated: GPU {best_gpu} for {domain}")
            
            # Confirm allocation to Training Conductor
            self.send_message(
                AgentType.CONDUCTOR,
                MessageType.RESOURCE_STATUS,
                {
                    "action": "allocation_complete",
                    "domain": domain,
                    "gpu_id": best_gpu,
                    "batch_size": self.gpu_configs[best_gpu]["optimal_batch_size"],
                    "cost_per_hour": self.gpu_configs[best_gpu]["cost_per_hour"]
                }
            )
        else:
            print(f"❌ No GPU resources available for {domain}")
            self.send_message(
                AgentType.CONDUCTOR,
                MessageType.ERROR_NOTIFICATION,
                {
                    "error_type": "resource_unavailable",
                    "domain": domain,
                    "message": "No GPU resources available"
                }
            )
            
    async def _release_training_resources(self, data: Dict[str, Any]):
        """Release resources after training completion"""
        domain = data.get("domain")
        
        if domain in self.current_allocations:
            allocation = self.current_allocations[domain]
            gpu_id = allocation["gpu_id"]
            
            # Release GPU
            self.gpu_configs[gpu_id]["available"] = True
            
            # Calculate final cost
            elapsed_time = datetime.now() - allocation["allocated_at"]
            final_cost = self.gpu_configs[gpu_id]["cost_per_hour"] * (elapsed_time.total_seconds() / 3600)
            
            print(f"🔓 Resources released for {domain}: GPU {gpu_id} (${final_cost:.2f})")
            
            # Remove allocation
            del self.current_allocations[domain]
            
            # Confirm release
            self.send_message(
                AgentType.CONDUCTOR,
                MessageType.RESOURCE_STATUS,
                {
                    "action": "resources_released",
                    "domain": domain,
                    "gpu_id": gpu_id,
                    "final_cost": final_cost
                }
            )

# Global instance
gpu_optimizer_agent = GPUOptimizerAgent() 
