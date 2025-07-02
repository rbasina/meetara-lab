"""
MeeTARA Lab - Training Conductor Agent
Master orchestrator of the entire training pipeline
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from .mcp_protocol import BaseAgent, AgentType, MessageType, MCPMessage, mcp_protocol

class TrainingConductorAgent(BaseAgent):
    """Master orchestrator of the entire training pipeline"""
    
    def __init__(self, mcp=None):
        super().__init__(AgentType.CONDUCTOR, mcp or mcp_protocol)
        self.training_queue: List[str] = []
        self.current_training: Optional[str] = None
        self.training_start_time: Optional[datetime] = None
        self.training_stats: Dict[str, Any] = {}
        self.coordination_strategies: Dict[str, Any] = {}
        
    async def start(self):
        """Start the Training Conductor Agent"""
        await super().start()
        
        # Initialize coordination strategies
        self.coordination_strategies = {
            "sequential_training": True,  # Start with proven approach
            "parallel_domains": False,    # Enable when resources allow
            "smart_scheduling": True,     # Optimize training order
            "cost_optimization": True,    # Enable cost intelligence
            "quality_first": True         # Prioritize quality over speed
        }
        
        # Start orchestration loop
        asyncio.create_task(self._orchestration_loop())
        
    async def handle_mcp_message(self, message: MCPMessage):
        """Handle incoming MCP messages"""
        if message.message_type == MessageType.RESOURCE_STATUS:
            await self._handle_resource_status(message.data)
        elif message.message_type == MessageType.QUALITY_METRICS:
            await self._handle_quality_metrics(message.data)
        elif message.message_type == MessageType.TRAINING_PROGRESS:
            await self._handle_training_progress(message.data)
        elif message.message_type == MessageType.ERROR_NOTIFICATION:
            await self._handle_error_notification(message.data)
        elif message.message_type == MessageType.OPTIMIZATION_REQUEST:
            await self._handle_optimization_request(message.data)
            
    async def _orchestration_loop(self):
        """Main orchestration loop"""
        while self.running:
            try:
                # Check if we need to start new training
                if not self.current_training and self.training_queue:
                    await self._start_next_training()
                
                # Monitor current training
                if self.current_training:
                    await self._monitor_current_training()
                
                # Optimize resource allocation
                await self._optimize_resources()
                
                # Update coordination strategies
                await self._update_strategies()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                print(f"❌ Conductor orchestration error: {e}")
                await asyncio.sleep(30)
                
    async def queue_domain_training(self, domain: str, priority: int = 1):
        """Queue a domain for training"""
        if domain not in self.training_queue:
            if priority == 3:  # High priority
                self.training_queue.insert(0, domain)
            else:
                self.training_queue.append(domain)
                
        print(f"📋 Domain queued for training: {domain} (queue length: {len(self.training_queue)})")
        
        # Notify other agents
        self.broadcast_message(
            MessageType.STATUS_UPDATE,
            {
                "action": "domain_queued",
                "domain": domain,
                "queue_length": len(self.training_queue),
                "priority": priority
            }
        )
        
    async def queue_batch_training(self, domains: List[str], batch_name: str = "custom"):
        """Queue multiple domains for batch training"""
        for domain in domains:
            await self.queue_domain_training(domain)
            
        print(f"📋 Batch queued: {batch_name} ({len(domains)} domains)")
        
        # Optimize training order based on knowledge transfer
        await self._optimize_training_order()
        
    async def _start_next_training(self):
        """Start training the next domain in queue"""
        if not self.training_queue:
            return
            
        domain = self.training_queue.pop(0)
        self.current_training = domain
        self.training_start_time = datetime.now()
        
        print(f"🚀 Starting training: {domain}")
        
        # Update context
        self.update_context({
            "current_domain": domain,
            "training_step": 0,
            "progress_percentage": 0.0
        })
        
        # Request resource allocation
        self.send_message(
            AgentType.GPU_OPTIMIZER,
            MessageType.COORDINATION_REQUEST,
            {
                "action": "allocate_resources",
                "domain": domain,
                "estimated_duration": self._estimate_training_duration(domain)
            }
        )
        
        # Request data preparation
        self.send_message(
            AgentType.DATA_GENERATOR,
            MessageType.COORDINATION_REQUEST,
            {
                "action": "prepare_training_data",
                "domain": domain,
                "quality_requirements": self._get_quality_requirements(domain)
            }
        )
        
        # Start quality monitoring
        self.send_message(
            AgentType.QUALITY_ASSURANCE,
            MessageType.COORDINATION_REQUEST,
            {
                "action": "start_monitoring",
                "domain": domain,
                "quality_thresholds": self._get_quality_thresholds(domain)
            }
        )
        
    async def _monitor_current_training(self):
        """Monitor the current training progress"""
        if not self.current_training:
            return
            
        context = self.get_context()
        
        # Calculate training duration
        if self.training_start_time:
            duration = datetime.now() - self.training_start_time
            
            # Check for stalled training
            if duration > timedelta(hours=4) and context.progress_percentage < 10:
                await self._handle_stalled_training()
                
            # Check for completion
            if context.progress_percentage >= 99:
                await self._complete_training()
                
    async def _complete_training(self):
        """Complete the current training"""
        domain = self.current_training
        duration = datetime.now() - self.training_start_time if self.training_start_time else None
        
        print(f"✅ Training completed: {domain} ({duration})")
        
        # Store training stats
        self.training_stats[domain] = {
            "completion_time": datetime.now(),
            "duration": duration,
            "final_quality": self.get_context().data_quality_score,
            "success": True
        }
        
        # Request GGUF creation
        self.send_message(
            AgentType.GGUF_CREATOR,
            MessageType.COORDINATION_REQUEST,
            {
                "action": "create_gguf",
                "domain": domain,
                "training_stats": self.training_stats[domain]
            }
        )
        
        # Update knowledge base
        self.send_message(
            AgentType.KNOWLEDGE_TRANSFER,
            MessageType.KNOWLEDGE_SHARE,
            {
                "domain": domain,
                "training_patterns": self._extract_training_patterns(domain),
                "optimization_insights": self._extract_optimization_insights(domain)
            }
        )
        
        # Reset current training
        self.current_training = None
        self.training_start_time = None
        
        # Update context
        self.update_context({
            "current_domain": None,
            "training_step": 0,
            "progress_percentage": 0.0
        })
        
    async def _handle_stalled_training(self):
        """Handle stalled training situations"""
        print(f"⚠️ Training appears stalled: {self.current_training}")
        
        # Request diagnostic information
        self.broadcast_message(
            MessageType.COORDINATION_REQUEST,
            {
                "action": "diagnostic_check",
                "domain": self.current_training,
                "issue": "stalled_training"
            },
            priority=2
        )
        
        # Implement recovery strategies
        await self._implement_recovery_strategies()
        
    async def _implement_recovery_strategies(self):
        """Implement training recovery strategies"""
        strategies = [
            "reduce_batch_size",
            "increase_checkpoint_frequency", 
            "optimize_memory_usage",
            "restart_with_last_checkpoint"
        ]
        
        for strategy in strategies:
            self.broadcast_message(
                MessageType.OPTIMIZATION_REQUEST,
                {
                    "strategy": strategy,
                    "domain": self.current_training,
                    "priority": "high"
                },
                priority=2
            )
            
    async def _optimize_training_order(self):
        """Optimize the training order based on knowledge transfer"""
        if len(self.training_queue) <= 1:
            return
            
        # Request optimization from Knowledge Transfer Agent
        self.send_message(
            AgentType.KNOWLEDGE_TRANSFER,
            MessageType.OPTIMIZATION_REQUEST,
            {
                "action": "optimize_training_order",
                "domains": self.training_queue.copy(),
                "current_knowledge": self._get_current_knowledge_base()
            }
        )
        
    async def _optimize_resources(self):
        """Optimize resource allocation"""
        context = self.get_context()
        
        # Check GPU utilization
        if any(util > 90 for util in context.gpu_utilization.values()):
            self.send_message(
                AgentType.GPU_OPTIMIZER,
                MessageType.OPTIMIZATION_REQUEST,
                {
                    "action": "scale_up_resources",
                    "current_utilization": context.gpu_utilization
                }
            )
            
        # Check cost efficiency
        if context.cost_tracking.get("daily_spend", 0) > 50:  # $50 daily limit
            self.send_message(
                AgentType.GPU_OPTIMIZER,
                MessageType.OPTIMIZATION_REQUEST,
                {
                    "action": "optimize_costs",
                    "current_spending": context.cost_tracking
                }
            )
            
    async def _update_strategies(self):
        """Update coordination strategies based on performance"""
        context = self.get_context()
        
        # Enable parallel training if resources allow
        if (len(context.gpu_utilization) > 1 and 
            all(util < 70 for util in context.gpu_utilization.values())):
            self.coordination_strategies["parallel_domains"] = True
            
        # Adjust quality vs speed balance
        if context.data_quality_score < 70:
            self.coordination_strategies["quality_first"] = True
        elif context.data_quality_score > 95:
            self.coordination_strategies["quality_first"] = False
            
    def _estimate_training_duration(self, domain: str) -> timedelta:
        """Estimate training duration for a domain"""
        # Base estimate on previous training stats
        if domain in self.training_stats:
            return self.training_stats[domain]["duration"]
            
        # Default estimates based on domain complexity
        complexity_map = {
            "healthcare": timedelta(hours=2),
            "mental_health": timedelta(hours=1.5),
            "business": timedelta(hours=2),
            "education": timedelta(hours=1.5),
            "creative": timedelta(hours=1),
        }
        
        return complexity_map.get(domain, timedelta(hours=1.5))
        
    def _get_quality_requirements(self, domain: str) -> Dict[str, Any]:
        """Get quality requirements for a domain"""
        # High-stakes domains need higher quality
        high_quality_domains = ["healthcare", "mental_health", "crisis_intervention"]
        
        if domain in high_quality_domains:
            return {
                "min_validation_score": 95,
                "min_data_quality": 90,
                "crisis_scenarios": True,
                "therapeutic_focus": True
            }
        else:
            return {
                "min_validation_score": 85,
                "min_data_quality": 80,
                "crisis_scenarios": False,
                "therapeutic_focus": False
            }
            
    def _get_quality_thresholds(self, domain: str) -> Dict[str, float]:
        """Get quality monitoring thresholds"""
        requirements = self._get_quality_requirements(domain)
        return {
            "validation_score": requirements["min_validation_score"],
            "data_quality": requirements["min_data_quality"],
            "early_stop_threshold": 50  # Stop if quality drops below 50%
        }
        
    def _extract_training_patterns(self, domain: str) -> Dict[str, Any]:
        """Extract successful training patterns"""
        context = self.get_context()
        return {
            "optimal_batch_size": 6,  # From proven TARA experience
            "successful_parameters": context.optimization_strategies.get(domain, {}),
            "quality_insights": {
                "final_score": context.data_quality_score,
                "validation_trend": context.validation_scores[-10:] if context.validation_scores else []
            }
        }
        
    def _extract_optimization_insights(self, domain: str) -> Dict[str, Any]:
        """Extract optimization insights from training"""
        return {
            "resource_efficiency": self._calculate_resource_efficiency(),
            "time_optimization": self._calculate_time_optimization(),
            "cost_effectiveness": self._calculate_cost_effectiveness(),
            "quality_achievement": self._calculate_quality_achievement()
        }
        
    def _calculate_resource_efficiency(self) -> float:
        """Calculate resource efficiency score"""
        context = self.get_context()
        avg_gpu_util = sum(context.gpu_utilization.values()) / len(context.gpu_utilization) if context.gpu_utilization else 0
        return min(avg_gpu_util / 85 * 100, 100)  # Target 85% utilization
        
    def _calculate_time_optimization(self) -> float:
        """Calculate time optimization score"""
        if not self.training_start_time:
            return 0
            
        actual_duration = datetime.now() - self.training_start_time
        estimated_duration = self._estimate_training_duration(self.current_training)
        
        if actual_duration <= estimated_duration:
            return 100
        else:
            return max(0, 100 - (actual_duration.total_seconds() / estimated_duration.total_seconds() - 1) * 100)
            
    def _calculate_cost_effectiveness(self) -> float:
        """Calculate cost effectiveness score"""
        context = self.get_context()
        daily_spend = context.cost_tracking.get("daily_spend", 0)
        
        if daily_spend <= 10:  # Excellent
            return 100
        elif daily_spend <= 25:  # Good
            return 80
        elif daily_spend <= 50:  # Acceptable
            return 60
        else:  # Needs optimization
            return 30
            
    def _calculate_quality_achievement(self) -> float:
        """Calculate quality achievement score"""
        context = self.get_context()
        return context.data_quality_score
        
    def _get_current_knowledge_base(self) -> Dict[str, Any]:
        """Get current knowledge base for optimization"""
        context = self.get_context()
        return {
            "successful_patterns": context.successful_patterns,
            "optimization_strategies": context.optimization_strategies,
            "training_stats": self.training_stats
        }
        
    async def _handle_resource_status(self, data: Dict[str, Any]):
        """Handle resource status updates"""
        # Update context with resource information
        self.update_context({
            "gpu_utilization": data.get("gpu_utilization", {}),
            "memory_usage": data.get("memory_usage", {}),
            "cost_tracking": data.get("cost_tracking", {})
        })
        
    async def _handle_quality_metrics(self, data: Dict[str, Any]):
        """Handle quality metrics updates"""
        # Update context with quality information
        context = self.get_context()
        if "validation_score" in data:
            context.validation_scores.append(data["validation_score"])
            
        self.update_context({
            "data_quality_score": data.get("data_quality_score", context.data_quality_score),
            "model_performance_metrics": data.get("performance_metrics", context.model_performance_metrics)
        })
        
    async def _handle_training_progress(self, data: Dict[str, Any]):
        """Handle training progress updates"""
        # Update context with progress information
        self.update_context({
            "training_step": data.get("current_step", 0),
            "total_steps": data.get("total_steps", 0),
            "progress_percentage": data.get("progress_percentage", 0.0)
        })
        
    async def _handle_error_notification(self, data: Dict[str, Any]):
        """Handle error notifications"""
        error_type = data.get("error_type", "unknown")
        domain = data.get("domain", self.current_training)
        
        print(f"🚨 Training error detected: {error_type} in {domain}")
        
        # Implement error recovery
        await self._implement_error_recovery(error_type, domain, data)
        
    async def _handle_optimization_request(self, data: Dict[str, Any]):
        """Handle optimization requests from other agents"""
        request_type = data.get("action", "")
        
        if request_type == "update_training_order":
            new_order = data.get("optimized_order", [])
            if new_order:
                self.training_queue = new_order
                print(f"📋 Training order optimized: {new_order}")
                
    async def _implement_error_recovery(self, error_type: str, domain: str, error_data: Dict[str, Any]):
        """Implement error recovery strategies"""
        recovery_strategies = {
            "out_of_memory": self._handle_memory_error,
            "training_stall": self._handle_stall_error,
            "quality_degradation": self._handle_quality_error,
            "resource_exhaustion": self._handle_resource_error
        }
        
        handler = recovery_strategies.get(error_type, self._handle_generic_error)
        await handler(domain, error_data)
        
    async def _handle_memory_error(self, domain: str, error_data: Dict[str, Any]):
        """Handle out of memory errors"""
        self.broadcast_message(
            MessageType.OPTIMIZATION_REQUEST,
            {
                "action": "reduce_memory_usage",
                "domain": domain,
                "suggested_batch_size": 4  # Reduce from 6 to 4
            },
            priority=2
        )
        
    async def _handle_stall_error(self, domain: str, error_data: Dict[str, Any]):
        """Handle training stall errors"""
        await self._implement_recovery_strategies()
        
    async def _handle_quality_error(self, domain: str, error_data: Dict[str, Any]):
        """Handle quality degradation errors"""
        self.send_message(
            AgentType.DATA_GENERATOR,
            MessageType.COORDINATION_REQUEST,
            {
                "action": "improve_data_quality",
                "domain": domain,
                "quality_issues": error_data.get("quality_issues", [])
            },
            priority=2
        )
        
    async def _handle_resource_error(self, domain: str, error_data: Dict[str, Any]):
        """Handle resource exhaustion errors"""
        self.send_message(
            AgentType.GPU_OPTIMIZER,
            MessageType.COORDINATION_REQUEST,
            {
                "action": "emergency_scaling",
                "domain": domain,
                "resource_issues": error_data.get("resource_issues", [])
            },
            priority=3
        )
        
    async def _handle_generic_error(self, domain: str, error_data: Dict[str, Any]):
        """Handle generic errors"""
        print(f"⚠️ Generic error recovery for {domain}: {error_data}")
        
        # Restart training with conservative parameters
        self.broadcast_message(
            MessageType.COORDINATION_REQUEST,
            {
                "action": "restart_training",
                "domain": domain,
                "conservative_mode": True
            },
            priority=2
        )

# Global conductor instance
training_conductor = TrainingConductorAgent() 