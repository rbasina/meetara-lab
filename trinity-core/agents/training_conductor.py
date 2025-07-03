"""
MeeTARA Lab - Training Conductor Agent
Orchestrates multi-domain training with intelligent resource management and quality assurance
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import yaml
from enum import Enum
from collections import defaultdict, deque
from .mcp_protocol import BaseAgent, AgentType, MessageType, MCPMessage, mcp_protocol

# Use centralized domain mapping
from ..domain_integration import (
    domain_integration, 
    get_domain_categories, 
    get_all_domains, 
    validate_domain, 
    get_model_for_domain,
    get_enhanced_feature_for_domain
)

class TrainingConductorAgent(BaseAgent):
    """Orchestrates multi-domain training with intelligent resource management"""
    
    def __init__(self, mcp=None):
        super().__init__(AgentType.TRAINING_CONDUCTOR, mcp or mcp_protocol)
        
        # Load domain configuration using centralized approach
        self._load_domain_configuration()
        
        # Training orchestration
        self.training_queue = deque()
        self.active_training = None
        self.training_history = []
        self.performance_metrics = defaultdict(list)
        
        # Resource management
        self.resource_allocation = {}
        self.optimization_strategies = {}
        
        # Quality assurance
        self.quality_requirements = {}
        self.quality_thresholds = {}
        
        # Training complexity analysis
        self.training_complexity = {}
        
        # Knowledge base for continuous learning
        self.knowledge_base = {}
        
    def _load_domain_configuration(self):
        """Load domain configuration using centralized domain integration"""
        try:
            # Use centralized domain mapping
            self.domain_categories = get_domain_categories()
            self.domain_mapping = self.domain_categories.copy()
            
            # Create reverse mapping for quick lookups
            self.domain_to_category = {}
            for category, domains in self.domain_categories.items():
                for domain in domains:
                    self.domain_to_category[domain] = category
            
            # Initialize category-based requirements
            self._initialize_category_requirements()
            
            # Load quality requirements per category
            self._initialize_quality_requirements()
            
            # Load training complexity per category
            self._initialize_training_complexity()
            
            total_domains = len(get_all_domains())
            print(f"✅ Training Conductor: Loaded {total_domains} domains across {len(self.domain_mapping)} categories")
            
            # Print category breakdown
            for category, domains in self.domain_mapping.items():
                print(f"   → {category}: {len(domains)} domains")
                
        except Exception as e:
            print(f"❌ Error loading domain configuration: {e}")
            raise Exception(f"Training Conductor requires valid domain configuration: {e}")
            
    def _initialize_category_requirements(self):
        """Initialize quality requirements based on domain categories"""
        self.category_requirements = {
            "healthcare": {
                "min_validation_score": 95,
                "min_data_quality": 90,
                "crisis_scenarios": True,
                "therapeutic_focus": True,
                "safety_critical": True,
                "complexity_hours": 3
            },
            "specialized": {
                "min_validation_score": 92,
                "min_data_quality": 88,
                "crisis_scenarios": True,
                "therapeutic_focus": False,
                "safety_critical": True,
                "complexity_hours": 2.5
            },
            "business": {
                "min_validation_score": 88,
                "min_data_quality": 85,
                "crisis_scenarios": False,
                "therapeutic_focus": False,
                "safety_critical": False,
                "complexity_hours": 2
            },
            "education": {
                "min_validation_score": 87,
                "min_data_quality": 82,
                "crisis_scenarios": False,
                "therapeutic_focus": False,
                "safety_critical": False,
                "complexity_hours": 1.5
            },
            "technology": {
                "min_validation_score": 87,
                "min_data_quality": 82,
                "crisis_scenarios": False,
                "therapeutic_focus": False,
                "safety_critical": False,
                "complexity_hours": 1.5
            },
            "daily_life": {
                "min_validation_score": 85,
                "min_data_quality": 80,
                "crisis_scenarios": False,
                "therapeutic_focus": False,
                "safety_critical": False,
                "complexity_hours": 1
            },
            "creative": {
                "min_validation_score": 82,
                "min_data_quality": 78,
                "crisis_scenarios": False,
                "therapeutic_focus": False,
                "safety_critical": False,
                "complexity_hours": 1
            }
        }
        
        # Apply requirements to all domains in each category
        for category, requirements in self.category_requirements.items():
            if category in self.domain_mapping:
                for domain in self.domain_mapping[category]:
                    self.quality_requirements[domain] = requirements.copy()
                    
    def _initialize_quality_requirements(self):
        """Initialize quality requirements based on domain categories"""
        # Category-based quality requirements
        category_requirements = {
            "healthcare": {
                "min_validation_score": 95,
                "min_data_quality": 90,
                "crisis_scenarios": True,
                "therapeutic_focus": True,
                "safety_critical": True
            },
            "specialized": {
                "min_validation_score": 92,
                "min_data_quality": 88,
                "crisis_scenarios": True,
                "therapeutic_focus": False,
                "safety_critical": True
            },
            "business": {
                "min_validation_score": 88,
                "min_data_quality": 85,
                "crisis_scenarios": False,
                "therapeutic_focus": False,
                "safety_critical": False
            },
            "education": {
                "min_validation_score": 87,
                "min_data_quality": 82,
                "crisis_scenarios": False,
                "therapeutic_focus": False,
                "safety_critical": False
            },
            "technology": {
                "min_validation_score": 87,
                "min_data_quality": 82,
                "crisis_scenarios": False,
                "therapeutic_focus": False,
                "safety_critical": False
            },
            "daily_life": {
                "min_validation_score": 85,
                "min_data_quality": 80,
                "crisis_scenarios": False,
                "therapeutic_focus": False,
                "safety_critical": False
            },
            "creative": {
                "min_validation_score": 82,
                "min_data_quality": 78,
                "crisis_scenarios": False,
                "therapeutic_focus": False,
                "safety_critical": False
            }
        }
        
        # Apply to all domains
        for category, requirements in category_requirements.items():
            if category in self.domain_mapping:
                for domain in self.domain_mapping[category]:
                    self.quality_thresholds[domain] = requirements.copy()
                    
    def _initialize_training_complexity(self):
        """Initialize training complexity estimates per domain category"""
        complexity_mapping = {
            "healthcare": {"base_hours": 3.0, "complexity_multiplier": 1.5},
            "specialized": {"base_hours": 2.5, "complexity_multiplier": 1.3},
            "business": {"base_hours": 2.0, "complexity_multiplier": 1.2},
            "education": {"base_hours": 1.5, "complexity_multiplier": 1.1},
            "technology": {"base_hours": 1.5, "complexity_multiplier": 1.1},
            "daily_life": {"base_hours": 1.0, "complexity_multiplier": 1.0},
            "creative": {"base_hours": 1.0, "complexity_multiplier": 1.0}
        }
        
        for category, complexity in complexity_mapping.items():
            if category in self.domain_mapping:
                for domain in self.domain_mapping[category]:
                    self.training_complexity[domain] = complexity.copy()

    async def start(self):
        """Start the Training Conductor Agent"""
        await super().start()
        
        # Initialize training orchestration
        await self._initialize_training_orchestration()
        
        # Start orchestration loop
        asyncio.create_task(self._orchestration_loop())
        
        print("🎯 Training Conductor Agent started")
        print(f"   → Domains managed: {len(get_all_domains())}")
        print(f"   → Categories: {len(self.domain_categories)}")
        print(f"   → Config-driven: Centralized domain integration")

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
        """Main orchestration loop for training management"""
        while self.running:
            try:
                # Check for new training requests
                if not self.active_training and self.training_queue:
                    await self._start_next_training()
                
                # Monitor active training
                if self.active_training:
                    await self._monitor_current_training()
                
                # Optimize training order
                await self._optimize_training_order()
                
                # Resource optimization
                await self._optimize_resources()
                
                # Update strategies based on performance
                await self._update_strategies()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                print(f"❌ Training orchestration error: {e}")
                await asyncio.sleep(30)

    async def queue_domain_training(self, domain: str, priority: int = 1):
        """Queue a domain for training with priority"""
        
        # Validate domain using centralized validation
        if not validate_domain(domain):
            raise ValueError(f"Invalid domain: {domain}")
        
        # Get domain category and model
        category = self.domain_to_category.get(domain, "daily_life")
        model = get_model_for_domain(domain)
        
        training_request = {
            "domain": domain,
            "category": category,
            "priority": priority,
            "model": model,
            "timestamp": datetime.now(),
            "estimated_duration": self._estimate_training_duration(domain),
            "quality_requirements": self._get_quality_requirements(domain),
            "enhanced_features": get_enhanced_feature_for_domain(domain, "training_orchestrator")
        }
        
        # Insert based on priority
        inserted = False
        for i, existing in enumerate(self.training_queue):
            if priority > existing["priority"]:
                self.training_queue.insert(i, training_request)
                inserted = True
                break
        
        if not inserted:
            self.training_queue.append(training_request)
            
        print(f"✅ Queued {domain} training (priority: {priority}, category: {category})")

    async def queue_batch_training(self, domains: List[str], batch_name: str = "custom"):
        """Queue multiple domains for batch training"""
        
        # Validate all domains
        invalid_domains = [d for d in domains if not validate_domain(d)]
        if invalid_domains:
            raise ValueError(f"Invalid domains: {invalid_domains}")
        
        # Queue all domains with batch priority
        for domain in domains:
            await self.queue_domain_training(domain, priority=2)
            
        print(f"✅ Queued batch training: {batch_name} ({len(domains)} domains)")

    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        return {
            "queue_status": {
                "pending": len(self.training_queue),
                "active": 1 if self.active_training else 0,
                "completed": len(self.training_history)
            },
            "domain_coverage": {
                "total_domains": len(get_all_domains()),
                "categories": len(self.domain_categories),
                "config_driven": True
            },
            "performance": {
                "average_training_time": self._calculate_time_optimization(),
                "resource_efficiency": self._calculate_resource_efficiency(),
                "quality_achievement": self._calculate_quality_achievement()
            },
            "domain_integration": {
                "centralized_mapping": True,
                "enhanced_features": len(domain_integration.enhanced_features),
                "config_loaded": domain_integration.domain_mapping.get("config_loaded", False)
            }
        }

    def refresh_domain_configuration(self):
        """Refresh domain configuration from centralized source"""
        domain_integration.refresh_config()
        self._load_domain_configuration()
        print(f"✅ Domain configuration refreshed - {len(get_all_domains())} domains loaded")

    async def _start_next_training(self):
        """Start training the next domain in queue"""
        if not self.training_queue:
            return
            
        training_request = self.training_queue.popleft()
        self.active_training = training_request["domain"]
        self.training_start_time = datetime.now()
        
        print(f"🚀 Starting training: {self.active_training}")
        
        # Update context
        self.update_context({
            "current_domain": self.active_training,
            "training_step": 0,
            "progress_percentage": 0.0
        })
        
        # Request resource allocation
        self.send_message(
            AgentType.GPU_OPTIMIZER,
            MessageType.COORDINATION_REQUEST,
            {
                "action": "allocate_resources",
                "domain": self.active_training,
                "estimated_duration": self._estimate_training_duration(self.active_training)
            }
        )
        
        # Request data preparation
        self.send_message(
            AgentType.DATA_GENERATOR,
            MessageType.COORDINATION_REQUEST,
            {
                "action": "prepare_training_data",
                "domain": self.active_training,
                "quality_requirements": self._get_quality_requirements(self.active_training)
            }
        )
        
        # Start quality monitoring
        self.send_message(
            AgentType.QUALITY_ASSURANCE,
            MessageType.COORDINATION_REQUEST,
            {
                "action": "start_monitoring",
                "domain": self.active_training,
                "quality_thresholds": self._get_quality_thresholds(self.active_training)
            }
        )
        
    async def _monitor_current_training(self):
        """Monitor the current training progress"""
        if not self.active_training:
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
        domain = self.active_training
        duration = datetime.now() - self.training_start_time if self.training_start_time else None
        
        print(f"✅ Training completed: {domain} ({duration})")
        
        # Store training stats
        self.training_history.append({
            "domain": domain,
            "completion_time": datetime.now(),
            "duration": duration,
            "final_quality": self.get_context().data_quality_score,
            "success": True
        })
        
        # Request GGUF creation
        self.send_message(
            AgentType.GGUF_CREATOR,
            MessageType.COORDINATION_REQUEST,
            {
                "action": "create_gguf",
                "domain": domain,
                "training_stats": self.training_history[-1]
            }
        )
        
        # Update knowledge base
        self.knowledge_base[domain] = {
            "training_patterns": self._extract_training_patterns(domain),
            "optimization_insights": self._extract_optimization_insights(domain)
        }
        
        # Reset current training
        self.active_training = None
        self.training_start_time = None
        
        # Update context
        self.update_context({
            "current_domain": None,
            "training_step": 0,
            "progress_percentage": 0.0
        })
        
    async def _handle_stalled_training(self):
        """Handle stalled training situations"""
        print(f"⚠️ Training appears stalled: {self.active_training}")
        
        # Request diagnostic information
        self.broadcast_message(
            MessageType.COORDINATION_REQUEST,
            {
                "action": "diagnostic_check",
                "domain": self.active_training,
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
                    "domain": self.active_training,
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
                "domains": list(self.training_queue),
                "current_knowledge": self.knowledge_base
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
            self.optimization_strategies["parallel_domains"] = True
            
        # Adjust quality vs speed balance
        if context.data_quality_score < 70:
            self.optimization_strategies["quality_first"] = True
        elif context.data_quality_score > 95:
            self.optimization_strategies["quality_first"] = False
            
    def _estimate_training_duration(self, domain: str) -> timedelta:
        """Estimate training duration for a domain based on category"""
        # Base estimate on previous training stats
        if domain in self.training_history:
            return self.training_history[-1]["duration"]
        
        # Get domain category for intelligent estimation
        category = self.domain_categories.get(domain)
        
        if category and category in self.category_requirements:
            hours = self.category_requirements[category]["complexity_hours"]
            return timedelta(hours=hours)
            
        # Default estimates based on domain complexity (fallback)
        complexity_map = {
            "healthcare": timedelta(hours=3),
            "mental_health": timedelta(hours=3),
            "specialized": timedelta(hours=2.5),
            "business": timedelta(hours=2),
            "education": timedelta(hours=1.5),
            "technology": timedelta(hours=1.5),
            "daily_life": timedelta(hours=1),
            "creative": timedelta(hours=1),
        }
        
        return complexity_map.get(domain, timedelta(hours=1.5))
        
    def _get_quality_requirements(self, domain: str) -> Dict[str, Any]:
        """Get quality requirements for a domain based on its category"""
        # Use domain category to determine requirements
        category = self.domain_categories.get(domain)
        
        if category and category in self.category_requirements:
            requirements = self.category_requirements[category]
            return {
                "min_validation_score": requirements["min_validation_score"],
                "min_data_quality": requirements["min_data_quality"],
                "crisis_scenarios": requirements["crisis_scenarios"],
                "therapeutic_focus": requirements["therapeutic_focus"]
            }
        
        # Fallback for unknown domains
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
        estimated_duration = self._estimate_training_duration(self.active_training)
        
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
        domain = data.get("domain", self.active_training)
        
        print(f"🚨 Training error detected: {error_type} in {domain}")
        
        # Implement error recovery
        await self._implement_error_recovery(error_type, domain, data)
        
    async def _handle_optimization_request(self, data: Dict[str, Any]):
        """Handle optimization requests from other agents"""
        request_type = data.get("action", "")
        
        if request_type == "update_training_order":
            new_order = data.get("optimized_order", [])
            if new_order:
                self.training_queue = deque(new_order)
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
