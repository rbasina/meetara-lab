"""
MeeTARA Lab - MCP Protocol Implementation
Standardized context sharing between all training agents
"""

import json
import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import queue

class AgentType(Enum):
    CONDUCTOR = "training_conductor"
    GPU_OPTIMIZER = "gpu_optimizer"
    QUALITY_ASSURANCE = "quality_assurance"
    DATA_GENERATOR = "data_generator"
    GGUF_CREATOR = "gguf_creator"
    KNOWLEDGE_TRANSFER = "knowledge_transfer"
    CROSS_DOMAIN = "cross_domain_intelligence"

class MessageType(Enum):
    STATUS_UPDATE = "status_update"
    OPTIMIZATION_REQUEST = "optimization_request"
    QUALITY_METRICS = "quality_metrics"
    RESOURCE_STATUS = "resource_status"
    TRAINING_PROGRESS = "training_progress"
    KNOWLEDGE_SHARE = "knowledge_share"
    ERROR_NOTIFICATION = "error_notification"
    COORDINATION_REQUEST = "coordination_request"

@dataclass
class MCPMessage:
    """Standardized MCP message format"""
    id: str
    timestamp: datetime
    sender: AgentType
    recipient: Optional[AgentType]  # None for broadcast
    message_type: MessageType
    data: Dict[str, Any]
    priority: int = 1  # 1=normal, 2=high, 3=critical

@dataclass
class TrainingContext:
    """Shared training context across all agents"""
    # Resource Status
    gpu_utilization: Dict[str, float]
    memory_usage: Dict[str, float]
    cost_tracking: Dict[str, float]
    
    # Training State
    current_domain: Optional[str]
    training_step: int
    total_steps: int
    progress_percentage: float
    
    # Quality Metrics
    validation_scores: List[float]
    data_quality_score: float
    model_performance_metrics: Dict[str, float]
    
    # Knowledge Base
    successful_patterns: Dict[str, Any]
    optimization_strategies: Dict[str, Any]
    error_solutions: Dict[str, Any]
    
    # Agent Status
    active_agents: List[AgentType]
    agent_health: Dict[AgentType, str]

class MCPProtocol:
    """Model Context Protocol for agent communication"""
    
    def __init__(self):
        self.agents: Dict[AgentType, 'BaseAgent'] = {}
        self.message_queue = queue.Queue()
        self.context = TrainingContext(
            gpu_utilization={},
            memory_usage={},
            cost_tracking={},
            current_domain=None,
            training_step=0,
            total_steps=0,
            progress_percentage=0.0,
            validation_scores=[],
            data_quality_score=0.0,
            model_performance_metrics={},
            successful_patterns={},
            optimization_strategies={},
            error_solutions={},
            active_agents=[],
            agent_health={}
        )
        self.message_handlers: Dict[MessageType, List[Callable]] = {}
        self.running = False
        self.thread = None
        
    def register_agent(self, agent_type: AgentType, agent: 'BaseAgent'):
        """Register an agent with the MCP protocol"""
        self.agents[agent_type] = agent
        self.context.active_agents.append(agent_type)
        self.context.agent_health[agent_type] = "healthy"
        print(f"🤖 Agent registered: {agent_type.value}")
        
    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """Register a message handler for specific message types"""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)
        
    def send_message(self, sender: AgentType, recipient: Optional[AgentType], 
                    message_type: MessageType, data: Dict[str, Any], priority: int = 1):
        """Send a message through the MCP protocol"""
        message = MCPMessage(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            sender=sender,
            recipient=recipient,
            message_type=message_type,
            data=data,
            priority=priority
        )
        self.message_queue.put(message)
        
    def broadcast_message(self, sender: AgentType, message_type: MessageType, 
                         data: Dict[str, Any], priority: int = 1):
        """Broadcast a message to all agents"""
        self.send_message(sender, None, message_type, data, priority)
        
    def update_context(self, updates: Dict[str, Any]):
        """Update the shared training context"""
        for key, value in updates.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
        
        # Notify all agents of context update
        self.broadcast_message(
            AgentType.CONDUCTOR,
            MessageType.STATUS_UPDATE,
            {"context_update": updates}
        )
        
    def get_context(self) -> TrainingContext:
        """Get the current training context"""
        return self.context
        
    def start(self):
        """Start the MCP protocol message processing"""
        self.running = True
        self.thread = threading.Thread(target=self._process_messages)
        self.thread.start()
        print("📡 MCP Protocol started")
        
    def stop(self):
        """Stop the MCP protocol"""
        self.running = False
        if self.thread:
            self.thread.join()
        print("📡 MCP Protocol stopped")
        
    def _process_messages(self):
        """Process messages in the queue"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while self.running:
            try:
                # Get message with timeout
                message = self.message_queue.get(timeout=1.0)
                
                # Process message
                loop.run_until_complete(self._handle_message(message))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error processing MCP message: {e}")
                
        loop.close()
                
    async def _handle_message(self, message: MCPMessage):
        """Handle an incoming message"""
        try:
            # Update agent health
            self.context.agent_health[message.sender] = "healthy"
            
            # Call registered handlers
            if message.message_type in self.message_handlers:
                for handler in self.message_handlers[message.message_type]:
                    await handler(message)
                    
            # Route to specific recipient or broadcast
            if message.recipient and message.recipient in self.agents:
                await self.agents[message.recipient].handle_mcp_message(message)
            elif message.recipient is None:  # Broadcast
                for agent in self.agents.values():
                    await agent.handle_mcp_message(message)
                    
        except Exception as e:
            print(f"❌ Error handling message {message.id}: {e}")

class BaseAgent:
    """Base class for all training agents"""
    
    def __init__(self, agent_type: AgentType, mcp: MCPProtocol):
        self.agent_type = agent_type
        self.mcp = mcp
        self.running = False
        
    async def start(self):
        """Start the agent"""
        self.running = True
        self.mcp.register_agent(self.agent_type, self)
        print(f"🚀 {self.agent_type.value} started")
        
    async def stop(self):
        """Stop the agent"""
        self.running = False
        print(f"🛑 {self.agent_type.value} stopped")
        
    async def handle_mcp_message(self, message: MCPMessage):
        """Handle incoming MCP messages"""
        # Override in subclasses
        pass
        
    def send_message(self, recipient: Optional[AgentType], message_type: MessageType, 
                    data: Dict[str, Any], priority: int = 1):
        """Send a message via MCP"""
        self.mcp.send_message(self.agent_type, recipient, message_type, data, priority)
        
    def broadcast_message(self, message_type: MessageType, data: Dict[str, Any], priority: int = 1):
        """Broadcast a message via MCP"""
        self.mcp.broadcast_message(self.agent_type, message_type, data, priority)
        
    def update_context(self, updates: Dict[str, Any]):
        """Update the shared context"""
        self.mcp.update_context(updates)
        
    def get_context(self) -> TrainingContext:
        """Get the current training context"""
        return self.mcp.get_context()

# Global MCP protocol instance
mcp_protocol = MCPProtocol()

def get_mcp_protocol() -> MCPProtocol:
    """Get the global MCP protocol instance"""
    return mcp_protocol 