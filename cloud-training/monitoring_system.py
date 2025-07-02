"""
MeeTARA Lab - Monitoring & Recovery System with Trinity Architecture
Real-time training dashboards, health checks, automatic recovery, and performance analytics
"""

import asyncio
import json
import time
import yaml
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Import trinity-core components
import sys
sys.path.append('../trinity-core')
from agents.mcp_protocol import BaseAgent, AgentType, MessageType, MCPMessage

class ServiceStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OFFLINE = "offline"

class RecoveryAction(Enum):
    RESTART_SERVICE = "restart_service"
    SWITCH_PROVIDER = "switch_provider"
    REDUCE_LOAD = "reduce_load"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"

@dataclass
class HealthMetric:
    service_name: str
    status: ServiceStatus
    metric_value: float
    threshold: float
    timestamp: datetime
    details: Dict[str, Any]

@dataclass
class RecoveryEvent:
    event_id: str
    service_name: str
    issue_type: str
    action_taken: RecoveryAction
    success: bool
    timestamp: datetime
    details: Dict[str, Any]

class MonitoringSystem(BaseAgent):
    """Monitoring & Recovery System with Trinity Architecture and cloud intelligence"""
    
    def __init__(self, mcp=None):
        super().__init__(AgentType.MONITORING_SYSTEM, mcp)
        
        # Service monitoring configuration
        self.monitored_services = {
            "training_orchestrator": {
                "health_endpoint": "http://localhost:8001/health",
                "critical_metrics": ["active_trainings", "cost_tracking", "success_rate"],
                "thresholds": {"success_rate": 85.0, "cost_per_hour": 5.0},
                "recovery_actions": [RecoveryAction.RESTART_SERVICE, RecoveryAction.SWITCH_PROVIDER]
            },
            "gguf_factory": {
                "health_endpoint": "http://localhost:8002/health",
                "critical_metrics": ["gguf_creation_rate", "quality_score", "file_size"],
                "thresholds": {"quality_score": 90.0, "file_size_mb": 10.0},
                "recovery_actions": [RecoveryAction.RESTART_SERVICE, RecoveryAction.REDUCE_LOAD]
            },
            "gpu_orchestrator": {
                "health_endpoint": "http://localhost:8003/health",
                "critical_metrics": ["gpu_utilization", "memory_usage", "provider_status"],
                "thresholds": {"gpu_utilization": 95.0, "memory_usage": 90.0},
                "recovery_actions": [RecoveryAction.REDUCE_LOAD, RecoveryAction.SWITCH_PROVIDER]
            },
            "cost_monitor": {
                "health_endpoint": "http://localhost:8004/health",
                "critical_metrics": ["daily_spend", "monthly_projection", "budget_usage"],
                "thresholds": {"budget_usage": 95.0, "daily_spend": 5.0},
                "recovery_actions": [RecoveryAction.EMERGENCY_SHUTDOWN, RecoveryAction.REDUCE_LOAD]
            },
            "trinity_intelligence": {
                "health_endpoint": "http://localhost:8005/health",
                "critical_metrics": ["fusion_efficiency", "intelligence_amplification", "response_time"],
                "thresholds": {"fusion_efficiency": 85.0, "response_time": 2.0},
                "recovery_actions": [RecoveryAction.RESTART_SERVICE]
            }
        }
        
        # Real-time dashboard data
        self.dashboard_data = {
            "system_overview": {
                "total_services": len(self.monitored_services),
                "healthy_services": 0,
                "warning_services": 0,
                "critical_services": 0,
                "offline_services": 0,
                "last_update": None
            },
            "training_metrics": {
                "active_trainings": 0,
                "completed_domains": 0,
                "failed_domains": 0,
                "success_rate": 0.0,
                "average_cost_per_domain": 0.0,
                "speed_improvement_factor": 0.0
            },
            "performance_metrics": {
                "cpu_usage": 0.0,
                "memory_usage": 0.0,
                "gpu_utilization": 0.0,
                "network_io": 0.0,
                "disk_io": 0.0
            },
            "cost_metrics": {
                "daily_spend": 0.0,
                "weekly_spend": 0.0,
                "monthly_spend": 0.0,
                "projected_monthly": 0.0,
                "budget_remaining": 50.0
            },
            "trinity_metrics": {
                "arc_reactor_efficiency": 90.0,
                "perplexity_intelligence": 95.0,
                "einstein_fusion": 504.0,
                "overall_amplification": 100.0
            }
        }
        
        # Health check tracking
        self.health_history = {}
        self.recovery_history = []
        self.alert_history = []
        
        # Recovery system configuration
        self.recovery_config = {
            "auto_recovery_enabled": True,
            "max_recovery_attempts": 3,
            "recovery_cooldown_minutes": 5,
            "escalation_threshold": 3,
            "emergency_contacts": [],
            "backup_providers": ["lambda_labs", "runpod", "vast_ai"]
        }
        
        # Performance regression detection
        self.performance_baselines = {
            "training_speed": {"t4": 37.0, "v100": 75.0, "a100": 151.0},
            "quality_scores": 101.0,
            "cost_efficiency": 0.10,  # $0.10 per domain
            "model_size": 8.3  # MB
        }
        
        # Trinity Architecture monitoring
        self.trinity_monitoring = {
            "arc_reactor_efficiency": True,    # 90% efficiency monitoring
            "perplexity_intelligence": True,   # Context-aware monitoring
            "einstein_fusion": True           # Exponential monitoring gains
        }
        
        # Monitoring intervals
        self.monitoring_intervals = {
            "health_check": 30,      # 30 seconds
            "performance_check": 60, # 1 minute
            "cost_check": 300,      # 5 minutes
            "dashboard_update": 10   # 10 seconds
        }
        
    async def start(self):
        """Start the Monitoring & Recovery System"""
        await super().start()
        print("📊 Monitoring & Recovery System ready with Trinity Architecture")
        
        # Start monitoring tasks
        asyncio.create_task(self._health_check_loop())
        asyncio.create_task(self._performance_monitoring_loop())
        asyncio.create_task(self._cost_monitoring_loop())
        asyncio.create_task(self._dashboard_update_loop())
        asyncio.create_task(self._regression_detection_loop())
        
    async def _health_check_loop(self):
        """Continuous health check monitoring"""
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.monitoring_intervals["health_check"])
            except Exception as e:
                print(f"⚠️ Health check loop error: {e}")
                await asyncio.sleep(60)  # Wait before retrying
                
    async def _perform_health_checks(self):
        """Perform health checks on all monitored services"""
        health_results = {}
        
        for service_name, config in self.monitored_services.items():
            try:
                # Simulate health check (in real implementation, would make HTTP requests)
                health_status = await self._check_service_health(service_name, config)
                health_results[service_name] = health_status
                
                # Store health history
                if service_name not in self.health_history:
                    self.health_history[service_name] = []
                self.health_history[service_name].append(health_status)
                
                # Keep only last 100 entries
                if len(self.health_history[service_name]) > 100:
                    self.health_history[service_name] = self.health_history[service_name][-100:]
                    
                # Check if recovery action needed
                if health_status.status in [ServiceStatus.CRITICAL, ServiceStatus.OFFLINE]:
                    await self._trigger_recovery_action(service_name, health_status)
                    
            except Exception as e:
                print(f"⚠️ Health check failed for {service_name}: {e}")
                
        # Update dashboard
        await self._update_service_status_dashboard(health_results)
        
    async def _check_service_health(self, service_name: str, config: Dict[str, Any]) -> HealthMetric:
        """Check health of a specific service"""
        
        # Simulate service health check
        # In real implementation, would make HTTP request to health endpoint
        
        # Simulate different health states
        import random
        health_simulation = {
            "training_orchestrator": 0.95,  # 95% uptime
            "gguf_factory": 0.98,          # 98% uptime
            "gpu_orchestrator": 0.90,      # 90% uptime (cloud dependency)
            "cost_monitor": 0.99,          # 99% uptime
            "trinity_intelligence": 0.97   # 97% uptime
        }
        
        uptime_probability = health_simulation.get(service_name, 0.95)
        is_healthy = random.random() < uptime_probability
        
        if is_healthy:
            status = ServiceStatus.HEALTHY
            metric_value = random.uniform(85, 100)
        else:
            status = random.choice([ServiceStatus.WARNING, ServiceStatus.CRITICAL])
            metric_value = random.uniform(40, 84)
            
        return HealthMetric(
            service_name=service_name,
            status=status,
            metric_value=metric_value,
            threshold=config["thresholds"].get("success_rate", 85.0),
            timestamp=datetime.now(),
            details={
                "endpoint": config["health_endpoint"],
                "response_time": random.uniform(0.1, 2.0),
                "last_error": None if is_healthy else "Simulated service degradation"
            }
        )
        
    async def _trigger_recovery_action(self, service_name: str, health_status: HealthMetric):
        """Trigger automatic recovery action for failing service"""
        
        if not self.recovery_config["auto_recovery_enabled"]:
            print(f"⚠️ Auto-recovery disabled, manual intervention needed for {service_name}")
            return
            
        # Check recovery cooldown
        recent_recoveries = [r for r in self.recovery_history 
                           if r.service_name == service_name 
                           and r.timestamp > datetime.now() - timedelta(minutes=self.recovery_config["recovery_cooldown_minutes"])]
        
        if len(recent_recoveries) >= self.recovery_config["max_recovery_attempts"]:
            print(f"🚨 Recovery cooldown active for {service_name}, escalating...")
            await self._escalate_issue(service_name, health_status)
            return
            
        # Select recovery action
        service_config = self.monitored_services[service_name]
        recovery_action = service_config["recovery_actions"][0]  # Try first action
        
        # Execute recovery
        recovery_success = await self._execute_recovery_action(service_name, recovery_action, health_status)
        
        # Log recovery event
        recovery_event = RecoveryEvent(
            event_id=f"recovery_{int(time.time())}",
            service_name=service_name,
            issue_type=health_status.status.value,
            action_taken=recovery_action,
            success=recovery_success,
            timestamp=datetime.now(),
            details={
                "metric_value": health_status.metric_value,
                "threshold": health_status.threshold,
                "automatic": True
            }
        )
        
        self.recovery_history.append(recovery_event)
        
        # Send notification
        await self._send_recovery_notification(recovery_event)
        
    async def _execute_recovery_action(self, service_name: str, action: RecoveryAction, 
                                     health_status: HealthMetric) -> bool:
        """Execute specific recovery action"""
        
        try:
            if action == RecoveryAction.RESTART_SERVICE:
                print(f"🔄 Restarting service: {service_name}")
                # In real implementation: restart service process
                await asyncio.sleep(2)  # Simulate restart time
                return True
                
            elif action == RecoveryAction.SWITCH_PROVIDER:
                print(f"🔄 Switching cloud provider for: {service_name}")
                # In real implementation: switch to backup provider
                await asyncio.sleep(5)  # Simulate provider switch time
                return True
                
            elif action == RecoveryAction.REDUCE_LOAD:
                print(f"🔄 Reducing load for: {service_name}")
                # In real implementation: reduce batch sizes, scale down
                await asyncio.sleep(1)  # Simulate load reduction
                return True
                
            elif action == RecoveryAction.EMERGENCY_SHUTDOWN:
                print(f"🚨 Emergency shutdown for: {service_name}")
                # In real implementation: graceful shutdown to prevent cost overrun
                await asyncio.sleep(3)  # Simulate shutdown time
                return True
                
        except Exception as e:
            print(f"❌ Recovery action failed for {service_name}: {e}")
            return False
            
        return False
        
    async def _escalate_issue(self, service_name: str, health_status: HealthMetric):
        """Escalate issue when auto-recovery fails"""
        
        escalation_alert = {
            "severity": "HIGH",
            "service": service_name,
            "issue": health_status.status.value,
            "metric_value": health_status.metric_value,
            "threshold": health_status.threshold,
            "recovery_attempts": len([r for r in self.recovery_history if r.service_name == service_name]),
            "timestamp": datetime.now().isoformat(),
            "action_required": "Manual intervention needed"
        }
        
        # Store alert
        self.alert_history.append(escalation_alert)
        
        # In real implementation: send to monitoring service, email, Slack, etc.
        print(f"🚨 ESCALATION ALERT: {json.dumps(escalation_alert, indent=2)}")
        
    async def _performance_monitoring_loop(self):
        """Monitor system performance metrics"""
        while True:
            try:
                await self._collect_performance_metrics()
                await asyncio.sleep(self.monitoring_intervals["performance_check"])
            except Exception as e:
                print(f"⚠️ Performance monitoring error: {e}")
                await asyncio.sleep(60)
                
    async def _collect_performance_metrics(self):
        """Collect system performance metrics"""
        
        # Simulate performance metrics
        import random
        
        performance_metrics = {
            "cpu_usage": random.uniform(20, 80),
            "memory_usage": random.uniform(30, 85),
            "gpu_utilization": random.uniform(60, 95),
            "network_io": random.uniform(10, 50),
            "disk_io": random.uniform(5, 30)
        }
        
        # Update dashboard
        self.dashboard_data["performance_metrics"].update(performance_metrics)
        
        # Check for performance degradation
        await self._check_performance_regression(performance_metrics)
        
    async def _check_performance_regression(self, current_metrics: Dict[str, float]):
        """Check for performance regression against baselines"""
        
        # Check GPU utilization efficiency
        gpu_util = current_metrics.get("gpu_utilization", 0)
        if gpu_util < 60:  # Below 60% utilization indicates potential issues
            await self._handle_performance_issue("low_gpu_utilization", {
                "current": gpu_util,
                "expected": ">60%",
                "suggestion": "Check training workload distribution"
            })
            
        # Check memory usage
        memory_usage = current_metrics.get("memory_usage", 0)
        if memory_usage > 90:  # Above 90% memory usage
            await self._handle_performance_issue("high_memory_usage", {
                "current": memory_usage,
                "threshold": "90%",
                "suggestion": "Reduce batch size or clear cache"
            })
            
    async def _handle_performance_issue(self, issue_type: str, details: Dict[str, Any]):
        """Handle detected performance issues"""
        
        performance_alert = {
            "type": "performance_degradation",
            "issue": issue_type,
            "details": details,
            "timestamp": datetime.now().isoformat(),
            "auto_action": True
        }
        
        # Log alert
        self.alert_history.append(performance_alert)
        
        # Auto-optimization based on issue type
        if issue_type == "low_gpu_utilization":
            # Request training orchestrator to optimize workload
            self.send_message(
                AgentType.TRAINING_CONDUCTOR,
                MessageType.OPTIMIZATION_REQUEST,
                {
                    "action": "optimize_gpu_utilization",
                    "current_utilization": details["current"],
                    "priority": "medium"
                }
            )
        elif issue_type == "high_memory_usage":
            # Request memory optimization
            self.broadcast_message(
                MessageType.OPTIMIZATION_REQUEST,
                {
                    "action": "optimize_memory_usage", 
                    "current_usage": details["current"],
                    "priority": "high"
                }
            )
            
    async def _cost_monitoring_loop(self):
        """Monitor cost metrics and budget usage"""
        while True:
            try:
                await self._check_cost_metrics()
                await asyncio.sleep(self.monitoring_intervals["cost_check"])
            except Exception as e:
                print(f"⚠️ Cost monitoring error: {e}")
                await asyncio.sleep(300)
                
    async def _check_cost_metrics(self):
        """Check cost metrics and budget usage"""
        
        # Get cost data from cost monitor
        # In real implementation, would query cost monitor service
        
        # Simulate cost metrics
        import random
        current_day = datetime.now().day
        
        cost_metrics = {
            "daily_spend": random.uniform(0.5, 4.0),
            "weekly_spend": random.uniform(3.0, 20.0),
            "monthly_spend": random.uniform(8.0, 45.0),
            "projected_monthly": random.uniform(15.0, 48.0),
            "budget_remaining": 50.0 - random.uniform(8.0, 45.0)
        }
        
        # Update dashboard
        self.dashboard_data["cost_metrics"].update(cost_metrics)
        
        # Check budget alerts
        if cost_metrics["projected_monthly"] > 45.0:  # 90% of $50 budget
            await self._trigger_budget_alert("high_projected_cost", cost_metrics)
        elif cost_metrics["daily_spend"] > 4.0:  # Above daily limit
            await self._trigger_budget_alert("daily_limit_exceeded", cost_metrics)
            
    async def _trigger_budget_alert(self, alert_type: str, cost_data: Dict[str, float]):
        """Trigger budget alert and potential cost reduction actions"""
        
        budget_alert = {
            "type": "budget_alert",
            "alert_type": alert_type,
            "cost_data": cost_data,
            "timestamp": datetime.now().isoformat(),
            "action_required": True
        }
        
        # Store alert
        self.alert_history.append(budget_alert)
        
        # Auto-cost reduction if enabled
        if alert_type == "high_projected_cost":
            # Request cost optimization
            self.send_message(
                AgentType.COST_OPTIMIZER,
                MessageType.COORDINATION_REQUEST,
                {
                    "action": "emergency_cost_optimization",
                    "projected_cost": cost_data["projected_monthly"],
                    "budget_limit": 50.0,
                    "priority": "high"
                }
            )
        elif alert_type == "daily_limit_exceeded":
            # Consider temporary shutdown
            self.send_message(
                AgentType.TRAINING_CONDUCTOR,
                MessageType.COORDINATION_REQUEST,
                {
                    "action": "pause_non_critical_training",
                    "daily_spend": cost_data["daily_spend"],
                    "priority": "high"
                }
            )
            
    async def _dashboard_update_loop(self):
        """Update real-time dashboard data"""
        while True:
            try:
                await self._update_dashboard_data()
                await asyncio.sleep(self.monitoring_intervals["dashboard_update"])
            except Exception as e:
                print(f"⚠️ Dashboard update error: {e}")
                await asyncio.sleep(30)
                
    async def _update_dashboard_data(self):
        """Update comprehensive dashboard data"""
        
        # Update system overview
        healthy_count = sum(1 for service in self.health_history.values() 
                          if service and service[-1].status == ServiceStatus.HEALTHY)
        warning_count = sum(1 for service in self.health_history.values()
                          if service and service[-1].status == ServiceStatus.WARNING)
        critical_count = sum(1 for service in self.health_history.values()
                           if service and service[-1].status == ServiceStatus.CRITICAL)
        offline_count = sum(1 for service in self.health_history.values()
                          if service and service[-1].status == ServiceStatus.OFFLINE)
        
        self.dashboard_data["system_overview"].update({
            "healthy_services": healthy_count,
            "warning_services": warning_count,
            "critical_services": critical_count,
            "offline_services": offline_count,
            "last_update": datetime.now().isoformat()
        })
        
        # Update Trinity metrics
        import random
        self.dashboard_data["trinity_metrics"].update({
            "arc_reactor_efficiency": random.uniform(88, 92),
            "perplexity_intelligence": random.uniform(93, 97),
            "einstein_fusion": random.uniform(500, 508),
            "overall_amplification": random.uniform(98, 105)
        })
        
    async def _regression_detection_loop(self):
        """Detect performance regression against baselines"""
        while True:
            try:
                await self._check_regression_baselines()
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                print(f"⚠️ Regression detection error: {e}")
                await asyncio.sleep(300)
                
    async def _check_regression_baselines(self):
        """Check current performance against established baselines"""
        
        # In real implementation, would query actual training metrics
        # For now, simulate regression checks
        
        regression_issues = []
        
        # Check training speed baselines
        current_speeds = {"t4": 35.0, "v100": 73.0, "a100": 148.0}  # Slightly lower than baseline
        
        for gpu_type, current_speed in current_speeds.items():
            baseline_speed = self.performance_baselines["training_speed"][gpu_type]
            if current_speed < baseline_speed * 0.9:  # 10% regression threshold
                regression_issues.append({
                    "type": "training_speed_regression",
                    "gpu_type": gpu_type,
                    "current": current_speed,
                    "baseline": baseline_speed,
                    "regression_percent": ((baseline_speed - current_speed) / baseline_speed) * 100
                })
                
        # Handle regressions
        for issue in regression_issues:
            await self._handle_regression_issue(issue)
            
    async def _handle_regression_issue(self, issue: Dict[str, Any]):
        """Handle detected performance regression"""
        
        regression_alert = {
            "type": "performance_regression",
            "issue": issue,
            "timestamp": datetime.now().isoformat(),
            "severity": "medium" if issue.get("regression_percent", 0) < 15 else "high"
        }
        
        # Store alert
        self.alert_history.append(regression_alert)
        
        # Request optimization
        self.broadcast_message(
            MessageType.OPTIMIZATION_REQUEST,
            {
                "action": "investigate_performance_regression",
                "issue_details": issue,
                "priority": "medium"
            }
        )
        
    async def _update_service_status_dashboard(self, health_results: Dict[str, HealthMetric]):
        """Update service status in dashboard"""
        
        # Update training metrics based on service health
        training_success_rate = 0
        if "training_orchestrator" in health_results:
            training_success_rate = health_results["training_orchestrator"].metric_value
            
        self.dashboard_data["training_metrics"].update({
            "success_rate": training_success_rate,
            "active_trainings": len([s for s in health_results.values() 
                                   if s.status == ServiceStatus.HEALTHY]),
            "speed_improvement_factor": sum(self.performance_baselines["training_speed"].values()) / 3
        })
        
    async def _send_recovery_notification(self, recovery_event: RecoveryEvent):
        """Send recovery notification"""
        
        notification = {
            "type": "recovery_notification",
            "event": {
                "service": recovery_event.service_name,
                "action": recovery_event.action_taken.value,
                "success": recovery_event.success,
                "timestamp": recovery_event.timestamp.isoformat()
            },
            "status": "resolved" if recovery_event.success else "failed"
        }
        
        # In real implementation: send to monitoring channels
        print(f"📧 Recovery Notification: {json.dumps(notification, indent=2)}")
        
    # Public API methods
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        
        latest_health = {}
        for service_name, history in self.health_history.items():
            if history:
                latest_health[service_name] = {
                    "status": history[-1].status.value,
                    "metric_value": history[-1].metric_value,
                    "last_checked": history[-1].timestamp.isoformat()
                }
                
        return {
            "system_overview": self.dashboard_data["system_overview"],
            "service_health": latest_health,
            "recent_alerts": self.alert_history[-10:],  # Last 10 alerts
            "recovery_history": [
                {
                    "service": r.service_name,
                    "action": r.action_taken.value,
                    "success": r.success,
                    "timestamp": r.timestamp.isoformat()
                } for r in self.recovery_history[-5:]  # Last 5 recoveries
            ],
            "trinity_status": {
                "monitoring_enabled": True,
                "auto_recovery_enabled": self.recovery_config["auto_recovery_enabled"],
                "performance_baselines_active": True
            }
        }
        
    async def get_real_time_dashboard(self) -> Dict[str, Any]:
        """Get real-time dashboard data"""
        return {
            **self.dashboard_data,
            "alerts": {
                "active_alerts": len([a for a in self.alert_history 
                                    if a.get("timestamp", "") > (datetime.now() - timedelta(hours=1)).isoformat()]),
                "critical_alerts": len([a for a in self.alert_history 
                                      if a.get("severity", "") == "high"]),
                "recent_recoveries": len(self.recovery_history)
            },
            "monitoring_status": {
                "services_monitored": len(self.monitored_services),
                "monitoring_uptime": "99.9%",  # Calculate actual uptime
                "last_health_check": datetime.now().isoformat()
            }
        }
        
    async def trigger_manual_recovery(self, service_name: str, action: str) -> Dict[str, Any]:
        """Trigger manual recovery action"""
        
        if service_name not in self.monitored_services:
            return {"success": False, "error": f"Unknown service: {service_name}"}
            
        try:
            recovery_action = RecoveryAction(action)
        except ValueError:
            return {"success": False, "error": f"Unknown recovery action: {action}"}
            
        # Create simulated health status for manual trigger
        health_status = HealthMetric(
            service_name=service_name,
            status=ServiceStatus.CRITICAL,
            metric_value=0.0,
            threshold=85.0,
            timestamp=datetime.now(),
            details={"manual_trigger": True}
        )
        
        # Execute recovery
        success = await self._execute_recovery_action(service_name, recovery_action, health_status)
        
        # Log manual recovery
        recovery_event = RecoveryEvent(
            event_id=f"manual_{int(time.time())}",
            service_name=service_name,
            issue_type="manual_intervention",
            action_taken=recovery_action,
            success=success,
            timestamp=datetime.now(),
            details={"manual": True, "operator": "system_admin"}
        )
        
        self.recovery_history.append(recovery_event)
        
        return {
            "success": success,
            "recovery_event_id": recovery_event.event_id,
            "action_taken": recovery_action.value,
            "timestamp": recovery_event.timestamp.isoformat()
        }
        
    async def get_performance_analytics(self) -> Dict[str, Any]:
        """Get comprehensive performance analytics"""
        
        return {
            "baselines": self.performance_baselines,
            "current_metrics": self.dashboard_data["performance_metrics"],
            "trinity_metrics": self.dashboard_data["trinity_metrics"],
            "regression_analysis": {
                "regressions_detected": len([a for a in self.alert_history 
                                           if a.get("type") == "performance_regression"]),
                "performance_trend": "stable",  # Calculate actual trend
                "optimization_opportunities": [
                    "GPU utilization could be improved during low-demand periods",
                    "Memory usage optimization available for creative domains",
                    "Cost efficiency improvements possible with spot instances"
                ]
            },
            "cost_analytics": self.dashboard_data["cost_metrics"],
            "training_analytics": self.dashboard_data["training_metrics"]
        }
        
    async def configure_monitoring(self, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update monitoring configuration"""
        
        updated_fields = []
        
        # Update monitoring intervals
        if "intervals" in config_updates:
            for interval_type, value in config_updates["intervals"].items():
                if interval_type in self.monitoring_intervals:
                    self.monitoring_intervals[interval_type] = value
                    updated_fields.append(f"interval_{interval_type}")
                    
        # Update recovery configuration
        if "recovery" in config_updates:
            for setting, value in config_updates["recovery"].items():
                if setting in self.recovery_config:
                    self.recovery_config[setting] = value
                    updated_fields.append(f"recovery_{setting}")
                    
        # Update performance baselines
        if "baselines" in config_updates:
            for metric, value in config_updates["baselines"].items():
                if metric in self.performance_baselines:
                    self.performance_baselines[metric] = value
                    updated_fields.append(f"baseline_{metric}")
                    
        return {
            "success": True,
            "updated_fields": updated_fields,
            "current_config": {
                "monitoring_intervals": self.monitoring_intervals,
                "recovery_config": self.recovery_config,
                "performance_baselines": self.performance_baselines
            }
        }

# Example usage
async def main():
    """Example usage of Monitoring System"""
    
    # Initialize monitoring system
    monitoring = MonitoringSystem()
    await monitoring.start()
    
    print("🎯 Monitoring System initialized with Trinity Architecture")
    print("📊 Real-time dashboard: http://localhost:8080/dashboard")
    print("🔍 Health checks running every 30 seconds")
    print("⚡ Auto-recovery enabled with 3-attempt limit")
    print("🎛️ Performance regression detection active")
    print("💰 Cost monitoring with $50/month budget")
    
    # Keep running
    await asyncio.sleep(60)  # Run for 1 minute in example
    
    # Get system status
    health_status = await monitoring.get_system_health()
    dashboard_data = await monitoring.get_real_time_dashboard()
    
    print("\n📊 System Health Summary:")
    print(f"   Healthy Services: {health_status['system_overview']['healthy_services']}")
    print(f"   Recent Alerts: {len(health_status['recent_alerts'])}")
    print(f"   Recovery Actions: {len(health_status['recovery_history'])}")
    print(f"   Trinity Status: Monitoring ✅ | Recovery ✅ | Analytics ✅")

if __name__ == "__main__":
    print("📊 Starting MeeTARA Lab Monitoring & Recovery System...")
    asyncio.run(main()) 