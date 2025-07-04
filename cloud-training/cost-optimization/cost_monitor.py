"""
MeeTARA Lab - Cost Optimization Monitor
Keep GPU training under $50/month target
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class CostMetrics:
    daily_spend: float = 0.0
    weekly_spend: float = 0.0
    monthly_spend: float = 0.0
    hourly_rate: float = 0.0
    projected_monthly: float = 0.0

class CostMonitor:
    """Monitor and optimize training costs"""
    
    def __init__(self):
        self.cost_metrics = CostMetrics()
        self.cost_limits = {
            "daily": 5.0,    # $5/day max
            "weekly": 25.0,  # $25/week max  
            "monthly": 50.0  # $50/month target
        }
        self.cost_alerts = []
        
    async def track_training_cost(self, provider: str, gpu_type: str, 
                                 duration_hours: float) -> float:
        """Track cost for a training session"""
        
        # GPU hourly rates (optimized pricing)
        rates = {
            "google_colab": {"T4": 0.35, "V100": 0.74, "A100": 1.28},
            "runpod": {"RTX_4090": 0.39, "A100": 0.79},
            "vast_ai": {"RTX_4090": 0.29, "A100": 0.59}
        }
        
        hourly_rate = rates.get(provider, {}).get(gpu_type, 1.0)
        session_cost = hourly_rate * duration_hours
        
        # Update metrics
        self.cost_metrics.daily_spend += session_cost
        self.cost_metrics.weekly_spend += session_cost
        self.cost_metrics.monthly_spend += session_cost
        self.cost_metrics.hourly_rate = hourly_rate
        
        # Calculate projection
        days_in_month = 30
        self.cost_metrics.projected_monthly = (
            self.cost_metrics.daily_spend * days_in_month
        )
        
        # Check limits
        await self._check_cost_limits()
        
        return session_cost
        
    async def _check_cost_limits(self):
        """Check if cost limits are exceeded"""
        alerts = []
        
        if self.cost_metrics.daily_spend > self.cost_limits["daily"]:
            alerts.append(f"⚠️ Daily limit exceeded: ${self.cost_metrics.daily_spend:.2f}")
            
        if self.cost_metrics.projected_monthly > self.cost_limits["monthly"]:
            alerts.append(f"📊 Monthly projection: ${self.cost_metrics.projected_monthly:.2f}")
            
        if alerts:
            self.cost_alerts.extend(alerts)
            await self._implement_cost_reduction()
            
    async def _implement_cost_reduction(self):
        """Implement cost reduction strategies"""
        strategies = [
            "Switch to cheaper GPU instances",
            "Use spot instances where available", 
            "Reduce batch size to speed up training",
            "Schedule training during off-peak hours"
        ]
        
        print("💰 Implementing cost reduction:")
        for strategy in strategies:
            print(f"  - {strategy}")
            
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get comprehensive cost summary"""
        return {
            "current_metrics": {
                "daily": f"${self.cost_metrics.daily_spend:.2f}",
                "weekly": f"${self.cost_metrics.weekly_spend:.2f}", 
                "monthly": f"${self.cost_metrics.monthly_spend:.2f}",
                "projected_monthly": f"${self.cost_metrics.projected_monthly:.2f}"
            },
            "limits": {
                "daily": f"${self.cost_limits['daily']:.2f}",
                "weekly": f"${self.cost_limits['weekly']:.2f}",
                "monthly": f"${self.cost_limits['monthly']:.2f}"
            },
            "status": {
                "under_budget": self.cost_metrics.projected_monthly < self.cost_limits["monthly"],
                "alerts": self.cost_alerts[-5:],  # Last 5 alerts
                "optimization_needed": len(self.cost_alerts) > 0
            },
            "recommendations": self._get_cost_recommendations()
        }
        
    def _get_cost_recommendations(self) -> List[str]:
        """Get cost optimization recommendations"""
        recommendations = []
        
        if self.cost_metrics.projected_monthly > 40:
            recommendations.append("Consider using Vast.ai for cheaper A100 instances")
            
        if self.cost_metrics.hourly_rate > 1.0:
            recommendations.append("Switch to T4 instances for less critical domains")
            
        if self.cost_metrics.daily_spend > 3:
            recommendations.append("Enable auto-shutdown after training completion")
            
        return recommendations

# Global cost monitor
cost_monitor = CostMonitor() 
