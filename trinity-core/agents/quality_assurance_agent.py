"""
MeeTARA Lab - Quality Assurance Agent
Monitors training quality, validates outputs, and ensures 101% validation scores
"""

import asyncio
import time
import json
import statistics
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import yaml
from .mcp_protocol import BaseAgent, AgentType, MessageType, MCPMessage, mcp_protocol

class QualityAssuranceAgent(BaseAgent):
    """Quality monitoring and validation for training with TARA-level standards"""
    
    def __init__(self, mcp=None):
        super().__init__(AgentType.QUALITY_ASSURANCE, mcp or mcp_protocol)
        
        # Quality thresholds (TARA proven standards)
        self.quality_thresholds = {
            "validation_score": 0.85,  # Minimum 85% validation
            "target_validation": 1.01,  # Target 101% (TARA achievement)
            "loss_threshold": 0.5,     # Maximum loss threshold
            "convergence_rate": 0.1,   # Minimum convergence rate
            "data_quality_min": 0.80,  # Minimum data quality
            "emotional_intelligence_min": 0.75,  # Minimum EI score
            "crisis_handling_min": 0.90  # Minimum crisis handling score
        }
        
        # 62-Domain Configuration
        self.domain_mapping = {}
        self.domain_categories = {}
        self.category_quality_requirements = {}
        
        # Load domain configuration
        self._load_domain_configuration()
        
        # Monitoring domains and their metrics
        self.monitoring_domains = {}
        
        # Quality validation patterns
        self.validation_patterns = {
            "convergence_analysis": True,
            "loss_pattern_detection": True,
            "overfitting_detection": True,
            "quality_regression_analysis": True,
            "real_time_monitoring": True
        }
        
        # Alert and recovery mechanisms
        self.alert_history = []
        self.recovery_actions = {}
        
        # Quality benchmarks from TARA success
        self.tara_benchmarks = {
            "validation_scores": [1.01, 0.98, 0.99, 1.02, 0.97],  # Historical TARA scores
            "training_efficiency": 0.92,  # 92% efficiency target
            "model_consistency": 0.95,    # 95% consistency target
            "user_satisfaction": 0.89     # 89% user satisfaction
        }
        
    def _load_domain_configuration(self):
        """Load 62-domain configuration from cloud-optimized mapping"""
        try:
            config_path = Path("config/cloud-optimized-domain-mapping.yaml")
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    
                # Extract domain categories and their domains
                domain_categories = ['healthcare', 'daily_life', 'business', 'education', 'creative', 'technology', 'specialized']
                
                for category in domain_categories:
                    if category in config:
                        self.domain_mapping[category] = list(config[category].keys())
                        
                        # Map each domain to its category
                        for domain in config[category].keys():
                            self.domain_categories[domain] = category
                            
                # Initialize category-based quality requirements
                self._initialize_category_quality_requirements()
                
                total_domains = sum(len(domains) for domains in self.domain_mapping.values())
                print(f"✅ Quality Assurance: Loaded {total_domains} domains across {len(self.domain_mapping)} categories")
                
            else:
                print("⚠️ Domain mapping not found, using default configuration")
                self._initialize_default_configuration()
                
        except Exception as e:
            print(f"⚠️ Error loading domain configuration: {e}")
            self._initialize_default_configuration()
            
    def _initialize_default_configuration(self):
        """Initialize default domain configuration if YAML not available"""
        self.domain_mapping = {
            "healthcare": ["general_health", "mental_health", "nutrition", "fitness"],
            "business": ["entrepreneurship", "marketing", "sales", "customer_service"],
            "education": ["academic_tutoring", "skill_development", "career_guidance"],
            "creative": ["writing", "storytelling", "content_creation"],
            "technology": ["programming", "ai_ml", "cybersecurity"],
            "specialized": ["legal", "financial", "scientific_research"]
        }
        
        # Map domains to categories
        for category, domains in self.domain_mapping.items():
            for domain in domains:
                self.domain_categories[domain] = category
                
        self._initialize_category_quality_requirements()
        
    def _initialize_category_quality_requirements(self):
        """Initialize quality requirements based on domain categories"""
        self.category_quality_requirements = {
            "healthcare": {
                "validation_score": 0.95,
                "crisis_handling_min": 0.95,
                "data_quality_min": 0.90,
                "emotional_intelligence_min": 0.90,
                "safety_critical": True
            },
            "specialized": {
                "validation_score": 0.92,
                "crisis_handling_min": 0.88,
                "data_quality_min": 0.88,
                "emotional_intelligence_min": 0.85,
                "safety_critical": True
            },
            "business": {
                "validation_score": 0.88,
                "crisis_handling_min": 0.85,
                "data_quality_min": 0.85,
                "emotional_intelligence_min": 0.80,
                "safety_critical": False
            },
            "education": {
                "validation_score": 0.87,
                "crisis_handling_min": 0.82,
                "data_quality_min": 0.82,
                "emotional_intelligence_min": 0.78,
                "safety_critical": False
            },
            "technology": {
                "validation_score": 0.87,
                "crisis_handling_min": 0.82,
                "data_quality_min": 0.82,
                "emotional_intelligence_min": 0.75,
                "safety_critical": False
            },
            "daily_life": {
                "validation_score": 0.85,
                "crisis_handling_min": 0.80,
                "data_quality_min": 0.80,
                "emotional_intelligence_min": 0.75,
                "safety_critical": False
            },
            "creative": {
                "validation_score": 0.82,
                "crisis_handling_min": 0.78,
                "data_quality_min": 0.78,
                "emotional_intelligence_min": 0.75,
                "safety_critical": False
            }
        }
        
    async def start(self):
        """Start the Quality Assurance Agent"""
        await super().start()
        
        # Initialize quality monitoring systems
        await self._initialize_quality_systems()
        
        # Start real-time monitoring loop
        asyncio.create_task(self._real_time_monitoring_loop())
        
        # Start quality analysis loop
        asyncio.create_task(self._quality_analysis_loop())
        
        print("🔍 Quality Assurance Agent started")
        print(f"   → Target validation score: {self.quality_thresholds['target_validation']}")
        print(f"   → Quality patterns: {len(self.validation_patterns)} active")
        print(f"   → TARA benchmarks loaded: {len(self.tara_benchmarks)} metrics")
        
    async def handle_mcp_message(self, message: MCPMessage):
        """Handle incoming MCP messages"""
        if message.message_type == MessageType.COORDINATION_REQUEST:
            await self._handle_coordination_request(message.data)
        elif message.message_type == MessageType.TRAINING_PROGRESS:
            await self._monitor_training_progress(message.data)
        elif message.message_type == MessageType.STATUS_UPDATE:
            await self._process_status_update(message.data)
            
    async def _initialize_quality_systems(self):
        """Initialize quality monitoring and validation systems"""
        
        # Load quality configuration if available
        try:
            config_path = Path("config/quality_standards.json")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    quality_config = json.load(f)
                    self.quality_thresholds.update(quality_config.get("thresholds", {}))
                    print("✅ Quality standards loaded from config")
        except Exception as e:
            print(f"⚠️ Using default quality standards: {e}")
            
        # Initialize validation metrics
        self.validation_metrics = {
            "accuracy_tracking": [],
            "loss_progression": [],
            "convergence_analysis": [],
            "quality_consistency": [],
            "emotional_intelligence_scores": [],
            "crisis_handling_scores": []
        }
        
    async def _real_time_monitoring_loop(self):
        """Real-time quality monitoring loop"""
        while self.running:
            try:
                # Monitor all active domains
                for domain in self.monitoring_domains:
                    await self._check_domain_quality(domain)
                    
                # Check for quality alerts
                await self._process_quality_alerts()
                
                # Broadcast quality status
                self.broadcast_message(
                    MessageType.QUALITY_METRICS,
                    {
                        "agent": "quality_assurance",
                        "overall_quality": await self._calculate_overall_quality(),
                        "domain_statuses": await self._get_domain_statuses(),
                        "alerts": len(self.alert_history),
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                print(f"❌ Real-time monitoring error: {e}")
                await asyncio.sleep(30)
                
    async def _quality_analysis_loop(self):
        """Deep quality analysis loop"""
        while self.running:
            try:
                # Perform comprehensive quality analysis
                await self._comprehensive_quality_analysis()
                
                # Generate quality reports
                await self._generate_quality_reports()
                
                # Update quality benchmarks
                await self._update_quality_benchmarks()
                
                await asyncio.sleep(60)  # Deep analysis every minute
                
            except Exception as e:
                print(f"❌ Quality analysis error: {e}")
                await asyncio.sleep(120)
                
    async def _handle_coordination_request(self, data: Dict[str, Any]):
        """Handle coordination requests from Training Conductor"""
        action = data.get("action")
        
        if action == "start_monitoring":
            await self._start_quality_monitoring(data)
        elif action == "validate_model_quality":
            await self._validate_model_quality(data)
        elif action == "quality_checkpoint":
            await self._perform_quality_checkpoint(data)
        elif action == "emergency_quality_check":
            await self._emergency_quality_check(data)
            
    async def _start_quality_monitoring(self, data: Dict[str, Any]):
        """Start quality monitoring for domain training"""
        domain = data.get("domain")
        quality_thresholds = data.get("quality_thresholds", self.quality_thresholds)
        monitoring_config = data.get("monitoring_config", {})
        
        print(f"🔍 Starting quality monitoring for {domain}")
        
        # Initialize domain monitoring
        self.monitoring_domains[domain] = {
            "start_time": datetime.now(),
            "quality_thresholds": quality_thresholds,
            "monitoring_config": monitoring_config,
            "samples_processed": 0,
            "quality_scores": [],
            "validation_scores": [],
            "loss_values": [],
            "convergence_metrics": [],
            "alerts": [],
            "status": "monitoring_active",
            "last_checkpoint": datetime.now()
        }
        
        # Set domain-specific quality targets based on category
        category = self.domain_categories.get(domain) if hasattr(self, 'domain_categories') else None
        
        if category and hasattr(self, 'category_quality_requirements') and category in self.category_quality_requirements:
            # Use category-based requirements
            category_reqs = self.category_quality_requirements[category]
            self.monitoring_domains[domain]["quality_thresholds"]["validation_score"] = category_reqs["validation_score"]
            self.monitoring_domains[domain]["quality_thresholds"]["crisis_handling_min"] = category_reqs["crisis_handling_min"]
            self.monitoring_domains[domain]["quality_thresholds"]["data_quality_min"] = category_reqs["data_quality_min"]
            self.monitoring_domains[domain]["quality_thresholds"]["emotional_intelligence_min"] = category_reqs["emotional_intelligence_min"]
        else:
            # Fallback to hardcoded mapping for backward compatibility
            if domain in ["healthcare", "mental_health"]:
                # Higher standards for critical domains
                self.monitoring_domains[domain]["quality_thresholds"]["validation_score"] = 0.90
                self.monitoring_domains[domain]["quality_thresholds"]["crisis_handling_min"] = 0.95
            elif domain in ["finance", "legal"]:
                # High accuracy requirements
                self.monitoring_domains[domain]["quality_thresholds"]["validation_score"] = 0.88
                self.monitoring_domains[domain]["quality_thresholds"]["data_quality_min"] = 0.85
            
        print(f"✅ Quality monitoring configured for {domain}")
        print(f"   → Validation target: {quality_thresholds.get('validation_score', 0.85)}")
        print(f"   → TARA target: {self.quality_thresholds['target_validation']}")
        
        # Send monitoring confirmation
        self.send_message(
            AgentType.CONDUCTOR,
            MessageType.STATUS_UPDATE,
            {
                "action": "monitoring_started",
                "domain": domain,
                "quality_thresholds": quality_thresholds,
                "monitoring_id": f"qa_{domain}_{int(time.time())}"
            }
        )
        
    async def _monitor_training_progress(self, data: Dict[str, Any]):
        """Monitor training progress and quality metrics"""
        domain = data.get("domain")
        
        if domain not in self.monitoring_domains:
            print(f"⚠️ Received progress update for unmonitored domain: {domain}")
            return
            
        # Extract quality metrics from training progress
        validation_score = data.get("validation_score", 0.0)
        loss = data.get("loss", 1.0)
        training_step = data.get("training_step", 0)
        
        # Update monitoring data
        monitoring = self.monitoring_domains[domain]
        monitoring["samples_processed"] = training_step
        monitoring["validation_scores"].append(validation_score)
        monitoring["loss_values"].append(loss)
        monitoring["quality_scores"].append(validation_score)
        
        # Calculate convergence metrics
        convergence_rate = await self._calculate_convergence_rate(domain)
        monitoring["convergence_metrics"].append(convergence_rate)
        
        print(f"📊 Quality check {domain}: validation={validation_score:.3f}, loss={loss:.3f}, convergence={convergence_rate:.3f}")
        
        # Check quality thresholds
        await self._check_quality_thresholds(domain, validation_score, loss, convergence_rate)
        
        # Detect quality patterns
        await self._detect_quality_patterns(domain)
        
    async def _check_quality_thresholds(self, domain: str, validation_score: float, 
                                      loss: float, convergence_rate: float):
        """Check if quality metrics meet thresholds"""
        
        monitoring = self.monitoring_domains[domain]
        thresholds = monitoring["quality_thresholds"]
        
        alerts = []
        
        # Validation score check
        if validation_score < thresholds["validation_score"]:
            alerts.append({
                "type": "validation_below_threshold",
                "severity": "warning",
                "metric": validation_score,
                "threshold": thresholds["validation_score"],
                "message": f"Validation score {validation_score:.3f} below threshold {thresholds['validation_score']}"
            })
            
        # Loss threshold check
        if loss > thresholds["loss_threshold"]:
            alerts.append({
                "type": "loss_above_threshold", 
                "severity": "warning",
                "metric": loss,
                "threshold": thresholds["loss_threshold"],
                "message": f"Loss {loss:.3f} above threshold {thresholds['loss_threshold']}"
            })
            
        # Convergence rate check
        if convergence_rate < thresholds["convergence_rate"]:
            alerts.append({
                "type": "slow_convergence",
                "severity": "info",
                "metric": convergence_rate,
                "threshold": thresholds["convergence_rate"],
                "message": f"Convergence rate {convergence_rate:.3f} below expected {thresholds['convergence_rate']}"
            })
            
        # Check for TARA-level achievement
        if validation_score >= self.quality_thresholds["target_validation"]:
            alerts.append({
                "type": "tara_level_achieved",
                "severity": "success",
                "metric": validation_score,
                "threshold": self.quality_thresholds["target_validation"],
                "message": f"🏆 TARA-level quality achieved: {validation_score:.3f}"
            })
            
        # Process alerts
        for alert in alerts:
            await self._process_quality_alert(domain, alert)
            
    async def _detect_quality_patterns(self, domain: str):
        """Detect quality patterns and trends"""
        
        monitoring = self.monitoring_domains[domain]
        validation_scores = monitoring["validation_scores"]
        loss_values = monitoring["loss_values"]
        
        if len(validation_scores) < 5:
            return  # Need more data for pattern detection
            
        # Check for quality regression
        recent_scores = validation_scores[-5:]
        if len(recent_scores) >= 3:
            trend = await self._calculate_trend(recent_scores)
            
            if trend < -0.05:  # Decreasing trend
                await self._process_quality_alert(domain, {
                    "type": "quality_regression",
                    "severity": "warning",
                    "message": f"Quality regression detected: trend {trend:.3f}",
                    "trend": trend
                })
                
        # Check for overfitting
        if len(validation_scores) >= 10:
            overfitting_score = await self._detect_overfitting(domain)
            if overfitting_score > 0.7:
                await self._process_quality_alert(domain, {
                    "type": "overfitting_detected",
                    "severity": "warning",
                    "message": f"Potential overfitting detected: score {overfitting_score:.3f}",
                    "overfitting_score": overfitting_score
                })
                
    async def _calculate_convergence_rate(self, domain: str) -> float:
        """Calculate training convergence rate"""
        
        monitoring = self.monitoring_domains[domain]
        loss_values = monitoring["loss_values"]
        
        if len(loss_values) < 3:
            return 0.0
            
        # Calculate rate of loss decrease
        recent_losses = loss_values[-5:]  # Last 5 measurements
        if len(recent_losses) >= 2:
            initial_loss = recent_losses[0]
            final_loss = recent_losses[-1]
            
            if initial_loss > 0:
                convergence_rate = (initial_loss - final_loss) / initial_loss
                return max(0.0, convergence_rate)
                
        return 0.0
        
    async def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in values (positive = increasing, negative = decreasing)"""
        
        if len(values) < 2:
            return 0.0
            
        # Simple linear trend calculation
        x_values = list(range(len(values)))
        n = len(values)
        
        sum_x = sum(x_values)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(x_values, values))
        sum_x2 = sum(x * x for x in x_values)
        
        # Linear regression slope
        numerator = n * sum_xy - sum_x * sum_y
        denominator = n * sum_x2 - sum_x * sum_x
        
        if denominator == 0:
            return 0.0
            
        slope = numerator / denominator
        return slope
        
    async def _detect_overfitting(self, domain: str) -> float:
        """Detect potential overfitting"""
        
        monitoring = self.monitoring_domains[domain]
        validation_scores = monitoring["validation_scores"]
        loss_values = monitoring["loss_values"]
        
        if len(validation_scores) < 10 or len(loss_values) < 10:
            return 0.0
            
        # Check for divergence between loss and validation
        recent_losses = loss_values[-5:]
        recent_validations = validation_scores[-5:]
        
        loss_trend = await self._calculate_trend(recent_losses)
        validation_trend = await self._calculate_trend(recent_validations)
        
        # Overfitting indicator: loss decreasing but validation stagnating/decreasing
        if loss_trend < -0.1 and validation_trend > -0.02:
            return 0.8  # High overfitting probability
        elif loss_trend < -0.05 and validation_trend > -0.01:
            return 0.5  # Medium overfitting probability
            
        return 0.2  # Low overfitting probability
        
    async def _process_quality_alert(self, domain: str, alert: Dict[str, Any]):
        """Process quality alert and take appropriate action"""
        
        alert["domain"] = domain
        alert["timestamp"] = datetime.now().isoformat()
        
        # Add to alert history
        self.alert_history.append(alert)
        self.monitoring_domains[domain]["alerts"].append(alert)
        
        # Log alert
        severity_icons = {
            "info": "ℹ️",
            "warning": "⚠️", 
            "error": "❌",
            "success": "✅"
        }
        
        icon = severity_icons.get(alert["severity"], "📋")
        print(f"{icon} Quality Alert [{domain}]: {alert['message']}")
        
        # Send alert to Training Conductor
        priority = 1 if alert["severity"] == "info" else 2 if alert["severity"] == "warning" else 3
        
        self.send_message(
            AgentType.CONDUCTOR,
            MessageType.ERROR_NOTIFICATION if alert["severity"] in ["warning", "error"] else MessageType.QUALITY_METRICS,
            {
                "alert_type": alert["type"],
                "domain": domain,
                "severity": alert["severity"],
                "message": alert["message"],
                "metric_value": alert.get("metric"),
                "threshold": alert.get("threshold"),
                "suggested_actions": await self._get_suggested_actions(alert)
            },
            priority=priority
        )
        
    async def _get_suggested_actions(self, alert: Dict[str, Any]) -> List[str]:
        """Get suggested actions for quality alerts"""
        
        alert_type = alert["type"]
        
        action_suggestions = {
            "validation_below_threshold": [
                "Increase training data quality",
                "Adjust learning rate",
                "Check for data distribution issues",
                "Consider model architecture changes"
            ],
            "loss_above_threshold": [
                "Reduce learning rate",
                "Increase training duration",
                "Check gradient clipping",
                "Verify data preprocessing"
            ],
            "quality_regression": [
                "Implement early stopping",
                "Reduce learning rate",
                "Check for overfitting",
                "Validate data quality"
            ],
            "overfitting_detected": [
                "Implement regularization",
                "Reduce model complexity",
                "Increase training data",
                "Add dropout layers"
            ],
            "slow_convergence": [
                "Increase learning rate",
                "Check optimizer settings",
                "Verify data preprocessing",
                "Consider learning rate scheduling"
            ]
        }
        
        return action_suggestions.get(alert_type, ["Review training parameters", "Consult quality guidelines"])
        
    async def _check_domain_quality(self, domain: str):
        """Check overall quality for a domain"""
        
        monitoring = self.monitoring_domains[domain]
        
        # Check if monitoring is stale
        last_update = monitoring.get("last_checkpoint", monitoring["start_time"])
        if datetime.now() - last_update > timedelta(minutes=10):
            await self._process_quality_alert(domain, {
                "type": "monitoring_stale",
                "severity": "warning",
                "message": f"No quality updates received for {domain} in 10 minutes"
            })
            
    async def _calculate_overall_quality(self) -> float:
        """Calculate overall quality score across all domains"""
        
        if not self.monitoring_domains:
            return 0.0
            
        domain_qualities = []
        
        for domain, monitoring in self.monitoring_domains.items():
            if monitoring["validation_scores"]:
                recent_quality = statistics.mean(monitoring["validation_scores"][-5:])
                domain_qualities.append(recent_quality)
                
        if domain_qualities:
            return statistics.mean(domain_qualities)
        else:
            return 0.0
            
    async def _get_domain_statuses(self) -> Dict[str, str]:
        """Get status for all monitored domains"""
        
        statuses = {}
        
        for domain, monitoring in self.monitoring_domains.items():
            if not monitoring["validation_scores"]:
                statuses[domain] = "initializing"
            else:
                recent_score = monitoring["validation_scores"][-1]
                target_score = monitoring["quality_thresholds"]["validation_score"]
                
                if recent_score >= self.quality_thresholds["target_validation"]:
                    statuses[domain] = "excellent"  # TARA level
                elif recent_score >= target_score:
                    statuses[domain] = "good"
                elif recent_score >= target_score * 0.9:
                    statuses[domain] = "acceptable"
                else:
                    statuses[domain] = "needs_improvement"
                    
        return statuses
        
    async def _comprehensive_quality_analysis(self):
        """Perform comprehensive quality analysis"""
        
        for domain in self.monitoring_domains:
            monitoring = self.monitoring_domains[domain]
            
            if len(monitoring["validation_scores"]) >= 10:
                # Comprehensive analysis for domains with sufficient data
                analysis = {
                    "domain": domain,
                    "quality_trend": await self._calculate_trend(monitoring["validation_scores"]),
                    "convergence_stability": await self._calculate_convergence_stability(domain),
                    "quality_consistency": await self._calculate_quality_consistency(domain),
                    "tara_benchmark_comparison": await self._compare_to_tara_benchmarks(domain)
                }
                
                # Store analysis results
                monitoring["latest_analysis"] = analysis
                
    async def _calculate_convergence_stability(self, domain: str) -> float:
        """Calculate how stable the convergence is"""
        
        monitoring = self.monitoring_domains[domain]
        convergence_metrics = monitoring["convergence_metrics"]
        
        if len(convergence_metrics) < 5:
            return 0.0
            
        # Calculate standard deviation of convergence rates
        recent_convergence = convergence_metrics[-10:]
        return 1.0 - min(1.0, statistics.stdev(recent_convergence))
        
    async def _calculate_quality_consistency(self, domain: str) -> float:
        """Calculate quality consistency score"""
        
        monitoring = self.monitoring_domains[domain]
        validation_scores = monitoring["validation_scores"]
        
        if len(validation_scores) < 5:
            return 0.0
            
        # Calculate coefficient of variation (lower is more consistent)
        recent_scores = validation_scores[-10:]
        mean_score = statistics.mean(recent_scores)
        
        if mean_score > 0:
            cv = statistics.stdev(recent_scores) / mean_score
            return max(0.0, 1.0 - cv)
        else:
            return 0.0
            
    async def _compare_to_tara_benchmarks(self, domain: str) -> Dict[str, float]:
        """Compare domain quality to TARA benchmarks"""
        
        monitoring = self.monitoring_domains[domain]
        validation_scores = monitoring["validation_scores"]
        
        if not validation_scores:
            return {"benchmark_comparison": 0.0}
            
        # Compare to TARA benchmark scores
        current_avg = statistics.mean(validation_scores[-5:])
        tara_avg = statistics.mean(self.tara_benchmarks["validation_scores"])
        
        comparison = current_avg / tara_avg if tara_avg > 0 else 0.0
        
        return {
            "benchmark_comparison": comparison,
            "current_average": current_avg,
            "tara_average": tara_avg,
            "meets_tara_standard": comparison >= 0.95
        }
        
    async def _generate_quality_reports(self):
        """Generate quality reports for all domains"""
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_quality": await self._calculate_overall_quality(),
            "domains": {},
            "alerts_summary": {
                "total_alerts": len(self.alert_history),
                "recent_alerts": len([a for a in self.alert_history if datetime.fromisoformat(a["timestamp"]) > datetime.now() - timedelta(hours=1)])
            }
        }
        
        for domain, monitoring in self.monitoring_domains.items():
            if monitoring["validation_scores"]:
                report["domains"][domain] = {
                    "average_quality": statistics.mean(monitoring["validation_scores"]),
                    "latest_quality": monitoring["validation_scores"][-1],
                    "total_samples": monitoring["samples_processed"],
                    "alerts": len(monitoring["alerts"]),
                    "status": monitoring["status"]
                }
                
        # Store report
        report_path = Path(f"reports/quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
    async def _update_quality_benchmarks(self):
        """Update quality benchmarks based on recent performance"""
        
        # Update benchmarks based on consistent high performance
        all_recent_scores = []
        
        for monitoring in self.monitoring_domains.values():
            if len(monitoring["validation_scores"]) >= 5:
                recent_scores = monitoring["validation_scores"][-5:]
                all_recent_scores.extend(recent_scores)
                
        if len(all_recent_scores) >= 10:
            avg_recent = statistics.mean(all_recent_scores)
            
            # Update benchmarks if consistently exceeding current ones
            if avg_recent > statistics.mean(self.tara_benchmarks["validation_scores"]) * 1.05:
                self.tara_benchmarks["validation_scores"].append(avg_recent)
                # Keep only last 10 benchmark scores
                self.tara_benchmarks["validation_scores"] = self.tara_benchmarks["validation_scores"][-10:]
                
                print(f"📈 Quality benchmarks updated: new average {avg_recent:.3f}")

# Global instance
quality_assurance_agent = QualityAssuranceAgent() 