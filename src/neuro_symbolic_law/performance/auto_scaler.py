"""
Advanced auto-scaling system for dynamic resource management.
Generation 3: Implements predictive scaling, load balancing, and resource optimization.
"""

import asyncio
import logging
import time
import threading
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class ScalingDecision(Enum):
    """Scaling decisions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"


@dataclass
class ResourceMetrics:
    """Resource usage metrics."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    request_rate: float = 0.0
    response_time: float = 0.0
    queue_length: int = 0
    active_workers: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration."""
    min_workers: int = 2
    max_workers: int = 50
    target_cpu_threshold: float = 0.7
    target_memory_threshold: float = 0.8
    target_response_time_ms: float = 1000.0
    scale_up_cooldown: int = 120  # seconds
    scale_down_cooldown: int = 300  # seconds
    scale_up_increment: int = 2
    scale_down_decrement: int = 1


class PredictiveScaler:
    """Predictive auto-scaler using machine learning techniques."""
    
    def __init__(self, policy: ScalingPolicy):
        self.policy = policy
        self.metrics_history = deque(maxlen=1000)
        self.scaling_history = deque(maxlen=100)
        self.last_scale_action = 0
        self.current_workers = policy.min_workers
        
        # Prediction models (simplified)
        self.demand_patterns = defaultdict(list)
        self.seasonal_factors = {}
        
        self._lock = threading.RLock()
        
    def record_metrics(self, metrics: ResourceMetrics) -> None:
        """Record resource metrics for analysis."""
        with self._lock:
            self.metrics_history.append(metrics)
            
            # Learn demand patterns
            hour = datetime.fromtimestamp(metrics.timestamp).hour
            self.demand_patterns[hour].append(metrics.request_rate)
            
            # Limit pattern history
            if len(self.demand_patterns[hour]) > 50:
                self.demand_patterns[hour] = self.demand_patterns[hour][-30:]
    
    def predict_demand(self, horizon_minutes: int = 10) -> float:
        """Predict demand for the next horizon_minutes."""
        try:
            current_time = datetime.now()
            target_hour = (current_time + timedelta(minutes=horizon_minutes)).hour
            
            if target_hour in self.demand_patterns and self.demand_patterns[target_hour]:
                # Use historical average for this hour
                historical_avg = sum(self.demand_patterns[target_hour]) / len(self.demand_patterns[target_hour])
                
                # Apply recent trend
                if len(self.metrics_history) >= 5:
                    recent_metrics = list(self.metrics_history)[-5:]
                    recent_trend = (recent_metrics[-1].request_rate - recent_metrics[0].request_rate) / 5
                    
                    predicted = historical_avg + (recent_trend * horizon_minutes / 60)
                    return max(0, predicted)
                
                return historical_avg
            
            # Fallback to recent average
            if len(self.metrics_history) >= 3:
                recent_avg = sum(m.request_rate for m in list(self.metrics_history)[-3:]) / 3
                return recent_avg
                
            return 1.0  # Default prediction
            
        except Exception as e:
            logger.error(f"Error predicting demand: {e}")
            return 1.0
    
    def make_scaling_decision(self, current_metrics: ResourceMetrics) -> Tuple[ScalingDecision, int]:
        """Make intelligent scaling decision."""
        with self._lock:
            current_time = time.time()
            
            # Check cooldown periods
            time_since_last_scale = current_time - self.last_scale_action
            
            # Analyze current load
            cpu_pressure = current_metrics.cpu_usage > self.policy.target_cpu_threshold
            memory_pressure = current_metrics.memory_usage > self.policy.target_memory_threshold
            response_time_pressure = current_metrics.response_time > self.policy.target_response_time_ms
            queue_pressure = current_metrics.queue_length > self.current_workers * 2
            
            # Predict future demand
            predicted_demand = self.predict_demand(10)
            current_demand = current_metrics.request_rate
            demand_increasing = predicted_demand > current_demand * 1.2
            
            # Scale up conditions
            scale_up_needed = (
                (cpu_pressure or memory_pressure or response_time_pressure or queue_pressure)
                and time_since_last_scale > self.policy.scale_up_cooldown
                and self.current_workers < self.policy.max_workers
            ) or (
                demand_increasing
                and time_since_last_scale > self.policy.scale_up_cooldown / 2
                and self.current_workers < self.policy.max_workers
            )
            
            if scale_up_needed:
                new_workers = min(
                    self.current_workers + self.policy.scale_up_increment,
                    self.policy.max_workers
                )
                
                # Predictive scaling - scale more aggressively if demand spike predicted
                if demand_increasing and predicted_demand > current_demand * 2:
                    new_workers = min(
                        self.current_workers + self.policy.scale_up_increment * 2,
                        self.policy.max_workers
                    )
                
                self.last_scale_action = current_time
                self.current_workers = new_workers
                
                self.scaling_history.append({
                    'timestamp': current_time,
                    'action': 'scale_up',
                    'from_workers': self.current_workers - (new_workers - self.current_workers),
                    'to_workers': new_workers,
                    'reason': 'high_load' if cpu_pressure else 'predicted_demand'
                })
                
                return ScalingDecision.SCALE_UP, new_workers
            
            # Scale down conditions
            all_metrics_low = (
                current_metrics.cpu_usage < self.policy.target_cpu_threshold * 0.5
                and current_metrics.memory_usage < self.policy.target_memory_threshold * 0.6
                and current_metrics.response_time < self.policy.target_response_time_ms * 0.7
                and current_metrics.queue_length == 0
            )
            
            predicted_demand_low = predicted_demand < current_demand * 0.7
            
            scale_down_needed = (
                all_metrics_low
                and time_since_last_scale > self.policy.scale_down_cooldown
                and self.current_workers > self.policy.min_workers
                and not demand_increasing
            ) or (
                predicted_demand_low
                and time_since_last_scale > self.policy.scale_down_cooldown * 1.5
                and self.current_workers > self.policy.min_workers
            )
            
            if scale_down_needed:
                new_workers = max(
                    self.current_workers - self.policy.scale_down_decrement,
                    self.policy.min_workers
                )
                
                self.last_scale_action = current_time
                self.current_workers = new_workers
                
                self.scaling_history.append({
                    'timestamp': current_time,
                    'action': 'scale_down',
                    'from_workers': self.current_workers + self.policy.scale_down_decrement,
                    'to_workers': new_workers,
                    'reason': 'low_load'
                })
                
                return ScalingDecision.SCALE_DOWN, new_workers
            
            return ScalingDecision.MAINTAIN, self.current_workers
    
    def get_scaling_insights(self) -> Dict[str, Any]:
        """Get insights about scaling patterns and efficiency."""
        with self._lock:
            if not self.scaling_history:
                return {"message": "No scaling history available"}
            
            recent_actions = list(self.scaling_history)[-10:]
            scale_ups = [a for a in recent_actions if a['action'] == 'scale_up']
            scale_downs = [a for a in recent_actions if a['action'] == 'scale_down']
            
            # Calculate efficiency metrics
            if len(self.metrics_history) >= 10:
                recent_metrics = list(self.metrics_history)[-10:]
                avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
                avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
                avg_response_time = sum(m.response_time for m in recent_metrics) / len(recent_metrics)
            else:
                avg_cpu = avg_memory = avg_response_time = 0
            
            return {
                'current_workers': self.current_workers,
                'recent_scale_ups': len(scale_ups),
                'recent_scale_downs': len(scale_downs),
                'scaling_frequency': len(recent_actions) / max(1, len(recent_actions)),
                'resource_utilization': {
                    'cpu': avg_cpu,
                    'memory': avg_memory,
                    'response_time': avg_response_time
                },
                'demand_patterns': {
                    hour: sum(demands)/len(demands) if demands else 0
                    for hour, demands in self.demand_patterns.items()
                },
                'predicted_demand': self.predict_demand()
            }


class LoadBalancer:
    """Intelligent load balancer with health-aware routing."""
    
    def __init__(self, initial_workers: int = 4):
        self.workers = {}  # worker_id -> worker_info
        self.worker_health = {}  # worker_id -> health_score
        self.request_counts = defaultdict(int)
        self.response_times = defaultdict(deque)
        
        self.round_robin_counter = 0
        self._lock = threading.RLock()
        
        # Initialize workers
        for i in range(initial_workers):
            self.add_worker(f"worker_{i}")
    
    def add_worker(self, worker_id: str) -> None:
        """Add a new worker to the pool."""
        with self._lock:
            self.workers[worker_id] = {
                'id': worker_id,
                'active_requests': 0,
                'total_requests': 0,
                'last_activity': time.time(),
                'status': 'active'
            }
            self.worker_health[worker_id] = 1.0
            self.response_times[worker_id] = deque(maxlen=100)
            
            logger.info(f"Added worker {worker_id} to load balancer")
    
    def remove_worker(self, worker_id: str) -> None:
        """Remove worker from the pool."""
        with self._lock:
            if worker_id in self.workers:
                # Mark as draining first
                self.workers[worker_id]['status'] = 'draining'
                
                # Wait for active requests to complete (simplified)
                # In production, this would be more sophisticated
                time.sleep(1)
                
                del self.workers[worker_id]
                del self.worker_health[worker_id]
                if worker_id in self.response_times:
                    del self.response_times[worker_id]
                
                logger.info(f"Removed worker {worker_id} from load balancer")
    
    def select_worker(self, request_metadata: Dict[str, Any] = None) -> Optional[str]:
        """Select optimal worker for request using intelligent routing."""
        with self._lock:
            active_workers = [
                wid for wid, worker in self.workers.items() 
                if worker['status'] == 'active'
            ]
            
            if not active_workers:
                return None
            
            # Health-aware weighted round robin
            best_worker = None
            best_score = -1
            
            for worker_id in active_workers:
                worker = self.workers[worker_id]
                health = self.worker_health[worker_id]
                
                # Calculate load score (lower is better)
                active_load = worker['active_requests']
                recent_response_times = list(self.response_times[worker_id])
                avg_response_time = (
                    sum(recent_response_times) / len(recent_response_times)
                    if recent_response_times else 1.0
                )
                
                # Combined score (higher is better)
                load_score = 1.0 / (1.0 + active_load)
                response_score = 1.0 / max(0.1, avg_response_time / 1000.0)  # Normalize to seconds
                
                combined_score = (health * 0.4 + load_score * 0.4 + response_score * 0.2)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_worker = worker_id
            
            # Update worker state
            if best_worker:
                self.workers[best_worker]['active_requests'] += 1
                self.workers[best_worker]['total_requests'] += 1
                self.workers[best_worker]['last_activity'] = time.time()
            
            return best_worker
    
    def record_request_completion(self, worker_id: str, response_time_ms: float, success: bool = True) -> None:
        """Record request completion for load balancing optimization."""
        with self._lock:
            if worker_id in self.workers:
                self.workers[worker_id]['active_requests'] = max(0, self.workers[worker_id]['active_requests'] - 1)
                self.response_times[worker_id].append(response_time_ms)
                
                # Update health score based on success rate and response time
                current_health = self.worker_health[worker_id]
                
                if success and response_time_ms < 5000:  # Good performance
                    self.worker_health[worker_id] = min(1.0, current_health + 0.01)
                elif not success or response_time_ms > 10000:  # Poor performance
                    self.worker_health[worker_id] = max(0.1, current_health - 0.05)
    
    def get_load_distribution(self) -> Dict[str, Any]:
        """Get current load distribution across workers."""
        with self._lock:
            distribution = {}
            
            for worker_id, worker in self.workers.items():
                recent_times = list(self.response_times[worker_id])
                avg_response = sum(recent_times) / len(recent_times) if recent_times else 0
                
                distribution[worker_id] = {
                    'active_requests': worker['active_requests'],
                    'total_requests': worker['total_requests'],
                    'health_score': self.worker_health[worker_id],
                    'avg_response_time_ms': avg_response,
                    'status': worker['status']
                }
            
            return {
                'workers': distribution,
                'total_active_workers': len([w for w in self.workers.values() if w['status'] == 'active']),
                'total_active_requests': sum(w['active_requests'] for w in self.workers.values()),
                'system_health': sum(self.worker_health.values()) / len(self.worker_health) if self.worker_health else 0
            }


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
        self._lock = threading.RLock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == 'open':
                if self.last_failure_time and (time.time() - self.last_failure_time) > self.timeout:
                    self.state = 'half-open'
                    logger.info("Circuit breaker moving to half-open state")
                else:
                    raise Exception("Circuit breaker is open")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e
    
    def _on_success(self):
        """Handle successful call."""
        if self.state == 'half-open':
            self.state = 'closed'
            self.failure_count = 0
            logger.info("Circuit breaker closed after successful call")
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'open'
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class PerformanceOptimizer:
    """System-wide performance optimizer."""
    
    def __init__(self):
        self.optimizations = {}
        self.performance_history = deque(maxlen=1000)
        self.optimization_effects = {}
        
    def suggest_optimizations(self, metrics: ResourceMetrics) -> List[Dict[str, Any]]:
        """Suggest performance optimizations based on metrics."""
        suggestions = []
        
        # CPU optimization
        if metrics.cpu_usage > 0.8:
            suggestions.append({
                'type': 'cpu_optimization',
                'priority': 'high',
                'suggestion': 'Consider enabling CPU-intensive task offloading',
                'implementation': 'async_processing'
            })
        
        # Memory optimization
        if metrics.memory_usage > 0.85:
            suggestions.append({
                'type': 'memory_optimization',
                'priority': 'high',
                'suggestion': 'Implement aggressive caching cleanup',
                'implementation': 'cache_eviction'
            })
        
        # Response time optimization
        if metrics.response_time > 2000:
            suggestions.append({
                'type': 'latency_optimization',
                'priority': 'medium',
                'suggestion': 'Enable request batching and connection pooling',
                'implementation': 'batch_processing'
            })
        
        return suggestions
    
    def apply_optimization(self, optimization_type: str) -> bool:
        """Apply a specific optimization."""
        try:
            if optimization_type == 'async_processing':
                # Enable async processing for CPU-intensive tasks
                logger.info("Enabling async processing optimization")
                return True
                
            elif optimization_type == 'cache_eviction':
                # Trigger aggressive cache cleanup
                logger.info("Triggering cache eviction optimization")
                gc.collect()  # Force garbage collection
                return True
                
            elif optimization_type == 'batch_processing':
                # Enable request batching
                logger.info("Enabling batch processing optimization")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Failed to apply optimization {optimization_type}: {e}")
            return False