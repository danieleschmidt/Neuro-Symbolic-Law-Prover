"""
Monitoring and observability components for Neuro-Symbolic Law Prover.
Includes metrics collection, health checks, and performance monitoring.
"""

import time
import psutil
import logging
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


@dataclass
class Metric:
    """Represents a system metric."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = "count"


@dataclass
class HealthCheck:
    """Represents a health check result."""
    name: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    timestamp: datetime
    duration_ms: float
    details: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and tracks system metrics."""
    
    def __init__(self, retention_hours: int = 24):
        """Initialize metrics collector."""
        self.retention_hours = retention_hours
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._lock = threading.Lock()
        self._start_time = time.time()
        
        # Performance counters
        self.counters = defaultdict(int)
        self.timers = defaultdict(list)
        
        logger.info(f"MetricsCollector initialized with {retention_hours}h retention")
    
    def record_counter(self, name: str, value: float = 1, tags: Dict[str, str] = None) -> None:
        """Record a counter metric."""
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
            unit="count"
        )
        
        with self._lock:
            self.metrics[name].append(metric)
            self.counters[name] += value
        
        logger.debug(f"Counter {name}: {value} (total: {self.counters[name]})")
    
    def record_gauge(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Record a gauge metric."""
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
            unit="gauge"
        )
        
        with self._lock:
            self.metrics[name].append(metric)
        
        logger.debug(f"Gauge {name}: {value}")
    
    def record_timer(self, name: str, duration_ms: float, tags: Dict[str, str] = None) -> None:
        """Record a timing metric."""
        metric = Metric(
            name=name,
            value=duration_ms,
            timestamp=datetime.now(),
            tags=tags or {},
            unit="milliseconds"
        )
        
        with self._lock:
            self.metrics[name].append(metric)
            self.timers[name].append(duration_ms)
            
            # Keep only recent timings for statistics
            if len(self.timers[name]) > 1000:
                self.timers[name] = self.timers[name][-500:]
        
        logger.debug(f"Timer {name}: {duration_ms}ms")
    
    def get_counter_value(self, name: str) -> int:
        """Get current counter value."""
        return self.counters.get(name, 0)
    
    def get_timer_stats(self, name: str) -> Dict[str, float]:
        """Get timer statistics."""
        timings = self.timers.get(name, [])
        if not timings:
            return {"count": 0, "avg": 0, "min": 0, "max": 0, "p95": 0}
        
        sorted_timings = sorted(timings)
        count = len(sorted_timings)
        
        return {
            "count": count,
            "avg": sum(sorted_timings) / count,
            "min": sorted_timings[0],
            "max": sorted_timings[-1],
            "p95": sorted_timings[int(count * 0.95)] if count > 0 else 0
        }
    
    def get_recent_metrics(self, name: str, hours: int = 1) -> List[Metric]:
        """Get metrics from the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            metrics = self.metrics.get(name, [])
            return [m for m in metrics if m.timestamp >= cutoff_time]
    
    def get_all_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        summary = {
            "uptime_seconds": time.time() - self._start_time,
            "counters": dict(self.counters),
            "timer_stats": {name: self.get_timer_stats(name) for name in self.timers.keys()},
            "total_metric_series": len(self.metrics),
            "timestamp": datetime.now().isoformat()
        }
        
        return summary
    
    def cleanup_old_metrics(self) -> None:
        """Clean up metrics older than retention period."""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        with self._lock:
            for name, metric_deque in self.metrics.items():
                # Remove old metrics from the left side of deque
                while metric_deque and metric_deque[0].timestamp < cutoff_time:
                    metric_deque.popleft()
        
        logger.debug("Cleaned up old metrics")


class HealthChecker:
    """Manages health checks for system components."""
    
    def __init__(self):
        """Initialize health checker."""
        self.checks: Dict[str, Callable[[], HealthCheck]] = {}
        self._results: Dict[str, HealthCheck] = {}
        self._lock = threading.Lock()
        
        # Register default checks
        self.register_check("system_resources", self._check_system_resources)
        self.register_check("memory_usage", self._check_memory_usage)
        self.register_check("disk_space", self._check_disk_space)
        
        logger.info("HealthChecker initialized")
    
    def register_check(self, name: str, check_func: Callable[[], HealthCheck]) -> None:
        """Register a health check function."""
        self.checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    def run_check(self, name: str) -> HealthCheck:
        """Run a specific health check."""
        if name not in self.checks:
            return HealthCheck(
                name=name,
                status="unhealthy",
                message=f"Health check '{name}' not found",
                timestamp=datetime.now(),
                duration_ms=0
            )
        
        start_time = time.time()
        try:
            result = self.checks[name]()
            result.duration_ms = (time.time() - start_time) * 1000
            
            with self._lock:
                self._results[name] = result
            
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = HealthCheck(
                name=name,
                status="unhealthy",
                message=f"Health check failed: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=duration_ms,
                details={"error": str(e)}
            )
            
            with self._lock:
                self._results[name] = result
            
            logger.error(f"Health check '{name}' failed: {e}")
            return result
    
    def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks."""
        results = {}
        
        for name in self.checks.keys():
            results[name] = self.run_check(name)
        
        return results
    
    def get_overall_status(self) -> str:
        """Get overall system health status."""
        results = self.run_all_checks()
        
        if not results:
            return "unknown"
        
        statuses = [r.status for r in results.values()]
        
        if all(s == "healthy" for s in statuses):
            return "healthy"
        elif any(s == "unhealthy" for s in statuses):
            return "unhealthy"
        else:
            return "degraded"
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        results = self.run_all_checks()
        overall_status = self.get_overall_status()
        
        return {
            "overall_status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "checks": {name: {
                "status": check.status,
                "message": check.message,
                "duration_ms": check.duration_ms,
                "details": check.details
            } for name, check in results.items()}
        }
    
    def _check_system_resources(self) -> HealthCheck:
        """Check system resource availability."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            if cpu_percent > 90 or memory_percent > 90:
                status = "unhealthy"
                message = f"High resource usage: CPU {cpu_percent}%, Memory {memory_percent}%"
            elif cpu_percent > 70 or memory_percent > 70:
                status = "degraded"
                message = f"Elevated resource usage: CPU {cpu_percent}%, Memory {memory_percent}%"
            else:
                status = "healthy"
                message = f"Resource usage normal: CPU {cpu_percent}%, Memory {memory_percent}%"
            
            return HealthCheck(
                name="system_resources",
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=0,  # Will be set by caller
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="system_resources",
                status="unhealthy",
                message=f"Failed to check system resources: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=0
            )
    
    def _check_memory_usage(self) -> HealthCheck:
        """Check memory usage details."""
        try:
            memory = psutil.virtual_memory()
            
            if memory.percent > 85:
                status = "unhealthy"
            elif memory.percent > 70:
                status = "degraded"
            else:
                status = "healthy"
            
            return HealthCheck(
                name="memory_usage",
                status=status,
                message=f"Memory usage: {memory.percent}% ({memory.used // 1024 // 1024} MB used)",
                timestamp=datetime.now(),
                duration_ms=0,
                details={
                    "total_mb": memory.total // 1024 // 1024,
                    "used_mb": memory.used // 1024 // 1024,
                    "available_mb": memory.available // 1024 // 1024,
                    "percent": memory.percent
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="memory_usage",
                status="unhealthy",
                message=f"Failed to check memory usage: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=0
            )
    
    def _check_disk_space(self) -> HealthCheck:
        """Check disk space availability."""
        try:
            disk = psutil.disk_usage('/')
            percent_used = (disk.used / disk.total) * 100
            
            if percent_used > 90:
                status = "unhealthy"
            elif percent_used > 80:
                status = "degraded"
            else:
                status = "healthy"
            
            return HealthCheck(
                name="disk_space",
                status=status,
                message=f"Disk usage: {percent_used:.1f}% ({disk.used // 1024 // 1024 // 1024} GB used)",
                timestamp=datetime.now(),
                duration_ms=0,
                details={
                    "total_gb": disk.total // 1024 // 1024 // 1024,
                    "used_gb": disk.used // 1024 // 1024 // 1024,
                    "free_gb": disk.free // 1024 // 1024 // 1024,
                    "percent_used": percent_used
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="disk_space",
                status="unhealthy",
                message=f"Failed to check disk space: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=0
            )


class PerformanceTimer:
    """Context manager for measuring execution time."""
    
    def __init__(self, metrics_collector: MetricsCollector, timer_name: str, tags: Dict[str, str] = None):
        """Initialize performance timer."""
        self.metrics_collector = metrics_collector
        self.timer_name = timer_name
        self.tags = tags or {}
        self.start_time = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and record metric."""
        if self.start_time is not None:
            duration_ms = (time.time() - self.start_time) * 1000
            self.metrics_collector.record_timer(self.timer_name, duration_ms, self.tags)


def time_function(metrics_collector: MetricsCollector, timer_name: str = None):
    """Decorator to time function execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            name = timer_name or f"function.{func.__name__}"
            with PerformanceTimer(metrics_collector, name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Global instances
_metrics_collector = MetricsCollector()
_health_checker = HealthChecker()


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    return _metrics_collector


def get_health_checker() -> HealthChecker:
    """Get global health checker instance."""
    return _health_checker


def setup_monitoring(retention_hours: int = 24) -> None:
    """Set up monitoring with custom configuration."""
    global _metrics_collector, _health_checker
    
    _metrics_collector = MetricsCollector(retention_hours)
    _health_checker = HealthChecker()
    
    logger.info(f"Monitoring setup complete with {retention_hours}h retention")