"""
Production monitoring and observability for neuro-symbolic law prover.
"""

__version__ = "1.0.0"

# Core monitoring components
from .metrics_collector import MetricsCollector, MetricType, Metric
from .alert_manager import AlertManager, AlertSeverity, Alert
from .dashboard import Dashboard, DashboardWidget
from .tracer import DistributedTracer, TraceSpan
from .profiler import PerformanceProfiler, ProfileReport

# Health monitoring
from .health_monitor import HealthMonitor, HealthCheck, HealthStatus
from .service_registry import ServiceRegistry, ServiceInstance

# Log aggregation
from .log_aggregator import LogAggregator, LogLevel, LogEntry

__all__ = [
    # Core components
    "MetricsCollector",
    "MetricType",
    "Metric",
    "AlertManager",
    "AlertSeverity",
    "Alert",
    "Dashboard",
    "DashboardWidget",
    "DistributedTracer",
    "TraceSpan",
    "PerformanceProfiler",
    "ProfileReport",
    
    # Health monitoring
    "HealthMonitor",
    "HealthCheck",
    "HealthStatus",
    "ServiceRegistry",
    "ServiceInstance",
    
    # Logging
    "LogAggregator",
    "LogLevel",
    "LogEntry"
]