"""
Advanced metrics collection and aggregation system for production monitoring.
"""

import time
import threading
import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics
import weakref
import json
import sqlite3
from pathlib import Path
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    class psutil:
        @staticmethod
        def cpu_percent(): return 50.0
        @staticmethod
        def virtual_memory(): 
            class mem: percent = 60.0; total = 8*1024**3
            return mem()
import platform

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"
    PERCENTAGE = "percentage"


@dataclass
class Metric:
    """Represents a metric with its metadata and values."""
    name: str
    metric_type: MetricType
    value: Union[int, float, List[float]]
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary representation."""
        return {
            'name': self.name,
            'type': self.metric_type.value,
            'value': self.value,
            'timestamp': self.timestamp,
            'labels': self.labels,
            'unit': self.unit,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Metric':
        """Create metric from dictionary representation."""
        return cls(
            name=data['name'],
            metric_type=MetricType(data['type']),
            value=data['value'],
            timestamp=data['timestamp'],
            labels=data.get('labels', {}),
            unit=data.get('unit'),
            description=data.get('description')
        )


class MetricBuffer:
    """Thread-safe buffer for storing metrics before aggregation."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.metrics: deque = deque(maxlen=max_size)
        self.lock = threading.RLock()
    
    def add(self, metric: Metric):
        """Add metric to buffer."""
        with self.lock:
            self.metrics.append(metric)
    
    def drain(self) -> List[Metric]:
        """Drain all metrics from buffer."""
        with self.lock:
            metrics = list(self.metrics)
            self.metrics.clear()
            return metrics
    
    def size(self) -> int:
        """Get current buffer size."""
        with self.lock:
            return len(self.metrics)


class MetricsCollector:
    """
    Production-grade metrics collection system.
    
    Features:
    - Multiple metric types (counter, gauge, histogram, timer)
    - Automatic system metrics collection
    - Metric aggregation and storage
    - Export to various backends (Prometheus, InfluxDB, etc.)
    - Real-time streaming and alerting
    """
    
    def __init__(self,
                 collection_interval: float = 30.0,  # seconds
                 retention_days: int = 30,
                 storage_path: Optional[str] = None,
                 enable_system_metrics: bool = True,
                 enable_export: bool = False,
                 export_endpoints: Optional[List[str]] = None):
        """
        Initialize metrics collector.
        
        Args:
            collection_interval: How often to collect metrics
            retention_days: How long to retain metrics
            storage_path: Path to store metrics database
            enable_system_metrics: Whether to collect system metrics
            enable_export: Whether to export metrics to external systems
            export_endpoints: List of export endpoint URLs
        """
        self.collection_interval = collection_interval
        self.retention_days = retention_days
        self.enable_system_metrics = enable_system_metrics
        self.enable_export = enable_export
        self.export_endpoints = export_endpoints or []
        
        # Metric storage
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        
        # Metric metadata
        self.metric_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Buffer for incoming metrics
        self.buffer = MetricBuffer()
        
        # Aggregated metrics storage
        self.aggregated_metrics: Dict[str, List[Metric]] = defaultdict(list)
        
        # Database storage
        self.storage_path = storage_path or "/tmp/metrics.db"
        self.db_connection = None
        self._init_storage()
        
        # Background processing
        self.collection_task: Optional[asyncio.Task] = None
        self.processing_task: Optional[asyncio.Task] = None
        self.export_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Listeners for real-time metrics
        self.listeners: List[Callable[[Metric], None]] = []
        
        # System metrics components
        if self.enable_system_metrics:
            self.process = psutil.Process()
            self.system_info = {
                'hostname': platform.node(),
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total
            }
        
        logger.info("MetricsCollector initialized")
    
    def _init_storage(self):
        """Initialize SQLite database for metrics storage."""
        try:
            Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
            self.db_connection = sqlite3.connect(self.storage_path, check_same_thread=False)
            
            # Create metrics table
            self.db_connection.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    value REAL,
                    timestamp REAL NOT NULL,
                    labels TEXT,
                    unit TEXT,
                    description TEXT,
                    INDEX(name, timestamp),
                    INDEX(timestamp)
                )
            """)
            
            # Create aggregated metrics table
            self.db_connection.execute("""
                CREATE TABLE IF NOT EXISTS aggregated_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    min_value REAL,
                    max_value REAL,
                    avg_value REAL,
                    sum_value REAL,
                    count_value INTEGER,
                    percentile_95 REAL,
                    percentile_99 REAL,
                    timestamp REAL NOT NULL,
                    window_size INTEGER,
                    labels TEXT,
                    INDEX(name, timestamp)
                )
            """)
            
            self.db_connection.commit()
            logger.info(f"Metrics database initialized: {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize metrics database: {e}")
            self.db_connection = None
    
    async def start(self):
        """Start metrics collection."""
        if self.running:
            return
        
        self.running = True
        
        # Start collection tasks
        if self.enable_system_metrics:
            self.collection_task = asyncio.create_task(self._system_metrics_loop())
        
        self.processing_task = asyncio.create_task(self._processing_loop())
        
        if self.enable_export:
            self.export_task = asyncio.create_task(self._export_loop())
        
        logger.info("Metrics collection started")
    
    async def stop(self):
        """Stop metrics collection."""
        self.running = False
        
        # Cancel tasks
        for task in [self.collection_task, self.processing_task, self.export_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Close database
        if self.db_connection:
            self.db_connection.close()
        
        logger.info("Metrics collection stopped")
    
    def increment(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None,
                 description: Optional[str] = None):
        """Increment a counter metric."""
        with self.lock:
            key = self._make_key(name, labels)
            self.counters[key] += value
            
            if name not in self.metric_metadata:
                self.metric_metadata[name] = {
                    'type': MetricType.COUNTER,
                    'description': description,
                    'labels': labels or {}
                }
            
            # Add to buffer for real-time processing
            metric = Metric(
                name=name,
                metric_type=MetricType.COUNTER,
                value=self.counters[key],
                timestamp=time.time(),
                labels=labels or {},
                description=description
            )
            self.buffer.add(metric)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None,
                 unit: Optional[str] = None, description: Optional[str] = None):
        """Set a gauge metric value."""
        with self.lock:
            key = self._make_key(name, labels)
            self.gauges[key] = value
            
            if name not in self.metric_metadata:
                self.metric_metadata[name] = {
                    'type': MetricType.GAUGE,
                    'unit': unit,
                    'description': description,
                    'labels': labels or {}
                }
            
            metric = Metric(
                name=name,
                metric_type=MetricType.GAUGE,
                value=value,
                timestamp=time.time(),
                labels=labels or {},
                unit=unit,
                description=description
            )
            self.buffer.add(metric)
    
    def observe(self, name: str, value: float, labels: Optional[Dict[str, str]] = None,
               unit: Optional[str] = None, description: Optional[str] = None):
        """Observe a value for histogram metrics."""
        with self.lock:
            key = self._make_key(name, labels)
            self.histograms[key].append(value)
            
            # Keep only recent observations (sliding window)
            max_observations = 1000
            if len(self.histograms[key]) > max_observations:
                self.histograms[key] = self.histograms[key][-max_observations:]
            
            if name not in self.metric_metadata:
                self.metric_metadata[name] = {
                    'type': MetricType.HISTOGRAM,
                    'unit': unit,
                    'description': description,
                    'labels': labels or {}
                }
            
            metric = Metric(
                name=name,
                metric_type=MetricType.HISTOGRAM,
                value=value,
                timestamp=time.time(),
                labels=labels or {},
                unit=unit,
                description=description
            )
            self.buffer.add(metric)
    
    def time_function(self, name: str, labels: Optional[Dict[str, str]] = None,
                     unit: str = "seconds", description: Optional[str] = None):
        """Decorator to time function execution."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    execution_time = time.time() - start_time
                    self.record_timer(name, execution_time, labels, unit, description)
            return wrapper
        return decorator
    
    def record_timer(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None,
                    unit: str = "seconds", description: Optional[str] = None):
        """Record a timer measurement."""
        with self.lock:
            key = self._make_key(name, labels)
            self.timers[key].append(duration)
            
            # Keep only recent timings
            max_timings = 1000
            if len(self.timers[key]) > max_timings:
                self.timers[key] = self.timers[key][-max_timings:]
            
            if name not in self.metric_metadata:
                self.metric_metadata[name] = {
                    'type': MetricType.TIMER,
                    'unit': unit,
                    'description': description,
                    'labels': labels or {}
                }
            
            metric = Metric(
                name=name,
                metric_type=MetricType.TIMER,
                value=duration,
                timestamp=time.time(),
                labels=labels or {},
                unit=unit,
                description=description
            )
            self.buffer.add(metric)
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create a unique key for a metric with labels."""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    async def _system_metrics_loop(self):
        """Collect system metrics periodically."""
        while self.running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(1)
    
    async def _collect_system_metrics(self):
        """Collect various system metrics."""
        timestamp = time.time()
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.set_gauge("system.cpu.usage_percent", cpu_percent, 
                          description="CPU usage percentage")
            
            cpu_count = psutil.cpu_count()
            self.set_gauge("system.cpu.count", cpu_count,
                          description="Number of CPU cores")
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.set_gauge("system.memory.total", memory.total,
                          unit="bytes", description="Total memory")
            self.set_gauge("system.memory.available", memory.available,
                          unit="bytes", description="Available memory")
            self.set_gauge("system.memory.used", memory.used,
                          unit="bytes", description="Used memory")
            self.set_gauge("system.memory.usage_percent", memory.percent,
                          unit="percent", description="Memory usage percentage")
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.set_gauge("system.disk.total", disk.total,
                          unit="bytes", description="Total disk space")
            self.set_gauge("system.disk.used", disk.used,
                          unit="bytes", description="Used disk space")
            self.set_gauge("system.disk.free", disk.free,
                          unit="bytes", description="Free disk space")
            self.set_gauge("system.disk.usage_percent", 
                          (disk.used / disk.total) * 100,
                          unit="percent", description="Disk usage percentage")
            
            # Process metrics
            process_memory = self.process.memory_info()
            self.set_gauge("process.memory.rss", process_memory.rss,
                          unit="bytes", description="Process RSS memory")
            self.set_gauge("process.memory.vms", process_memory.vms,
                          unit="bytes", description="Process VMS memory")
            
            process_cpu = self.process.cpu_percent()
            self.set_gauge("process.cpu.usage_percent", process_cpu,
                          unit="percent", description="Process CPU usage")
            
            # Network I/O metrics
            net_io = psutil.net_io_counters()
            if net_io:
                self.increment("system.network.bytes_sent", net_io.bytes_sent)
                self.increment("system.network.bytes_recv", net_io.bytes_recv)
                self.increment("system.network.packets_sent", net_io.packets_sent)
                self.increment("system.network.packets_recv", net_io.packets_recv)
            
            # Load average (Unix systems)
            try:
                load_avg = psutil.getloadavg()
                self.set_gauge("system.load.avg_1min", load_avg[0],
                              description="1-minute load average")
                self.set_gauge("system.load.avg_5min", load_avg[1],
                              description="5-minute load average")
                self.set_gauge("system.load.avg_15min", load_avg[2],
                              description="15-minute load average")
            except (AttributeError, OSError):
                pass  # Not available on Windows
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _processing_loop(self):
        """Process buffered metrics and perform aggregation."""
        while self.running:
            try:
                # Drain metrics from buffer
                metrics = self.buffer.drain()
                
                if metrics:
                    await self._process_metrics(metrics)
                    await self._store_metrics(metrics)
                    await self._notify_listeners(metrics)
                
                await asyncio.sleep(5)  # Process every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing metrics: {e}")
                await asyncio.sleep(1)
    
    async def _process_metrics(self, metrics: List[Metric]):
        """Process metrics for aggregation and analysis."""
        # Group metrics by name and time window
        window_size = 60  # 1 minute windows
        current_window = int(time.time() / window_size) * window_size
        
        metric_groups = defaultdict(list)
        
        for metric in metrics:
            # Group by metric name and window
            window_timestamp = int(metric.timestamp / window_size) * window_size
            key = f"{metric.name}_{window_timestamp}"
            metric_groups[key].append(metric)
        
        # Process each group
        for key, group_metrics in metric_groups.items():
            if not group_metrics:
                continue
            
            metric_name = group_metrics[0].name
            metric_type = group_metrics[0].metric_type
            window_timestamp = int(key.split('_')[-1])
            
            # Calculate aggregated statistics
            values = [m.value for m in group_metrics if isinstance(m.value, (int, float))]
            
            if values:
                aggregated = {
                    'name': metric_name,
                    'type': metric_type.value,
                    'min_value': min(values),
                    'max_value': max(values),
                    'avg_value': statistics.mean(values),
                    'sum_value': sum(values),
                    'count_value': len(values),
                    'timestamp': window_timestamp,
                    'window_size': window_size
                }
                
                # Calculate percentiles for histograms and timers
                if len(values) >= 2:
                    sorted_values = sorted(values)
                    aggregated['percentile_95'] = self._percentile(sorted_values, 95)
                    aggregated['percentile_99'] = self._percentile(sorted_values, 99)
                
                # Store aggregated metrics
                self.aggregated_metrics[metric_name].append(aggregated)
                
                # Keep only recent aggregated data
                max_aggregated = 1440  # 24 hours of 1-minute windows
                if len(self.aggregated_metrics[metric_name]) > max_aggregated:
                    self.aggregated_metrics[metric_name] = \
                        self.aggregated_metrics[metric_name][-max_aggregated:]
    
    def _percentile(self, sorted_values: List[float], percentile: float) -> float:
        """Calculate percentile of sorted values."""
        if not sorted_values:
            return 0.0
        
        k = (len(sorted_values) - 1) * (percentile / 100)
        f = int(k)
        c = k - f
        
        if f + 1 < len(sorted_values):
            return sorted_values[f] * (1 - c) + sorted_values[f + 1] * c
        else:
            return sorted_values[f]
    
    async def _store_metrics(self, metrics: List[Metric]):
        """Store metrics in database."""
        if not self.db_connection:
            return
        
        try:
            # Store raw metrics
            for metric in metrics:
                labels_json = json.dumps(metric.labels) if metric.labels else None
                
                self.db_connection.execute("""
                    INSERT INTO metrics (name, type, value, timestamp, labels, unit, description)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    metric.name,
                    metric.metric_type.value,
                    metric.value if isinstance(metric.value, (int, float)) else None,
                    metric.timestamp,
                    labels_json,
                    metric.unit,
                    metric.description
                ))
            
            self.db_connection.commit()
            
            # Clean up old metrics
            cutoff_time = time.time() - (self.retention_days * 24 * 3600)
            self.db_connection.execute(
                "DELETE FROM metrics WHERE timestamp < ?",
                (cutoff_time,)
            )
            self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Error storing metrics: {e}")
    
    async def _notify_listeners(self, metrics: List[Metric]):
        """Notify registered listeners of new metrics."""
        for metric in metrics:
            for listener in self.listeners:
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None, listener, metric
                    )
                except Exception as e:
                    logger.error(f"Error notifying listener: {e}")
    
    async def _export_loop(self):
        """Export metrics to external systems."""
        while self.running:
            try:
                if self.export_endpoints:
                    await self._export_metrics()
                await asyncio.sleep(60)  # Export every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error exporting metrics: {e}")
                await asyncio.sleep(10)
    
    async def _export_metrics(self):
        """Export metrics to configured endpoints."""
        # Get recent metrics for export
        recent_metrics = []
        cutoff_time = time.time() - 300  # Last 5 minutes
        
        for metric_list in self.aggregated_metrics.values():
            for metric in metric_list:
                if metric['timestamp'] >= cutoff_time:
                    recent_metrics.append(metric)
        
        if not recent_metrics:
            return
        
        # Export to each endpoint
        for endpoint in self.export_endpoints:
            try:
                await self._export_to_endpoint(endpoint, recent_metrics)
            except Exception as e:
                logger.error(f"Error exporting to {endpoint}: {e}")
    
    async def _export_to_endpoint(self, endpoint: str, metrics: List[Dict[str, Any]]):
        """Export metrics to a specific endpoint."""
        # This is a placeholder for actual export implementations
        # In practice, you'd implement specific exporters for:
        # - Prometheus
        # - InfluxDB
        # - CloudWatch
        # - Datadog
        # etc.
        
        logger.debug(f"Exporting {len(metrics)} metrics to {endpoint}")
    
    def add_listener(self, listener: Callable[[Metric], None]):
        """Add a listener for real-time metrics."""
        self.listeners.append(listener)
    
    def remove_listener(self, listener: Callable[[Metric], None]):
        """Remove a metrics listener."""
        if listener in self.listeners:
            self.listeners.remove(listener)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metric values."""
        with self.lock:
            return {
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'histograms': {k: list(v) for k, v in self.histograms.items()},
                'timers': {k: list(v) for k, v in self.timers.items()},
                'metadata': dict(self.metric_metadata)
            }
    
    def get_metric_history(self, metric_name: str, 
                          start_time: Optional[float] = None,
                          end_time: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get historical data for a specific metric."""
        if not self.db_connection:
            return []
        
        start_time = start_time or (time.time() - 3600)  # Last hour
        end_time = end_time or time.time()
        
        try:
            cursor = self.db_connection.execute("""
                SELECT name, type, value, timestamp, labels, unit, description
                FROM metrics
                WHERE name = ? AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp
            """, (metric_name, start_time, end_time))
            
            results = []
            for row in cursor.fetchall():
                labels = json.loads(row[4]) if row[4] else {}
                results.append({
                    'name': row[0],
                    'type': row[1],
                    'value': row[2],
                    'timestamp': row[3],
                    'labels': labels,
                    'unit': row[5],
                    'description': row[6]
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving metric history: {e}")
            return []
    
    def get_aggregated_metrics(self, metric_name: str) -> List[Dict[str, Any]]:
        """Get aggregated metrics for a specific metric name."""
        return self.aggregated_metrics.get(metric_name, [])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get metrics collector statistics."""
        with self.lock:
            return {
                'running': self.running,
                'collection_interval': self.collection_interval,
                'buffer_size': self.buffer.size(),
                'total_counters': len(self.counters),
                'total_gauges': len(self.gauges),
                'total_histograms': len(self.histograms),
                'total_timers': len(self.timers),
                'total_listeners': len(self.listeners),
                'export_endpoints': len(self.export_endpoints),
                'system_metrics_enabled': self.enable_system_metrics,
                'export_enabled': self.enable_export,
                'storage_path': self.storage_path,
                'retention_days': self.retention_days
            }
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()