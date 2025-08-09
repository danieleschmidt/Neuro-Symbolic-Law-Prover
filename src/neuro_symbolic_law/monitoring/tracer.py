"""
Distributed tracing and performance profiling for microservices architecture.
"""

import time
import uuid
import asyncio
import logging
import json
import threading
from typing import Dict, List, Optional, Any, Callable, Set, Tuple, Union, ContextManager
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
from contextlib import contextmanager, asynccontextmanager
import contextvars
from pathlib import Path

logger = logging.getLogger(__name__)

# Context variables for distributed tracing
trace_context: contextvars.ContextVar[Optional['TraceContext']] = contextvars.ContextVar('trace_context', default=None)


class SpanType(Enum):
    """Types of trace spans."""
    REQUEST = "request"
    DATABASE = "database"
    CACHE = "cache"
    EXTERNAL_SERVICE = "external_service"
    COMPUTATION = "computation"
    NETWORK = "network"
    FILE_IO = "file_io"
    CUSTOM = "custom"


class SpanStatus(Enum):
    """Status of trace spans."""
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class TraceContext:
    """Context information for distributed tracing."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)
    flags: int = 0
    
    def child_context(self) -> 'TraceContext':
        """Create a child trace context."""
        return TraceContext(
            trace_id=self.trace_id,
            span_id=str(uuid.uuid4()),
            parent_span_id=self.span_id,
            baggage=self.baggage.copy(),
            flags=self.flags
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'trace_id': self.trace_id,
            'span_id': self.span_id,
            'parent_span_id': self.parent_span_id,
            'baggage': self.baggage,
            'flags': self.flags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TraceContext':
        """Create from dictionary."""
        return cls(
            trace_id=data['trace_id'],
            span_id=data['span_id'],
            parent_span_id=data.get('parent_span_id'),
            baggage=data.get('baggage', {}),
            flags=data.get('flags', 0)
        )


@dataclass
class TraceSpan:
    """Represents a single span in a distributed trace."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    service_name: str
    span_type: SpanType
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    status: SpanStatus = SpanStatus.OK
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    baggage: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.end_time and not self.duration:
            self.duration = self.end_time - self.start_time
    
    def finish(self, status: SpanStatus = SpanStatus.OK):
        """Finish the span."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.status = status
    
    def add_tag(self, key: str, value: Any):
        """Add a tag to the span."""
        self.tags[key] = value
    
    def add_log(self, message: str, level: str = "info", **fields):
        """Add a log entry to the span."""
        log_entry = {
            'timestamp': time.time(),
            'message': message,
            'level': level,
            **fields
        }
        self.logs.append(log_entry)
    
    def set_error(self, error: Exception):
        """Mark span as error and add error information."""
        self.status = SpanStatus.ERROR
        self.add_tag('error', True)
        self.add_tag('error.type', type(error).__name__)
        self.add_tag('error.message', str(error))
        
        self.add_log(
            message=f"Error occurred: {error}",
            level="error",
            error_type=type(error).__name__,
            error_message=str(error)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary representation."""
        return {
            'trace_id': self.trace_id,
            'span_id': self.span_id,
            'parent_span_id': self.parent_span_id,
            'operation_name': self.operation_name,
            'service_name': self.service_name,
            'span_type': self.span_type.value,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'status': self.status.value,
            'tags': self.tags,
            'logs': self.logs,
            'baggage': self.baggage
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TraceSpan':
        """Create span from dictionary."""
        return cls(
            trace_id=data['trace_id'],
            span_id=data['span_id'],
            parent_span_id=data.get('parent_span_id'),
            operation_name=data['operation_name'],
            service_name=data['service_name'],
            span_type=SpanType(data['span_type']),
            start_time=data['start_time'],
            end_time=data.get('end_time'),
            duration=data.get('duration'),
            status=SpanStatus(data.get('status', 'ok')),
            tags=data.get('tags', {}),
            logs=data.get('logs', []),
            baggage=data.get('baggage', {})
        )


class DistributedTracer:
    """
    Distributed tracing system for microservices.
    
    Features:
    - Hierarchical span tracking
    - Context propagation
    - Sampling strategies
    - Export to multiple backends
    - Performance analysis
    - Error tracking
    """
    
    def __init__(self,
                 service_name: str,
                 sampling_rate: float = 1.0,
                 max_spans_per_trace: int = 1000,
                 storage_path: Optional[str] = None,
                 export_batch_size: int = 100,
                 export_interval: float = 30.0):
        """
        Initialize distributed tracer.
        
        Args:
            service_name: Name of the service
            sampling_rate: Trace sampling rate (0.0 to 1.0)
            max_spans_per_trace: Maximum spans per trace
            storage_path: Path to store traces
            export_batch_size: Batch size for exporting traces
            export_interval: Export interval in seconds
        """
        self.service_name = service_name
        self.sampling_rate = sampling_rate
        self.max_spans_per_trace = max_spans_per_trace
        self.storage_path = storage_path or f"/tmp/traces_{service_name}.json"
        self.export_batch_size = export_batch_size
        self.export_interval = export_interval
        
        # Span storage
        self.active_spans: Dict[str, TraceSpan] = {}
        self.completed_traces: Dict[str, List[TraceSpan]] = {}
        self.span_buffer: deque = deque(maxlen=10000)
        
        # Export configuration
        self.exporters: List[Callable[[List[TraceSpan]], None]] = []
        
        # Background tasks
        self.export_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'total_traces': 0,
            'total_spans': 0,
            'sampled_traces': 0,
            'dropped_traces': 0,
            'export_count': 0,
            'error_count': 0
        }
        
        logger.info(f"DistributedTracer initialized for service: {service_name}")
    
    async def start(self):
        """Start the tracer."""
        if self.running:
            return
        
        self.running = True
        
        # Start background tasks
        self.export_task = asyncio.create_task(self._export_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Distributed tracer started")
    
    async def stop(self):
        """Stop the tracer."""
        self.running = False
        
        # Cancel tasks
        for task in [self.export_task, self.cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Export remaining spans
        await self._export_pending_spans()
        
        logger.info("Distributed tracer stopped")
    
    def start_trace(self, operation_name: str, 
                   span_type: SpanType = SpanType.REQUEST,
                   parent_context: Optional[TraceContext] = None,
                   tags: Optional[Dict[str, Any]] = None) -> TraceContext:
        """
        Start a new trace or span.
        
        Args:
            operation_name: Name of the operation
            span_type: Type of span
            parent_context: Parent trace context
            tags: Initial tags for the span
            
        Returns:
            New trace context
        """
        # Check sampling
        import random
        if random.random() > self.sampling_rate:
            self.stats['dropped_traces'] += 1
            return TraceContext(
                trace_id=str(uuid.uuid4()),
                span_id=str(uuid.uuid4())
            )
        
        # Create context
        if parent_context:
            context = parent_context.child_context()
        else:
            context = TraceContext(
                trace_id=str(uuid.uuid4()),
                span_id=str(uuid.uuid4())
            )
            self.stats['total_traces'] += 1
            self.stats['sampled_traces'] += 1
        
        # Create span
        span = TraceSpan(
            trace_id=context.trace_id,
            span_id=context.span_id,
            parent_span_id=context.parent_span_id,
            operation_name=operation_name,
            service_name=self.service_name,
            span_type=span_type,
            start_time=time.time(),
            tags=tags or {},
            baggage=context.baggage.copy()
        )
        
        with self.lock:
            self.active_spans[context.span_id] = span
        
        self.stats['total_spans'] += 1
        
        return context
    
    def finish_span(self, context: TraceContext, 
                   status: SpanStatus = SpanStatus.OK,
                   error: Optional[Exception] = None):
        """
        Finish a span.
        
        Args:
            context: Trace context
            status: Span status
            error: Error if span failed
        """
        with self.lock:
            span = self.active_spans.get(context.span_id)
            if not span:
                return
            
            # Finish the span
            span.finish(status)
            
            if error:
                span.set_error(error)
                self.stats['error_count'] += 1
            
            # Move to buffer for export
            self.span_buffer.append(span)
            
            # Add to completed traces for analysis
            if span.trace_id not in self.completed_traces:
                self.completed_traces[span.trace_id] = []
            
            self.completed_traces[span.trace_id].append(span)
            
            # Limit traces in memory
            if len(self.completed_traces[span.trace_id]) > self.max_spans_per_trace:
                self.completed_traces[span.trace_id] = self.completed_traces[span.trace_id][-self.max_spans_per_trace:]
            
            # Remove from active spans
            del self.active_spans[context.span_id]
    
    @contextmanager
    def trace(self, operation_name: str, 
              span_type: SpanType = SpanType.CUSTOM,
              tags: Optional[Dict[str, Any]] = None):
        """
        Context manager for tracing operations.
        
        Args:
            operation_name: Name of the operation
            span_type: Type of span
            tags: Initial tags
        """
        # Get current context from context variable
        parent_context = trace_context.get()
        
        # Start span
        context = self.start_trace(operation_name, span_type, parent_context, tags)
        
        # Set context variable
        token = trace_context.set(context)
        
        try:
            yield context
            self.finish_span(context, SpanStatus.OK)
        except Exception as e:
            self.finish_span(context, SpanStatus.ERROR, e)
            raise
        finally:
            trace_context.reset(token)
    
    @asynccontextmanager
    async def trace_async(self, operation_name: str,
                         span_type: SpanType = SpanType.CUSTOM,
                         tags: Optional[Dict[str, Any]] = None):
        """
        Async context manager for tracing operations.
        
        Args:
            operation_name: Name of the operation
            span_type: Type of span
            tags: Initial tags
        """
        # Get current context from context variable
        parent_context = trace_context.get()
        
        # Start span
        context = self.start_trace(operation_name, span_type, parent_context, tags)
        
        # Set context variable
        token = trace_context.set(context)
        
        try:
            yield context
            self.finish_span(context, SpanStatus.OK)
        except Exception as e:
            self.finish_span(context, SpanStatus.ERROR, e)
            raise
        finally:
            trace_context.reset(token)
    
    def trace_function(self, operation_name: Optional[str] = None,
                      span_type: SpanType = SpanType.COMPUTATION,
                      tags: Optional[Dict[str, Any]] = None):
        """
        Decorator for tracing functions.
        
        Args:
            operation_name: Operation name (defaults to function name)
            span_type: Type of span
            tags: Initial tags
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                
                with self.trace(op_name, span_type, tags):
                    return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def add_tag(self, key: str, value: Any):
        """Add tag to current span."""
        context = trace_context.get()
        if context:
            with self.lock:
                span = self.active_spans.get(context.span_id)
                if span:
                    span.add_tag(key, value)
    
    def add_log(self, message: str, level: str = "info", **fields):
        """Add log to current span."""
        context = trace_context.get()
        if context:
            with self.lock:
                span = self.active_spans.get(context.span_id)
                if span:
                    span.add_log(message, level, **fields)
    
    def set_baggage(self, key: str, value: str):
        """Set baggage item for trace propagation."""
        context = trace_context.get()
        if context:
            context.baggage[key] = value
            
            # Update span baggage
            with self.lock:
                span = self.active_spans.get(context.span_id)
                if span:
                    span.baggage[key] = value
    
    def get_baggage(self, key: str) -> Optional[str]:
        """Get baggage item from current trace."""
        context = trace_context.get()
        if context:
            return context.baggage.get(key)
        return None
    
    def add_exporter(self, exporter: Callable[[List[TraceSpan]], None]):
        """Add trace exporter."""
        self.exporters.append(exporter)
        logger.info("Added trace exporter")
    
    def remove_exporter(self, exporter: Callable[[List[TraceSpan]], None]):
        """Remove trace exporter."""
        if exporter in self.exporters:
            self.exporters.remove(exporter)
            logger.info("Removed trace exporter")
    
    async def _export_loop(self):
        """Background loop for exporting traces."""
        while self.running:
            try:
                await self._export_pending_spans()
                await asyncio.sleep(self.export_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in export loop: {e}")
                await asyncio.sleep(5)
    
    async def _export_pending_spans(self):
        """Export pending spans to configured exporters."""
        spans_to_export = []
        
        with self.lock:
            # Collect spans for export
            while len(spans_to_export) < self.export_batch_size and self.span_buffer:
                spans_to_export.append(self.span_buffer.popleft())
        
        if not spans_to_export:
            return
        
        # Export to all configured exporters
        for exporter in self.exporters:
            try:
                await asyncio.get_event_loop().run_in_executor(None, exporter, spans_to_export)
            except Exception as e:
                logger.error(f"Error in trace exporter: {e}")
        
        self.stats['export_count'] += len(spans_to_export)
        
        # Store to file
        if self.storage_path:
            await self._store_spans(spans_to_export)
    
    async def _store_spans(self, spans: List[TraceSpan]):
        """Store spans to file."""
        try:
            Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Append to file
            with open(self.storage_path, 'a') as f:
                for span in spans:
                    f.write(json.dumps(span.to_dict()) + '\n')
        
        except Exception as e:
            logger.error(f"Error storing spans: {e}")
    
    async def _cleanup_loop(self):
        """Background cleanup of old traces."""
        while self.running:
            try:
                await self._cleanup_old_traces()
                await asyncio.sleep(300)  # Clean up every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_old_traces(self):
        """Clean up old completed traces."""
        current_time = time.time()
        cleanup_age = 3600  # 1 hour
        
        with self.lock:
            traces_to_remove = []
            
            for trace_id, spans in self.completed_traces.items():
                if spans:
                    latest_span = max(spans, key=lambda s: s.end_time or s.start_time)
                    span_time = latest_span.end_time or latest_span.start_time
                    
                    if current_time - span_time > cleanup_age:
                        traces_to_remove.append(trace_id)
            
            for trace_id in traces_to_remove:
                del self.completed_traces[trace_id]
            
            if traces_to_remove:
                logger.debug(f"Cleaned up {len(traces_to_remove)} old traces")
    
    def get_trace(self, trace_id: str) -> Optional[List[TraceSpan]]:
        """Get all spans for a trace."""
        with self.lock:
            return self.completed_traces.get(trace_id, []).copy()
    
    def get_traces(self, limit: int = 100) -> List[Tuple[str, List[TraceSpan]]]:
        """Get recent traces."""
        with self.lock:
            traces = list(self.completed_traces.items())
            
            # Sort by most recent span in trace
            def trace_time(item):
                trace_id, spans = item
                if spans:
                    return max(span.end_time or span.start_time for span in spans)
                return 0
            
            traces.sort(key=trace_time, reverse=True)
            return traces[:limit]
    
    def analyze_trace_performance(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Analyze performance metrics for a trace."""
        spans = self.get_trace(trace_id)
        if not spans:
            return None
        
        # Calculate metrics
        total_duration = max(span.end_time or span.start_time for span in spans) - \
                        min(span.start_time for span in spans)
        
        span_durations = [span.duration for span in spans if span.duration]
        operation_times = defaultdict(list)
        service_times = defaultdict(list)
        error_count = 0
        
        for span in spans:
            if span.duration:
                operation_times[span.operation_name].append(span.duration)
                service_times[span.service_name].append(span.duration)
            
            if span.status == SpanStatus.ERROR:
                error_count += 1
        
        # Calculate statistics
        def calc_stats(durations):
            if not durations:
                return {}
            return {
                'count': len(durations),
                'total': sum(durations),
                'avg': sum(durations) / len(durations),
                'min': min(durations),
                'max': max(durations)
            }
        
        return {
            'trace_id': trace_id,
            'total_spans': len(spans),
            'total_duration': total_duration,
            'error_count': error_count,
            'error_rate': (error_count / len(spans)) * 100 if spans else 0,
            'span_statistics': calc_stats(span_durations),
            'operation_times': {op: calc_stats(times) for op, times in operation_times.items()},
            'service_times': {svc: calc_stats(times) for svc, times in service_times.items()},
            'critical_path': self._calculate_critical_path(spans)
        }
    
    def _calculate_critical_path(self, spans: List[TraceSpan]) -> List[Dict[str, Any]]:
        """Calculate the critical path through a trace."""
        if not spans:
            return []
        
        # Build span hierarchy
        span_map = {span.span_id: span for span in spans}
        root_spans = [span for span in spans if not span.parent_span_id]
        
        def find_longest_path(span: TraceSpan) -> Tuple[float, List[TraceSpan]]:
            children = [s for s in spans if s.parent_span_id == span.span_id]
            
            if not children:
                return span.duration or 0, [span]
            
            longest_duration = 0
            longest_path = [span]
            
            for child in children:
                child_duration, child_path = find_longest_path(child)
                total_duration = (span.duration or 0) + child_duration
                
                if total_duration > longest_duration:
                    longest_duration = total_duration
                    longest_path = [span] + child_path
            
            return longest_duration, longest_path
        
        # Find critical path from each root
        critical_paths = []
        for root in root_spans:
            duration, path = find_longest_path(root)
            critical_paths.append((duration, path))
        
        if not critical_paths:
            return []
        
        # Return the longest critical path
        _, longest_path = max(critical_paths, key=lambda x: x[0])
        
        return [
            {
                'span_id': span.span_id,
                'operation_name': span.operation_name,
                'service_name': span.service_name,
                'duration': span.duration,
                'cumulative_duration': sum(s.duration or 0 for s in longest_path[:i+1])
            }
            for i, span in enumerate(longest_path)
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tracer statistics."""
        with self.lock:
            return {
                'service_name': self.service_name,
                'running': self.running,
                'sampling_rate': self.sampling_rate,
                'active_spans': len(self.active_spans),
                'completed_traces': len(self.completed_traces),
                'pending_export': len(self.span_buffer),
                'exporters': len(self.exporters),
                **self.stats
            }
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()


# Built-in exporters
class ConsoleExporter:
    """Export traces to console for debugging."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def __call__(self, spans: List[TraceSpan]):
        for span in spans:
            status_symbol = "✓" if span.status == SpanStatus.OK else "✗"
            duration_ms = (span.duration * 1000) if span.duration else 0
            
            print(f"{status_symbol} {span.service_name}.{span.operation_name} "
                  f"({span.span_type.value}) - {duration_ms:.2f}ms")
            
            if self.verbose:
                if span.tags:
                    print(f"  Tags: {span.tags}")
                if span.logs:
                    for log in span.logs[-3:]:  # Show last 3 logs
                        print(f"  Log: {log['message']}")


class JSONFileExporter:
    """Export traces to JSON file."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def __call__(self, spans: List[TraceSpan]):
        try:
            Path(self.file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.file_path, 'a') as f:
                for span in spans:
                    f.write(json.dumps(span.to_dict()) + '\n')
        
        except Exception as e:
            logger.error(f"Error exporting to JSON file: {e}")


class HTTPExporter:
    """Export traces to HTTP endpoint (e.g., Jaeger, Zipkin)."""
    
    def __init__(self, endpoint_url: str, batch_size: int = 100):
        self.endpoint_url = endpoint_url
        self.batch_size = batch_size
    
    def __call__(self, spans: List[TraceSpan]):
        try:
            # This is a placeholder for actual HTTP export implementation
            # In practice, you'd format spans according to the target system's API
            
            payload = {
                'spans': [span.to_dict() for span in spans],
                'timestamp': time.time()
            }
            
            # Would use aiohttp or requests to POST to endpoint
            logger.debug(f"Would export {len(spans)} spans to {self.endpoint_url}")
            
        except Exception as e:
            logger.error(f"Error exporting via HTTP: {e}")


# Global tracer instance
_global_tracer: Optional[DistributedTracer] = None

def init_global_tracer(service_name: str, **kwargs) -> DistributedTracer:
    """Initialize global tracer instance."""
    global _global_tracer
    _global_tracer = DistributedTracer(service_name, **kwargs)
    return _global_tracer

def get_global_tracer() -> Optional[DistributedTracer]:
    """Get global tracer instance."""
    return _global_tracer

def trace(operation_name: str, span_type: SpanType = SpanType.CUSTOM, **kwargs):
    """Convenience decorator using global tracer."""
    tracer = get_global_tracer()
    if tracer:
        return tracer.trace_function(operation_name, span_type, **kwargs)
    else:
        # No-op decorator if no tracer
        def decorator(func):
            return func
        return decorator