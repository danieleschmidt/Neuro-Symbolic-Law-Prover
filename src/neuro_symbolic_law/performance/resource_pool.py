"""
Advanced resource pooling for concurrent processing and connection management.
"""

import asyncio
import threading
import time
import logging
from typing import Dict, List, Optional, Any, Callable, Generic, TypeVar, Union
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue, PriorityQueue, Empty
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
import weakref
from contextlib import asynccontextmanager, contextmanager
import psutil

logger = logging.getLogger(__name__)

T = TypeVar('T')


class PoolType(Enum):
    """Types of resource pools."""
    THREAD_POOL = "thread_pool"
    PROCESS_POOL = "process_pool"
    CONNECTION_POOL = "connection_pool"
    WORKER_POOL = "worker_pool"
    GPU_POOL = "gpu_pool"


class ResourceState(Enum):
    """Resource states in pool."""
    AVAILABLE = "available"
    IN_USE = "in_use"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    RETIRED = "retired"


@dataclass
class PoolStats:
    """Resource pool statistics."""
    pool_size: int = 0
    active_resources: int = 0
    available_resources: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_wait_time: float = 0.0
    avg_usage_time: float = 0.0
    peak_usage: int = 0
    created_resources: int = 0
    destroyed_resources: int = 0
    
    def update_request(self, wait_time: float, usage_time: float, success: bool):
        """Update statistics for a resource request."""
        self.total_requests += 1
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        # Update average wait time
        if self.total_requests == 1:
            self.avg_wait_time = wait_time
            self.avg_usage_time = usage_time
        else:
            self.avg_wait_time = (
                (self.avg_wait_time * (self.total_requests - 1) + wait_time) / 
                self.total_requests
            )
            self.avg_usage_time = (
                (self.avg_usage_time * (self.total_requests - 1) + usage_time) / 
                self.total_requests
            )
        
        # Update peak usage
        self.peak_usage = max(self.peak_usage, self.active_resources)
    
    def get_success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    def get_utilization_rate(self) -> float:
        """Calculate pool utilization percentage."""
        if self.pool_size == 0:
            return 0.0
        return (self.active_resources / self.pool_size) * 100


@dataclass
class PoolResource(Generic[T]):
    """Wrapper for pooled resources."""
    resource: T
    resource_id: str
    state: ResourceState
    created_at: float
    last_used_at: float
    usage_count: int = 0
    error_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_healthy(self, max_age: Optional[float] = None, max_errors: int = 5) -> bool:
        """Check if resource is healthy and usable."""
        if self.state != ResourceState.AVAILABLE:
            return False
        
        if self.error_count > max_errors:
            return False
        
        if max_age and (time.time() - self.created_at) > max_age:
            return False
        
        return True
    
    def mark_used(self):
        """Mark resource as used."""
        self.last_used_at = time.time()
        self.usage_count += 1
        self.state = ResourceState.IN_USE
    
    def mark_available(self):
        """Mark resource as available."""
        self.state = ResourceState.AVAILABLE
    
    def mark_error(self):
        """Mark resource as having an error."""
        self.error_count += 1
        self.state = ResourceState.ERROR


class ResourcePool(Generic[T]):
    """
    Advanced resource pool with health monitoring and auto-scaling.
    
    Supports different types of resources (connections, workers, GPU contexts)
    with intelligent lifecycle management and performance optimization.
    """
    
    def __init__(self,
                 resource_factory: Callable[[], T],
                 resource_destroyer: Optional[Callable[[T], None]] = None,
                 min_size: int = 1,
                 max_size: int = 10,
                 max_idle_time: float = 300,  # 5 minutes
                 max_resource_age: Optional[float] = None,
                 health_check_interval: float = 60,  # 1 minute
                 health_checker: Optional[Callable[[T], bool]] = None,
                 pool_type: PoolType = PoolType.WORKER_POOL):
        """
        Initialize resource pool.
        
        Args:
            resource_factory: Function to create new resources
            resource_destroyer: Function to clean up resources
            min_size: Minimum pool size
            max_size: Maximum pool size
            max_idle_time: Maximum idle time before resource cleanup
            max_resource_age: Maximum resource age before replacement
            health_check_interval: Health check frequency in seconds
            health_checker: Function to check resource health
            pool_type: Type of resource pool
        """
        self.resource_factory = resource_factory
        self.resource_destroyer = resource_destroyer
        self.min_size = min_size
        self.max_size = max_size
        self.max_idle_time = max_idle_time
        self.max_resource_age = max_resource_age
        self.health_check_interval = health_check_interval
        self.health_checker = health_checker
        self.pool_type = pool_type
        
        # Pool state
        self.resources: Dict[str, PoolResource[T]] = {}
        self.available_queue: Queue[str] = Queue()
        self.stats = PoolStats()
        
        # Synchronization
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        
        # Background maintenance
        self.maintenance_thread: Optional[threading.Thread] = None
        self.maintenance_stop_event = threading.Event()
        self.last_health_check = 0
        
        # Auto-scaling
        self.scaling_enabled = True
        self.scale_up_threshold = 0.8  # Scale up at 80% utilization
        self.scale_down_threshold = 0.3  # Scale down at 30% utilization
        self.last_scale_time = 0
        self.scale_cooldown = 30  # seconds
        
        # Initialize pool
        self._initialize_pool()
        self._start_maintenance()
        
        logger.info(f"ResourcePool initialized: type={pool_type.value}, size={min_size}-{max_size}")
    
    def _initialize_pool(self):
        """Initialize pool with minimum resources."""
        with self.lock:
            for _ in range(self.min_size):
                self._create_resource()
    
    def _create_resource(self) -> Optional[str]:
        """Create a new resource and add it to the pool."""
        if len(self.resources) >= self.max_size:
            return None
        
        try:
            resource = self.resource_factory()
            resource_id = f"{self.pool_type.value}_{len(self.resources)}_{time.time()}"
            
            pool_resource = PoolResource(
                resource=resource,
                resource_id=resource_id,
                state=ResourceState.AVAILABLE,
                created_at=time.time(),
                last_used_at=time.time()
            )
            
            self.resources[resource_id] = pool_resource
            self.available_queue.put(resource_id)
            self.stats.pool_size += 1
            self.stats.available_resources += 1
            self.stats.created_resources += 1
            
            logger.debug(f"Created resource {resource_id}")
            return resource_id
            
        except Exception as e:
            logger.error(f"Failed to create resource: {e}")
            return None
    
    def _destroy_resource(self, resource_id: str):
        """Destroy a resource and remove it from the pool."""
        if resource_id not in self.resources:
            return
        
        pool_resource = self.resources[resource_id]
        
        # Clean up resource if destroyer provided
        if self.resource_destroyer:
            try:
                self.resource_destroyer(pool_resource.resource)
            except Exception as e:
                logger.error(f"Error destroying resource {resource_id}: {e}")
        
        # Update state
        if pool_resource.state == ResourceState.AVAILABLE:
            self.stats.available_resources -= 1
        elif pool_resource.state == ResourceState.IN_USE:
            self.stats.active_resources -= 1
        
        # Remove from pool
        del self.resources[resource_id]
        self.stats.pool_size -= 1
        self.stats.destroyed_resources += 1
        
        logger.debug(f"Destroyed resource {resource_id}")
    
    @asynccontextmanager
    async def acquire(self, timeout: Optional[float] = None):
        """
        Acquire a resource from the pool (async context manager).
        
        Args:
            timeout: Maximum wait time for resource acquisition
            
        Yields:
            The acquired resource
        """
        start_time = time.time()
        resource_id = None
        pool_resource = None
        
        try:
            # Acquire resource
            resource_id = await self._acquire_resource(timeout)
            if resource_id is None:
                raise TimeoutError("Failed to acquire resource within timeout")
            
            with self.lock:
                pool_resource = self.resources[resource_id]
                pool_resource.mark_used()
                self.stats.active_resources += 1
                self.stats.available_resources -= 1
            
            wait_time = time.time() - start_time
            logger.debug(f"Acquired resource {resource_id} after {wait_time:.3f}s")
            
            usage_start = time.time()
            
            try:
                yield pool_resource.resource
                success = True
            except Exception as e:
                logger.error(f"Error using resource {resource_id}: {e}")
                pool_resource.mark_error()
                success = False
                raise
            
        finally:
            # Release resource
            if resource_id and pool_resource:
                usage_time = time.time() - usage_start if 'usage_start' in locals() else 0
                wait_time = time.time() - start_time
                
                await self._release_resource(resource_id)
                
                # Update statistics
                self.stats.update_request(wait_time, usage_time, success if 'success' in locals() else False)
    
    @contextmanager
    def acquire_sync(self, timeout: Optional[float] = None):
        """
        Acquire a resource from the pool (sync context manager).
        
        Args:
            timeout: Maximum wait time for resource acquisition
            
        Yields:
            The acquired resource
        """
        start_time = time.time()
        resource_id = None
        pool_resource = None
        
        try:
            # Acquire resource synchronously
            resource_id = self._acquire_resource_sync(timeout)
            if resource_id is None:
                raise TimeoutError("Failed to acquire resource within timeout")
            
            with self.lock:
                pool_resource = self.resources[resource_id]
                pool_resource.mark_used()
                self.stats.active_resources += 1
                self.stats.available_resources -= 1
            
            wait_time = time.time() - start_time
            logger.debug(f"Acquired resource {resource_id} after {wait_time:.3f}s")
            
            usage_start = time.time()
            success = True
            
            try:
                yield pool_resource.resource
            except Exception as e:
                logger.error(f"Error using resource {resource_id}: {e}")
                pool_resource.mark_error()
                success = False
                raise
            
        finally:
            # Release resource
            if resource_id and pool_resource:
                usage_time = time.time() - usage_start if 'usage_start' in locals() else 0
                wait_time = time.time() - start_time
                
                self._release_resource_sync(resource_id)
                
                # Update statistics
                self.stats.update_request(wait_time, usage_time, success)
    
    async def _acquire_resource(self, timeout: Optional[float] = None) -> Optional[str]:
        """Acquire a resource asynchronously."""
        start_time = time.time()
        
        while True:
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                return None
            
            # Try to get available resource
            try:
                resource_id = self.available_queue.get_nowait()
                
                # Verify resource is still valid
                with self.lock:
                    if resource_id in self.resources:
                        pool_resource = self.resources[resource_id]
                        if pool_resource.is_healthy(self.max_resource_age):
                            return resource_id
                        else:
                            # Resource is unhealthy, destroy it
                            self._destroy_resource(resource_id)
                
            except Empty:
                pass
            
            # Try to create new resource if needed
            with self.lock:
                if len(self.resources) < self.max_size:
                    new_resource_id = self._create_resource()
                    if new_resource_id:
                        # Remove from available queue and return
                        try:
                            self.available_queue.get_nowait()  # Should be our new resource
                            return new_resource_id
                        except Empty:
                            pass
            
            # Wait a bit before retrying
            await asyncio.sleep(0.01)
    
    def _acquire_resource_sync(self, timeout: Optional[float] = None) -> Optional[str]:
        """Acquire a resource synchronously."""
        with self.condition:
            start_time = time.time()
            
            while True:
                # Check timeout
                if timeout and (time.time() - start_time) > timeout:
                    return None
                
                # Try to get available resource
                try:
                    resource_id = self.available_queue.get_nowait()
                    
                    # Verify resource is still valid
                    if resource_id in self.resources:
                        pool_resource = self.resources[resource_id]
                        if pool_resource.is_healthy(self.max_resource_age):
                            return resource_id
                        else:
                            # Resource is unhealthy, destroy it
                            self._destroy_resource(resource_id)
                
                except Empty:
                    pass
                
                # Try to create new resource if needed
                if len(self.resources) < self.max_size:
                    new_resource_id = self._create_resource()
                    if new_resource_id:
                        # Remove from available queue and return
                        try:
                            self.available_queue.get_nowait()
                            return new_resource_id
                        except Empty:
                            pass
                
                # Wait for resource to become available
                remaining_time = None
                if timeout:
                    remaining_time = timeout - (time.time() - start_time)
                    if remaining_time <= 0:
                        return None
                
                self.condition.wait(min(remaining_time or 0.1, 0.1))
    
    async def _release_resource(self, resource_id: str):
        """Release a resource back to the pool asynchronously."""
        with self.lock:
            if resource_id not in self.resources:
                return
            
            pool_resource = self.resources[resource_id]
            
            # Check if resource should be retired
            if not pool_resource.is_healthy(self.max_resource_age):
                self._destroy_resource(resource_id)
                # Create replacement if below minimum
                if len(self.resources) < self.min_size:
                    self._create_resource()
            else:
                # Return to available pool
                pool_resource.mark_available()
                self.available_queue.put(resource_id)
                self.stats.active_resources -= 1
                self.stats.available_resources += 1
            
            # Notify waiting threads
            self.condition.notify()
        
        # Trigger auto-scaling check
        await self._check_auto_scaling()
    
    def _release_resource_sync(self, resource_id: str):
        """Release a resource back to the pool synchronously."""
        with self.condition:
            if resource_id not in self.resources:
                return
            
            pool_resource = self.resources[resource_id]
            
            # Check if resource should be retired
            if not pool_resource.is_healthy(self.max_resource_age):
                self._destroy_resource(resource_id)
                # Create replacement if below minimum
                if len(self.resources) < self.min_size:
                    self._create_resource()
            else:
                # Return to available pool
                pool_resource.mark_available()
                self.available_queue.put(resource_id)
                self.stats.active_resources -= 1
                self.stats.available_resources += 1
            
            # Notify waiting threads
            self.condition.notify_all()
        
        # Trigger auto-scaling check (sync version)
        self._check_auto_scaling_sync()
    
    async def _check_auto_scaling(self):
        """Check if auto-scaling is needed."""
        if not self.scaling_enabled:
            return
        
        current_time = time.time()
        if current_time - self.last_scale_time < self.scale_cooldown:
            return
        
        with self.lock:
            utilization = self.stats.get_utilization_rate() / 100
            
            # Scale up if high utilization and not at max size
            if (utilization > self.scale_up_threshold and 
                len(self.resources) < self.max_size and
                self.available_queue.qsize() == 0):
                
                new_resource_id = self._create_resource()
                if new_resource_id:
                    self.last_scale_time = current_time
                    logger.info(f"Scaled up pool: created resource {new_resource_id}")
            
            # Scale down if low utilization and above min size
            elif (utilization < self.scale_down_threshold and 
                  len(self.resources) > self.min_size and
                  self.available_queue.qsize() > 1):
                
                try:
                    resource_id = self.available_queue.get_nowait()
                    self._destroy_resource(resource_id)
                    self.last_scale_time = current_time
                    logger.info(f"Scaled down pool: removed resource {resource_id}")
                except Empty:
                    pass
    
    def _check_auto_scaling_sync(self):
        """Synchronous version of auto-scaling check."""
        if not self.scaling_enabled:
            return
        
        current_time = time.time()
        if current_time - self.last_scale_time < self.scale_cooldown:
            return
        
        with self.lock:
            utilization = self.stats.get_utilization_rate() / 100
            
            # Scale up if high utilization
            if (utilization > self.scale_up_threshold and 
                len(self.resources) < self.max_size):
                
                new_resource_id = self._create_resource()
                if new_resource_id:
                    self.last_scale_time = current_time
                    logger.info(f"Scaled up pool: created resource {new_resource_id}")
            
            # Scale down if low utilization
            elif (utilization < self.scale_down_threshold and 
                  len(self.resources) > self.min_size):
                
                try:
                    resource_id = self.available_queue.get_nowait()
                    self._destroy_resource(resource_id)
                    self.last_scale_time = current_time
                    logger.info(f"Scaled down pool: removed resource {resource_id}")
                except Empty:
                    pass
    
    def _start_maintenance(self):
        """Start background maintenance thread."""
        if self.maintenance_thread is not None:
            return
        
        self.maintenance_thread = threading.Thread(
            target=self._maintenance_worker,
            name=f"{self.pool_type.value}_maintenance",
            daemon=True
        )
        self.maintenance_thread.start()
    
    def _maintenance_worker(self):
        """Background maintenance worker."""
        while not self.maintenance_stop_event.is_set():
            try:
                current_time = time.time()
                
                # Perform health checks
                if current_time - self.last_health_check > self.health_check_interval:
                    self._perform_health_checks()
                    self.last_health_check = current_time
                
                # Clean up idle resources
                self._cleanup_idle_resources()
                
                # Sleep before next maintenance cycle
                self.maintenance_stop_event.wait(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Maintenance worker error: {e}")
                time.sleep(1)
    
    def _perform_health_checks(self):
        """Perform health checks on all resources."""
        if not self.health_checker:
            return
        
        unhealthy_resources = []
        
        with self.lock:
            for resource_id, pool_resource in self.resources.items():
                if pool_resource.state == ResourceState.AVAILABLE:
                    try:
                        if not self.health_checker(pool_resource.resource):
                            unhealthy_resources.append(resource_id)
                            pool_resource.mark_error()
                    except Exception as e:
                        logger.error(f"Health check error for resource {resource_id}: {e}")
                        unhealthy_resources.append(resource_id)
                        pool_resource.mark_error()
        
        # Remove unhealthy resources
        for resource_id in unhealthy_resources:
            with self.lock:
                self._destroy_resource(resource_id)
                # Create replacement if below minimum
                if len(self.resources) < self.min_size:
                    self._create_resource()
        
        if unhealthy_resources:
            logger.info(f"Removed {len(unhealthy_resources)} unhealthy resources")
    
    def _cleanup_idle_resources(self):
        """Clean up resources that have been idle too long."""
        if not self.max_idle_time:
            return
        
        current_time = time.time()
        idle_resources = []
        
        with self.lock:
            for resource_id, pool_resource in self.resources.items():
                if (pool_resource.state == ResourceState.AVAILABLE and
                    (current_time - pool_resource.last_used_at) > self.max_idle_time and
                    len(self.resources) > self.min_size):
                    idle_resources.append(resource_id)
        
        # Remove idle resources
        for resource_id in idle_resources:
            with self.lock:
                # Double-check it's still idle and we're above min size
                if (resource_id in self.resources and
                    len(self.resources) > self.min_size):
                    try:
                        # Try to remove from available queue
                        temp_queue = Queue()
                        found = False
                        
                        while not self.available_queue.empty():
                            item = self.available_queue.get_nowait()
                            if item == resource_id and not found:
                                found = True  # Don't put back in queue
                            else:
                                temp_queue.put(item)
                        
                        # Restore queue
                        while not temp_queue.empty():
                            self.available_queue.put(temp_queue.get_nowait())
                        
                        if found:
                            self._destroy_resource(resource_id)
                    except Empty:
                        pass
        
        if idle_resources:
            logger.debug(f"Cleaned up {len(idle_resources)} idle resources")
    
    def get_stats(self) -> PoolStats:
        """Get current pool statistics."""
        with self.lock:
            # Update current counts
            self.stats.pool_size = len(self.resources)
            self.stats.available_resources = self.available_queue.qsize()
            self.stats.active_resources = len([
                r for r in self.resources.values() 
                if r.state == ResourceState.IN_USE
            ])
            
            return self.stats
    
    def get_resource_details(self) -> List[Dict[str, Any]]:
        """Get detailed information about all resources."""
        details = []
        
        with self.lock:
            for resource_id, pool_resource in self.resources.items():
                details.append({
                    'resource_id': resource_id,
                    'state': pool_resource.state.value,
                    'created_at': pool_resource.created_at,
                    'last_used_at': pool_resource.last_used_at,
                    'usage_count': pool_resource.usage_count,
                    'error_count': pool_resource.error_count,
                    'age_seconds': time.time() - pool_resource.created_at,
                    'idle_seconds': time.time() - pool_resource.last_used_at,
                    'metadata': pool_resource.metadata.copy()
                })
        
        return details
    
    def resize(self, new_min_size: int, new_max_size: int):
        """Resize the pool with new minimum and maximum sizes."""
        with self.lock:
            old_min, old_max = self.min_size, self.max_size
            self.min_size = max(0, new_min_size)
            self.max_size = max(self.min_size, new_max_size)
            
            current_size = len(self.resources)
            
            # Scale up if below new minimum
            if current_size < self.min_size:
                for _ in range(self.min_size - current_size):
                    self._create_resource()
            
            # Scale down if above new maximum
            elif current_size > self.max_size:
                resources_to_remove = current_size - self.max_size
                removed = 0
                
                while removed < resources_to_remove and not self.available_queue.empty():
                    try:
                        resource_id = self.available_queue.get_nowait()
                        self._destroy_resource(resource_id)
                        removed += 1
                    except Empty:
                        break
            
            logger.info(f"Pool resized: {old_min}-{old_max} -> {self.min_size}-{self.max_size}")
    
    def shutdown(self):
        """Shutdown the pool and clean up all resources."""
        logger.info(f"Shutting down {self.pool_type.value} pool")
        
        # Stop maintenance thread
        if self.maintenance_thread:
            self.maintenance_stop_event.set()
            self.maintenance_thread.join(timeout=5)
        
        # Destroy all resources
        with self.lock:
            resource_ids = list(self.resources.keys())
            for resource_id in resource_ids:
                try:
                    self._destroy_resource(resource_id)
                except Exception as e:
                    logger.error(f"Error destroying resource {resource_id}: {e}")
        
        logger.info("Pool shutdown complete")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


class ConnectionPool(ResourcePool):
    """
    Specialized connection pool for database/network connections.
    """
    
    def __init__(self,
                 connection_factory: Callable[[], Any],
                 connection_validator: Optional[Callable[[Any], bool]] = None,
                 max_connections: int = 20,
                 max_connection_age: float = 3600,  # 1 hour
                 connection_timeout: float = 30):
        """
        Initialize connection pool.
        
        Args:
            connection_factory: Function to create new connections
            connection_validator: Function to validate connection health
            max_connections: Maximum number of connections
            max_connection_age: Maximum connection age in seconds
            connection_timeout: Connection timeout in seconds
        """
        
        def connection_destroyer(conn):
            """Close connection safely."""
            try:
                if hasattr(conn, 'close'):
                    conn.close()
                elif hasattr(conn, 'disconnect'):
                    conn.disconnect()
            except Exception as e:
                logger.error(f"Error closing connection: {e}")
        
        super().__init__(
            resource_factory=connection_factory,
            resource_destroyer=connection_destroyer,
            min_size=1,
            max_size=max_connections,
            max_idle_time=300,  # 5 minutes
            max_resource_age=max_connection_age,
            health_check_interval=60,
            health_checker=connection_validator,
            pool_type=PoolType.CONNECTION_POOL
        )
        
        self.connection_timeout = connection_timeout
    
    @asynccontextmanager
    async def get_connection(self, timeout: Optional[float] = None):
        """Get a connection from the pool."""
        async with self.acquire(timeout or self.connection_timeout) as connection:
            yield connection
    
    @contextmanager
    def get_connection_sync(self, timeout: Optional[float] = None):
        """Get a connection from the pool (synchronous)."""
        with self.acquire_sync(timeout or self.connection_timeout) as connection:
            yield connection


class WorkerPool:
    """
    High-performance worker pool for CPU and I/O bound tasks.
    
    Combines thread and process pools with intelligent task scheduling.
    """
    
    def __init__(self,
                 cpu_workers: int = None,
                 io_workers: int = None,
                 process_workers: int = None,
                 enable_gpu: bool = False):
        """
        Initialize worker pool.
        
        Args:
            cpu_workers: Number of CPU-bound task workers
            io_workers: Number of I/O-bound task workers  
            process_workers: Number of process workers for CPU-intensive tasks
            enable_gpu: Enable GPU workers if available
        """
        # Auto-detect optimal worker counts
        cpu_count = psutil.cpu_count(logical=False) or 4
        
        self.cpu_workers = cpu_workers or min(cpu_count, 8)
        self.io_workers = io_workers or min(cpu_count * 2, 20)
        self.process_workers = process_workers or max(1, cpu_count - 1)
        
        # Create executor pools
        self.cpu_executor = ThreadPoolExecutor(
            max_workers=self.cpu_workers,
            thread_name_prefix="nsl_cpu"
        )
        
        self.io_executor = ThreadPoolExecutor(
            max_workers=self.io_workers,
            thread_name_prefix="nsl_io"
        )
        
        self.process_executor = ProcessPoolExecutor(
            max_workers=self.process_workers
        )
        
        # Task tracking
        self.active_tasks: Dict[str, Future] = {}
        self.task_stats = {
            'cpu_tasks': 0,
            'io_tasks': 0,
            'process_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0
        }
        
        # GPU support
        self.gpu_available = False
        if enable_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    self.gpu_available = True
                    self.gpu_devices = torch.cuda.device_count()
                    logger.info(f"GPU support enabled: {self.gpu_devices} devices")
            except ImportError:
                logger.info("GPU requested but PyTorch not available")
        
        logger.info(f"WorkerPool initialized: CPU={self.cpu_workers}, IO={self.io_workers}, Process={self.process_workers}")
    
    async def submit_cpu_task(self, func: Callable, *args, **kwargs) -> Any:
        """Submit CPU-bound task."""
        loop = asyncio.get_event_loop()
        
        task_id = f"cpu_{time.time()}_{id(func)}"
        future = loop.run_in_executor(self.cpu_executor, func, *args, **kwargs)
        
        self.active_tasks[task_id] = future
        self.task_stats['cpu_tasks'] += 1
        
        try:
            result = await future
            self.task_stats['completed_tasks'] += 1
            return result
        except Exception as e:
            self.task_stats['failed_tasks'] += 1
            raise
        finally:
            self.active_tasks.pop(task_id, None)
    
    async def submit_io_task(self, func: Callable, *args, **kwargs) -> Any:
        """Submit I/O-bound task."""
        loop = asyncio.get_event_loop()
        
        task_id = f"io_{time.time()}_{id(func)}"
        future = loop.run_in_executor(self.io_executor, func, *args, **kwargs)
        
        self.active_tasks[task_id] = future
        self.task_stats['io_tasks'] += 1
        
        try:
            result = await future
            self.task_stats['completed_tasks'] += 1
            return result
        except Exception as e:
            self.task_stats['failed_tasks'] += 1
            raise
        finally:
            self.active_tasks.pop(task_id, None)
    
    async def submit_process_task(self, func: Callable, *args, **kwargs) -> Any:
        """Submit process-based task for CPU-intensive work."""
        loop = asyncio.get_event_loop()
        
        task_id = f"process_{time.time()}_{id(func)}"
        future = loop.run_in_executor(self.process_executor, func, *args, **kwargs)
        
        self.active_tasks[task_id] = future
        self.task_stats['process_tasks'] += 1
        
        try:
            result = await future
            self.task_stats['completed_tasks'] += 1
            return result
        except Exception as e:
            self.task_stats['failed_tasks'] += 1
            raise
        finally:
            self.active_tasks.pop(task_id, None)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        return {
            'active_tasks': len(self.active_tasks),
            'cpu_workers': self.cpu_workers,
            'io_workers': self.io_workers,
            'process_workers': self.process_workers,
            'gpu_available': self.gpu_available,
            'gpu_devices': getattr(self, 'gpu_devices', 0),
            **self.task_stats
        }
    
    def shutdown(self, wait: bool = True):
        """Shutdown all worker pools."""
        logger.info("Shutting down worker pools")
        
        # Cancel active tasks
        for task_id, future in self.active_tasks.items():
            if not future.done():
                future.cancel()
        
        # Shutdown executors
        self.cpu_executor.shutdown(wait=wait)
        self.io_executor.shutdown(wait=wait)
        self.process_executor.shutdown(wait=wait)
        
        logger.info("Worker pools shutdown complete")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()