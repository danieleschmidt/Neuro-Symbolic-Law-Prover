"""
Comprehensive health monitoring and service discovery for distributed systems.
"""

import time
import asyncio
import logging
import json
import threading
import platform
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
            class mem: percent = 60.0; total = 8*1024**3; available = 4*1024**3
            return mem()
from typing import Dict, List, Optional, Any, Callable, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
from pathlib import Path
import weakref

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    MAINTENANCE = "maintenance"


class CheckType(Enum):
    """Types of health checks."""
    READINESS = "readiness"
    LIVENESS = "liveness"
    STARTUP = "startup"
    DEPENDENCY = "dependency"
    RESOURCE = "resource"
    CUSTOM = "custom"


@dataclass
class HealthCheck:
    """Defines a health check configuration."""
    name: str
    check_type: CheckType
    check_function: Callable[[], Union[bool, Tuple[bool, str]]]
    interval: float  # seconds
    timeout: float  # seconds
    enabled: bool = True
    critical: bool = False  # If true, failure marks service as critical
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    
    # Failure handling
    failure_threshold: int = 3  # Consecutive failures before marking as unhealthy
    success_threshold: int = 1  # Consecutive successes before marking as healthy
    
    # State tracking
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_check_time: float = 0
    last_success_time: float = 0
    last_failure_time: float = 0
    last_result: Optional[bool] = None
    last_message: str = ""
    
    def should_check(self) -> bool:
        """Check if it's time to run this health check."""
        return (time.time() - self.last_check_time) >= self.interval
    
    async def execute(self) -> Tuple[bool, str]:
        """Execute the health check with timeout."""
        if not self.enabled:
            return True, "Check disabled"
        
        try:
            # Run check with timeout
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, self.check_function),
                timeout=self.timeout
            )
            
            self.last_check_time = time.time()
            
            # Handle result
            if isinstance(result, tuple):
                success, message = result
            else:
                success, message = result, ""
            
            # Update counters
            if success:
                self.consecutive_successes += 1
                self.consecutive_failures = 0
                self.last_success_time = time.time()
            else:
                self.consecutive_failures += 1
                self.consecutive_successes = 0
                self.last_failure_time = time.time()
            
            self.last_result = success
            self.last_message = message
            
            return success, message
            
        except asyncio.TimeoutError:
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            self.last_failure_time = time.time()
            self.last_result = False
            self.last_message = f"Health check timed out after {self.timeout}s"
            
            return False, self.last_message
            
        except Exception as e:
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            self.last_failure_time = time.time()
            self.last_result = False
            self.last_message = f"Health check failed: {e}"
            
            return False, self.last_message
    
    def is_healthy(self) -> bool:
        """Determine if this check is currently healthy."""
        if self.last_result is None:
            return False
        
        if self.last_result:
            return self.consecutive_successes >= self.success_threshold
        else:
            return self.consecutive_failures < self.failure_threshold
    
    def get_status(self) -> HealthStatus:
        """Get the current health status."""
        if not self.enabled:
            return HealthStatus.UNKNOWN
        
        if self.last_result is None:
            return HealthStatus.UNKNOWN
        
        is_healthy = self.is_healthy()
        
        if is_healthy:
            return HealthStatus.HEALTHY
        elif self.critical:
            return HealthStatus.CRITICAL
        else:
            return HealthStatus.WARNING
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'name': self.name,
            'check_type': self.check_type.value,
            'enabled': self.enabled,
            'critical': self.critical,
            'description': self.description,
            'tags': self.tags,
            'interval': self.interval,
            'timeout': self.timeout,
            'failure_threshold': self.failure_threshold,
            'success_threshold': self.success_threshold,
            'consecutive_failures': self.consecutive_failures,
            'consecutive_successes': self.consecutive_successes,
            'last_check_time': self.last_check_time,
            'last_success_time': self.last_success_time,
            'last_failure_time': self.last_failure_time,
            'last_result': self.last_result,
            'last_message': self.last_message,
            'is_healthy': self.is_healthy(),
            'status': self.get_status().value
        }


@dataclass
class ServiceInstance:
    """Represents a service instance in the registry."""
    id: str
    name: str
    version: str
    host: str
    port: int
    health_check_url: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Registration info
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    
    # Health status
    health_status: HealthStatus = HealthStatus.UNKNOWN
    health_message: str = ""
    health_checks: Dict[str, bool] = field(default_factory=dict)
    
    def is_alive(self, heartbeat_timeout: float = 60.0) -> bool:
        """Check if service instance is alive based on heartbeat."""
        return (time.time() - self.last_heartbeat) <= heartbeat_timeout
    
    def update_heartbeat(self):
        """Update the heartbeat timestamp."""
        self.last_heartbeat = time.time()
    
    def update_health(self, status: HealthStatus, message: str = ""):
        """Update health status."""
        self.health_status = status
        self.health_message = message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ServiceInstance':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            name=data['name'],
            version=data['version'],
            host=data['host'],
            port=data['port'],
            health_check_url=data.get('health_check_url'),
            tags=data.get('tags', {}),
            metadata=data.get('metadata', {}),
            registered_at=data.get('registered_at', time.time()),
            last_heartbeat=data.get('last_heartbeat', time.time()),
            health_status=HealthStatus(data.get('health_status', 'unknown')),
            health_message=data.get('health_message', ''),
            health_checks=data.get('health_checks', {})
        )


class ServiceRegistry:
    """Service registry for discovering and monitoring service instances."""
    
    def __init__(self, 
                 heartbeat_timeout: float = 60.0,
                 cleanup_interval: float = 30.0):
        """
        Initialize service registry.
        
        Args:
            heartbeat_timeout: Timeout for service heartbeats
            cleanup_interval: Interval for cleaning up dead services
        """
        self.heartbeat_timeout = heartbeat_timeout
        self.cleanup_interval = cleanup_interval
        
        # Service storage
        self.services: Dict[str, ServiceInstance] = {}
        self.services_by_name: Dict[str, List[str]] = defaultdict(list)
        
        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Change listeners
        self.listeners: List[Callable[[str, ServiceInstance, str], None]] = []
        
        logger.info("Service registry initialized")
    
    async def start(self):
        """Start the service registry."""
        if self.running:
            return
        
        self.running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Service registry started")
    
    async def stop(self):
        """Stop the service registry."""
        self.running = False
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Service registry stopped")
    
    def register_service(self, service: ServiceInstance) -> bool:
        """Register a service instance."""
        with self.lock:
            # Check if already registered
            if service.id in self.services:
                logger.warning(f"Service {service.id} already registered")
                return False
            
            # Register service
            self.services[service.id] = service
            self.services_by_name[service.name].append(service.id)
            
            # Notify listeners
            self._notify_listeners(service.id, service, "registered")
            
            logger.info(f"Registered service: {service.name}@{service.host}:{service.port} (id: {service.id})")
            return True
    
    def deregister_service(self, service_id: str) -> bool:
        """Deregister a service instance."""
        with self.lock:
            if service_id not in self.services:
                return False
            
            service = self.services[service_id]
            
            # Remove from registries
            del self.services[service_id]
            if service_id in self.services_by_name[service.name]:
                self.services_by_name[service.name].remove(service_id)
            
            # Clean up empty service name entries
            if not self.services_by_name[service.name]:
                del self.services_by_name[service.name]
            
            # Notify listeners
            self._notify_listeners(service_id, service, "deregistered")
            
            logger.info(f"Deregistered service: {service.name} (id: {service_id})")
            return True
    
    def update_heartbeat(self, service_id: str) -> bool:
        """Update service heartbeat."""
        with self.lock:
            service = self.services.get(service_id)
            if service:
                service.update_heartbeat()
                return True
        return False
    
    def update_service_health(self, service_id: str, status: HealthStatus, 
                            message: str = "") -> bool:
        """Update service health status."""
        with self.lock:
            service = self.services.get(service_id)
            if service:
                old_status = service.health_status
                service.update_health(status, message)
                
                if old_status != status:
                    self._notify_listeners(service_id, service, "health_changed")
                
                return True
        return False
    
    def get_service(self, service_id: str) -> Optional[ServiceInstance]:
        """Get service by ID."""
        with self.lock:
            return self.services.get(service_id)
    
    def get_services_by_name(self, service_name: str) -> List[ServiceInstance]:
        """Get all instances of a service by name."""
        with self.lock:
            service_ids = self.services_by_name.get(service_name, [])
            return [self.services[sid] for sid in service_ids if sid in self.services]
    
    def get_healthy_services(self, service_name: str) -> List[ServiceInstance]:
        """Get healthy instances of a service."""
        services = self.get_services_by_name(service_name)
        return [
            service for service in services
            if service.health_status == HealthStatus.HEALTHY and service.is_alive(self.heartbeat_timeout)
        ]
    
    def get_all_services(self) -> List[ServiceInstance]:
        """Get all registered services."""
        with self.lock:
            return list(self.services.values())
    
    def add_listener(self, listener: Callable[[str, ServiceInstance, str], None]):
        """Add service change listener."""
        self.listeners.append(listener)
    
    def remove_listener(self, listener: Callable[[str, ServiceInstance, str], None]):
        """Remove service change listener."""
        if listener in self.listeners:
            self.listeners.remove(listener)
    
    def _notify_listeners(self, service_id: str, service: ServiceInstance, event: str):
        """Notify listeners of service changes."""
        for listener in self.listeners:
            try:
                listener(service_id, service, event)
            except Exception as e:
                logger.error(f"Error in service registry listener: {e}")
    
    async def _cleanup_loop(self):
        """Clean up dead services periodically."""
        while self.running:
            try:
                await self._cleanup_dead_services()
                await asyncio.sleep(self.cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(10)
    
    async def _cleanup_dead_services(self):
        """Remove services that haven't sent heartbeats."""
        current_time = time.time()
        dead_services = []
        
        with self.lock:
            for service_id, service in self.services.items():
                if not service.is_alive(self.heartbeat_timeout):
                    dead_services.append(service_id)
        
        # Remove dead services
        for service_id in dead_services:
            self.deregister_service(service_id)
            logger.info(f"Removed dead service: {service_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        with self.lock:
            status_counts = defaultdict(int)
            service_counts = defaultdict(int)
            
            for service in self.services.values():
                status_counts[service.health_status.value] += 1
                service_counts[service.name] += 1
            
            return {
                'total_services': len(self.services),
                'unique_service_names': len(self.services_by_name),
                'status_distribution': dict(status_counts),
                'service_counts': dict(service_counts),
                'heartbeat_timeout': self.heartbeat_timeout,
                'cleanup_interval': self.cleanup_interval,
                'running': self.running
            }


class HealthMonitor:
    """
    Comprehensive health monitoring system.
    
    Features:
    - Multiple health check types
    - Service registry integration
    - Automatic recovery detection
    - Health status aggregation
    - Notification integration
    """
    
    def __init__(self,
                 service_name: str,
                 service_version: str = "1.0.0",
                 check_interval: float = 30.0,
                 storage_path: Optional[str] = None):
        """
        Initialize health monitor.
        
        Args:
            service_name: Name of the service being monitored
            service_version: Version of the service
            check_interval: Default health check interval
            storage_path: Path to store health check results
        """
        self.service_name = service_name
        self.service_version = service_version
        self.check_interval = check_interval
        self.storage_path = storage_path or f"/tmp/health_{service_name}.json"
        
        # Health checks
        self.health_checks: Dict[str, HealthCheck] = {}
        
        # Overall health status
        self.overall_status = HealthStatus.UNKNOWN
        self.overall_message = "No health checks configured"
        self.status_history: deque = deque(maxlen=1000)
        
        # Background tasks
        self.monitor_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Status change listeners
        self.status_listeners: List[Callable[[HealthStatus, str], None]] = []
        
        # Built-in health checks
        self._register_builtin_checks()
        
        logger.info(f"Health monitor initialized for {service_name} v{service_version}")
    
    def _register_builtin_checks(self):
        """Register built-in system health checks."""
        # CPU usage check
        def cpu_check():
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                return False, f"CPU usage too high: {cpu_percent}%"
            elif cpu_percent > 80:
                return True, f"CPU usage warning: {cpu_percent}%"
            return True, f"CPU usage normal: {cpu_percent}%"
        
        self.add_check(HealthCheck(
            name="cpu_usage",
            check_type=CheckType.RESOURCE,
            check_function=cpu_check,
            interval=60,
            timeout=5,
            critical=True,
            description="Monitor CPU usage"
        ))
        
        # Memory usage check
        def memory_check():
            memory = psutil.virtual_memory()
            if memory.percent > 95:
                return False, f"Memory usage critical: {memory.percent}%"
            elif memory.percent > 85:
                return True, f"Memory usage warning: {memory.percent}%"
            return True, f"Memory usage normal: {memory.percent}%"
        
        self.add_check(HealthCheck(
            name="memory_usage",
            check_type=CheckType.RESOURCE,
            check_function=memory_check,
            interval=60,
            timeout=5,
            critical=True,
            description="Monitor memory usage"
        ))
        
        # Disk space check
        def disk_check():
            disk = psutil.disk_usage('/')
            percent_used = (disk.used / disk.total) * 100
            if percent_used > 95:
                return False, f"Disk space critical: {percent_used:.1f}%"
            elif percent_used > 85:
                return True, f"Disk space warning: {percent_used:.1f}%"
            return True, f"Disk space normal: {percent_used:.1f}%"
        
        self.add_check(HealthCheck(
            name="disk_space",
            check_type=CheckType.RESOURCE,
            check_function=disk_check,
            interval=300,  # 5 minutes
            timeout=5,
            critical=False,
            description="Monitor disk space"
        ))
        
        # Basic connectivity check
        def connectivity_check():
            try:
                import socket
                socket.create_connection(("8.8.8.8", 53), timeout=3)
                return True, "Network connectivity OK"
            except Exception as e:
                return False, f"Network connectivity failed: {e}"
        
        self.add_check(HealthCheck(
            name="network_connectivity",
            check_type=CheckType.DEPENDENCY,
            check_function=connectivity_check,
            interval=120,  # 2 minutes
            timeout=5,
            critical=False,
            description="Check basic network connectivity"
        ))
    
    def add_check(self, health_check: HealthCheck):
        """Add a health check."""
        with self.lock:
            self.health_checks[health_check.name] = health_check
        
        logger.info(f"Added health check: {health_check.name} ({health_check.check_type.value})")
    
    def remove_check(self, check_name: str) -> bool:
        """Remove a health check."""
        with self.lock:
            if check_name in self.health_checks:
                del self.health_checks[check_name]
                logger.info(f"Removed health check: {check_name}")
                return True
        return False
    
    def enable_check(self, check_name: str) -> bool:
        """Enable a health check."""
        with self.lock:
            check = self.health_checks.get(check_name)
            if check:
                check.enabled = True
                return True
        return False
    
    def disable_check(self, check_name: str) -> bool:
        """Disable a health check."""
        with self.lock:
            check = self.health_checks.get(check_name)
            if check:
                check.enabled = False
                return True
        return False
    
    async def start(self):
        """Start health monitoring."""
        if self.running:
            return
        
        self.running = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Health monitoring started")
    
    async def stop(self):
        """Stop health monitoring."""
        self.running = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        # Save final state
        await self._save_health_state()
        
        logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                await self._run_health_checks()
                await self._update_overall_status()
                await asyncio.sleep(min(check.interval for check in self.health_checks.values()) if self.health_checks else 30)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)
    
    async def _run_health_checks(self):
        """Run all due health checks."""
        tasks = []
        
        with self.lock:
            for check in self.health_checks.values():
                if check.should_check():
                    tasks.append(self._run_single_check(check))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _run_single_check(self, health_check: HealthCheck):
        """Run a single health check."""
        try:
            success, message = await health_check.execute()
            
            logger.debug(f"Health check {health_check.name}: {'PASS' if success else 'FAIL'} - {message}")
            
        except Exception as e:
            logger.error(f"Error running health check {health_check.name}: {e}")
    
    async def _update_overall_status(self):
        """Update overall health status based on all checks."""
        with self.lock:
            if not self.health_checks:
                new_status = HealthStatus.UNKNOWN
                new_message = "No health checks configured"
            else:
                critical_failures = 0
                warnings = 0
                healthy_checks = 0
                total_enabled = 0
                
                for check in self.health_checks.values():
                    if not check.enabled:
                        continue
                    
                    total_enabled += 1
                    check_status = check.get_status()
                    
                    if check_status == HealthStatus.CRITICAL:
                        critical_failures += 1
                    elif check_status == HealthStatus.WARNING:
                        warnings += 1
                    elif check_status == HealthStatus.HEALTHY:
                        healthy_checks += 1
                
                # Determine overall status
                if critical_failures > 0:
                    new_status = HealthStatus.CRITICAL
                    new_message = f"{critical_failures} critical health check failures"
                elif warnings > 0:
                    new_status = HealthStatus.WARNING
                    new_message = f"{warnings} health check warnings, {healthy_checks} healthy"
                elif healthy_checks == total_enabled and total_enabled > 0:
                    new_status = HealthStatus.HEALTHY
                    new_message = f"All {total_enabled} health checks passing"
                else:
                    new_status = HealthStatus.UNKNOWN
                    new_message = "Health status unknown"
            
            # Update status if changed
            if new_status != self.overall_status or new_message != self.overall_message:
                old_status = self.overall_status
                self.overall_status = new_status
                self.overall_message = new_message
                
                # Record status change
                self.status_history.append({
                    'timestamp': time.time(),
                    'status': new_status.value,
                    'message': new_message,
                    'previous_status': old_status.value if old_status else None
                })
                
                # Notify listeners
                await self._notify_status_listeners(new_status, new_message)
                
                logger.info(f"Overall health status changed: {old_status.value if old_status else 'None'} -> {new_status.value}")
    
    async def _notify_status_listeners(self, status: HealthStatus, message: str):
        """Notify status change listeners."""
        for listener in self.status_listeners:
            try:
                await asyncio.get_event_loop().run_in_executor(None, listener, status, message)
            except Exception as e:
                logger.error(f"Error in status listener: {e}")
    
    def add_status_listener(self, listener: Callable[[HealthStatus, str], None]):
        """Add status change listener."""
        self.status_listeners.append(listener)
    
    def remove_status_listener(self, listener: Callable[[HealthStatus, str], None]):
        """Remove status change listener."""
        if listener in self.status_listeners:
            self.status_listeners.remove(listener)
    
    async def check_now(self, check_name: Optional[str] = None) -> Dict[str, Any]:
        """Run health checks immediately."""
        results = {}
        
        with self.lock:
            checks_to_run = [self.health_checks[check_name]] if check_name and check_name in self.health_checks else list(self.health_checks.values())
        
        for check in checks_to_run:
            try:
                success, message = await check.execute()
                results[check.name] = {
                    'success': success,
                    'message': message,
                    'status': check.get_status().value,
                    'timestamp': check.last_check_time
                }
            except Exception as e:
                results[check.name] = {
                    'success': False,
                    'message': f"Check failed: {e}",
                    'status': 'critical',
                    'timestamp': time.time()
                }
        
        await self._update_overall_status()
        return results
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        with self.lock:
            check_details = {}
            
            for name, check in self.health_checks.items():
                check_details[name] = check.to_dict()
            
            return {
                'service_name': self.service_name,
                'service_version': self.service_version,
                'overall_status': self.overall_status.value,
                'overall_message': self.overall_message,
                'timestamp': time.time(),
                'uptime': time.time() - (self.status_history[0]['timestamp'] if self.status_history else time.time()),
                'checks': check_details,
                'status_history': list(self.status_history)[-10:]  # Last 10 status changes
            }
    
    def get_readiness_status(self) -> Dict[str, Any]:
        """Get readiness status (for Kubernetes readiness probes)."""
        with self.lock:
            readiness_checks = [
                check for check in self.health_checks.values()
                if check.check_type in [CheckType.READINESS, CheckType.DEPENDENCY]
            ]
            
            all_ready = all(check.is_healthy() for check in readiness_checks if check.enabled)
            
            return {
                'ready': all_ready,
                'status': self.overall_status.value,
                'message': self.overall_message,
                'timestamp': time.time()
            }
    
    def get_liveness_status(self) -> Dict[str, Any]:
        """Get liveness status (for Kubernetes liveness probes)."""
        with self.lock:
            liveness_checks = [
                check for check in self.health_checks.values()
                if check.check_type in [CheckType.LIVENESS, CheckType.RESOURCE] and check.critical
            ]
            
            alive = all(check.is_healthy() for check in liveness_checks if check.enabled)
            
            return {
                'alive': alive,
                'status': 'healthy' if alive else 'critical',
                'message': self.overall_message if not alive else "Service is alive",
                'timestamp': time.time()
            }
    
    async def _save_health_state(self):
        """Save health state to storage."""
        try:
            if not self.storage_path:
                return
            
            state = self.get_health_status()
            
            Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump(state, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save health state: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get health monitor statistics."""
        with self.lock:
            check_types = defaultdict(int)
            check_statuses = defaultdict(int)
            
            for check in self.health_checks.values():
                check_types[check.check_type.value] += 1
                check_statuses[check.get_status().value] += 1
            
            return {
                'service_name': self.service_name,
                'service_version': self.service_version,
                'running': self.running,
                'overall_status': self.overall_status.value,
                'total_checks': len(self.health_checks),
                'enabled_checks': sum(1 for c in self.health_checks.values() if c.enabled),
                'check_types': dict(check_types),
                'check_statuses': dict(check_statuses),
                'status_changes': len(self.status_history),
                'listeners': len(self.status_listeners)
            }
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()