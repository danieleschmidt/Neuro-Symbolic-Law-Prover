"""
Intelligent load balancing and health checking for high availability.
"""

import asyncio
import time
import random
import logging
import hashlib
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import weakref
import psutil
import socket

logger = logging.getLogger(__name__)


class LoadBalancingAlgorithm(Enum):
    """Load balancing algorithms."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    IP_HASH = "ip_hash"
    LEAST_RESPONSE_TIME = "least_response_time"
    ADAPTIVE = "adaptive"


class HealthStatus(Enum):
    """Health status of backend servers."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"


@dataclass
class BackendServer:
    """Represents a backend server instance."""
    id: str
    host: str
    port: int
    weight: float = 1.0
    max_connections: int = 100
    
    # Runtime state
    current_connections: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    last_health_check: float = 0.0
    health_status: HealthStatus = HealthStatus.UNKNOWN
    health_failures: int = 0
    
    # Metrics tracking
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    error_rate_window: deque = field(default_factory=lambda: deque(maxlen=50))
    
    def __post_init__(self):
        self.url = f"http://{self.host}:{self.port}"
    
    def record_request(self, response_time: float, success: bool):
        """Record request metrics."""
        self.total_requests += 1
        
        if success:
            self.successful_requests += 1
            self.response_times.append(response_time)
            
            # Update average response time
            if self.successful_requests == 1:
                self.avg_response_time = response_time
            else:
                self.avg_response_time = (
                    (self.avg_response_time * (self.successful_requests - 1) + response_time) / 
                    self.successful_requests
                )
        else:
            self.failed_requests += 1
        
        # Track error rate in sliding window
        self.error_rate_window.append(0 if success else 1)
    
    def get_error_rate(self) -> float:
        """Calculate current error rate percentage."""
        if not self.error_rate_window:
            return 0.0
        return (sum(self.error_rate_window) / len(self.error_rate_window)) * 100
    
    def get_success_rate(self) -> float:
        """Calculate overall success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    def get_load_score(self) -> float:
        """Calculate load score for selection algorithms."""
        # Weighted combination of factors
        connection_factor = self.current_connections / max(self.max_connections, 1)
        response_time_factor = min(self.avg_response_time / 1000, 1.0)  # Normalize to 0-1
        error_rate_factor = self.get_error_rate() / 100
        
        # Higher score means higher load (worse for selection)
        return (connection_factor * 0.4 + response_time_factor * 0.4 + error_rate_factor * 0.2)
    
    def is_available(self) -> bool:
        """Check if server is available for requests."""
        return (
            self.health_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED] and
            self.current_connections < self.max_connections
        )
    
    def increment_connections(self):
        """Increment active connection count."""
        self.current_connections += 1
    
    def decrement_connections(self):
        """Decrement active connection count."""
        self.current_connections = max(0, self.current_connections - 1)


@dataclass
class HealthCheckConfig:
    """Configuration for health checks."""
    enabled: bool = True
    interval: float = 30.0  # seconds
    timeout: float = 5.0
    unhealthy_threshold: int = 3  # consecutive failures
    healthy_threshold: int = 2   # consecutive successes
    check_path: str = "/health"
    expected_status: int = 200
    expected_response: Optional[str] = None


class HealthChecker:
    """
    Advanced health checking system for backend servers.
    
    Supports multiple health check types:
    - HTTP health checks
    - TCP port checks
    - Custom health check functions
    """
    
    def __init__(self, config: HealthCheckConfig = None):
        """Initialize health checker."""
        self.config = config or HealthCheckConfig()
        self.servers: Dict[str, BackendServer] = {}
        self.health_check_tasks: Dict[str, asyncio.Task] = {}
        self.running = False
        
        # HTTP client for health checks
        try:
            import aiohttp
            self.http_client = None  # Will be initialized when needed
            self.aiohttp_available = True
        except ImportError:
            self.aiohttp_available = False
            logger.warning("aiohttp not available, using basic TCP health checks")
    
    def add_server(self, server: BackendServer):
        """Add server for health monitoring."""
        self.servers[server.id] = server
        
        if self.running and self.config.enabled:
            # Start health checking for this server
            task = asyncio.create_task(self._health_check_loop(server))
            self.health_check_tasks[server.id] = task
        
        logger.info(f"Added server for health checking: {server.id}")
    
    def remove_server(self, server_id: str):
        """Remove server from health monitoring."""
        if server_id in self.health_check_tasks:
            self.health_check_tasks[server_id].cancel()
            del self.health_check_tasks[server_id]
        
        if server_id in self.servers:
            del self.servers[server_id]
        
        logger.info(f"Removed server from health checking: {server_id}")
    
    async def start(self):
        """Start health checking for all servers."""
        if not self.config.enabled:
            logger.info("Health checking disabled")
            return
        
        self.running = True
        
        # Initialize HTTP client
        if self.aiohttp_available:
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.http_client = aiohttp.ClientSession(timeout=timeout)
        
        # Start health check tasks for all servers
        for server in self.servers.values():
            task = asyncio.create_task(self._health_check_loop(server))
            self.health_check_tasks[server.id] = task
        
        logger.info(f"Health checker started for {len(self.servers)} servers")
    
    async def stop(self):
        """Stop health checking."""
        self.running = False
        
        # Cancel all health check tasks
        for task in self.health_check_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self.health_check_tasks:
            await asyncio.gather(*self.health_check_tasks.values(), return_exceptions=True)
        
        # Close HTTP client
        if self.http_client:
            await self.http_client.close()
        
        self.health_check_tasks.clear()
        logger.info("Health checker stopped")
    
    async def _health_check_loop(self, server: BackendServer):
        """Health check loop for a single server."""
        consecutive_failures = 0
        consecutive_successes = 0
        
        while self.running:
            try:
                # Perform health check
                is_healthy = await self._perform_health_check(server)
                server.last_health_check = time.time()
                
                if is_healthy:
                    consecutive_successes += 1
                    consecutive_failures = 0
                    
                    # Mark as healthy if enough consecutive successes
                    if (server.health_status != HealthStatus.HEALTHY and 
                        consecutive_successes >= self.config.healthy_threshold):
                        
                        old_status = server.health_status
                        server.health_status = HealthStatus.HEALTHY
                        server.health_failures = 0
                        
                        logger.info(f"Server {server.id} recovered: {old_status.value} -> healthy")
                
                else:
                    consecutive_failures += 1
                    consecutive_successes = 0
                    server.health_failures += 1
                    
                    # Mark as unhealthy if enough consecutive failures
                    if consecutive_failures >= self.config.unhealthy_threshold:
                        if server.health_status != HealthStatus.UNHEALTHY:
                            old_status = server.health_status
                            server.health_status = HealthStatus.UNHEALTHY
                            
                            logger.warning(f"Server {server.id} unhealthy: {old_status.value} -> unhealthy")
                
                # Wait for next check
                await asyncio.sleep(self.config.interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error for server {server.id}: {e}")
                await asyncio.sleep(self.config.interval)
    
    async def _perform_health_check(self, server: BackendServer) -> bool:
        """Perform health check on a server."""
        try:
            if self.aiohttp_available and self.http_client:
                return await self._http_health_check(server)
            else:
                return await self._tcp_health_check(server)
        
        except Exception as e:
            logger.debug(f"Health check failed for {server.id}: {e}")
            return False
    
    async def _http_health_check(self, server: BackendServer) -> bool:
        """Perform HTTP health check."""
        try:
            url = f"http://{server.host}:{server.port}{self.config.check_path}"
            
            async with self.http_client.get(url) as response:
                # Check status code
                if response.status != self.config.expected_status:
                    return False
                
                # Check response content if specified
                if self.config.expected_response:
                    text = await response.text()
                    if self.config.expected_response not in text:
                        return False
                
                return True
        
        except Exception:
            return False
    
    async def _tcp_health_check(self, server: BackendServer) -> bool:
        """Perform TCP port health check."""
        try:
            # Create connection with timeout
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(server.host, server.port),
                timeout=self.config.timeout
            )
            
            # Close connection
            writer.close()
            await writer.wait_closed()
            
            return True
        
        except Exception:
            return False
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health status summary for all servers."""
        summary = {
            'total_servers': len(self.servers),
            'healthy_servers': 0,
            'degraded_servers': 0,
            'unhealthy_servers': 0,
            'servers': []
        }
        
        for server in self.servers.values():
            status = server.health_status
            
            if status == HealthStatus.HEALTHY:
                summary['healthy_servers'] += 1
            elif status == HealthStatus.DEGRADED:
                summary['degraded_servers'] += 1
            elif status == HealthStatus.UNHEALTHY:
                summary['unhealthy_servers'] += 1
            
            summary['servers'].append({
                'id': server.id,
                'host': server.host,
                'port': server.port,
                'status': status.value,
                'success_rate': server.get_success_rate(),
                'error_rate': server.get_error_rate(),
                'avg_response_time': server.avg_response_time,
                'current_connections': server.current_connections,
                'total_requests': server.total_requests,
                'health_failures': server.health_failures,
                'last_health_check': server.last_health_check
            })
        
        return summary


class LoadBalancer:
    """
    Advanced load balancer with multiple algorithms and adaptive routing.
    
    Features:
    - Multiple load balancing algorithms
    - Health checking and auto-failover
    - Adaptive routing based on performance metrics
    - Circuit breaker pattern for failing backends
    - Request routing and sticky sessions
    """
    
    def __init__(self,
                 algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.ADAPTIVE,
                 health_check_config: Optional[HealthCheckConfig] = None,
                 enable_circuit_breaker: bool = True,
                 circuit_breaker_threshold: float = 50.0,  # Error rate percentage
                 circuit_breaker_timeout: float = 60.0):   # Recovery timeout
        """
        Initialize load balancer.
        
        Args:
            algorithm: Load balancing algorithm to use
            health_check_config: Health check configuration
            enable_circuit_breaker: Enable circuit breaker pattern
            circuit_breaker_threshold: Error rate threshold for circuit breaker
            circuit_breaker_timeout: Circuit breaker recovery timeout
        """
        self.algorithm = algorithm
        self.servers: List[BackendServer] = []
        self.server_map: Dict[str, BackendServer] = {}
        
        # Algorithm state
        self.round_robin_index = 0
        self.last_selection_time = 0
        
        # Health checking
        self.health_checker = HealthChecker(health_check_config)
        
        # Circuit breaker
        self.enable_circuit_breaker = enable_circuit_breaker
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.algorithm_switches = 0
        
        # Adaptive algorithm learning
        self.performance_history: deque = deque(maxlen=1000)
        self.algorithm_performance: Dict[LoadBalancingAlgorithm, float] = defaultdict(float)
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info(f"LoadBalancer initialized with algorithm: {algorithm.value}")
    
    def add_server(self, host: str, port: int, weight: float = 1.0, 
                  max_connections: int = 100) -> str:
        """
        Add backend server to load balancer.
        
        Args:
            host: Server hostname/IP
            port: Server port
            weight: Server weight for weighted algorithms
            max_connections: Maximum concurrent connections
            
        Returns:
            Server ID for future reference
        """
        server_id = f"{host}:{port}"
        
        server = BackendServer(
            id=server_id,
            host=host,
            port=port,
            weight=weight,
            max_connections=max_connections
        )
        
        with self.lock:
            self.servers.append(server)
            self.server_map[server_id] = server
        
        # Add to health checker
        self.health_checker.add_server(server)
        
        logger.info(f"Added backend server: {server_id} (weight: {weight})")
        return server_id
    
    def remove_server(self, server_id: str) -> bool:
        """Remove backend server from load balancer."""
        with self.lock:
            if server_id not in self.server_map:
                return False
            
            server = self.server_map[server_id]
            self.servers.remove(server)
            del self.server_map[server_id]
        
        # Remove from health checker
        self.health_checker.remove_server(server_id)
        
        # Remove circuit breaker state
        self.circuit_breakers.pop(server_id, None)
        
        logger.info(f"Removed backend server: {server_id}")
        return True
    
    async def start(self):
        """Start load balancer and health checking."""
        await self.health_checker.start()
        logger.info("Load balancer started")
    
    async def stop(self):
        """Stop load balancer and cleanup."""
        await self.health_checker.stop()
        logger.info("Load balancer stopped")
    
    async def select_server(self, client_id: Optional[str] = None,
                          request_info: Optional[Dict[str, Any]] = None) -> Optional[BackendServer]:
        """
        Select optimal backend server for request.
        
        Args:
            client_id: Client identifier for sticky sessions
            request_info: Additional request information
            
        Returns:
            Selected backend server or None if no servers available
        """
        with self.lock:
            available_servers = [
                server for server in self.servers 
                if server.is_available() and not self._is_circuit_breaker_open(server.id)
            ]
            
            if not available_servers:
                logger.warning("No available backend servers")
                return None
            
            # Select server based on algorithm
            if self.algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
                server = self._round_robin_selection(available_servers)
            elif self.algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
                server = self._least_connections_selection(available_servers)
            elif self.algorithm == LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
                server = self._weighted_round_robin_selection(available_servers)
            elif self.algorithm == LoadBalancingAlgorithm.IP_HASH:
                server = self._ip_hash_selection(available_servers, client_id)
            elif self.algorithm == LoadBalancingAlgorithm.LEAST_RESPONSE_TIME:
                server = self._least_response_time_selection(available_servers)
            elif self.algorithm == LoadBalancingAlgorithm.ADAPTIVE:
                server = self._adaptive_selection(available_servers, request_info)
            else:
                server = self._round_robin_selection(available_servers)
            
            if server:
                server.increment_connections()
                logger.debug(f"Selected server: {server.id} ({self.algorithm.value})")
            
            return server
    
    async def handle_request_completion(self, server_id: str, response_time: float, 
                                      success: bool, error: Optional[Exception] = None):
        """
        Handle request completion and update server metrics.
        
        Args:
            server_id: Server that handled the request
            response_time: Request response time in seconds
            success: Whether request was successful
            error: Error if request failed
        """
        with self.lock:
            server = self.server_map.get(server_id)
            if server:
                server.decrement_connections()
                server.record_request(response_time, success)
                
                # Update circuit breaker
                self._update_circuit_breaker(server_id, success)
                
                # Update global statistics
                self.total_requests += 1
                if success:
                    self.successful_requests += 1
                else:
                    self.failed_requests += 1
                
                # Record performance for adaptive algorithm
                self.performance_history.append({
                    'server_id': server_id,
                    'algorithm': self.algorithm,
                    'response_time': response_time,
                    'success': success,
                    'timestamp': time.time()
                })
                
                # Update algorithm performance tracking
                if success:
                    self.algorithm_performance[self.algorithm] += 1.0 / response_time
                else:
                    self.algorithm_performance[self.algorithm] -= 1.0
    
    def _round_robin_selection(self, servers: List[BackendServer]) -> BackendServer:
        """Round-robin server selection."""
        if not servers:
            return None
        
        self.round_robin_index = (self.round_robin_index + 1) % len(servers)
        return servers[self.round_robin_index]
    
    def _least_connections_selection(self, servers: List[BackendServer]) -> BackendServer:
        """Select server with least active connections."""
        return min(servers, key=lambda s: s.current_connections)
    
    def _weighted_round_robin_selection(self, servers: List[BackendServer]) -> BackendServer:
        """Weighted round-robin selection based on server weights."""
        if not servers:
            return None
        
        # Calculate cumulative weights
        cumulative_weights = []
        total_weight = 0
        
        for server in servers:
            total_weight += server.weight
            cumulative_weights.append(total_weight)
        
        # Select based on weight
        if total_weight == 0:
            return servers[0]
        
        random_weight = random.uniform(0, total_weight)
        
        for i, cumulative_weight in enumerate(cumulative_weights):
            if random_weight <= cumulative_weight:
                return servers[i]
        
        return servers[-1]
    
    def _ip_hash_selection(self, servers: List[BackendServer], client_id: Optional[str]) -> BackendServer:
        """Consistent hash-based selection for sticky sessions."""
        if not servers or not client_id:
            return self._round_robin_selection(servers)
        
        # Use consistent hashing
        hash_value = int(hashlib.md5(client_id.encode()).hexdigest(), 16)
        index = hash_value % len(servers)
        
        return servers[index]
    
    def _least_response_time_selection(self, servers: List[BackendServer]) -> BackendServer:
        """Select server with lowest average response time."""
        return min(servers, key=lambda s: s.avg_response_time if s.avg_response_time > 0 else 0.001)
    
    def _adaptive_selection(self, servers: List[BackendServer], 
                          request_info: Optional[Dict[str, Any]]) -> BackendServer:
        """
        Adaptive server selection based on multiple factors.
        
        Combines load score, health status, and historical performance.
        """
        if len(servers) == 1:
            return servers[0]
        
        # Score each server
        server_scores = []
        
        for server in servers:
            score = 0.0
            
            # Load factor (lower is better)
            load_score = server.get_load_score()
            score -= load_score * 40  # Weight: 40%
            
            # Health factor
            if server.health_status == HealthStatus.HEALTHY:
                score += 30  # Weight: 30%
            elif server.health_status == HealthStatus.DEGRADED:
                score += 15
            
            # Performance factor
            if server.avg_response_time > 0:
                # Faster response times get higher scores
                performance_score = max(0, 10 - server.avg_response_time)
                score += performance_score * 20  # Weight: 20%
            
            # Success rate factor
            success_rate = server.get_success_rate()
            score += (success_rate / 100) * 10  # Weight: 10%
            
            server_scores.append((server, score))
        
        # Select server with highest score
        best_server, _ = max(server_scores, key=lambda x: x[1])
        return best_server
    
    def _is_circuit_breaker_open(self, server_id: str) -> bool:
        """Check if circuit breaker is open for server."""
        if not self.enable_circuit_breaker:
            return False
        
        cb_state = self.circuit_breakers.get(server_id)
        if not cb_state:
            return False
        
        if cb_state['state'] == 'open':
            # Check if timeout has passed for recovery attempt
            if time.time() - cb_state['opened_at'] > self.circuit_breaker_timeout:
                cb_state['state'] = 'half_open'
                logger.info(f"Circuit breaker half-open for server {server_id}")
            return cb_state['state'] == 'open'
        
        return False
    
    def _update_circuit_breaker(self, server_id: str, success: bool):
        """Update circuit breaker state based on request result."""
        if not self.enable_circuit_breaker:
            return
        
        server = self.server_map.get(server_id)
        if not server:
            return
        
        if server_id not in self.circuit_breakers:
            self.circuit_breakers[server_id] = {
                'state': 'closed',
                'failure_count': 0,
                'opened_at': 0
            }
        
        cb_state = self.circuit_breakers[server_id]
        
        if cb_state['state'] == 'closed':
            if success:
                cb_state['failure_count'] = max(0, cb_state['failure_count'] - 1)
            else:
                cb_state['failure_count'] += 1
                
                # Check if threshold exceeded
                error_rate = server.get_error_rate()
                if error_rate >= self.circuit_breaker_threshold:
                    cb_state['state'] = 'open'
                    cb_state['opened_at'] = time.time()
                    logger.warning(f"Circuit breaker opened for server {server_id} (error rate: {error_rate:.1f}%)")
        
        elif cb_state['state'] == 'half_open':
            if success:
                cb_state['state'] = 'closed'
                cb_state['failure_count'] = 0
                logger.info(f"Circuit breaker closed for server {server_id}")
            else:
                cb_state['state'] = 'open'
                cb_state['opened_at'] = time.time()
                logger.warning(f"Circuit breaker re-opened for server {server_id}")
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get comprehensive server statistics."""
        with self.lock:
            stats = {
                'algorithm': self.algorithm.value,
                'total_servers': len(self.servers),
                'available_servers': sum(1 for s in self.servers if s.is_available()),
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests,
                'failed_requests': self.failed_requests,
                'success_rate': (self.successful_requests / max(1, self.total_requests)) * 100,
                'servers': []
            }
            
            for server in self.servers:
                cb_state = self.circuit_breakers.get(server.id, {})
                
                server_stats = {
                    'id': server.id,
                    'host': server.host,
                    'port': server.port,
                    'weight': server.weight,
                    'health_status': server.health_status.value,
                    'current_connections': server.current_connections,
                    'max_connections': server.max_connections,
                    'total_requests': server.total_requests,
                    'successful_requests': server.successful_requests,
                    'failed_requests': server.failed_requests,
                    'success_rate': server.get_success_rate(),
                    'error_rate': server.get_error_rate(),
                    'avg_response_time': server.avg_response_time,
                    'load_score': server.get_load_score(),
                    'circuit_breaker_state': cb_state.get('state', 'closed'),
                    'is_available': server.is_available()
                }
                
                stats['servers'].append(server_stats)
            
            return stats
    
    def set_algorithm(self, algorithm: LoadBalancingAlgorithm):
        """Change load balancing algorithm."""
        old_algorithm = self.algorithm
        self.algorithm = algorithm
        self.algorithm_switches += 1
        
        logger.info(f"Changed load balancing algorithm: {old_algorithm.value} -> {algorithm.value}")
    
    def optimize_algorithm(self):
        """Automatically optimize algorithm based on performance history."""
        if len(self.performance_history) < 100:  # Need sufficient data
            return
        
        # Analyze performance of each algorithm
        algorithm_scores = defaultdict(list)
        
        for record in list(self.performance_history)[-500:]:  # Last 500 requests
            if record['success']:
                score = 1.0 / max(record['response_time'], 0.001)  # Higher is better
                algorithm_scores[record['algorithm']].append(score)
        
        # Find best performing algorithm
        best_algorithm = None
        best_score = 0
        
        for algorithm, scores in algorithm_scores.items():
            if len(scores) >= 20:  # Need sufficient samples
                avg_score = sum(scores) / len(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_algorithm = algorithm
        
        # Switch to best algorithm if significantly better
        if (best_algorithm and best_algorithm != self.algorithm and 
            best_score > self.algorithm_performance[self.algorithm] * 1.2):
            
            self.set_algorithm(best_algorithm)
            logger.info(f"Auto-optimized to {best_algorithm.value} (score: {best_score:.3f})")
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()