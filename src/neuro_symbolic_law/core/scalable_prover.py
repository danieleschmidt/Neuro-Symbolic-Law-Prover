"""
Generation 3: Scalable Legal Prover with advanced performance optimization.
Implements adaptive caching, concurrent processing, and auto-scaling capabilities.
"""

import asyncio
import concurrent.futures
import logging
import time
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
from functools import lru_cache, wraps
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import weakref
import gc

from .enhanced_prover import EnhancedLegalProver
from .exceptions import ResourceError, handle_exception_gracefully, log_security_event
from .monitoring import get_metrics_collector, get_health_checker, PerformanceTimer
from .compliance_result import ComplianceResult, ComplianceReport, ComplianceStatus
from ..parsing.contract_parser import ParsedContract
from ..regulations.base_regulation import BaseRegulation
from ..performance.auto_scaler import (
    PredictiveScaler, LoadBalancer, CircuitBreaker, PerformanceOptimizer,
    ResourceMetrics, ScalingPolicy, ScalingDecision
)

logger = logging.getLogger(__name__)


class AdaptiveCache:
    """Adaptive cache that learns access patterns and optimizes performance."""
    
    def __init__(self, initial_size: int = 1000, max_size: int = 50000):
        """Initialize adaptive cache."""
        self.initial_size = initial_size
        self.max_size = max_size
        self.current_size = initial_size
        
        self._cache: Dict[str, Any] = {}
        self._access_times: Dict[str, float] = {}
        self._access_counts: Dict[str, int] = defaultdict(int)
        self._access_patterns: deque = deque(maxlen=10000)
        
        # Cache adaptation parameters
        self._last_adaptation = time.time()
        self._adaptation_interval = 300  # 5 minutes
        self._hit_rate_threshold = 0.8
        self._recent_hit_rates = deque(maxlen=100)
        
        self._lock = threading.RLock()
        
        logger.info(f"AdaptiveCache initialized: size={initial_size}, max={max_size}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache and record access pattern."""
        with self._lock:
            current_time = time.time()
            
            if key in self._cache:
                # Update access patterns
                self._access_times[key] = current_time
                self._access_counts[key] += 1
                self._access_patterns.append(('hit', key, current_time))
                
                # Trigger adaptation check
                self._maybe_adapt_cache()
                
                return self._cache[key]
            else:
                self._access_patterns.append(('miss', key, current_time))
                self._maybe_adapt_cache()
                return None
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache with intelligent eviction."""
        with self._lock:
            current_time = time.time()
            
            # Check if we need to evict
            if len(self._cache) >= self.current_size and key not in self._cache:
                self._intelligent_evict()
            
            self._cache[key] = value
            self._access_times[key] = current_time
            self._access_counts[key] = self._access_counts.get(key, 0) + 1
    
    def _intelligent_evict(self) -> None:
        """Evict items intelligently based on access patterns."""
        if not self._cache:
            return
        
        current_time = time.time()
        
        # Calculate scores for each item (lower = more likely to evict)
        scores = {}
        for key in self._cache.keys():
            recency_score = current_time - self._access_times.get(key, 0)
            frequency_score = 1.0 / max(1, self._access_counts.get(key, 1))
            
            # Combined score (higher recency + higher frequency = lower eviction score)
            scores[key] = recency_score * frequency_score
        
        # Evict items with highest scores (least valuable)
        items_to_evict = max(1, len(self._cache) // 10)  # Evict 10% or at least 1
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        for key, _ in sorted_items[:items_to_evict]:
            del self._cache[key]
            self._access_times.pop(key, None)
            # Keep access counts for future reference
    
    def _maybe_adapt_cache(self) -> None:
        """Adapt cache size based on performance metrics."""
        current_time = time.time()
        
        if current_time - self._last_adaptation < self._adaptation_interval:
            return
        
        # Calculate recent hit rate
        recent_accesses = [access for access in self._access_patterns 
                          if current_time - access[2] < self._adaptation_interval]
        
        if not recent_accesses:
            return
        
        hits = sum(1 for access in recent_accesses if access[0] == 'hit')
        hit_rate = hits / len(recent_accesses)
        self._recent_hit_rates.append(hit_rate)
        
        # Adapt cache size based on hit rate trends
        if len(self._recent_hit_rates) >= 5:
            avg_hit_rate = sum(self._recent_hit_rates) / len(self._recent_hit_rates)
            
            if avg_hit_rate < self._hit_rate_threshold and self.current_size < self.max_size:
                # Low hit rate - increase cache size
                old_size = self.current_size
                self.current_size = min(self.max_size, int(self.current_size * 1.2))
                logger.info(f"Cache size adapted: {old_size} -> {self.current_size} (hit_rate: {avg_hit_rate:.2f})")
                
            elif avg_hit_rate > 0.95 and self.current_size > self.initial_size:
                # Very high hit rate - can reduce cache size to free memory
                old_size = self.current_size
                self.current_size = max(self.initial_size, int(self.current_size * 0.9))
                logger.info(f"Cache size reduced: {old_size} -> {self.current_size} (hit_rate: {avg_hit_rate:.2f})")
        
        self._last_adaptation = current_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_accesses = len(self._access_patterns)
            hits = sum(1 for access in self._access_patterns if access[0] == 'hit')
            hit_rate = hits / total_accesses if total_accesses > 0 else 0
            
            return {
                'size': len(self._cache),
                'current_limit': self.current_size,
                'max_limit': self.max_size,
                'utilization': len(self._cache) / self.current_size if self.current_size > 0 else 0,
                'hit_rate': hit_rate,
                'total_accesses': total_accesses,
                'adaptation_active': len(self._recent_hit_rates) >= 5
            }


class ResourceManager:
    """Manages system resources and implements auto-scaling."""
    
    def __init__(self, max_workers: int = None, memory_threshold: float = 0.8):
        """Initialize resource manager."""
        # Use the psutil from monitoring module (with fallback)
        from .monitoring import psutil, PSUTIL_AVAILABLE
        
        self.cpu_count = psutil.cpu_count()
        self.max_workers = max_workers or min(32, (self.cpu_count or 1) + 4)
        self.memory_threshold = memory_threshold
        
        self._current_workers = 2  # Start conservative
        self._worker_pools: Dict[str, concurrent.futures.ThreadPoolExecutor] = {}
        self._load_history = deque(maxlen=100)
        self._last_scaling = time.time()
        
        self.metrics_collector = get_metrics_collector()
        
        logger.info(f"ResourceManager initialized: max_workers={self.max_workers}")
    
    def get_executor(self, pool_name: str = "default") -> concurrent.futures.ThreadPoolExecutor:
        """Get or create thread pool executor."""
        if pool_name not in self._worker_pools:
            workers = min(self._current_workers, self.max_workers)
            self._worker_pools[pool_name] = concurrent.futures.ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix=f"nsl-{pool_name}"
            )
            logger.info(f"Created thread pool '{pool_name}' with {workers} workers")
        
        return self._worker_pools[pool_name]
    
    def check_resources(self) -> Dict[str, Any]:
        """Check system resource availability."""
        try:
            from .monitoring import psutil, PSUTIL_AVAILABLE
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Record load metrics
            load_info = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'timestamp': time.time()
            }
            self._load_history.append(load_info)
            
            # Auto-scale workers based on load
            self._auto_scale_workers(cpu_percent, memory.percent)
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'current_workers': self._current_workers,
                'max_workers': self.max_workers,
                'worker_pools': len(self._worker_pools),
                'resource_status': self._get_resource_status(cpu_percent, memory.percent)
            }
            
        except (ImportError, Exception) as e:
            logger.warning(f"Resource checking not available: {e} - using fallback")
            return {
                'cpu_percent': 50,  # Assume moderate load
                'memory_percent': 50,
                'resource_status': 'unknown',
                'current_workers': self._current_workers
            }
    
    def _auto_scale_workers(self, cpu_percent: float, memory_percent: float) -> None:
        """Auto-scale worker threads based on system load."""
        current_time = time.time()
        
        # Only scale every 30 seconds
        if current_time - self._last_scaling < 30:
            return
        
        old_workers = self._current_workers
        
        # Scale up conditions
        if cpu_percent > 70 and memory_percent < self.memory_threshold * 100:
            if self._current_workers < self.max_workers:
                self._current_workers = min(self.max_workers, self._current_workers + 1)
                logger.info(f"Scaling up workers: {old_workers} -> {self._current_workers}")
                
        # Scale down conditions
        elif cpu_percent < 30 and self._current_workers > 2:
            self._current_workers = max(2, self._current_workers - 1)
            logger.info(f"Scaling down workers: {old_workers} -> {self._current_workers}")
        
        if old_workers != self._current_workers:
            self._last_scaling = current_time
            # Recreate pools with new worker count
            self._recreate_worker_pools()
            
            self.metrics_collector.record_gauge("resource_manager.workers", self._current_workers)
    
    def _recreate_worker_pools(self) -> None:
        """Recreate worker pools with updated worker count."""
        # Shutdown old pools
        for pool in self._worker_pools.values():
            pool.shutdown(wait=False)
        
        self._worker_pools.clear()
        # New pools will be created on demand with updated worker count
    
    def _get_resource_status(self, cpu_percent: float, memory_percent: float) -> str:
        """Get overall resource status."""
        if cpu_percent > 90 or memory_percent > 90:
            return 'critical'
        elif cpu_percent > 70 or memory_percent > 70:
            return 'high'
        elif cpu_percent > 40 or memory_percent > 40:
            return 'moderate'
        else:
            return 'low'
    
    def cleanup(self) -> None:
        """Cleanup all resources."""
        for pool_name, pool in self._worker_pools.items():
            logger.info(f"Shutting down thread pool: {pool_name}")
            pool.shutdown(wait=True)
        
        self._worker_pools.clear()


class ScalableLegalProver(EnhancedLegalProver):
    """
    Generation 3: Scalable Legal Prover with advanced performance features.
    
    Features:
    - Adaptive caching based on access patterns
    - Concurrent processing with auto-scaling
    - Performance optimization and resource pooling
    - Load balancing and intelligent request routing
    - Self-healing with circuit breakers
    """
    
    def __init__(
        self,
        initial_cache_size: int = 1000,
        max_cache_size: int = 50000,
        max_workers: int = None,
        enable_adaptive_caching: bool = True,
        enable_concurrent_processing: bool = True,
        memory_threshold: float = 0.8,
        **kwargs
    ):
        """Initialize scalable legal prover."""
        super().__init__(**kwargs)
        
        # Advanced caching
        self.enable_adaptive_caching = enable_adaptive_caching
        if enable_adaptive_caching:
            self.adaptive_cache = AdaptiveCache(initial_cache_size, max_cache_size)
        else:
            self.adaptive_cache = None
        
        # Resource management
        self.enable_concurrent_processing = enable_concurrent_processing
        self.resource_manager = ResourceManager(max_workers, memory_threshold)
        
        # Performance optimization
        self._request_queue = asyncio.Queue(maxsize=1000)
        self._active_requests: Dict[str, float] = {}  # request_id -> start_time
        self._performance_history = deque(maxlen=1000)
        
        # Auto-scaling components
        scaling_policy = ScalingPolicy(
            min_workers=2,
            max_workers=max_workers or 20,
            target_cpu_threshold=0.7,
            target_memory_threshold=memory_threshold
        )
        self.auto_scaler = PredictiveScaler(scaling_policy)
        # Use the new LoadBalancer from auto_scaler instead of the simple one
        from ..performance.auto_scaler import LoadBalancer as NewLoadBalancer
        self.load_balancer = NewLoadBalancer(scaling_policy.min_workers)
        self.circuit_breaker = CircuitBreaker()
        self.performance_optimizer = PerformanceOptimizer()
        
        self._optimization_enabled = True
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = threading.Event()
        
        logger.info("ScalableLegalProver initialized with Generation 3 features")
        self.metrics_collector.record_counter("scalable_legal_prover.initialized")
    
    @handle_exception_gracefully
    async def verify_compliance_concurrent(
        self,
        contracts: List[ParsedContract],
        regulation: BaseRegulation,
        focus_areas: Optional[List[str]] = None,
        max_concurrent: int = None
    ) -> Dict[str, Dict[str, ComplianceResult]]:
        """
        Verify compliance for multiple contracts concurrently.
        
        Args:
            contracts: List of contracts to verify
            regulation: Regulation to check against
            focus_areas: Specific areas to focus on
            max_concurrent: Maximum concurrent verifications
            
        Returns:
            Dictionary mapping contract IDs to compliance results
        """
        if not self.enable_concurrent_processing or len(contracts) <= 1:
            # Fall back to sequential processing
            results = {}
            for contract in contracts:
                results[contract.id] = self.verify_compliance(contract, regulation, focus_areas)
            return results
        
        # Determine optimal concurrency level
        if max_concurrent is None:
            resource_info = self.resource_manager.check_resources()
            max_concurrent = min(len(contracts), resource_info['current_workers'])
        
        start_time = time.time()
        logger.info(f"Starting concurrent verification of {len(contracts)} contracts (max_concurrent={max_concurrent})")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def verify_single_contract(contract: ParsedContract) -> Tuple[str, Dict[str, ComplianceResult]]:
            async with semaphore:
                # Run verification in thread pool
                executor = self.resource_manager.get_executor("verification")
                loop = asyncio.get_event_loop()
                
                result = await loop.run_in_executor(
                    executor, 
                    self.verify_compliance, 
                    contract, 
                    regulation, 
                    focus_areas
                )
                
                return contract.id, result
        
        # Create and run concurrent tasks
        tasks = [verify_single_contract(contract) for contract in contracts]
        
        try:
            completed = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            results = {}
            errors = 0
            
            for result in completed:
                if isinstance(result, Exception):
                    logger.error(f"Concurrent verification error: {result}")
                    errors += 1
                else:
                    contract_id, compliance_results = result
                    results[contract_id] = compliance_results
            
            total_time = time.time() - start_time
            
            # Record performance metrics
            self.metrics_collector.record_timer("concurrent_verification.total_time", total_time * 1000)
            self.metrics_collector.record_counter("concurrent_verification.completed", len(results))
            self.metrics_collector.record_counter("concurrent_verification.errors", errors)
            
            logger.info(f"Concurrent verification completed: {len(results)} success, {errors} errors in {total_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Concurrent verification failed: {e}")
            self.metrics_collector.record_counter("concurrent_verification.failure")
            
            # Fallback to sequential processing
            logger.info("Falling back to sequential verification")
            results = {}
            for contract in contracts:
                try:
                    results[contract.id] = self.verify_compliance(contract, regulation, focus_areas)
                except Exception as contract_error:
                    logger.error(f"Sequential fallback failed for {contract.id}: {contract_error}")
            
            return results
    
    def verify_compliance(
        self,
        contract: ParsedContract,
        regulation: BaseRegulation,
        focus_areas: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, ComplianceResult]:
        """
        Enhanced verify_compliance with adaptive caching and optimization.
        """
        # Check adaptive cache first
        cache_key = self._get_adaptive_cache_key(contract, regulation, focus_areas)
        
        if self.enable_adaptive_caching and self.adaptive_cache:
            cached_result = self.adaptive_cache.get(cache_key)
            if cached_result is not None:
                self.metrics_collector.record_counter("adaptive_cache.hit")
                return cached_result
            else:
                self.metrics_collector.record_counter("adaptive_cache.miss")
        
        # Perform verification with performance tracking
        start_time = time.time()
        
        try:
            # Use parent's enhanced verification
            results = super().verify_compliance(contract, regulation, focus_areas, **kwargs)
            
            verification_time = time.time() - start_time
            
            # Store in adaptive cache
            if self.enable_adaptive_caching and self.adaptive_cache:
                self.adaptive_cache.set(cache_key, results)
            
            # Record performance metrics
            self._performance_history.append({
                'verification_time': verification_time,
                'contract_size': len(contract.clauses),
                'requirements_checked': len(results),
                'timestamp': time.time()
            })
            
            self.metrics_collector.record_timer("scalable_verification.time", verification_time * 1000)
            
            # Trigger optimization if needed
            if self._optimization_enabled:
                self._maybe_optimize_performance()
            
            return results
            
        except Exception as e:
            verification_time = time.time() - start_time
            logger.error(f"Scalable verification failed in {verification_time:.2f}s: {e}")
            raise
    
    def _get_adaptive_cache_key(
        self, 
        contract: ParsedContract, 
        regulation: BaseRegulation, 
        focus_areas: Optional[List[str]]
    ) -> str:
        """Generate cache key for adaptive cache."""
        # Include contract hash for better cache efficiency
        contract_hash = hash(contract.text) if hasattr(contract, 'text') else hash(str(contract.clauses))
        focus_str = ','.join(sorted(focus_areas)) if focus_areas else 'all'
        
        return f"scalable:v1:{contract.id}:{contract_hash}:{regulation.name}:{focus_str}"
    
    def _maybe_optimize_performance(self) -> None:
        """Optimize performance based on historical data."""
        if len(self._performance_history) < 10:
            return
        
        # Analyze recent performance
        recent_performance = list(self._performance_history)[-50:]  # Last 50 requests
        avg_time = sum(p['verification_time'] for p in recent_performance) / len(recent_performance)
        
        # Optimize cache if performance is degrading
        if avg_time > 5.0:  # Slow performance threshold
            if self.adaptive_cache:
                cache_stats = self.adaptive_cache.get_stats()
                if cache_stats['hit_rate'] < 0.5:
                    logger.info("Optimizing cache due to low hit rate")
                    # Cache optimization happens automatically in AdaptiveCache
        
        # Adjust resource allocation
        resource_info = self.resource_manager.check_resources()
        if resource_info['resource_status'] in ['high', 'critical']:
            logger.warning(f"High resource usage detected: {resource_info['resource_status']}")
            # Resource scaling happens automatically in ResourceManager
    
    async def batch_verify_compliance(
        self,
        batch_requests: List[Dict[str, Any]],
        batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Process multiple verification requests in optimized batches.
        
        Args:
            batch_requests: List of verification requests
            batch_size: Size of each processing batch
            
        Returns:
            List of verification results
        """
        start_time = time.time()
        logger.info(f"Starting batch verification: {len(batch_requests)} requests, batch_size={batch_size}")
        
        # Split requests into batches
        batches = [batch_requests[i:i+batch_size] for i in range(0, len(batch_requests), batch_size)]
        
        all_results = []
        
        for i, batch in enumerate(batches):
            batch_start = time.time()
            logger.debug(f"Processing batch {i+1}/{len(batches)} ({len(batch)} requests)")
            
            # Process batch concurrently
            batch_tasks = []
            for request in batch:
                task = self._process_single_request(request)
                batch_tasks.append(task)
            
            # Wait for batch completion
            try:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Handle results and errors
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Batch request failed: {result}")
                        all_results.append({'error': str(result)})
                    else:
                        all_results.append(result)
                        
            except Exception as e:
                logger.error(f"Batch {i+1} failed: {e}")
                # Add error results for the entire batch
                for _ in batch:
                    all_results.append({'error': f'Batch processing failed: {str(e)}'})
            
            batch_time = time.time() - batch_start
            logger.debug(f"Batch {i+1} completed in {batch_time:.2f}s")
        
        total_time = time.time() - start_time
        
        # Record metrics
        self.metrics_collector.record_timer("batch_verification.total_time", total_time * 1000)
        self.metrics_collector.record_counter("batch_verification.requests", len(batch_requests))
        self.metrics_collector.record_counter("batch_verification.batches", len(batches))
        
        logger.info(f"Batch verification completed: {len(all_results)} results in {total_time:.2f}s")
        
        return all_results
    
    async def _process_single_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single verification request."""
        try:
            # Extract request parameters
            contract = request['contract']
            regulation = request['regulation']
            focus_areas = request.get('focus_areas')
            
            # Perform verification
            results = self.verify_compliance(contract, regulation, focus_areas)
            
            return {
                'contract_id': contract.id,
                'regulation': regulation.name,
                'results': results,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Request processing failed: {e}")
            return {
                'contract_id': request.get('contract', {}).get('id', 'unknown'),
                'status': 'error',
                'error': str(e)
            }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        base_metrics = super().get_cache_stats()
        
        # Add scalable-specific metrics
        scalable_metrics = {
            'adaptive_cache': self.adaptive_cache.get_stats() if self.adaptive_cache else None,
            'resource_manager': self.resource_manager.check_resources(),
            'performance_history_size': len(self._performance_history),
            'active_requests': len(self._active_requests),
            'optimization_enabled': self._optimization_enabled
        }
        
        # Performance statistics
        if self._performance_history:
            recent_perf = list(self._performance_history)[-100:]  # Last 100 requests
            scalable_metrics['performance_stats'] = {
                'avg_verification_time': sum(p['verification_time'] for p in recent_perf) / len(recent_perf),
                'avg_contract_size': sum(p['contract_size'] for p in recent_perf) / len(recent_perf),
                'avg_requirements_checked': sum(p['requirements_checked'] for p in recent_perf) / len(recent_perf),
                'requests_per_second': len(recent_perf) / max(1, time.time() - recent_perf[0]['timestamp'])
            }
        
        return {**base_metrics, **scalable_metrics}
    
    def optimize_system(self) -> Dict[str, Any]:
        """Manually trigger system optimization."""
        logger.info("Manual system optimization triggered")
        
        optimization_results = {}
        
        # Force cache adaptation
        if self.adaptive_cache:
            old_stats = self.adaptive_cache.get_stats()
            self.adaptive_cache._maybe_adapt_cache()
            new_stats = self.adaptive_cache.get_stats()
            
            optimization_results['cache_optimization'] = {
                'before': old_stats,
                'after': new_stats,
                'size_changed': old_stats['current_limit'] != new_stats['current_limit']
            }
        
        # Force resource scaling check
        resource_info = self.resource_manager.check_resources()
        optimization_results['resource_check'] = resource_info
        
        # Garbage collection
        collected = gc.collect()
        optimization_results['garbage_collection'] = {
            'objects_collected': collected,
            'memory_freed': True
        }
        
        # Performance analysis
        if self._performance_history:
            self._maybe_optimize_performance()
            optimization_results['performance_analysis'] = 'completed'
        
        self.metrics_collector.record_counter("manual_optimization.triggered")
        
        return optimization_results
    
    def get_resource_metrics(self) -> ResourceMetrics:
        """Get current resource utilization metrics."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            
            return ResourceMetrics(
                cpu_usage=process.cpu_percent() / 100.0,
                memory_usage=process.memory_percent() / 100.0,
                request_rate=len(self._active_requests) * 60.0,  # Requests per minute estimate
                response_time=sum(time.time() - start for start in self._active_requests.values()) / max(1, len(self._active_requests)),
                queue_length=self._request_queue.qsize() if hasattr(self._request_queue, 'qsize') else 0,
                active_workers=self.resource_manager.current_workers if hasattr(self.resource_manager, 'current_workers') else 4
            )
        except (ImportError, Exception):
            # Fallback when psutil is not available or fails
            return ResourceMetrics(
                cpu_usage=0.5,  # Assume moderate usage
                memory_usage=0.6,
                request_rate=len(self._active_requests) * 60.0,
                response_time=1000.0,  # 1 second default
                queue_length=0,
                active_workers=4
            )
    
    def adaptive_scale_resources(self) -> Dict[str, Any]:
        """Adaptively scale resources based on current demand."""
        metrics = self.get_resource_metrics()
        
        # Record metrics for learning
        self.auto_scaler.record_metrics(metrics)
        
        # Make scaling decision
        decision, new_workers = self.auto_scaler.make_scaling_decision(metrics)
        
        scaling_result = {
            'metrics': {
                'cpu_usage': metrics.cpu_usage,
                'memory_usage': metrics.memory_usage,
                'request_rate': metrics.request_rate,
                'response_time': metrics.response_time,
                'queue_length': metrics.queue_length
            },
            'decision': decision.value,
            'previous_workers': self.load_balancer.total_active_workers if hasattr(self.load_balancer, 'total_active_workers') else 4,
            'new_workers': new_workers,
            'timestamp': time.time()
        }
        
        # Execute scaling decision
        if decision == ScalingDecision.SCALE_UP:
            # Add workers to load balancer
            current_count = len(self.load_balancer.workers) if hasattr(self.load_balancer, 'workers') else 4
            for i in range(new_workers - current_count):
                worker_id = f"scaled_worker_{int(time.time())}_{i}"
                self.load_balancer.add_worker(worker_id)
            
            logger.info(f"Scaled up to {new_workers} workers")
            self.metrics_collector.record_counter("auto_scaling.scale_up")
            
        elif decision == ScalingDecision.SCALE_DOWN:
            # Remove workers from load balancer
            if hasattr(self.load_balancer, 'workers'):
                workers_to_remove = list(self.load_balancer.workers.keys())[new_workers:]
                for worker_id in workers_to_remove:
                    self.load_balancer.remove_worker(worker_id)
            
            logger.info(f"Scaled down to {new_workers} workers")
            self.metrics_collector.record_counter("auto_scaling.scale_down")
        
        # Apply performance optimizations if needed
        optimizations = self.performance_optimizer.suggest_optimizations(metrics)
        if optimizations:
            for optimization in optimizations[:3]:  # Apply top 3 suggestions
                success = self.performance_optimizer.apply_optimization(optimization['implementation'])
                scaling_result['applied_optimizations'] = scaling_result.get('applied_optimizations', [])
                scaling_result['applied_optimizations'].append({
                    'type': optimization['type'],
                    'success': success
                })
        
        return scaling_result
    
    async def intelligent_request_routing(self, request_metadata: Dict[str, Any]) -> str:
        """Route request to optimal worker using intelligent load balancing."""
        # Use circuit breaker for fault tolerance
        try:
            worker_id = self.circuit_breaker.call(
                self.load_balancer.select_worker,
                request_metadata
            )
            
            if worker_id is None:
                # No healthy workers available - trigger scaling
                scaling_result = self.adaptive_scale_resources()
                logger.warning(f"No healthy workers available, triggered scaling: {scaling_result}")
                
                # Retry after scaling
                worker_id = self.load_balancer.select_worker(request_metadata)
            
            return worker_id or "default_worker"
            
        except Exception as e:
            logger.error(f"Request routing failed: {e}")
            
            # Fallback routing
            if hasattr(self.load_balancer, 'workers') and self.load_balancer.workers:
                return list(self.load_balancer.workers.keys())[0]
            else:
                return "fallback_worker"
    
    def get_scaling_insights(self) -> Dict[str, Any]:
        """Get comprehensive insights about system scaling and performance."""
        insights = {
            'auto_scaling': self.auto_scaler.get_scaling_insights(),
            'load_distribution': self.load_balancer.get_load_distribution(),
            'circuit_breaker_status': {
                'state': self.circuit_breaker.state,
                'failure_count': self.circuit_breaker.failure_count,
                'last_failure': self.circuit_breaker.last_failure_time
            },
            'resource_metrics': self.get_resource_metrics().__dict__,
            'performance_history_size': len(self._performance_history),
            'active_requests': len(self._active_requests)
        }
        
        # Add performance trends
        if len(self._performance_history) >= 5:
            recent_perf = list(self._performance_history)[-5:]
            insights['performance_trend'] = {
                'avg_response_time': sum(p.get('response_time', 0) for p in recent_perf) / len(recent_perf),
                'trend': 'improving' if recent_perf[0].get('response_time', 0) > recent_perf[-1].get('response_time', 0) else 'degrading'
            }
        
        return insights
    
    def cleanup(self) -> None:
        """Cleanup all resources and background tasks."""
        logger.info("Cleaning up ScalableLegalProver resources")
        
        # Set shutdown flag
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Cleanup resource manager
        self.resource_manager.cleanup()
        
        # Clear caches
        if self.adaptive_cache:
            self.adaptive_cache._cache.clear()
        
        # Call parent cleanup
        super().clear_cache()
        
        logger.info("ScalableLegalProver cleanup completed")


class LoadBalancer:
    """Simple load balancer for distributing requests."""
    
    def __init__(self):
        """Initialize load balancer."""
        self._request_counts = defaultdict(int)
        self._response_times = defaultdict(list)
        self._last_reset = time.time()
    
    def route_request(self, request_id: str) -> str:
        """Route request to optimal worker (placeholder)."""
        # Simple round-robin for now
        worker_id = f"worker_{hash(request_id) % 4}"
        self._request_counts[worker_id] += 1
        return worker_id
    
    def record_response_time(self, worker_id: str, response_time: float) -> None:
        """Record response time for worker."""
        self._response_times[worker_id].append(response_time)
        
        # Keep only recent response times
        if len(self._response_times[worker_id]) > 100:
            self._response_times[worker_id] = self._response_times[worker_id][-50:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        return {
            'request_counts': dict(self._request_counts),
            'avg_response_times': {
                worker: sum(times) / len(times) if times else 0
                for worker, times in self._response_times.items()
            }
        }