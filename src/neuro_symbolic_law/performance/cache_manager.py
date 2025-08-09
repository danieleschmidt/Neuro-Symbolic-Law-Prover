"""
Advanced caching system for high-performance compliance verification.
"""

import asyncio
import json
import time
import hashlib
import logging
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor
import weakref

# Redis imports with fallback
try:
    import redis
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Memory profiling
import psutil

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache hierarchy levels."""
    L1_MEMORY = "l1_memory"       # In-process memory cache
    L2_LOCAL = "l2_local"         # Local file-based cache  
    L3_DISTRIBUTED = "l3_distributed"  # Redis/distributed cache
    L4_PERSISTENT = "l4_persistent"    # Database/persistent cache


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"                   # Least Recently Used
    LFU = "lfu"                   # Least Frequently Used
    TTL = "ttl"                   # Time To Live
    ADAPTIVE = "adaptive"         # ML-based adaptive caching


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    avg_access_time: float = 0.0
    hit_rate: float = 0.0
    memory_usage: int = 0
    last_reset: datetime = field(default_factory=datetime.now)
    
    def update_hit(self, access_time: float):
        self.hits += 1
        self.total_requests += 1
        self._update_avg_time(access_time)
        self._update_hit_rate()
    
    def update_miss(self, access_time: float):
        self.misses += 1
        self.total_requests += 1
        self._update_avg_time(access_time)
        self._update_hit_rate()
    
    def _update_avg_time(self, access_time: float):
        if self.total_requests == 1:
            self.avg_access_time = access_time
        else:
            self.avg_access_time = (
                (self.avg_access_time * (self.total_requests - 1) + access_time) / 
                self.total_requests
            )
    
    def _update_hit_rate(self):
        if self.total_requests > 0:
            self.hit_rate = self.hits / self.total_requests * 100
    
    def reset(self):
        """Reset all statistics."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_requests = 0
        self.avg_access_time = 0.0
        self.hit_rate = 0.0
        self.memory_usage = 0
        self.last_reset = datetime.now()


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    ttl_seconds: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds
    
    def update_access(self):
        """Update access statistics."""
        self.last_accessed = datetime.now()
        self.access_count += 1


class AdaptiveCachePolicy:
    """ML-based adaptive caching policy."""
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.access_patterns: Dict[str, List[float]] = {}
        self.prediction_weights: Dict[str, float] = {}
        self.global_weights = {
            'recency': 0.3,
            'frequency': 0.3, 
            'size': 0.2,
            'prediction': 0.2
        }
    
    def predict_future_access(self, key: str) -> float:
        """Predict likelihood of future access."""
        if key not in self.access_patterns:
            return 0.5  # Default probability
        
        # Simple time-series prediction based on access intervals
        intervals = self.access_patterns[key]
        if len(intervals) < 2:
            return 0.7  # Recently accessed items have higher probability
        
        # Calculate trend
        recent_avg = sum(intervals[-3:]) / min(3, len(intervals))
        overall_avg = sum(intervals) / len(intervals)
        
        # Predict based on trend
        if recent_avg < overall_avg:
            return min(0.9, 0.5 + (overall_avg - recent_avg) / overall_avg)
        else:
            return max(0.1, 0.5 - (recent_avg - overall_avg) / overall_avg)
    
    def calculate_priority(self, entry: CacheEntry) -> float:
        """Calculate cache retention priority."""
        now = datetime.now()
        
        # Recency score (0-1, higher is more recent)
        recency = max(0, 1 - (now - entry.last_accessed).total_seconds() / 3600)
        
        # Frequency score (normalized by time since creation)
        age_hours = max(1, (now - entry.created_at).total_seconds() / 3600)
        frequency = min(1, entry.access_count / age_hours)
        
        # Size penalty (0-1, smaller is better)
        size_penalty = max(0, 1 - entry.size_bytes / (1024 * 1024))  # Penalty for >1MB items
        
        # Future access prediction
        prediction = self.predict_future_access(entry.key)
        
        # Weighted combination
        priority = (
            self.global_weights['recency'] * recency +
            self.global_weights['frequency'] * frequency +
            self.global_weights['size'] * size_penalty +
            self.global_weights['prediction'] * prediction
        )
        
        return priority
    
    def record_access(self, key: str, access_time: float):
        """Record access pattern for learning."""
        if key not in self.access_patterns:
            self.access_patterns[key] = []
        
        # Keep only recent access intervals
        if len(self.access_patterns[key]) >= 10:
            self.access_patterns[key] = self.access_patterns[key][-9:]
        
        self.access_patterns[key].append(access_time)
    
    def adapt_weights(self, cache_performance: CacheStats):
        """Adapt policy weights based on performance."""
        # Simple adaptation based on hit rate
        if cache_performance.hit_rate < 70:  # Low hit rate
            # Increase weight on prediction and frequency
            self.global_weights['prediction'] *= 1.1
            self.global_weights['frequency'] *= 1.05
            self.global_weights['recency'] *= 0.95
        elif cache_performance.hit_rate > 90:  # High hit rate
            # Can afford to optimize for size
            self.global_weights['size'] *= 1.1
            self.global_weights['prediction'] *= 0.95
        
        # Normalize weights
        total_weight = sum(self.global_weights.values())
        for key in self.global_weights:
            self.global_weights[key] /= total_weight


class CacheManager:
    """
    Advanced multi-level caching system with adaptive policies.
    
    Implements hierarchical caching with L1-L4 levels:
    - L1: In-memory cache for hot data
    - L2: Local file cache for warm data  
    - L3: Distributed Redis cache for shared data
    - L4: Persistent database cache for cold data
    """
    
    def __init__(self,
                 l1_max_size: int = 1000,
                 l1_ttl: int = 300,
                 l2_max_size: int = 10000,
                 l2_ttl: int = 3600,
                 redis_url: Optional[str] = None,
                 policy: CachePolicy = CachePolicy.ADAPTIVE,
                 enable_compression: bool = True,
                 enable_encryption: bool = False):
        """
        Initialize multi-level cache manager.
        
        Args:
            l1_max_size: Maximum L1 cache entries
            l1_ttl: L1 cache TTL in seconds
            l2_max_size: Maximum L2 cache entries  
            l2_ttl: L2 cache TTL in seconds
            redis_url: Redis connection URL for L3 cache
            policy: Cache eviction policy
            enable_compression: Enable value compression
            enable_encryption: Enable value encryption
        """
        self.l1_max_size = l1_max_size
        self.l1_ttl = l1_ttl
        self.l2_max_size = l2_max_size
        self.l2_ttl = l2_ttl
        self.policy = policy
        self.enable_compression = enable_compression
        self.enable_encryption = enable_encryption
        
        # Cache storage
        self.l1_cache: Dict[str, CacheEntry] = {}
        self.l2_cache: Dict[str, CacheEntry] = {}
        
        # Cache statistics
        self.l1_stats = CacheStats()
        self.l2_stats = CacheStats()
        self.l3_stats = CacheStats()
        
        # Adaptive policy
        if policy == CachePolicy.ADAPTIVE:
            self.adaptive_policy = AdaptiveCachePolicy()
        else:
            self.adaptive_policy = None
        
        # Thread safety
        self.l1_lock = threading.RLock()
        self.l2_lock = threading.RLock()
        
        # Background maintenance
        self.maintenance_executor = ThreadPoolExecutor(max_workers=2)
        self.maintenance_interval = 60  # seconds
        self.last_maintenance = time.time()
        
        # Redis connection for L3 cache
        self.redis_client = None
        if redis_url and REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()  # Test connection
                logger.info("L3 distributed cache (Redis) initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis cache: {e}")
        
        # Compression/encryption setup
        if enable_compression:
            try:
                import zlib
                self.compressor = zlib
            except ImportError:
                logger.warning("Compression requested but zlib not available")
                self.enable_compression = False
        
        if enable_encryption:
            try:
                from cryptography.fernet import Fernet
                self.cipher = Fernet(Fernet.generate_key())
            except ImportError:
                logger.warning("Encryption requested but cryptography not available")
                self.enable_encryption = False
        
        logger.info(f"CacheManager initialized with policy: {policy.value}")
    
    async def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache with hierarchical lookup.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        start_time = time.time()
        
        # Try L1 cache first (fastest)
        try:
            value = await self._get_l1(key)
            if value is not None:
                self.l1_stats.update_hit(time.time() - start_time)
                return value
            else:
                self.l1_stats.update_miss(time.time() - start_time)
        except Exception as e:
            logger.error(f"L1 cache error: {e}")
        
        # Try L2 cache
        try:
            value = await self._get_l2(key)
            if value is not None:
                self.l2_stats.update_hit(time.time() - start_time)
                # Promote to L1 for future access
                asyncio.create_task(self._set_l1(key, value, self.l1_ttl))
                return value
            else:
                self.l2_stats.update_miss(time.time() - start_time)
        except Exception as e:
            logger.error(f"L2 cache error: {e}")
        
        # Try L3 cache (distributed)
        if self.redis_client:
            try:
                value = await self._get_l3(key)
                if value is not None:
                    self.l3_stats.update_hit(time.time() - start_time)
                    # Promote to L1 and L2
                    asyncio.create_task(self._set_l1(key, value, self.l1_ttl))
                    asyncio.create_task(self._set_l2(key, value, self.l2_ttl))
                    return value
                else:
                    self.l3_stats.update_miss(time.time() - start_time)
            except Exception as e:
                logger.error(f"L3 cache error: {e}")
        
        # Cache miss at all levels
        return default
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache with intelligent placement.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successfully cached
        """
        success = False
        
        # Calculate value size
        try:
            serialized_value = self._serialize(value)
            value_size = len(serialized_value.encode('utf-8'))
        except Exception as e:
            logger.error(f"Failed to serialize cache value: {e}")
            return False
        
        # Intelligent cache level selection
        cache_levels = self._select_cache_levels(key, value_size, ttl)
        
        # Set in selected cache levels
        for level in cache_levels:
            try:
                if level == CacheLevel.L1_MEMORY:
                    await self._set_l1(key, value, ttl or self.l1_ttl)
                    success = True
                elif level == CacheLevel.L2_LOCAL:
                    await self._set_l2(key, value, ttl or self.l2_ttl)
                    success = True
                elif level == CacheLevel.L3_DISTRIBUTED and self.redis_client:
                    await self._set_l3(key, value, ttl or 3600)
                    success = True
            except Exception as e:
                logger.error(f"Failed to set {level.value} cache: {e}")
        
        return success
    
    async def delete(self, key: str) -> bool:
        """Delete key from all cache levels."""
        success = True
        
        # Delete from all levels
        try:
            await self._delete_l1(key)
        except Exception as e:
            logger.error(f"L1 delete error: {e}")
            success = False
        
        try:
            await self._delete_l2(key)
        except Exception as e:
            logger.error(f"L2 delete error: {e}")
            success = False
        
        if self.redis_client:
            try:
                await self._delete_l3(key)
            except Exception as e:
                logger.error(f"L3 delete error: {e}")
                success = False
        
        return success
    
    async def clear(self, levels: Optional[List[CacheLevel]] = None) -> bool:
        """Clear specified cache levels or all levels."""
        if levels is None:
            levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_LOCAL, CacheLevel.L3_DISTRIBUTED]
        
        success = True
        
        for level in levels:
            try:
                if level == CacheLevel.L1_MEMORY:
                    with self.l1_lock:
                        self.l1_cache.clear()
                        self.l1_stats.reset()
                elif level == CacheLevel.L2_LOCAL:
                    with self.l2_lock:
                        self.l2_cache.clear()
                        self.l2_stats.reset()
                elif level == CacheLevel.L3_DISTRIBUTED and self.redis_client:
                    # Clear only our namespace in Redis
                    pattern = "nsl:*"  # Neuro-Symbolic Law prefix
                    keys = self.redis_client.keys(pattern)
                    if keys:
                        self.redis_client.delete(*keys)
                    self.l3_stats.reset()
            except Exception as e:
                logger.error(f"Failed to clear {level.value}: {e}")
                success = False
        
        return success
    
    def get_stats(self) -> Dict[str, CacheStats]:
        """Get cache statistics for all levels."""
        return {
            'l1_memory': self.l1_stats,
            'l2_local': self.l2_stats,
            'l3_distributed': self.l3_stats
        }
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage for cache levels."""
        return {
            'l1_bytes': sum(entry.size_bytes for entry in self.l1_cache.values()),
            'l2_bytes': sum(entry.size_bytes for entry in self.l2_cache.values()),
            'process_memory_mb': psutil.Process().memory_info().rss // (1024 * 1024)
        }
    
    async def _get_l1(self, key: str) -> Any:
        """Get from L1 memory cache."""
        with self.l1_lock:
            if key in self.l1_cache:
                entry = self.l1_cache[key]
                if entry.is_expired():
                    del self.l1_cache[key]
                    return None
                
                entry.update_access()
                if self.adaptive_policy:
                    self.adaptive_policy.record_access(key, time.time())
                
                return self._deserialize(entry.value)
        return None
    
    async def _set_l1(self, key: str, value: Any, ttl: int):
        """Set in L1 memory cache."""
        with self.l1_lock:
            # Check if eviction needed
            if len(self.l1_cache) >= self.l1_max_size:
                await self._evict_l1()
            
            serialized_value = self._serialize(value)
            size_bytes = len(serialized_value.encode('utf-8'))
            
            entry = CacheEntry(
                key=key,
                value=serialized_value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                size_bytes=size_bytes,
                ttl_seconds=ttl
            )
            
            self.l1_cache[key] = entry
    
    async def _get_l2(self, key: str) -> Any:
        """Get from L2 local cache."""
        with self.l2_lock:
            if key in self.l2_cache:
                entry = self.l2_cache[key]
                if entry.is_expired():
                    del self.l2_cache[key]
                    return None
                
                entry.update_access()
                return self._deserialize(entry.value)
        return None
    
    async def _set_l2(self, key: str, value: Any, ttl: int):
        """Set in L2 local cache."""
        with self.l2_lock:
            # Check if eviction needed
            if len(self.l2_cache) >= self.l2_max_size:
                await self._evict_l2()
            
            serialized_value = self._serialize(value)
            size_bytes = len(serialized_value.encode('utf-8'))
            
            entry = CacheEntry(
                key=key,
                value=serialized_value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                size_bytes=size_bytes,
                ttl_seconds=ttl
            )
            
            self.l2_cache[key] = entry
    
    async def _get_l3(self, key: str) -> Any:
        """Get from L3 distributed cache."""
        if not self.redis_client:
            return None
        
        try:
            cache_key = f"nsl:{key}"
            data = self.redis_client.get(cache_key)
            if data:
                return self._deserialize(data)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
        
        return None
    
    async def _set_l3(self, key: str, value: Any, ttl: int):
        """Set in L3 distributed cache."""
        if not self.redis_client:
            return
        
        try:
            cache_key = f"nsl:{key}"
            serialized_value = self._serialize(value)
            self.redis_client.setex(cache_key, ttl, serialized_value)
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    async def _delete_l1(self, key: str):
        """Delete from L1 cache."""
        with self.l1_lock:
            self.l1_cache.pop(key, None)
    
    async def _delete_l2(self, key: str):
        """Delete from L2 cache."""
        with self.l2_lock:
            self.l2_cache.pop(key, None)
    
    async def _delete_l3(self, key: str):
        """Delete from L3 cache."""
        if self.redis_client:
            cache_key = f"nsl:{key}"
            self.redis_client.delete(cache_key)
    
    async def _evict_l1(self):
        """Evict entries from L1 cache based on policy."""
        if not self.l1_cache:
            return
        
        entries_to_remove = max(1, len(self.l1_cache) // 10)  # Remove 10%
        
        if self.policy == CachePolicy.LRU:
            # Remove least recently used
            sorted_entries = sorted(
                self.l1_cache.items(),
                key=lambda x: x[1].last_accessed
            )
        elif self.policy == CachePolicy.LFU:
            # Remove least frequently used
            sorted_entries = sorted(
                self.l1_cache.items(),
                key=lambda x: x[1].access_count
            )
        elif self.policy == CachePolicy.ADAPTIVE and self.adaptive_policy:
            # Use adaptive policy
            sorted_entries = sorted(
                self.l1_cache.items(),
                key=lambda x: self.adaptive_policy.calculate_priority(x[1])
            )
        else:
            # Default to LRU
            sorted_entries = sorted(
                self.l1_cache.items(),
                key=lambda x: x[1].last_accessed
            )
        
        for i in range(entries_to_remove):
            if i < len(sorted_entries):
                key_to_remove = sorted_entries[i][0]
                del self.l1_cache[key_to_remove]
                self.l1_stats.evictions += 1
    
    async def _evict_l2(self):
        """Evict entries from L2 cache based on policy."""
        # Similar to L1 eviction but for L2
        if not self.l2_cache:
            return
        
        entries_to_remove = max(1, len(self.l2_cache) // 10)
        
        sorted_entries = sorted(
            self.l2_cache.items(),
            key=lambda x: x[1].last_accessed
        )
        
        for i in range(entries_to_remove):
            if i < len(sorted_entries):
                key_to_remove = sorted_entries[i][0]
                del self.l2_cache[key_to_remove]
                self.l2_stats.evictions += 1
    
    def _select_cache_levels(self, key: str, value_size: int, ttl: Optional[int]) -> List[CacheLevel]:
        """Intelligently select which cache levels to use."""
        levels = []
        
        # Small, frequently accessed items go to L1
        if value_size < 10 * 1024:  # Less than 10KB
            levels.append(CacheLevel.L1_MEMORY)
        
        # Medium items go to L2
        if value_size < 1024 * 1024:  # Less than 1MB
            levels.append(CacheLevel.L2_LOCAL)
        
        # All items can go to L3 if available
        if self.redis_client:
            levels.append(CacheLevel.L3_DISTRIBUTED)
        
        # If no levels selected, default to L1
        if not levels:
            levels.append(CacheLevel.L1_MEMORY)
        
        return levels
    
    def _serialize(self, value: Any) -> str:
        """Serialize value for caching."""
        try:
            serialized = json.dumps(value, default=str)
            
            if self.enable_compression:
                compressed = self.compressor.compress(serialized.encode('utf-8'))
                serialized = compressed.hex()
            
            if self.enable_encryption:
                encrypted = self.cipher.encrypt(serialized.encode('utf-8'))
                serialized = encrypted.hex()
            
            return serialized
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            raise
    
    def _deserialize(self, serialized: str) -> Any:
        """Deserialize cached value."""
        try:
            value = serialized
            
            if self.enable_encryption:
                decrypted = self.cipher.decrypt(bytes.fromhex(value))
                value = decrypted.decode('utf-8')
            
            if self.enable_compression:
                decompressed = self.compressor.decompress(bytes.fromhex(value))
                value = decompressed.decode('utf-8')
            
            return json.loads(value)
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            raise
    
    async def maintenance(self):
        """Perform periodic cache maintenance."""
        current_time = time.time()
        
        if current_time - self.last_maintenance < self.maintenance_interval:
            return
        
        logger.debug("Performing cache maintenance")
        
        # Remove expired entries
        await self._cleanup_expired()
        
        # Update adaptive policy if enabled
        if self.adaptive_policy:
            overall_stats = CacheStats()
            overall_stats.hits = self.l1_stats.hits + self.l2_stats.hits + self.l3_stats.hits
            overall_stats.total_requests = (
                self.l1_stats.total_requests + 
                self.l2_stats.total_requests + 
                self.l3_stats.total_requests
            )
            if overall_stats.total_requests > 0:
                overall_stats.hit_rate = overall_stats.hits / overall_stats.total_requests * 100
            
            self.adaptive_policy.adapt_weights(overall_stats)
        
        self.last_maintenance = current_time
        logger.debug("Cache maintenance completed")
    
    async def _cleanup_expired(self):
        """Remove expired entries from all cache levels."""
        # L1 cleanup
        with self.l1_lock:
            expired_keys = [
                key for key, entry in self.l1_cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                del self.l1_cache[key]
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired L1 entries")
        
        # L2 cleanup
        with self.l2_lock:
            expired_keys = [
                key for key, entry in self.l2_cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                del self.l2_cache[key]
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired L2 entries")


class DistributedCache:
    """
    Distributed cache implementation using Redis with advanced features.
    """
    
    def __init__(self, redis_cluster_urls: List[str], 
                 enable_sharding: bool = True,
                 replication_factor: int = 2):
        """
        Initialize distributed cache with Redis cluster.
        
        Args:
            redis_cluster_urls: List of Redis node URLs
            enable_sharding: Enable data sharding across nodes
            replication_factor: Number of replicas for each cache entry
        """
        self.redis_urls = redis_cluster_urls
        self.enable_sharding = enable_sharding
        self.replication_factor = replication_factor
        self.clients: List[redis.Redis] = []
        
        # Initialize Redis clients
        for url in redis_cluster_urls:
            try:
                client = redis.from_url(url, decode_responses=True)
                client.ping()
                self.clients.append(client)
                logger.info(f"Connected to Redis node: {url}")
            except Exception as e:
                logger.error(f"Failed to connect to Redis node {url}: {e}")
        
        if not self.clients:
            raise RuntimeError("No Redis nodes available")
        
        logger.info(f"DistributedCache initialized with {len(self.clients)} nodes")
    
    def _get_shard(self, key: str) -> int:
        """Get shard index for key using consistent hashing."""
        if not self.enable_sharding:
            return 0
        
        # Simple hash-based sharding
        hash_value = hashlib.md5(key.encode()).hexdigest()
        return int(hash_value, 16) % len(self.clients)
    
    def _get_replica_shards(self, primary_shard: int) -> List[int]:
        """Get replica shard indices."""
        replicas = []
        for i in range(1, self.replication_factor):
            replica_shard = (primary_shard + i) % len(self.clients)
            replicas.append(replica_shard)
        return replicas
    
    async def get(self, key: str) -> Any:
        """Get value from distributed cache with fault tolerance."""
        primary_shard = self._get_shard(key)
        
        # Try primary shard first
        try:
            client = self.clients[primary_shard]
            data = client.get(f"nsl:{key}")
            if data:
                return json.loads(data)
        except Exception as e:
            logger.warning(f"Primary shard {primary_shard} failed for key {key}: {e}")
        
        # Try replica shards
        replica_shards = self._get_replica_shards(primary_shard)
        for shard in replica_shards:
            try:
                client = self.clients[shard]
                data = client.get(f"nsl:{key}")
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.warning(f"Replica shard {shard} failed for key {key}: {e}")
        
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in distributed cache with replication."""
        serialized_value = json.dumps(value, default=str)
        primary_shard = self._get_shard(key)
        success_count = 0
        
        # Set in primary shard
        try:
            client = self.clients[primary_shard]
            client.setex(f"nsl:{key}", ttl, serialized_value)
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to set key {key} in primary shard {primary_shard}: {e}")
        
        # Set in replica shards
        replica_shards = self._get_replica_shards(primary_shard)
        for shard in replica_shards:
            try:
                client = self.clients[shard]
                client.setex(f"nsl:{key}", ttl, serialized_value)
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to set key {key} in replica shard {shard}: {e}")
        
        # Consider success if at least one replica succeeded
        return success_count > 0
    
    async def delete(self, key: str) -> bool:
        """Delete key from all shards."""
        success_count = 0
        
        for i, client in enumerate(self.clients):
            try:
                result = client.delete(f"nsl:{key}")
                if result:
                    success_count += 1
            except Exception as e:
                logger.error(f"Failed to delete key {key} from shard {i}: {e}")
        
        return success_count > 0
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get cluster-wide cache statistics."""
        stats = {
            'nodes': len(self.clients),
            'replication_factor': self.replication_factor,
            'node_stats': []
        }
        
        for i, client in enumerate(self.clients):
            try:
                info = client.info('memory')
                node_stats = {
                    'node_id': i,
                    'memory_used': info.get('used_memory', 0),
                    'memory_peak': info.get('used_memory_peak', 0),
                    'connected': True
                }
            except Exception as e:
                node_stats = {
                    'node_id': i,
                    'error': str(e),
                    'connected': False
                }
            
            stats['node_stats'].append(node_stats)
        
        return stats