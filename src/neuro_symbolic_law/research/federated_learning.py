"""
Generation 5: Federated Learning for Global Legal AI
Enables distributed learning across global deployments while preserving privacy.
"""

import asyncio
import hashlib
import json
import logging
import time
try:
    import numpy as np
except ImportError:
    # Fallback for environments without numpy
    class MockNumpy:
        def mean(self, arr): return sum(arr) / len(arr) if arr else 0
        def var(self, arr): 
            if not arr: return 0
            mean_val = self.mean(arr)
            return sum((x - mean_val) ** 2 for x in arr) / len(arr)
        def std(self, arr): return self.var(arr) ** 0.5
        def array(self, arr): return arr
        @property
        def random(self): 
            import random
            class MockRandom:
                def normal(self, mu, sigma, shape=None): 
                    if shape:
                        return [[random.gauss(mu, sigma) for _ in range(shape[1])] for _ in range(shape[0])]
                    return random.gauss(mu, sigma)
                def uniform(self, a, b): return random.uniform(a, b)
                def randint(self, a, b): return random.randint(a, b-1)
                def choice(self, arr, size=1, replace=True): 
                    import random
                    return [random.choice(arr) for _ in range(size)]
            return MockRandom()
    np = MockNumpy()
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import threading
from pathlib import Path
import pickle
import base64

logger = logging.getLogger(__name__)


class FederatedRole(Enum):
    """Roles in federated learning network."""
    COORDINATOR = "coordinator"
    PARTICIPANT = "participant"
    VALIDATOR = "validator"
    AGGREGATOR = "aggregator"


@dataclass
class ModelUpdate:
    """Federated model update package."""
    participant_id: str
    model_weights: Dict[str, Any]
    training_samples: int
    validation_accuracy: float
    privacy_budget: float
    update_hash: str
    timestamp: float = field(default_factory=time.time)
    
    def serialize(self) -> str:
        """Serialize model update for transmission."""
        data = {
            'participant_id': self.participant_id,
            'model_weights': {k: v.tolist() for k, v in self.model_weights.items()},
            'training_samples': self.training_samples,
            'validation_accuracy': self.validation_accuracy,
            'privacy_budget': self.privacy_budget,
            'update_hash': self.update_hash,
            'timestamp': self.timestamp
        }
        return base64.b64encode(json.dumps(data).encode()).decode()
    
    @classmethod
    def deserialize(cls, serialized: str) -> 'ModelUpdate':
        """Deserialize model update from transmission."""
        data = json.loads(base64.b64decode(serialized).decode())
        model_weights = {k: np.array(v) for k, v in data['model_weights'].items()}
        
        return cls(
            participant_id=data['participant_id'],
            model_weights=model_weights,
            training_samples=data['training_samples'],
            validation_accuracy=data['validation_accuracy'],
            privacy_budget=data['privacy_budget'],
            update_hash=data['update_hash'],
            timestamp=data['timestamp']
        )


@dataclass
class FederatedMetrics:
    """Metrics for federated learning performance."""
    round_number: int
    participating_nodes: int
    average_accuracy: float
    consensus_score: float
    communication_overhead: float
    privacy_preservation_score: float
    convergence_rate: float
    global_model_hash: str


class DifferentialPrivacy:
    """Differential privacy mechanism for federated learning."""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
        self.privacy_budget_used = 0.0
    
    def add_noise(self, gradients: Dict[str, Any], sensitivity: float = 1.0) -> Dict[str, Any]:
        """Add calibrated noise to gradients for privacy."""
        noisy_gradients = {}
        noise_scale = sensitivity / self.epsilon
        
        for layer_name, gradient in gradients.items():
            # Gaussian noise for differential privacy
            noise = np.random.normal(0, noise_scale, gradient.shape)
            noisy_gradients[layer_name] = gradient + noise
            
        self.privacy_budget_used += self.epsilon
        logger.debug(f"Added DP noise with scale {noise_scale:.4f}, budget used: {self.privacy_budget_used:.2f}")
        
        return noisy_gradients
    
    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget."""
        return max(0, 10.0 - self.privacy_budget_used)  # Total budget of 10.0


class SecureAggregator:
    """Secure aggregation for federated learning."""
    
    def __init__(self):
        self.aggregation_threshold = 3  # Minimum participants for aggregation
        self.byzantine_tolerance = 0.33  # Tolerate up to 33% malicious nodes
    
    def aggregate_updates(self, updates: List[ModelUpdate]) -> Dict[str, Any]:
        """Securely aggregate model updates."""
        if len(updates) < self.aggregation_threshold:
            raise ValueError(f"Need at least {self.aggregation_threshold} updates for secure aggregation")
        
        # Validate updates
        validated_updates = self._validate_updates(updates)
        
        # Remove potential outliers (Byzantine fault tolerance)
        filtered_updates = self._filter_byzantine_updates(validated_updates)
        
        # Weighted federated averaging
        aggregated_weights = self._federated_averaging(filtered_updates)
        
        logger.info(f"Aggregated {len(filtered_updates)} updates from {len(updates)} submissions")
        return aggregated_weights
    
    def _validate_updates(self, updates: List[ModelUpdate]) -> List[ModelUpdate]:
        """Validate model updates for consistency and authenticity."""
        validated = []
        
        for update in updates:
            # Verify hash integrity
            calculated_hash = self._calculate_update_hash(update)
            if calculated_hash == update.update_hash:
                # Check weight dimensions consistency
                if self._validate_weight_dimensions(update.model_weights):
                    validated.append(update)
                else:
                    logger.warning(f"Invalid weight dimensions from {update.participant_id}")
            else:
                logger.warning(f"Hash mismatch for update from {update.participant_id}")
        
        return validated
    
    def _calculate_update_hash(self, update: ModelUpdate) -> str:
        """Calculate hash of model update for integrity checking."""
        # Create deterministic hash of model weights
        weight_data = []
        for layer_name in sorted(update.model_weights.keys()):
            weight_data.append(update.model_weights[layer_name].tobytes())
        
        combined_data = b''.join(weight_data) + str(update.training_samples).encode()
        return hashlib.sha256(combined_data).hexdigest()
    
    def _validate_weight_dimensions(self, weights: Dict[str, Any]) -> bool:
        """Validate that weight dimensions are consistent."""
        expected_layers = ['input_layer', 'hidden_layer_1', 'hidden_layer_2', 'output_layer']
        
        for layer in expected_layers:
            if layer not in weights:
                return False
            
            # Check reasonable weight ranges
            weight_array = weights[layer]
            if np.any(np.isnan(weight_array)) or np.any(np.isinf(weight_array)):
                return False
            
            if np.max(np.abs(weight_array)) > 100:  # Sanity check
                return False
        
        return True
    
    def _filter_byzantine_updates(self, updates: List[ModelUpdate]) -> List[ModelUpdate]:
        """Filter out potentially malicious updates."""
        if len(updates) <= 3:
            return updates  # Too few updates to filter
        
        # Calculate accuracy statistics
        accuracies = [update.validation_accuracy for update in updates]
        accuracy_mean = np.mean(accuracies)
        accuracy_std = np.std(accuracies)
        
        # Filter outliers (potential Byzantine nodes)
        filtered_updates = []
        for update in updates:
            # Remove updates with suspiciously high or low accuracy
            if abs(update.validation_accuracy - accuracy_mean) <= 2 * accuracy_std:
                filtered_updates.append(update)
            else:
                logger.warning(f"Filtered suspicious update from {update.participant_id} (accuracy: {update.validation_accuracy:.3f})")
        
        # Ensure we don't remove too many (Byzantine tolerance)
        max_filtered = int(len(updates) * self.byzantine_tolerance)
        if len(updates) - len(filtered_updates) > max_filtered:
            # Keep the most recent updates if too many filtered
            sorted_updates = sorted(updates, key=lambda x: x.timestamp, reverse=True)
            filtered_updates = sorted_updates[:len(updates) - max_filtered]
        
        return filtered_updates
    
    def _federated_averaging(self, updates: List[ModelUpdate]) -> Dict[str, Any]:
        """Perform weighted federated averaging."""
        total_samples = sum(update.training_samples for update in updates)
        
        # Initialize aggregated weights
        aggregated_weights = {}
        first_update = updates[0]
        
        for layer_name in first_update.model_weights.keys():
            layer_shape = first_update.model_weights[layer_name].shape
            aggregated_weights[layer_name] = np.zeros(layer_shape)
        
        # Weighted averaging based on training samples
        for update in updates:
            weight = update.training_samples / total_samples
            
            for layer_name, weights in update.model_weights.items():
                aggregated_weights[layer_name] += weight * weights
        
        return aggregated_weights


class FederatedLearningNode:
    """
    Federated learning node for legal AI systems.
    
    Generation 5 Features:
    - Privacy-preserving distributed learning
    - Secure multi-party computation
    - Byzantine fault tolerance
    - Adaptive optimization strategies
    - Cross-jurisdiction knowledge sharing
    """
    
    def __init__(self, node_id: str, role: FederatedRole = FederatedRole.PARTICIPANT):
        self.node_id = node_id
        self.role = role
        self.connected_nodes: Set[str] = set()
        self.model_weights = self._initialize_model_weights()
        self.training_history = []
        self.federated_rounds_participated = 0
        
        # Privacy and security
        self.differential_privacy = DifferentialPrivacy()
        self.secure_aggregator = SecureAggregator()
        
        # Communication
        self.message_queue = asyncio.Queue()
        self.update_buffer: List[ModelUpdate] = []
        
        # Metrics
        self.federated_metrics: List[FederatedMetrics] = []
        self._lock = threading.RLock()
        
        logger.info(f"Initialized federated learning node {node_id} as {role.value}")
    
    def _initialize_model_weights(self) -> Dict[str, Any]:
        """Initialize model weights for legal AI components."""
        return {
            'input_layer': np.random.normal(0, 0.1, (768, 256)),  # BERT embeddings to hidden
            'hidden_layer_1': np.random.normal(0, 0.1, (256, 128)),
            'hidden_layer_2': np.random.normal(0, 0.1, (128, 64)),
            'output_layer': np.random.normal(0, 0.1, (64, 10))  # Classification outputs
        }
    
    async def participate_in_round(self, round_number: int, global_weights: Optional[Dict[str, Any]] = None) -> ModelUpdate:
        """Participate in a federated learning round."""
        with self._lock:
            logger.info(f"Node {self.node_id} participating in round {round_number}")
            
            # Update local model with global weights if provided
            if global_weights:
                self.model_weights = global_weights.copy()
            
            # Simulate local training
            local_update = await self._local_training_simulation()
            
            # Apply differential privacy
            private_weights = self.differential_privacy.add_noise(
                local_update['gradients'], 
                sensitivity=1.0
            )
            
            # Create model update
            update_hash = hashlib.sha256(
                json.dumps({k: v.tolist() for k, v in private_weights.items()}).encode()
            ).hexdigest()
            
            model_update = ModelUpdate(
                participant_id=self.node_id,
                model_weights=private_weights,
                training_samples=local_update['samples'],
                validation_accuracy=local_update['accuracy'],
                privacy_budget=self.differential_privacy.privacy_budget_used,
                update_hash=update_hash
            )
            
            self.federated_rounds_participated += 1
            self.training_history.append({
                'round': round_number,
                'accuracy': local_update['accuracy'],
                'privacy_budget_used': self.differential_privacy.privacy_budget_used,
                'timestamp': time.time()
            })
            
            return model_update
    
    async def _local_training_simulation(self) -> Dict[str, Any]:
        """Simulate local training on jurisdiction-specific legal data."""
        # Simulate training process
        await asyncio.sleep(0.1)  # Simulate training time
        
        # Simulate gradient computation
        gradients = {}
        for layer_name, weights in self.model_weights.items():
            # Simulate gradients with some improvement signal
            gradient_noise = np.random.normal(0, 0.01, weights.shape)
            gradients[layer_name] = weights * 0.001 + gradient_noise  # Small update
        
        # Simulate training metrics
        base_accuracy = 0.85
        accuracy_improvement = np.random.normal(0.02, 0.01)  # Small improvement
        simulated_accuracy = min(0.99, base_accuracy + accuracy_improvement)
        
        # Simulate varying training data sizes
        simulated_samples = np.random.randint(100, 1000)
        
        return {
            'gradients': gradients,
            'accuracy': simulated_accuracy,
            'samples': simulated_samples
        }
    
    async def coordinate_federated_round(self, participant_nodes: List['FederatedLearningNode'], round_number: int) -> FederatedMetrics:
        """Coordinate a federated learning round (coordinator role)."""
        if self.role != FederatedRole.COORDINATOR:
            raise ValueError("Only coordinator nodes can coordinate rounds")
        
        logger.info(f"Coordinating federated round {round_number} with {len(participant_nodes)} participants")
        
        # Collect updates from participants
        updates = []
        current_global_weights = self.model_weights if hasattr(self, 'global_weights') else None
        
        for participant in participant_nodes:
            try:
                update = await participant.participate_in_round(round_number, current_global_weights)
                updates.append(update)
            except Exception as e:
                logger.error(f"Failed to get update from {participant.node_id}: {e}")
        
        if len(updates) < self.secure_aggregator.aggregation_threshold:
            raise ValueError(f"Insufficient updates received: {len(updates)}")
        
        # Secure aggregation
        aggregated_weights = self.secure_aggregator.aggregate_updates(updates)
        self.model_weights = aggregated_weights
        
        # Calculate round metrics
        metrics = self._calculate_round_metrics(updates, round_number)
        self.federated_metrics.append(metrics)
        
        logger.info(f"Round {round_number} completed: avg_accuracy={metrics.average_accuracy:.3f}, consensus={metrics.consensus_score:.3f}")
        
        return metrics
    
    def _calculate_round_metrics(self, updates: List[ModelUpdate], round_number: int) -> FederatedMetrics:
        """Calculate metrics for the federated round."""
        accuracies = [update.validation_accuracy for update in updates]
        average_accuracy = sum(accuracies) / len(accuracies)
        
        # Consensus score based on accuracy variance
        accuracy_variance = np.var(accuracies)
        consensus_score = max(0, 1 - accuracy_variance)  # Higher consensus = lower variance
        
        # Privacy preservation score
        avg_privacy_budget = sum(update.privacy_budget for update in updates) / len(updates)
        privacy_score = max(0, 1 - avg_privacy_budget / 10.0)  # Normalize to [0,1]
        
        # Communication overhead (simplified)
        total_weights = sum(
            sum(w.size for w in update.model_weights.values()) 
            for update in updates
        )
        communication_overhead = total_weights / (1024 * 1024)  # MB
        
        # Convergence rate (simplified)
        if len(self.federated_metrics) > 0:
            prev_accuracy = self.federated_metrics[-1].average_accuracy
            convergence_rate = (average_accuracy - prev_accuracy) / max(prev_accuracy, 0.01)
        else:
            convergence_rate = 0.0
        
        # Global model hash for integrity
        global_model_hash = hashlib.sha256(
            json.dumps({k: v.tolist() for k, v in self.model_weights.items()}).encode()
        ).hexdigest()[:16]
        
        return FederatedMetrics(
            round_number=round_number,
            participating_nodes=len(updates),
            average_accuracy=average_accuracy,
            consensus_score=consensus_score,
            communication_overhead=communication_overhead,
            privacy_preservation_score=privacy_score,
            convergence_rate=convergence_rate,
            global_model_hash=global_model_hash
        )
    
    def get_federated_insights(self) -> Dict[str, Any]:
        """Get insights from federated learning participation."""
        with self._lock:
            if not self.federated_metrics:
                return {"message": "No federated learning rounds completed"}
            
            recent_metrics = self.federated_metrics[-10:]  # Last 10 rounds
            
            avg_accuracy_trend = [m.average_accuracy for m in recent_metrics]
            consensus_trend = [m.consensus_score for m in recent_metrics]
            privacy_trend = [m.privacy_preservation_score for m in recent_metrics]
            
            return {
                'node_id': self.node_id,
                'role': self.role.value,
                'total_rounds_participated': self.federated_rounds_participated,
                'recent_average_accuracy': np.mean(avg_accuracy_trend),
                'accuracy_improvement': avg_accuracy_trend[-1] - avg_accuracy_trend[0] if len(avg_accuracy_trend) > 1 else 0,
                'average_consensus_score': np.mean(consensus_trend),
                'privacy_preservation_score': np.mean(privacy_trend),
                'remaining_privacy_budget': self.differential_privacy.get_remaining_budget(),
                'connected_nodes': len(self.connected_nodes),
                'total_federated_metrics': len(self.federated_metrics),
                'latest_global_model_hash': self.federated_metrics[-1].global_model_hash if self.federated_metrics else None
            }


class GlobalFederatedCoordinator:
    """
    Global coordinator for cross-jurisdiction federated learning.
    """
    
    def __init__(self):
        self.regional_coordinators: Dict[str, FederatedLearningNode] = {}
        self.global_model_weights = {}
        self.cross_jurisdiction_insights = []
        self.regulatory_adaptation_history = []
        
    def register_regional_coordinator(self, region: str, coordinator: FederatedLearningNode):
        """Register a regional coordinator."""
        self.regional_coordinators[region] = coordinator
        logger.info(f"Registered regional coordinator for {region}")
    
    async def cross_jurisdiction_learning_round(self) -> Dict[str, Any]:
        """Coordinate learning across multiple jurisdictions."""
        if len(self.regional_coordinators) < 2:
            return {"error": "Need at least 2 regional coordinators"}
        
        logger.info(f"Starting cross-jurisdiction learning with {len(self.regional_coordinators)} regions")
        
        # Collect regional model updates
        regional_updates = {}
        for region, coordinator in self.regional_coordinators.items():
            try:
                # Simulate getting regional consensus model
                regional_weights = coordinator.model_weights
                regional_accuracy = (
                    coordinator.federated_metrics[-1].average_accuracy 
                    if coordinator.federated_metrics else 0.85
                )
                
                regional_updates[region] = {
                    'weights': regional_weights,
                    'accuracy': regional_accuracy,
                    'participants': coordinator.federated_rounds_participated
                }
                
            except Exception as e:
                logger.error(f"Failed to get update from region {region}: {e}")
        
        # Cross-jurisdiction aggregation
        global_weights = self._aggregate_regional_models(regional_updates)
        self.global_model_weights = global_weights
        
        # Generate cross-jurisdiction insights
        insights = self._analyze_cross_jurisdiction_patterns(regional_updates)
        self.cross_jurisdiction_insights.append(insights)
        
        return {
            'participating_regions': list(regional_updates.keys()),
            'global_model_hash': hashlib.sha256(str(global_weights).encode()).hexdigest()[:16],
            'cross_jurisdiction_insights': insights,
            'regulatory_harmonization_score': self._calculate_harmonization_score(regional_updates)
        }
    
    def _aggregate_regional_models(self, regional_updates: Dict[str, Dict]) -> Dict[str, Any]:
        """Aggregate models from different jurisdictions."""
        # Weight regions by accuracy and participation
        total_weight = 0
        aggregated_weights = {}
        
        # Initialize with first region's structure
        first_region = list(regional_updates.keys())[0]
        first_weights = regional_updates[first_region]['weights']
        
        for layer_name in first_weights.keys():
            aggregated_weights[layer_name] = np.zeros_like(first_weights[layer_name])
        
        # Weighted aggregation
        for region, update in regional_updates.items():
            region_weight = update['accuracy'] * np.log(1 + update['participants'])
            total_weight += region_weight
            
            for layer_name, weights in update['weights'].items():
                aggregated_weights[layer_name] += region_weight * weights
        
        # Normalize
        if total_weight > 0:
            for layer_name in aggregated_weights:
                aggregated_weights[layer_name] /= total_weight
        
        return aggregated_weights
    
    def _analyze_cross_jurisdiction_patterns(self, regional_updates: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze patterns across jurisdictions."""
        accuracies = [update['accuracy'] for update in regional_updates.values()]
        
        # Identify leading and lagging regions
        region_performance = {
            region: update['accuracy'] 
            for region, update in regional_updates.items()
        }
        
        best_region = max(region_performance, key=region_performance.get)
        worst_region = min(region_performance, key=region_performance.get)
        
        accuracy_variance = np.var(accuracies)
        
        return {
            'timestamp': time.time(),
            'average_cross_jurisdiction_accuracy': np.mean(accuracies),
            'accuracy_variance': accuracy_variance,
            'leading_region': best_region,
            'leading_region_accuracy': region_performance[best_region],
            'lagging_region': worst_region,
            'lagging_region_accuracy': region_performance[worst_region],
            'harmonization_opportunity': accuracy_variance > 0.1,
            'knowledge_transfer_potential': region_performance[best_region] - region_performance[worst_region]
        }
    
    def _calculate_harmonization_score(self, regional_updates: Dict[str, Dict]) -> float:
        """Calculate how well-harmonized the regional models are."""
        accuracies = [update['accuracy'] for update in regional_updates.values()]
        
        # Lower variance = higher harmonization
        accuracy_variance = np.var(accuracies)
        harmonization_score = max(0, 1 - accuracy_variance * 10)  # Scale variance
        
        return harmonization_score


# Global federated learning instances
_federated_nodes: Dict[str, FederatedLearningNode] = {}
_global_coordinator: Optional[GlobalFederatedCoordinator] = None

def create_federated_node(node_id: str, role: FederatedRole = FederatedRole.PARTICIPANT) -> FederatedLearningNode:
    """Create a new federated learning node."""
    node = FederatedLearningNode(node_id, role)
    _federated_nodes[node_id] = node
    return node

def get_global_federated_coordinator() -> GlobalFederatedCoordinator:
    """Get global federated coordinator instance."""
    global _global_coordinator
    if _global_coordinator is None:
        _global_coordinator = GlobalFederatedCoordinator()
    return _global_coordinator

def get_federated_node(node_id: str) -> Optional[FederatedLearningNode]:
    """Get federated learning node by ID."""
    return _federated_nodes.get(node_id)