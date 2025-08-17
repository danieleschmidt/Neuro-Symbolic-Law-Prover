"""
Generation 5: Causal Reasoning for Legal Precedent Analysis
Advanced causal inference and reasoning for legal AI systems.
"""

import logging
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
        def zeros(self, shape): 
            if isinstance(shape, tuple):
                return [[0] * shape[1] for _ in range(shape[0])]
            return [0] * shape
        def ones(self, shape):
            if isinstance(shape, tuple):
                return [[1] * shape[1] for _ in range(shape[0])]
            return [1] * shape
        def eye(self, n): return [[1 if i==j else 0 for j in range(n)] for i in range(n)]
        def corrcoef(self, x, y): 
            # Simplified correlation coefficient
            return [[1.0, 0.5], [0.5, 1.0]]
        def dot(self, a, b): return sum(a[i] * b[i] for i in range(len(a)))
        def sqrt(self, x): return x ** 0.5
        @property
        def random(self): 
            import random
            class MockRandom:
                def normal(self, mu, sigma): return random.gauss(mu, sigma)
                def uniform(self, a, b): return random.uniform(a, b)
                def randint(self, a, b): return random.randint(a, b-1)
                def choice(self, arr): return random.choice(arr)
            return MockRandom()
        def argmax(self, arr): return arr.index(max(arr))
        def unique(self, arr, return_counts=False):
            unique_vals = list(set(arr))
            if return_counts:
                counts = [arr.count(val) for val in unique_vals]
                return unique_vals, counts
            return unique_vals
        def min(self, arr): return min(arr)
        def max(self, arr): return max(arr)
        def abs(self, x): return abs(x)
        def any(self, arr): return any(arr)
        def isnan(self, x): return x != x
        def isinf(self, x): return x == float('inf') or x == float('-inf')
    np = MockNumpy()

try:
    import networkx as nx
except ImportError:
    # Fallback for environments without networkx
    class MockNetworkX:
        class DiGraph:
            def __init__(self):
                self._nodes = {}
                self._edges = {}
            def add_node(self, node, **attrs): 
                self._nodes[node] = attrs
            def add_edge(self, u, v, **attrs): 
                if u not in self._edges: self._edges[u] = {}
                self._edges[u][v] = attrs
            def nodes(self): return list(self._nodes.keys())
            def edges(self): return [(u, v) for u in self._edges for v in self._edges[u]]
            def number_of_nodes(self): return len(self._nodes)
            def number_of_edges(self): return sum(len(targets) for targets in self._edges.values())
            def has_edge(self, u, v): return u in self._edges and v in self._edges[u]
            def copy(self): 
                new_graph = MockNetworkX.DiGraph()
                new_graph._nodes = self._nodes.copy()
                new_graph._edges = self._edges.copy()
                return new_graph
            def predecessors(self, node): 
                return [u for u in self._edges if node in self._edges[u]]
            def out_degree(self, node): return len(self._edges.get(node, {}))
            def in_degree(self, node): return len(self.predecessors(node))
            def __getitem__(self, node): return self._edges.get(node, {})
        def topological_sort(self, graph): return list(graph.nodes())
        def all_simple_paths(self, graph, source, target, cutoff=None): 
            return [[source, target]]  # Simplified
        class NetworkXError(Exception): pass
        class NetworkXNoPath(NetworkXError): pass
    nx = MockNetworkX()
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class CausalRelationType(Enum):
    """Types of causal relationships in legal reasoning."""
    DIRECT_CAUSE = "direct_cause"
    NECESSARY_CONDITION = "necessary_condition"
    SUFFICIENT_CONDITION = "sufficient_condition"
    CONTRIBUTORY_CAUSE = "contributory_cause"
    INTERVENING_CAUSE = "intervening_cause"
    PROXIMATE_CAUSE = "proximate_cause"
    BUT_FOR_CAUSE = "but_for_cause"
    LEGAL_CONSEQUENCE = "legal_consequence"


@dataclass
class CausalFactor:
    """A factor in causal reasoning."""
    id: str
    name: str
    description: str
    factor_type: str  # 'legal_rule', 'factual_event', 'precedent', 'statute'
    strength: float  # 0.0 to 1.0
    temporal_order: int  # Sequence in time
    jurisdiction: str
    legal_domain: str
    evidence_support: float = 0.0
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class CausalRelation:
    """A causal relationship between factors."""
    cause_factor: CausalFactor
    effect_factor: CausalFactor
    relation_type: CausalRelationType
    strength: float  # Causal strength 0.0 to 1.0
    confidence: float  # Confidence in the relationship
    evidence: List[str] = field(default_factory=list)
    precedent_support: List[str] = field(default_factory=list)
    temporal_gap: int = 0  # Time units between cause and effect
    
    def __hash__(self):
        return hash((self.cause_factor.id, self.effect_factor.id, self.relation_type.value))


@dataclass
class CausalChain:
    """A chain of causal relationships."""
    chain_id: str
    factors: List[CausalFactor]
    relations: List[CausalRelation]
    total_strength: float
    legal_outcome: str
    supporting_precedents: List[str] = field(default_factory=list)
    alternative_chains: List[str] = field(default_factory=list)


class CausalInferenceEngine:
    """
    Causal inference engine for legal reasoning.
    
    Generation 5 Features:
    - Causal discovery from legal precedents
    - Counterfactual reasoning
    - Causal chain analysis
    - Precedent-based causal models
    - Intervention analysis
    """
    
    def __init__(self):
        self.causal_graph = nx.DiGraph()
        self.causal_factors: Dict[str, CausalFactor] = {}
        self.causal_relations: Dict[str, CausalRelation] = {}
        self.causal_chains: List[CausalChain] = []
        self.precedent_database = defaultdict(list)
        self.intervention_effects = {}
        
        # Learning components
        self.causal_discoveries = []
        self.counterfactual_cache = {}
        self.temporal_patterns = defaultdict(list)
        
        logger.info("Initialized causal inference engine for legal reasoning")
    
    def discover_causal_structure(self, legal_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Discover causal structure from legal cases using causal discovery algorithms.
        """
        logger.info(f"Discovering causal structure from {len(legal_cases)} legal cases")
        
        # Extract factors and outcomes from cases
        factors, outcomes = self._extract_factors_and_outcomes(legal_cases)
        
        # Build temporal factor matrix
        factor_matrix, temporal_matrix = self._build_factor_matrices(legal_cases, factors)
        
        # Apply causal discovery algorithm (simplified PC algorithm)
        causal_structure = self._pc_algorithm(factor_matrix, temporal_matrix)
        
        # Validate discovered structure with legal domain knowledge
        validated_structure = self._validate_causal_structure(causal_structure, factors)
        
        # Update internal causal graph
        self._update_causal_graph(validated_structure, factors)
        
        discovery_result = {
            'discovered_factors': len(factors),
            'discovered_relations': len(validated_structure['edges']),
            'causal_strength_distribution': self._analyze_strength_distribution(validated_structure),
            'temporal_patterns': self._identify_temporal_patterns(validated_structure),
            'legal_domains_covered': list(set(f.legal_domain for f in factors.values())),
            'discovery_confidence': self._calculate_discovery_confidence(validated_structure)
        }
        
        self.causal_discoveries.append({
            'timestamp': time.time(),
            'result': discovery_result,
            'cases_analyzed': len(legal_cases)
        })
        
        return discovery_result
    
    def _extract_factors_and_outcomes(self, legal_cases: List[Dict[str, Any]]) -> Tuple[Dict[str, CausalFactor], List[str]]:
        """Extract causal factors and outcomes from legal cases."""
        factors = {}
        outcomes = []
        
        for i, case in enumerate(legal_cases):
            case_id = case.get('id', f'case_{i}')
            
            # Extract factual factors
            for j, fact in enumerate(case.get('facts', [])):
                factor_id = f"fact_{case_id}_{j}"
                factors[factor_id] = CausalFactor(
                    id=factor_id,
                    name=fact.get('description', f'Fact {j}'),
                    description=fact.get('full_description', ''),
                    factor_type='factual_event',
                    strength=fact.get('importance', 0.5),
                    temporal_order=fact.get('temporal_order', j),
                    jurisdiction=case.get('jurisdiction', 'unknown'),
                    legal_domain=case.get('legal_domain', 'general'),
                    evidence_support=fact.get('evidence_strength', 0.5)
                )
            
            # Extract legal rules applied
            for j, rule in enumerate(case.get('legal_rules', [])):
                factor_id = f"rule_{case_id}_{j}"
                factors[factor_id] = CausalFactor(
                    id=factor_id,
                    name=rule.get('name', f'Rule {j}'),
                    description=rule.get('description', ''),
                    factor_type='legal_rule',
                    strength=rule.get('weight', 0.8),
                    temporal_order=1000 + j,  # Rules typically applied after facts
                    jurisdiction=case.get('jurisdiction', 'unknown'),
                    legal_domain=case.get('legal_domain', 'general')
                )
            
            # Extract precedents cited
            for j, precedent in enumerate(case.get('precedents', [])):
                factor_id = f"precedent_{case_id}_{j}"
                factors[factor_id] = CausalFactor(
                    id=factor_id,
                    name=precedent.get('case_name', f'Precedent {j}'),
                    description=precedent.get('principle', ''),
                    factor_type='precedent',
                    strength=precedent.get('authority', 0.7),
                    temporal_order=2000 + j,  # Precedents applied conceptually
                    jurisdiction=precedent.get('jurisdiction', case.get('jurisdiction', 'unknown')),
                    legal_domain=case.get('legal_domain', 'general')
                )
            
            # Record outcome
            outcomes.append(case.get('outcome', 'unknown'))
        
        return factors, outcomes
    
    def _build_factor_matrices(self, legal_cases: List[Dict[str, Any]], factors: Dict[str, CausalFactor]) -> Tuple[Any, Any]:
        """Build matrices for causal discovery algorithms."""
        factor_list = list(factors.keys())
        n_factors = len(factor_list)
        n_cases = len(legal_cases)
        
        # Factor presence matrix (cases x factors)
        factor_matrix = np.zeros((n_cases, n_factors))
        
        # Temporal relationship matrix (factors x factors)
        temporal_matrix = np.zeros((n_factors, n_factors))
        
        for case_idx, case in enumerate(legal_cases):
            case_id = case.get('id', f'case_{case_idx}')
            
            # Mark factor presence in this case
            for factor_idx, factor_id in enumerate(factor_list):
                if case_id in factor_id:
                    factor_matrix[case_idx, factor_idx] = factors[factor_id].strength
        
        # Build temporal relationships
        for i, factor_id_1 in enumerate(factor_list):
            for j, factor_id_2 in enumerate(factor_list):
                if i != j:
                    factor_1 = factors[factor_id_1]
                    factor_2 = factors[factor_id_2]
                    
                    # Temporal precedence
                    if factor_1.temporal_order < factor_2.temporal_order:
                        temporal_matrix[i, j] = 1.0
                    elif factor_1.temporal_order == factor_2.temporal_order:
                        temporal_matrix[i, j] = 0.5  # Simultaneous
        
        return factor_matrix, temporal_matrix
    
    def _pc_algorithm(self, factor_matrix: Any, temporal_matrix: Any) -> Dict[str, Any]:
        """
        Simplified PC (Peter-Clark) algorithm for causal discovery.
        """
        n_factors = factor_matrix.shape[1]
        
        # Start with complete graph
        adjacency_matrix = np.ones((n_factors, n_factors)) - np.eye(n_factors)
        
        # Test for conditional independence
        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                if adjacency_matrix[i, j] == 1:
                    # Test independence of i and j given subsets of other variables
                    independence_score = self._test_conditional_independence(
                        factor_matrix[:, i], 
                        factor_matrix[:, j], 
                        factor_matrix, 
                        i, j
                    )
                    
                    # Remove edge if independent
                    if independence_score > 0.05:  # p-value threshold
                        adjacency_matrix[i, j] = 0
                        adjacency_matrix[j, i] = 0
        
        # Orient edges using temporal information
        oriented_matrix = self._orient_edges(adjacency_matrix, temporal_matrix)
        
        # Extract causal relationships
        edges = []
        for i in range(n_factors):
            for j in range(n_factors):
                if oriented_matrix[i, j] > 0:
                    strength = oriented_matrix[i, j]
                    edges.append({
                        'from': i,
                        'to': j,
                        'strength': strength,
                        'type': self._determine_causal_type(strength, temporal_matrix[i, j])
                    })
        
        return {
            'adjacency_matrix': oriented_matrix,
            'edges': edges,
            'n_factors': n_factors
        }
    
    def _test_conditional_independence(self, x: Any, y: Any, 
                                     full_matrix: Any, x_idx: int, y_idx: int) -> float:
        """Test conditional independence between two variables."""
        # Simplified independence test using correlation
        # In practice, would use more sophisticated tests like partial correlation
        
        # Calculate correlation without conditioning
        unconditional_corr = abs(np.corrcoef(x, y)[0, 1])
        
        # Calculate partial correlation conditioning on most relevant variables
        conditioning_vars = []
        for k in range(full_matrix.shape[1]):
            if k != x_idx and k != y_idx:
                corr_x_k = abs(np.corrcoef(x, full_matrix[:, k])[0, 1])
                corr_y_k = abs(np.corrcoef(y, full_matrix[:, k])[0, 1])
                if corr_x_k > 0.3 or corr_y_k > 0.3:  # Strong relationship
                    conditioning_vars.append(k)
        
        if not conditioning_vars:
            return 1.0 - unconditional_corr  # Return p-value analog
        
        # Simplified partial correlation (using strongest conditioning variable)
        if conditioning_vars:
            z = full_matrix[:, conditioning_vars[0]]
            
            # Remove linear effect of z from x and y
            x_residual = x - np.dot(x, z) / np.dot(z, z) * z
            y_residual = y - np.dot(y, z) / np.dot(z, z) * z
            
            # Calculate correlation of residuals
            partial_corr = abs(np.corrcoef(x_residual, y_residual)[0, 1])
            return 1.0 - partial_corr
        
        return 1.0 - unconditional_corr
    
    def _orient_edges(self, adjacency_matrix: Any, temporal_matrix: Any) -> Any:
        """Orient edges using temporal information and causal constraints."""
        n_factors = adjacency_matrix.shape[0]
        oriented_matrix = np.zeros((n_factors, n_factors))
        
        for i in range(n_factors):
            for j in range(n_factors):
                if adjacency_matrix[i, j] == 1:
                    # Use temporal information to orient edge
                    if temporal_matrix[i, j] > temporal_matrix[j, i]:
                        # i precedes j temporally, so i -> j
                        oriented_matrix[i, j] = adjacency_matrix[i, j] * temporal_matrix[i, j]
                    elif temporal_matrix[j, i] > temporal_matrix[i, j]:
                        # j precedes i temporally, so j -> i
                        oriented_matrix[j, i] = adjacency_matrix[i, j] * temporal_matrix[j, i]
                    else:
                        # Simultaneous or unclear temporal relationship
                        # Use additional heuristics or leave unoriented
                        strength = adjacency_matrix[i, j] * 0.5
                        oriented_matrix[i, j] = strength
                        oriented_matrix[j, i] = strength
        
        return oriented_matrix
    
    def _determine_causal_type(self, strength: float, temporal_precedence: float) -> CausalRelationType:
        """Determine the type of causal relationship."""
        if strength > 0.8 and temporal_precedence > 0.8:
            return CausalRelationType.DIRECT_CAUSE
        elif strength > 0.6:
            return CausalRelationType.CONTRIBUTORY_CAUSE
        elif temporal_precedence > 0.9:
            return CausalRelationType.NECESSARY_CONDITION
        else:
            return CausalRelationType.LEGAL_CONSEQUENCE
    
    def _validate_causal_structure(self, structure: Dict[str, Any], factors: Dict[str, CausalFactor]) -> Dict[str, Any]:
        """Validate discovered causal structure with legal domain knowledge."""
        validated_edges = []
        
        for edge in structure['edges']:
            from_idx, to_idx = edge['from'], edge['to']
            factor_list = list(factors.keys())
            
            from_factor = factors[factor_list[from_idx]]
            to_factor = factors[factor_list[to_idx]]
            
            # Legal validation rules
            is_valid = True
            
            # Rule 1: Facts should precede legal rules
            if (from_factor.factor_type == 'legal_rule' and 
                to_factor.factor_type == 'factual_event'):
                is_valid = False
            
            # Rule 2: Precedents can influence outcomes but not facts
            if (from_factor.factor_type == 'precedent' and 
                to_factor.factor_type == 'factual_event'):
                is_valid = False
            
            # Rule 3: Same-jurisdiction precedents have stronger influence
            if (from_factor.factor_type == 'precedent' and 
                from_factor.jurisdiction != to_factor.jurisdiction):
                edge['strength'] *= 0.7  # Reduce strength for cross-jurisdiction
            
            # Rule 4: Boost strength for domain-consistent relationships
            if from_factor.legal_domain == to_factor.legal_domain:
                edge['strength'] *= 1.2
            
            if is_valid:
                validated_edges.append(edge)
        
        structure['edges'] = validated_edges
        return structure
    
    def _update_causal_graph(self, structure: Dict[str, Any], factors: Dict[str, CausalFactor]):
        """Update internal causal graph with discovered structure."""
        factor_list = list(factors.keys())
        
        # Add factors as nodes
        for factor_id, factor in factors.items():
            self.causal_factors[factor_id] = factor
            self.causal_graph.add_node(factor_id, factor=factor)
        
        # Add causal relationships as edges
        for edge in structure['edges']:
            from_factor_id = factor_list[edge['from']]
            to_factor_id = factor_list[edge['to']]
            
            relation = CausalRelation(
                cause_factor=factors[from_factor_id],
                effect_factor=factors[to_factor_id],
                relation_type=edge['type'],
                strength=edge['strength'],
                confidence=0.8  # Default confidence
            )
            
            relation_id = f"{from_factor_id}→{to_factor_id}"
            self.causal_relations[relation_id] = relation
            
            self.causal_graph.add_edge(
                from_factor_id, to_factor_id,
                relation=relation,
                weight=edge['strength']
            )
    
    def counterfactual_reasoning(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform counterfactual reasoning: "What would have happened if...?"
        """
        logger.info(f"Performing counterfactual reasoning for query: {query.get('description', 'unnamed')}")
        
        # Parse counterfactual query
        intervention_factors = query.get('intervention_factors', [])
        target_outcome = query.get('target_outcome')
        factual_scenario = query.get('factual_scenario', {})
        
        # Check cache
        query_hash = self._hash_query(query)
        if query_hash in self.counterfactual_cache:
            return self.counterfactual_cache[query_hash]
        
        # Perform intervention on causal graph
        intervened_graph = self._perform_intervention(intervention_factors)
        
        # Calculate counterfactual outcome
        counterfactual_outcome = self._calculate_counterfactual_outcome(
            intervened_graph, factual_scenario, target_outcome
        )
        
        # Calculate causal effect
        causal_effect = self._calculate_causal_effect(
            factual_scenario.get('outcome'),
            counterfactual_outcome
        )
        
        # Find supporting and contradicting precedents
        precedent_analysis = self._analyze_precedent_support(intervention_factors, counterfactual_outcome)
        
        result = {
            'factual_outcome': factual_scenario.get('outcome'),
            'counterfactual_outcome': counterfactual_outcome,
            'causal_effect': causal_effect,
            'intervention_factors': intervention_factors,
            'confidence': self._calculate_counterfactual_confidence(intervened_graph, intervention_factors),
            'precedent_support': precedent_analysis,
            'alternative_scenarios': self._generate_alternative_scenarios(intervention_factors),
            'robustness_score': self._calculate_robustness_score(intervened_graph)
        }
        
        # Cache result
        self.counterfactual_cache[query_hash] = result
        
        return result
    
    def _perform_intervention(self, intervention_factors: List[Dict[str, Any]]) -> nx.DiGraph:
        """Perform intervention on causal graph (do-calculus)."""
        intervened_graph = self.causal_graph.copy()
        
        for intervention in intervention_factors:
            factor_id = intervention.get('factor_id')
            new_value = intervention.get('value')
            
            if factor_id in intervened_graph:
                # Remove incoming edges (break causal dependencies)
                incoming_edges = list(intervened_graph.in_edges(factor_id))
                intervened_graph.remove_edges_from(incoming_edges)
                
                # Set fixed value
                intervened_graph.nodes[factor_id]['intervention_value'] = new_value
        
        return intervened_graph
    
    def _calculate_counterfactual_outcome(self, intervened_graph: nx.DiGraph, 
                                        factual_scenario: Dict[str, Any], 
                                        target_outcome: str) -> Dict[str, Any]:
        """Calculate outcome under counterfactual intervention."""
        # Topological sort for causal ordering
        try:
            causal_order = list(nx.topological_sort(intervened_graph))
        except nx.NetworkXError:
            # Handle cycles by using best approximation
            causal_order = list(intervened_graph.nodes())
        
        # Propagate effects through causal chain
        node_values = {}
        
        for node_id in causal_order:
            if 'intervention_value' in intervened_graph.nodes[node_id]:
                # Use intervention value
                node_values[node_id] = intervened_graph.nodes[node_id]['intervention_value']
            else:
                # Calculate based on causal parents
                parent_effects = []
                for parent_id in intervened_graph.predecessors(node_id):
                    if parent_id in node_values:
                        edge_data = intervened_graph[parent_id][node_id]
                        parent_value = node_values[parent_id]
                        causal_strength = edge_data.get('weight', 0.5)
                        
                        parent_effects.append(parent_value * causal_strength)
                
                # Combine parent effects
                if parent_effects:
                    combined_effect = np.mean(parent_effects)  # Simple averaging
                    node_values[node_id] = min(1.0, max(0.0, combined_effect))
                else:
                    # Use baseline value from factual scenario
                    node_values[node_id] = factual_scenario.get(node_id, 0.5)
        
        # Determine final outcome
        outcome_probability = 0.5  # Default
        outcome_factors = []
        
        for node_id, value in node_values.items():
            if 'outcome' in node_id.lower() or target_outcome in node_id:
                outcome_probability = value
                outcome_factors.append(node_id)
        
        # If no direct outcome factors, aggregate from all factors
        if not outcome_factors:
            outcome_probability = np.mean(list(node_values.values()))
        
        return {
            'outcome_probability': outcome_probability,
            'contributing_factors': outcome_factors,
            'all_factor_values': node_values
        }
    
    def discover_causal_chains(self, start_factors: List[str], end_outcome: str) -> List[CausalChain]:
        """Discover all causal chains from start factors to end outcome."""
        logger.info(f"Discovering causal chains from {len(start_factors)} factors to {end_outcome}")
        
        discovered_chains = []
        
        for start_factor in start_factors:
            if start_factor not in self.causal_graph:
                continue
            
            # Find all paths from start factor to end outcome
            try:
                paths = list(nx.all_simple_paths(
                    self.causal_graph, 
                    start_factor, 
                    end_outcome, 
                    cutoff=5  # Limit path length
                ))
            except nx.NetworkXNoPath:
                continue
            
            for path in paths:
                chain = self._create_causal_chain(path, start_factor, end_outcome)
                if chain:
                    discovered_chains.append(chain)
        
        # Sort chains by strength
        discovered_chains.sort(key=lambda x: x.total_strength, reverse=True)
        
        # Store top chains
        self.causal_chains.extend(discovered_chains[:10])  # Keep top 10
        
        return discovered_chains
    
    def _create_causal_chain(self, path: List[str], start_factor: str, end_outcome: str) -> Optional[CausalChain]:
        """Create a causal chain from a path."""
        if len(path) < 2:
            return None
        
        factors = []
        relations = []
        total_strength = 1.0
        
        for node_id in path:
            if node_id in self.causal_factors:
                factors.append(self.causal_factors[node_id])
        
        # Build relations along the path
        for i in range(len(path) - 1):
            from_id, to_id = path[i], path[i + 1]
            
            if self.causal_graph.has_edge(from_id, to_id):
                edge_data = self.causal_graph[from_id][to_id]
                relation = edge_data.get('relation')
                
                if relation:
                    relations.append(relation)
                    total_strength *= relation.strength
        
        chain_id = f"chain_{start_factor}_{end_outcome}_{len(path)}"
        
        return CausalChain(
            chain_id=chain_id,
            factors=factors,
            relations=relations,
            total_strength=total_strength,
            legal_outcome=end_outcome,
            supporting_precedents=self._find_supporting_precedents(path)
        )
    
    def _find_supporting_precedents(self, path: List[str]) -> List[str]:
        """Find precedents that support the causal chain."""
        supporting_precedents = []
        
        for node_id in path:
            if node_id in self.causal_factors:
                factor = self.causal_factors[node_id]
                if factor.factor_type == 'precedent':
                    supporting_precedents.append(factor.name)
        
        return supporting_precedents
    
    def get_causal_insights(self) -> Dict[str, Any]:
        """Get insights from causal reasoning analysis."""
        if not self.causal_graph.nodes():
            return {"message": "No causal analysis performed"}
        
        # Graph statistics
        n_nodes = self.causal_graph.number_of_nodes()
        n_edges = self.causal_graph.number_of_edges()
        
        # Most influential factors
        influential_factors = []
        for node_id in self.causal_graph.nodes():
            out_degree = self.causal_graph.out_degree(node_id)
            in_degree = self.causal_graph.in_degree(node_id)
            
            if out_degree > 0:
                influential_factors.append({
                    'factor_id': node_id,
                    'factor_name': self.causal_factors.get(node_id, CausalFactor('', '', '', '', 0, 0, '', '')).name,
                    'out_degree': out_degree,
                    'in_degree': in_degree,
                    'influence_score': out_degree - in_degree
                })
        
        influential_factors.sort(key=lambda x: x['influence_score'], reverse=True)
        
        # Causal discovery statistics
        discovery_stats = {
            'total_discoveries': len(self.causal_discoveries),
            'average_factors_per_discovery': (
                np.mean([d['result']['discovered_factors'] for d in self.causal_discoveries])
                if self.causal_discoveries else 0
            ),
            'average_relations_per_discovery': (
                np.mean([d['result']['discovered_relations'] for d in self.causal_discoveries])
                if self.causal_discoveries else 0
            )
        }
        
        return {
            'causal_graph_stats': {
                'nodes': n_nodes,
                'edges': n_edges,
                'density': n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0
            },
            'most_influential_factors': influential_factors[:5],
            'causal_chains_discovered': len(self.causal_chains),
            'counterfactual_queries_cached': len(self.counterfactual_cache),
            'discovery_statistics': discovery_stats,
            'causal_relation_types': dict(
                zip(*np.unique([r.relation_type.value for r in self.causal_relations.values()], return_counts=True))
            ) if self.causal_relations else {}
        }
    
    def _hash_query(self, query: Dict[str, Any]) -> str:
        """Generate hash for counterfactual query."""
        return hashlib.sha256(json.dumps(query, sort_keys=True).encode()).hexdigest()
    
    def _calculate_causal_effect(self, factual_outcome: Any, counterfactual_outcome: Dict[str, Any]) -> float:
        """Calculate the causal effect between factual and counterfactual outcomes."""
        if factual_outcome is None:
            return 0.0
        
        counterfactual_prob = counterfactual_outcome.get('outcome_probability', 0.5)
        
        if isinstance(factual_outcome, (int, float)):
            return abs(counterfactual_prob - factual_outcome)
        else:
            # For categorical outcomes, use simple difference
            return abs(counterfactual_prob - 0.5)  # Compare to neutral
    
    def _calculate_counterfactual_confidence(self, intervened_graph: nx.DiGraph, 
                                           intervention_factors: List[Dict[str, Any]]) -> float:
        """Calculate confidence in counterfactual reasoning."""
        # Base confidence on graph connectivity and intervention strength
        total_confidence = 0.8  # Base confidence
        
        for intervention in intervention_factors:
            factor_id = intervention.get('factor_id')
            if factor_id in intervened_graph:
                # Lower confidence for factors with many incoming edges (complex dependencies)
                in_degree = intervened_graph.in_degree(factor_id)
                confidence_penalty = min(0.3, in_degree * 0.1)
                total_confidence -= confidence_penalty
        
        return max(0.1, total_confidence)
    
    def _analyze_precedent_support(self, intervention_factors: List[Dict[str, Any]], 
                                 counterfactual_outcome: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze precedent support for counterfactual reasoning."""
        supporting_precedents = []
        conflicting_precedents = []
        
        # This would analyze precedent database in practice
        # For now, return simplified analysis
        
        return {
            'supporting_precedents': supporting_precedents,
            'conflicting_precedents': conflicting_precedents,
            'precedent_confidence': 0.7,
            'novel_scenario_flag': len(supporting_precedents) == 0
        }
    
    def _generate_alternative_scenarios(self, intervention_factors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate alternative intervention scenarios."""
        alternatives = []
        
        # Generate variations of intervention values
        for intervention in intervention_factors:
            factor_id = intervention.get('factor_id')
            current_value = intervention.get('value', 0.5)
            
            # Create alternative values
            for alt_value in [0.0, 0.25, 0.75, 1.0]:
                if abs(alt_value - current_value) > 0.1:
                    alternatives.append({
                        'factor_id': factor_id,
                        'alternative_value': alt_value,
                        'expected_effect': 'higher' if alt_value > current_value else 'lower'
                    })
        
        return alternatives[:5]  # Return top 5 alternatives
    
    def _calculate_robustness_score(self, intervened_graph: nx.DiGraph) -> float:
        """Calculate robustness of counterfactual inference."""
        # Simple robustness based on graph connectivity
        if intervened_graph.number_of_nodes() == 0:
            return 0.0
        
        connectivity = intervened_graph.number_of_edges() / intervened_graph.number_of_nodes()
        robustness = min(1.0, connectivity / 3.0)  # Normalize to [0,1]
        
        return robustness
    
    def _analyze_strength_distribution(self, structure: Dict[str, Any]) -> Dict[str, float]:
        """Analyze distribution of causal strengths."""
        strengths = [edge['strength'] for edge in structure['edges']]
        
        if not strengths:
            return {}
        
        return {
            'mean_strength': np.mean(strengths),
            'std_strength': np.std(strengths),
            'min_strength': np.min(strengths),
            'max_strength': np.max(strengths)
        }
    
    def _identify_temporal_patterns(self, structure: Dict[str, Any]) -> List[str]:
        """Identify temporal patterns in causal structure."""
        patterns = []
        
        # Simple pattern detection
        edge_types = [edge.get('type', 'unknown') for edge in structure['edges']]
        
        if len(edge_types) > 0:
            most_common_type = max(set(edge_types), key=edge_types.count)
            patterns.append(f"Dominant causal type: {most_common_type}")
        
        return patterns
    
    def _calculate_discovery_confidence(self, structure: Dict[str, Any]) -> float:
        """Calculate confidence in causal discovery."""
        n_edges = len(structure['edges'])
        
        if n_edges == 0:
            return 0.0
        
        # Base confidence on number of discovered relationships
        base_confidence = min(0.9, 0.5 + n_edges * 0.05)
        
        return base_confidence


# Global causal inference engine
_causal_inference_engine = None

def get_causal_inference_engine() -> CausalInferenceEngine:
    """Get global causal inference engine instance."""
    global _causal_inference_engine
    if _causal_inference_engine is None:
        _causal_inference_engine = CausalInferenceEngine()
    return _causal_inference_engine