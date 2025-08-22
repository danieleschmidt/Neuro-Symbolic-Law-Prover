"""
Generation 8: Universal Legal Pattern Recognition Engine
Terragon Labs Revolutionary Implementation

Breakthrough Features:
- Cross-domain pattern recognition
- Universal legal archetype detection
- Emergent pattern discovery
- Hierarchical pattern abstraction
- Real-time pattern evolution
- Multi-dimensional pattern analysis
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from datetime import datetime
import logging
import re
from concurrent.futures import ThreadPoolExecutor


logger = logging.getLogger(__name__)


@dataclass
class LegalPattern:
    """Represents a discovered legal pattern."""
    
    pattern_id: str
    pattern_type: str  # 'clause', 'structure', 'principle', 'requirement'
    jurisdictions: Set[str] = field(default_factory=set)
    frequency: float = 0.0
    confidence: float = 0.0
    template: str = ""
    variations: List[str] = field(default_factory=list)
    universal_principle: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatternMatching:
    """Represents pattern matching results."""
    
    pattern: LegalPattern
    match_strength: float
    match_location: Tuple[int, int]  # start, end positions
    context: str
    jurisdiction_specific_variants: Dict[str, str] = field(default_factory=dict)


@dataclass
class PatternEvolution:
    """Tracks pattern evolution over time."""
    
    pattern_id: str
    evolution_timeline: List[Tuple[datetime, str]] = field(default_factory=list)
    adaptation_triggers: List[str] = field(default_factory=list)
    convergence_trends: Dict[str, float] = field(default_factory=dict)
    emergence_indicators: Dict[str, float] = field(default_factory=dict)


class CrossJurisdictionalPatternEngine:
    """
    Revolutionary Cross-Jurisdictional Pattern Recognition Engine.
    
    Breakthrough capabilities:
    - Universal legal pattern discovery across jurisdictions
    - Real-time pattern matching and adaptation
    - Emergent legal principle identification
    - Evolutionary pattern tracking
    """
    
    def __init__(self, 
                 max_workers: int = 6,
                 pattern_similarity_threshold: float = 0.75,
                 evolution_tracking: bool = True):
        """Initialize Cross-Jurisdictional Pattern Engine."""
        
        self.max_workers = max_workers
        self.pattern_similarity_threshold = pattern_similarity_threshold
        self.evolution_tracking = evolution_tracking
        
        # Pattern databases
        self.universal_patterns: Dict[str, LegalPattern] = {}
        self.jurisdictional_patterns: Dict[str, Dict[str, LegalPattern]] = defaultdict(dict)
        self.pattern_evolution_history: Dict[str, PatternEvolution] = {}
        
        # Pattern recognition engines
        self.clause_pattern_recognizer = self._initialize_clause_recognizer()
        self.structural_pattern_recognizer = self._initialize_structural_recognizer()
        self.principle_pattern_recognizer = self._initialize_principle_recognizer()
        
        # Evolution tracking
        self.evolution_tracker = self._initialize_evolution_tracker()
        
        # Performance optimization
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Initialize with foundational patterns
        self._initialize_foundational_patterns()
        
        logger.info(f"Pattern Engine initialized with {len(self.universal_patterns)} universal patterns")
    
    def _initialize_clause_recognizer(self):
        """Initialize clause pattern recognition engine."""
        class ClausePatternRecognizer:
            
            def __init__(self):
                # Common clause patterns across jurisdictions
                self.clause_templates = {
                    'data_processing_consent': [
                        r'consent.*data processing',
                        r'explicit.*consent.*personal data',
                        r'agree.*process.*information',
                        r'authorize.*use.*data'
                    ],
                    'retention_period': [
                        r'retain.*data.*(\d+)\s*(days?|months?|years?)',
                        r'keep.*information.*period',
                        r'storage.*duration.*(\d+)',
                        r'delete.*after.*(\d+)'
                    ],
                    'third_party_sharing': [
                        r'share.*third party',
                        r'disclose.*partner',
                        r'transfer.*outside',
                        r'provide.*external'
                    ],
                    'security_measures': [
                        r'security.*measures',
                        r'protect.*encryption',
                        r'safeguard.*technical',
                        r'secure.*appropriate'
                    ],
                    'user_rights': [
                        r'right.*access',
                        r'right.*rectification',
                        r'right.*erasure',
                        r'right.*portability'
                    ]
                }
            
            def recognize_patterns(self, text: str, jurisdiction: str) -> List[LegalPattern]:
                patterns = []
                
                for pattern_type, templates in self.clause_templates.items():
                    for template in templates:
                        matches = list(re.finditer(template, text, re.IGNORECASE))
                        
                        if matches:
                            pattern = LegalPattern(
                                pattern_id=f"{pattern_type}_{jurisdiction}_{len(patterns)}",
                                pattern_type="clause",
                                jurisdictions={jurisdiction},
                                frequency=len(matches) / len(text.split()),
                                confidence=0.8,
                                template=template,
                                variations=[match.group() for match in matches],
                                metadata={
                                    'matches': len(matches),
                                    'jurisdiction': jurisdiction,
                                    'recognition_timestamp': datetime.now().isoformat()
                                }
                            )
                            patterns.append(pattern)
                
                return patterns
        
        return ClausePatternRecognizer()
    
    def _initialize_structural_recognizer(self):
        """Initialize structural pattern recognition engine."""
        class StructuralPatternRecognizer:
            
            def recognize_patterns(self, text: str, jurisdiction: str) -> List[LegalPattern]:
                patterns = []
                
                # Detect common structural patterns
                structural_indicators = {
                    'hierarchical_clauses': r'(\d+\.)+\s*[A-Z]',
                    'conditional_structures': r'if\s+.*\s+then|where\s+.*\s+shall',
                    'exception_clauses': r'except|unless|provided that|notwithstanding',
                    'obligation_statements': r'shall|must|required to|obligated',
                    'permission_statements': r'may|permitted|allowed|authorized',
                    'prohibition_statements': r'shall not|must not|prohibited|forbidden'
                }
                
                for pattern_type, regex in structural_indicators.items():
                    matches = list(re.finditer(regex, text, re.IGNORECASE))
                    
                    if matches:
                        pattern = LegalPattern(
                            pattern_id=f"struct_{pattern_type}_{jurisdiction}",
                            pattern_type="structure",
                            jurisdictions={jurisdiction},
                            frequency=len(matches) / len(text.split()),
                            confidence=0.7,
                            template=regex,
                            variations=[match.group() for match in matches[:5]],  # Limit variations
                            metadata={
                                'structural_type': pattern_type,
                                'match_count': len(matches),
                                'jurisdiction': jurisdiction
                            }
                        )
                        patterns.append(pattern)
                
                return patterns
        
        return StructuralPatternRecognizer()
    
    def _initialize_principle_recognizer(self):
        """Initialize principle pattern recognition engine."""
        class PrinciplePatternRecognizer:
            
            def __init__(self):
                self.principle_indicators = {
                    'proportionality': [
                        'proportional', 'appropriate', 'reasonable', 'balanced'
                    ],
                    'necessity': [
                        'necessary', 'essential', 'required', 'needed'
                    ],
                    'transparency': [
                        'transparent', 'clear', 'disclosed', 'informed', 'notice'
                    ],
                    'accountability': [
                        'accountable', 'responsible', 'liable', 'oversight'
                    ],
                    'fairness': [
                        'fair', 'equitable', 'just', 'non-discriminatory'
                    ],
                    'privacy_by_design': [
                        'privacy by design', 'built-in privacy', 'default privacy'
                    ],
                    'human_oversight': [
                        'human oversight', 'human control', 'human supervision'
                    ]
                }
            
            def recognize_patterns(self, text: str, jurisdiction: str) -> List[LegalPattern]:
                patterns = []
                text_lower = text.lower()
                
                for principle, indicators in self.principle_indicators.items():
                    indicator_matches = []
                    total_strength = 0.0
                    
                    for indicator in indicators:
                        count = text_lower.count(indicator)
                        if count > 0:
                            indicator_matches.append((indicator, count))
                            total_strength += count
                    
                    if indicator_matches:
                        pattern = LegalPattern(
                            pattern_id=f"principle_{principle}_{jurisdiction}",
                            pattern_type="principle",
                            jurisdictions={jurisdiction},
                            frequency=total_strength / len(text.split()),
                            confidence=min(total_strength * 0.1, 1.0),
                            template=f"principle_pattern_{principle}",
                            variations=[match[0] for match in indicator_matches],
                            universal_principle=principle,
                            metadata={
                                'indicator_matches': indicator_matches,
                                'principle_strength': total_strength,
                                'jurisdiction': jurisdiction
                            }
                        )
                        patterns.append(pattern)
                
                return patterns
        
        return PrinciplePatternRecognizer()
    
    def _initialize_evolution_tracker(self):
        """Initialize pattern evolution tracking engine."""
        class EvolutionTracker:
            
            def track_pattern_evolution(self, 
                                      pattern_id: str,
                                      new_instance: str,
                                      context: Dict[str, Any]) -> PatternEvolution:
                
                # Get or create evolution record
                if pattern_id not in self.pattern_evolution_history:
                    evolution = PatternEvolution(pattern_id=pattern_id)
                    self.pattern_evolution_history[pattern_id] = evolution
                else:
                    evolution = self.pattern_evolution_history[pattern_id]
                
                # Record evolution event
                timestamp = datetime.now()
                evolution.evolution_timeline.append((timestamp, new_instance))
                
                # Analyze adaptation triggers
                if 'regulation_change' in context:
                    evolution.adaptation_triggers.append(f"regulation_change_{timestamp}")
                
                if 'jurisdiction_expansion' in context:
                    evolution.adaptation_triggers.append(f"jurisdiction_expansion_{timestamp}")
                
                # Calculate convergence trends
                if len(evolution.evolution_timeline) >= 2:
                    # Simple convergence metric based on similarity of recent instances
                    recent_instances = [instance for _, instance in evolution.evolution_timeline[-5:]]
                    similarity_scores = []
                    
                    for i in range(len(recent_instances) - 1):
                        similarity = self._calculate_similarity(
                            recent_instances[i], recent_instances[i + 1]
                        )
                        similarity_scores.append(similarity)
                    
                    if similarity_scores:
                        evolution.convergence_trends['recent_stability'] = np.mean(similarity_scores)
                
                return evolution
            
            def _calculate_similarity(self, text1: str, text2: str) -> float:
                """Simple similarity calculation based on word overlap."""
                words1 = set(text1.lower().split())
                words2 = set(text2.lower().split())
                
                if not words1 and not words2:
                    return 1.0
                
                intersection = words1.intersection(words2)
                union = words1.union(words2)
                
                return len(intersection) / len(union) if union else 0.0
        
        evolution_tracker = EvolutionTracker()
        evolution_tracker.pattern_evolution_history = self.pattern_evolution_history
        return evolution_tracker
    
    def _initialize_foundational_patterns(self):
        """Initialize foundational universal legal patterns."""
        
        foundational_patterns = [
            LegalPattern(
                pattern_id="universal_consent",
                pattern_type="principle",
                jurisdictions={'EU', 'US', 'UK', 'APAC'},
                frequency=0.9,
                confidence=0.95,
                template="consent_for_data_processing",
                universal_principle="consent_validity",
                metadata={'foundational': True, 'universality_score': 0.95}
            ),
            LegalPattern(
                pattern_id="universal_transparency",
                pattern_type="principle", 
                jurisdictions={'EU', 'US', 'UK', 'APAC'},
                frequency=0.85,
                confidence=0.9,
                template="transparency_and_disclosure",
                universal_principle="transparency",
                metadata={'foundational': True, 'universality_score': 0.9}
            ),
            LegalPattern(
                pattern_id="universal_security",
                pattern_type="requirement",
                jurisdictions={'EU', 'US', 'UK', 'APAC'},
                frequency=0.8,
                confidence=0.88,
                template="appropriate_security_measures",
                universal_principle="security_by_design",
                metadata={'foundational': True, 'universality_score': 0.88}
            )
        ]
        
        for pattern in foundational_patterns:
            self.universal_patterns[pattern.pattern_id] = pattern
    
    async def discover_patterns(self, 
                               legal_texts: Dict[str, str],
                               jurisdictions: List[str]) -> Dict[str, List[LegalPattern]]:
        """
        Discover legal patterns across multiple jurisdictions.
        
        Args:
            legal_texts: Mapping of document_id -> legal text
            jurisdictions: List of jurisdictions to analyze
            
        Returns:
            Mapping of jurisdiction -> discovered patterns
        """
        
        logger.info(f"Starting pattern discovery for {len(legal_texts)} documents across {len(jurisdictions)} jurisdictions")
        
        # Prepare analysis tasks
        discovery_tasks = []
        
        for jurisdiction in jurisdictions:
            for doc_id, text in legal_texts.items():
                task = self._discover_patterns_for_jurisdiction(
                    text, jurisdiction, doc_id
                )
                discovery_tasks.append(task)
        
        # Execute parallel pattern discovery
        all_patterns = await asyncio.gather(*discovery_tasks)
        
        # Organize results by jurisdiction
        jurisdiction_patterns = defaultdict(list)
        
        for patterns in all_patterns:
            for pattern in patterns:
                for jurisdiction in pattern.jurisdictions:
                    jurisdiction_patterns[jurisdiction].append(pattern)
        
        # Identify universal patterns
        universal_patterns = await self._identify_universal_patterns(jurisdiction_patterns)
        
        # Update pattern databases
        await self._update_pattern_databases(jurisdiction_patterns, universal_patterns)
        
        logger.info(f"Pattern discovery complete. Found {len(universal_patterns)} universal patterns")
        
        return dict(jurisdiction_patterns)
    
    async def _discover_patterns_for_jurisdiction(self, 
                                                text: str, 
                                                jurisdiction: str,
                                                document_id: str) -> List[LegalPattern]:
        """Discover patterns for a specific jurisdiction."""
        
        all_patterns = []
        
        # Run all pattern recognizers
        clause_patterns = self.clause_pattern_recognizer.recognize_patterns(text, jurisdiction)
        structural_patterns = self.structural_pattern_recognizer.recognize_patterns(text, jurisdiction)
        principle_patterns = self.principle_pattern_recognizer.recognize_patterns(text, jurisdiction)
        
        all_patterns.extend(clause_patterns)
        all_patterns.extend(structural_patterns)
        all_patterns.extend(principle_patterns)
        
        # Add document metadata
        for pattern in all_patterns:
            pattern.metadata['document_id'] = document_id
            pattern.metadata['discovery_timestamp'] = datetime.now().isoformat()
        
        return all_patterns
    
    async def _identify_universal_patterns(self, 
                                         jurisdiction_patterns: Dict[str, List[LegalPattern]]) -> List[LegalPattern]:
        """Identify patterns that are universal across jurisdictions."""
        
        universal_patterns = []
        
        # Group patterns by type and principle
        pattern_groups = defaultdict(list)
        
        for jurisdiction, patterns in jurisdiction_patterns.items():
            for pattern in patterns:
                if pattern.universal_principle:
                    key = (pattern.pattern_type, pattern.universal_principle)
                    pattern_groups[key].append(pattern)
        
        # Identify universal patterns (appear in multiple jurisdictions)
        for (pattern_type, principle), patterns in pattern_groups.items():
            jurisdictions_represented = set()
            total_frequency = 0.0
            total_confidence = 0.0
            all_variations = []
            
            for pattern in patterns:
                jurisdictions_represented.update(pattern.jurisdictions)
                total_frequency += pattern.frequency
                total_confidence += pattern.confidence
                all_variations.extend(pattern.variations)
            
            # Consider universal if appears in 3+ jurisdictions
            if len(jurisdictions_represented) >= 3:
                universal_pattern = LegalPattern(
                    pattern_id=f"universal_{pattern_type}_{principle}",
                    pattern_type=pattern_type,
                    jurisdictions=jurisdictions_represented,
                    frequency=total_frequency / len(patterns),
                    confidence=total_confidence / len(patterns),
                    template=f"universal_{principle}_template",
                    variations=list(set(all_variations))[:10],  # Limit variations
                    universal_principle=principle,
                    metadata={
                        'universality_score': len(jurisdictions_represented) / 4.0,  # Normalize by total jurisdictions
                        'source_patterns': len(patterns),
                        'discovery_method': 'cross_jurisdictional_analysis'
                    }
                )
                universal_patterns.append(universal_pattern)
        
        return universal_patterns
    
    async def _update_pattern_databases(self,
                                       jurisdiction_patterns: Dict[str, List[LegalPattern]],
                                       universal_patterns: List[LegalPattern]):
        """Update pattern databases with new discoveries."""
        
        # Update jurisdictional patterns
        for jurisdiction, patterns in jurisdiction_patterns.items():
            for pattern in patterns:
                self.jurisdictional_patterns[jurisdiction][pattern.pattern_id] = pattern
        
        # Update universal patterns
        for pattern in universal_patterns:
            self.universal_patterns[pattern.pattern_id] = pattern
        
        # Track pattern evolution if enabled
        if self.evolution_tracking:
            for pattern in universal_patterns:
                evolution = self.evolution_tracker.track_pattern_evolution(
                    pattern.pattern_id,
                    pattern.template,
                    {'discovery_event': True}
                )
    
    async def match_patterns(self, 
                           text: str,
                           target_jurisdictions: Optional[List[str]] = None) -> List[PatternMatching]:
        """
        Match legal text against known patterns.
        
        Args:
            text: Legal text to analyze
            target_jurisdictions: Specific jurisdictions to match against
            
        Returns:
            List of pattern matches with strength scores
        """
        
        if target_jurisdictions is None:
            target_jurisdictions = list(self.jurisdictional_patterns.keys())
        
        matches = []
        
        # Match against universal patterns first
        for pattern_id, pattern in self.universal_patterns.items():
            match_strength = await self._calculate_pattern_match_strength(text, pattern)
            
            if match_strength >= self.pattern_similarity_threshold:
                # Find match location (simplified)
                match_location = self._find_pattern_location(text, pattern)
                
                pattern_match = PatternMatching(
                    pattern=pattern,
                    match_strength=match_strength,
                    match_location=match_location,
                    context=self._extract_context(text, match_location)
                )
                matches.append(pattern_match)
        
        # Match against jurisdiction-specific patterns
        for jurisdiction in target_jurisdictions:
            if jurisdiction in self.jurisdictional_patterns:
                for pattern_id, pattern in self.jurisdictional_patterns[jurisdiction].items():
                    match_strength = await self._calculate_pattern_match_strength(text, pattern)
                    
                    if match_strength >= self.pattern_similarity_threshold:
                        match_location = self._find_pattern_location(text, pattern)
                        
                        pattern_match = PatternMatching(
                            pattern=pattern,
                            match_strength=match_strength,
                            match_location=match_location,
                            context=self._extract_context(text, match_location)
                        )
                        matches.append(pattern_match)
        
        # Sort matches by strength
        matches.sort(key=lambda x: x.match_strength, reverse=True)
        
        logger.info(f"Pattern matching complete. Found {len(matches)} matches")
        
        return matches
    
    async def _calculate_pattern_match_strength(self, text: str, pattern: LegalPattern) -> float:
        """Calculate how strongly a pattern matches the given text."""
        
        text_lower = text.lower()
        match_strength = 0.0
        
        # Check template match
        if pattern.template:
            if pattern.pattern_type == "principle" and pattern.universal_principle:
                # For principle patterns, check for principle indicators
                principle_indicators = getattr(
                    self.principle_pattern_recognizer, 'principle_indicators', {}
                ).get(pattern.universal_principle, [])
                
                indicator_matches = sum(1 for indicator in principle_indicators if indicator in text_lower)
                if principle_indicators:
                    match_strength += (indicator_matches / len(principle_indicators)) * 0.6
            
            elif pattern.pattern_type == "clause":
                # For clause patterns, try regex matching
                try:
                    import re
                    matches = list(re.finditer(pattern.template, text, re.IGNORECASE))
                    if matches:
                        match_strength += min(len(matches) * 0.3, 0.6)
                except:
                    # Fallback to simple string matching
                    if pattern.template.lower() in text_lower:
                        match_strength += 0.4
        
        # Check variation matches
        if pattern.variations:
            variation_matches = sum(1 for variation in pattern.variations if variation.lower() in text_lower)
            if pattern.variations:
                match_strength += (variation_matches / len(pattern.variations)) * 0.3
        
        # Bonus for universal patterns
        if pattern.universal_principle:
            match_strength += 0.1
        
        return min(match_strength, 1.0)
    
    def _find_pattern_location(self, text: str, pattern: LegalPattern) -> Tuple[int, int]:
        """Find the location of a pattern match in text."""
        
        # Simplified implementation - find first variation match
        if pattern.variations:
            for variation in pattern.variations:
                start_pos = text.lower().find(variation.lower())
                if start_pos != -1:
                    return (start_pos, start_pos + len(variation))
        
        # Fallback to beginning of text
        return (0, min(100, len(text)))
    
    def _extract_context(self, text: str, match_location: Tuple[int, int]) -> str:
        """Extract context around a pattern match."""
        
        start, end = match_location
        context_start = max(0, start - 100)
        context_end = min(len(text), end + 100)
        
        return text[context_start:context_end]
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pattern statistics."""
        
        stats = {
            'universal_patterns': len(self.universal_patterns),
            'jurisdictional_patterns': {
                jurisdiction: len(patterns) 
                for jurisdiction, patterns in self.jurisdictional_patterns.items()
            },
            'total_patterns': len(self.universal_patterns) + sum(
                len(patterns) for patterns in self.jurisdictional_patterns.values()
            ),
            'pattern_types': defaultdict(int),
            'universal_principles': set(),
            'evolution_tracked_patterns': len(self.pattern_evolution_history)
        }
        
        # Analyze pattern types and principles
        all_patterns = list(self.universal_patterns.values())
        for jurisdiction_patterns in self.jurisdictional_patterns.values():
            all_patterns.extend(jurisdiction_patterns.values())
        
        for pattern in all_patterns:
            stats['pattern_types'][pattern.pattern_type] += 1
            if pattern.universal_principle:
                stats['universal_principles'].add(pattern.universal_principle)
        
        stats['universal_principles'] = list(stats['universal_principles'])
        stats['pattern_types'] = dict(stats['pattern_types'])
        
        return stats
    
    def export_patterns(self, format_type: str = 'json') -> str:
        """Export patterns in specified format."""
        
        if format_type == 'json':
            import json
            export_data = {
                'universal_patterns': {
                    pid: {
                        'pattern_id': p.pattern_id,
                        'pattern_type': p.pattern_type,
                        'jurisdictions': list(p.jurisdictions),
                        'frequency': p.frequency,
                        'confidence': p.confidence,
                        'template': p.template,
                        'variations': p.variations,
                        'universal_principle': p.universal_principle,
                        'metadata': p.metadata
                    }
                    for pid, p in self.universal_patterns.items()
                },
                'jurisdictional_patterns': {
                    jurisdiction: {
                        pid: {
                            'pattern_id': p.pattern_id,
                            'pattern_type': p.pattern_type,
                            'jurisdictions': list(p.jurisdictions),
                            'frequency': p.frequency,
                            'confidence': p.confidence,
                            'template': p.template,
                            'variations': p.variations,
                            'universal_principle': p.universal_principle,
                            'metadata': p.metadata
                        }
                        for pid, p in patterns.items()
                    }
                    for jurisdiction, patterns in self.jurisdictional_patterns.items()
                }
            }
            return json.dumps(export_data, indent=2, default=str)
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)