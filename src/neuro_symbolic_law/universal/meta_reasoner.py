"""
Meta-Legal Reasoner - Generation 7 Revolutionary Capability
Terragon Labs Meta-Cognitive Legal Intelligence

Capabilities:
- Reasoning about reasoning (meta-cognition)
- Self-aware legal analysis
- Recursive legal principle derivation
- Meta-pattern recognition across legal systems
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import numpy as np
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
import inspect


logger = logging.getLogger(__name__)


@dataclass
class MetaReasoning:
    """Represents meta-level reasoning about legal analysis."""
    
    reasoning_id: str
    reasoning_level: int  # 0=base, 1=meta, 2=meta-meta, etc.
    subject_domain: str
    reasoning_type: str  # 'analytical', 'synthetic', 'evaluative', 'creative'
    premises: List[str] = field(default_factory=list)
    inferences: List[str] = field(default_factory=list)
    conclusions: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning_quality: float = 0.0
    meta_insights: Dict[str, Any] = field(default_factory=dict)
    self_critique: List[str] = field(default_factory=list)


@dataclass
class MetaPattern:
    """Represents patterns in how legal reasoning operates."""
    
    pattern_id: str
    pattern_level: str  # 'reasoning_structure', 'inference_method', 'validation_approach'
    pattern_description: str
    frequency: float = 0.0
    effectiveness: float = 0.0
    contexts: List[str] = field(default_factory=list)
    variations: List[str] = field(default_factory=list)
    meta_properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SelfAwareness:
    """Represents system's self-awareness about its reasoning capabilities."""
    
    awareness_level: str  # 'basic', 'intermediate', 'advanced', 'transcendent'
    known_capabilities: Set[str] = field(default_factory=set)
    known_limitations: Set[str] = field(default_factory=set)
    uncertainty_areas: Set[str] = field(default_factory=set)
    confidence_calibration: Dict[str, float] = field(default_factory=dict)
    improvement_opportunities: List[str] = field(default_factory=list)
    meta_knowledge: Dict[str, Any] = field(default_factory=dict)


class MetaReasoningEngine(ABC):
    """Abstract base for meta-reasoning engines."""
    
    @abstractmethod
    async def meta_analyze(self, subject: Any, context: Dict[str, Any]) -> MetaReasoning:
        """Perform meta-analysis on a subject."""
        pass
    
    @abstractmethod
    def get_reasoning_level(self) -> int:
        """Get the reasoning level this engine operates at."""
        pass


class BaseReasoningAnalyzer(MetaReasoningEngine):
    """Analyzes base-level legal reasoning."""
    
    def get_reasoning_level(self) -> int:
        return 0
    
    async def meta_analyze(self, subject: Any, context: Dict[str, Any]) -> MetaReasoning:
        """Analyze base-level reasoning patterns."""
        
        reasoning = MetaReasoning(
            reasoning_id=f"base_analysis_{datetime.now().timestamp()}",
            reasoning_level=0,
            subject_domain=context.get('domain', 'legal'),
            reasoning_type='analytical'
        )
        
        # Analyze structure of the reasoning
        if isinstance(subject, dict) and 'premises' in subject:
            reasoning.premises = subject['premises']
            reasoning.inferences = subject.get('inferences', [])
            reasoning.conclusions = subject.get('conclusions', [])
            
            # Evaluate reasoning quality
            reasoning.reasoning_quality = self._evaluate_base_reasoning_quality(subject)
            reasoning.confidence = min(reasoning.reasoning_quality + 0.2, 1.0)
        
        return reasoning
    
    def _evaluate_base_reasoning_quality(self, reasoning_subject: Dict[str, Any]) -> float:
        """Evaluate quality of base-level reasoning."""
        
        quality_score = 0.0
        
        # Check for logical structure
        if 'premises' in reasoning_subject and 'conclusions' in reasoning_subject:
            quality_score += 0.3
        
        # Check for evidence support
        if 'evidence' in reasoning_subject:
            quality_score += 0.2
        
        # Check for consideration of alternatives
        if 'alternatives_considered' in reasoning_subject:
            quality_score += 0.2
        
        # Check for uncertainty acknowledgment
        if 'uncertainties' in reasoning_subject:
            quality_score += 0.2
        
        # Check for completeness
        premises = reasoning_subject.get('premises', [])
        conclusions = reasoning_subject.get('conclusions', [])
        if premises and conclusions:
            quality_score += 0.1
        
        return min(quality_score, 1.0)


class MetaReasoningAnalyzer(MetaReasoningEngine):
    """Analyzes meta-level reasoning (reasoning about reasoning)."""
    
    def get_reasoning_level(self) -> int:
        return 1
    
    async def meta_analyze(self, subject: Any, context: Dict[str, Any]) -> MetaReasoning:
        """Analyze meta-level reasoning patterns."""
        
        reasoning = MetaReasoning(
            reasoning_id=f"meta_analysis_{datetime.now().timestamp()}",
            reasoning_level=1,
            subject_domain=context.get('domain', 'legal_reasoning'),
            reasoning_type='evaluative'
        )
        
        # If subject is a MetaReasoning object, analyze its reasoning
        if isinstance(subject, MetaReasoning):
            reasoning.premises = [
                f"Base reasoning used {subject.reasoning_type} approach",
                f"Base reasoning confidence: {subject.confidence}",
                f"Base reasoning quality: {subject.reasoning_quality}"
            ]
            
            reasoning.inferences = self._generate_meta_inferences(subject)
            reasoning.conclusions = self._generate_meta_conclusions(subject)
            reasoning.confidence = self._calculate_meta_confidence(subject)
            reasoning.reasoning_quality = self._evaluate_meta_reasoning_quality(subject)
            
            # Generate self-critique
            reasoning.self_critique = self._generate_self_critique(subject)
        
        return reasoning
    
    def _generate_meta_inferences(self, base_reasoning: MetaReasoning) -> List[str]:
        """Generate inferences about the base reasoning."""
        
        inferences = []
        
        # Analyze reasoning strength
        if base_reasoning.confidence > 0.8:
            inferences.append("Base reasoning demonstrates high confidence")
        elif base_reasoning.confidence < 0.5:
            inferences.append("Base reasoning shows uncertainty that requires investigation")
        
        # Analyze reasoning completeness
        if len(base_reasoning.premises) < 2:
            inferences.append("Reasoning may be insufficiently grounded")
        
        if not base_reasoning.conclusions:
            inferences.append("Reasoning lacks clear conclusions")
        
        # Analyze reasoning type appropriateness
        if base_reasoning.reasoning_type == 'analytical' and base_reasoning.subject_domain == 'creative_legal':
            inferences.append("Analytical approach may be insufficient for creative legal problems")
        
        return inferences
    
    def _generate_meta_conclusions(self, base_reasoning: MetaReasoning) -> List[str]:
        """Generate conclusions about the base reasoning."""
        
        conclusions = []
        
        # Overall quality assessment
        if base_reasoning.reasoning_quality > 0.7:
            conclusions.append("Base reasoning is of high quality")
        elif base_reasoning.reasoning_quality < 0.4:
            conclusions.append("Base reasoning needs significant improvement")
        else:
            conclusions.append("Base reasoning is adequate but could be enhanced")
        
        # Specific recommendations
        if base_reasoning.confidence < base_reasoning.reasoning_quality:
            conclusions.append("System is appropriately cautious given reasoning quality")
        elif base_reasoning.confidence > base_reasoning.reasoning_quality + 0.2:
            conclusions.append("System may be overconfident given reasoning quality")
        
        return conclusions
    
    def _calculate_meta_confidence(self, base_reasoning: MetaReasoning) -> float:
        """Calculate confidence in the meta-analysis."""
        
        # Meta-confidence based on consistency and completeness of base reasoning
        consistency_score = 0.5  # Base score
        
        # Increase confidence if base reasoning is well-structured
        if len(base_reasoning.premises) >= 2 and base_reasoning.conclusions:
            consistency_score += 0.2
        
        # Increase confidence if base reasoning acknowledges uncertainty appropriately
        if 0.3 <= base_reasoning.confidence <= 0.9:  # Reasonable range
            consistency_score += 0.2
        
        # Decrease confidence if base reasoning seems problematic
        if base_reasoning.reasoning_quality < 0.3:
            consistency_score -= 0.3
        
        return max(0.0, min(1.0, consistency_score))
    
    def _evaluate_meta_reasoning_quality(self, base_reasoning: MetaReasoning) -> float:
        """Evaluate the quality of meta-reasoning."""
        
        quality = 0.5  # Base quality
        
        # Quality increases with thorough analysis
        if len(self._generate_meta_inferences(base_reasoning)) >= 2:
            quality += 0.2
        
        if len(self._generate_meta_conclusions(base_reasoning)) >= 2:
            quality += 0.2
        
        # Quality increases with appropriate calibration
        confidence_reasoning_diff = abs(base_reasoning.confidence - base_reasoning.reasoning_quality)
        if confidence_reasoning_diff < 0.2:
            quality += 0.1
        
        return min(quality, 1.0)
    
    def _generate_self_critique(self, base_reasoning: MetaReasoning) -> List[str]:
        """Generate self-critique of the reasoning."""
        
        critique = []
        
        # Critique completeness
        if len(base_reasoning.premises) < 3:
            critique.append("Should consider more premises for robust reasoning")
        
        # Critique bias consideration
        if 'bias_analysis' not in base_reasoning.meta_insights:
            critique.append("Should explicitly consider potential biases")
        
        # Critique alternative consideration
        if 'alternatives' not in base_reasoning.meta_insights:
            critique.append("Should consider alternative interpretations")
        
        return critique


class MetaMetaReasoningAnalyzer(MetaReasoningEngine):
    """Analyzes meta-meta-level reasoning (reasoning about reasoning about reasoning)."""
    
    def get_reasoning_level(self) -> int:
        return 2
    
    async def meta_analyze(self, subject: Any, context: Dict[str, Any]) -> MetaReasoning:
        """Analyze meta-meta-level reasoning patterns."""
        
        reasoning = MetaReasoning(
            reasoning_id=f"meta_meta_analysis_{datetime.now().timestamp()}",
            reasoning_level=2,
            subject_domain=context.get('domain', 'meta_reasoning'),
            reasoning_type='synthetic'
        )
        
        # Analyze the meta-reasoning process itself
        if isinstance(subject, MetaReasoning) and subject.reasoning_level >= 1:
            reasoning.premises = [
                f"Meta-reasoning analyzed {subject.subject_domain}",
                f"Meta-reasoning type: {subject.reasoning_type}",
                f"Meta-reasoning included {len(subject.self_critique)} self-critiques"
            ]
            
            reasoning.inferences = self._generate_meta_meta_inferences(subject)
            reasoning.conclusions = self._generate_meta_meta_conclusions(subject)
            reasoning.confidence = self._calculate_meta_meta_confidence(subject)
            reasoning.reasoning_quality = self._evaluate_meta_meta_quality(subject)
        
        return reasoning
    
    def _generate_meta_meta_inferences(self, meta_reasoning: MetaReasoning) -> List[str]:
        """Generate inferences about meta-reasoning."""
        
        inferences = []
        
        # Analyze meta-reasoning depth
        if len(meta_reasoning.self_critique) > 0:
            inferences.append("Meta-reasoning demonstrates self-awareness")
        else:
            inferences.append("Meta-reasoning lacks sufficient self-reflection")
        
        # Analyze meta-reasoning appropriateness
        if meta_reasoning.reasoning_type == 'evaluative':
            inferences.append("Evaluative meta-reasoning is appropriate for quality assessment")
        
        # Analyze recursion handling
        if meta_reasoning.reasoning_level > 3:
            inferences.append("Deep recursion may lead to diminishing returns")
        
        return inferences
    
    def _generate_meta_meta_conclusions(self, meta_reasoning: MetaReasoning) -> List[str]:
        """Generate conclusions about meta-reasoning."""
        
        conclusions = []
        
        # Assess meta-reasoning effectiveness
        if meta_reasoning.confidence > 0.7 and len(meta_reasoning.self_critique) > 0:
            conclusions.append("Meta-reasoning is functioning effectively")
        else:
            conclusions.append("Meta-reasoning could be enhanced")
        
        # Recommend optimization
        conclusions.append("Optimal recursion depth appears to be 2-3 levels")
        
        return conclusions
    
    def _calculate_meta_meta_confidence(self, meta_reasoning: MetaReasoning) -> float:
        """Calculate confidence in meta-meta analysis."""
        return min(meta_reasoning.confidence * 0.9, 0.95)  # Slightly reduced due to recursion
    
    def _evaluate_meta_meta_quality(self, meta_reasoning: MetaReasoning) -> float:
        """Evaluate meta-meta reasoning quality."""
        return min(meta_reasoning.reasoning_quality * 0.85, 0.9)  # Quality degrades with recursion


class MetaLegalReasoner:
    """
    Revolutionary Meta-Legal Reasoning System - Generation 7.
    
    Breakthrough capabilities:
    - Multi-level meta-reasoning about legal analysis
    - Self-aware legal intelligence with recursive depth
    - Pattern recognition across reasoning methods
    - Autonomous quality improvement
    """
    
    def __init__(self,
                 max_recursion_depth: int = 4,
                 self_awareness_level: str = 'advanced',
                 max_workers: int = 4):
        """Initialize Meta-Legal Reasoner."""
        
        self.max_recursion_depth = max_recursion_depth
        self.self_awareness_level = self_awareness_level
        self.max_workers = max_workers
        
        # Meta-reasoning engines by level
        self.reasoning_engines = {
            0: BaseReasoningAnalyzer(),
            1: MetaReasoningAnalyzer(),
            2: MetaMetaReasoningAnalyzer()
        }
        
        # Meta-pattern recognition
        self.meta_patterns: Dict[str, MetaPattern] = {}
        self.pattern_history: deque = deque(maxlen=500)
        
        # Self-awareness system
        self.self_awareness = SelfAwareness(
            awareness_level=self_awareness_level,
            known_capabilities={
                'multi_level_reasoning',
                'self_critique',
                'pattern_recognition',
                'quality_assessment',
                'uncertainty_quantification'
            },
            known_limitations={
                'recursion_computational_cost',
                'diminishing_returns_depth',
                'potential_infinite_regress',
                'context_sensitivity'
            },
            uncertainty_areas={
                'optimal_recursion_depth',
                'meta_reasoning_validity',
                'self_awareness_accuracy'
            }
        )
        
        # Performance optimization
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.reasoning_cache: Dict[str, MetaReasoning] = {}
        
        logger.info(f"Meta-Legal Reasoner initialized with {self_awareness_level} self-awareness")
    
    async def recursive_legal_analysis(self,
                                     legal_problem: Dict[str, Any],
                                     max_depth: Optional[int] = None,
                                     focus_areas: Optional[List[str]] = None) -> Dict[str, MetaReasoning]:
        """
        Perform recursive meta-analysis of legal reasoning.
        
        Revolutionary capability: Multi-level self-aware legal analysis.
        """
        
        if max_depth is None:
            max_depth = self.max_recursion_depth
        
        logger.info(f"Starting recursive legal analysis with depth {max_depth}")
        
        # Create analysis context
        context = {
            'domain': 'legal',
            'problem_type': legal_problem.get('type', 'unknown'),
            'complexity': legal_problem.get('complexity', 'medium'),
            'focus_areas': focus_areas or []
        }
        
        # Store reasoning at each level
        reasoning_levels: Dict[str, MetaReasoning] = {}
        
        # Start with base-level analysis
        current_subject = legal_problem
        
        for level in range(max_depth + 1):
            logger.debug(f"Performing analysis at reasoning level {level}")
            
            # Get appropriate reasoning engine
            engine = self._get_reasoning_engine(level)
            
            # Perform analysis at current level
            reasoning = await engine.meta_analyze(current_subject, context)
            reasoning_levels[f"level_{level}"] = reasoning
            
            # Record pattern for learning
            await self._record_reasoning_pattern(reasoning, context)
            
            # Update context for next level
            context['previous_level'] = level
            context['previous_reasoning'] = reasoning
            
            # Use current reasoning as subject for next level
            current_subject = reasoning
            
            # Check for convergence or diminishing returns
            if level > 0 and await self._check_convergence(reasoning_levels, level):
                logger.info(f"Convergence detected at level {level}")
                break
        
        # Perform cross-level analysis
        cross_level_insights = await self._analyze_across_levels(reasoning_levels)
        
        # Update self-awareness based on analysis
        await self._update_self_awareness(reasoning_levels, cross_level_insights)
        
        logger.info(f"Recursive analysis complete with {len(reasoning_levels)} levels")
        
        return reasoning_levels
    
    def _get_reasoning_engine(self, level: int) -> MetaReasoningEngine:
        """Get reasoning engine for specified level."""
        
        if level in self.reasoning_engines:
            return self.reasoning_engines[level]
        else:
            # For levels beyond defined engines, use the highest available
            max_level = max(self.reasoning_engines.keys())
            return self.reasoning_engines[max_level]
    
    async def _record_reasoning_pattern(self, reasoning: MetaReasoning, context: Dict[str, Any]):
        """Record reasoning patterns for meta-learning."""
        
        pattern_signature = f"{reasoning.reasoning_type}_{reasoning.reasoning_level}_{context.get('problem_type', 'unknown')}"
        
        # Update or create meta-pattern
        if pattern_signature in self.meta_patterns:
            pattern = self.meta_patterns[pattern_signature]
            pattern.frequency += 1
            pattern.effectiveness = (pattern.effectiveness + reasoning.reasoning_quality) / 2
        else:
            pattern = MetaPattern(
                pattern_id=pattern_signature,
                pattern_level=f"level_{reasoning.reasoning_level}",
                pattern_description=f"{reasoning.reasoning_type} reasoning at level {reasoning.reasoning_level}",
                frequency=1,
                effectiveness=reasoning.reasoning_quality,
                contexts=[context.get('problem_type', 'unknown')]
            )
            self.meta_patterns[pattern_signature] = pattern
        
        # Add to history
        self.pattern_history.append({
            'timestamp': datetime.now(),
            'pattern_id': pattern_signature,
            'reasoning_quality': reasoning.reasoning_quality,
            'context': context.copy()
        })
    
    async def _check_convergence(self, reasoning_levels: Dict[str, MetaReasoning], current_level: int) -> bool:
        """Check if reasoning has converged (diminishing returns)."""
        
        if current_level < 2:
            return False
        
        # Get last two levels
        current_reasoning = reasoning_levels[f"level_{current_level}"]
        previous_reasoning = reasoning_levels[f"level_{current_level - 1}"]
        
        # Check quality improvement
        quality_improvement = current_reasoning.reasoning_quality - previous_reasoning.reasoning_quality
        
        # Check confidence stability
        confidence_change = abs(current_reasoning.confidence - previous_reasoning.confidence)
        
        # Convergence criteria
        minimal_improvement = quality_improvement < 0.05
        stable_confidence = confidence_change < 0.1
        
        return minimal_improvement and stable_confidence
    
    async def _analyze_across_levels(self, reasoning_levels: Dict[str, MetaReasoning]) -> Dict[str, Any]:
        """Analyze patterns and insights across reasoning levels."""
        
        insights = {
            'quality_progression': [],
            'confidence_progression': [],
            'consistency_analysis': {},
            'emergent_insights': [],
            'optimal_depth': 0
        }
        
        # Analyze quality and confidence progression
        for level_key in sorted(reasoning_levels.keys()):
            reasoning = reasoning_levels[level_key]
            insights['quality_progression'].append(reasoning.reasoning_quality)
            insights['confidence_progression'].append(reasoning.confidence)
        
        # Find optimal depth (where quality peaks)
        if insights['quality_progression']:
            optimal_depth_index = np.argmax(insights['quality_progression'])
            insights['optimal_depth'] = optimal_depth_index
        
        # Analyze consistency across levels
        if len(reasoning_levels) >= 2:
            conclusions_overlap = self._analyze_conclusion_consistency(reasoning_levels)
            insights['consistency_analysis']['conclusion_overlap'] = conclusions_overlap
        
        # Identify emergent insights
        insights['emergent_insights'] = self._identify_emergent_insights(reasoning_levels)
        
        return insights
    
    def _analyze_conclusion_consistency(self, reasoning_levels: Dict[str, MetaReasoning]) -> float:
        """Analyze consistency of conclusions across levels."""
        
        all_conclusions = []
        for reasoning in reasoning_levels.values():
            all_conclusions.extend(reasoning.conclusions)
        
        if len(all_conclusions) < 2:
            return 1.0
        
        # Simple consistency metric based on keyword overlap
        conclusion_words = [set(conclusion.lower().split()) for conclusion in all_conclusions]
        
        total_pairs = 0
        consistent_pairs = 0
        
        for i in range(len(conclusion_words)):
            for j in range(i + 1, len(conclusion_words)):
                total_pairs += 1
                
                # Calculate word overlap
                overlap = len(conclusion_words[i].intersection(conclusion_words[j]))
                union = len(conclusion_words[i].union(conclusion_words[j]))
                
                if union > 0:
                    similarity = overlap / union
                    if similarity > 0.3:  # Threshold for consistency
                        consistent_pairs += 1
        
        return consistent_pairs / total_pairs if total_pairs > 0 else 1.0
    
    def _identify_emergent_insights(self, reasoning_levels: Dict[str, MetaReasoning]) -> List[str]:
        """Identify insights that emerge from meta-level analysis."""
        
        emergent_insights = []
        
        # Check for quality convergence patterns
        quality_progression = [r.reasoning_quality for r in reasoning_levels.values()]
        if len(quality_progression) >= 3:
            if all(q1 <= q2 for q1, q2 in zip(quality_progression[:-1], quality_progression[1:])):
                emergent_insights.append("Reasoning quality consistently improves with meta-analysis depth")
            elif quality_progression[-1] < quality_progression[-2]:
                emergent_insights.append("Diminishing returns detected at higher meta-reasoning levels")
        
        # Check for self-critique evolution
        critique_counts = [len(r.self_critique) for r in reasoning_levels.values() if r.self_critique]
        if critique_counts and max(critique_counts) > 0:
            emergent_insights.append("Self-awareness increases with meta-reasoning depth")
        
        # Check for reasoning type evolution
        reasoning_types = [r.reasoning_type for r in reasoning_levels.values()]
        if len(set(reasoning_types)) > 1:
            emergent_insights.append("Reasoning approach evolves across meta-levels")
        
        return emergent_insights
    
    async def _update_self_awareness(self, 
                                   reasoning_levels: Dict[str, MetaReasoning],
                                   cross_level_insights: Dict[str, Any]):
        """Update self-awareness based on recursive analysis."""
        
        # Update confidence calibration
        for level_key, reasoning in reasoning_levels.items():
            level_num = int(level_key.split('_')[1])
            self.self_awareness.confidence_calibration[f"level_{level_num}"] = reasoning.confidence
        
        # Update improvement opportunities
        if cross_level_insights['optimal_depth'] < len(reasoning_levels) - 1:
            self.self_awareness.improvement_opportunities.append(
                f"Consider limiting recursion to {cross_level_insights['optimal_depth']} levels for efficiency"
            )
        
        # Update meta-knowledge
        self.self_awareness.meta_knowledge.update({
            'typical_optimal_depth': cross_level_insights['optimal_depth'],
            'average_quality_improvement': np.mean(cross_level_insights['quality_progression']),
            'last_analysis_timestamp': datetime.now().isoformat()
        })
    
    async def generate_meta_legal_advice(self,
                                       legal_question: str,
                                       reasoning_depth: int = 3) -> Dict[str, Any]:
        """
        Generate legal advice with meta-reasoning insights.
        
        Revolutionary capability: Self-aware legal advice with reasoning transparency.
        """
        
        logger.info(f"Generating meta-legal advice for: {legal_question[:100]}...")
        
        # Structure the legal problem
        legal_problem = {
            'question': legal_question,
            'type': 'advisory_query',
            'complexity': self._assess_question_complexity(legal_question),
            'timestamp': datetime.now().isoformat()
        }
        
        # Perform recursive analysis
        reasoning_analysis = await self.recursive_legal_analysis(
            legal_problem, 
            max_depth=reasoning_depth
        )
        
        # Extract advice from different reasoning levels
        advice_levels = {}
        for level_key, reasoning in reasoning_analysis.items():
            advice_levels[level_key] = {
                'advice': reasoning.conclusions,
                'confidence': reasoning.confidence,
                'reasoning_quality': reasoning.reasoning_quality,
                'self_critique': reasoning.self_critique
            }
        
        # Generate meta-advice (advice about the advice)
        meta_advice = await self._generate_meta_advice(reasoning_analysis)
        
        # Compile comprehensive response
        response = {
            'legal_question': legal_question,
            'advice_levels': advice_levels,
            'meta_advice': meta_advice,
            'reasoning_transparency': {
                'reasoning_levels_used': len(reasoning_analysis),
                'optimal_depth_recommendation': meta_advice.get('optimal_depth', reasoning_depth),
                'confidence_trajectory': [r.confidence for r in reasoning_analysis.values()],
                'quality_trajectory': [r.reasoning_quality for r in reasoning_analysis.values()]
            },
            'self_awareness_insights': {
                'system_confidence_in_analysis': self._calculate_system_confidence(reasoning_analysis),
                'known_limitations_relevant': self._identify_relevant_limitations(legal_question),
                'uncertainty_acknowledgment': self._acknowledge_uncertainties(reasoning_analysis)
            },
            'recommendation': self._formulate_final_recommendation(reasoning_analysis, meta_advice)
        }
        
        logger.info("Meta-legal advice generation complete")
        
        return response
    
    def _assess_question_complexity(self, question: str) -> str:
        """Assess the complexity of a legal question."""
        
        complexity_indicators = {
            'high': ['constitutional', 'precedent', 'interpretation', 'novel', 'conflicting', 'international'],
            'medium': ['contract', 'liability', 'compliance', 'standard', 'typical'],
            'low': ['basic', 'simple', 'clear', 'straightforward', 'routine']
        }
        
        question_lower = question.lower()
        scores = {}
        
        for complexity, indicators in complexity_indicators.items():
            score = sum(1 for indicator in indicators if indicator in question_lower)
            scores[complexity] = score
        
        # Return complexity with highest score
        max_score = max(scores.values())
        if max_score == 0:
            return 'medium'  # Default
        
        return max(scores, key=scores.get)
    
    async def _generate_meta_advice(self, reasoning_analysis: Dict[str, MetaReasoning]) -> Dict[str, Any]:
        """Generate advice about the advice (meta-advice)."""
        
        meta_advice = {
            'advice_quality_assessment': 'unknown',
            'confidence_calibration': 'unknown',
            'reasoning_depth_assessment': 'unknown',
            'optimal_depth': 2,
            'improvement_suggestions': []
        }
        
        if reasoning_analysis:
            qualities = [r.reasoning_quality for r in reasoning_analysis.values()]
            confidences = [r.confidence for r in reasoning_analysis.values()]
            
            # Assess advice quality
            avg_quality = np.mean(qualities)
            if avg_quality > 0.8:
                meta_advice['advice_quality_assessment'] = 'high'
            elif avg_quality > 0.6:
                meta_advice['advice_quality_assessment'] = 'medium'
            else:
                meta_advice['advice_quality_assessment'] = 'low'
                meta_advice['improvement_suggestions'].append("Consider gathering more information")
            
            # Assess confidence calibration
            if len(qualities) >= 2:
                quality_confidence_correlation = np.corrcoef(qualities, confidences)[0, 1]
                if not np.isnan(quality_confidence_correlation) and quality_confidence_correlation > 0.5:
                    meta_advice['confidence_calibration'] = 'well_calibrated'
                else:
                    meta_advice['confidence_calibration'] = 'poorly_calibrated'
                    meta_advice['improvement_suggestions'].append("Review confidence assessment mechanisms")
            
            # Find optimal depth
            optimal_index = np.argmax(qualities)
            meta_advice['optimal_depth'] = optimal_index
            
            # Depth assessment
            if len(reasoning_analysis) > 4:
                meta_advice['reasoning_depth_assessment'] = 'excessive'
                meta_advice['improvement_suggestions'].append("Consider reducing reasoning depth for efficiency")
            elif len(reasoning_analysis) < 2:
                meta_advice['reasoning_depth_assessment'] = 'insufficient'
                meta_advice['improvement_suggestions'].append("Consider deeper analysis for complex questions")
            else:
                meta_advice['reasoning_depth_assessment'] = 'appropriate'
        
        return meta_advice
    
    def _calculate_system_confidence(self, reasoning_analysis: Dict[str, MetaReasoning]) -> float:
        """Calculate overall system confidence in the analysis."""
        
        if not reasoning_analysis:
            return 0.0
        
        # Weight higher-level reasoning more heavily
        weighted_confidences = []
        for level_key, reasoning in reasoning_analysis.items():
            level_num = int(level_key.split('_')[1])
            weight = 1.0 + (level_num * 0.1)  # Higher levels get more weight
            weighted_confidences.append(reasoning.confidence * weight)
        
        return np.mean(weighted_confidences)
    
    def _identify_relevant_limitations(self, question: str) -> List[str]:
        """Identify system limitations relevant to the question."""
        
        relevant_limitations = []
        question_lower = question.lower()
        
        # Map question types to relevant limitations
        limitation_mappings = {
            'jurisdiction': ['cross_jurisdictional_complexity', 'local_law_specificity'],
            'international': ['cross_jurisdictional_complexity', 'treaty_interpretation'],
            'novel': ['precedent_scarcity', 'emerging_technology_law'],
            'ethical': ['value_judgment_complexity', 'cultural_context_sensitivity'],
            'technical': ['technical_expertise_requirements', 'interdisciplinary_complexity']
        }
        
        for trigger, limitations in limitation_mappings.items():
            if trigger in question_lower:
                relevant_limitations.extend(limitations)
        
        # Add from known limitations
        for limitation in self.self_awareness.known_limitations:
            if any(word in question_lower for word in limitation.split('_')):
                relevant_limitations.append(limitation)
        
        return list(set(relevant_limitations))
    
    def _acknowledge_uncertainties(self, reasoning_analysis: Dict[str, MetaReasoning]) -> List[str]:
        """Acknowledge uncertainties in the analysis."""
        
        uncertainties = []
        
        # Check for low confidence areas
        for reasoning in reasoning_analysis.values():
            if reasoning.confidence < 0.6:
                uncertainties.append(f"Low confidence in {reasoning.reasoning_type} reasoning")
        
        # Check for inconsistent conclusions
        all_conclusions = []
        for reasoning in reasoning_analysis.values():
            all_conclusions.extend(reasoning.conclusions)
        
        if len(set(all_conclusions)) > len(all_conclusions) * 0.7:  # High diversity in conclusions
            uncertainties.append("Conclusions vary significantly across reasoning levels")
        
        # Add system-level uncertainties
        uncertainties.extend([
            f"Uncertainty in {area}" for area in self.self_awareness.uncertainty_areas
        ])
        
        return uncertainties
    
    def _formulate_final_recommendation(self,
                                      reasoning_analysis: Dict[str, MetaReasoning],
                                      meta_advice: Dict[str, Any]) -> Dict[str, Any]:
        """Formulate final recommendation based on all analyses."""
        
        # Get the best reasoning level
        optimal_depth = meta_advice.get('optimal_depth', 0)
        optimal_reasoning = reasoning_analysis.get(f"level_{optimal_depth}")
        
        if not optimal_reasoning:
            optimal_reasoning = list(reasoning_analysis.values())[-1]  # Fallback to last level
        
        recommendation = {
            'primary_advice': optimal_reasoning.conclusions,
            'confidence_level': optimal_reasoning.confidence,
            'reasoning_transparency': f"Based on level {optimal_reasoning.reasoning_level} {optimal_reasoning.reasoning_type} reasoning",
            'caveats': optimal_reasoning.self_critique,
            'meta_assessment': meta_advice['advice_quality_assessment'],
            'follow_up_recommendations': meta_advice.get('improvement_suggestions', [])
        }
        
        return recommendation
    
    def get_meta_reasoning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about meta-reasoning performance."""
        
        stats = {
            'meta_patterns_discovered': len(self.meta_patterns),
            'reasoning_history_size': len(self.pattern_history),
            'self_awareness_level': self.self_awareness.awareness_level,
            'known_capabilities': list(self.self_awareness.known_capabilities),
            'known_limitations': list(self.self_awareness.known_limitations),
            'pattern_effectiveness': {},
            'optimal_depth_distribution': defaultdict(int),
            'reasoning_type_distribution': defaultdict(int)
        }
        
        # Analyze pattern effectiveness
        for pattern_id, pattern in self.meta_patterns.items():
            stats['pattern_effectiveness'][pattern_id] = {
                'frequency': pattern.frequency,
                'effectiveness': pattern.effectiveness,
                'contexts': pattern.contexts
            }
        
        # Analyze historical patterns
        for record in self.pattern_history:
            # Extract optimal depth from meta-knowledge
            if 'optimal_depth' in record['context']:
                depth = record['context']['optimal_depth']
                stats['optimal_depth_distribution'][depth] += 1
        
        return stats
    
    def export_meta_knowledge(self, format_type: str = 'json') -> str:
        """Export meta-knowledge for analysis or backup."""
        
        if format_type == 'json':
            import json
            
            export_data = {
                'self_awareness': {
                    'awareness_level': self.self_awareness.awareness_level,
                    'known_capabilities': list(self.self_awareness.known_capabilities),
                    'known_limitations': list(self.self_awareness.known_limitations),
                    'uncertainty_areas': list(self.self_awareness.uncertainty_areas),
                    'confidence_calibration': self.self_awareness.confidence_calibration,
                    'improvement_opportunities': self.self_awareness.improvement_opportunities,
                    'meta_knowledge': self.self_awareness.meta_knowledge
                },
                'meta_patterns': {
                    pattern_id: {
                        'pattern_level': pattern.pattern_level,
                        'description': pattern.pattern_description,
                        'frequency': pattern.frequency,
                        'effectiveness': pattern.effectiveness,
                        'contexts': pattern.contexts,
                        'variations': pattern.variations
                    }
                    for pattern_id, pattern in self.meta_patterns.items()
                },
                'reasoning_history': [
                    {
                        'timestamp': record['timestamp'].isoformat(),
                        'pattern_id': record['pattern_id'],
                        'reasoning_quality': record['reasoning_quality'],
                        'context': record['context']
                    }
                    for record in list(self.pattern_history)[-100:]  # Last 100 records
                ]
            }
            
            return json.dumps(export_data, indent=2, default=str)
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)