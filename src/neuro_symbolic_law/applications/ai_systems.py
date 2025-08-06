"""
AI system contract analysis and AI Act compliance.
"""

from typing import Dict, List, Optional, Any, Set
import logging
from dataclasses import dataclass
from enum import Enum

from ..core.enhanced_prover import EnhancedLegalProver
from ..parsing.neural_parser import NeuralContractParser
from ..regulations import AIAct
from ..core.compliance_result import ComplianceResult

logger = logging.getLogger(__name__)


class AIRiskLevel(Enum):
    """AI system risk levels according to AI Act."""
    MINIMAL = "minimal"
    LIMITED = "limited" 
    HIGH = "high"
    PROHIBITED = "prohibited"


@dataclass
class AISystemClassification:
    """Classification of AI system based on AI Act."""
    risk_level: AIRiskLevel
    application_areas: List[str]
    use_cases: List[str]
    regulatory_requirements: List[str]
    confidence: float = 1.0


@dataclass
class AIComplianceGap:
    """AI Act compliance gap."""
    requirement_id: str
    gap_type: str
    severity: str
    description: str
    article_reference: str
    remediation_steps: List[str]


class AISystemContractAnalyzer:
    """
    Specialized analyzer for AI system contracts and AI Act compliance.
    
    Features:
    - AI system risk classification
    - AI Act compliance verification
    - Transparency requirement analysis
    - Human oversight assessment
    - Bias and fairness evaluation
    """
    
    def __init__(self, debug: bool = False):
        """Initialize AI system contract analyzer."""
        self.prover = EnhancedLegalProver(debug=debug)
        self.parser = NeuralContractParser(debug=debug)
        self.ai_act = AIAct()
        self.debug = debug
        
        # AI system classification rules
        self.risk_classification_rules = self._initialize_risk_classification()
        
        # High-risk AI system categories from Annex III
        self.high_risk_categories = self._initialize_high_risk_categories()
        
        # Transparency requirements mapping
        self.transparency_requirements = self._initialize_transparency_requirements()
    
    def analyze_ai_contract(
        self,
        contract_text: str,
        contract_id: Optional[str] = None,
        system_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive AI system contract analysis.
        
        Args:
            contract_text: AI system contract text
            contract_id: Contract identifier
            system_description: Additional AI system description
            
        Returns:
            Comprehensive AI Act compliance analysis
        """
        logger.info(f"Analyzing AI system contract {contract_id}")
        
        try:
            # Enhanced parsing with AI-specific extraction
            parsed_contract, contract_graph = self.parser.parse_enhanced(
                contract_text, contract_id, extract_semantics=True, build_graph=True
            )
            
            # AI system classification
            ai_classification = self.classify_ai_system(parsed_contract, system_description)
            
            # AI Act compliance verification
            ai_act_compliance = self._verify_ai_act_compliance(
                parsed_contract, ai_classification
            )
            
            # Transparency analysis
            transparency_analysis = self._analyze_transparency_requirements(
                parsed_contract, ai_classification
            )
            
            # Human oversight assessment
            human_oversight_analysis = self._analyze_human_oversight(
                parsed_contract, ai_classification
            )
            
            # Bias and fairness assessment
            bias_analysis = self._analyze_bias_and_fairness(parsed_contract)
            
            # Risk assessment
            risk_assessment = self._assess_ai_risks(
                ai_classification, ai_act_compliance, transparency_analysis
            )
            
            # Generate recommendations
            recommendations = self._generate_ai_recommendations(
                ai_classification, ai_act_compliance, risk_assessment
            )
            
            return {
                'contract_id': parsed_contract.id,
                'ai_classification': ai_classification,
                'ai_act_compliance': ai_act_compliance,
                'transparency_analysis': transparency_analysis,
                'human_oversight_analysis': human_oversight_analysis,
                'bias_analysis': bias_analysis,
                'risk_assessment': risk_assessment,
                'recommendations': recommendations,
                'contract_graph': contract_graph
            }
            
        except Exception as e:
            logger.error(f"Error analyzing AI contract: {e}")
            return {'error': str(e)}
    
    def classify_ai_system(
        self,
        contract,
        system_description: Optional[str] = None
    ) -> AISystemClassification:
        """
        Classify AI system according to AI Act risk categories.
        
        Args:
            contract: Parsed contract
            system_description: Additional system description
            
        Returns:
            AI system classification
        """
        logger.info("Classifying AI system risk level")
        
        try:
            # Extract AI-related content
            ai_content = self._extract_ai_system_content(contract, system_description)
            
            # Check for prohibited practices
            if self._check_prohibited_practices(ai_content):
                return AISystemClassification(
                    risk_level=AIRiskLevel.PROHIBITED,
                    application_areas=['prohibited'],
                    use_cases=['prohibited_ai_practice'],
                    regulatory_requirements=['immediate_cessation'],
                    confidence=0.9
                )
            
            # Check for high-risk categories
            high_risk_matches = self._check_high_risk_categories(ai_content)
            if high_risk_matches:
                return AISystemClassification(
                    risk_level=AIRiskLevel.HIGH,
                    application_areas=list(high_risk_matches),
                    use_cases=self._extract_use_cases(ai_content),
                    regulatory_requirements=self._get_high_risk_requirements(),
                    confidence=0.85
                )
            
            # Check for limited risk (transparency requirements)
            if self._check_limited_risk_indicators(ai_content):
                return AISystemClassification(
                    risk_level=AIRiskLevel.LIMITED,
                    application_areas=self._extract_application_areas(ai_content),
                    use_cases=self._extract_use_cases(ai_content),
                    regulatory_requirements=self._get_limited_risk_requirements(),
                    confidence=0.75
                )
            
            # Default to minimal risk
            return AISystemClassification(
                risk_level=AIRiskLevel.MINIMAL,
                application_areas=self._extract_application_areas(ai_content),
                use_cases=self._extract_use_cases(ai_content),
                regulatory_requirements=self._get_minimal_risk_requirements(),
                confidence=0.6
            )
            
        except Exception as e:
            logger.error(f"Error classifying AI system: {e}")
            return AISystemClassification(
                risk_level=AIRiskLevel.MINIMAL,
                application_areas=[],
                use_cases=[],
                regulatory_requirements=[],
                confidence=0.0
            )
    
    def assess_transparency_compliance(self, contract) -> Dict[str, Any]:
        """Assess AI transparency compliance (Article 13)."""
        
        logger.info("Assessing transparency compliance")
        
        transparency_indicators = [
            'transparent', 'transparency', 'explainable', 'explanation',
            'interpretable', 'user information', 'disclosure',
            'ai interaction', 'automated decision'
        ]
        
        transparency_clauses = contract.get_clauses_containing(transparency_indicators)
        
        assessment = {
            'transparency_clauses_count': len(transparency_clauses),
            'user_informed': False,
            'explanation_provided': False,
            'ai_disclosed': False,
            'decision_transparency': False,
            'compliance_score': 0.0
        }
        
        score_components = []
        
        for clause in transparency_clauses:
            clause_text = clause.text.lower()
            
            # Check specific transparency requirements
            if any(inform in clause_text for inform in ['inform', 'notify', 'disclose']):
                assessment['user_informed'] = True
                score_components.append(0.3)
            
            if any(explain in clause_text for explain in ['explain', 'explanation', 'rationale']):
                assessment['explanation_provided'] = True
                score_components.append(0.3)
            
            if any(ai in clause_text for ai in ['ai', 'artificial intelligence', 'automated']):
                assessment['ai_disclosed'] = True
                score_components.append(0.2)
            
            if any(decision in clause_text for decision in ['decision', 'recommendation', 'output']):
                assessment['decision_transparency'] = True
                score_components.append(0.2)
        
        assessment['compliance_score'] = min(1.0, sum(score_components))
        
        return assessment
    
    def _verify_ai_act_compliance(
        self,
        contract,
        ai_classification: AISystemClassification
    ) -> Dict[str, ComplianceResult]:
        """Verify AI Act compliance based on system classification."""
        
        # Focus on relevant requirements based on risk level
        if ai_classification.risk_level == AIRiskLevel.HIGH:
            focus_areas = [
                'risk_management', 'data_governance', 'transparency',
                'human_oversight', 'accuracy', 'technical_documentation'
            ]
        elif ai_classification.risk_level == AIRiskLevel.LIMITED:
            focus_areas = ['transparency', 'ai_disclosure']
        else:
            focus_areas = []
        
        # Verify compliance
        results = self.prover.verify_compliance(
            contract=contract,
            regulation=self.ai_act,
            focus_areas=focus_areas
        )
        
        return results
    
    def _analyze_transparency_requirements(
        self,
        contract,
        ai_classification: AISystemClassification
    ) -> Dict[str, Any]:
        """Analyze transparency requirements compliance."""
        
        transparency_analysis = self.assess_transparency_compliance(contract)
        
        # Add specific requirements based on classification
        if ai_classification.risk_level in [AIRiskLevel.HIGH, AIRiskLevel.LIMITED]:
            required_disclosures = self._get_required_disclosures(ai_classification)
            
            disclosure_compliance = {}
            for disclosure_type in required_disclosures:
                disclosure_compliance[disclosure_type] = self._check_disclosure_compliance(
                    contract, disclosure_type
                )
            
            transparency_analysis['required_disclosures'] = disclosure_compliance
        
        return transparency_analysis
    
    def _analyze_human_oversight(
        self,
        contract,
        ai_classification: AISystemClassification
    ) -> Dict[str, Any]:
        """Analyze human oversight requirements (Article 14)."""
        
        if ai_classification.risk_level != AIRiskLevel.HIGH:
            return {'required': False, 'analysis': 'Human oversight not required for this risk level'}
        
        oversight_keywords = [
            'human oversight', 'human supervision', 'human intervention',
            'human control', 'human review', 'manual override',
            'stop system', 'disable system', 'human operator'
        ]
        
        oversight_clauses = contract.get_clauses_containing(oversight_keywords)
        
        analysis = {
            'required': True,
            'oversight_clauses_count': len(oversight_clauses),
            'human_intervention': False,
            'system_control': False,
            'competent_operators': False,
            'oversight_measures': [],
            'compliance_score': 0.0
        }
        
        score_components = []
        
        for clause in oversight_clauses:
            clause_text = clause.text.lower()
            
            if any(intervention in clause_text for intervention in ['intervention', 'override', 'stop']):
                analysis['human_intervention'] = True
                analysis['oversight_measures'].append('human_intervention')
                score_components.append(0.4)
            
            if any(control in clause_text for control in ['control', 'supervise', 'monitor']):
                analysis['system_control'] = True
                analysis['oversight_measures'].append('system_control')
                score_components.append(0.3)
            
            if any(competent in clause_text for competent in ['competent', 'qualified', 'trained']):
                analysis['competent_operators'] = True
                analysis['oversight_measures'].append('competent_operators')
                score_components.append(0.3)
        
        analysis['compliance_score'] = min(1.0, sum(score_components))
        
        return analysis
    
    def _analyze_bias_and_fairness(self, contract) -> Dict[str, Any]:
        """Analyze bias mitigation and fairness provisions."""
        
        bias_keywords = [
            'bias', 'discrimination', 'fairness', 'fair', 'equitable',
            'representative data', 'diverse data', 'bias mitigation',
            'algorithmic fairness', 'protected attributes'
        ]
        
        bias_clauses = contract.get_clauses_containing(bias_keywords)
        
        analysis = {
            'bias_clauses_count': len(bias_clauses),
            'bias_mitigation': False,
            'data_quality': False,
            'fairness_testing': False,
            'monitoring_bias': False,
            'protected_groups': [],
            'mitigation_measures': []
        }
        
        for clause in bias_clauses:
            clause_text = clause.text.lower()
            
            if any(mitigation in clause_text for mitigation in ['mitigate', 'reduce', 'prevent']):
                analysis['bias_mitigation'] = True
                analysis['mitigation_measures'].append('bias_mitigation')
            
            if any(quality in clause_text for quality in ['representative', 'diverse', 'quality']):
                analysis['data_quality'] = True
                analysis['mitigation_measures'].append('data_quality')
            
            if any(test in clause_text for test in ['test', 'evaluate', 'assess']):
                analysis['fairness_testing'] = True
                analysis['mitigation_measures'].append('fairness_testing')
            
            if any(monitor in clause_text for monitor in ['monitor', 'track', 'measure']):
                analysis['monitoring_bias'] = True
                analysis['mitigation_measures'].append('monitoring')
            
            # Extract protected groups mentioned
            protected_terms = ['gender', 'race', 'ethnicity', 'age', 'disability', 'religion']
            for term in protected_terms:
                if term in clause_text:
                    analysis['protected_groups'].append(term)
        
        return analysis
    
    def _initialize_risk_classification(self) -> Dict[str, Dict[str, Any]]:
        """Initialize AI system risk classification rules."""
        
        return {
            'prohibited_patterns': [
                'subliminal techniques', 'manipulative techniques',
                'vulnerable persons', 'cognitive disabilities',
                'social scoring', 'real-time biometric identification'
            ],
            'high_risk_patterns': [
                'biometric identification', 'critical infrastructure',
                'education assessment', 'employment recruitment',
                'credit scoring', 'law enforcement',
                'migration control', 'justice administration'
            ],
            'limited_risk_patterns': [
                'chatbot', 'virtual assistant', 'emotion recognition',
                'biometric categorization', 'deepfake', 'synthetic media'
            ]
        }
    
    def _initialize_high_risk_categories(self) -> Dict[str, List[str]]:
        """Initialize high-risk AI system categories from AI Act Annex III."""
        
        return {
            'biometric_identification': [
                'facial recognition', 'biometric verification',
                'identity verification', 'biometric authentication'
            ],
            'critical_infrastructure': [
                'traffic management', 'power grid', 'water supply',
                'gas supply', 'heating', 'digital infrastructure'
            ],
            'education_training': [
                'educational assessment', 'student evaluation',
                'training assessment', 'skill evaluation'
            ],
            'employment': [
                'recruitment', 'hiring', 'promotion decision',
                'performance evaluation', 'work assignment',
                'monitoring workers'
            ],
            'essential_services': [
                'credit scoring', 'creditworthiness assessment',
                'insurance pricing', 'healthcare diagnosis',
                'medical treatment'
            ],
            'law_enforcement': [
                'crime prediction', 'risk assessment',
                'evidence evaluation', 'investigation'
            ],
            'migration_asylum': [
                'visa application', 'residence permit',
                'asylum application', 'border control'
            ],
            'justice_democracy': [
                'court decision support', 'democratic process',
                'election management', 'legal interpretation'
            ]
        }
    
    def _initialize_transparency_requirements(self) -> Dict[str, List[str]]:
        """Initialize transparency requirements mapping."""
        
        return {
            'high_risk': [
                'system_description', 'intended_use', 'limitations',
                'accuracy_measures', 'robustness_measures',
                'human_oversight_measures', 'data_requirements'
            ],
            'limited_risk': [
                'ai_interaction_disclosure', 'system_capabilities',
                'limitations', 'human_interaction_notice'
            ],
            'specific_systems': {
                'emotion_recognition': ['emotion_detection_disclosure'],
                'biometric_categorization': ['biometric_processing_disclosure'],
                'deepfake': ['synthetic_content_disclosure']
            }
        }
    
    def _extract_ai_system_content(self, contract, system_description: Optional[str]) -> str:
        """Extract AI system related content from contract."""
        
        ai_keywords = [
            'artificial intelligence', 'ai', 'machine learning', 'ml',
            'neural network', 'deep learning', 'algorithm', 'automated',
            'intelligent system', 'cognitive system', 'predictive',
            'recommendation', 'classification', 'decision support'
        ]
        
        ai_clauses = contract.get_clauses_containing(ai_keywords)
        ai_content = ' '.join([clause.text for clause in ai_clauses])
        
        if system_description:
            ai_content += ' ' + system_description
        
        return ai_content
    
    def _check_prohibited_practices(self, ai_content: str) -> bool:
        """Check for prohibited AI practices."""
        
        prohibited_terms = self.risk_classification_rules['prohibited_patterns']
        ai_content_lower = ai_content.lower()
        
        return any(term in ai_content_lower for term in prohibited_terms)
    
    def _check_high_risk_categories(self, ai_content: str) -> Set[str]:
        """Check for high-risk category indicators."""
        
        matches = set()
        ai_content_lower = ai_content.lower()
        
        for category, keywords in self.high_risk_categories.items():
            if any(keyword in ai_content_lower for keyword in keywords):
                matches.add(category)
        
        return matches
    
    def _check_limited_risk_indicators(self, ai_content: str) -> bool:
        """Check for limited risk indicators."""
        
        limited_risk_terms = self.risk_classification_rules['limited_risk_patterns']
        ai_content_lower = ai_content.lower()
        
        return any(term in ai_content_lower for term in limited_risk_terms)
    
    def _extract_application_areas(self, ai_content: str) -> List[str]:
        """Extract application areas from AI content."""
        
        areas = []
        ai_content_lower = ai_content.lower()
        
        area_keywords = {
            'healthcare': ['health', 'medical', 'diagnosis', 'treatment'],
            'finance': ['financial', 'banking', 'credit', 'investment'],
            'education': ['education', 'learning', 'training', 'assessment'],
            'employment': ['employment', 'hiring', 'recruitment', 'hr'],
            'law_enforcement': ['law enforcement', 'police', 'crime', 'security'],
            'transportation': ['transport', 'autonomous', 'vehicle', 'traffic'],
            'marketing': ['marketing', 'advertising', 'recommendation', 'personalization']
        }
        
        for area, keywords in area_keywords.items():
            if any(keyword in ai_content_lower for keyword in keywords):
                areas.append(area)
        
        return areas or ['general']
    
    def _extract_use_cases(self, ai_content: str) -> List[str]:
        """Extract use cases from AI content."""
        
        use_cases = []
        ai_content_lower = ai_content.lower()
        
        use_case_patterns = [
            'recommendation', 'classification', 'prediction', 'detection',
            'recognition', 'analysis', 'optimization', 'automation',
            'decision support', 'risk assessment', 'fraud detection',
            'content filtering', 'personalization', 'matching'
        ]
        
        for pattern in use_case_patterns:
            if pattern in ai_content_lower:
                use_cases.append(pattern)
        
        return use_cases or ['general_ai_system']
    
    def _get_high_risk_requirements(self) -> List[str]:
        """Get regulatory requirements for high-risk AI systems."""
        
        return [
            'risk_management_system',
            'data_governance',
            'technical_documentation', 
            'record_keeping',
            'transparency',
            'human_oversight',
            'accuracy_robustness',
            'cybersecurity',
            'quality_management',
            'conformity_assessment',
            'ce_marking',
            'post_market_monitoring'
        ]
    
    def _get_limited_risk_requirements(self) -> List[str]:
        """Get regulatory requirements for limited risk AI systems."""
        
        return [
            'transparency_obligations',
            'user_information',
            'ai_interaction_disclosure'
        ]
    
    def _get_minimal_risk_requirements(self) -> List[str]:
        """Get regulatory requirements for minimal risk AI systems."""
        
        return [
            'voluntary_codes_of_conduct',
            'best_practices_adherence'
        ]
    
    def _get_required_disclosures(self, ai_classification: AISystemClassification) -> List[str]:
        """Get required disclosures based on AI classification."""
        
        if ai_classification.risk_level == AIRiskLevel.HIGH:
            return self.transparency_requirements['high_risk']
        elif ai_classification.risk_level == AIRiskLevel.LIMITED:
            return self.transparency_requirements['limited_risk']
        else:
            return []
    
    def _check_disclosure_compliance(self, contract, disclosure_type: str) -> Dict[str, Any]:
        """Check compliance with specific disclosure requirement."""
        
        disclosure_keywords = {
            'system_description': ['system description', 'ai system', 'functionality'],
            'intended_use': ['intended use', 'purpose', 'application'],
            'limitations': ['limitation', 'constraint', 'restriction'],
            'ai_interaction_disclosure': ['ai interaction', 'automated', 'artificial intelligence'],
            'emotion_detection_disclosure': ['emotion', 'emotional state', 'mood'],
            'biometric_processing_disclosure': ['biometric', 'facial', 'identification'],
            'synthetic_content_disclosure': ['synthetic', 'generated', 'artificial', 'deepfake']
        }
        
        keywords = disclosure_keywords.get(disclosure_type, [disclosure_type])
        relevant_clauses = contract.get_clauses_containing(keywords)
        
        return {
            'compliant': len(relevant_clauses) > 0,
            'clauses_count': len(relevant_clauses),
            'adequacy_score': min(1.0, len(relevant_clauses) * 0.5)
        }
    
    def _assess_ai_risks(
        self,
        ai_classification: AISystemClassification,
        compliance_results: Dict[str, ComplianceResult],
        transparency_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess overall AI risks."""
        
        # Base risk from classification
        risk_scores = {
            AIRiskLevel.MINIMAL: 10,
            AIRiskLevel.LIMITED: 30,
            AIRiskLevel.HIGH: 70,
            AIRiskLevel.PROHIBITED: 100
        }
        
        base_risk = risk_scores[ai_classification.risk_level]
        
        # Adjust for compliance violations
        non_compliant_count = sum(1 for result in compliance_results.values() if not result.compliant)
        compliance_penalty = min(20, non_compliant_count * 5)
        
        # Adjust for transparency gaps
        transparency_score = transparency_analysis.get('compliance_score', 1.0)
        transparency_penalty = (1 - transparency_score) * 15
        
        total_risk_score = min(100, base_risk + compliance_penalty + transparency_penalty)
        
        # Risk categorization
        if total_risk_score >= 80:
            risk_level = 'CRITICAL'
        elif total_risk_score >= 60:
            risk_level = 'HIGH'
        elif total_risk_score >= 40:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        return {
            'risk_score': total_risk_score,
            'risk_level': risk_level,
            'base_risk': base_risk,
            'compliance_penalty': compliance_penalty,
            'transparency_penalty': transparency_penalty,
            'ai_classification': ai_classification.risk_level.value
        }
    
    def _generate_ai_recommendations(
        self,
        ai_classification: AISystemClassification,
        compliance_results: Dict[str, ComplianceResult],
        risk_assessment: Dict[str, Any]
    ) -> List[str]:
        """Generate AI-specific recommendations."""
        
        recommendations = []
        
        # Classification-based recommendations
        if ai_classification.risk_level == AIRiskLevel.PROHIBITED:
            recommendations.append("CRITICAL: System involves prohibited AI practices - immediate cessation required")
        
        elif ai_classification.risk_level == AIRiskLevel.HIGH:
            recommendations.extend([
                "Implement comprehensive risk management system",
                "Establish human oversight mechanisms",
                "Ensure transparent operation and user information",
                "Conduct thorough testing and validation",
                "Implement post-market monitoring"
            ])
        
        elif ai_classification.risk_level == AIRiskLevel.LIMITED:
            recommendations.extend([
                "Implement transparency obligations",
                "Ensure users are informed of AI interaction",
                "Provide clear system capabilities and limitations"
            ])
        
        # Compliance-based recommendations
        non_compliant = [r for r in compliance_results.values() if not r.compliant]
        for result in non_compliant[:3]:  # Top 3
            if result.suggestion:
                recommendations.append(f"AI Act: {result.suggestion}")
        
        # Risk-based recommendations
        if risk_assessment['risk_level'] in ['CRITICAL', 'HIGH']:
            recommendations.insert(0, f"HIGH PRIORITY: Address {risk_assessment['risk_level']} risk level")
        
        return recommendations