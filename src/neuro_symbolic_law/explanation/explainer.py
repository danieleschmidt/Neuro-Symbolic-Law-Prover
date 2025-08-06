"""
Natural language explanation generation for compliance results.
"""

from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum

from ..parsing.contract_parser import ParsedContract
from ..core.compliance_result import ComplianceResult, ComplianceViolation, ComplianceStatus

logger = logging.getLogger(__name__)


class ExplanationStyle(Enum):
    """Styles of explanation generation."""
    LEGAL_PROFESSIONAL = "legal_professional"
    BUSINESS_EXECUTIVE = "business_executive" 
    TECHNICAL_TEAM = "technical_team"
    DATA_SUBJECT = "data_subject"


@dataclass
class Explanation:
    """Generated explanation for compliance result."""
    legal_explanation: str
    business_explanation: str
    technical_explanation: str
    remediation_steps: List[str]
    confidence: float = 1.0


class ExplainabilityEngine:
    """
    Generates natural language explanations for compliance verification results.
    
    Translates formal logical reasoning into human-readable explanations
    tailored to different audiences (legal, business, technical).
    """
    
    def __init__(
        self,
        llm_model: Optional[str] = None,
        style: ExplanationStyle = ExplanationStyle.LEGAL_PROFESSIONAL
    ):
        """
        Initialize explainability engine.
        
        Args:
            llm_model: Language model to use (placeholder for Generation 3)
            style: Default explanation style
        """
        self.llm_model = llm_model
        self.default_style = style
        
        # Template-based explanations for Generation 2
        self.explanation_templates = self._initialize_templates()
    
    def explain_violation(
        self,
        violation: ComplianceViolation,
        contract_context: ParsedContract,
        audience: str = "legal_team"
    ) -> Explanation:
        """
        Generate explanation for a compliance violation.
        
        Args:
            violation: Compliance violation to explain
            contract_context: Contract context for explanation
            audience: Target audience for explanation
            
        Returns:
            Multi-audience explanation
        """
        logger.info(f"Generating explanation for violation {violation.rule_id}")
        
        try:
            # Get base explanation templates
            templates = self.explanation_templates.get(
                violation.rule_id, 
                self.explanation_templates["default"]
            )
            
            # Generate audience-specific explanations
            legal_explanation = self._generate_legal_explanation(
                violation, contract_context, templates
            )
            
            business_explanation = self._generate_business_explanation(
                violation, contract_context, templates
            )
            
            technical_explanation = self._generate_technical_explanation(
                violation, contract_context, templates
            )
            
            remediation_steps = self._generate_remediation_steps(
                violation, contract_context, templates
            )
            
            return Explanation(
                legal_explanation=legal_explanation,
                business_explanation=business_explanation,
                technical_explanation=technical_explanation,
                remediation_steps=remediation_steps,
                confidence=0.8  # Template-based confidence
            )
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return self._create_fallback_explanation(violation)
    
    def explain_compliance_result(
        self,
        result: ComplianceResult,
        contract_context: ParsedContract,
        include_proof: bool = True
    ) -> Explanation:
        """
        Generate explanation for overall compliance result.
        
        Args:
            result: Compliance result to explain
            contract_context: Contract context
            include_proof: Whether to include formal proof details
            
        Returns:
            Comprehensive explanation
        """
        logger.info(f"Explaining compliance result for {result.requirement_id}")
        
        try:
            if result.status == ComplianceStatus.COMPLIANT:
                return self._explain_compliant_result(result, contract_context, include_proof)
            elif result.status == ComplianceStatus.NON_COMPLIANT:
                return self._explain_non_compliant_result(result, contract_context, include_proof)
            else:
                return self._explain_unknown_result(result, contract_context)
                
        except Exception as e:
            logger.error(f"Error explaining result: {e}")
            return self._create_fallback_explanation(result)
    
    def _generate_legal_explanation(
        self,
        violation: ComplianceViolation,
        contract: ParsedContract,
        templates: Dict[str, str]
    ) -> str:
        """Generate legal professional explanation."""
        
        template = templates.get("legal", templates["default"])
        
        # Extract relevant legal context
        regulation_name = self._extract_regulation_name(violation.rule_id)
        article_reference = self._extract_article_reference(violation.rule_id)
        
        explanation = template.format(
            rule_id=violation.rule_id,
            regulation=regulation_name,
            article=article_reference,
            violation_text=violation.violation_text,
            severity=violation.severity.value,
            contract_type=contract.contract_type or "contract"
        )
        
        # Add specific legal context
        if "GDPR" in violation.rule_id:
            explanation += self._add_gdpr_legal_context(violation)
        elif "AI-ACT" in violation.rule_id:
            explanation += self._add_ai_act_legal_context(violation)
        
        return explanation
    
    def _generate_business_explanation(
        self,
        violation: ComplianceViolation,
        contract: ParsedContract,
        templates: Dict[str, str]
    ) -> str:
        """Generate business executive explanation."""
        
        template = templates.get("business", "This compliance issue may impact business operations.")
        
        # Focus on business impact and risk
        risk_level = self._assess_business_risk(violation)
        potential_impact = self._assess_potential_impact(violation)
        
        explanation = template.format(
            rule_id=violation.rule_id,
            risk_level=risk_level,
            potential_impact=potential_impact,
            violation_text=violation.violation_text
        )
        
        # Add cost implications
        explanation += self._add_cost_implications(violation)
        
        return explanation
    
    def _generate_technical_explanation(
        self,
        violation: ComplianceViolation,
        contract: ParsedContract,
        templates: Dict[str, str]
    ) -> str:
        """Generate technical team explanation."""
        
        template = templates.get("technical", "Technical implementation required for compliance.")
        
        # Focus on implementation details
        technical_requirements = self._extract_technical_requirements(violation)
        implementation_approach = self._suggest_implementation_approach(violation)
        
        explanation = template.format(
            rule_id=violation.rule_id,
            technical_requirements=technical_requirements,
            implementation_approach=implementation_approach,
            violation_text=violation.violation_text
        )
        
        return explanation
    
    def _generate_remediation_steps(
        self,
        violation: ComplianceViolation,
        contract: ParsedContract,
        templates: Dict[str, str]
    ) -> List[str]:
        """Generate specific remediation steps."""
        
        steps = []
        
        # Get rule-specific remediation steps
        if "GDPR-5.1.c" in violation.rule_id:  # Data minimization
            steps.extend([
                "Review all data collection practices to identify unnecessary data",
                "Update privacy policy to specify exact data categories collected",
                "Implement data minimization controls in collection systems",
                "Establish regular audits of data collection necessity"
            ])
        
        elif "GDPR-32.1" in violation.rule_id:  # Security
            steps.extend([
                "Conduct comprehensive security risk assessment", 
                "Implement encryption for personal data at rest and in transit",
                "Establish access controls and authentication mechanisms",
                "Document security measures in technical documentation"
            ])
        
        elif "AI-ACT-13.1" in violation.rule_id:  # AI transparency
            steps.extend([
                "Develop user-facing explanations for AI decision-making",
                "Create transparency documentation for AI system operations",
                "Implement user notification when AI systems are in use",
                "Establish processes for responding to explanation requests"
            ])
        
        else:
            # Generic remediation steps
            steps.extend([
                "Review contract clauses addressing this requirement",
                "Consult with legal counsel on compliance approach",
                "Update contract language to address identified gaps",
                "Implement monitoring for ongoing compliance"
            ])
        
        # Add suggested fix if available
        if violation.suggested_fix:
            steps.append(f"Specific action: {violation.suggested_fix}")
        
        return steps
    
    def _explain_compliant_result(
        self,
        result: ComplianceResult,
        contract: ParsedContract,
        include_proof: bool
    ) -> Explanation:
        """Explain why a requirement is compliant."""
        
        legal_explanation = (
            f"The contract satisfies {result.requirement_id}: {result.requirement_description}. "
            f"Our analysis identified {len(result.supporting_clauses)} supporting clause(s) "
            f"that address this requirement with {result.confidence:.1%} confidence."
        )
        
        if result.formal_proof and include_proof:
            legal_explanation += f" Formal verification: {result.formal_proof}"
        
        business_explanation = (
            f"Your contract is compliant with this regulatory requirement. "
            f"This reduces compliance risk and demonstrates good governance practices."
        )
        
        technical_explanation = (
            f"The compliance check for {result.requirement_id} passed validation. "
            f"No technical changes are required for this requirement."
        )
        
        return Explanation(
            legal_explanation=legal_explanation,
            business_explanation=business_explanation,
            technical_explanation=technical_explanation,
            remediation_steps=["Monitor for regulatory changes that might affect this requirement"],
            confidence=result.confidence
        )
    
    def _explain_non_compliant_result(
        self,
        result: ComplianceResult,
        contract: ParsedContract,
        include_proof: bool
    ) -> Explanation:
        """Explain why a requirement is non-compliant."""
        
        legal_explanation = (
            f"The contract does not adequately address {result.requirement_id}: "
            f"{result.requirement_description}. "
        )
        
        if result.issue:
            legal_explanation += f"Specific issue: {result.issue}. "
        
        if result.formal_proof and include_proof:
            legal_explanation += f"Formal analysis: {result.formal_proof}"
        
        business_explanation = (
            f"This non-compliance creates regulatory risk. "
            f"Addressing this issue is recommended to avoid potential penalties "
            f"and ensure regulatory compliance."
        )
        
        technical_explanation = (
            f"The system failed compliance validation for {result.requirement_id}. "
            f"Technical implementation or contract updates are needed."
        )
        
        remediation_steps = ["Address the identified compliance gap"]
        if result.suggestion:
            remediation_steps.append(result.suggestion)
        
        # Add violation-specific remediation
        for violation in result.violations:
            explanation = self.explain_violation(violation, contract)
            remediation_steps.extend(explanation.remediation_steps)
        
        return Explanation(
            legal_explanation=legal_explanation,
            business_explanation=business_explanation,
            technical_explanation=technical_explanation,
            remediation_steps=list(set(remediation_steps)),  # Remove duplicates
            confidence=result.confidence
        )
    
    def _explain_unknown_result(
        self,
        result: ComplianceResult,
        contract: ParsedContract
    ) -> Explanation:
        """Explain unknown compliance status."""
        
        legal_explanation = (
            f"Unable to definitively determine compliance status for {result.requirement_id}. "
            f"This may require manual legal review."
        )
        
        if result.issue:
            legal_explanation += f" Issue encountered: {result.issue}"
        
        business_explanation = (
            "The compliance status is uncertain and requires further investigation. "
            "Consider seeking legal counsel for definitive assessment."
        )
        
        technical_explanation = (
            f"Automated analysis could not determine compliance for {result.requirement_id}. "
            "Manual review or additional context may be needed."
        )
        
        return Explanation(
            legal_explanation=legal_explanation,
            business_explanation=business_explanation, 
            technical_explanation=technical_explanation,
            remediation_steps=[
                "Conduct manual legal review of this requirement",
                "Gather additional context or documentation",
                "Consult with legal counsel if needed"
            ],
            confidence=result.confidence
        )
    
    def _initialize_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize explanation templates."""
        
        return {
            "GDPR-5.1.c": {
                "legal": "Article 5(1)(c) of the GDPR requires data minimization - personal data must be adequate, relevant and limited to what is necessary for the processing purposes. The identified violation indicates that the contract may permit collection or processing of data beyond what is necessary.",
                "business": "Your data collection practices may exceed what is legally necessary, creating compliance risk and potential liability.",
                "technical": "Implement data minimization controls to collect only necessary personal data."
            },
            "GDPR-32.1": {
                "legal": "Article 32 of the GDPR mandates appropriate technical and organizational security measures. The contract must specify adequate security protections for personal data.",
                "business": "Insufficient security measures expose your organization to data breaches and regulatory penalties.",
                "technical": "Implement encryption, access controls, and other technical security measures."
            },
            "default": {
                "legal": "The contract does not adequately address the regulatory requirement {rule_id}.",
                "business": "This compliance gap creates regulatory risk for your organization.",
                "technical": "Technical implementation or contract updates are needed for compliance."
            }
        }
    
    def _extract_regulation_name(self, rule_id: str) -> str:
        """Extract regulation name from rule ID."""
        if "GDPR" in rule_id:
            return "General Data Protection Regulation (GDPR)"
        elif "AI-ACT" in rule_id:
            return "EU AI Act"
        elif "CCPA" in rule_id:
            return "California Consumer Privacy Act (CCPA)"
        else:
            return "Regulation"
    
    def _extract_article_reference(self, rule_id: str) -> str:
        """Extract article reference from rule ID."""
        # Parse rule ID to extract article reference
        parts = rule_id.split("-")
        if len(parts) >= 2:
            return f"Article {parts[1]}"
        return "Article"
    
    def _assess_business_risk(self, violation: ComplianceViolation) -> str:
        """Assess business risk level."""
        if violation.severity in [ViolationSeverity.CRITICAL, ViolationSeverity.HIGH]:
            return "HIGH"
        elif violation.severity == ViolationSeverity.MEDIUM:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _assess_potential_impact(self, violation: ComplianceViolation) -> str:
        """Assess potential business impact."""
        if "GDPR" in violation.rule_id:
            return "Potential GDPR fines up to 4% of annual turnover"
        elif "AI-ACT" in violation.rule_id:
            return "Potential AI Act penalties and market access restrictions"
        else:
            return "Regulatory enforcement action and reputational damage"
    
    def _add_cost_implications(self, violation: ComplianceViolation) -> str:
        """Add cost implications to explanation."""
        return " Implementation costs should be weighed against potential penalties and business disruption."
    
    def _extract_technical_requirements(self, violation: ComplianceViolation) -> str:
        """Extract technical requirements from violation."""
        if "security" in violation.violation_text.lower():
            return "encryption, access controls, security monitoring"
        elif "transparency" in violation.violation_text.lower():
            return "user interfaces, explanation systems, documentation"
        else:
            return "system updates, process changes, documentation"
    
    def _suggest_implementation_approach(self, violation: ComplianceViolation) -> str:
        """Suggest technical implementation approach."""
        if "GDPR" in violation.rule_id:
            return "privacy-by-design implementation with automated compliance controls"
        elif "AI-ACT" in violation.rule_id:
            return "AI governance framework with technical transparency measures"
        else:
            return "systematic compliance implementation with monitoring"
    
    def _add_gdpr_legal_context(self, violation: ComplianceViolation) -> str:
        """Add GDPR-specific legal context."""
        return " Under GDPR, this may result in administrative fines and enforcement action by supervisory authorities."
    
    def _add_ai_act_legal_context(self, violation: ComplianceViolation) -> str:
        """Add AI Act-specific legal context.""" 
        return " The EU AI Act imposes specific obligations on AI system providers and may restrict market access for non-compliant systems."
    
    def _create_fallback_explanation(self, violation_or_result) -> Explanation:
        """Create fallback explanation when generation fails."""
        
        return Explanation(
            legal_explanation="A compliance issue was identified that requires legal review.",
            business_explanation="This compliance gap should be addressed to reduce regulatory risk.",
            technical_explanation="Technical or contractual changes may be needed for compliance.",
            remediation_steps=["Conduct detailed legal review", "Consult with compliance experts"],
            confidence=0.3
        )