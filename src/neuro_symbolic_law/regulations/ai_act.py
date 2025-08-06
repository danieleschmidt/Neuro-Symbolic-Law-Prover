"""
EU AI Act compliance model.
"""

from typing import Set
from .base_regulation import BaseRegulation, ComplianceRequirement, RequirementType


class AIAct(BaseRegulation):
    """
    EU AI Act compliance model implementing key AI system requirements.
    
    Based on EU AI Act (Regulation 2024/1689) with focus on:
    - Risk-based approach
    - High-risk AI systems
    - Transparency requirements
    - Human oversight
    - Accuracy and robustness
    """
    
    def __init__(self):
        super().__init__(name="EU AI Act", version="2024/1689")
    
    def _initialize_requirements(self) -> None:
        """Initialize AI Act compliance requirements."""
        
        # Article 9 - Risk management system
        self.add_requirement(ComplianceRequirement(
            id="AI-ACT-9.1",
            description="High-risk AI systems must have a risk management system",
            article_reference="Article 9(1)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["risk management", "high-risk", "ai system", "risk assessment"],
            categories={"risk_management", "high_risk"}
        ))
        
        # Article 10 - Data governance
        self.add_requirement(ComplianceRequirement(
            id="AI-ACT-10.1",
            description="Training data must be relevant, representative and free of errors",
            article_reference="Article 10(1)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["training data", "data quality", "representative", "bias", "errors"],
            categories={"data_governance", "training_data", "bias"}
        ))
        
        self.add_requirement(ComplianceRequirement(
            id="AI-ACT-10.2", 
            description="Data sets must be examined for bias and appropriate mitigation measures taken",
            article_reference="Article 10(2)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["bias", "discrimination", "mitigation", "data examination", "fairness"],
            categories={"bias_mitigation", "fairness", "discrimination"}
        ))
        
        # Article 11 - Technical documentation
        self.add_requirement(ComplianceRequirement(
            id="AI-ACT-11.1",
            description="High-risk AI systems must have comprehensive technical documentation",
            article_reference="Article 11(1)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["technical documentation", "system description", "documentation"],
            categories={"documentation", "technical_documentation"}
        ))
        
        # Article 12 - Record keeping
        self.add_requirement(ComplianceRequirement(
            id="AI-ACT-12.1",
            description="High-risk AI systems must automatically log operations",
            article_reference="Article 12(1)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["logging", "record keeping", "audit trail", "operations log"],
            categories={"logging", "audit_trail", "record_keeping"}
        ))
        
        # Article 13 - Transparency and information to users
        self.add_requirement(ComplianceRequirement(
            id="AI-ACT-13.1",
            description="High-risk AI systems must be designed to ensure transparency to users",
            article_reference="Article 13(1)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["transparency", "user information", "interpretability", "explainability"],
            categories={"transparency", "user_information", "explainability"}
        ))
        
        self.add_requirement(ComplianceRequirement(
            id="AI-ACT-13.2",
            description="Instructions for use must be provided in appropriate format",
            article_reference="Article 13(2)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["instructions", "user manual", "usage instructions", "format"],
            categories={"user_instructions", "documentation"}
        ))
        
        # Article 14 - Human oversight
        self.add_requirement(ComplianceRequirement(
            id="AI-ACT-14.1",
            description="High-risk AI systems must be designed to ensure effective human oversight",
            article_reference="Article 14(1)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["human oversight", "human intervention", "human control", "supervision"],
            categories={"human_oversight", "human_intervention"}
        ))
        
        # Article 15 - Accuracy, robustness and cybersecurity
        self.add_requirement(ComplianceRequirement(
            id="AI-ACT-15.1",
            description="High-risk AI systems must achieve appropriate accuracy levels",
            article_reference="Article 15(1)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["accuracy", "performance", "robustness", "reliability"],
            categories={"accuracy", "performance", "robustness"}
        ))
        
        self.add_requirement(ComplianceRequirement(
            id="AI-ACT-15.2",
            description="High-risk AI systems must be resilient against attacks and secure",
            article_reference="Article 15(2)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["cybersecurity", "security", "resilient", "attacks", "secure"],
            categories={"cybersecurity", "security", "resilience"}
        ))
        
        # Article 16 - Quality management system
        self.add_requirement(ComplianceRequirement(
            id="AI-ACT-16.1",
            description="Providers must establish a quality management system",
            article_reference="Article 16(1)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["quality management", "qms", "quality system", "provider"],
            categories={"quality_management", "governance"}
        ))
        
        # Article 50 - Transparency obligations for certain AI systems
        self.add_requirement(ComplianceRequirement(
            id="AI-ACT-50.1",
            description="AI systems interacting with humans must inform users they are interacting with AI",
            article_reference="Article 50(1)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["human interaction", "ai disclosure", "chatbot", "virtual assistant"],
            categories={"transparency", "ai_disclosure", "human_interaction"}
        ))
        
        self.add_requirement(ComplianceRequirement(
            id="AI-ACT-50.2",
            description="Emotion recognition systems must inform users of their operation",
            article_reference="Article 50(2)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["emotion recognition", "biometric", "emotion detection", "disclosure"],
            categories={"emotion_recognition", "biometric", "transparency"}
        ))
        
        self.add_requirement(ComplianceRequirement(
            id="AI-ACT-50.3",
            description="Biometric categorization systems must inform users",
            article_reference="Article 50(3)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["biometric categorization", "biometric classification", "disclosure"],
            categories={"biometric_categorization", "transparency"}
        ))
        
        self.add_requirement(ComplianceRequirement(
            id="AI-ACT-50.4",
            description="AI-generated content must be disclosed as artificially generated",
            article_reference="Article 50(4)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["deepfake", "synthetic media", "ai-generated", "artificial content"],
            categories={"synthetic_media", "deepfake", "ai_generated_content"}
        ))
        
        # Article 59 - Post-market monitoring
        self.add_requirement(ComplianceRequirement(
            id="AI-ACT-59.1",
            description="Providers must establish post-market monitoring system",
            article_reference="Article 59(1)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["post-market monitoring", "monitoring", "surveillance", "performance monitoring"],
            categories={"post_market_monitoring", "surveillance"}
        ))
        
        # Article 61 - Reporting of serious incidents
        self.add_requirement(ComplianceRequirement(
            id="AI-ACT-61.1",
            description="Serious incidents must be reported to market surveillance authorities",
            article_reference="Article 61(1)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["incident reporting", "serious incidents", "market surveillance", "reporting"],
            categories={"incident_reporting", "market_surveillance"}
        ))
        
        # Prohibited AI practices (Article 5)
        self.add_requirement(ComplianceRequirement(
            id="AI-ACT-5.1.a",
            description="AI systems using subliminal techniques are prohibited",
            article_reference="Article 5(1)(a)",
            requirement_type=RequirementType.PROHIBITED,
            keywords=["subliminal", "manipulation", "deception", "prohibited"],
            categories={"prohibited_practices", "manipulation"}
        ))
        
        self.add_requirement(ComplianceRequirement(
            id="AI-ACT-5.1.b",
            description="AI systems exploiting vulnerabilities are prohibited",
            article_reference="Article 5(1)(b)",
            requirement_type=RequirementType.PROHIBITED,
            keywords=["vulnerability exploitation", "vulnerable groups", "harm", "prohibited"],
            categories={"prohibited_practices", "vulnerable_groups"}
        ))
        
        # High-risk AI system categories (Annex III)
        self.add_requirement(ComplianceRequirement(
            id="AI-ACT-ANNEX-III.1",
            description="Biometric identification systems are high-risk",
            article_reference="Annex III(1)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["biometric identification", "facial recognition", "high-risk"],
            categories={"high_risk_classification", "biometric"}
        ))
        
        self.add_requirement(ComplianceRequirement(
            id="AI-ACT-ANNEX-III.2",
            description="Critical infrastructure AI systems are high-risk",
            article_reference="Annex III(2)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["critical infrastructure", "safety component", "high-risk"],
            categories={"high_risk_classification", "critical_infrastructure"}
        ))
        
        self.add_requirement(ComplianceRequirement(
            id="AI-ACT-ANNEX-III.5",
            description="Employment and worker management AI systems are high-risk",
            article_reference="Annex III(5)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["employment", "recruitment", "worker evaluation", "hr", "high-risk"],
            categories={"high_risk_classification", "employment"}
        ))