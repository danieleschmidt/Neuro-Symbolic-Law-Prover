"""
GDPR (General Data Protection Regulation) compliance model.
"""

from typing import Set
from .base_regulation import BaseRegulation, ComplianceRequirement, RequirementType


class GDPR(BaseRegulation):
    """
    GDPR compliance model implementing key data protection requirements.
    
    Based on EU GDPR (Regulation 2016/679) with focus on:
    - Data subject rights
    - Lawful basis for processing  
    - Data minimization
    - Purpose limitation
    - Storage limitation
    - Security measures
    """
    
    def __init__(self):
        super().__init__(name="GDPR", version="2016/679")
    
    def _initialize_requirements(self) -> None:
        """Initialize GDPR compliance requirements."""
        
        # Article 5 - Principles of processing
        self.add_requirement(ComplianceRequirement(
            id="GDPR-5.1.a",
            description="Personal data must be processed lawfully, fairly and transparently",
            article_reference="Article 5(1)(a)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["lawful", "fair", "transparent", "processing", "personal data"],
            categories={"lawfulness", "transparency", "fairness"}
        ))
        
        self.add_requirement(ComplianceRequirement(
            id="GDPR-5.1.b", 
            description="Personal data must be collected for specified, explicit and legitimate purposes",
            article_reference="Article 5(1)(b)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["purpose", "specified", "explicit", "legitimate", "purpose limitation"],
            categories={"purpose_limitation", "collection"}
        ))
        
        self.add_requirement(ComplianceRequirement(
            id="GDPR-5.1.c",
            description="Personal data must be adequate, relevant and limited to what is necessary",
            article_reference="Article 5(1)(c)", 
            requirement_type=RequirementType.MANDATORY,
            keywords=["data minimization", "adequate", "relevant", "necessary", "limited"],
            categories={"data_minimization", "necessity"}
        ))
        
        self.add_requirement(ComplianceRequirement(
            id="GDPR-5.1.d",
            description="Personal data must be accurate and kept up to date",
            article_reference="Article 5(1)(d)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["accurate", "up to date", "correct", "data quality"],
            categories={"data_quality", "accuracy"}
        ))
        
        self.add_requirement(ComplianceRequirement(
            id="GDPR-5.1.e",
            description="Personal data must not be kept longer than necessary",
            article_reference="Article 5(1)(e)",
            requirement_type=RequirementType.MANDATORY, 
            keywords=["retention", "storage limitation", "delete", "necessary period"],
            categories={"data_retention", "storage_limitation"}
        ))
        
        self.add_requirement(ComplianceRequirement(
            id="GDPR-5.1.f",
            description="Personal data must be processed securely with appropriate technical measures",
            article_reference="Article 5(1)(f)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["security", "technical measures", "organizational measures", "protection"],
            categories={"security", "technical_measures"}
        ))
        
        # Article 6 - Lawfulness of processing
        self.add_requirement(ComplianceRequirement(
            id="GDPR-6.1",
            description="Processing must have a lawful basis under Article 6",
            article_reference="Article 6(1)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["lawful basis", "consent", "contract", "legal obligation", "vital interests"],
            categories={"lawful_basis", "consent"}
        ))
        
        # Article 7 - Conditions for consent
        self.add_requirement(ComplianceRequirement(
            id="GDPR-7.1",
            description="Consent must be freely given, specific, informed and unambiguous",
            article_reference="Article 7(1)",
            requirement_type=RequirementType.CONDITIONAL,
            keywords=["consent", "freely given", "specific", "informed", "unambiguous"],
            categories={"consent", "consent_conditions"}
        ))
        
        # Article 12-14 - Information to data subjects
        self.add_requirement(ComplianceRequirement(
            id="GDPR-12.1",
            description="Information must be provided in concise, transparent and intelligible form",
            article_reference="Article 12(1)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["privacy notice", "information", "concise", "transparent", "intelligible"],
            categories={"transparency", "privacy_notice"}
        ))
        
        self.add_requirement(ComplianceRequirement(
            id="GDPR-13.1",
            description="Information must be provided at time of data collection",
            article_reference="Article 13(1)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["collection notice", "data collection", "information provided"],
            categories={"transparency", "collection_notice"}
        ))
        
        # Article 15-22 - Data subject rights
        self.add_requirement(ComplianceRequirement(
            id="GDPR-15.1",
            description="Data subjects have the right to access their personal data",
            article_reference="Article 15(1)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["access", "data subject access", "right of access", "copy"],
            categories={"data_subject_rights", "access_rights"}
        ))
        
        self.add_requirement(ComplianceRequirement(
            id="GDPR-16",
            description="Data subjects have the right to rectification of inaccurate data",
            article_reference="Article 16", 
            requirement_type=RequirementType.MANDATORY,
            keywords=["rectification", "correct", "inaccurate", "right to rectification"],
            categories={"data_subject_rights", "rectification"}
        ))
        
        self.add_requirement(ComplianceRequirement(
            id="GDPR-17.1",
            description="Data subjects have the right to erasure (right to be forgotten)",
            article_reference="Article 17(1)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["erasure", "deletion", "right to be forgotten", "delete"],
            categories={"data_subject_rights", "erasure", "deletion"}
        ))
        
        self.add_requirement(ComplianceRequirement(
            id="GDPR-20.1",
            description="Data subjects have the right to data portability",
            article_reference="Article 20(1)",
            requirement_type=RequirementType.CONDITIONAL,
            keywords=["data portability", "machine readable", "portable format"],
            categories={"data_subject_rights", "portability"}
        ))
        
        # Article 25 - Data protection by design and default
        self.add_requirement(ComplianceRequirement(
            id="GDPR-25.1",
            description="Data protection by design and by default must be implemented",
            article_reference="Article 25(1)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["privacy by design", "data protection by design", "by default", "technical measures"],
            categories={"privacy_by_design", "technical_measures"}
        ))
        
        # Article 28 - Processor obligations
        self.add_requirement(ComplianceRequirement(
            id="GDPR-28.3",
            description="Processing by a processor must be governed by a contract",
            article_reference="Article 28(3)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["data processing agreement", "processor", "contract", "dpa"],
            categories={"processor_agreements", "contracts"}
        ))
        
        # Article 32 - Security of processing
        self.add_requirement(ComplianceRequirement(
            id="GDPR-32.1",
            description="Appropriate technical and organizational security measures must be implemented",
            article_reference="Article 32(1)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["security measures", "encryption", "pseudonymization", "confidentiality", "integrity"],
            categories={"security", "technical_measures", "organizational_measures"}
        ))
        
        # Article 33-34 - Breach notification
        self.add_requirement(ComplianceRequirement(
            id="GDPR-33.1",
            description="Personal data breaches must be notified to supervisory authority within 72 hours",
            article_reference="Article 33(1)", 
            requirement_type=RequirementType.MANDATORY,
            keywords=["data breach", "notification", "supervisory authority", "72 hours"],
            categories={"breach_notification", "incident_response"}
        ))
        
        self.add_requirement(ComplianceRequirement(
            id="GDPR-34.1",
            description="High-risk breaches must be communicated to affected data subjects",
            article_reference="Article 34(1)",
            requirement_type=RequirementType.CONDITIONAL,
            keywords=["breach notification", "data subjects", "high risk", "communication"],
            categories={"breach_notification", "data_subject_notification"}
        ))
        
        # Article 35 - Data protection impact assessment
        self.add_requirement(ComplianceRequirement(
            id="GDPR-35.1",
            description="DPIA required for high risk processing operations", 
            article_reference="Article 35(1)",
            requirement_type=RequirementType.CONDITIONAL,
            keywords=["data protection impact assessment", "dpia", "high risk", "privacy impact"],
            categories={"dpia", "risk_assessment"}
        ))
        
        # Article 37 - Data Protection Officer
        self.add_requirement(ComplianceRequirement(
            id="GDPR-37.1",
            description="DPO must be designated in certain circumstances",
            article_reference="Article 37(1)",
            requirement_type=RequirementType.CONDITIONAL,
            keywords=["data protection officer", "dpo", "designate"],
            categories={"dpo", "governance"}
        ))