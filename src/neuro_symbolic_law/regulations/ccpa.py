"""
CCPA (California Consumer Privacy Act) compliance model.
"""

from typing import Set
from .base_regulation import BaseRegulation, ComplianceRequirement, RequirementType


class CCPA(BaseRegulation):
    """
    CCPA compliance model implementing California privacy requirements.
    
    Based on California Consumer Privacy Act with focus on:
    - Consumer rights
    - Personal information handling
    - Disclosure requirements
    - Opt-out mechanisms
    """
    
    def __init__(self):
        super().__init__(name="CCPA", version="2020")
    
    def _initialize_requirements(self) -> None:
        """Initialize CCPA compliance requirements."""
        
        # Section 1798.100 - Right to know
        self.add_requirement(ComplianceRequirement(
            id="CCPA-1798.100.a",
            description="Consumers have the right to request information about personal information collected",
            article_reference="Civil Code 1798.100(a)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["right to know", "personal information", "collection", "disclosure"],
            categories={"consumer_rights", "right_to_know", "disclosure"}
        ))
        
        # Section 1798.105 - Right to delete
        self.add_requirement(ComplianceRequirement(
            id="CCPA-1798.105.a",
            description="Consumers have the right to request deletion of personal information",
            article_reference="Civil Code 1798.105(a)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["right to delete", "deletion", "personal information", "consumer request"],
            categories={"consumer_rights", "deletion", "right_to_delete"}
        ))
        
        # Section 1798.110 - Right to know specific pieces and categories
        self.add_requirement(ComplianceRequirement(
            id="CCPA-1798.110.a",
            description="Business must disclose specific pieces of personal information collected",
            article_reference="Civil Code 1798.110(a)", 
            requirement_type=RequirementType.MANDATORY,
            keywords=["specific pieces", "categories", "personal information", "disclosure"],
            categories={"disclosure", "transparency", "consumer_rights"}
        ))
        
        # Section 1798.115 - Right to know about sale/sharing
        self.add_requirement(ComplianceRequirement(
            id="CCPA-1798.115.a",
            description="Business must disclose categories of personal information sold or shared",
            article_reference="Civil Code 1798.115(a)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["sale", "sharing", "third parties", "categories", "disclosure"],
            categories={"sale_disclosure", "sharing", "third_parties"}
        ))
        
        # Section 1798.120 - Right to opt-out of sale/sharing
        self.add_requirement(ComplianceRequirement(
            id="CCPA-1798.120.a",
            description="Consumers have the right to opt out of sale/sharing of personal information",
            article_reference="Civil Code 1798.120(a)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["opt-out", "sale", "sharing", "do not sell", "consumer choice"],
            categories={"consumer_rights", "opt_out", "sale", "sharing"}
        ))
        
        # Section 1798.121 - Right to limit use of sensitive personal information
        self.add_requirement(ComplianceRequirement(
            id="CCPA-1798.121.a",
            description="Consumers can limit use of sensitive personal information",
            article_reference="Civil Code 1798.121(a)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["sensitive personal information", "limit use", "consumer choice"],
            categories={"sensitive_data", "consumer_rights", "limit_use"}
        ))
        
        # Section 1798.125 - Non-discrimination
        self.add_requirement(ComplianceRequirement(
            id="CCPA-1798.125.a",
            description="Business cannot discriminate against consumers exercising privacy rights",
            article_reference="Civil Code 1798.125(a)",
            requirement_type=RequirementType.PROHIBITED,
            keywords=["discrimination", "retaliation", "privacy rights", "equal treatment"],
            categories={"non_discrimination", "consumer_protection"}
        ))
        
        # Section 1798.130 - Notice at collection
        self.add_requirement(ComplianceRequirement(
            id="CCPA-1798.130.a",
            description="Business must inform consumers about personal information collection at time of collection",
            article_reference="Civil Code 1798.130(a)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["notice at collection", "collection notice", "categories", "purposes"],
            categories={"notice", "transparency", "collection_notice"}
        ))
        
        # Section 1798.130(b) - Privacy policy requirements
        self.add_requirement(ComplianceRequirement(
            id="CCPA-1798.130.b",
            description="Business must maintain comprehensive privacy policy",
            article_reference="Civil Code 1798.130(b)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["privacy policy", "categories", "purposes", "sources", "sharing"],
            categories={"privacy_policy", "transparency", "disclosure"}
        ))
        
        # Section 1798.135 - Opt-out methods
        self.add_requirement(ComplianceRequirement(
            id="CCPA-1798.135.a",
            description="Business must provide clear methods for consumers to opt out",
            article_reference="Civil Code 1798.135(a)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["opt-out method", "do not sell my personal information", "clear and conspicuous"],
            categories={"opt_out_methods", "user_interface", "accessibility"}
        ))
        
        # Section 1798.140 - Definitions and sensitive personal information
        self.add_requirement(ComplianceRequirement(
            id="CCPA-1798.140.ae",
            description="Sensitive personal information requires special handling",
            article_reference="Civil Code 1798.140(ae)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["sensitive personal information", "biometric", "geolocation", "racial origin"],
            categories={"sensitive_data", "special_categories", "protection"}
        ))
        
        # Verification requirements (Regulations 7002-7006)
        self.add_requirement(ComplianceRequirement(
            id="CCPA-REG-7003",
            description="Business must verify consumer identity for requests",
            article_reference="Regulation 7003",
            requirement_type=RequirementType.MANDATORY,
            keywords=["verification", "identity verification", "consumer requests", "authentication"],
            categories={"verification", "identity", "security"}
        ))
        
        # Response timeframes (Regulation 7001)
        self.add_requirement(ComplianceRequirement(
            id="CCPA-REG-7001",
            description="Business must respond to consumer requests within 45 days",
            article_reference="Regulation 7001",
            requirement_type=RequirementType.MANDATORY,
            keywords=["response time", "45 days", "consumer requests", "timeframe"],
            categories={"response_time", "consumer_service"}
        ))
        
        # Authorized agent requirements
        self.add_requirement(ComplianceRequirement(
            id="CCPA-REG-7004",
            description="Business must accept requests from authorized agents",
            article_reference="Regulation 7004",
            requirement_type=RequirementType.MANDATORY,
            keywords=["authorized agent", "power of attorney", "consumer authorization"],
            categories={"authorized_agents", "consumer_representation"}
        ))
        
        # Record keeping requirements
        self.add_requirement(ComplianceRequirement(
            id="CCPA-1798.185.a.15",
            description="Business must maintain records of consumer requests and responses",
            article_reference="Civil Code 1798.185(a)(15)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["record keeping", "consumer requests", "compliance records", "audit trail"],
            categories={"record_keeping", "audit_trail", "compliance"}
        ))
        
        # Training requirements for employees
        self.add_requirement(ComplianceRequirement(
            id="CCPA-REG-7100",
            description="Business must train employees handling consumer requests",
            article_reference="Regulation 7100",
            requirement_type=RequirementType.RECOMMENDED,
            keywords=["employee training", "privacy training", "consumer requests"],
            categories={"training", "employee_education", "governance"}
        ))
        
        # Third-party sharing agreements
        self.add_requirement(ComplianceRequirement(
            id="CCPA-1798.115.c",
            description="Business must have contractual restrictions on third-party use",
            article_reference="Civil Code 1798.115(c)",
            requirement_type=RequirementType.MANDATORY,
            keywords=["third party", "contractual restrictions", "service provider", "contractor"],
            categories={"third_party_agreements", "contractual_requirements"}
        ))