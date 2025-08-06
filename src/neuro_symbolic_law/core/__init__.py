"""
Core legal prover functionality.
"""

from .legal_prover import LegalProver
from .compliance_result import ComplianceResult, ComplianceReport, ComplianceStatus

__all__ = ["LegalProver", "ComplianceResult", "ComplianceReport", "ComplianceStatus"]