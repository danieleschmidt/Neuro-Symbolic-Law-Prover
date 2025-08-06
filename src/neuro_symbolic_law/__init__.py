"""
Neuro-Symbolic Law Prover

Combines Graph Neural Networks with Z3 SMT solving to automatically prove 
regulatory compliance and identify counter-examples in legal contracts.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.ai"

from .core.legal_prover import LegalProver
from .parsing.contract_parser import ContractParser
from .core.compliance_result import ComplianceResult

__all__ = [
    "LegalProver",
    "ContractParser", 
    "ComplianceResult",
]