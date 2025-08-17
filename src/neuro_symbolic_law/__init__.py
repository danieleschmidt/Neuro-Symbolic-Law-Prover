"""
Neuro-Symbolic Law Prover

Combines Graph Neural Networks with Z3 SMT solving to automatically prove 
regulatory compliance and identify counter-examples in legal contracts.

Generation 5: Autonomous Evolution with advanced research capabilities.
"""

__version__ = "0.5.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.ai"

from .core.legal_prover import LegalProver
from .parsing.contract_parser import ContractParser
from .core.compliance_result import ComplianceResult

# Enhanced generations
try:
    from .core.enhanced_prover import EnhancedLegalProver
    from .core.scalable_prover import ScalableLegalProver
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False

# Generation 4: Advanced features
try:
    from .research.autonomous_learning import get_autonomous_learning_engine
    from .advanced.quantum_optimization import get_quantum_legal_optimizer
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    ADVANCED_FEATURES_AVAILABLE = False

# Generation 5: Autonomous Evolution features
try:
    from .research.federated_learning import create_federated_node, get_global_federated_coordinator
    from .research.causal_reasoning import get_causal_inference_engine
    GENERATION_5_AVAILABLE = True
except ImportError:
    GENERATION_5_AVAILABLE = False

__all__ = [
    "LegalProver",
    "ContractParser", 
    "ComplianceResult",
]

if ENHANCED_FEATURES_AVAILABLE:
    __all__.extend([
        "EnhancedLegalProver",
        "ScalableLegalProver"
    ])

if ADVANCED_FEATURES_AVAILABLE:
    __all__.extend([
        "get_autonomous_learning_engine",
        "get_quantum_legal_optimizer"
    ])

if GENERATION_5_AVAILABLE:
    __all__.extend([
        "create_federated_node",
        "get_global_federated_coordinator", 
        "get_causal_inference_engine"
    ])