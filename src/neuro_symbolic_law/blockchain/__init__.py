"""
Blockchain Integration Module - Generation 6 Enhancement

Immutable legal record storage and verification system including:
- Smart contract deployment for legal agreements
- Immutable audit trails for compliance decisions
- Distributed legal precedent storage
- Cryptographic proof generation
- Cross-chain legal verification
"""

from .legal_blockchain import LegalBlockchainManager, BlockchainVerifier

# Try to import advanced blockchain components
try:
    from .smart_contracts import SmartLegalContract, ContractDeploymentManager
    from .immutable_records import ImmutableLegalRecords, AuditTrailManager
    from .crypto_proofs import LegalCryptoProofs, ProofGenerator
    ADVANCED_BLOCKCHAIN_AVAILABLE = True
except ImportError:
    ADVANCED_BLOCKCHAIN_AVAILABLE = False

__all__ = [
    "LegalBlockchainManager",
    "BlockchainVerifier"
]

if ADVANCED_BLOCKCHAIN_AVAILABLE:
    __all__.extend([
        "SmartLegalContract", 
        "ContractDeploymentManager",
        "ImmutableLegalRecords",
        "AuditTrailManager",
        "LegalCryptoProofs",
        "ProofGenerator"
    ])