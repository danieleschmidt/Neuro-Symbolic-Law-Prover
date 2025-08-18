"""
AGI Legal Reasoning Module - Generation 6 Enhancement

Advanced General Intelligence framework for legal analysis including:
- Multi-modal reasoning across text, vision, and audio
- Emergent legal reasoning capabilities
- Self-improving legal knowledge graphs
- Advanced neural-symbolic integration
- Consciousness-inspired legal decision making
"""

from .agi_legal_reasoner import AGILegalReasoner, EmergentReasoningEngine

# Try to import advanced AGI components
try:
    from .consciousness_engine import LegalConsciousnessEngine, AwarenessProcessor
    from .emergent_intelligence import EmergentLegalIntelligence, SelfImprovingReasoner
    from .neural_symbolic_fusion import AdvancedNeuralSymbolicFusion, HybridReasoningEngine
    ADVANCED_AGI_AVAILABLE = True
except ImportError:
    ADVANCED_AGI_AVAILABLE = False

__all__ = [
    "AGILegalReasoner",
    "EmergentReasoningEngine"
]

if ADVANCED_AGI_AVAILABLE:
    __all__.extend([
        "LegalConsciousnessEngine", 
        "AwarenessProcessor",
        "EmergentLegalIntelligence",
        "SelfImprovingReasoner",
        "AdvancedNeuralSymbolicFusion",
        "HybridReasoningEngine"
    ])