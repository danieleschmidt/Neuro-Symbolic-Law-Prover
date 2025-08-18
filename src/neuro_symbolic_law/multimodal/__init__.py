"""
Multi-Modal Legal Analysis Module

Generation 6: Next-Generation Enhancement
Comprehensive multi-modal analysis for legal documents including:
- Visual document analysis (PDFs, images, charts)
- Audio processing (legal proceedings, depositions)
- Video analysis (legal presentations, evidence)
- Cross-modal correlation and reasoning
"""

from .vision_analyzer import LegalVisionAnalyzer, DocumentImageProcessor

# Try to import advanced multimodal components
try:
    from .audio_processor import LegalAudioProcessor, ProceedingTranscriber
    from .video_analyzer import LegalVideoAnalyzer, EvidenceAnalyzer
    from .multimodal_fusion import MultiModalLegalFusion
    ADVANCED_MULTIMODAL_AVAILABLE = True
except ImportError:
    ADVANCED_MULTIMODAL_AVAILABLE = False

__all__ = [
    "LegalVisionAnalyzer",
    "DocumentImageProcessor"
]

if ADVANCED_MULTIMODAL_AVAILABLE:
    __all__.extend([
        "LegalAudioProcessor",
        "ProceedingTranscriber",
        "LegalVideoAnalyzer",
        "EvidenceAnalyzer",
        "MultiModalLegalFusion"
    ])