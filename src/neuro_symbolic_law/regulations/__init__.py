"""
Regulation models and compliance checking.
"""

from .gdpr import GDPR
from .ai_act import AIAct
from .ccpa import CCPA

__all__ = ["GDPR", "AIAct", "CCPA"]