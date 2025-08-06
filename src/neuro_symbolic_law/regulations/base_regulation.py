"""
Base classes for regulatory compliance definitions.
"""

from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum


class RequirementType(Enum):
    """Types of compliance requirements."""
    MANDATORY = "mandatory"
    RECOMMENDED = "recommended"
    CONDITIONAL = "conditional"
    PROHIBITED = "prohibited"


@dataclass
class ComplianceRequirement:
    """Represents a single compliance requirement."""
    id: str
    description: str
    article_reference: Optional[str]
    requirement_type: RequirementType
    keywords: List[str]
    categories: Set[str]
    mandatory: bool = True
    
    def __post_init__(self):
        if self.requirement_type == RequirementType.MANDATORY:
            self.mandatory = True
        elif self.requirement_type == RequirementType.RECOMMENDED:
            self.mandatory = False


class BaseRegulation(ABC):
    """
    Abstract base class for all regulatory compliance models.
    
    Each regulation (GDPR, AI Act, CCPA, etc.) should inherit from this class
    and implement the required methods.
    """
    
    def __init__(self, name: str, version: str = "1.0"):
        self.name = name
        self.version = version
        self._requirements: Dict[str, ComplianceRequirement] = {}
        self._initialize_requirements()
    
    @abstractmethod
    def _initialize_requirements(self) -> None:
        """Initialize the regulation's requirements. Must be implemented by subclasses."""
        pass
    
    def get_requirements(self, categories: Optional[List[str]] = None) -> Dict[str, ComplianceRequirement]:
        """
        Get all requirements, optionally filtered by categories.
        
        Args:
            categories: Filter requirements by these categories
            
        Returns:
            Dictionary mapping requirement IDs to requirements
        """
        if categories is None:
            return self._requirements.copy()
        
        filtered = {}
        for req_id, req in self._requirements.items():
            if any(cat in req.categories for cat in categories):
                filtered[req_id] = req
        
        return filtered
    
    def get_requirement(self, requirement_id: str) -> Optional[ComplianceRequirement]:
        """Get a specific requirement by ID."""
        return self._requirements.get(requirement_id)
    
    def add_requirement(self, requirement: ComplianceRequirement) -> None:
        """Add a new requirement to the regulation."""
        self._requirements[requirement.id] = requirement
    
    def get_mandatory_requirements(self) -> Dict[str, ComplianceRequirement]:
        """Get only mandatory requirements."""
        return {
            req_id: req for req_id, req in self._requirements.items()
            if req.mandatory
        }
    
    def get_categories(self) -> Set[str]:
        """Get all categories across all requirements."""
        categories = set()
        for req in self._requirements.values():
            categories.update(req.categories)
        return categories
    
    def __len__(self) -> int:
        """Return number of requirements."""
        return len(self._requirements)
    
    def __str__(self) -> str:
        return f"{self.name} v{self.version} ({len(self._requirements)} requirements)"