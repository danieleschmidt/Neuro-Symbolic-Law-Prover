"""
Compliance verification results and related data structures.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum


class ComplianceStatus(Enum):
    """Status of compliance verification."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class ViolationSeverity(Enum):
    """Severity levels for compliance violations."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


@dataclass
class CounterExample:
    """Counter-example showing how compliance can be violated."""
    scenario: str
    variables: Dict[str, Any]
    violation_path: List[str]
    description: str


@dataclass
class ComplianceViolation:
    """Details of a specific compliance violation."""
    rule_id: str
    rule_description: str
    violation_text: str
    severity: ViolationSeverity
    clause_location: Optional[str] = None
    counter_example: Optional[CounterExample] = None
    suggested_fix: Optional[str] = None


@dataclass
class ComplianceResult:
    """Result of compliance verification for a single requirement."""
    
    requirement_id: str
    requirement_description: str
    status: ComplianceStatus
    confidence: float = 1.0
    
    # Details for non-compliant results
    violations: List[ComplianceViolation] = None
    issue: Optional[str] = None
    counter_example: Optional[Dict[str, Any]] = None
    suggestion: Optional[str] = None
    
    # Supporting evidence
    supporting_clauses: List[str] = None
    formal_proof: Optional[str] = None
    
    def __post_init__(self):
        if self.violations is None:
            self.violations = []
        if self.supporting_clauses is None:
            self.supporting_clauses = []
    
    @property 
    def compliant(self) -> bool:
        """Check if requirement is compliant."""
        return self.status == ComplianceStatus.COMPLIANT
    
    @property
    def has_violations(self) -> bool:
        """Check if result contains violations."""
        return len(self.violations) > 0
    
    def add_violation(self, violation: ComplianceViolation) -> None:
        """Add a violation to this result."""
        self.violations.append(violation)
        if self.status == ComplianceStatus.COMPLIANT:
            self.status = ComplianceStatus.NON_COMPLIANT
    
    def get_severity(self) -> ViolationSeverity:
        """Get the highest severity violation."""
        if not self.violations:
            return ViolationSeverity.INFORMATIONAL
        
        severities = [v.severity for v in self.violations]
        severity_order = [
            ViolationSeverity.CRITICAL,
            ViolationSeverity.HIGH, 
            ViolationSeverity.MEDIUM,
            ViolationSeverity.LOW,
            ViolationSeverity.INFORMATIONAL
        ]
        
        for severity in severity_order:
            if severity in severities:
                return severity
        
        return ViolationSeverity.INFORMATIONAL


@dataclass 
class ComplianceReport:
    """Comprehensive compliance report for a contract."""
    
    contract_id: str
    regulation_name: str
    results: Dict[str, ComplianceResult]
    overall_status: ComplianceStatus
    timestamp: str
    
    # Summary statistics
    total_requirements: int = 0
    compliant_count: int = 0
    violation_count: int = 0
    
    def __post_init__(self):
        self.total_requirements = len(self.results)
        self.compliant_count = sum(1 for r in self.results.values() if r.compliant)
        self.violation_count = sum(len(r.violations) for r in self.results.values())
    
    @property
    def compliance_rate(self) -> float:
        """Calculate compliance rate as percentage."""
        if self.total_requirements == 0:
            return 0.0
        return (self.compliant_count / self.total_requirements) * 100.0
    
    def get_violations(self) -> List[ComplianceViolation]:
        """Get all violations across all requirements."""
        violations = []
        for result in self.results.values():
            violations.extend(result.violations)
        return violations
    
    def get_violations_by_severity(self, severity: ViolationSeverity) -> List[ComplianceViolation]:
        """Get violations of specific severity."""
        return [v for v in self.get_violations() if v.severity == severity]
    
    def get_critical_violations(self) -> List[ComplianceViolation]:
        """Get all critical violations."""
        return self.get_violations_by_severity(ViolationSeverity.CRITICAL)