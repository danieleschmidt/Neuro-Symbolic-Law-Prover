"""
Z3 SMT encoding for legal compliance verification.
"""

from typing import Dict, List, Optional, Any, Union
import logging
from dataclasses import dataclass

# Z3 imports with fallback for testing
try:
    from z3 import *
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    # Fallback classes for testing without Z3
    class Solver:
        def __init__(self): pass
        def add(self, *args): pass
        def check(self): return "sat"
        def model(self): return {}
    
    class Bool:
        def __init__(self, name): self.name = name
    
    class Real:
        def __init__(self, name): self.name = name
    
    class String:
        def __init__(self, name): self.name = name
    
    def sat(): return "sat"
    def unsat(): return "unsat"

from ..parsing.contract_parser import ParsedContract, Clause
from ..core.compliance_result import ComplianceResult, ComplianceViolation, ComplianceStatus, ViolationSeverity, CounterExample

logger = logging.getLogger(__name__)


@dataclass
class LogicalConstraint:
    """Represents a logical constraint in Z3."""
    name: str
    formula: Any  # Z3 formula
    description: str
    constraint_type: str = "requirement"


class Z3Encoder:
    """
    Encodes legal compliance requirements as Z3 SMT formulas.
    
    Translates natural language legal requirements into formal logical
    constraints that can be verified using Z3 SMT solving.
    """
    
    def __init__(self, debug: bool = False):
        """Initialize Z3 encoder."""
        self.debug = debug
        self.constraints: List[LogicalConstraint] = []
        self._variable_registry: Dict[str, Any] = {}
        
        if debug:
            logging.basicConfig(level=logging.DEBUG)
        
        if not Z3_AVAILABLE:
            logger.warning("Z3 not available - using fallback implementation")
    
    def encode_gdpr_data_minimization(self, contract: ParsedContract) -> List[LogicalConstraint]:
        """
        Encode GDPR Article 5(1)(c) - Data minimization principle.
        
        Formula: ∀ data_item: collected(data_item) → necessary(data_item, purpose)
        """
        constraints = []
        
        try:
            solver = Solver()
            
            # Variables for data minimization
            data_collected = Real('data_collected')
            data_necessary = Real('data_necessary')
            collection_justified = Bool('collection_justified')
            
            # Extract data collection information from contract
            collection_clauses = self._extract_data_collection_clauses(contract)
            purposes = self._extract_stated_purposes(contract)
            
            # Constraint: collected data should not exceed necessary data
            minimization_constraint = data_collected <= data_necessary
            
            # Justification constraint: all collection must be justified by purpose
            purpose_justification = collection_justified
            
            # Add constraints
            constraint = LogicalConstraint(
                name="gdpr_data_minimization",
                formula=And(minimization_constraint, purpose_justification),
                description="Personal data must be adequate, relevant and limited to what is necessary",
                constraint_type="gdpr_requirement"
            )
            
            constraints.append(constraint)
            
            # Store variables for later use
            self._variable_registry['data_collected'] = data_collected
            self._variable_registry['data_necessary'] = data_necessary
            self._variable_registry['collection_justified'] = collection_justified
            
        except Exception as e:
            logger.error(f"Error encoding data minimization: {e}")
            # Return empty constraint that will be handled gracefully
            constraints.append(LogicalConstraint(
                name="gdpr_data_minimization_error",
                formula=Bool('error_placeholder'),
                description=f"Error encoding data minimization: {e}",
                constraint_type="error"
            ))
        
        return constraints
    
    def encode_gdpr_purpose_limitation(self, contract: ParsedContract) -> List[LogicalConstraint]:
        """
        Encode GDPR Article 5(1)(b) - Purpose limitation principle.
        
        Formula: ∀ use: data_use(use) → ∃ purpose ∈ stated_purposes: compatible(use, purpose)
        """
        constraints = []
        
        try:
            # Variables for purpose limitation
            purpose_compliant = Bool('purpose_compliant')
            stated_purposes_complete = Bool('stated_purposes_complete')
            uses_compatible = Bool('uses_compatible')
            
            # Extract purposes and uses from contract
            stated_purposes = self._extract_stated_purposes(contract)
            actual_uses = self._extract_data_uses(contract)
            
            # Purpose limitation constraint
            purpose_constraint = And(
                stated_purposes_complete,  # All purposes are stated
                uses_compatible,           # All uses are compatible with purposes
                purpose_compliant          # Overall purpose compliance
            )
            
            constraint = LogicalConstraint(
                name="gdpr_purpose_limitation",
                formula=purpose_constraint,
                description="Personal data must be collected for specified, explicit and legitimate purposes",
                constraint_type="gdpr_requirement"
            )
            
            constraints.append(constraint)
            
            # Store variables
            self._variable_registry['purpose_compliant'] = purpose_compliant
            self._variable_registry['stated_purposes_complete'] = stated_purposes_complete
            self._variable_registry['uses_compatible'] = uses_compatible
            
        except Exception as e:
            logger.error(f"Error encoding purpose limitation: {e}")
            constraints.append(LogicalConstraint(
                name="gdpr_purpose_limitation_error",
                formula=Bool('error_placeholder'),
                description=f"Error encoding purpose limitation: {e}",
                constraint_type="error"
            ))
        
        return constraints
    
    def encode_gdpr_retention_limits(self, contract: ParsedContract) -> List[LogicalConstraint]:
        """
        Encode GDPR Article 5(1)(e) - Storage limitation principle.
        
        Formula: ∀ data: retention_period(data) ≤ necessary_period(data, purpose)
        """
        constraints = []
        
        try:
            # Variables for retention
            retention_period = Real('retention_period')
            necessary_period = Real('necessary_period')
            retention_compliant = Bool('retention_compliant')
            deletion_mechanism = Bool('deletion_mechanism_exists')
            
            # Extract retention information
            retention_clauses = self._extract_retention_clauses(contract)
            
            # Retention constraint
            retention_constraint = And(
                retention_period <= necessary_period,  # Period not excessive
                retention_compliant,                   # Overall retention compliance
                deletion_mechanism                     # Deletion mechanism exists
            )
            
            constraint = LogicalConstraint(
                name="gdpr_retention_limits",
                formula=retention_constraint,
                description="Personal data must not be kept longer than necessary",
                constraint_type="gdpr_requirement"
            )
            
            constraints.append(constraint)
            
            # Store variables
            self._variable_registry['retention_period'] = retention_period
            self._variable_registry['necessary_period'] = necessary_period
            self._variable_registry['retention_compliant'] = retention_compliant
            
        except Exception as e:
            logger.error(f"Error encoding retention limits: {e}")
            constraints.append(LogicalConstraint(
                name="gdpr_retention_limits_error",
                formula=Bool('error_placeholder'),
                description=f"Error encoding retention limits: {e}",
                constraint_type="error"
            ))
        
        return constraints
    
    def encode_ai_act_transparency(self, contract: ParsedContract) -> List[LogicalConstraint]:
        """
        Encode AI Act Article 13 - Transparency requirements.
        
        Formula: ai_system(x) ∧ high_risk(x) → transparent(x) ∧ explainable(x)
        """
        constraints = []
        
        try:
            # Variables for AI transparency
            is_ai_system = Bool('is_ai_system')
            is_high_risk = Bool('is_high_risk')
            is_transparent = Bool('is_transparent')
            has_explanations = Bool('has_explanations')
            user_informed = Bool('user_informed')
            
            # AI transparency constraint
            transparency_constraint = Implies(
                And(is_ai_system, is_high_risk),
                And(is_transparent, has_explanations, user_informed)
            )
            
            constraint = LogicalConstraint(
                name="ai_act_transparency",
                formula=transparency_constraint,
                description="High-risk AI systems must be designed to ensure transparency to users",
                constraint_type="ai_act_requirement"
            )
            
            constraints.append(constraint)
            
            # Store variables
            self._variable_registry['is_ai_system'] = is_ai_system
            self._variable_registry['is_high_risk'] = is_high_risk
            self._variable_registry['is_transparent'] = is_transparent
            
        except Exception as e:
            logger.error(f"Error encoding AI transparency: {e}")
            constraints.append(LogicalConstraint(
                name="ai_act_transparency_error",
                formula=Bool('error_placeholder'),
                description=f"Error encoding AI transparency: {e}",
                constraint_type="error"
            ))
        
        return constraints
    
    def encode_ai_act_human_oversight(self, contract: ParsedContract) -> List[LogicalConstraint]:
        """
        Encode AI Act Article 14 - Human oversight requirements.
        
        Formula: ai_system(x) ∧ high_risk(x) → human_oversight(x) ∧ can_intervene(x)
        """
        constraints = []
        
        try:
            # Variables for human oversight
            is_ai_system = self._variable_registry.get('is_ai_system', Bool('is_ai_system'))
            is_high_risk = self._variable_registry.get('is_high_risk', Bool('is_high_risk'))
            has_human_oversight = Bool('has_human_oversight')
            can_intervene = Bool('can_intervene')
            human_can_stop = Bool('human_can_stop')
            
            # Human oversight constraint
            oversight_constraint = Implies(
                And(is_ai_system, is_high_risk),
                And(has_human_oversight, can_intervene, human_can_stop)
            )
            
            constraint = LogicalConstraint(
                name="ai_act_human_oversight",
                formula=oversight_constraint,
                description="High-risk AI systems must be designed to ensure effective human oversight",
                constraint_type="ai_act_requirement"
            )
            
            constraints.append(constraint)
            
            # Store variables
            self._variable_registry['has_human_oversight'] = has_human_oversight
            self._variable_registry['can_intervene'] = can_intervene
            
        except Exception as e:
            logger.error(f"Error encoding human oversight: {e}")
            constraints.append(LogicalConstraint(
                name="ai_act_human_oversight_error",
                formula=Bool('error_placeholder'),
                description=f"Error encoding human oversight: {e}",
                constraint_type="error"
            ))
        
        return constraints
    
    def _extract_data_collection_clauses(self, contract: ParsedContract) -> List[Clause]:
        """Extract clauses related to data collection."""
        keywords = ['collect', 'collection', 'gather', 'obtain', 'acquire', 'personal data']
        return contract.get_clauses_containing(keywords)
    
    def _extract_stated_purposes(self, contract: ParsedContract) -> List[str]:
        """Extract stated purposes from contract."""
        purposes = []
        purpose_keywords = ['purpose', 'use', 'process', 'processing']
        
        purpose_clauses = contract.get_clauses_containing(purpose_keywords)
        for clause in purpose_clauses:
            # Simple extraction - in Generation 3 this would use advanced NLP
            if 'purpose' in clause.text.lower():
                purposes.append(clause.text)
        
        return purposes
    
    def _extract_data_uses(self, contract: ParsedContract) -> List[str]:
        """Extract actual data uses from contract."""
        uses = []
        use_keywords = ['analytics', 'marketing', 'service', 'support', 'research']
        
        use_clauses = contract.get_clauses_containing(use_keywords)
        for clause in use_clauses:
            uses.append(clause.text)
        
        return uses
    
    def _extract_retention_clauses(self, contract: ParsedContract) -> List[Clause]:
        """Extract clauses related to data retention."""
        keywords = ['retention', 'retain', 'keep', 'store', 'delete', 'deletion']
        return contract.get_clauses_containing(keywords)
    
    def get_all_constraints(self) -> List[LogicalConstraint]:
        """Get all registered constraints."""
        return self.constraints.copy()
    
    def clear_constraints(self) -> None:
        """Clear all constraints and variables."""
        self.constraints.clear()
        self._variable_registry.clear()
    
    def get_variable(self, name: str) -> Optional[Any]:
        """Get a registered Z3 variable by name."""
        return self._variable_registry.get(name)