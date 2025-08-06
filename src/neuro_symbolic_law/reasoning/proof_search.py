"""
Proof search strategies for legal compliance verification.
"""

from typing import Dict, List, Optional, Set, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum

from ..parsing.contract_parser import ParsedContract, Clause
from ..core.compliance_result import ComplianceResult, ComplianceStatus
from .z3_encoder import LogicalConstraint

logger = logging.getLogger(__name__)


class ProofStrategy(Enum):
    """Strategies for proof search."""
    FORWARD_CHAINING = "forward_chaining"
    BACKWARD_CHAINING = "backward_chaining" 
    RESOLUTION = "resolution"
    NATURAL_DEDUCTION = "natural_deduction"


@dataclass
class ProofStep:
    """Single step in a proof."""
    step_number: int
    rule_applied: str
    premises: List[str]
    conclusion: str
    justification: str


@dataclass
class Proof:
    """Complete proof structure."""
    goal: str
    strategy: ProofStrategy
    steps: List[ProofStep]
    valid: bool
    confidence: float = 1.0


class ProofSearcher:
    """
    Searches for proofs of legal compliance using various strategies.
    
    Implements automated theorem proving techniques adapted for legal
    reasoning to construct formal proofs of compliance or non-compliance.
    """
    
    def __init__(self, default_strategy: ProofStrategy = ProofStrategy.FORWARD_CHAINING):
        """Initialize proof searcher with default strategy."""
        self.default_strategy = default_strategy
        self.proof_cache: Dict[str, Proof] = {}
        
        # Legal reasoning rules
        self.legal_rules = self._initialize_legal_rules()
    
    def search_proof(
        self,
        goal: str,
        premises: List[str],
        contract: ParsedContract,
        strategy: Optional[ProofStrategy] = None
    ) -> Optional[Proof]:
        """
        Search for a proof of the given goal from premises.
        
        Args:
            goal: Statement to prove
            premises: Known facts/premises
            contract: Contract context
            strategy: Proof strategy to use
            
        Returns:
            Proof if found, None otherwise
        """
        strategy = strategy or self.default_strategy
        
        logger.info(f"Searching proof for '{goal}' using {strategy.value}")
        
        # Check cache first
        cache_key = self._get_cache_key(goal, premises, strategy)
        if cache_key in self.proof_cache:
            return self.proof_cache[cache_key]
        
        proof = None
        
        try:
            if strategy == ProofStrategy.FORWARD_CHAINING:
                proof = self._forward_chaining_search(goal, premises, contract)
            elif strategy == ProofStrategy.BACKWARD_CHAINING:
                proof = self._backward_chaining_search(goal, premises, contract)
            elif strategy == ProofStrategy.RESOLUTION:
                proof = self._resolution_search(goal, premises, contract)
            else:
                logger.warning(f"Strategy {strategy} not implemented, using forward chaining")
                proof = self._forward_chaining_search(goal, premises, contract)
            
            # Cache result
            if proof:
                self.proof_cache[cache_key] = proof
                
        except Exception as e:
            logger.error(f"Error in proof search: {e}")
        
        return proof
    
    def prove_compliance(
        self,
        requirement_id: str,
        contract: ParsedContract,
        regulation_facts: List[str]
    ) -> Tuple[bool, Optional[Proof]]:
        """
        Attempt to prove compliance with a specific requirement.
        
        Args:
            requirement_id: ID of requirement to prove
            contract: Contract to check
            regulation_facts: Facts from regulation
            
        Returns:
            (is_compliant, proof_if_found)
        """
        logger.info(f"Attempting to prove compliance for {requirement_id}")
        
        # Extract contract facts
        contract_facts = self._extract_contract_facts(contract)
        
        # Combine all premises
        premises = contract_facts + regulation_facts
        
        # Define compliance goal
        goal = f"compliant({requirement_id})"
        
        # Search for proof
        proof = self.search_proof(goal, premises, contract)
        
        if proof and proof.valid:
            return True, proof
        
        # If no proof found, try to prove non-compliance
        negated_goal = f"not_compliant({requirement_id})"
        negative_proof = self.search_proof(negated_goal, premises, contract)
        
        if negative_proof and negative_proof.valid:
            return False, negative_proof
        
        # Neither compliance nor non-compliance could be proven
        return False, None
    
    def _forward_chaining_search(
        self,
        goal: str,
        premises: List[str],
        contract: ParsedContract
    ) -> Optional[Proof]:
        """Forward chaining proof search."""
        
        steps = []
        derived_facts = set(premises)
        step_count = 0
        max_steps = 50  # Prevent infinite loops
        
        while step_count < max_steps:
            step_count += 1
            new_facts = set()
            
            # Apply all possible rules
            for rule_name, rule in self.legal_rules.items():
                new_conclusions = self._apply_rule(rule, derived_facts)
                
                for conclusion in new_conclusions:
                    if conclusion not in derived_facts:
                        new_facts.add(conclusion)
                        
                        # Record proof step
                        steps.append(ProofStep(
                            step_number=len(steps) + 1,
                            rule_applied=rule_name,
                            premises=list(rule["premises"]),
                            conclusion=conclusion,
                            justification=f"Applied {rule_name} rule"
                        ))
                        
                        # Check if we reached the goal
                        if conclusion == goal:
                            return Proof(
                                goal=goal,
                                strategy=ProofStrategy.FORWARD_CHAINING,
                                steps=steps,
                                valid=True,
                                confidence=0.8
                            )
            
            # Add new facts to derived set
            derived_facts.update(new_facts)
            
            # If no new facts derived, stop
            if not new_facts:
                break
        
        # Goal not reached
        return Proof(
            goal=goal,
            strategy=ProofStrategy.FORWARD_CHAINING,
            steps=steps,
            valid=False,
            confidence=0.0
        )
    
    def _backward_chaining_search(
        self,
        goal: str,
        premises: List[str],
        contract: ParsedContract
    ) -> Optional[Proof]:
        """Backward chaining proof search."""
        
        # Simplified backward chaining implementation
        # Generation 3 will have full implementation
        
        steps = []
        
        # Check if goal is already in premises
        if goal in premises:
            steps.append(ProofStep(
                step_number=1,
                rule_applied="premise",
                premises=[goal],
                conclusion=goal,
                justification="Goal is a given premise"
            ))
            
            return Proof(
                goal=goal,
                strategy=ProofStrategy.BACKWARD_CHAINING,
                steps=steps,
                valid=True,
                confidence=1.0
            )
        
        # Try to find rules that conclude the goal
        for rule_name, rule in self.legal_rules.items():
            if rule["conclusion"] == goal:
                # Check if all premises of the rule are satisfied
                rule_premises = rule["premises"]
                
                if all(premise in premises for premise in rule_premises):
                    steps.append(ProofStep(
                        step_number=1,
                        rule_applied=rule_name,
                        premises=rule_premises,
                        conclusion=goal,
                        justification=f"Applied {rule_name} rule backwards"
                    ))
                    
                    return Proof(
                        goal=goal,
                        strategy=ProofStrategy.BACKWARD_CHAINING,
                        steps=steps,
                        valid=True,
                        confidence=0.7
                    )
        
        # Goal not provable
        return Proof(
            goal=goal,
            strategy=ProofStrategy.BACKWARD_CHAINING,
            steps=steps,
            valid=False,
            confidence=0.0
        )
    
    def _resolution_search(
        self,
        goal: str,
        premises: List[str],
        contract: ParsedContract
    ) -> Optional[Proof]:
        """Resolution-based proof search."""
        
        # Placeholder for resolution implementation
        # Generation 3 will implement full resolution theorem proving
        
        steps = []
        
        # Convert to clause form and apply resolution
        # For now, return invalid proof
        return Proof(
            goal=goal,
            strategy=ProofStrategy.RESOLUTION,
            steps=steps,
            valid=False,
            confidence=0.0
        )
    
    def _apply_rule(self, rule: Dict[str, Any], facts: Set[str]) -> Set[str]:
        """Apply a logical rule to derive new facts."""
        
        conclusions = set()
        
        try:
            premises = set(rule["premises"])
            
            # Check if all premises are satisfied
            if premises.issubset(facts):
                conclusion = rule["conclusion"]
                conclusions.add(conclusion)
                
                # Apply any transformations
                if "transformation" in rule:
                    transformed = rule["transformation"](facts)
                    conclusions.update(transformed)
        
        except Exception as e:
            logger.error(f"Error applying rule: {e}")
        
        return conclusions
    
    def _extract_contract_facts(self, contract: ParsedContract) -> List[str]:
        """Extract logical facts from contract clauses."""
        
        facts = []
        
        # Basic fact extraction from clauses
        for clause in contract.clauses:
            clause_text = clause.text.lower()
            
            # Extract specific types of facts
            if "personal data" in clause_text:
                facts.append("processes_personal_data(contract)")
                
                if "security" in clause_text or "secure" in clause_text:
                    facts.append("has_security_measures(contract)")
                
                if "delete" in clause_text or "deletion" in clause_text:
                    facts.append("has_deletion_procedure(contract)")
                
                if "consent" in clause_text:
                    facts.append("requires_consent(contract)")
            
            if "ai" in clause_text or "artificial intelligence" in clause_text:
                facts.append("uses_ai_system(contract)")
                
                if "transparent" in clause_text or "explainable" in clause_text:
                    facts.append("has_transparency(contract)")
                
                if "human" in clause_text and "oversight" in clause_text:
                    facts.append("has_human_oversight(contract)")
            
            # Extract obligations
            if clause.obligations:
                for obligation in clause.obligations:
                    facts.append(f"obligation({obligation})")
        
        # Add party facts
        for party in contract.parties:
            facts.append(f"party({party.name}, {party.role})")
        
        return facts
    
    def _initialize_legal_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize legal reasoning rules."""
        
        rules = {}
        
        # GDPR rules
        rules["gdpr_data_minimization"] = {
            "premises": ["processes_personal_data(contract)", "has_purpose_limitation(contract)"],
            "conclusion": "compliant(GDPR-5.1.c)",
            "confidence": 0.8
        }
        
        rules["gdpr_security"] = {
            "premises": ["processes_personal_data(contract)", "has_security_measures(contract)"],
            "conclusion": "compliant(GDPR-32.1)",
            "confidence": 0.9
        }
        
        rules["gdpr_data_subject_rights"] = {
            "premises": ["processes_personal_data(contract)", "has_access_procedure(contract)"],
            "conclusion": "compliant(GDPR-15.1)",
            "confidence": 0.8
        }
        
        # AI Act rules
        rules["ai_transparency"] = {
            "premises": ["uses_ai_system(contract)", "has_transparency(contract)"],
            "conclusion": "compliant(AI-ACT-13.1)",
            "confidence": 0.85
        }
        
        rules["ai_human_oversight"] = {
            "premises": ["uses_ai_system(contract)", "has_human_oversight(contract)"],
            "conclusion": "compliant(AI-ACT-14.1)",
            "confidence": 0.9
        }
        
        # General rules
        rules["modus_ponens"] = {
            "premises": ["implies(A, B)", "A"],
            "conclusion": "B",
            "confidence": 1.0
        }
        
        return rules
    
    def _get_cache_key(
        self,
        goal: str,
        premises: List[str],
        strategy: ProofStrategy
    ) -> str:
        """Generate cache key for proof search."""
        
        premises_str = "|".join(sorted(premises))
        return f"{goal}:{premises_str}:{strategy.value}"
    
    def clear_cache(self) -> None:
        """Clear proof cache."""
        self.proof_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get proof cache statistics."""
        return {
            "cached_proofs": len(self.proof_cache)
        }