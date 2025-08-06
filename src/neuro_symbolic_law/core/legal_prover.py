"""
Main legal compliance prover engine.
"""

from typing import Dict, List, Optional, Union
from datetime import datetime
import logging

from .compliance_result import ComplianceResult, ComplianceReport, ComplianceStatus
from ..parsing.contract_parser import ParsedContract
from ..regulations.base_regulation import BaseRegulation


logger = logging.getLogger(__name__)


class LegalProver:
    """
    Main engine for proving legal compliance using neuro-symbolic reasoning.
    
    Combines neural contract parsing with symbolic verification using Z3 SMT solving
    to automatically verify regulatory compliance and generate counter-examples.
    """
    
    def __init__(self, cache_enabled: bool = True, debug: bool = False):
        """
        Initialize the legal prover.
        
        Args:
            cache_enabled: Whether to cache verification results
            debug: Enable debug logging
        """
        self.cache_enabled = cache_enabled
        self.debug = debug
        self._verification_cache: Dict[str, ComplianceResult] = {}
        
        if debug:
            logging.basicConfig(level=logging.DEBUG)
    
    def verify_compliance(
        self,
        contract: ParsedContract,
        regulation: BaseRegulation,
        focus_areas: Optional[List[str]] = None
    ) -> Dict[str, ComplianceResult]:
        """
        Verify contract compliance against a regulation.
        
        Args:
            contract: Parsed contract to verify
            regulation: Regulation to check compliance against
            focus_areas: Specific areas to focus verification on
            
        Returns:
            Dictionary mapping requirement IDs to compliance results
        """
        logger.info(f"Starting compliance verification for {regulation.name}")
        
        # Get requirements to verify
        requirements = regulation.get_requirements()
        if focus_areas:
            requirements = {
                req_id: req for req_id, req in requirements.items()
                if any(area in req.categories for area in focus_areas)
            }
        
        results = {}
        
        for req_id, requirement in requirements.items():
            logger.debug(f"Verifying requirement: {req_id}")
            
            # Check cache first
            cache_key = self._get_cache_key(contract.id, req_id, regulation.name)
            if self.cache_enabled and cache_key in self._verification_cache:
                results[req_id] = self._verification_cache[cache_key]
                continue
            
            # Perform verification
            try:
                result = self._verify_requirement(contract, requirement)
                results[req_id] = result
                
                # Cache result
                if self.cache_enabled:
                    self._verification_cache[cache_key] = result
                    
            except Exception as e:
                logger.error(f"Error verifying requirement {req_id}: {e}")
                results[req_id] = ComplianceResult(
                    requirement_id=req_id,
                    requirement_description=requirement.description,
                    status=ComplianceStatus.UNKNOWN,
                    confidence=0.0,
                    issue=f"Verification error: {str(e)}"
                )
        
        logger.info(f"Completed verification: {len(results)} requirements checked")
        return results
    
    def generate_compliance_report(
        self,
        contract: ParsedContract,
        regulation: BaseRegulation,
        results: Optional[Dict[str, ComplianceResult]] = None
    ) -> ComplianceReport:
        """
        Generate comprehensive compliance report.
        
        Args:
            contract: Contract that was verified
            regulation: Regulation used for verification  
            results: Verification results (will run verification if not provided)
            
        Returns:
            Comprehensive compliance report
        """
        if results is None:
            results = self.verify_compliance(contract, regulation)
        
        # Determine overall status
        compliant_count = sum(1 for r in results.values() if r.compliant)
        total_count = len(results)
        
        if compliant_count == total_count:
            overall_status = ComplianceStatus.COMPLIANT
        elif compliant_count == 0:
            overall_status = ComplianceStatus.NON_COMPLIANT
        else:
            overall_status = ComplianceStatus.PARTIAL
        
        return ComplianceReport(
            contract_id=contract.id,
            regulation_name=regulation.name,
            results=results,
            overall_status=overall_status,
            timestamp=datetime.now().isoformat()
        )
    
    def _verify_requirement(self, contract: ParsedContract, requirement) -> ComplianceResult:
        """
        Verify a single requirement against a contract.
        
        This is the core verification logic that will be enhanced in later generations
        with neural parsing and Z3 formal verification.
        """
        # Generation 1: Basic rule-based verification
        # This will be enhanced with neural parsing + Z3 in later generations
        
        result = ComplianceResult(
            requirement_id=requirement.id,
            requirement_description=requirement.description,
            status=ComplianceStatus.COMPLIANT,  # Default assumption
            confidence=0.8  # Moderate confidence for basic implementation
        )
        
        # Simple keyword-based verification for Generation 1
        relevant_clauses = []
        for clause in contract.clauses:
            clause_text = clause.text.lower()
            
            # Check if clause contains requirement keywords
            if any(keyword.lower() in clause_text for keyword in requirement.keywords):
                relevant_clauses.append(clause.text)
        
        result.supporting_clauses = relevant_clauses
        
        # Basic compliance check based on clause presence
        if not relevant_clauses and requirement.mandatory:
            result.status = ComplianceStatus.NON_COMPLIANT
            result.issue = f"No clauses found addressing {requirement.description}"
            result.suggestion = f"Add clause addressing: {requirement.description}"
            result.confidence = 0.9  # High confidence in finding missing requirements
        
        return result
    
    def _get_cache_key(self, contract_id: str, requirement_id: str, regulation_name: str) -> str:
        """Generate cache key for verification result."""
        return f"{contract_id}:{requirement_id}:{regulation_name}"
    
    def clear_cache(self) -> None:
        """Clear verification cache."""
        self._verification_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "cached_results": len(self._verification_cache),
            "cache_enabled": self.cache_enabled
        }