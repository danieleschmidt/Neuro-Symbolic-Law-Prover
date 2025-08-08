"""
Enhanced legal compliance prover with Generation 2 robustness features.
Includes comprehensive error handling, validation, monitoring, and security.
"""

from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import logging
import time
from functools import wraps

from .exceptions import (
    ComplianceVerificationError, ValidationError, ResourceError,
    validate_contract_text, validate_contract_id, validate_focus_areas,
    handle_exception_gracefully, log_security_event
)
from .monitoring import get_metrics_collector, PerformanceTimer
from .compliance_result import ComplianceResult, ComplianceReport, ComplianceStatus
from ..parsing.contract_parser import ParsedContract
from ..regulations.base_regulation import BaseRegulation

logger = logging.getLogger(__name__)


class EnhancedLegalProver:
    """
    Enhanced legal compliance prover with robust error handling and monitoring.
    
    Generation 2 Features:
    - Comprehensive input validation
    - Error handling with graceful fallbacks
    - Performance monitoring and metrics
    - Security event logging
    - Resource management
    - Circuit breaker pattern for external dependencies
    """
    
    def __init__(
        self, 
        cache_enabled: bool = True, 
        debug: bool = False, 
        max_cache_size: int = 10000,
        verification_timeout_seconds: int = 300
    ):
        """
        Initialize the enhanced legal prover.
        
        Args:
            cache_enabled: Whether to cache verification results
            debug: Enable debug logging
            max_cache_size: Maximum number of cached results
            verification_timeout_seconds: Timeout for verification operations
        """
        self.cache_enabled = cache_enabled
        self.debug = debug
        self.max_cache_size = max_cache_size
        self.verification_timeout_seconds = verification_timeout_seconds
        
        # Core data structures
        self._verification_cache: Dict[str, ComplianceResult] = {}
        self.metrics_collector = get_metrics_collector()
        
        # Circuit breaker state for external dependencies
        self._circuit_breaker = {
            'failures': 0,
            'last_failure': None,
            'state': 'closed'  # closed, open, half-open
        }
        
        # Configure logging
        if debug:
            logging.basicConfig(level=logging.DEBUG)
        
        # Track initialization
        self.metrics_collector.record_counter("enhanced_legal_prover.initialized")
        log_security_event("enhanced_legal_prover_initialized", {
            "cache_enabled": cache_enabled,
            "max_cache_size": max_cache_size,
            "verification_timeout": verification_timeout_seconds
        })
        
        logger.info(f"EnhancedLegalProver initialized with cache_size={max_cache_size}")
    
    @handle_exception_gracefully
    def verify_compliance(
        self,
        contract: ParsedContract,
        regulation: BaseRegulation,
        focus_areas: Optional[List[str]] = None,
        strict_validation: bool = True
    ) -> Dict[str, ComplianceResult]:
        """
        Verify contract compliance with comprehensive error handling.
        
        Args:
            contract: Parsed contract to verify
            regulation: Regulation to check compliance against
            focus_areas: Specific areas to focus verification on
            strict_validation: Whether to perform strict input validation
            
        Returns:
            Dictionary mapping requirement IDs to compliance results
            
        Raises:
            ValidationError: If input validation fails
            ComplianceVerificationError: If verification fails
            ResourceError: If system resources are insufficient
        """
        # Input validation
        self._validate_verification_inputs(contract, regulation, focus_areas, strict_validation)
        
        # Check circuit breaker
        if self._circuit_breaker['state'] == 'open':
            logger.warning("Circuit breaker is open, using fallback verification")
            return self._fallback_verification(contract, regulation)
        
        with PerformanceTimer(self.metrics_collector, "compliance_verification.total_time", {
            "regulation": regulation.name,
            "contract_id": contract.id
        }):
            try:
                return self._perform_verification(contract, regulation, focus_areas)
            except Exception as e:
                self._handle_verification_failure(e, regulation.name)
                # Try fallback verification
                return self._fallback_verification(contract, regulation)
    
    def _validate_verification_inputs(
        self, 
        contract: ParsedContract, 
        regulation: BaseRegulation, 
        focus_areas: Optional[List[str]],
        strict_validation: bool
    ) -> None:
        """Validate inputs for compliance verification."""
        
        if not contract:
            raise ValidationError("Contract cannot be None", "contract", None)
            
        if not regulation:
            raise ValidationError("Regulation cannot be None", "regulation", None)
        
        # Validate contract structure
        if not hasattr(contract, 'id') or not contract.id:
            raise ValidationError("Contract must have a valid ID", "contract.id", contract.id)
            
        if not hasattr(contract, 'clauses') or not contract.clauses:
            raise ValidationError("Contract must have clauses", "contract.clauses", len(contract.clauses) if hasattr(contract, 'clauses') else 0)
        
        # Validate regulation structure
        if not hasattr(regulation, 'name') or not regulation.name:
            raise ValidationError("Regulation must have a name", "regulation.name", regulation.name)
        
        try:
            requirements = regulation.get_requirements()
            if not requirements:
                raise ValidationError("Regulation must have requirements", "regulation.requirements", 0)
        except Exception as e:
            raise ValidationError(f"Invalid regulation: {str(e)}", "regulation", str(e))
        
        # Validate focus areas
        validate_focus_areas(focus_areas)
        
        # Strict validation checks
        if strict_validation:
            # Check contract content
            for i, clause in enumerate(contract.clauses):
                if not hasattr(clause, 'text') or not clause.text:
                    raise ValidationError(f"Clause {i} has empty text", f"contract.clauses[{i}].text", clause.text if hasattr(clause, 'text') else None)
                
                if len(clause.text) > 50000:  # Very long clause
                    logger.warning(f"Clause {i} is very long ({len(clause.text)} chars)")
        
        logger.debug("Input validation completed successfully")
    
    def _perform_verification(
        self, 
        contract: ParsedContract, 
        regulation: BaseRegulation, 
        focus_areas: Optional[List[str]]
    ) -> Dict[str, ComplianceResult]:
        """Perform the actual compliance verification."""
        
        logger.info(f"Starting enhanced compliance verification for {regulation.name}")
        self.metrics_collector.record_counter("compliance_verification.started", tags={
            "regulation": regulation.name,
            "contract_id": contract.id
        })
        
        # Get requirements to verify
        requirements = regulation.get_requirements()
        if focus_areas:
            original_count = len(requirements)
            requirements = {
                req_id: req for req_id, req in requirements.items()
                if any(area in req.categories for area in focus_areas)
            }
            logger.info(f"Filtered requirements: {len(requirements)}/{original_count} (focus: {focus_areas})")
        
        results = {}
        verification_start = time.time()
        
        for req_id, requirement in requirements.items():
            # Check timeout
            if time.time() - verification_start > self.verification_timeout_seconds:
                logger.warning(f"Verification timeout reached, stopping at requirement {req_id}")
                self.metrics_collector.record_counter("compliance_verification.timeout")
                break
            
            logger.debug(f"Verifying requirement: {req_id}")
            
            # Check cache first
            cache_key = self._get_cache_key(contract.id, req_id, regulation.name)
            if self.cache_enabled and cache_key in self._verification_cache:
                results[req_id] = self._verification_cache[cache_key]
                self.metrics_collector.record_counter("compliance_verification.cache_hit")
                continue
            
            # Perform verification with error handling
            try:
                with PerformanceTimer(self.metrics_collector, "requirement_verification.time", {
                    "requirement_id": req_id
                }):
                    result = self._verify_requirement_enhanced(contract, requirement)
                    results[req_id] = result
                    
                    # Cache result with size management
                    if self.cache_enabled:
                        self._verification_cache[cache_key] = result
                        self._manage_cache_size()
                        self.metrics_collector.record_counter("compliance_verification.cached")
                        
            except Exception as e:
                logger.error(f"Error verifying requirement {req_id}: {e}")
                self.metrics_collector.record_counter("compliance_verification.requirement_error", tags={
                    "requirement_id": req_id,
                    "error_type": type(e).__name__
                })
                
                # Create error result
                results[req_id] = ComplianceResult(
                    requirement_id=req_id,
                    requirement_description=requirement.description,
                    status=ComplianceStatus.UNKNOWN,
                    confidence=0.0,
                    issue=f"Verification error: {str(e)}",
                    suggestion="Please review this requirement manually"
                )
        
        # Record completion metrics
        total_time = time.time() - verification_start
        compliant_count = sum(1 for r in results.values() if r.compliant)
        
        logger.info(f"Completed verification: {len(results)} requirements checked in {total_time:.2f}s")
        
        self.metrics_collector.record_counter("compliance_verification.completed", tags={
            "regulation": regulation.name
        })
        self.metrics_collector.record_gauge("compliance_verification.compliance_rate", 
                                          (compliant_count / len(results)) * 100 if results else 0)
        self.metrics_collector.record_timer("compliance_verification.total_duration", total_time * 1000)
        
        # Reset circuit breaker on success
        self._reset_circuit_breaker()
        
        return results
    
    def _verify_requirement_enhanced(self, contract: ParsedContract, requirement) -> ComplianceResult:
        """Enhanced requirement verification with better error handling."""
        
        try:
            result = ComplianceResult(
                requirement_id=requirement.id,
                requirement_description=requirement.description,
                status=ComplianceStatus.COMPLIANT,  # Default assumption
                confidence=0.8
            )
            
            # Enhanced keyword-based verification with error handling
            relevant_clauses = []
            processed_clauses = 0
            
            for clause in contract.clauses:
                try:
                    processed_clauses += 1
                    clause_text = clause.text.lower() if clause.text else ""
                    
                    # Validate clause text
                    if not clause_text.strip():
                        logger.debug(f"Skipping empty clause {clause.id}")
                        continue
                    
                    # Check if clause contains requirement keywords
                    matching_keywords = []
                    for keyword in requirement.keywords:
                        if keyword and keyword.lower() in clause_text:
                            matching_keywords.append(keyword)
                    
                    if matching_keywords:
                        relevant_clauses.append(clause.text)
                        logger.debug(f"Clause {clause.id} matches keywords: {matching_keywords}")
                        
                except Exception as e:
                    logger.warning(f"Error processing clause {getattr(clause, 'id', 'unknown')}: {e}")
                    continue
            
            result.supporting_clauses = relevant_clauses
            
            # Enhanced compliance assessment
            if not relevant_clauses and requirement.mandatory:
                result.status = ComplianceStatus.NON_COMPLIANT
                result.issue = f"No clauses found addressing {requirement.description}"
                result.suggestion = f"Add clause addressing: {requirement.description}"
                result.confidence = 0.9
                
                # Log compliance violation with context
                log_security_event("compliance_violation_detected", {
                    "requirement_id": requirement.id,
                    "regulation": requirement.article_reference if hasattr(requirement, 'article_reference') else "unknown",
                    "contract_id": contract.id,
                    "violation_type": "missing_clause"
                }, "WARNING")
                
                self.metrics_collector.record_counter("compliance_verification.violation", tags={
                    "requirement_id": requirement.id,
                    "violation_type": "missing_clause"
                })
            
            elif relevant_clauses:
                # Assess quality of matching clauses
                clause_quality_score = self._assess_clause_quality(relevant_clauses, requirement)
                result.confidence = min(0.95, 0.7 + (clause_quality_score * 0.25))
                
                if clause_quality_score < 0.3:
                    result.status = ComplianceStatus.PARTIAL
                    result.issue = "Found relevant clauses but they may not fully address the requirement"
                    result.suggestion = "Review and strengthen the relevant clauses"
            
            # Add metadata
            result.metadata = {
                "processed_clauses": processed_clauses,
                "matching_clauses": len(relevant_clauses),
                "verification_timestamp": datetime.now().isoformat(),
                "verification_method": "enhanced_keyword_matching"
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced requirement verification: {e}")
            raise ComplianceVerificationError(
                f"Failed to verify requirement {requirement.id}: {str(e)}",
                regulation_name=getattr(requirement, 'regulation_name', 'unknown'),
                requirement_id=requirement.id
            )
    
    def _assess_clause_quality(self, clauses: List[str], requirement) -> float:
        """Assess the quality of matching clauses."""
        
        if not clauses:
            return 0.0
        
        try:
            quality_score = 0.0
            keyword_matches = 0
            total_keywords = len(requirement.keywords)
            
            # Check keyword coverage
            all_clause_text = " ".join(clauses).lower()
            for keyword in requirement.keywords:
                if keyword and keyword.lower() in all_clause_text:
                    keyword_matches += 1
            
            # Base score from keyword coverage
            if total_keywords > 0:
                quality_score = keyword_matches / total_keywords
            
            # Bonus for specific legal language
            legal_terms = ['shall', 'must', 'required', 'obligation', 'compliance', 'accordance']
            legal_term_matches = sum(1 for term in legal_terms if term in all_clause_text)
            quality_score += min(0.3, legal_term_matches * 0.05)
            
            # Penalty for vague language
            vague_terms = ['may consider', 'if possible', 'best effort', 'reasonable']
            vague_matches = sum(1 for term in vague_terms if term in all_clause_text)
            quality_score -= min(0.2, vague_matches * 0.05)
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.warning(f"Error assessing clause quality: {e}")
            return 0.5  # Default moderate score
    
    def _fallback_verification(self, contract: ParsedContract, regulation: BaseRegulation) -> Dict[str, ComplianceResult]:
        """Fallback verification when main verification fails."""
        
        logger.info("Using fallback verification method")
        self.metrics_collector.record_counter("compliance_verification.fallback_used")
        
        try:
            requirements = regulation.get_requirements()
            results = {}
            
            for req_id, requirement in requirements.items():
                # Very basic fallback - just check if any clauses exist
                has_clauses = bool(contract.clauses)
                
                result = ComplianceResult(
                    requirement_id=req_id,
                    requirement_description=requirement.description,
                    status=ComplianceStatus.UNKNOWN if not has_clauses else ComplianceStatus.PARTIAL,
                    confidence=0.3,  # Low confidence for fallback
                    issue="Verified using fallback method due to system limitations",
                    suggestion="Manual review recommended"
                )
                
                results[req_id] = result
            
            return results
            
        except Exception as e:
            logger.error(f"Fallback verification failed: {e}")
            # Return empty results rather than crashing
            return {}
    
    def _handle_verification_failure(self, error: Exception, regulation_name: str) -> None:
        """Handle verification failures and update circuit breaker."""
        
        logger.error(f"Verification failure for {regulation_name}: {error}")
        
        # Update circuit breaker
        self._circuit_breaker['failures'] += 1
        self._circuit_breaker['last_failure'] = datetime.now()
        
        if self._circuit_breaker['failures'] >= 5:  # Open circuit after 5 failures
            self._circuit_breaker['state'] = 'open'
            logger.warning("Circuit breaker opened due to repeated failures")
        
        # Log security event
        log_security_event("verification_failure", {
            "regulation_name": regulation_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "circuit_breaker_state": self._circuit_breaker['state']
        }, "ERROR")
        
        self.metrics_collector.record_counter("compliance_verification.failure", tags={
            "regulation": regulation_name,
            "error_type": type(error).__name__
        })
    
    def _reset_circuit_breaker(self) -> None:
        """Reset circuit breaker on successful operations."""
        
        if self._circuit_breaker['state'] != 'closed':
            logger.info("Resetting circuit breaker - operations successful")
            self._circuit_breaker = {
                'failures': 0,
                'last_failure': None,
                'state': 'closed'
            }
    
    def _manage_cache_size(self) -> None:
        """Manage cache size to prevent memory issues."""
        
        if len(self._verification_cache) > self.max_cache_size:
            # Remove oldest entries (simple LRU approximation)
            items_to_remove = len(self._verification_cache) - int(self.max_cache_size * 0.8)
            cache_keys = list(self._verification_cache.keys())
            
            for key in cache_keys[:items_to_remove]:
                del self._verification_cache[key]
            
            logger.info(f"Cache size management: removed {items_to_remove} old entries")
            self.metrics_collector.record_counter("cache.evicted", items_to_remove)
    
    def _get_cache_key(self, contract_id: str, requirement_id: str, regulation_name: str) -> str:
        """Generate cache key for verification result."""
        return f"{contract_id}:{requirement_id}:{regulation_name}"
    
    @handle_exception_gracefully
    def generate_compliance_report(
        self,
        contract: ParsedContract,
        regulation: BaseRegulation,
        results: Optional[Dict[str, ComplianceResult]] = None
    ) -> ComplianceReport:
        """Generate enhanced compliance report with additional metrics."""
        
        with PerformanceTimer(self.metrics_collector, "compliance_report.generation_time"):
            if results is None:
                results = self.verify_compliance(contract, regulation)
            
            # Enhanced status determination
            compliant_count = sum(1 for r in results.values() if r.compliant)
            total_count = len(results)
            partial_count = sum(1 for r in results.values() if r.status == ComplianceStatus.PARTIAL)
            unknown_count = sum(1 for r in results.values() if r.status == ComplianceStatus.UNKNOWN)
            
            if compliant_count == total_count:
                overall_status = ComplianceStatus.COMPLIANT
            elif compliant_count == 0:
                overall_status = ComplianceStatus.NON_COMPLIANT
            else:
                overall_status = ComplianceStatus.PARTIAL
            
            report = ComplianceReport(
                contract_id=contract.id,
                regulation_name=regulation.name,
                results=results,
                overall_status=overall_status,
                timestamp=datetime.now().isoformat()
            )
            
            # Add enhanced metadata
            report.metadata = {
                "verification_summary": {
                    "total_requirements": total_count,
                    "compliant": compliant_count,
                    "non_compliant": total_count - compliant_count - partial_count - unknown_count,
                    "partial": partial_count,
                    "unknown": unknown_count,
                    "compliance_percentage": (compliant_count / total_count * 100) if total_count > 0 else 0
                },
                "system_info": {
                    "cache_enabled": self.cache_enabled,
                    "cache_size": len(self._verification_cache),
                    "circuit_breaker_state": self._circuit_breaker['state'],
                    "verification_method": "enhanced_prover_v2"
                }
            }
            
            self.metrics_collector.record_counter("compliance_report.generated")
            
            return report
    
    def clear_cache(self) -> None:
        """Clear verification cache with logging."""
        
        cache_size = len(self._verification_cache)
        self._verification_cache.clear()
        
        self.metrics_collector.record_counter("cache.cleared")
        self.metrics_collector.record_gauge("cache.size", 0)
        
        logger.info(f"Cleared {cache_size} items from verification cache")
        log_security_event("cache_cleared", {
            "previous_size": cache_size,
            "cleared_by": "user_request"
        })
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        
        stats = {
            "cached_results": len(self._verification_cache),
            "cache_enabled": self.cache_enabled,
            "max_cache_size": self.max_cache_size,
            "cache_utilization": len(self._verification_cache) / self.max_cache_size if self.max_cache_size > 0 else 0,
            "circuit_breaker_state": self._circuit_breaker['state'],
            "circuit_breaker_failures": self._circuit_breaker['failures']
        }
        
        # Update metrics
        self.metrics_collector.record_gauge("cache.size", len(self._verification_cache))
        self.metrics_collector.record_gauge("cache.utilization", stats["cache_utilization"])
        
        return stats
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health information."""
        
        from .monitoring import get_health_checker
        health_checker = get_health_checker()
        
        return {
            "enhanced_prover_status": "operational",
            "cache_status": "healthy" if len(self._verification_cache) < self.max_cache_size * 0.9 else "degraded",
            "circuit_breaker": self._circuit_breaker,
            "system_health": health_checker.get_overall_status(),
            "metrics_summary": self.metrics_collector.get_all_metrics_summary()
        }