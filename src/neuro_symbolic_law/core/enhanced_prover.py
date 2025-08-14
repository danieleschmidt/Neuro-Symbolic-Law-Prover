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
from ..parsing.neural_parser import NeuralContractParser
from ..reasoning.z3_encoder import Z3Encoder
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
        verification_timeout_seconds: int = 300,
        enable_neural_parsing: bool = True,
        enable_z3_verification: bool = True
    ):
        """
        Initialize the enhanced legal prover.
        
        Args:
            cache_enabled: Whether to cache verification results
            debug: Enable debug logging
            max_cache_size: Maximum number of cached results
            verification_timeout_seconds: Timeout for verification operations
            enable_neural_parsing: Enable neural contract parsing
            enable_z3_verification: Enable Z3 formal verification
        """
        self.cache_enabled = cache_enabled
        self.debug = debug
        self.max_cache_size = max_cache_size
        self.verification_timeout_seconds = verification_timeout_seconds
        self.enable_neural_parsing = enable_neural_parsing
        self.enable_z3_verification = enable_z3_verification
        
        # Core data structures
        self._verification_cache: Dict[str, ComplianceResult] = {}
        self.metrics_collector = get_metrics_collector()
        
        # Initialize neural parser if enabled
        self.neural_parser = None
        if enable_neural_parsing:
            try:
                self.neural_parser = NeuralContractParser(debug=debug)
                logger.info("Neural parser initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize neural parser: {e}")
                self.enable_neural_parsing = False
        
        # Initialize Z3 encoder if enabled
        self.z3_encoder = None
        if enable_z3_verification:
            try:
                self.z3_encoder = Z3Encoder(debug=debug)
                logger.info("Z3 encoder initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Z3 encoder: {e}")
                self.enable_z3_verification = False
        
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
        """Enhanced requirement verification with neural parsing and Z3 formal verification."""
        
        try:
            result = ComplianceResult(
                requirement_id=requirement.id,
                requirement_description=requirement.description,
                status=ComplianceStatus.COMPLIANT,  # Default assumption
                confidence=0.8
            )
            
            # Stage 1: Neural semantic analysis (if enabled)
            semantic_analysis = None
            if self.enable_neural_parsing and self.neural_parser:
                try:
                    enhanced_clauses = self.neural_parser.enhance_clauses(contract.clauses)
                    semantic_analysis = self._perform_semantic_analysis(enhanced_clauses, requirement)
                    self.metrics_collector.record_counter("verification.neural_parsing.success")
                except Exception as e:
                    logger.warning(f"Neural parsing failed for {requirement.id}: {e}")
                    self.metrics_collector.record_counter("verification.neural_parsing.failure")
            
            # Stage 2: Z3 formal verification (if enabled and applicable)
            formal_verification = None
            if self.enable_z3_verification and self.z3_encoder:
                try:
                    formal_verification = self._perform_formal_verification(contract, requirement)
                    self.metrics_collector.record_counter("verification.z3_verification.success")
                except Exception as e:
                    logger.warning(f"Z3 verification failed for {requirement.id}: {e}")
                    self.metrics_collector.record_counter("verification.z3_verification.failure")
            
            # Stage 3: Traditional keyword-based verification (fallback)
            keyword_verification = self._perform_keyword_verification(contract, requirement)
            
            # Combine results from all verification stages
            result = self._combine_verification_results(
                keyword_verification, semantic_analysis, formal_verification, requirement
            )
            
            # Enhanced metadata with all verification methods
            result.metadata = {
                "processed_clauses": len(contract.clauses),
                "verification_timestamp": datetime.now().isoformat(),
                "verification_methods": {
                    "keyword_matching": True,
                    "neural_parsing": semantic_analysis is not None,
                    "z3_formal": formal_verification is not None
                },
                "semantic_analysis": semantic_analysis,
                "formal_verification": formal_verification
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced requirement verification: {e}")
            raise ComplianceVerificationError(
                f"Failed to verify requirement {requirement.id}: {str(e)}",
                regulation_name=getattr(requirement, 'regulation_name', 'unknown'),
                requirement_id=requirement.id
            )
    
    def _perform_semantic_analysis(self, enhanced_clauses, requirement) -> Dict[str, Any]:
        """Perform semantic analysis using neural parser."""
        try:
            # Find semantically relevant clauses
            relevant_clauses = []
            semantic_types = []
            
            for clause in enhanced_clauses:
                # Check semantic type alignment
                if clause.semantic_type and any(cat in requirement.categories for cat in [clause.semantic_type]):
                    relevant_clauses.append(clause)
                    semantic_types.append(clause.semantic_type)
                
                # Check obligation strength for mandatory requirements
                if requirement.mandatory and clause.obligation_strength > 0.7:
                    if clause not in relevant_clauses:
                        relevant_clauses.append(clause)
            
            # Extract entities and temporal constraints
            entities = []
            temporal_constraints = []
            
            for clause in relevant_clauses:
                entities.extend(clause.legal_entities)
                temporal_constraints.extend(clause.temporal_expressions)
            
            return {
                "relevant_clauses_count": len(relevant_clauses),
                "semantic_types": list(set(semantic_types)),
                "entities": list(set(entities)),
                "temporal_constraints": list(set(temporal_constraints)),
                "avg_obligation_strength": sum(c.obligation_strength for c in relevant_clauses) / len(relevant_clauses) if relevant_clauses else 0,
                "confidence_score": 0.8 if relevant_clauses else 0.2
            }
            
        except Exception as e:
            logger.error(f"Error in semantic analysis: {e}")
            return {"error": str(e), "confidence_score": 0.1}
    
    def _perform_formal_verification(self, contract: ParsedContract, requirement) -> Dict[str, Any]:
        """Perform formal verification using Z3."""
        try:
            if not self.z3_encoder:
                return None
            
            # Choose appropriate Z3 encoding based on requirement type
            constraints = []
            
            # GDPR-specific formal verification
            if hasattr(requirement, 'article_reference') and 'GDPR' in str(requirement.article_reference):
                if 'data minimization' in requirement.description.lower():
                    constraints = self.z3_encoder.encode_gdpr_data_minimization(contract)
                elif 'purpose limitation' in requirement.description.lower():
                    constraints = self.z3_encoder.encode_gdpr_purpose_limitation(contract)
                elif 'retention' in requirement.description.lower():
                    constraints = self.z3_encoder.encode_gdpr_retention_limits(contract)
            
            # AI Act-specific formal verification
            elif hasattr(requirement, 'article_reference') and 'AI Act' in str(requirement.article_reference):
                if 'transparency' in requirement.description.lower():
                    constraints = self.z3_encoder.encode_ai_act_transparency(contract)
                elif 'human oversight' in requirement.description.lower():
                    constraints = self.z3_encoder.encode_ai_act_human_oversight(contract)
            
            # Verify constraints if any were generated
            if constraints:
                verification_results = []
                for constraint in constraints:
                    if constraint.constraint_type != "error":
                        constraint_result = self.z3_encoder.verify_constraint(constraint)
                        verification_results.append({
                            "constraint_name": constraint.name,
                            "status": constraint_result.status.value,
                            "confidence": constraint_result.confidence,
                            "issue": constraint_result.issue
                        })
                
                return {
                    "constraints_verified": len(verification_results),
                    "results": verification_results,
                    "overall_status": "compliant" if all(r["status"] == "compliant" for r in verification_results) else "non_compliant",
                    "confidence_score": sum(r["confidence"] for r in verification_results) / len(verification_results) if verification_results else 0
                }
            
            return {"message": "No applicable formal constraints", "confidence_score": 0.5}
            
        except Exception as e:
            logger.error(f"Error in formal verification: {e}")
            return {"error": str(e), "confidence_score": 0.1}
    
    def _perform_keyword_verification(self, contract: ParsedContract, requirement) -> ComplianceResult:
        """Traditional keyword-based verification."""
        result = ComplianceResult(
            requirement_id=requirement.id,
            requirement_description=requirement.description,
            status=ComplianceStatus.COMPLIANT,
            confidence=0.6  # Lower baseline confidence
        )
        
        # Enhanced keyword-based verification with error handling
        relevant_clauses = []
        processed_clauses = 0
        
        for clause in contract.clauses:
            try:
                processed_clauses += 1
                clause_text = clause.text.lower() if clause.text else ""
                
                if not clause_text.strip():
                    continue
                
                # Check if clause contains requirement keywords
                matching_keywords = []
                for keyword in requirement.keywords:
                    if keyword and keyword.lower() in clause_text:
                        matching_keywords.append(keyword)
                
                if matching_keywords:
                    relevant_clauses.append(clause.text)
                    
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
        elif relevant_clauses:
            clause_quality_score = self._assess_clause_quality(relevant_clauses, requirement)
            result.confidence = min(0.85, 0.6 + (clause_quality_score * 0.25))
            
            if clause_quality_score < 0.3:
                result.status = ComplianceStatus.PARTIAL
                result.issue = "Found relevant clauses but they may not fully address the requirement"
                result.suggestion = "Review and strengthen the relevant clauses"
        
        return result
    
    def _combine_verification_results(self, keyword_result, semantic_analysis, formal_verification, requirement) -> ComplianceResult:
        """Combine results from multiple verification methods."""
        
        # Start with keyword verification result
        final_result = keyword_result
        
        # Enhance with semantic analysis
        if semantic_analysis and semantic_analysis.get('confidence_score', 0) > 0.5:
            # Boost confidence if semantic analysis agrees
            if semantic_analysis.get('relevant_clauses_count', 0) > 0:
                final_result.confidence = min(0.95, final_result.confidence + 0.1)
                
                # Add semantic insights to suggestion
                if 'entities' in semantic_analysis:
                    entities = [e.split(':')[1] if ':' in e else e for e in semantic_analysis['entities'][:3]]
                    if entities:
                        final_result.suggestion = (final_result.suggestion or "") + f" Consider entities: {', '.join(entities)}"
        
        # Enhance with formal verification
        if formal_verification and formal_verification.get('confidence_score', 0) > 0.5:
            formal_status = formal_verification.get('overall_status')
            
            if formal_status == "compliant" and final_result.status == ComplianceStatus.COMPLIANT:
                # Both methods agree on compliance - high confidence
                final_result.confidence = min(0.98, final_result.confidence + 0.15)
                final_result.supporting_evidence = final_result.supporting_evidence or []
                final_result.supporting_evidence.append("Formal verification confirms compliance")
                
            elif formal_status == "non_compliant":
                # Formal verification found issues - override other results
                final_result.status = ComplianceStatus.NON_COMPLIANT
                final_result.confidence = 0.95
                final_result.issue = "Formal verification detected compliance violations"
                
                # Add specific issues from formal verification
                if 'results' in formal_verification:
                    issues = [r.get('issue') for r in formal_verification['results'] if r.get('issue')]
                    if issues:
                        final_result.suggestion = f"Address formal verification issues: {'; '.join(issues[:2])}"
        
        # Final confidence adjustment based on agreement between methods
        confidence_scores = [keyword_result.confidence]
        if semantic_analysis:
            confidence_scores.append(semantic_analysis.get('confidence_score', 0.5))
        if formal_verification:
            confidence_scores.append(formal_verification.get('confidence_score', 0.5))
        
        # Use weighted average with higher weight for methods that agree
        if len(confidence_scores) > 1:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            variance = sum((c - avg_confidence) ** 2 for c in confidence_scores) / len(confidence_scores)
            
            # Lower variance means methods agree more - boost confidence
            if variance < 0.1:  # Low variance - good agreement
                final_result.confidence = min(0.98, final_result.confidence + 0.05)
            elif variance > 0.3:  # High variance - poor agreement
                final_result.confidence = max(0.3, final_result.confidence - 0.1)
        
        return final_result
    
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