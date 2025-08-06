"""
Enhanced legal prover with neural parsing and formal verification.
"""

from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import cachetools
from dataclasses import dataclass

from .compliance_result import ComplianceResult, ComplianceReport, ComplianceStatus, ViolationSeverity
from .legal_prover import LegalProver
from ..parsing.contract_parser import ParsedContract
from ..regulations.base_regulation import BaseRegulation
from ..reasoning.solver import ComplianceSolver
from ..reasoning.proof_search import ProofSearcher
from ..explanation.explainer import ExplainabilityEngine

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for compliance verification."""
    total_time: float
    parsing_time: float
    verification_time: float
    explanation_time: float
    cache_hits: int
    cache_misses: int
    parallel_workers: int


class EnhancedLegalProver(LegalProver):
    """
    Enhanced legal prover with advanced capabilities:
    
    - Parallel verification processing
    - Advanced caching strategies  
    - Formal verification with Z3
    - Performance optimization
    - Comprehensive explanations
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        cache_size: int = 1000,
        cache_ttl: int = 3600,
        enable_formal_verification: bool = True,
        enable_parallel_processing: bool = True,
        debug: bool = False
    ):
        """
        Initialize enhanced legal prover.
        
        Args:
            max_workers: Maximum parallel workers
            cache_size: Cache size for verification results
            cache_ttl: Cache time-to-live in seconds
            enable_formal_verification: Use Z3 formal verification
            enable_parallel_processing: Enable parallel processing
            debug: Enable debug logging
        """
        super().__init__(cache_enabled=True, debug=debug)
        
        self.max_workers = max_workers
        self.enable_formal_verification = enable_formal_verification
        self.enable_parallel_processing = enable_parallel_processing
        
        # Advanced caching
        self._advanced_cache = cachetools.TTLCache(
            maxsize=cache_size,
            ttl=cache_ttl
        )
        
        # Specialized components
        self.compliance_solver = ComplianceSolver(debug=debug)
        self.proof_searcher = ProofSearcher()
        self.explainer = ExplainabilityEngine()
        
        # Performance tracking
        self.performance_metrics = PerformanceMetrics(
            total_time=0.0,
            parsing_time=0.0,
            verification_time=0.0,
            explanation_time=0.0,
            cache_hits=0,
            cache_misses=0,
            parallel_workers=max_workers
        )
        
        # Thread pool for parallel processing
        if self.enable_parallel_processing:
            self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        else:
            self.thread_pool = None
    
    async def verify_compliance_async(
        self,
        contract: ParsedContract,
        regulation: BaseRegulation,
        focus_areas: Optional[List[str]] = None,
        enable_explanations: bool = True
    ) -> Dict[str, ComplianceResult]:
        """
        Asynchronous compliance verification with enhanced features.
        
        Args:
            contract: Contract to verify
            regulation: Regulation to check against
            focus_areas: Specific areas to focus on
            enable_explanations: Generate explanations for results
            
        Returns:
            Enhanced compliance results with explanations
        """
        start_time = datetime.now()
        logger.info(f"Starting async compliance verification for {regulation.name}")
        
        try:
            # Get requirements to verify
            requirements = regulation.get_requirements()
            if focus_areas:
                requirements = {
                    req_id: req for req_id, req in requirements.items()
                    if any(area in req.categories for area in focus_areas)
                }
            
            # Parallel verification if enabled
            if self.enable_parallel_processing and len(requirements) > 1:
                results = await self._verify_requirements_parallel(
                    contract, requirements, regulation.name
                )
            else:
                results = await self._verify_requirements_sequential(
                    contract, requirements, regulation.name
                )
            
            # Generate explanations if requested
            if enable_explanations:
                await self._add_explanations_to_results(results, contract)
            
            # Update performance metrics
            total_time = (datetime.now() - start_time).total_seconds()
            self.performance_metrics.total_time = total_time
            self.performance_metrics.verification_time = total_time * 0.8  # Estimate
            
            logger.info(f"Async verification completed in {total_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Error in async verification: {e}")
            return {}
    
    def verify_with_formal_methods(
        self,
        contract: ParsedContract,
        requirement_id: str,
        regulation_type: str = "gdpr"
    ) -> ComplianceResult:
        """
        Verify compliance using formal methods and Z3 SMT solving.
        
        Args:
            contract: Contract to verify
            requirement_id: Specific requirement to verify
            regulation_type: Type of regulation
            
        Returns:
            Formal verification result
        """
        logger.info(f"Formal verification of {requirement_id}")
        
        if not self.enable_formal_verification:
            logger.warning("Formal verification disabled")
            return self._verify_requirement(contract, requirement_id)
        
        try:
            # Use SMT solver for formal verification
            result = self.compliance_solver.verify_compliance(
                contract, requirement_id, regulation_type
            )
            
            # Enhance with proof search
            if result.status == ComplianceStatus.NON_COMPLIANT:
                counter_examples = self.compliance_solver.find_counter_examples(
                    contract, requirement_id, regulation_type
                )
                
                if counter_examples:
                    result.counter_example = {
                        'examples': [ce.__dict__ for ce in counter_examples],
                        'count': len(counter_examples)
                    }
            
            # Add formal proof details
            if result.formal_proof:
                proof = self.proof_searcher.search_proof(
                    goal=f"compliant({requirement_id})",
                    premises=[],
                    contract=contract
                )
                
                if proof:
                    result.formal_proof += f" | Proof steps: {len(proof.steps)}"
            
            return result
            
        except Exception as e:
            logger.error(f"Formal verification error: {e}")
            return self._create_error_result(requirement_id, str(e))
    
    def generate_comprehensive_report(
        self,
        contract: ParsedContract,
        regulation: BaseRegulation,
        results: Optional[Dict[str, ComplianceResult]] = None,
        include_formal_proofs: bool = True,
        include_explanations: bool = True
    ) -> ComplianceReport:
        """
        Generate comprehensive compliance report with enhanced features.
        
        Args:
            contract: Contract analyzed
            regulation: Regulation checked
            results: Verification results
            include_formal_proofs: Include formal verification details
            include_explanations: Include natural language explanations
            
        Returns:
            Enhanced compliance report
        """
        if results is None:
            results = self.verify_compliance(contract, regulation)
        
        # Create base report
        report = super().generate_compliance_report(contract, regulation, results)
        
        # Enhance with formal verification details
        if include_formal_proofs:
            for req_id, result in results.items():
                if not result.formal_proof and self.enable_formal_verification:
                    # Generate formal proof
                    formal_result = self.verify_with_formal_methods(
                        contract, req_id, regulation.name.lower().replace(" ", "_")
                    )
                    result.formal_proof = formal_result.formal_proof
                    result.counter_example = formal_result.counter_example
        
        # Enhance with explanations
        if include_explanations:
            for result in results.values():
                if result.violations:
                    for violation in result.violations:
                        explanation = self.explainer.explain_violation(
                            violation, contract, "legal_team"
                        )
                        violation.suggested_fix = explanation.remediation_steps[0] if explanation.remediation_steps else violation.suggested_fix
        
        return report
    
    async def _verify_requirements_parallel(
        self,
        contract: ParsedContract,
        requirements: Dict[str, Any],
        regulation_name: str
    ) -> Dict[str, ComplianceResult]:
        """Verify requirements in parallel."""
        
        logger.info(f"Parallel verification of {len(requirements)} requirements")
        
        loop = asyncio.get_event_loop()
        
        # Create verification tasks
        tasks = []
        for req_id, requirement in requirements.items():
            task = loop.run_in_executor(
                self.thread_pool,
                self._verify_single_requirement_enhanced,
                contract, req_id, requirement, regulation_name
            )
            tasks.append((req_id, task))
        
        # Wait for all tasks to complete
        results = {}
        for req_id, task in tasks:
            try:
                result = await task
                results[req_id] = result
            except Exception as e:
                logger.error(f"Error verifying {req_id}: {e}")
                results[req_id] = self._create_error_result(req_id, str(e))
        
        return results
    
    async def _verify_requirements_sequential(
        self,
        contract: ParsedContract,
        requirements: Dict[str, Any],
        regulation_name: str
    ) -> Dict[str, ComplianceResult]:
        """Verify requirements sequentially."""
        
        logger.info(f"Sequential verification of {len(requirements)} requirements")
        
        results = {}
        for req_id, requirement in requirements.items():
            try:
                result = self._verify_single_requirement_enhanced(
                    contract, req_id, requirement, regulation_name
                )
                results[req_id] = result
            except Exception as e:
                logger.error(f"Error verifying {req_id}: {e}")
                results[req_id] = self._create_error_result(req_id, str(e))
        
        return results
    
    def _verify_single_requirement_enhanced(
        self,
        contract: ParsedContract,
        req_id: str,
        requirement: Any,
        regulation_name: str
    ) -> ComplianceResult:
        """Enhanced verification of single requirement."""
        
        # Check advanced cache first
        cache_key = self._get_advanced_cache_key(contract.id, req_id, regulation_name)
        
        if cache_key in self._advanced_cache:
            self.performance_metrics.cache_hits += 1
            return self._advanced_cache[cache_key]
        
        self.performance_metrics.cache_misses += 1
        
        # Use formal verification if enabled
        if self.enable_formal_verification:
            result = self.verify_with_formal_methods(
                contract, req_id, regulation_name.lower().replace(" ", "_")
            )
        else:
            # Fallback to basic verification
            result = self._verify_requirement(contract, requirement)
        
        # Cache result
        self._advanced_cache[cache_key] = result
        
        return result
    
    async def _add_explanations_to_results(
        self,
        results: Dict[str, ComplianceResult],
        contract: ParsedContract
    ) -> None:
        """Add explanations to verification results."""
        
        explanation_start = datetime.now()
        
        for result in results.values():
            if result.violations:
                for violation in result.violations:
                    try:
                        explanation = self.explainer.explain_violation(
                            violation, contract, "legal_team"
                        )
                        
                        # Enhance violation with explanation details
                        if not violation.suggested_fix and explanation.remediation_steps:
                            violation.suggested_fix = explanation.remediation_steps[0]
                        
                    except Exception as e:
                        logger.error(f"Error generating explanation: {e}")
        
        explanation_time = (datetime.now() - explanation_start).total_seconds()
        self.performance_metrics.explanation_time = explanation_time
    
    def _get_advanced_cache_key(self, contract_id: str, req_id: str, regulation_name: str) -> str:
        """Generate advanced cache key with versioning."""
        version = "v3"  # Cache version for enhanced prover
        return f"{version}:{contract_id}:{req_id}:{regulation_name}"
    
    def _create_error_result(self, requirement_id: str, error_message: str) -> ComplianceResult:
        """Create error result for failed verification."""
        return ComplianceResult(
            requirement_id=requirement_id,
            requirement_description=f"Requirement {requirement_id}",
            status=ComplianceStatus.UNKNOWN,
            confidence=0.0,
            issue=error_message
        )
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get performance metrics for the prover."""
        return self.performance_metrics
    
    def optimize_cache(self) -> Dict[str, Any]:
        """Optimize cache performance."""
        
        # Cache statistics
        cache_info = {
            "size": len(self._advanced_cache),
            "max_size": self._advanced_cache.maxsize,
            "hits": self.performance_metrics.cache_hits,
            "misses": self.performance_metrics.cache_misses,
            "hit_rate": self.performance_metrics.cache_hits / max(1, self.performance_metrics.cache_hits + self.performance_metrics.cache_misses)
        }
        
        # Clear expired entries
        # TTL cache handles this automatically
        
        logger.info(f"Cache optimization: {cache_info}")
        return cache_info
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        self._advanced_cache.clear()
        self.clear_cache()  # Parent cache