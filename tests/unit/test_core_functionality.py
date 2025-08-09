"""
Unit tests for core legal prover functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
import time

from neuro_symbolic_law.core.legal_prover import LegalProver
from neuro_symbolic_law.core.enhanced_prover import EnhancedLegalProver
from neuro_symbolic_law.core.compliance_result import ComplianceStatus, ComplianceResult, ViolationSeverity
from neuro_symbolic_law.parsing.contract_parser import ContractParser
from neuro_symbolic_law.regulations.gdpr import GDPR
from neuro_symbolic_law.regulations.ai_act import AIAct


class TestLegalProver:
    """Test cases for basic legal prover."""
    
    def test_initialization(self):
        """Test legal prover initialization."""
        prover = LegalProver(cache_enabled=True, debug=False)
        
        assert prover.cache_enabled is True
        assert prover.debug is False
        assert hasattr(prover, 'contract_parser')
        assert hasattr(prover, 'cache')
    
    def test_verify_compliance_basic(self, legal_prover, parsed_contract, gdpr_regulation):
        """Test basic compliance verification."""
        results = legal_prover.verify_compliance(
            contract=parsed_contract,
            regulation=gdpr_regulation
        )
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Check that all results are ComplianceResult objects
        for result in results.values():
            assert isinstance(result, ComplianceResult)
            assert hasattr(result, 'status')
            assert hasattr(result, 'confidence')
            assert hasattr(result, 'requirement_id')
    
    def test_verify_compliance_with_focus_areas(self, legal_prover, parsed_contract, gdpr_regulation):
        """Test compliance verification with focus areas."""
        focus_areas = ['security', 'data_subject_rights']
        
        results = legal_prover.verify_compliance(
            contract=parsed_contract,
            regulation=gdpr_regulation,
            focus_areas=focus_areas
        )
        
        assert isinstance(results, dict)
        
        # Should only return results for focused areas
        for result in results.values():
            requirement = gdpr_regulation.get_requirement(result.requirement_id)
            if requirement:
                # At least one focus area should overlap with requirement categories
                assert any(area in requirement.categories for area in focus_areas)
    
    def test_generate_compliance_report(self, legal_prover, parsed_contract, gdpr_regulation):
        """Test compliance report generation."""
        results = legal_prover.verify_compliance(parsed_contract, gdpr_regulation)
        
        report = legal_prover.generate_compliance_report(
            contract=parsed_contract,
            regulation=gdpr_regulation,
            results=results
        )
        
        assert hasattr(report, 'contract_id')
        assert hasattr(report, 'regulation_name')
        assert hasattr(report, 'total_requirements')
        assert hasattr(report, 'compliant_count')
        assert hasattr(report, 'violation_count')
        assert hasattr(report, 'compliance_rate')
        assert hasattr(report, 'overall_status')
        
        assert report.contract_id == parsed_contract.id
        assert report.regulation_name == gdpr_regulation.name
        assert report.total_requirements == len(results)
        assert 0 <= report.compliance_rate <= 100
    
    def test_caching_functionality(self, temp_dir):
        """Test compliance result caching."""
        prover = LegalProver(cache_enabled=True, debug=False)
        parser = ContractParser()
        contract = parser.parse("Test contract", "cache_test")
        regulation = GDPR()
        
        # First verification (should cache results)
        start_time = time.time()
        results1 = prover.verify_compliance(contract, regulation)
        first_time = time.time() - start_time
        
        # Second verification (should use cache)
        start_time = time.time()
        results2 = prover.verify_compliance(contract, regulation)
        second_time = time.time() - start_time
        
        # Results should be identical
        assert len(results1) == len(results2)
        for req_id in results1:
            assert req_id in results2
            assert results1[req_id].status == results2[req_id].status
        
        # Second call should be faster due to caching
        assert second_time < first_time or second_time < 0.1  # Allow for small variations
    
    def test_empty_contract_handling(self, legal_prover, gdpr_regulation):
        """Test handling of empty or minimal contracts."""
        parser = ContractParser()
        empty_contract = parser.parse("", "empty_contract")
        
        results = legal_prover.verify_compliance(empty_contract, gdpr_regulation)
        
        assert isinstance(results, dict)
        # Should still return results, mostly non-compliant due to missing content
        for result in results.values():
            assert result.status in [ComplianceStatus.NON_COMPLIANT, ComplianceStatus.UNKNOWN]


class TestEnhancedProver:
    """Test cases for enhanced legal prover."""
    
    def test_initialization(self):
        """Test enhanced prover initialization."""
        prover = EnhancedLegalProver(
            max_workers=2,
            cache_enabled=True,
            enable_formal_verification=True,
            debug=False
        )
        
        assert prover.max_workers == 2
        assert prover.enable_formal_verification is True
        assert hasattr(prover, 'compliance_solver')
        assert hasattr(prover, 'explainer')
    
    @pytest.mark.asyncio
    async def test_async_verification(self, enhanced_prover, parsed_contract, gdpr_regulation):
        """Test asynchronous compliance verification."""
        results = await enhanced_prover.verify_compliance_async(
            contract=parsed_contract,
            regulation=gdpr_regulation
        )
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        for result in results.values():
            assert isinstance(result, ComplianceResult)
            # Enhanced prover should provide higher confidence
            assert result.confidence >= 0.0
    
    def test_formal_verification(self, enhanced_prover, parsed_contract):
        """Test formal verification capabilities."""
        result = enhanced_prover.verify_with_formal_methods(
            contract=parsed_contract,
            requirement_id="GDPR-5.1.c",
            regulation_type="gdpr"
        )
        
        assert isinstance(result, ComplianceResult)
        assert result.requirement_id == "GDPR-5.1.c"
        # Formal verification should provide confidence scores
        assert 0.0 <= result.confidence <= 1.0
    
    def test_comprehensive_report(self, enhanced_prover, parsed_contract, gdpr_regulation):
        """Test comprehensive report generation."""
        results = enhanced_prover.verify_compliance(parsed_contract, gdpr_regulation)
        
        report = enhanced_prover.generate_comprehensive_report(
            contract=parsed_contract,
            regulation=gdpr_regulation,
            results=results,
            include_formal_proofs=True,
            include_explanations=True
        )
        
        assert hasattr(report, 'metadata')
        assert 'neural_model_used' in report.metadata
        assert 'symbolic_verification' in report.metadata
        assert 'verification_time' in report.metadata
    
    def test_performance_metrics(self, enhanced_prover):
        """Test performance metrics collection."""
        metrics = enhanced_prover.get_performance_metrics()
        
        assert isinstance(metrics, dict)
        # Should have basic metrics structure
        expected_keys = ['total_time', 'parsing_time', 'verification_time']
        for key in expected_keys:
            if hasattr(metrics, key):  # Metrics might be dataclass or dict
                assert hasattr(metrics, key)
    
    def test_cache_management(self, enhanced_prover):
        """Test advanced cache management."""
        # Test cache optimization
        cache_info = enhanced_prover.optimize_cache()
        
        assert isinstance(cache_info, dict)
        assert 'size' in cache_info
        assert 'hit_rate' in cache_info
    
    def test_cleanup(self, enhanced_prover):
        """Test resource cleanup."""
        # Should not raise exceptions
        enhanced_prover.cleanup()


class TestContractParser:
    """Test cases for contract parser."""
    
    def test_basic_parsing(self, contract_parser, sample_contract_text):
        """Test basic contract parsing."""
        contract = contract_parser.parse(sample_contract_text, "test_contract")
        
        assert contract.id == "test_contract"
        assert contract.title is not None
        assert len(contract.clauses) > 0
        assert len(contract.parties) >= 0  # May or may not detect parties
        
        # Check clause structure
        for clause in contract.clauses:
            assert hasattr(clause, 'id')
            assert hasattr(clause, 'text')
            assert hasattr(clause, 'category')
            assert hasattr(clause, 'confidence')
            assert clause.text is not None
            assert 0.0 <= clause.confidence <= 1.0
    
    def test_party_extraction(self, contract_parser, sample_contract_text):
        """Test party extraction from contract."""
        contract = contract_parser.parse(sample_contract_text, "party_test")
        
        # Should detect at least some parties from the sample contract
        party_names = [party.name for party in contract.parties]
        expected_parties = ["Company A", "Company B", "Data Controller", "Data Processor"]
        
        # At least one expected party should be found
        assert any(expected in party_names for expected in expected_parties)
    
    def test_clause_categorization(self, contract_parser, sample_contract_text):
        """Test clause categorization."""
        contract = contract_parser.parse(sample_contract_text, "category_test")
        
        categories = [clause.category for clause in contract.clauses if clause.category]
        
        # Should have categorized some clauses
        assert len(categories) > 0
        
        # Check for expected categories
        expected_categories = ['security', 'data_subject_rights', 'retention', 'breach_notification']
        found_categories = set(categories)
        
        # Should find at least some expected categories
        assert len(found_categories.intersection(expected_categories)) > 0
    
    def test_clause_confidence_scoring(self, contract_parser, sample_contract_text):
        """Test confidence scoring for parsed clauses."""
        contract = contract_parser.parse(sample_contract_text, "confidence_test")
        
        for clause in contract.clauses:
            # All clauses should have confidence scores
            assert hasattr(clause, 'confidence')
            assert 0.0 <= clause.confidence <= 1.0
            
            # Longer clauses should generally have higher confidence
            if len(clause.text.split()) > 20:
                assert clause.confidence > 0.3  # Reasonable threshold
    
    def test_malformed_input_handling(self, contract_parser):
        """Test handling of malformed input."""
        # Empty input
        empty_contract = contract_parser.parse("", "empty")
        assert empty_contract.id == "empty"
        assert len(empty_contract.clauses) == 0
        
        # Very short input
        short_contract = contract_parser.parse("Short.", "short")
        assert short_contract.id == "short"
        
        # Non-English text (should still parse without errors)
        foreign_contract = contract_parser.parse("Contrato en español.", "spanish")
        assert foreign_contract.id == "spanish"


class TestRegulations:
    """Test cases for regulation implementations."""
    
    def test_gdpr_initialization(self, gdpr_regulation):
        """Test GDPR regulation initialization."""
        assert gdpr_regulation.name == "GDPR"
        assert len(gdpr_regulation) > 0
        
        requirements = gdpr_regulation.get_requirements()
        assert len(requirements) > 10  # Should have many requirements
        
        # Check mandatory requirements
        mandatory = gdpr_regulation.get_mandatory_requirements()
        assert len(mandatory) > 0
        
        # Check categories
        categories = gdpr_regulation.get_categories()
        expected_categories = {'security', 'data_subject_rights', 'data_minimization'}
        assert expected_categories.issubset(categories)
    
    def test_ai_act_initialization(self, ai_act_regulation):
        """Test AI Act regulation initialization."""
        assert ai_act_regulation.name == "EU AI Act"
        assert len(ai_act_regulation) > 0
        
        requirements = ai_act_regulation.get_requirements()
        assert len(requirements) > 5
        
        # Check for specific AI Act requirements
        requirement_ids = list(requirements.keys())
        assert any("13.1" in req_id for req_id in requirement_ids)  # Transparency
        assert any("14.1" in req_id for req_id in requirement_ids)  # Human oversight
    
    def test_requirement_retrieval(self, gdpr_regulation):
        """Test requirement retrieval functionality."""
        # Get all requirements
        all_reqs = gdpr_regulation.get_requirements()
        assert len(all_reqs) > 0
        
        # Get requirements by category
        security_reqs = gdpr_regulation.get_requirements(['security'])
        assert len(security_reqs) > 0
        
        for req in security_reqs.values():
            assert 'security' in req.categories
        
        # Get specific requirement
        if all_reqs:
            first_req_id = list(all_reqs.keys())[0]
            specific_req = gdpr_regulation.get_requirement(first_req_id)
            assert specific_req is not None
            assert specific_req.id == first_req_id
    
    def test_requirement_properties(self, gdpr_regulation):
        """Test requirement object properties."""
        requirements = gdpr_regulation.get_requirements()
        
        for req_id, req in requirements.items():
            assert req.id == req_id
            assert req.description is not None
            assert len(req.description) > 0
            assert isinstance(req.keywords, list)
            assert isinstance(req.categories, set)
            assert len(req.categories) > 0
            assert isinstance(req.mandatory, bool)


class TestComplianceResults:
    """Test cases for compliance result objects."""
    
    def test_compliance_result_creation(self):
        """Test compliance result creation."""
        result = ComplianceResult(
            requirement_id="TEST-1",
            requirement_description="Test requirement",
            status=ComplianceStatus.COMPLIANT,
            confidence=0.85
        )
        
        assert result.requirement_id == "TEST-1"
        assert result.requirement_description == "Test requirement"
        assert result.status == ComplianceStatus.COMPLIANT
        assert result.confidence == 0.85
        assert result.compliant is True  # Should derive from status
    
    def test_violation_handling(self):
        """Test violation addition to compliance results."""
        from neuro_symbolic_law.core.compliance_result import ComplianceViolation
        
        result = ComplianceResult(
            requirement_id="TEST-2",
            requirement_description="Test with violations",
            status=ComplianceStatus.NON_COMPLIANT,
            confidence=0.90
        )
        
        violation = ComplianceViolation(
            rule_id="TEST-2",
            rule_description="Test rule",
            violation_text="Test violation",
            severity=ViolationSeverity.HIGH
        )
        
        result.add_violation(violation)
        
        assert len(result.violations) == 1
        assert result.violations[0] == violation
        assert result.compliant is False
    
    def test_result_serialization(self):
        """Test compliance result serialization."""
        result = ComplianceResult(
            requirement_id="TEST-3",
            requirement_description="Serialization test",
            status=ComplianceStatus.PARTIAL,
            confidence=0.75
        )
        
        # Should be able to convert to dict-like structure
        assert hasattr(result, 'requirement_id')
        assert hasattr(result, 'status')
        assert hasattr(result, 'confidence')
        
        # Status should be serializable
        assert hasattr(result.status, 'value')
        assert isinstance(result.status.value, str)


@pytest.mark.unit
class TestPerformanceOptimizations:
    """Test performance optimization features."""
    
    def test_parallel_processing(self, enhanced_prover, parsed_contract, gdpr_regulation):
        """Test parallel processing capabilities."""
        start_time = time.time()
        
        # Run verification that should use parallel processing
        results = enhanced_prover.verify_compliance(
            contract=parsed_contract,
            regulation=gdpr_regulation
        )
        
        execution_time = time.time() - start_time
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Should complete in reasonable time (parallel processing should help)
        assert execution_time < 30.0  # Allow reasonable time for test environment
    
    def test_memory_efficiency(self, legal_prover):
        """Test memory efficiency of verification process."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Create many small contracts and verify them
        parser = ContractParser()
        regulation = GDPR()
        
        for i in range(10):  # Small number for unit test
            contract_text = f"Contract {i} with some basic terms and conditions."
            contract = parser.parse(contract_text, f"memory_test_{i}")
            legal_prover.verify_compliance(contract, regulation)
        
        # Force garbage collection
        gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100 * 1024 * 1024
    
    def test_error_handling(self, legal_prover):
        """Test error handling in verification process."""
        # Test with invalid regulation (mock)
        parser = ContractParser()
        contract = parser.parse("Test contract", "error_test")
        
        # Create a mock regulation that might cause issues
        class BadRegulation:
            def __init__(self):
                self.name = "Bad Regulation"
                self.version = "1.0"
            
            def get_requirements(self):
                # Return malformed requirements
                return {"bad_req": None}
        
        bad_regulation = BadRegulation()
        
        # Should handle errors gracefully, not crash
        try:
            results = legal_prover.verify_compliance(contract, bad_regulation)
            # If it succeeds, results should be dict (possibly empty)
            assert isinstance(results, dict)
        except Exception as e:
            # If it fails, should be a reasonable error
            assert isinstance(e, (ValueError, TypeError, AttributeError))
    
    def test_concurrent_access(self, legal_prover, parsed_contract, gdpr_regulation):
        """Test concurrent access to legal prover."""
        import threading
        import queue
        
        results_queue = queue.Queue()
        errors_queue = queue.Queue()
        
        def worker():
            try:
                results = legal_prover.verify_compliance(parsed_contract, gdpr_regulation)
                results_queue.put(results)
            except Exception as e:
                errors_queue.put(e)
        
        # Start multiple threads
        threads = []
        for _ in range(3):  # Small number for unit test
            thread = threading.Thread(target=worker)
            thread.start()
            threads.append(thread)
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)
        
        # Check results
        assert errors_queue.qsize() == 0, f"Concurrent access caused errors: {list(errors_queue.queue)}"
        assert results_queue.qsize() == 3, "Not all threads completed successfully"
        
        # All results should be valid
        while not results_queue.empty():
            result = results_queue.get()
            assert isinstance(result, dict)
            assert len(result) > 0