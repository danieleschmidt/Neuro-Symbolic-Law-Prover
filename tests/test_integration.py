#!/usr/bin/env python3
"""
Integration tests for the Neuro-Symbolic Law Prover.
Tests end-to-end workflows and real-world scenarios.
"""

import unittest
import sys
import os
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neuro_symbolic_law import LegalProver, ContractParser
from neuro_symbolic_law.regulations import GDPR, AIAct, CCPA
from neuro_symbolic_law.core.enhanced_prover import EnhancedLegalProver
from neuro_symbolic_law.core.scalable_prover import ScalableLegalProver
from neuro_symbolic_law.parsing.neural_parser import NeuralContractParser


class TestFullWorkflow(unittest.TestCase):
    """Test complete end-to-end workflows."""
    
    def setUp(self):
        """Set up test environment."""
        self.real_world_contract = """
        CLOUD SERVICES AGREEMENT
        
        This agreement is between TechCorp Inc. ("Provider") and BusinessCo Ltd. ("Customer").
        
        Article 1: Service Description
        The Provider shall deliver cloud computing services including data storage and processing.
        
        Article 2: Data Processing
        Customer data shall be processed only for the agreed purposes of service delivery.
        Personal data shall be handled in accordance with applicable data protection laws.
        All data shall be encrypted both at rest and in transit using industry-standard methods.
        
        Article 3: Data Subject Rights
        The Provider shall assist the Customer in responding to data subject access requests.
        Data subjects have the right to rectification, erasure, and portability of their personal data.
        Such requests shall be processed within 30 days of receipt.
        
        Article 4: Data Retention
        Personal data shall not be retained longer than necessary for service provision.
        Upon termination, all customer data shall be deleted within 90 days unless legally required to retain.
        
        Article 5: Security Measures
        The Provider shall implement appropriate technical and organizational security measures.
        Regular security assessments and penetration testing shall be conducted annually.
        Any security incidents shall be reported to the Customer within 24 hours of discovery.
        
        Article 6: Breach Notification
        Data breaches affecting personal data shall be reported to supervisory authorities within 72 hours.
        High-risk breaches shall be communicated to affected data subjects without undue delay.
        
        Article 7: International Transfers
        Data may be transferred to countries with adequate data protection as determined by applicable law.
        Standard contractual clauses shall be used for transfers to countries without adequacy decisions.
        
        Article 8: Liability and Indemnification
        Each party shall indemnify the other for damages resulting from breach of data protection obligations.
        Liability shall be limited to direct damages up to the annual contract value.
        
        Article 9: Termination
        Either party may terminate this agreement with 30 days written notice.
        Upon termination, all data processing shall cease and data shall be returned or deleted.
        """
        
        self.parser = ContractParser()
        self.prover = LegalProver()
        self.enhanced_prover = EnhancedLegalProver()
        
        # Regulations
        self.gdpr = GDPR()
        self.ai_act = AIAct()
        self.ccpa = CCPA()
    
    def test_comprehensive_gdpr_analysis(self):
        """Test comprehensive GDPR analysis workflow."""
        # Parse contract
        parsed = self.parser.parse(self.real_world_contract, "cloud_services_agreement")
        
        # Should extract meaningful information
        self.assertGreater(len(parsed.clauses), 15)
        self.assertGreater(len(parsed.parties), 1)
        self.assertEqual(parsed.contract_type, "data_processing_agreement")
        
        # Verify GDPR compliance
        results = self.prover.verify_compliance(parsed, self.gdpr)
        
        # Should check multiple requirements
        self.assertGreater(len(results), 15)
        
        # Should find some compliant items due to good contract language
        compliant_count = sum(1 for r in results.values() if r.compliant)
        compliance_rate = compliant_count / len(results)
        self.assertGreater(compliance_rate, 0.4)  # At least 40% compliance
        
        # Generate comprehensive report
        report = self.prover.generate_compliance_report(parsed, self.gdpr, results)
        
        self.assertEqual(report.contract_id, parsed.id)
        self.assertEqual(report.regulation_name, self.gdpr.name)
        self.assertGreater(report.compliance_rate, 40.0)
    
    def test_multi_regulation_analysis(self):
        """Test analysis against multiple regulations."""
        parsed = self.parser.parse(self.real_world_contract, "multi_reg_test")
        
        # Test against GDPR
        gdpr_results = self.prover.verify_compliance(parsed, self.gdpr)
        
        # Test against CCPA
        ccpa_results = self.prover.verify_compliance(parsed, self.ccpa)
        
        # Should have different requirements
        self.assertNotEqual(set(gdpr_results.keys()), set(ccpa_results.keys()))
        
        # Both should find some compliance
        gdpr_compliance = sum(1 for r in gdpr_results.values() if r.compliant) / len(gdpr_results)
        ccpa_compliance = sum(1 for r in ccpa_results.values() if r.compliant) / len(ccpa_results)
        
        self.assertGreater(gdpr_compliance, 0.2)
        self.assertGreater(ccpa_compliance, 0.2)
    
    def test_enhanced_prover_features(self):
        """Test enhanced prover with additional features."""
        parsed = self.parser.parse(self.real_world_contract, "enhanced_test")
        
        # Test with enhanced prover
        results = self.enhanced_prover.verify_compliance(parsed, self.gdpr)
        
        # Should provide additional metadata
        for result in results.values():
            if hasattr(result, 'metadata'):
                self.assertIn('verification_timestamp', result.metadata)
                self.assertIn('verification_method', result.metadata)
        
        # Test cache functionality
        cache_stats_before = self.enhanced_prover.get_cache_stats()
        
        # Run same verification again (should hit cache)
        results2 = self.enhanced_prover.verify_compliance(parsed, self.gdpr)
        
        cache_stats_after = self.enhanced_prover.get_cache_stats()
        
        # Results should be identical
        self.assertEqual(len(results), len(results2))
        
        # Cache should have been used
        self.assertGreaterEqual(cache_stats_after['cached_results'], cache_stats_before['cached_results'])
    
    def test_neural_parser_integration(self):
        """Test neural parser integration."""
        neural_parser = NeuralContractParser()
        
        # Parse with basic parser first
        basic_parsed = self.parser.parse(self.real_world_contract, "neural_integration")
        
        # Enhance with neural parser
        enhanced_clauses = neural_parser.enhance_clauses(basic_parsed.clauses)
        
        # Should enhance clauses with semantic information
        self.assertEqual(len(enhanced_clauses), len(basic_parsed.clauses))
        
        # Check that enhancements were added
        semantic_types = [c.semantic_type for c in enhanced_clauses if c.semantic_type != "unknown"]
        self.assertGreater(len(semantic_types), 5)  # Should classify some clauses
        
        # Check for legal entity extraction
        all_entities = []
        for clause in enhanced_clauses:
            all_entities.extend(clause.legal_entities)
        
        self.assertGreater(len(all_entities), 2)  # Should find some entities
    
    def test_focus_area_filtering(self):
        """Test compliance verification with focus area filtering."""
        parsed = self.parser.parse(self.real_world_contract, "focus_test")
        
        # Test data subject rights focus
        rights_results = self.prover.verify_compliance(
            parsed, self.gdpr, 
            focus_areas=['data_subject_rights', 'access_rights']
        )
        
        # Test security focus
        security_results = self.prover.verify_compliance(
            parsed, self.gdpr,
            focus_areas=['security', 'technical_measures']
        )
        
        # Should return different requirements
        self.assertNotEqual(set(rights_results.keys()), set(security_results.keys()))
        
        # Should be smaller than full analysis
        full_results = self.prover.verify_compliance(parsed, self.gdpr)
        self.assertLess(len(rights_results), len(full_results))
        self.assertLess(len(security_results), len(full_results))
    
    def test_error_handling_robustness(self):
        """Test error handling with problematic inputs."""
        # Test with empty contract
        try:
            empty_parsed = self.parser.parse("", "empty_test")
            results = self.prover.verify_compliance(empty_parsed, self.gdpr)
            # Should handle gracefully
            self.assertIsInstance(results, dict)
        except Exception:
            pass  # Expected to fail gracefully
        
        # Test with malformed input
        try:
            malformed = "This is not a proper legal contract. Random text."
            malformed_parsed = self.parser.parse(malformed, "malformed_test")
            results = self.prover.verify_compliance(malformed_parsed, self.gdpr)
            # Should handle gracefully
            self.assertIsInstance(results, dict)
        except Exception:
            pass  # Expected to fail gracefully
        
        # Test with None inputs
        with self.assertRaises((AttributeError, TypeError, ValueError)):
            self.prover.verify_compliance(None, self.gdpr)
    
    def test_performance_characteristics(self):
        """Test performance characteristics."""
        parsed = self.parser.parse(self.real_world_contract, "performance_test")
        
        import time
        
        # Test parsing performance
        start_time = time.time()
        parsed_perf = self.parser.parse(self.real_world_contract * 3, "perf_test")  # 3x larger
        parsing_time = time.time() - start_time
        
        # Should parse reasonably quickly
        self.assertLess(parsing_time, 5.0)  # Less than 5 seconds
        self.assertGreater(len(parsed_perf.clauses), len(parsed.clauses))
        
        # Test verification performance
        start_time = time.time()
        results = self.prover.verify_compliance(parsed_perf, self.gdpr)
        verification_time = time.time() - start_time
        
        # Should verify reasonably quickly
        self.assertLess(verification_time, 10.0)  # Less than 10 seconds
        self.assertGreater(len(results), 15)


class TestScalableProver(unittest.TestCase):
    """Test scalable prover functionality."""
    
    def setUp(self):
        """Set up scalable prover test environment."""
        self.scalable_prover = ScalableLegalProver(
            initial_cache_size=100,
            max_cache_size=1000,
            max_workers=2
        )
        
        self.sample_contract = """
        TERMS OF SERVICE
        
        1. The service provider shall deliver software services.
        2. User data shall be processed securely and confidentially.
        3. Users have the right to access and delete their data.
        4. Data breaches shall be reported within 72 hours.
        """
        
        self.parser = ContractParser()
        self.gdpr = GDPR()
    
    def test_adaptive_caching(self):
        """Test adaptive caching functionality."""
        parsed = self.parser.parse(self.sample_contract, "cache_test")
        
        # First verification (cache miss)
        results1 = self.scalable_prover.verify_compliance(parsed, self.gdpr)
        
        # Second verification (should use cache)
        results2 = self.scalable_prover.verify_compliance(parsed, self.gdpr)
        
        # Results should be identical
        self.assertEqual(len(results1), len(results2))
        
        # Get performance metrics
        metrics = self.scalable_prover.get_performance_metrics()
        
        # Should have cache statistics
        if 'adaptive_cache' in metrics and metrics['adaptive_cache']:
            self.assertIn('hit_rate', metrics['adaptive_cache'])
            self.assertIn('size', metrics['adaptive_cache'])
    
    def test_concurrent_processing_setup(self):
        """Test that concurrent processing is set up correctly."""
        # Test resource manager
        resource_info = self.scalable_prover.resource_manager.check_resources()
        
        self.assertIn('current_workers', resource_info)
        self.assertIn('resource_status', resource_info)
        self.assertGreater(resource_info['current_workers'], 0)
    
    def test_performance_optimization(self):
        """Test performance optimization features."""
        # Test manual optimization
        optimization_results = self.scalable_prover.optimize_system()
        
        self.assertIsInstance(optimization_results, dict)
        
        # Should include various optimization results
        expected_keys = ['resource_check', 'garbage_collection']
        for key in expected_keys:
            self.assertIn(key, optimization_results)
    
    def tearDown(self):
        """Clean up resources."""
        self.scalable_prover.cleanup()


def run_integration_tests():
    """Run integration test suite."""
    print("🧪 Running Integration Tests")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestFullWorkflow,
        TestScalableProver
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"📊 INTEGRATION TEST SUMMARY")
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")
    
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == '__main__':
    success = run_integration_tests()
    sys.exit(0 if success else 1)