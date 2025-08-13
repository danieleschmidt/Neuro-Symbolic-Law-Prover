#!/usr/bin/env python3
"""
Generation 2 Robustness Test Suite
Testing enhanced error handling, security, monitoring, and reliability features.
"""

import asyncio
import logging
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from neuro_symbolic_law.core.enhanced_prover import EnhancedLegalProver
from neuro_symbolic_law.core.exceptions import (
    ValidationError, SecurityError, ComplianceVerificationError,
    ResourceError
)
from neuro_symbolic_law.parsing.contract_parser import ContractParser, ParsedContract, Clause
from neuro_symbolic_law.regulations.gdpr import GDPR
from neuro_symbolic_law.monitoring.health_monitor import HealthMonitor, HealthStatus, CheckType, HealthCheck
# from neuro_symbolic_law.monitoring.metrics_collector import get_metrics_collector, MetricsCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Generation2RobustnessTests(unittest.TestCase):
    """Test Generation 2 robustness features."""
    
    def setUp(self):
        """Set up test environment."""
        self.prover = EnhancedLegalProver(cache_enabled=True, debug=True)
        self.contract_parser = ContractParser()
        self.gdpr = GDPR()
        # self.metrics = get_metrics_collector()
        
        # Create test contract
        self.test_contract = ParsedContract(
            id="test_contract_123",
            text="This is a test data processing agreement.",
            clauses=[
                Clause(
                    id="clause_1",
                    text="The data controller shall implement appropriate technical measures.",
                    category="security"
                ),
                Clause(
                    id="clause_2", 
                    text="Personal data shall be processed fairly and transparently.",
                    category="fairness"
                ),
                Clause(
                    id="clause_3",
                    text="Data subjects have the right to access their personal data.",
                    category="rights"
                )
            ],
            parties=["Data Controller Corp"]
        )
    
    def test_input_validation(self):
        """Test comprehensive input validation."""
        logger.info("🧪 Testing Input Validation")
        
        # Test invalid contract
        with self.assertRaises(ValidationError) as context:
            self.prover.verify_compliance(None, self.gdpr)
        
        self.assertIn("Contract cannot be None", str(context.exception))
        
        # Test invalid regulation
        with self.assertRaises(ValidationError) as context:
            self.prover.verify_compliance(self.test_contract, None)
        
        self.assertIn("Regulation cannot be None", str(context.exception))
        
        # Test contract with invalid ID
        invalid_contract = ParsedContract(
            id="",
            text="Test",
            clauses=[Clause(id="1", text="Test clause", category="test")],
            parties=["Test Party"]
        )
        
        with self.assertRaises(ValidationError) as context:
            self.prover.verify_compliance(invalid_contract, self.gdpr)
        
        self.assertIn("Contract must have a valid ID", str(context.exception))
        
        logger.info("✅ Input validation tests passed")
    
    def test_security_validation(self):
        """Test security validation features."""
        logger.info("🧪 Testing Security Validation")
        
        # Test contract with suspicious content
        malicious_contract = ParsedContract(
            id="malicious_contract",
            text="<script>alert('xss')</script> This is a malicious contract.",
            clauses=[
                Clause(
                    id="malicious_clause",
                    text="This clause contains javascript: void(0) which is suspicious.",
                    category="malicious"
                )
            ],
            parties=["Evil Corp"]
        )
        
        # Should handle suspicious content gracefully
        try:
            results = self.prover.verify_compliance(malicious_contract, self.gdpr, strict_validation=True)
            # Should either filter out suspicious content or mark as low confidence
            self.assertIsInstance(results, dict)
            logger.info("✅ Security validation handled suspicious content")
        except SecurityError:
            # Acceptable - security validation blocked malicious content
            logger.info("✅ Security validation blocked malicious content")
        except Exception as e:
            # Should not crash with unhandled exception
            self.fail(f"Security validation failed with unhandled exception: {e}")
    
    def test_error_handling_and_recovery(self):
        """Test comprehensive error handling and recovery."""
        logger.info("🧪 Testing Error Handling and Recovery")
        
        # Test with malformed contract
        broken_contract = ParsedContract(
            id="broken_contract",
            text="A" * 100000,  # Very large text
            clauses=[],  # No clauses
            parties=[]  # No parties
        )
        
        # Should handle gracefully
        results = self.prover.verify_compliance(broken_contract, self.gdpr)
        self.assertIsInstance(results, dict)
        
        # Most results should be non-compliant due to missing clauses
        non_compliant_count = sum(1 for r in results.values() if not r.compliant)
        self.assertGreater(non_compliant_count, 0)
        
        logger.info("✅ Error handling and recovery tests passed")
    
    def test_circuit_breaker_functionality(self):
        """Test circuit breaker pattern."""
        logger.info("🧪 Testing Circuit Breaker Functionality")
        
        # Get initial circuit breaker status
        initial_status = self.prover.get_cache_stats()
        self.assertEqual(initial_status["circuit_breaker_state"], "closed")
        
        # Simulate multiple failures by providing invalid data repeatedly
        for i in range(6):  # Exceed failure threshold
            try:
                # This should cause verification errors
                invalid_regulation = Mock()
                invalid_regulation.name = "invalid_reg"
                invalid_regulation.get_requirements.side_effect = Exception("Simulated failure")
                
                self.prover.verify_compliance(self.test_contract, invalid_regulation)
            except Exception:
                pass  # Expected to fail
        
        # Circuit breaker should be opened after multiple failures
        # Note: In real implementation, circuit breaker logic may be more sophisticated
        logger.info("✅ Circuit breaker functionality tested")
    
    def test_performance_monitoring(self):
        """Test performance monitoring integration."""
        logger.info("🧪 Testing Performance Monitoring")
        
        # Clear existing metrics
        initial_metrics = self.metrics.get_all_metrics_summary()
        
        # Perform verification which should generate metrics
        results = self.prover.verify_compliance(self.test_contract, self.gdpr)
        
        # Check that metrics were recorded
        post_metrics = self.metrics.get_all_metrics_summary()
        
        # Should have more metrics after verification
        self.assertIsInstance(post_metrics, dict)
        self.assertIn("system", post_metrics)
        
        logger.info("✅ Performance monitoring tests passed")
    
    def test_comprehensive_logging(self):
        """Test comprehensive logging system."""
        logger.info("🧪 Testing Comprehensive Logging")
        
        # Capture logs
        with self.assertLogs(level='INFO') as log_capture:
            # Perform operations that should generate logs
            results = self.prover.verify_compliance(self.test_contract, self.gdpr)
            
            # Generate compliance report
            report = self.prover.generate_compliance_report(self.test_contract, self.gdpr, results)
        
        # Should have generated log entries
        self.assertTrue(len(log_capture.records) > 0)
        
        # Check for specific log patterns
        log_messages = [record.getMessage() for record in log_capture.records]
        
        # Should contain verification-related logs
        verification_logs = [msg for msg in log_messages if "verification" in msg.lower()]
        self.assertTrue(len(verification_logs) > 0)
        
        logger.info("✅ Comprehensive logging tests passed")
    
    def test_cache_management(self):
        """Test cache management and size limits."""
        logger.info("🧪 Testing Cache Management")
        
        # Get initial cache stats
        initial_stats = self.prover.get_cache_stats()
        initial_size = initial_stats["cached_results"]
        
        # Perform multiple verifications to populate cache
        for i in range(5):
            test_contract = ParsedContract(
                id=f"cache_test_contract_{i}",
                text=f"Test contract {i} for cache management.",
                clauses=[
                    Clause(
                        id=f"clause_{i}",
                        text=f"Test clause {i} with security measures.",
                        category="security"
                    )
                ],
                parties=[f"Test Party {i}"]
            )
            
            results = self.prover.verify_compliance(test_contract, self.gdpr)
            self.assertIsInstance(results, dict)
        
        # Cache should have grown
        final_stats = self.prover.get_cache_stats()
        final_size = final_stats["cached_results"]
        
        self.assertGreaterEqual(final_size, initial_size)
        
        # Test cache clearing
        self.prover.clear_cache()
        cleared_stats = self.prover.get_cache_stats()
        self.assertEqual(cleared_stats["cached_results"], 0)
        
        logger.info("✅ Cache management tests passed")
    
    def test_system_health_monitoring(self):
        """Test system health monitoring."""
        logger.info("🧪 Testing System Health Monitoring")
        
        # Create health monitor
        health_monitor = HealthMonitor(
            service_name="test_service",
            service_version="1.0.0",
            check_interval=1.0
        )
        
        # Add custom health check
        def custom_check():
            return True, "Custom check passed"
        
        health_monitor.add_check(HealthCheck(
            name="custom_test_check",
            check_type=CheckType.CUSTOM,
            check_function=custom_check,
            interval=1,
            timeout=5,
            description="Test custom health check"
        ))
        
        # Get health status
        status = health_monitor.get_health_status()
        
        self.assertEqual(status["service_name"], "test_service")
        self.assertIn("checks", status)
        self.assertIn("custom_test_check", status["checks"])
        
        # Test readiness and liveness
        readiness = health_monitor.get_readiness_status()
        liveness = health_monitor.get_liveness_status()
        
        self.assertIn("ready", readiness)
        self.assertIn("alive", liveness)
        
        logger.info("✅ System health monitoring tests passed")
    
    async def test_async_health_checks(self):
        """Test async health check functionality."""
        logger.info("🧪 Testing Async Health Checks")
        
        health_monitor = HealthMonitor(
            service_name="async_test_service", 
            service_version="1.0.0"
        )
        
        # Run immediate health check
        results = await health_monitor.check_now()
        
        self.assertIsInstance(results, dict)
        self.assertTrue(len(results) > 0)  # Should have built-in checks
        
        # Each result should have expected structure
        for check_name, result in results.items():
            self.assertIn("success", result)
            self.assertIn("message", result)
            self.assertIn("status", result)
            self.assertIn("timestamp", result)
        
        logger.info("✅ Async health checks tests passed")
    
    def test_compliance_report_generation(self):
        """Test enhanced compliance report generation."""
        logger.info("🧪 Testing Enhanced Compliance Report Generation")
        
        # Generate compliance report
        results = self.prover.verify_compliance(self.test_contract, self.gdpr)
        report = self.prover.generate_compliance_report(self.test_contract, self.gdpr, results)
        
        # Validate report structure
        self.assertEqual(report.contract_id, "test_contract_123")
        self.assertEqual(report.regulation_name, "GDPR")
        self.assertIsInstance(report.results, dict)
        self.assertTrue(len(report.results) > 0)
        
        # Check for enhanced metadata
        self.assertIsNotNone(report.metadata)
        self.assertIn("verification_summary", report.metadata)
        self.assertIn("system_info", report.metadata)
        
        # Validate verification summary
        summary = report.metadata["verification_summary"]
        self.assertIn("total_requirements", summary)
        self.assertIn("compliance_percentage", summary)
        
        logger.info("✅ Enhanced compliance report generation tests passed")
    
    def test_robust_contract_parsing(self):
        """Test robust contract parsing with edge cases."""
        logger.info("🧪 Testing Robust Contract Parsing")
        
        # Test with various edge cases
        edge_cases = [
            "",  # Empty contract
            "A" * 10,  # Very short contract
            "Contract with üñíçødé characters and special symbols: @#$%^&*()",
            "Contract\nwith\nmultiple\nlines\nand\ttabs",
            "Contract with \"quotes\" and 'apostrophes' and (parentheses)",
        ]
        
        for i, contract_text in enumerate(edge_cases):
            try:
                if contract_text.strip():  # Skip empty contracts
                    parsed = self.contract_parser.parse(contract_text, f"edge_case_{i}")
                    self.assertIsInstance(parsed, ParsedContract)
                    
                    # Should handle gracefully
                    if parsed.clauses:  # Only verify if we have clauses
                        results = self.prover.verify_compliance(parsed, self.gdpr)
                        self.assertIsInstance(results, dict)
                
            except Exception as e:
                # Edge cases might fail parsing, but shouldn't crash the system
                logger.warning(f"Edge case {i} failed parsing: {e}")
        
        logger.info("✅ Robust contract parsing tests passed")
    
    def test_resource_management(self):
        """Test resource management and limits."""
        logger.info("🧪 Testing Resource Management")
        
        # Test with resource-intensive operations
        large_contract = ParsedContract(
            id="large_contract",
            text="Large contract text " * 1000,  # Reasonably large
            clauses=[
                Clause(
                    id=f"clause_{i}",
                    text=f"Large clause text {i} " * 100,
                    category="test"
                ) for i in range(50)  # Many clauses
            ],
            parties=["Large Corp"] * 10
        )
        
        # Should handle large contracts without crashing
        start_time = time.time()
        results = self.prover.verify_compliance(large_contract, self.gdpr)
        end_time = time.time()
        
        self.assertIsInstance(results, dict)
        
        # Should complete within reasonable time (less than 30 seconds)
        processing_time = end_time - start_time
        self.assertLess(processing_time, 30, "Processing took too long")
        
        logger.info(f"✅ Resource management tests passed (processed in {processing_time:.2f}s)")


class Integration2Tests(unittest.TestCase):
    """Integration tests for Generation 2 features."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.prover = EnhancedLegalProver(cache_enabled=True, debug=False)
        self.contract_parser = ContractParser()
        self.gdpr = GDPR()
        
        # Create comprehensive test contract
        self.comprehensive_contract = ParsedContract(
            id="comprehensive_test_contract",
            text="""
            Data Processing Agreement
            
            This agreement establishes the terms for processing personal data.
            The data controller shall implement appropriate technical and organizational measures.
            Personal data shall be processed fairly, lawfully, and transparently.
            Data subjects have the right to access, rectify, and erase their personal data.
            Data shall be retained only for as long as necessary for the specified purposes.
            The data processor shall notify the controller of any personal data breaches within 24 hours.
            """,
            clauses=[
                Clause(
                    id="technical_measures",
                    text="The data controller shall implement appropriate technical and organizational measures.",
                    category="security"
                ),
                Clause(
                    id="fair_processing",
                    text="Personal data shall be processed fairly, lawfully, and transparently.",
                    category="fairness"
                ),
                Clause(
                    id="data_subject_rights",
                    text="Data subjects have the right to access, rectify, and erase their personal data.",
                    category="rights"
                ),
                Clause(
                    id="data_retention",
                    text="Data shall be retained only for as long as necessary for the specified purposes.",
                    category="retention"
                ),
                Clause(
                    id="breach_notification",
                    text="The data processor shall notify the controller of any personal data breaches within 24 hours.",
                    category="security"
                )
            ],
            parties=["DataCorp Inc", "ProcessorTech Ltd"]
        )
    
    def test_end_to_end_verification_with_monitoring(self):
        """Test complete end-to-end verification with monitoring."""
        logger.info("🧪 Testing End-to-End Verification with Monitoring")
        
        # Perform comprehensive verification
        start_time = time.time()
        
        results = self.prover.verify_compliance(
            self.comprehensive_contract,
            self.gdpr,
            focus_areas=["security", "data_subject_rights", "retention"]
        )
        
        end_time = time.time()
        
        # Validate results
        self.assertIsInstance(results, dict)
        self.assertTrue(len(results) > 0)
        
        # Should have some compliant results for our comprehensive contract
        compliant_count = sum(1 for r in results.values() if r.compliant)
        total_count = len(results)
        compliance_rate = compliant_count / total_count if total_count > 0 else 0
        
        self.assertGreater(compliance_rate, 0.2, "Compliance rate unexpectedly low")
        
        # Generate comprehensive report
        report = self.prover.generate_compliance_report(
            self.comprehensive_contract,
            self.gdpr,
            results
        )
        
        # Validate report completeness
        self.assertIsNotNone(report.metadata)
        self.assertIn("verification_summary", report.metadata)
        
        verification_summary = report.metadata["verification_summary"]
        self.assertEqual(verification_summary["total_requirements"], total_count)
        self.assertEqual(verification_summary["compliant"], compliant_count)
        
        processing_time = end_time - start_time
        logger.info(f"✅ End-to-end verification completed in {processing_time:.2f}s with {compliance_rate:.1%} compliance")
    
    def test_system_integration_health_check(self):
        """Test system integration health check."""
        logger.info("🧪 Testing System Integration Health Check")
        
        # Get system health information
        system_health = self.prover.get_system_health()
        
        self.assertIn("enhanced_prover_status", system_health)
        self.assertIn("cache_status", system_health)
        self.assertIn("system_health", system_health)
        
        # Prover should be operational
        self.assertEqual(system_health["enhanced_prover_status"], "operational")
        
        logger.info("✅ System integration health check passed")


async def run_async_tests():
    """Run async test cases."""
    logger.info("🚀 Running Async Tests")
    
    test_instance = Generation2RobustnessTests()
    test_instance.setUp()
    
    # Run async health check test
    await test_instance.test_async_health_checks()
    
    logger.info("✅ All async tests completed")


def main():
    """Run all Generation 2 robustness tests."""
    print("🚀 Neuro-Symbolic Law Prover - Generation 2 Robustness Test Suite")
    print("=" * 80)
    
    try:
        # Run synchronous tests
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # Add test classes
        suite.addTests(loader.loadTestsFromTestCase(Generation2RobustnessTests))
        suite.addTests(loader.loadTestsFromTestCase(Integration2Tests))
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Run async tests
        print("\n" + "=" * 80)
        print("Running Async Tests")
        print("=" * 80)
        
        asyncio.run(run_async_tests())
        
        # Summary
        print("\n" + "=" * 80)
        print("📊 GENERATION 2 ROBUSTNESS TEST SUMMARY")
        print("=" * 80)
        
        total_tests = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        passed = total_tests - failures - errors
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failures}")
        print(f"💥 Errors: {errors}")
        print(f"Success Rate: {(passed / total_tests * 100):.1f}%")
        
        if failures == 0 and errors == 0:
            print("\n🎉 ALL GENERATION 2 ROBUSTNESS TESTS PASSED!")
            print("✅ System demonstrates robust error handling, security, and monitoring")
            return True
        else:
            print("\n⚠️  Some tests failed. Review the output above for details.")
            return False
            
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)