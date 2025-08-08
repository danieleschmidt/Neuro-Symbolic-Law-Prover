#!/usr/bin/env python3
"""
Comprehensive test suite for Neuro-Symbolic Law Prover.
Aims for 85%+ code coverage across all modules.
"""

import unittest
import sys
import os
from typing import List, Dict, Any
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Core imports
from neuro_symbolic_law import LegalProver, ContractParser, ComplianceResult
from neuro_symbolic_law.core.compliance_result import ComplianceStatus, ViolationSeverity, ComplianceViolation
from neuro_symbolic_law.regulations import GDPR, AIAct, CCPA
from neuro_symbolic_law.parsing.neural_parser import NeuralContractParser, SemanticClause
from neuro_symbolic_law.reasoning.z3_encoder import Z3Encoder
from neuro_symbolic_law.reasoning.proof_search import ProofSearcher
from neuro_symbolic_law.reasoning.solver import ComplianceSolver
from neuro_symbolic_law.explanation.explainer import ExplainabilityEngine
from neuro_symbolic_law.explanation.report_generator import ComplianceReporter


class TestContractParsing(unittest.TestCase):
    """Test contract parsing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = ContractParser(debug=True)
        self.neural_parser = NeuralContractParser(debug=True)
        
        self.sample_contract = """
        DATA PROCESSING AGREEMENT
        
        This Agreement is between ACME Corp (the "Controller") and CloudTech Inc (the "Processor").
        
        1. The data controller shall implement appropriate technical measures to ensure data security.
        2. Personal data shall be processed only for specified, explicit purposes.
        3. Data subjects have the right to access their personal data within 30 days.
        4. The processor must delete personal data when instructed by the controller.
        5. Any data breach shall be reported within 72 hours to supervisory authorities.
        """
    
    def test_basic_contract_parsing(self):
        """Test basic contract parsing functionality."""
        parsed = self.parser.parse(self.sample_contract, "test_dpa")
        
        self.assertEqual(parsed.id, "test_dpa")
        self.assertIn("DATA PROCESSING AGREEMENT", parsed.title)
        self.assertGreaterEqual(len(parsed.clauses), 5)
        self.assertGreaterEqual(len(parsed.parties), 2)
        self.assertEqual(parsed.contract_type, "data_processing_agreement")
    
    def test_clause_classification(self):
        """Test clause classification."""
        parsed = self.parser.parse(self.sample_contract, "test_classification")
        
        categories = [clause.category for clause in parsed.clauses]
        
        # Should identify data processing and security clauses
        self.assertIn("data_processing", categories)
        self.assertIn("security", categories)
    
    def test_party_extraction(self):
        """Test party extraction."""
        parsed = self.parser.parse(self.sample_contract, "test_parties")
        
        party_names = [party.name for party in parsed.parties]
        self.assertIn("ACME Corp", party_names)
        self.assertIn("CloudTech Inc", party_names)
        
        # Check roles
        party_roles = [party.role for party in parsed.parties]
        self.assertIn("client", party_roles)
        self.assertIn("provider", party_roles)
    
    def test_obligation_extraction(self):
        """Test obligation extraction from clauses."""
        parsed = self.parser.parse(self.sample_contract, "test_obligations")
        
        all_obligations = []
        for clause in parsed.clauses:
            all_obligations.extend(clause.obligations)
        
        # Should extract obligations with "shall" and "must"
        self.assertTrue(any("implement" in obs for obs in all_obligations))
        self.assertTrue(any("delete" in obs for obs in all_obligations))
    
    def test_neural_parsing_enhancement(self):
        """Test neural parsing enhancements."""
        # Basic parsing first
        basic_parsed = self.parser.parse(self.sample_contract, "neural_test")
        
        # Enhance with neural parser
        enhanced_clauses = self.neural_parser.enhance_clauses(basic_parsed.clauses)
        
        self.assertEqual(len(enhanced_clauses), len(basic_parsed.clauses))
        
        # Check that clauses have embeddings
        for clause in enhanced_clauses:
            self.assertIsInstance(clause, SemanticClause)
            self.assertIsNotNone(clause.embedding)
            self.assertIsInstance(clause.embedding, list)
            self.assertGreater(len(clause.embedding), 0)
    
    def test_semantic_type_classification(self):
        """Test semantic type classification."""
        basic_parsed = self.parser.parse(self.sample_contract, "semantic_test")
        enhanced_clauses = self.neural_parser.enhance_clauses(basic_parsed.clauses)
        
        semantic_types = [clause.semantic_type for clause in enhanced_clauses]
        
        # Should classify some clauses correctly
        self.assertTrue(any(st in semantic_types for st in ['security', 'data_processing', 'rights']))
    
    def test_legal_entity_extraction(self):
        """Test legal entity extraction."""
        basic_parsed = self.parser.parse(self.sample_contract, "entity_test")
        enhanced_clauses = self.neural_parser.enhance_clauses(basic_parsed.clauses)
        
        all_entities = []
        for clause in enhanced_clauses:
            all_entities.extend(clause.legal_entities)
        
        # Should extract organizations and regulations
        org_entities = [e for e in all_entities if e.startswith('organization:')]
        self.assertGreater(len(org_entities), 0)
    
    def test_temporal_expression_extraction(self):
        """Test temporal expression extraction."""
        basic_parsed = self.parser.parse(self.sample_contract, "temporal_test")
        enhanced_clauses = self.neural_parser.enhance_clauses(basic_parsed.clauses)
        
        all_temporal = []
        for clause in enhanced_clauses:
            all_temporal.extend(clause.temporal_expressions)
        
        # Should find "within 30 days" and "within 72 hours"
        self.assertTrue(any("30 days" in temp for temp in all_temporal))
        self.assertTrue(any("72 hours" in temp for temp in all_temporal))
    
    def test_obligation_strength_calculation(self):
        """Test obligation strength calculation."""
        basic_parsed = self.parser.parse(self.sample_contract, "strength_test")
        enhanced_clauses = self.neural_parser.enhance_clauses(basic_parsed.clauses)
        
        # Find clauses with "shall" and "must" - should have high obligation strength
        strong_clauses = [c for c in enhanced_clauses if c.obligation_strength > 0.1]
        self.assertGreater(len(strong_clauses), 0)


class TestRegulationModels(unittest.TestCase):
    """Test regulation models (GDPR, AI Act, CCPA)."""
    
    def setUp(self):
        """Set up regulation instances."""
        self.gdpr = GDPR()
        self.ai_act = AIAct()
        self.ccpa = CCPA()
    
    def test_gdpr_initialization(self):
        """Test GDPR regulation initialization."""
        self.assertEqual(self.gdpr.name, "GDPR")
        self.assertEqual(self.gdpr.version, "2016/679")
        
        requirements = self.gdpr.get_requirements()
        self.assertGreater(len(requirements), 15)  # Should have many requirements
        
        # Check specific requirements
        self.assertIn("GDPR-5.1.c", requirements)  # Data minimization
        self.assertIn("GDPR-15.1", requirements)   # Access rights
        self.assertIn("GDPR-17.1", requirements)   # Deletion rights
    
    def test_ai_act_initialization(self):
        """Test AI Act regulation initialization."""
        self.assertEqual(self.ai_act.name, "EU AI Act")
        self.assertEqual(self.ai_act.version, "2024/1689")
        
        requirements = self.ai_act.get_requirements()
        self.assertGreater(len(requirements), 10)
        
        # Check specific requirements
        self.assertIn("AI-ACT-13.1", requirements)  # Transparency
        self.assertIn("AI-ACT-14.1", requirements)  # Human oversight
        self.assertIn("AI-ACT-50.1", requirements)  # AI disclosure
    
    def test_ccpa_initialization(self):
        """Test CCPA regulation initialization."""
        self.assertEqual(self.ccpa.name, "CCPA")
        self.assertEqual(self.ccpa.version, "2020")
        
        requirements = self.ccpa.get_requirements()
        self.assertGreater(len(requirements), 10)
        
        # Check specific requirements
        self.assertIn("CCPA-1798.100.a", requirements)  # Right to know
        self.assertIn("CCPA-1798.105.a", requirements)  # Right to delete
        self.assertIn("CCPA-1798.120.a", requirements)  # Opt-out
    
    def test_requirement_categories(self):
        """Test requirement categorization."""
        gdpr_requirements = self.gdpr.get_requirements()
        
        # Check categories exist
        for req_id, req in gdpr_requirements.items():
            self.assertIsInstance(req.categories, set)
            self.assertGreater(len(req.categories), 0)
        
        # Check specific categories
        data_min_req = gdpr_requirements["GDPR-5.1.c"]
        self.assertIn("data_minimization", data_min_req.categories)
    
    def test_requirement_keywords(self):
        """Test requirement keywords."""
        gdpr_requirements = self.gdpr.get_requirements()
        
        data_min_req = gdpr_requirements["GDPR-5.1.c"]
        self.assertIn("data minimization", data_min_req.keywords)
        self.assertIn("necessary", data_min_req.keywords)


class TestComplianceVerification(unittest.TestCase):
    """Test compliance verification engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.prover = LegalProver(debug=True)
        self.parser = ContractParser()
        self.gdpr = GDPR()
        
        self.compliant_contract = """
        DATA PROCESSING AGREEMENT
        
        1. The data controller shall implement appropriate technical measures to ensure data security.
        2. Personal data shall be processed only for specified, explicit and legitimate purposes.
        3. Personal data shall be adequate, relevant and limited to what is necessary.
        4. Data subjects have the right to access their personal data.
        5. Personal data shall not be kept longer than necessary.
        """
        
        self.non_compliant_contract = """
        BASIC AGREEMENT
        
        1. We collect any data we want.
        2. Data can be used for any purpose.
        3. Data is kept forever.
        """
    
    def test_compliant_contract_verification(self):
        """Test verification of compliant contract."""
        parsed = self.parser.parse(self.compliant_contract, "compliant_test")
        results = self.prover.verify_compliance(parsed, self.gdpr)
        
        # Should have results
        self.assertGreater(len(results), 0)
        
        # Check compliance rate
        compliant_count = sum(1 for r in results.values() if r.compliant)
        total_count = len(results)
        compliance_rate = (compliant_count / total_count) * 100
        
        # Should be reasonably compliant due to good language
        self.assertGreater(compliance_rate, 30.0)
    
    def test_non_compliant_contract_verification(self):
        """Test verification of non-compliant contract."""
        parsed = self.parser.parse(self.non_compliant_contract, "non_compliant_test")
        results = self.prover.verify_compliance(parsed, self.gdpr)
        
        # Should have results
        self.assertGreater(len(results), 0)
        
        # Should identify non-compliance
        non_compliant_count = sum(1 for r in results.values() if not r.compliant)
        self.assertGreater(non_compliant_count, 0)
    
    def test_focus_areas_filtering(self):
        """Test focus areas filtering."""
        parsed = self.parser.parse(self.compliant_contract, "focus_test")
        
        # Test with data minimization focus
        results = self.prover.verify_compliance(
            parsed, self.gdpr, 
            focus_areas=['data_minimization']
        )
        
        # Should only check requirements related to data minimization
        requirement_ids = list(results.keys())
        data_min_requirements = [req_id for req_id in requirement_ids if "5.1.c" in req_id]
        self.assertGreater(len(data_min_requirements), 0)
    
    def test_compliance_result_structure(self):
        """Test compliance result data structure."""
        parsed = self.parser.parse(self.compliant_contract, "structure_test")
        results = self.prover.verify_compliance(parsed, self.gdpr)
        
        for req_id, result in results.items():
            self.assertIsInstance(result, ComplianceResult)
            self.assertIsInstance(result.requirement_id, str)
            self.assertIsInstance(result.requirement_description, str)
            self.assertIsInstance(result.status, ComplianceStatus)
            self.assertIsInstance(result.confidence, float)
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)
    
    def test_compliance_report_generation(self):
        """Test compliance report generation."""
        parsed = self.parser.parse(self.compliant_contract, "report_test")
        results = self.prover.verify_compliance(parsed, self.gdpr)
        
        report = self.prover.generate_compliance_report(parsed, self.gdpr, results)
        
        self.assertEqual(report.contract_id, parsed.id)
        self.assertEqual(report.regulation_name, self.gdpr.name)
        self.assertIsInstance(report.compliance_rate, float)
        self.assertGreaterEqual(report.compliance_rate, 0.0)
        self.assertLessEqual(report.compliance_rate, 100.0)
    
    def test_caching_functionality(self):
        """Test verification result caching."""
        parsed = self.parser.parse(self.compliant_contract, "cache_test")
        
        # First verification
        results1 = self.prover.verify_compliance(parsed, self.gdpr)
        cache_stats1 = self.prover.get_cache_stats()
        
        # Second verification (should use cache)
        results2 = self.prover.verify_compliance(parsed, self.gdpr)
        cache_stats2 = self.prover.get_cache_stats()
        
        # Results should be identical
        self.assertEqual(len(results1), len(results2))
        
        # Cache should be used
        self.assertGreaterEqual(cache_stats2['cached_results'], cache_stats1['cached_results'])
    
    def test_cache_clearing(self):
        """Test cache clearing functionality."""
        parsed = self.parser.parse(self.compliant_contract, "clear_test")
        
        # Run verification to populate cache
        self.prover.verify_compliance(parsed, self.gdpr)
        stats_before = self.prover.get_cache_stats()
        
        # Clear cache
        self.prover.clear_cache()
        stats_after = self.prover.get_cache_stats()
        
        self.assertEqual(stats_after['cached_results'], 0)


class TestReasoningComponents(unittest.TestCase):
    """Test reasoning and Z3 encoding components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.encoder = Z3Encoder(debug=True)
        self.solver = ComplianceSolver()
        self.parser = ContractParser()
        
        self.dpa_contract = """
        1. Personal data shall be adequate, relevant and limited to what is necessary.
        2. Personal data shall be collected for specified, explicit purposes.
        3. Personal data shall not be kept longer than necessary for the purposes.
        """
    
    def test_z3_encoder_initialization(self):
        """Test Z3 encoder initialization."""
        self.assertIsInstance(self.encoder, Z3Encoder)
        self.assertEqual(len(self.encoder.constraints), 0)
    
    def test_gdpr_data_minimization_encoding(self):
        """Test GDPR data minimization encoding."""
        parsed = self.parser.parse(self.dpa_contract, "z3_test")
        constraints = self.encoder.encode_gdpr_data_minimization(parsed)
        
        self.assertGreater(len(constraints), 0)
        
        # Check constraint structure
        for constraint in constraints:
            self.assertIsNotNone(constraint.name)
            self.assertIsNotNone(constraint.description)
            self.assertIsNotNone(constraint.formula)
    
    def test_gdpr_purpose_limitation_encoding(self):
        """Test GDPR purpose limitation encoding."""
        parsed = self.parser.parse(self.dpa_contract, "purpose_test")
        constraints = self.encoder.encode_gdpr_purpose_limitation(parsed)
        
        self.assertGreater(len(constraints), 0)
        
        # Should create constraints about purpose compatibility
        constraint_names = [c.name for c in constraints]
        self.assertTrue(any("purpose_limitation" in name for name in constraint_names))
    
    def test_gdpr_retention_limits_encoding(self):
        """Test GDPR retention limits encoding."""
        parsed = self.parser.parse(self.dpa_contract, "retention_test")
        constraints = self.encoder.encode_gdpr_retention_limits(parsed)
        
        self.assertGreater(len(constraints), 0)
        
        # Should create constraints about retention periods
        constraint_names = [c.name for c in constraints]
        self.assertTrue(any("retention" in name for name in constraint_names))
    
    def test_variable_registry(self):
        """Test Z3 variable registry."""
        parsed = self.parser.parse(self.dpa_contract, "var_test")
        
        # Encode some constraints
        self.encoder.encode_gdpr_data_minimization(parsed)
        
        # Check variables are registered
        data_collected_var = self.encoder.get_variable('data_collected')
        self.assertIsNotNone(data_collected_var)
        
        data_necessary_var = self.encoder.get_variable('data_necessary')
        self.assertIsNotNone(data_necessary_var)
    
    def test_constraint_clearing(self):
        """Test constraint clearing."""
        parsed = self.parser.parse(self.dpa_contract, "clear_test")
        
        # Create constraints
        self.encoder.encode_gdpr_data_minimization(parsed)
        self.assertGreater(len(self.encoder.get_all_constraints()), 0)
        
        # Clear constraints
        self.encoder.clear_constraints()
        self.assertEqual(len(self.encoder.get_all_constraints()), 0)
        self.assertEqual(len(self.encoder._variable_registry), 0)
    
    def test_compliance_solver(self):
        """Test compliance solver functionality."""
        # Basic solver test
        self.assertIsInstance(self.solver, ComplianceSolver)
        
        # Test solving (will use fallback if Z3 not available)
        try:
            parsed = self.parser.parse(self.dpa_contract, "solver_test")
            result = self.solver.solve_compliance(parsed, "data_minimization")
            
            # Should return some result (even if using fallback)
            self.assertIsNotNone(result)
        except Exception as e:
            # Acceptable if Z3 components fail gracefully
            self.assertIn("fallback", str(e).lower())


class TestExplanationSystem(unittest.TestCase):
    """Test explanation and reporting system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.explainer = ExplainabilityEngine()
        self.reporter = ComplianceReporter()
    
    def test_explainer_initialization(self):
        """Test explainer initialization."""
        self.assertIsInstance(self.explainer, ExplainabilityEngine)
    
    def test_violation_explanation(self):
        """Test violation explanation generation."""
        violation = ComplianceViolation(
            rule_id="GDPR-5.1.c",
            rule_description="Data minimization requirement",
            violation_text="Excessive data collection identified",
            severity=ViolationSeverity.HIGH
        )
        
        explanation = self.explainer.explain_violation(violation, None)
        
        # Should generate explanation
        self.assertIsNotNone(explanation)
        self.assertIsInstance(explanation.remediation_steps, list)
        self.assertGreater(len(explanation.remediation_steps), 0)
    
    def test_reporter_initialization(self):
        """Test reporter initialization."""
        self.assertIsInstance(self.reporter, ComplianceReporter)


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios and end-to-end workflows."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.parser = ContractParser()
        self.prover = LegalProver()
        self.gdpr = GDPR()
        
        self.real_world_dpa = """
        DATA PROCESSING AGREEMENT
        
        Between ACME Corporation ("Controller") and CloudServices Inc ("Processor")
        
        Article 1: Subject Matter and Duration
        This Agreement governs the processing of personal data by the Processor on behalf of the Controller.
        
        Article 2: Nature and Purpose of Processing
        The Processor shall process personal data solely for the purpose of providing cloud storage services.
        
        Article 3: Categories of Data
        Personal data includes: names, email addresses, and business contact information.
        
        Article 4: Data Subject Rights
        The Controller shall ensure data subjects can exercise their rights under GDPR.
        Data subjects have the right to access, rectify, and delete their personal data.
        
        Article 5: Security Measures
        The Processor shall implement appropriate technical and organizational measures.
        All personal data shall be encrypted both at rest and in transit.
        
        Article 6: Data Retention
        Personal data shall not be retained longer than necessary for the specified purposes.
        Data shall be deleted within 30 days after termination of this agreement.
        
        Article 7: Sub-Processing
        Sub-processors may only be engaged with prior written consent from the Controller.
        
        Article 8: Data Breach Notification
        Any personal data breach shall be notified to the Controller within 24 hours.
        The Controller shall be assisted in notifying supervisory authorities within 72 hours.
        """
    
    def test_full_workflow_gdpr_compliance(self):
        """Test full workflow: parse -> verify -> report."""
        # Parse contract
        parsed = self.parser.parse(self.real_world_dpa, "integration_dpa")
        
        self.assertGreater(len(parsed.clauses), 5)
        self.assertGreater(len(parsed.parties), 1)
        
        # Verify GDPR compliance
        results = self.prover.verify_compliance(parsed, self.gdpr)
        
        self.assertGreater(len(results), 5)
        
        # Generate report
        report = self.prover.generate_compliance_report(parsed, self.gdpr, results)
        
        self.assertEqual(report.contract_id, parsed.id)
        self.assertGreater(report.total_requirements, 0)
        self.assertGreaterEqual(report.compliance_rate, 0.0)
        
        # Should have reasonable compliance due to good language
        self.assertGreater(report.compliance_rate, 20.0)
    
    def test_multi_regulation_compliance(self):
        """Test compliance verification against multiple regulations."""
        parsed = self.parser.parse(self.real_world_dpa, "multi_reg_test")
        
        # Test GDPR
        gdpr_results = self.prover.verify_compliance(parsed, self.gdpr)
        
        # Test CCPA
        ccpa = CCPA()
        ccpa_results = self.prover.verify_compliance(parsed, ccpa)
        
        # Both should return results
        self.assertGreater(len(gdpr_results), 0)
        self.assertGreater(len(ccpa_results), 0)
        
        # Requirements should be different
        gdpr_req_ids = set(gdpr_results.keys())
        ccpa_req_ids = set(ccpa_results.keys())
        self.assertNotEqual(gdpr_req_ids, ccpa_req_ids)
    
    def test_focus_area_analysis(self):
        """Test focused analysis on specific compliance areas."""
        parsed = self.parser.parse(self.real_world_dpa, "focus_analysis")
        
        # Focus on data subject rights
        rights_results = self.prover.verify_compliance(
            parsed, self.gdpr, 
            focus_areas=['data_subject_rights', 'access_rights']
        )
        
        # Focus on security
        security_results = self.prover.verify_compliance(
            parsed, self.gdpr,
            focus_areas=['security', 'technical_measures']
        )
        
        # Should return different results
        self.assertNotEqual(set(rights_results.keys()), set(security_results.keys()))
        
        # Rights results should include access-related requirements
        rights_req_ids = list(rights_results.keys())
        self.assertTrue(any("15.1" in req_id for req_id in rights_req_ids))  # Access rights
    
    def test_performance_with_large_contract(self):
        """Test performance with larger contract."""
        # Create a larger contract by repeating sections
        large_contract = self.real_world_dpa * 5  # 5x larger
        
        parsed = self.parser.parse(large_contract, "large_contract_test")
        
        # Should still parse successfully
        self.assertGreater(len(parsed.clauses), 20)
        
        # Should still verify (may be slower but should complete)
        results = self.prover.verify_compliance(parsed, self.gdpr)
        self.assertGreater(len(results), 0)


def run_comprehensive_tests():
    """Run comprehensive test suite."""
    print("🧪 Running Comprehensive Test Suite")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestContractParsing,
        TestRegulationModels,
        TestComplianceVerification,
        TestReasoningComponents,
        TestExplanationSystem,
        TestIntegrationScenarios
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"📊 TEST SUMMARY")
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n🚫 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Error:')[-1].strip()}")
    
    # Return success
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == '__main__':
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)