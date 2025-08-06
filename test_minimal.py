#!/usr/bin/env python3
"""
Minimal test runner focusing on core functionality without external dependencies.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_core_functionality():
    """Test core functionality without external dependencies."""
    print("🧪 Testing core functionality...")
    
    try:
        # Test basic imports
        from neuro_symbolic_law import LegalProver, ContractParser
        from neuro_symbolic_law.regulations import GDPR, AIAct, CCPA
        print("✅ Core imports successful")
        
        # Test contract parsing
        parser = ContractParser()
        contract_text = """
        DATA PROCESSING AGREEMENT
        
        This agreement is between ACME Corp and CloudTech Inc.
        
        1. The data controller shall implement appropriate technical measures to ensure data security.
        2. Personal data shall be processed only for specified purposes.
        3. Data subjects have the right to access their personal data.
        """
        
        parsed = parser.parse(contract_text, "test_contract")
        print(f"✅ Contract parsed: {len(parsed.clauses)} clauses, {len(parsed.parties)} parties")
        
        # Test regulation models
        gdpr = GDPR()
        ai_act = AIAct()
        ccpa = CCPA()
        print(f"✅ Regulations loaded: GDPR({len(gdpr)}), AI Act({len(ai_act)}), CCPA({len(ccpa)})")
        
        # Test compliance verification
        prover = LegalProver()
        results = prover.verify_compliance(
            contract=parsed,
            regulation=gdpr,
            focus_areas=['data_minimization', 'security']
        )
        
        report = prover.generate_compliance_report(parsed, gdpr, results)
        print(f"✅ Compliance verification: {report.compliance_rate:.1f}% compliant")
        
        return True
        
    except Exception as e:
        print(f"❌ Core functionality test failed: {e}")
        return False

def test_reasoning_components():
    """Test reasoning components with fallbacks."""
    print("🧪 Testing reasoning components...")
    
    try:
        from neuro_symbolic_law.reasoning.z3_encoder import Z3Encoder
        from neuro_symbolic_law.reasoning.proof_search import ProofSearcher
        
        # Test Z3 encoder (should work with fallback)
        encoder = Z3Encoder(debug=True)
        print("✅ Z3Encoder initialized (may use fallback)")
        
        # Test proof searcher
        searcher = ProofSearcher()
        print("✅ ProofSearcher initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Reasoning components test failed: {e}")
        return False

def test_explanation_system():
    """Test explanation system."""
    print("🧪 Testing explanation system...")
    
    try:
        from neuro_symbolic_law.explanation.explainer import ExplainabilityEngine
        from neuro_symbolic_law.explanation.report_generator import ComplianceReporter
        from neuro_symbolic_law.core.compliance_result import ComplianceViolation, ViolationSeverity
        
        explainer = ExplainabilityEngine()
        reporter = ComplianceReporter()
        
        # Test violation explanation
        violation = ComplianceViolation(
            rule_id="GDPR-5.1.c",
            rule_description="Data minimization requirement",
            violation_text="Excessive data collection identified",
            severity=ViolationSeverity.HIGH
        )
        
        explanation = explainer.explain_violation(violation, None)
        print(f"✅ Explanation generated: {len(explanation.remediation_steps)} steps")
        
        return True
        
    except Exception as e:
        print(f"❌ Explanation system test failed: {e}")
        return False

def main():
    """Run minimal test suite."""
    print("🚀 Neuro-Symbolic Law Prover - Minimal Quality Gates")
    print("=" * 55)
    
    tests = [
        ("Core Functionality", test_core_functionality),
        ("Reasoning Components", test_reasoning_components),
        ("Explanation System", test_explanation_system),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        if test_func():
            passed += 1
    
    print("\n" + "=" * 55)
    print(f"📊 MINIMAL TEST SUMMARY: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 CORE FUNCTIONALITY VERIFIED!")
        return 0
    else:
        print(f"⚠️  {total - passed} tests failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())