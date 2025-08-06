#!/usr/bin/env python3
"""
Simple test runner for the neuro-symbolic law prover.
"""

import sys
import os
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test basic imports work correctly."""
    print("🧪 Testing basic imports...")
    try:
        from neuro_symbolic_law import LegalProver, ContractParser
        from neuro_symbolic_law.regulations import GDPR, AIAct, CCPA
        print("✅ Basic imports successful")
        return True
    except Exception as e:
        print(f"❌ Basic import failed: {e}")
        return False

def test_contract_parser():
    """Test contract parsing functionality."""
    print("🧪 Testing contract parser...")
    try:
        from neuro_symbolic_law import ContractParser
        
        parser = ContractParser()
        
        contract_text = """
        DATA PROCESSING AGREEMENT
        
        This agreement is between ACME Corp and CloudTech Inc.
        
        1. The data controller shall implement appropriate technical measures to ensure data security.
        2. Personal data shall be processed only for specified purposes.
        3. Data subjects have the right to access their personal data.
        """
        
        parsed = parser.parse(contract_text, "test_contract")
        
        assert parsed.id == "test_contract"
        assert len(parsed.clauses) > 0
        assert len(parsed.parties) >= 2
        
        print(f"✅ Parser test passed - {len(parsed.clauses)} clauses, {len(parsed.parties)} parties")
        return True
        
    except Exception as e:
        print(f"❌ Parser test failed: {e}")
        traceback.print_exc()
        return False

def test_legal_prover():
    """Test legal prover functionality."""
    print("🧪 Testing legal prover...")
    try:
        from neuro_symbolic_law import LegalProver, ContractParser
        from neuro_symbolic_law.regulations import GDPR
        
        parser = ContractParser()
        prover = LegalProver()
        gdpr = GDPR()
        
        contract_text = """
        PRIVACY POLICY
        
        1. Personal data shall be processed lawfully and transparently.
        2. Data collection is limited to what is necessary for specified purposes.
        3. Personal data will be kept secure with appropriate technical measures.
        4. Data subjects have the right to access their personal data.
        5. Personal data will be deleted when no longer necessary.
        """
        
        parsed_contract = parser.parse(contract_text, "gdpr_test")
        results = prover.verify_compliance(
            contract=parsed_contract,
            regulation=gdpr,
            focus_areas=['data_minimization', 'security', 'data_subject_rights']
        )
        
        assert len(results) > 0
        assert all(hasattr(result, 'status') for result in results.values())
        
        # Generate report
        report = prover.generate_compliance_report(
            contract=parsed_contract,
            regulation=gdpr,
            results=results
        )
        
        assert report.contract_id == "gdpr_test"
        assert report.regulation_name == "GDPR"
        assert 0.0 <= report.compliance_rate <= 100.0
        
        print(f"✅ Prover test passed - {len(results)} requirements checked, {report.compliance_rate:.1f}% compliant")
        return True
        
    except Exception as e:
        print(f"❌ Prover test failed: {e}")
        traceback.print_exc()
        return False

def test_regulations():
    """Test regulation models."""
    print("🧪 Testing regulation models...")
    try:
        from neuro_symbolic_law.regulations import GDPR, AIAct, CCPA
        
        # Test GDPR
        gdpr = GDPR()
        assert len(gdpr) > 0
        assert gdpr.name == "GDPR"
        
        requirements = gdpr.get_requirements()
        assert "GDPR-5.1.c" in requirements  # Data minimization
        assert "GDPR-15.1" in requirements   # Right of access
        
        categories = gdpr.get_categories()
        assert 'data_minimization' in categories
        assert 'security' in categories
        
        # Test AI Act
        ai_act = AIAct()
        assert len(ai_act) > 0
        assert ai_act.name == "EU AI Act"
        
        ai_requirements = ai_act.get_requirements()
        assert "AI-ACT-9.1" in ai_requirements   # Risk management
        assert "AI-ACT-13.1" in ai_requirements  # Transparency
        
        # Test CCPA
        ccpa = CCPA()
        assert len(ccpa) > 0
        assert ccpa.name == "CCPA"
        
        ccpa_requirements = ccpa.get_requirements()
        assert "CCPA-1798.100.a" in ccpa_requirements  # Right to know
        assert "CCPA-1798.105.a" in ccpa_requirements  # Right to delete
        
        print(f"✅ Regulation tests passed - GDPR: {len(gdpr)}, AI Act: {len(ai_act)}, CCPA: {len(ccpa)} requirements")
        return True
        
    except Exception as e:
        print(f"❌ Regulation test failed: {e}")
        traceback.print_exc()
        return False

def test_cli():
    """Test CLI functionality."""
    print("🧪 Testing CLI...")
    try:
        from neuro_symbolic_law.cli import main
        # Basic smoke test - just check CLI imports
        print("✅ CLI import successful")
        return True
        
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        traceback.print_exc()
        return False

def test_enhanced_features():
    """Test enhanced features."""
    print("🧪 Testing enhanced features...")
    try:
        from neuro_symbolic_law.core.enhanced_prover import EnhancedLegalProver
        from neuro_symbolic_law.parsing.neural_parser import NeuralContractParser
        from neuro_symbolic_law.reasoning.z3_encoder import Z3Encoder
        from neuro_symbolic_law.explanation.explainer import ExplainabilityEngine
        
        # Basic initialization tests
        enhanced_prover = EnhancedLegalProver(debug=True)
        neural_parser = NeuralContractParser(debug=True)
        z3_encoder = Z3Encoder(debug=True)
        explainer = ExplainabilityEngine()
        
        print("✅ Enhanced features initialization successful")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced features test failed: {e}")
        traceback.print_exc()
        return False

def test_applications():
    """Test application modules."""
    print("🧪 Testing application modules...")
    try:
        from neuro_symbolic_law.applications import (
            SaaSContractAnalyzer,
            AISystemContractAnalyzer,
            DataSharingAgreementAnalyzer
        )
        
        # Basic initialization tests
        saas_analyzer = SaaSContractAnalyzer(debug=True)
        ai_analyzer = AISystemContractAnalyzer(debug=True)
        data_analyzer = DataSharingAgreementAnalyzer(debug=True)
        
        print("✅ Application modules initialization successful")
        return True
        
    except Exception as e:
        print(f"❌ Application modules test failed: {e}")
        traceback.print_exc()
        return False

def run_example():
    """Run the basic usage example."""
    print("🧪 Running basic usage example...")
    try:
        # Import example
        sys.path.insert(0, 'examples')
        from basic_usage import main as example_main
        
        # Run example (this might take a moment)
        gdpr_report, ai_report = example_main()
        
        assert gdpr_report is not None
        assert ai_report is not None
        assert hasattr(gdpr_report, 'compliance_rate')
        assert hasattr(ai_report, 'compliance_rate')
        
        print(f"✅ Example completed - GDPR: {gdpr_report.compliance_rate:.1f}%, AI Act: {ai_report.compliance_rate:.1f}%")
        return True
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Neuro-Symbolic Law Prover - Quality Gates Testing")
    print("=" * 60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Contract Parser", test_contract_parser),
        ("Legal Prover", test_legal_prover),
        ("Regulation Models", test_regulations),
        ("CLI", test_cli),
        ("Enhanced Features", test_enhanced_features),
        ("Applications", test_applications),
        ("Basic Usage Example", run_example),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        try:
            if test_func():
                passed += 1
            else:
                print(f"💥 {test_name} failed")
        except Exception as e:
            print(f"💥 {test_name} crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 TEST SUMMARY: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"⚠️  {total - passed} tests failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())