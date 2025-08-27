#!/usr/bin/env python3
"""
Generation 1: MAKE IT WORK - Simple functionality test
Test basic legal prover capabilities
"""

import sys
sys.path.insert(0, 'src')

from neuro_symbolic_law import LegalProver, ContractParser, ComplianceResult
from neuro_symbolic_law.regulations.gdpr import GDPR

def test_generation1_functionality():
    """Test Generation 1 basic functionality."""
    print("🧪 Testing Generation 1: MAKE IT WORK (Simple)")
    
    # Initialize basic components
    try:
        prover = LegalProver(debug=True)
        parser = ContractParser()
        gdpr = GDPR()
        print("✅ Basic initialization successful")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False
    
    # Test basic contract parsing
    try:
        sample_contract = """
        Data Processing Agreement
        
        1. Data Controller agrees to implement appropriate technical measures.
        2. Personal data shall be processed lawfully and transparently.  
        3. Data retention period shall not exceed 24 months.
        4. Data subjects have the right to access their personal data.
        """
        
        parsed_contract = parser.parse(sample_contract)
        print(f"✅ Contract parsed: {len(parsed_contract.clauses)} clauses extracted")
    except Exception as e:
        print(f"❌ Contract parsing failed: {e}")
        return False
    
    # Test basic compliance verification
    try:
        results = prover.verify_compliance(
            contract=parsed_contract,
            regulation=gdpr,
            focus_areas=['data_retention', 'data_subject_rights']
        )
        
        print(f"✅ Compliance verification completed: {len(results)} requirements checked")
        
        # Display results
        compliant_count = sum(1 for r in results.values() if r.compliant)
        print(f"📊 Compliance rate: {compliant_count}/{len(results)} requirements")
        
        return True
        
    except Exception as e:
        print(f"❌ Compliance verification failed: {e}")
        return False

def test_report_generation():
    """Test compliance report generation."""
    print("\n📋 Testing Compliance Report Generation")
    
    try:
        prover = LegalProver()
        parser = ContractParser()
        gdpr = GDPR()
        
        sample_contract = "Data processing for analytics purposes with 12-month retention."
        parsed_contract = parser.parse(sample_contract)
        
        report = prover.generate_compliance_report(
            contract=parsed_contract,
            regulation=gdpr
        )
        
        print(f"✅ Report generated: {report.overall_status}")
        print(f"📅 Timestamp: {report.timestamp}")
        return True
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 GENERATION 1: MAKE IT WORK - Testing Basic Functionality\n")
    
    success = True
    success &= test_generation1_functionality()
    success &= test_report_generation()
    
    if success:
        print(f"\n🎉 GENERATION 1 COMPLETE: Basic functionality verified!")
        print("📈 Ready to proceed to Generation 2: MAKE IT ROBUST")
        exit(0)
    else:
        print(f"\n❌ GENERATION 1 ISSUES: Some tests failed")
        exit(1)