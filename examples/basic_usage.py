#!/usr/bin/env python3
"""
Basic usage example for neuro-symbolic law prover.
"""

from neuro_symbolic_law import LegalProver, ContractParser
from neuro_symbolic_law.regulations import GDPR, AIAct


def main():
    """Demonstrate basic usage of the library."""
    
    # Initialize components
    prover = LegalProver()
    parser = ContractParser(model='basic')
    
    # Sample contract text
    contract_text = """
    DATA PROCESSING AGREEMENT
    
    This agreement is between ACME Corp (Controller) and CloudTech Inc (Processor).
    
    1. PURPOSES: Personal data shall be processed only for the following purposes:
       - Customer service provision
       - Analytics and business intelligence
       - Legal compliance and fraud prevention
    
    2. DATA CATEGORIES: The following categories of personal data may be processed:
       - Identity data (name, email, phone)
       - Usage data (login times, feature usage)
       - Financial data (payment information, billing address)
    
    3. SECURITY: The Processor shall implement appropriate technical and organizational 
       measures including encryption at rest and in transit, access controls, and 
       regular security assessments.
    
    4. DATA SUBJECT RIGHTS: Data subjects have the right to access, rectify, delete,
       and port their personal data. The Processor shall assist in responding to 
       data subject requests within 30 days.
    
    5. RETENTION: Personal data shall be deleted when no longer necessary for the 
       processing purposes, but in any event within 36 months of collection.
    
    6. BREACH NOTIFICATION: Any personal data breach shall be reported to the 
       Controller within 24 hours and to supervisory authorities within 72 hours.
    
    7. SUB-PROCESSORS: The Processor may engage sub-processors provided they 
       ensure equivalent data protection guarantees.
    """
    
    print("🧠 Neuro-Symbolic Law Prover Demo")
    print("=" * 50)
    
    # Parse the contract
    print("\n1. 📋 Parsing Contract...")
    parsed_contract = parser.parse(contract_text, contract_id='demo_dpa')
    
    print(f"   ✓ Contract parsed successfully")
    print(f"   ✓ Extracted {len(parsed_contract.clauses)} clauses")
    print(f"   ✓ Identified {len(parsed_contract.parties)} parties")
    print(f"   ✓ Contract type: {parsed_contract.contract_type or 'Unknown'}")
    
    # Show extracted parties
    if parsed_contract.parties:
        print(f"\n   📝 Parties:")
        for party in parsed_contract.parties:
            print(f"      • {party.name} ({party.role})")
    
    # Verify GDPR compliance
    print("\n2. ⚖️  Verifying GDPR Compliance...")
    gdpr = GDPR()
    
    gdpr_results = prover.verify_compliance(
        contract=parsed_contract,
        regulation=gdpr,
        focus_areas=['data_retention', 'purpose_limitation', 'data_subject_rights', 'security']
    )
    
    # Generate compliance report
    gdpr_report = prover.generate_compliance_report(
        contract=parsed_contract,
        regulation=gdpr,
        results=gdpr_results
    )
    
    print(f"   ✓ GDPR analysis complete")
    print(f"   ✓ Overall status: {gdpr_report.overall_status.value.upper()}")
    print(f"   ✓ Compliance rate: {gdpr_report.compliance_rate:.1f}%")
    print(f"   ✓ Requirements checked: {gdpr_report.total_requirements}")
    
    # Show compliance details
    compliant_count = sum(1 for r in gdpr_results.values() if r.compliant)
    non_compliant_count = len(gdpr_results) - compliant_count
    
    print(f"\n   📊 Results Summary:")
    print(f"      ✅ Compliant: {compliant_count}")
    print(f"      ❌ Non-compliant: {non_compliant_count}")
    
    # Show some specific results
    print(f"\n   🔍 Sample Results:")
    for req_id, result in list(gdpr_results.items())[:5]:
        status_emoji = "✅" if result.compliant else "❌"
        confidence = f"({result.confidence:.1%} confidence)"
        print(f"      {status_emoji} {req_id}: {result.requirement_description[:60]}... {confidence}")
        
        if result.supporting_clauses:
            print(f"         📄 Supporting: {len(result.supporting_clauses)} clause(s)")
        
        if result.issue:
            print(f"         ⚠️  Issue: {result.issue}")
    
    # Show violations if any
    violations = gdpr_report.get_violations()
    if violations:
        print(f"\n   🚨 Violations Found:")
        for violation in violations[:3]:  # Show first 3
            print(f"      • {violation.rule_id}: {violation.violation_text}")
            if violation.suggested_fix:
                print(f"        💡 Fix: {violation.suggested_fix}")
    
    print(f"\n3. 🤖 AI Act Compliance Check...")
    ai_act = AIAct()
    
    # For demonstration, let's assume this is an AI system contract
    ai_contract_text = contract_text.replace(
        "DATA PROCESSING AGREEMENT",
        "AI SYSTEM SERVICE AGREEMENT"
    ) + """
    
    8. AI SYSTEM: The service includes AI-powered analytics and recommendation 
       systems that process personal data to provide personalized insights.
    
    9. HUMAN OVERSIGHT: Human reviewers validate all AI-generated recommendations 
       before implementation.
    
    10. TRANSPARENCY: Users are informed when interacting with AI systems and 
        can request explanations of automated decisions.
    """
    
    ai_parsed = parser.parse(ai_contract_text, contract_id='ai_system_agreement')
    
    ai_results = prover.verify_compliance(
        contract=ai_parsed,
        regulation=ai_act,
        focus_areas=['transparency', 'human_oversight', 'risk_management']
    )
    
    ai_report = prover.generate_compliance_report(
        contract=ai_parsed,
        regulation=ai_act,
        results=ai_results
    )
    
    print(f"   ✓ AI Act analysis complete")
    print(f"   ✓ Overall status: {ai_report.overall_status.value.upper()}")
    print(f"   ✓ Compliance rate: {ai_report.compliance_rate:.1f}%")
    print(f"   ✓ Requirements checked: {ai_report.total_requirements}")
    
    # Show AI Act specific results
    ai_compliant = sum(1 for r in ai_results.values() if r.compliant)
    print(f"   ✅ AI Act compliant requirements: {ai_compliant}")
    
    print(f"\n4. 📈 Summary & Recommendations")
    print(f"   Overall assessment:")
    print(f"   • GDPR compliance: {gdpr_report.compliance_rate:.1f}%")
    print(f"   • AI Act compliance: {ai_report.compliance_rate:.1f}%")
    
    total_violations = len(gdpr_report.get_violations()) + len(ai_report.get_violations())
    
    if total_violations == 0:
        print(f"   🎉 No major compliance issues found!")
    else:
        print(f"   ⚠️  {total_violations} compliance issues require attention")
        print(f"   💡 Review specific violations above for remediation steps")
    
    print(f"\n5. 🚀 Next Steps")
    print(f"   • Review and address any compliance violations")
    print(f"   • Implement suggested fixes from the analysis")
    print(f"   • Consider regular compliance monitoring")
    print(f"   • Consult legal counsel for final compliance verification")
    
    print(f"\n✨ Demo completed successfully!")
    return gdpr_report, ai_report


if __name__ == '__main__':
    main()