#!/usr/bin/env python3
"""
Neuro-Symbolic Law Prover — End-to-End Demo
============================================

Pipeline:
  1. Build a legal entity graph from a sample contract (or hand-crafted entities)
  2. Run GNN message passing → dense node embeddings
  3. Project embeddings → property scores (blended with keyword signals)
  4. Feed properties to Z3ComplianceVerifier
  5. Z3 checks GDPR and AI Act rules and reports violations with counter-examples

Usage:
  python demo.py
  python demo.py --contract path/to/contract.txt
  python demo.py --entities   # use hand-crafted entity graph (no contract needed)
"""

import argparse
import sys
import os

# Ensure src is on the path when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from neuro_symbolic_law.parsing.contract_parser import ContractParser
from neuro_symbolic_law.gnn import LegalEntityGraph, LegalGNN, GNNConfig, GNNPropertyExtractor
from neuro_symbolic_law.reasoning.z3_compliance_verifier import Z3ComplianceVerifier, ComplianceReport


# ---------------------------------------------------------------------------
# Sample contract text (GDPR / AI Act scenarios embedded)
# ---------------------------------------------------------------------------
SAMPLE_CONTRACT = """
CLOUD SERVICES & AI ANALYTICS AGREEMENT

This agreement is between DataCorp Ltd ("Controller") and CloudAI Inc ("Processor").

1. DATA PROCESSING
   CloudAI Inc shall process personal data on behalf of DataCorp Ltd solely for the
   agreed purposes of service delivery, analytics, and fraud prevention.
   Personal data includes identity data (name, email), usage data, and financial data.

2. LAWFUL BASIS & CONSENT
   DataCorp Ltd confirms that all personal data processed under this agreement has
   been collected with explicit consent from data subjects in accordance with GDPR
   Article 6. Consent records are maintained and auditable.

3. DATA RETENTION
   Personal data shall be deleted within 90 days of the end of the service period.
   No data shall be retained longer than necessary for its stated processing purpose.

4. SECURITY MEASURES
   CloudAI Inc shall implement encryption at rest (AES-256) and in transit (TLS 1.3).
   Access controls, role-based permissions, and annual penetration testing are mandatory.
   Security assessments and organizational measures align with ISO 27001.

5. INTERNATIONAL TRANSFERS
   Any transfer of personal data to third countries shall be governed by standard
   contractual clauses (SCCs) approved by the European Commission.
   No transfers shall occur without adequacy decisions or equivalent safeguards.

6. AI SYSTEM — RISK CLASSIFICATION
   CloudAI Inc operates a credit-scoring AI system used for employment decisions.
   This system is classified as HIGH-RISK under the EU AI Act Annex III.
   Automated decision-making is subject to human review and override at all times.

7. HUMAN OVERSIGHT
   The high-risk AI system includes a mandatory human-in-the-loop review stage.
   Human operators may intervene, override, or disable the system at any time.
   Audit logs of all automated decisions are maintained for 24 months.

8. TRANSPARENCY & EXPLAINABILITY
   The AI system provides explanations for each decision in plain language.
   Data subjects and affected employees are informed when automated decisions apply.
   Model cards and technical documentation are maintained and available upon request.

9. BREACH NOTIFICATION
   Security incidents affecting personal data shall be reported to the Controller
   within 24 hours and to supervisory authorities within 72 hours as required by
   GDPR Article 33.
"""

# ---------------------------------------------------------------------------
# Hand-crafted entity graph — illustrates a NON-COMPLIANT scenario
# (missing consent, no retention limit on sensitive data)
# ---------------------------------------------------------------------------
NON_COMPLIANT_ENTITIES = [
    {
        "id": "acme_controller",
        "name": "ACME Controller",
        "type": "party",
        "text": "ACME Corp processes personal data for marketing without explicit consent",
        "is_personal_data": True,
        "has_consent": False,           # VIOLATION: no lawful basis
        "has_retention_limit": False,   # VIOLATION: no retention policy
        "has_security_measure": True,
        "has_transfer_safeguard": False, # VIOLATION: no transfer safeguards
        "is_sensitive_data": False,
        "is_high_risk_ai": False,
        "has_human_oversight": False,
        "has_transparency": False,
    },
    {
        "id": "biometric_system",
        "name": "BiometricAI System",
        "type": "system",
        "text": "Biometric identification system for law enforcement — high risk, no oversight",
        "is_personal_data": True,
        "has_consent": False,
        "has_retention_limit": False,
        "has_security_measure": False,
        "has_transfer_safeguard": False,
        "is_sensitive_data": True,      # biometric = sensitive
        "is_high_risk_ai": True,        # law enforcement AI = high risk
        "has_human_oversight": False,   # VIOLATION: no oversight
        "has_transparency": False,      # VIOLATION: not transparent
    },
]


def run_demo(contract_text: str, scenario_name: str = "Contract") -> None:
    print(f"\n{'='*60}")
    print(f"  Neuro-Symbolic Law Prover — {scenario_name}")
    print(f"{'='*60}\n")

    # Step 1: Parse contract → build legal entity graph
    print("📋 Step 1: Parsing contract & building legal entity graph...")
    parser = ContractParser()
    parsed = parser.parse(contract_text, "demo_contract")
    graph = LegalEntityGraph.from_contract(parsed)
    print(f"   ✓ {len(parsed.clauses)} clauses parsed")
    print(f"   ✓ {len(graph.entities)} graph nodes, {len(graph.edges)} edges")

    # Step 2: Run GNN
    print("\n🧠 Step 2: Running GNN message passing...")
    config = GNNConfig(in_features=14, hidden_features=32, out_features=16, num_layers=2)
    extractor = GNNPropertyExtractor(config)
    entity_properties = extractor.extract(graph)
    print(f"   ✓ GNN produced embeddings for {len(entity_properties)} entities")

    # Step 3: Z3 compliance verification
    print("\n⚖️  Step 3: Z3 SMT compliance verification...")
    verifier = Z3ComplianceVerifier()
    reports = verifier.verify_all(entity_properties)

    # Print results
    print()
    total_rules = sum(len(r.results) for r in reports)
    total_passed = sum(sum(1 for res in r.results if res.satisfied) for r in reports)
    total_failed = total_rules - total_passed

    for report in reports:
        if not report.results:
            continue
        print(f"  Entity: {report.entity_name} [{entity_properties[reports.index(report)].entity_type}]")
        for res in report.results:
            icon = "✅" if res.satisfied else "❌"
            print(f"    {icon} [{res.rule_id}] {res.rule_description[:70]}")
            if not res.satisfied and res.counter_example:
                print(f"       Counter-example: {res.counter_example}")
            if res.error:
                print(f"       Error: {res.error}")
        print()

    print(f"{'─'*60}")
    print(f"  Summary: {total_passed}/{total_rules} rules passed  "
          f"({'✅ COMPLIANT' if total_failed == 0 else f'❌ {total_failed} VIOLATION(S)'})")
    print(f"{'='*60}\n")


def run_entities_demo(entities: list, scenario_name: str = "Hand-crafted entities") -> None:
    print(f"\n{'='*60}")
    print(f"  Neuro-Symbolic Law Prover — {scenario_name}")
    print(f"{'='*60}\n")

    print("📋 Step 1: Building graph from entity definitions...")
    graph = LegalEntityGraph.from_dict(entities)
    print(f"   ✓ {len(graph.entities)} nodes, {len(graph.edges)} edges")

    print("\n🧠 Step 2: Running GNN message passing...")
    config = GNNConfig(in_features=14, hidden_features=32, out_features=16, num_layers=2)
    extractor = GNNPropertyExtractor(config)
    entity_properties = extractor.extract(graph)
    print(f"   ✓ GNN produced embeddings for {len(entity_properties)} entities")

    print("\n⚖️  Step 3: Z3 SMT compliance verification...")
    verifier = Z3ComplianceVerifier()
    reports = verifier.verify_all(entity_properties)

    print()
    total_rules = sum(len(r.results) for r in reports)
    total_passed = sum(sum(1 for res in r.results if res.satisfied) for r in reports)
    total_failed = total_rules - total_passed

    for i, report in enumerate(reports):
        if not report.results:
            continue
        ep = entity_properties[i]
        print(f"  Entity: {report.entity_name} [{ep.entity_type}]")
        for res in report.results:
            icon = "✅" if res.satisfied else "❌"
            print(f"    {icon} [{res.rule_id}] {res.rule_description[:70]}")
            if not res.satisfied and res.counter_example:
                print(f"       Counter-example: {res.counter_example}")
        print()

    print(f"{'─'*60}")
    print(f"  Summary: {total_passed}/{total_rules} rules passed  "
          f"({'✅ COMPLIANT' if total_failed == 0 else f'❌ {total_failed} VIOLATION(S)'})")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Neuro-Symbolic Law Prover demo")
    parser.add_argument("--contract", help="Path to contract text file")
    parser.add_argument("--entities", action="store_true",
                        help="Use hand-crafted non-compliant entity graph instead of contract")
    parser.add_argument("--both", action="store_true",
                        help="Run both compliant and non-compliant scenarios")
    args = parser.parse_args()

    if args.contract:
        with open(args.contract) as f:
            contract_text = f.read()
        run_demo(contract_text, scenario_name="Custom Contract")

    elif args.entities:
        run_entities_demo(NON_COMPLIANT_ENTITIES, "Non-compliant Entity Graph")

    elif args.both:
        run_demo(SAMPLE_CONTRACT, "Compliant Contract (sample)")
        run_entities_demo(NON_COMPLIANT_ENTITIES, "Non-compliant Entity Graph")

    else:
        # Default: run the compliant contract scenario
        run_demo(SAMPLE_CONTRACT, "Compliant Contract (sample)")


if __name__ == "__main__":
    main()
