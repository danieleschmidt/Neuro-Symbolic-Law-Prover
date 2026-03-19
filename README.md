# Neuro-Symbolic-Law-Prover ⚖️🤖

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Z3](https://img.shields.io/badge/Z3-4.12+-purple.svg)](https://github.com/Z3Prover/z3)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

Combines **Graph Neural Networks (GNN)** with the **Z3 SMT solver** to automatically prove
regulatory compliance and identify counter-examples in legal contracts,
focusing on **GDPR** and **EU AI Act** requirements.

## 🌟 Key Features

- **Legal Entity Graph**: builds a graph from contract text — nodes are parties, data categories,
  purposes, and obligations; edges encode relationships like _processes_, _transfers_to_, _stores_
- **GNN Message Passing**: 2-layer mean-aggregation GNN (pure PyTorch, no PyG required)
  produces dense embeddings for each graph node
- **Property Extraction**: blended keyword + GNN projection to scalar compliance property
  scores (consent, retention, security, transfer safeguards, high-risk AI, …)
- **Z3 Formal Verification**: encodes GDPR Art. 6/9/32/46 and EU AI Act Art. 6/13/14
  as SMT formulas; proves satisfiability and generates counter-examples for violations
- **Multi-Regulation**: GDPR, EU AI Act, CCPA, and custom regulations
- **Demo ready**: `python demo.py` runs the full pipeline end-to-end

## 🏗️ Architecture

```
Contract Text
      │
      ▼
ContractParser (rule-based + NLP)
      │
      ▼
LegalEntityGraph  ←── nodes: parties, data, purposes, obligations
      │                edges: processes, transfers, stores, oversees
      ▼
LegalGNN  ←── 2-layer mean-aggregation message passing (pure PyTorch)
      │
      ▼
GNNPropertyExtractor  ←── sigmoid projection → [0,1] compliance scores
      │                    blended with keyword-based evidence
      ▼
Z3ComplianceVerifier  ←── encodes rules as SMT formulas
      │                    e.g. Implies(is_personal_data, has_consent)
      ▼
ComplianceReport  ←── per-entity: passed rules, violations, counter-examples
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/danieleschmidt/Neuro-Symbolic-Law-Prover.git
cd Neuro-Symbolic-Law-Prover
pip install z3-solver torch
pip install -e ".[dev]"
```

### Run the Demo

```bash
# Compliant contract (sample)
python demo.py

# Non-compliant hand-crafted entities (shows violations + counter-examples)
python demo.py --entities

# Both scenarios
python demo.py --both

# Your own contract
python demo.py --contract path/to/your/contract.txt
```

### GNN + Z3 Pipeline (Python API)

```python
from neuro_symbolic_law.parsing.contract_parser import ContractParser
from neuro_symbolic_law.gnn import LegalEntityGraph, GNNPropertyExtractor
from neuro_symbolic_law.reasoning.z3_compliance_verifier import Z3ComplianceVerifier

# 1. Parse contract
parser = ContractParser()
contract = parser.parse(open("contract.txt").read(), "my_contract")

# 2. Build legal entity graph
graph = LegalEntityGraph.from_contract(contract)

# 3. Run GNN → extract compliance properties
extractor = GNNPropertyExtractor()
entity_props = extractor.extract(graph)

# 4. Verify GDPR + AI Act rules with Z3
verifier = Z3ComplianceVerifier()
for report in verifier.verify_all(entity_props):
    print(report.summary())
    for violation in report.violations:
        print(f"  ❌ {violation.rule_id}: {violation.rule_description}")
        if violation.counter_example:
            print(f"     Counter-example: {violation.counter_example}")
```

### Classic Contract-Level Compliance

```python
from neuro_symbolic_law import LegalProver, ContractParser
from neuro_symbolic_law.regulations import GDPR, AIAct

prover = LegalProver()
parser = ContractParser()

contract_text = open('data_processing_agreement.txt').read()
parsed_contract = parser.parse(contract_text)

print(f"Extracted {len(parsed_contract.clauses)} clauses")
print(f"Identified parties: {parsed_contract.parties}")

# Verify GDPR compliance
gdpr_results = prover.verify_compliance(
    contract=parsed_contract,
    regulation=GDPR(),
    focus_areas=['data_retention', 'purpose_limitation', 'data_subject_rights']
)

# Display results
for requirement, result in gdpr_results.items():
    print(f"\n{requirement}:")
    print(f"  Compliant: {'✓' if result.compliant else '✗'}")
    if not result.compliant:
        print(f"  Issue: {result.issue}")
        print(f"  Counter-example: {result.counter_example}")
        print(f"  Suggestion: {result.suggestion}")
```

## 🏗️ Architecture

```
neuro-symbolic-law-prover/
├── parsing/                # Contract parsing
│   ├── models/            # Neural models
│   │   ├── legal_bert.py  # Fine-tuned BERT
│   │   ├── graph_builder.py # Contract graph construction
│   │   └── clause_extractor.py
│   ├── preprocessing/     # Text preprocessing
│   └── templates/         # Parsing templates
├── reasoning/             # Symbolic reasoning
│   ├── z3_encoder.py     # Z3 formula encoding
│   ├── solver.py         # SMT solving
│   ├── proof_search.py   # Proof strategies
│   └── counter_examples.py
├── regulations/          # Regulation models
│   ├── gdpr/            # GDPR rules
│   │   ├── articles.py   # Article definitions
│   │   ├── principles.py # Core principles
│   │   └── requirements.py
│   ├── ai_act/          # EU AI Act
│   ├── ccpa/            # California privacy
│   └── custom/          # Custom regulations
├── knowledge/            # Legal knowledge base
│   ├── ontologies/      # Legal ontologies
│   ├── precedents/      # Case law
│   └── templates/       # Contract templates
├── explanation/         # Explanation generation
│   ├── natural_language.py
│   ├── visualizations.py
│   └── report_generator.py
└── applications/        # Domain applications
    ├── saas_contracts/  # SaaS agreements
    ├── ai_systems/      # AI model contracts
    └── data_sharing/    # Data sharing agreements
```

## 🧠 Neural Contract Understanding

### Graph Neural Network Parser

```python
from neuro_symbolic_law.parsing import LegalGraphBuilder
import torch_geometric

# Build contract graph representation
graph_builder = LegalGraphBuilder(
    node_features=['entity', 'obligation', 'right', 'condition'],
    edge_types=['defines', 'obligates', 'permits', 'restricts']
)

# Parse contract into graph
contract_graph = graph_builder.build_graph(contract_text)

# Visualize contract structure
graph_builder.visualize(
    contract_graph,
    highlight_obligations=True,
    show_data_flows=True
)

# Extract structured information
class ContractGNN(torch.nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.node_encoder = torch.nn.Linear(768, hidden_dim)  # BERT embeddings
        self.gnn_layers = torch.nn.ModuleList([
            torch_geometric.nn.GATConv(hidden_dim, hidden_dim, heads=4)
            for _ in range(3)
        ])
        self.classifier = torch.nn.Linear(hidden_dim, num_clause_types)
        
    def forward(self, x, edge_index):
        x = self.node_encoder(x)
        for gnn in self.gnn_layers:
            x = gnn(x, edge_index)
            x = torch.nn.functional.relu(x)
        return self.classifier(x)

# Extract compliance-relevant clauses
model = ContractGNN()
clause_classifications = model(
    contract_graph.node_features,
    contract_graph.edge_index
)
```

### Semantic Role Labeling

```python
from neuro_symbolic_law.parsing import LegalSRL

srl = LegalSRL()

# Extract legal roles from clauses
clause = "The data controller shall implement appropriate technical measures to ensure data security."

roles = srl.extract_roles(clause)
print(roles)
# Output:
# {
#   'actor': 'data controller',
#   'action': 'implement',
#   'object': 'appropriate technical measures',
#   'purpose': 'ensure data security',
#   'modality': 'obligation'
# }

# Convert to logical representation
logical_form = srl.to_logic(roles)
print(logical_form)
# Output: Obligation(DataController, Implement(TechnicalMeasures, Purpose(DataSecurity)))
```

## ⚖️ Symbolic Legal Reasoning

### Z3 Compliance Checking

```python
from z3 import *
from neuro_symbolic_law.reasoning import ComplianceChecker

# Define GDPR requirements in Z3
checker = ComplianceChecker()

# Data minimization principle
@checker.rule("GDPR Article 5(1)(c)")
def data_minimization(contract):
    solver = Solver()
    
    # Variables
    data_collected = Real('data_collected')
    data_necessary = Real('data_necessary')
    purpose = String('purpose')
    
    # Contract constraints
    for clause in contract.data_collection_clauses:
        solver.add(parse_data_scope(clause))
    
    # GDPR requirement: collected <= necessary
    solver.add(data_collected > data_necessary)
    
    # Check for violation
    if solver.check() == sat:
        model = solver.model()
        return ComplianceResult(
            compliant=False,
            violation="Collecting more data than necessary",
            counter_example={
                'collected': model[data_collected],
                'necessary': model[data_necessary],
                'excess': model[data_collected] - model[data_necessary]
            }
        )
    
    return ComplianceResult(compliant=True)

# Purpose limitation
@checker.rule("GDPR Article 5(1)(b)")
def purpose_limitation(contract):
    solver = Solver()
    
    # Extract stated purposes
    stated_purposes = extract_purposes(contract)
    
    # Extract actual uses
    actual_uses = extract_data_uses(contract)
    
    # Check each use against purposes
    for use in actual_uses:
        if not any(compatible(use, purpose) for purpose in stated_purposes):
            return ComplianceResult(
                compliant=False,
                violation=f"Data used for '{use}' not covered by stated purposes",
                counter_example={'unauthorized_use': use},
                suggestion=f"Add '{use}' to stated purposes or remove this use"
            )
    
    return ComplianceResult(compliant=True)
```

### Temporal Logic for Data Retention

```python
from neuro_symbolic_law.reasoning import TemporalChecker

temporal = TemporalChecker()

# Define retention requirements
@temporal.ltl_property("Data retention limits")
def retention_compliance(contract):
    """
    G (data_collected -> F[<=retention_period] data_deleted)
    Globally: if data collected, then eventually (within period) deleted
    """
    
    # Extract retention clauses
    retention_rules = contract.get_retention_rules()
    
    # Build temporal model
    for data_type, rule in retention_rules.items():
        if rule.retention_period > rule.legal_maximum:
            return ComplianceResult(
                compliant=False,
                violation=f"{data_type} retained for {rule.retention_period} days, "
                         f"but legal maximum is {rule.legal_maximum} days",
                suggestion=f"Reduce retention to {rule.legal_maximum} days"
            )
    
    # Verify deletion mechanisms exist
    if not contract.has_deletion_procedure():
        return ComplianceResult(
            compliant=False,
            violation="No automatic deletion procedure specified",
            suggestion="Implement automatic deletion after retention period"
        )
    
    return ComplianceResult(compliant=True)
```

## 🤖 AI Act Compliance

### High-Risk AI System Verification

```python
from neuro_symbolic_law.regulations import AIActChecker

ai_checker = AIActChecker()

# Parse AI system contract
ai_contract = parser.parse(open('ai_service_agreement.txt').read())

# Classify risk level
risk_assessment = ai_checker.assess_risk_level(
    application_area=ai_contract.extract_field('application_area'),
    deployment_context=ai_contract.extract_field('deployment_context')
)

print(f"Risk level: {risk_assessment.level}")  # e.g., "HIGH_RISK"

if risk_assessment.level == "HIGH_RISK":
    # Verify high-risk requirements
    requirements = ai_checker.get_high_risk_requirements()
    
    results = {}
    for req_id, requirement in requirements.items():
        result = ai_checker.verify_requirement(
            contract=ai_contract,
            requirement=requirement
        )
        results[req_id] = result
        
        if not result.compliant:
            print(f"\n{req_id}: {requirement.description}")
            print(f"❌ Non-compliant: {result.issue}")
            print(f"📝 Required clause: {result.missing_clause_template}")
```

### Transparency Requirements

```python
# AI Act Article 13 - Transparency
transparency_checker = ai_checker.transparency_checker()

# Check for required disclosures
required_disclosures = [
    'ai_system_interaction',
    'emotion_recognition_use', 
    'biometric_categorization',
    'deepfake_generation'
]

for disclosure in required_disclosures:
    if disclosure in ai_contract.system_capabilities:
        if not transparency_checker.has_disclosure(ai_contract, disclosure):
            print(f"Missing required disclosure for: {disclosure}")
            
            # Generate compliant disclosure clause
            clause = transparency_checker.generate_disclosure_clause(
                capability=disclosure,
                context=ai_contract.context
            )
            print(f"Suggested clause:\n{clause}")
```

## 📊 Compliance Analytics

### Contract Portfolio Analysis

```python
from neuro_symbolic_law.analytics import PortfolioAnalyzer

analyzer = PortfolioAnalyzer()

# Analyze multiple contracts
contracts = load_contract_portfolio('contracts/*.pdf')

portfolio_report = analyzer.analyze_portfolio(
    contracts=contracts,
    regulations=[GDPR(), AIAct(), CCPA()],
    risk_threshold=0.8
)

# Risk heatmap
analyzer.plot_risk_heatmap(
    portfolio_report,
    dimensions=['regulation', 'contract_type', 'risk_level']
)

# Identify systemic issues
systemic_issues = analyzer.find_systemic_issues(
    portfolio_report,
    min_occurrence=0.3  # Issues in >30% of contracts
)

for issue in systemic_issues:
    print(f"\nSystemic issue: {issue.description}")
    print(f"Affects {issue.affected_contracts} contracts ({issue.percentage:.1%})")
    print(f"Recommended fix: {issue.portfolio_wide_solution}")
```

### Compliance Trends

```python
from neuro_symbolic_law.analytics import ComplianceTrends

trends = ComplianceTrends()

# Track compliance over time
historical_data = trends.load_historical_audits('audits_2020_2024.db')

trend_analysis = trends.analyze_trends(
    historical_data,
    metrics=['compliance_rate', 'common_violations', 'fix_time']
)

# Predictive compliance
predictor = trends.train_predictor(historical_data)

future_risks = predictor.predict_future_risks(
    time_horizon='6_months',
    factors=['regulation_changes', 'business_growth', 'tech_stack_changes']
)

print("Predicted compliance risks:")
for risk in future_risks:
    print(f"- {risk.area}: {risk.probability:.1%} chance of issues")
    print(f"  Mitigation: {risk.suggested_mitigation}")
```

## 🔧 Custom Regulation Support

### Define Custom Regulations

```python
from neuro_symbolic_law.regulations import CustomRegulation

# Define industry-specific regulation
class HealthDataRegulation(CustomRegulation):
    def __init__(self):
        super().__init__(name="Health Data Protection Act 2025")
        
    def define_rules(self):
        # Rule 1: Encryption requirements
        self.add_rule(
            id="HDPA-1.1",
            description="Health data must be encrypted at rest and in transit",
            formal_rule=lambda contract: self.check_encryption(contract),
            severity="critical"
        )
        
        # Rule 2: Access controls
        self.add_rule(
            id="HDPA-2.1", 
            description="Multi-factor authentication required for health data access",
            formal_rule=lambda contract: self.check_mfa(contract),
            severity="high"
        )
        
        # Rule 3: Audit logging
        self.add_rule(
            id="HDPA-3.1",
            description="All health data access must be logged with immutable audit trail",
            formal_rule=lambda contract: self.check_audit_logging(contract),
            severity="high"
        )
    
    def check_encryption(self, contract):
        # Z3-based verification of encryption requirements
        solver = Solver()
        
        # Extract security clauses
        security_clauses = contract.get_security_clauses()
        
        # Verify encryption specifications
        has_rest_encryption = Bool('has_rest_encryption')
        has_transit_encryption = Bool('has_transit_encryption')
        
        # ... Z3 encoding logic ...
        
        return solver.check() == sat

# Use custom regulation
health_reg = HealthDataRegulation()
health_compliance = prover.verify_compliance(
    contract=medical_contract,
    regulation=health_reg
)
```

## 🎯 Contract Generation

### Generate Compliant Templates

```python
from neuro_symbolic_law.generation import ContractGenerator

generator = ContractGenerator()

# Generate GDPR-compliant DPA
dpa_template = generator.generate_template(
    contract_type='data_processing_agreement',
    regulations=[GDPR()],
    parties={
        'controller': 'ACME Corp',
        'processor': 'CloudTech Inc'
    },
    data_categories=['personal', 'sensitive', 'financial'],
    processing_purposes=['analytics', 'backup', 'support'],
    options={
        'include_sccs': True,  # Standard Contractual Clauses
        'sub_processors': True,
        'data_retention': '24_months'
    }
)

# Verify generated template
verification = prover.verify_compliance(
    contract=parser.parse(dpa_template),
    regulation=GDPR()
)

assert all(r.compliant for r in verification.values())

# Export in various formats
generator.export(dpa_template, 'dpa_template.docx')
generator.export(dpa_template, 'dpa_template.pdf')
generator.export(dpa_template, 'dpa_template.md')
```

## 📈 Explanation & Reporting

### Natural Language Explanations

```python
from neuro_symbolic_law.explanation import ExplainabilityEngine

explainer = ExplainabilityEngine(
    llm_model='gpt-4',
    style='legal_professional'
)

# Explain non-compliance
for violation in gdpr_results.get_violations():
    explanation = explainer.explain_violation(
        violation=violation,
        contract_context=parsed_contract,
        audience='legal_team'
    )
    
    print(f"\n{violation.rule_id}: {violation.description}")
    print(f"\nExplanation for legal team:")
    print(explanation.legal_explanation)
    print(f"\nExplanation for business:")
    print(explanation.business_explanation)
    print(f"\nSuggested remediation:")
    print(explanation.remediation_steps)
```

### Compliance Reports

```python
from neuro_symbolic_law.reporting import ComplianceReporter

reporter = ComplianceReporter()

# Generate comprehensive report
report = reporter.generate_report(
    contract=parsed_contract,
    compliance_results=gdpr_results,
    include_sections=[
        'executive_summary',
        'detailed_findings',
        'risk_assessment',
        'remediation_plan',
        'technical_appendix'
    ]
)

# Interactive dashboard
dashboard = reporter.create_dashboard(
    contracts=[contract1, contract2, contract3],
    regulations=[GDPR(), AIAct()],
    update_frequency='daily'
)

dashboard.launch(port=8080)
```

## 🔍 Advanced Analysis

### Cross-Contract Consistency

```python
from neuro_symbolic_law.analysis import ConsistencyChecker

consistency = ConsistencyChecker()

# Check consistency across related contracts
contract_suite = {
    'master_agreement': parse('master_services_agreement.txt'),
    'dpa': parse('data_processing_agreement.txt'),
    'sla': parse('service_level_agreement.txt')
}

inconsistencies = consistency.check_suite(contract_suite)

for issue in inconsistencies:
    print(f"\nInconsistency found:")
    print(f"- {issue.contract1}:{issue.clause1}")
    print(f"- {issue.contract2}:{issue.clause2}")
    print(f"Issue: {issue.description}")
    print(f"Resolution: {issue.suggested_resolution}")
```

### Legal Change Impact Analysis

```python
from neuro_symbolic_law.analysis import ChangeImpactAnalyzer

impact_analyzer = ChangeImpactAnalyzer()

# Analyze impact of regulatory changes
new_regulation = parse_regulation_update('gdpr_amendment_2025.xml')

impact_report = impact_analyzer.analyze_impact(
    regulation_change=new_regulation,
    contract_portfolio=contracts,
    implementation_deadline='2025-07-01'
)

print(f"Contracts requiring updates: {impact_report.affected_count}")
print(f"Estimated effort: {impact_report.total_effort_hours} hours")
print(f"Critical changes: {len(impact_report.critical_changes)}")

# Generate action plan
action_plan = impact_analyzer.generate_action_plan(
    impact_report,
    resources=['legal_team', 'compliance_team'],
    prioritization='risk_based'
)
```

## 📚 Research & Citations

```bibtex
@inproceedings{neuro_symbolic_law2025,
  title={Neuro-Symbolic Reasoning for Automated Legal Compliance Verification},
  author={Daniel Schmidt},
  booktitle={International Conference on AI and Law},
  year={2025}
}

@article{smt_legal_reasoning2024,
  title={SMT-based Verification of Data Protection Regulations},
  author={Daniel Schmidt},
  journal={Artificial Intelligence and Law},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions:
- Additional regulation models
- Legal ontology extensions
- Contract parsing improvements
- Industry-specific templates

See [CONTRIBUTING.md](CONTRIBUTING.md)

## ⚖️ Legal Notice

This tool provides automated analysis but does not constitute legal advice. Always consult qualified legal professionals for final compliance decisions.

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 🔗 Resources

- [Documentation](https://neuro-symbolic-law.readthedocs.io)
- [Legal Ontologies](https://github.com/yourusername/legal-ontologies)
- [Regulation Updates](https://neuro-symbolic-law.org/regulations)
- [Blog Post](https://medium.com/@yourusername/neuro-symbolic-legal-ai)
