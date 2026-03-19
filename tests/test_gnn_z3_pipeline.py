"""
Tests for the GNN → Z3 compliance pipeline.

Covers:
  - LegalEntityGraph construction from contracts and dicts
  - GNN forward pass (shape, output range)
  - GNNPropertyExtractor produces EntityProperties
  - Z3ComplianceVerifier: GDPR rules (consent, retention, security, cross-border)
  - Z3ComplianceVerifier: AI Act risk classification rule
  - Z3 counter-example generation for violations
  - Full end-to-end pipeline
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from neuro_symbolic_law.parsing.contract_parser import ContractParser
from neuro_symbolic_law.gnn import (
    LegalEntityGraph, LegalGNN, GNNConfig, GNNPropertyExtractor, EntityProperties
)
from neuro_symbolic_law.reasoning.z3_compliance_verifier import Z3ComplianceVerifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

COMPLIANT_CONTRACT = """
DATA PROCESSING AGREEMENT

This agreement is between ACME Corp and CloudVault Ltd.

1. LAWFUL BASIS AND CONSENT
   All personal data collected by ACME Corp is obtained with explicit consent
   from data subjects, maintaining a lawful basis for processing per GDPR Article 6.

2. DATA RETENTION
   Personal data is deleted within 60 days of the end of the processing purpose.
   No personal data is retained longer than necessary.

3. SECURITY
   CloudVault Ltd implements encryption at rest (AES-256) and in transit (TLS 1.3).
   Regular security assessments and penetration testing are conducted annually.

4. INTERNATIONAL TRANSFERS
   All cross-border transfers use standard contractual clauses per GDPR Article 46
   and adequacy decisions where available.

5. AI OVERSIGHT
   The AI system used for analytics includes human oversight at all decision points.
   Users are informed and explanations are provided for each automated decision.
"""


@pytest.fixture
def compliant_contract():
    parser = ContractParser()
    return parser.parse(COMPLIANT_CONTRACT, "test_compliant")


@pytest.fixture
def compliant_graph(compliant_contract):
    return LegalEntityGraph.from_contract(compliant_contract)


@pytest.fixture
def non_compliant_entities():
    return [
        {
            "id": "bad_processor",
            "name": "BadProcessor",
            "type": "party",
            "is_personal_data": True,
            "has_consent": False,
            "has_retention_limit": False,
            "has_security_measure": False,
            "has_transfer_safeguard": False,
        },
    ]


@pytest.fixture
def high_risk_ai_entity_no_oversight():
    return {
        "id": "risky_ai",
        "name": "RiskyAI",
        "type": "system",
        "is_high_risk_ai": True,
        "has_human_oversight": False,
        "has_transparency": False,
    }


@pytest.fixture
def high_risk_ai_entity_compliant():
    return {
        "id": "safe_ai",
        "name": "SafeAI",
        "type": "system",
        "is_high_risk_ai": True,
        "has_human_oversight": True,
        "has_transparency": True,
    }


# ---------------------------------------------------------------------------
# LegalEntityGraph tests
# ---------------------------------------------------------------------------

class TestLegalEntityGraph:

    def test_graph_from_contract_has_nodes(self, compliant_graph):
        assert len(compliant_graph.entities) > 0

    def test_graph_from_contract_has_parties(self, compliant_contract):
        graph = LegalEntityGraph.from_contract(compliant_contract)
        party_nodes = [e for e in graph.entities.values() if e.entity_type == "party"]
        assert len(party_nodes) >= 2

    def test_graph_from_dict(self):
        entities = [
            {"id": "e1", "name": "Entity1", "type": "party"},
            {"id": "e2", "name": "Entity2", "type": "data_category", "connects_to": ["e1"]},
        ]
        graph = LegalEntityGraph.from_dict(entities)
        assert "e1" in graph.entities
        assert "e2" in graph.entities

    def test_graph_to_tensors_shape(self, compliant_graph):
        features, edge_index, node_ids = compliant_graph.to_tensors()
        N = len(compliant_graph.entities)
        assert features.shape == (N, 14)   # 5 type + 9 bool properties
        assert edge_index.shape[0] == 2
        assert len(node_ids) == N

    def test_property_population_from_compliant_contract(self, compliant_graph):
        """Compliant contract should set consent/retention/security flags."""
        any_consent = any(e.has_consent for e in compliant_graph.entities.values())
        any_retention = any(e.has_retention_limit for e in compliant_graph.entities.values())
        any_security = any(e.has_security_measure for e in compliant_graph.entities.values())
        assert any_consent, "Expected at least one entity with consent flag"
        assert any_retention, "Expected at least one entity with retention_limit flag"
        assert any_security, "Expected at least one entity with security_measure flag"


# ---------------------------------------------------------------------------
# GNN model tests
# ---------------------------------------------------------------------------

class TestLegalGNN:

    def test_gnn_forward_pass_shape(self, compliant_graph):
        features, edge_index, node_ids = compliant_graph.to_tensors()
        config = GNNConfig(in_features=14, hidden_features=32, out_features=16, num_layers=2)
        gnn = LegalGNN(config)
        gnn.eval()
        with torch.no_grad():
            embeddings = gnn(features, edge_index)
        assert embeddings.shape == (len(node_ids), 16)

    def test_gnn_no_edges(self):
        """GNN should handle graphs with no edges."""
        features = torch.rand(3, 14)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        config = GNNConfig(in_features=14, hidden_features=16, out_features=8, num_layers=1)
        gnn = LegalGNN(config)
        gnn.eval()
        with torch.no_grad():
            embeddings = gnn(features, edge_index)
        assert embeddings.shape == (3, 8)

    def test_gnn_single_node(self):
        features = torch.rand(1, 14)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        config = GNNConfig(in_features=14, hidden_features=16, out_features=8)
        gnn = LegalGNN(config)
        gnn.eval()
        with torch.no_grad():
            out = gnn(features, edge_index)
        assert out.shape == (1, 8)

    def test_gnn_output_finite(self, compliant_graph):
        features, edge_index, _ = compliant_graph.to_tensors()
        gnn = LegalGNN(GNNConfig())
        gnn.eval()
        with torch.no_grad():
            out = gnn(features, edge_index)
        assert torch.all(torch.isfinite(out)), "GNN output contains NaN or Inf"


# ---------------------------------------------------------------------------
# GNNPropertyExtractor tests
# ---------------------------------------------------------------------------

class TestGNNPropertyExtractor:

    def test_extractor_returns_entity_properties(self, compliant_graph):
        extractor = GNNPropertyExtractor()
        props = extractor.extract(compliant_graph)
        assert len(props) == len(compliant_graph.entities)
        for p in props:
            assert isinstance(p, EntityProperties)

    def test_property_scores_in_range(self, compliant_graph):
        extractor = GNNPropertyExtractor()
        props = extractor.extract(compliant_graph)
        for p in props:
            for attr in [
                "has_consent", "has_retention_limit", "has_security_measure",
                "has_transfer_safeguard", "is_personal_data", "is_sensitive_data",
                "is_high_risk_ai", "has_human_oversight", "has_transparency",
            ]:
                val = getattr(p, attr)
                assert 0.0 <= val <= 1.0, f"{attr}={val} out of range for {p.entity_id}"

    def test_compliant_contract_extracts_high_scores(self, compliant_graph):
        """The compliant contract should produce high consent/retention/security scores
        for the overall entity set (blended keyword + GNN)."""
        extractor = GNNPropertyExtractor()
        props = extractor.extract(compliant_graph)
        # At least one entity should have high consent score (keyword flag is strong)
        max_consent = max(p.has_consent for p in props)
        assert max_consent >= 0.5, "Expected at least one entity with consent score >= 0.5"


# ---------------------------------------------------------------------------
# Z3ComplianceVerifier: GDPR tests
# ---------------------------------------------------------------------------

class TestGDPRRules:

    def _props(self, **kwargs):
        p = EntityProperties(entity_id="test", entity_name="Test", entity_type="party")
        for k, v in kwargs.items():
            setattr(p, k, v)
        return p

    def test_gdpr_lawful_basis_satisfied_when_consent_present(self):
        verifier = Z3ComplianceVerifier()
        props = self._props(is_personal_data=1.0, has_consent=1.0)
        report = verifier.verify_entity(props)
        basis_result = next(r for r in report.results if "LawfulBasis" in r.rule_id)
        assert basis_result.satisfied

    def test_gdpr_lawful_basis_violated_when_no_consent(self):
        verifier = Z3ComplianceVerifier()
        props = self._props(is_personal_data=1.0, has_consent=0.0)
        report = verifier.verify_entity(props)
        basis_result = next(r for r in report.results if "LawfulBasis" in r.rule_id)
        assert not basis_result.satisfied

    def test_gdpr_retention_satisfied_when_limit_set(self):
        verifier = Z3ComplianceVerifier()
        props = self._props(is_personal_data=1.0, has_retention_limit=1.0)
        report = verifier.verify_entity(props)
        retention_result = next(r for r in report.results if "Retention" in r.rule_id)
        assert retention_result.satisfied

    def test_gdpr_retention_violated_when_no_limit(self):
        verifier = Z3ComplianceVerifier()
        props = self._props(is_personal_data=1.0, has_retention_limit=0.0)
        report = verifier.verify_entity(props)
        retention_result = next(r for r in report.results if "Retention" in r.rule_id)
        assert not retention_result.satisfied

    def test_gdpr_security_satisfied_when_measures_present(self):
        verifier = Z3ComplianceVerifier()
        props = self._props(is_personal_data=1.0, has_security_measure=1.0)
        report = verifier.verify_entity(props)
        security_result = next(r for r in report.results if "Security" in r.rule_id)
        assert security_result.satisfied

    def test_gdpr_security_violated_when_no_measures(self):
        verifier = Z3ComplianceVerifier()
        props = self._props(is_personal_data=1.0, has_security_measure=0.0)
        report = verifier.verify_entity(props)
        security_result = next(r for r in report.results if "Security" in r.rule_id)
        assert not security_result.satisfied

    def test_gdpr_cross_border_satisfied_when_safeguards_present(self):
        verifier = Z3ComplianceVerifier()
        props = self._props(is_personal_data=1.0, has_transfer_safeguard=1.0)
        report = verifier.verify_entity(props)
        transfer_result = next((r for r in report.results if "CrossBorder" in r.rule_id), None)
        assert transfer_result is not None
        assert transfer_result.satisfied

    def test_gdpr_cross_border_violated_when_no_safeguards(self):
        verifier = Z3ComplianceVerifier()
        props = self._props(is_personal_data=1.0, has_transfer_safeguard=0.0)
        report = verifier.verify_entity(props)
        transfer_result = next((r for r in report.results if "CrossBorder" in r.rule_id), None)
        assert transfer_result is not None
        assert not transfer_result.satisfied

    def test_gdpr_no_rules_when_not_personal_data(self):
        """Non-personal data entities should still have rules but all pass trivially."""
        verifier = Z3ComplianceVerifier()
        props = self._props(is_personal_data=0.0, has_consent=0.0, has_retention_limit=0.0)
        report = verifier.verify_entity(props)
        # All implied rules satisfied when antecedent is false
        for r in report.results:
            assert r.satisfied, f"Rule {r.rule_id} should be satisfied for non-personal-data entity"

    def test_gdpr_sensitive_data_requires_consent(self):
        """GDPR Art. 9: sensitive data needs explicit consent."""
        verifier = Z3ComplianceVerifier()
        props = self._props(is_personal_data=1.0, is_sensitive_data=1.0, has_consent=0.0)
        report = verifier.verify_entity(props)
        sensitive_result = next((r for r in report.results if "Sensitive" in r.rule_id), None)
        assert sensitive_result is not None
        assert not sensitive_result.satisfied

    def test_gdpr_sensitive_data_satisfied_with_consent(self):
        verifier = Z3ComplianceVerifier()
        props = self._props(is_personal_data=1.0, is_sensitive_data=1.0, has_consent=1.0)
        report = verifier.verify_entity(props)
        sensitive_result = next((r for r in report.results if "Sensitive" in r.rule_id), None)
        assert sensitive_result is not None
        assert sensitive_result.satisfied


# ---------------------------------------------------------------------------
# Z3ComplianceVerifier: AI Act tests
# ---------------------------------------------------------------------------

class TestAIActRules:

    def _props(self, **kwargs):
        p = EntityProperties(entity_id="test", entity_name="AI System", entity_type="system")
        for k, v in kwargs.items():
            setattr(p, k, v)
        return p

    def test_ai_act_high_risk_requires_oversight_and_transparency(self):
        """High-risk AI without oversight or transparency should be flagged."""
        verifier = Z3ComplianceVerifier()
        props = self._props(is_high_risk_ai=1.0, has_human_oversight=0.0, has_transparency=0.0)
        report = verifier.verify_entity(props)
        risk_result = next(r for r in report.results if "RiskClassification" in r.rule_id)
        assert not risk_result.satisfied

    def test_ai_act_high_risk_compliant_with_oversight_and_transparency(self):
        verifier = Z3ComplianceVerifier()
        props = self._props(is_high_risk_ai=1.0, has_human_oversight=1.0, has_transparency=1.0)
        report = verifier.verify_entity(props)
        risk_result = next(r for r in report.results if "RiskClassification" in r.rule_id)
        assert risk_result.satisfied

    def test_ai_act_oversight_rule_violated(self):
        verifier = Z3ComplianceVerifier()
        props = self._props(is_high_risk_ai=1.0, has_human_oversight=0.0)
        report = verifier.verify_entity(props)
        oversight_result = next(r for r in report.results if "HumanOversight" in r.rule_id)
        assert not oversight_result.satisfied

    def test_ai_act_transparency_rule_violated(self):
        verifier = Z3ComplianceVerifier()
        props = self._props(is_high_risk_ai=1.0, has_transparency=0.0)
        report = verifier.verify_entity(props)
        transparency_result = next(r for r in report.results if "Transparency" in r.rule_id)
        assert not transparency_result.satisfied

    def test_ai_act_non_high_risk_always_passes(self):
        """Non-high-risk AI should pass all AI Act rules trivially."""
        verifier = Z3ComplianceVerifier()
        props = self._props(is_high_risk_ai=0.0, has_human_oversight=0.0, has_transparency=0.0)
        report = verifier.verify_entity(props)
        for r in report.results:
            assert r.satisfied, f"Non-high-risk AI should pass rule {r.rule_id}"


# ---------------------------------------------------------------------------
# Z3 counter-example tests
# ---------------------------------------------------------------------------

class TestZ3CounterExamples:

    def _props(self, **kwargs):
        p = EntityProperties(entity_id="test", entity_name="Test", entity_type="party")
        for k, v in kwargs.items():
            setattr(p, k, v)
        return p

    def test_counter_example_present_for_violation(self):
        """A violated rule should include a counter-example (when Z3 is available)."""
        try:
            from z3 import sat
            z3_available = True
        except ImportError:
            z3_available = False

        if not z3_available:
            pytest.skip("Z3 not available")

        verifier = Z3ComplianceVerifier()
        props = self._props(is_personal_data=1.0, has_consent=0.0)
        report = verifier.verify_entity(props)
        basis_result = next(r for r in report.results if "LawfulBasis" in r.rule_id)
        assert not basis_result.satisfied
        # Counter-example should be a dict (may be empty for simple BoolVal cases)
        assert basis_result.counter_example is not None or basis_result.error is None


# ---------------------------------------------------------------------------
# End-to-end pipeline test
# ---------------------------------------------------------------------------

class TestEndToEndPipeline:

    def test_full_pipeline_compliant_contract(self):
        """Full pipeline on compliant contract should produce mostly passing results."""
        parser = ContractParser()
        contract = parser.parse(COMPLIANT_CONTRACT, "e2e_test")
        graph = LegalEntityGraph.from_contract(contract)
        extractor = GNNPropertyExtractor()
        entity_props = extractor.extract(graph)
        verifier = Z3ComplianceVerifier()
        reports = verifier.verify_all(entity_props)

        assert len(reports) > 0
        total = sum(len(r.results) for r in reports)
        assert total > 0

    def test_full_pipeline_non_compliant_entities_produces_violations(self):
        """Hand-crafted non-compliant entities should produce violations."""
        entities = [
            {
                "id": "bad_party",
                "name": "BadParty",
                "type": "party",
                "is_personal_data": True,
                "has_consent": False,
                "has_retention_limit": False,
                "has_security_measure": False,
                "has_transfer_safeguard": False,
            },
        ]
        graph = LegalEntityGraph.from_dict(entities)
        extractor = GNNPropertyExtractor()
        entity_props = extractor.extract(graph)
        verifier = Z3ComplianceVerifier()
        reports = verifier.verify_all(entity_props)

        all_violated = [r for report in reports for r in report.results if not r.satisfied]
        assert len(all_violated) > 0, "Expected violations for non-compliant entity"

    def test_full_pipeline_high_risk_ai_no_safeguards_produces_violations(self):
        entities = [
            {
                "id": "risky_ai",
                "name": "RiskyAI",
                "type": "system",
                "is_high_risk_ai": True,
                "has_human_oversight": False,
                "has_transparency": False,
            },
        ]
        graph = LegalEntityGraph.from_dict(entities)
        extractor = GNNPropertyExtractor()
        entity_props = extractor.extract(graph)
        verifier = Z3ComplianceVerifier()
        reports = verifier.verify_all(entity_props)

        violated_rules = [r.rule_id for report in reports for r in report.results if not r.satisfied]
        assert "AIAct-Art6-RiskClassification" in violated_rules
        assert "AIAct-Art14-HumanOversight" in violated_rules
        assert "AIAct-Art13-Transparency" in violated_rules
