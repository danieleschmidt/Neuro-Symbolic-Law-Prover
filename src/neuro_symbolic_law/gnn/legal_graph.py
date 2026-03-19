"""
Legal entity graph construction from parsed contracts.

Builds a graph where:
  - Nodes = legal entities (parties, data categories, purposes, obligations)
  - Edges = relationships between entities (processes, transfers, stores, etc.)

Node features encode semantic properties extracted from contract text:
  - Entity type (one-hot)
  - Presence of key legal terms (boolean bag-of-features)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import re
import logging

logger = logging.getLogger(__name__)


# --- Data classes -----------------------------------------------------------

@dataclass
class LegalEntity:
    """A node in the legal entity graph."""
    id: str
    name: str
    entity_type: str          # 'party', 'data_category', 'purpose', 'obligation', 'system'
    text_snippet: str = ""    # raw text from which this entity was extracted
    properties: Dict[str, bool] = field(default_factory=dict)

    # Property flags (populated during graph construction)
    has_consent: bool = False
    has_retention_limit: bool = False
    has_security_measure: bool = False
    has_transfer_safeguard: bool = False
    is_personal_data: bool = False
    is_sensitive_data: bool = False
    is_high_risk_ai: bool = False
    has_human_oversight: bool = False
    has_transparency: bool = False


@dataclass
class GraphEdge:
    """A directed edge in the legal entity graph."""
    source_id: str
    target_id: str
    relation: str     # 'processes', 'transfers_to', 'stores', 'oversees', 'controls'
    weight: float = 1.0


class LegalEntityGraph:
    """
    Graph of legal entities extracted from a contract.

    Usage::

        graph = LegalEntityGraph.from_contract(parsed_contract)
        nodes, edge_index, features = graph.to_tensors()
    """

    # --- legal property signals ------------------------------------------
    _CONSENT_KEYWORDS = [
        "consent", "lawful basis", "legitimate interest", "explicit consent",
        "opt-in", "permission", "authorised", "authorized",
    ]
    _RETENTION_KEYWORDS = [
        "retention", "retain", "storage limit", "deleted", "delete",
        "no longer than", "within.*days", "years after",
    ]
    _SECURITY_KEYWORDS = [
        "encryption", "encrypted", "access control", "security measure",
        "technical measure", "organizational measure", "penetration test",
        "firewall", "pseudonymisation",
    ]
    _TRANSFER_KEYWORDS = [
        "standard contractual clause", "adequacy decision", "binding corporate rule",
        "international transfer", "third country", "cross-border",
    ]
    _PERSONAL_DATA_KEYWORDS = [
        "personal data", "personal information", "pii", "data subject",
        "identity data", "contact data",
    ]
    _SENSITIVE_KEYWORDS = [
        "sensitive", "special category", "biometric", "health", "racial",
        "religious", "sexual orientation", "criminal",
    ]
    _HIGH_RISK_AI_KEYWORDS = [
        "high-risk", "high risk ai", "automated decision", "profiling",
        "credit scoring", "biometric identification", "law enforcement",
        "critical infrastructure", "employment decision",
    ]
    _OVERSIGHT_KEYWORDS = [
        "human oversight", "human review", "human intervention", "human control",
        "override", "audit", "monitor",
    ]
    _TRANSPARENCY_KEYWORDS = [
        "transparency", "explainability", "explain", "interpretable",
        "disclose", "inform", "notification", "notice",
    ]

    def __init__(self) -> None:
        self.entities: Dict[str, LegalEntity] = {}
        self.edges: List[GraphEdge] = []

    # -----------------------------------------------------------------------
    @classmethod
    def from_contract(cls, parsed_contract) -> "LegalEntityGraph":
        """
        Build a LegalEntityGraph from a ParsedContract.

        Extracts:
          - Parties as entity nodes
          - Data categories from clause text
          - Purposes from clause text
          - Obligations from clause text
        Then wires them with typed edges.
        """
        graph = cls()

        # 1. Add party nodes
        for party in parsed_contract.parties:
            eid = f"party:{party.name.lower().replace(' ', '_')}"
            entity = LegalEntity(
                id=eid,
                name=party.name,
                entity_type="party",
                text_snippet=party.name,
            )
            graph._populate_properties(entity, party.name + " " + parsed_contract.text)
            graph.entities[eid] = entity

        # 2. Extract data categories and purposes from clauses
        data_cats = graph._extract_data_categories(parsed_contract.clauses)
        purposes = graph._extract_purposes(parsed_contract.clauses)
        obligations = graph._extract_obligations(parsed_contract.clauses)

        for dc in data_cats:
            eid = f"data:{dc.lower().replace(' ', '_')}"
            if eid not in graph.entities:
                entity = LegalEntity(id=eid, name=dc, entity_type="data_category", text_snippet=dc)
                graph._populate_properties(entity, parsed_contract.text)
                graph.entities[eid] = entity

        for purpose in purposes:
            eid = f"purpose:{purpose.lower().replace(' ', '_')[:40]}"
            if eid not in graph.entities:
                entity = LegalEntity(id=eid, name=purpose, entity_type="purpose", text_snippet=purpose)
                graph._populate_properties(entity, parsed_contract.text)
                graph.entities[eid] = entity

        for obligation in obligations:
            eid = f"obligation:{obligation.lower().replace(' ', '_')[:40]}"
            if eid not in graph.entities:
                entity = LegalEntity(id=eid, name=obligation, entity_type="obligation",
                                     text_snippet=obligation)
                graph._populate_properties(entity, parsed_contract.text)
                graph.entities[eid] = entity

        # 3. Wire edges: parties → data_categories (processes)
        party_ids = [eid for eid, e in graph.entities.items() if e.entity_type == "party"]
        data_ids = [eid for eid, e in graph.entities.items() if e.entity_type == "data_category"]
        purpose_ids = [eid for eid, e in graph.entities.items() if e.entity_type == "purpose"]
        obligation_ids = [eid for eid, e in graph.entities.items() if e.entity_type == "obligation"]

        for pid in party_ids:
            for did in data_ids:
                graph.edges.append(GraphEdge(pid, did, "processes"))
            for oid in obligation_ids:
                graph.edges.append(GraphEdge(pid, oid, "subject_to"))

        for did in data_ids:
            for puid in purpose_ids:
                graph.edges.append(GraphEdge(did, puid, "used_for"))

        logger.info(
            f"LegalEntityGraph: {len(graph.entities)} nodes, {len(graph.edges)} edges"
        )
        return graph

    # -----------------------------------------------------------------------
    @classmethod
    def from_dict(cls, entity_dicts: List[Dict]) -> "LegalEntityGraph":
        """Build a graph from a list of entity dicts (for demo/testing)."""
        graph = cls()
        for d in entity_dicts:
            eid = d.get("id", d["name"].lower().replace(" ", "_"))
            entity = LegalEntity(
                id=eid,
                name=d["name"],
                entity_type=d.get("type", "party"),
                text_snippet=d.get("text", ""),
                has_consent=d.get("has_consent", False),
                has_retention_limit=d.get("has_retention_limit", False),
                has_security_measure=d.get("has_security_measure", False),
                has_transfer_safeguard=d.get("has_transfer_safeguard", False),
                is_personal_data=d.get("is_personal_data", False),
                is_sensitive_data=d.get("is_sensitive_data", False),
                is_high_risk_ai=d.get("is_high_risk_ai", False),
                has_human_oversight=d.get("has_human_oversight", False),
                has_transparency=d.get("has_transparency", False),
            )
            graph.entities[eid] = entity

        for i, d in enumerate(entity_dicts):
            for target_name in d.get("connects_to", []):
                target_id = target_name.lower().replace(" ", "_")
                if target_id in graph.entities:
                    src_id = d.get("id", d["name"].lower().replace(" ", "_"))
                    graph.edges.append(GraphEdge(src_id, target_id, "relates_to"))

        return graph

    # -----------------------------------------------------------------------
    def to_tensors(self):
        """
        Convert graph to PyTorch tensors.

        Returns:
            node_features: FloatTensor [N, F]
            edge_index:    LongTensor  [2, E]
            node_ids:      list of entity ids (length N)
        """
        import torch

        node_ids = list(self.entities.keys())
        id_to_idx = {eid: i for i, eid in enumerate(node_ids)}

        BOOL_PROPS = [
            "has_consent", "has_retention_limit", "has_security_measure",
            "has_transfer_safeguard", "is_personal_data", "is_sensitive_data",
            "is_high_risk_ai", "has_human_oversight", "has_transparency",
        ]
        TYPE_ORDER = ["party", "data_category", "purpose", "obligation", "system"]

        features = []
        for eid in node_ids:
            entity = self.entities[eid]
            # One-hot entity type
            type_onehot = [1.0 if entity.entity_type == t else 0.0 for t in TYPE_ORDER]
            # Boolean properties
            bool_feats = [1.0 if getattr(entity, p, False) else 0.0 for p in BOOL_PROPS]
            features.append(type_onehot + bool_feats)

        node_features = torch.tensor(features, dtype=torch.float32)

        if self.edges:
            src_list = [id_to_idx[e.source_id] for e in self.edges
                        if e.source_id in id_to_idx and e.target_id in id_to_idx]
            tgt_list = [id_to_idx[e.target_id] for e in self.edges
                        if e.source_id in id_to_idx and e.target_id in id_to_idx]
            edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return node_features, edge_index, node_ids

    # -----------------------------------------------------------------------
    def _populate_properties(self, entity: LegalEntity, context: str) -> None:
        """Populate boolean property flags from keyword matching on context text."""
        ctx = context.lower()
        entity.has_consent = self._matches_any(ctx, self._CONSENT_KEYWORDS)
        entity.has_retention_limit = self._matches_any(ctx, self._RETENTION_KEYWORDS)
        entity.has_security_measure = self._matches_any(ctx, self._SECURITY_KEYWORDS)
        entity.has_transfer_safeguard = self._matches_any(ctx, self._TRANSFER_KEYWORDS)
        entity.is_personal_data = self._matches_any(ctx, self._PERSONAL_DATA_KEYWORDS)
        entity.is_sensitive_data = self._matches_any(ctx, self._SENSITIVE_KEYWORDS)
        entity.is_high_risk_ai = self._matches_any(ctx, self._HIGH_RISK_AI_KEYWORDS)
        entity.has_human_oversight = self._matches_any(ctx, self._OVERSIGHT_KEYWORDS)
        entity.has_transparency = self._matches_any(ctx, self._TRANSPARENCY_KEYWORDS)

    @staticmethod
    def _matches_any(text: str, keywords: List[str]) -> bool:
        return any(re.search(kw, text) for kw in keywords)

    def _extract_data_categories(self, clauses) -> List[str]:
        DATA_PATTERNS = [
            r"(identity data|usage data|financial data|payment information|"
            r"personal data|contact data|location data|health data|biometric data|"
            r"behavioral data|communication data)",
        ]
        found = set()
        for clause in clauses:
            for pattern in DATA_PATTERNS:
                for m in re.finditer(pattern, clause.text, re.IGNORECASE):
                    found.add(m.group(1).lower().strip())
        return list(found) if found else ["personal_data"]

    def _extract_purposes(self, clauses) -> List[str]:
        PURPOSE_PATTERNS = [
            r"(?:for the purposes? of|for|including)\s+([^,.\n]{5,60})",
        ]
        found = set()
        purpose_keywords = ["service", "analytics", "compliance", "delivery", "support",
                            "fraud", "billing", "marketing", "research"]
        for clause in clauses:
            text = clause.text.lower()
            for kw in purpose_keywords:
                if kw in text:
                    found.add(kw)
        return list(found) if found else ["service_delivery"]

    def _extract_obligations(self, clauses) -> List[str]:
        OBLIGATION_KEYWORDS = ["encrypt", "delete", "notify", "report", "implement",
                               "maintain", "assess", "audit", "restrict"]
        found = set()
        for clause in clauses:
            text = clause.text.lower()
            for kw in OBLIGATION_KEYWORDS:
                if kw in text:
                    found.add(kw)
        return list(found) if found else ["process_lawfully"]
