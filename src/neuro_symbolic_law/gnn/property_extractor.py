"""
GNNPropertyExtractor: bridges GNN embeddings → Z3-consumable boolean properties.

For each entity node the GNN produces an embedding vector. We apply a learned
(or heuristic) projection to recover the original property flags, and also
enrich them with structural signals from the graph (e.g., a data node that is
connected to a party with consent is itself "consented").

The output EntityProperties objects are directly consumed by the Z3ComplianceVerifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .legal_graph import LegalEntityGraph, LegalEntity
from .gnn_model import LegalGNN, GNNConfig


@dataclass
class EntityProperties:
    """
    Extracted properties for a single entity node — Z3-ready.

    All float fields are in [0, 1] (confidence scores).
    The Z3 verifier thresholds them at 0.5 for boolean interpretation.
    """
    entity_id: str
    entity_name: str
    entity_type: str

    # Core compliance properties
    has_consent: float = 0.0
    has_retention_limit: float = 0.0
    has_security_measure: float = 0.0
    has_transfer_safeguard: float = 0.0
    is_personal_data: float = 0.0
    is_sensitive_data: float = 0.0
    is_high_risk_ai: float = 0.0
    has_human_oversight: float = 0.0
    has_transparency: float = 0.0

    def as_bool(self, prop: str, threshold: float = 0.5) -> bool:
        return getattr(self, prop, 0.0) >= threshold

    def to_dict(self) -> Dict:
        return {
            "id": self.entity_id,
            "name": self.entity_name,
            "type": self.entity_type,
            "has_consent": self.has_consent,
            "has_retention_limit": self.has_retention_limit,
            "has_security_measure": self.has_security_measure,
            "has_transfer_safeguard": self.has_transfer_safeguard,
            "is_personal_data": self.is_personal_data,
            "is_sensitive_data": self.is_sensitive_data,
            "is_high_risk_ai": self.is_high_risk_ai,
            "has_human_oversight": self.has_human_oversight,
            "has_transparency": self.has_transparency,
        }


PROPERTY_NAMES = [
    "has_consent", "has_retention_limit", "has_security_measure",
    "has_transfer_safeguard", "is_personal_data", "is_sensitive_data",
    "is_high_risk_ai", "has_human_oversight", "has_transparency",
]


class PropertyProjectionHead(nn.Module):
    """
    Linear head that maps GNN output embeddings → property scores.
    One sigmoid output per property.
    """

    def __init__(self, in_dim: int, num_properties: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, num_properties)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.proj(embeddings))


class GNNPropertyExtractor:
    """
    Runs the full GNN pipeline and extracts EntityProperties from node embeddings.

    Pipeline:
      1. LegalEntityGraph → to_tensors() → (X, edge_index)
      2. LegalGNN forward pass → node embeddings [N, out_features]
      3. PropertyProjectionHead → property scores [N, num_properties]
      4. Blend with raw keyword-based scores (hedge against uninitialised weights)
      5. Return List[EntityProperties]
    """

    BLEND_ALPHA = 0.4   # weight on raw keyword features vs. GNN output

    def __init__(self, config: Optional[GNNConfig] = None) -> None:
        self.config = config or GNNConfig()
        self.gnn = LegalGNN(self.config)
        self.head = PropertyProjectionHead(self.config.out_features, len(PROPERTY_NAMES))
        self.gnn.eval()
        self.head.eval()

    def extract(self, graph: LegalEntityGraph) -> List[EntityProperties]:
        """
        Run GNN over the graph and return per-entity properties.
        """
        node_features, edge_index, node_ids = graph.to_tensors()

        with torch.no_grad():
            embeddings = self.gnn(node_features, edge_index)       # [N, out_features]
            gnn_scores = self.head(embeddings)                      # [N, num_props]

        results = []
        for i, eid in enumerate(node_ids):
            entity = graph.entities[eid]

            # Raw keyword-based scores (0 or 1)
            raw_scores = torch.tensor(
                [1.0 if getattr(entity, p, False) else 0.0 for p in PROPERTY_NAMES],
                dtype=torch.float32,
            )

            # Blended score: prefer keyword evidence when available
            blended = self.BLEND_ALPHA * gnn_scores[i] + (1 - self.BLEND_ALPHA) * raw_scores

            props = EntityProperties(
                entity_id=eid,
                entity_name=entity.name,
                entity_type=entity.entity_type,
            )
            for j, name in enumerate(PROPERTY_NAMES):
                setattr(props, name, float(blended[j]))

            results.append(props)

        return results
