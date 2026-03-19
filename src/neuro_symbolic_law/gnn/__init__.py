"""
GNN (Graph Neural Network) component for legal entity graph processing.

This module provides:
- LegalEntityGraph: builds a graph from legal entities extracted from contracts
- LegalGNN: message-passing GNN that produces embeddings for each entity node
- GNNPropertyExtractor: converts GNN node embeddings → boolean property scores
  that Z3 can consume for formal compliance verification
"""

from .legal_graph import LegalEntityGraph, LegalEntity, GraphEdge
from .gnn_model import LegalGNN, GNNConfig
from .property_extractor import GNNPropertyExtractor, EntityProperties

__all__ = [
    "LegalEntityGraph",
    "LegalEntity",
    "GraphEdge",
    "LegalGNN",
    "GNNConfig",
    "GNNPropertyExtractor",
    "EntityProperties",
]
