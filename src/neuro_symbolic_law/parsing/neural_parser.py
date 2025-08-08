"""
Advanced neural parsing for legal contracts using transformer models.

Generation 1: Basic sentence transformers for semantic similarity
Generation 2: Will add fine-tuned legal BERT and GNN graph construction
Generation 3: Will add full semantic role labeling and formal logic extraction
"""

from typing import List, Dict, Optional, Tuple, Any
import logging
import re
from dataclasses import dataclass, field

# Neural model imports with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    import random
    
    class SentenceTransformer:
        def __init__(self, model_name): pass
        def encode(self, texts): return [[random.random() for _ in range(384)] for _ in texts]
    
    class np:
        @staticmethod
        def array(x): return x
        @staticmethod
        def dot(a, b): return sum(x*y for x, y in zip(a, b))
        @staticmethod
        def linalg(): pass
        class linalg:
            @staticmethod
            def norm(x): return sum(i**2 for i in x)**0.5

from .contract_parser import Clause, ContractParty, ParsedContract

logger = logging.getLogger(__name__)


@dataclass
class SemanticClause(Clause):
    """Enhanced clause with semantic analysis."""
    embedding: Optional[List[float]] = None
    semantic_type: Optional[str] = None
    legal_entities: List[str] = field(default_factory=list)
    temporal_expressions: List[str] = field(default_factory=list)
    conditional_expressions: List[str] = field(default_factory=list)
    obligation_strength: float = 0.0  # 0.0-1.0 scale
    

@dataclass 
class LegalEntity:
    """Represents a legal entity extracted from text."""
    name: str
    entity_type: str  # person, organization, location, law, regulation
    confidence: float
    context: str
    

class NeuralContractParser:
    """Neural-enhanced contract parser for advanced semantic understanding."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", debug: bool = False):
        """
        Initialize neural parser.
        
        Args:
            model_name: Sentence transformer model name
            debug: Enable debug logging
        """
        self.model_name = model_name
        self.debug = debug
        self.model = None
        
        if debug:
            logging.basicConfig(level=logging.DEBUG)
            
        # Initialize sentence transformer model
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.model = SentenceTransformer(model_name)
                logger.info(f"Loaded neural model: {model_name}")
            else:
                logger.warning("Sentence transformers not available - using fallback")
                self.model = SentenceTransformer(model_name)
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            self.model = SentenceTransformer("fallback")
        
        # Legal concept embeddings for semantic matching
        self.legal_concepts = {
            'data_processing': ['data processing', 'personal data', 'information processing', 'data handling'],
            'consent': ['consent', 'agreement', 'authorization', 'permission', 'approval'],
            'liability': ['liability', 'responsible', 'damages', 'indemnification', 'compensation'],
            'termination': ['terminate', 'end', 'expiry', 'cancellation', 'dissolution'],
            'confidentiality': ['confidential', 'non-disclosure', 'proprietary', 'trade secret'],
            'intellectual_property': ['copyright', 'patent', 'trademark', 'intellectual property'],
            'compliance': ['comply', 'compliance', 'regulation', 'law', 'requirement'],
            'security': ['security', 'protection', 'safeguard', 'encryption', 'secure'],
            'retention': ['retention', 'storage', 'keep', 'maintain', 'preserve'],
            'deletion': ['delete', 'deletion', 'remove', 'destroy', 'erase'],
            'transparency': ['transparency', 'disclosure', 'information', 'notify', 'inform'],
            'rights': ['rights', 'entitlement', 'privilege', 'access', 'control']
        }
        
        # Pre-compute concept embeddings
        self.concept_embeddings = self._compute_concept_embeddings()
        
        # Legal entity patterns
        self.entity_patterns = {
            'organization': r'\b(?:[A-Z][a-zA-Z\s]+(?:Inc\.|Corp\.|LLC|Ltd\.|Company|Corporation))\b',
            'person': r'\b(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b',
            'regulation': r'\b(?:GDPR|CCPA|AI Act|Article \d+|Section \d+|Regulation \d+)\b',
            'location': r'\b(?:California|Europe|EU|United States|US|UK)\b'
        }
        
        # Temporal expression patterns
        self.temporal_patterns = [
            r'\b(?:within|after|before|during)\s+\d+\s+(?:days?|months?|years?)\b',
            r'\b\d+\s+(?:days?|months?|years?)\s+(?:from|after|before)\b',
            r'\b(?:immediately|promptly|without delay|forthwith)\b',
            r'\b(?:annually|monthly|weekly|daily)\b'
        ]
        
        # Conditional expression patterns
        self.conditional_patterns = [
            r'\b(?:if|when|unless|provided that|subject to|in the event)\b.*?\b(?:then|shall|must|will)\b',
            r'\b(?:where|whereas)\b.*?,',
            r'\b(?:notwithstanding|except|save for)\b.*?\b'
        ]
        
        # Obligation strength keywords
        self.obligation_keywords = {
            'mandatory': ['shall', 'must', 'will', 'required', 'obligated', 'bound'],
            'permissive': ['may', 'can', 'permitted', 'allowed', 'authorized'],
            'prohibited': ['shall not', 'must not', 'cannot', 'prohibited', 'forbidden']
        }
    
    def _compute_concept_embeddings(self) -> Dict[str, List[float]]:
        """Pre-compute embeddings for legal concepts."""
        concept_embeddings = {}
        
        for concept, terms in self.legal_concepts.items():
            # Get embeddings for all terms in concept
            term_embeddings = self.model.encode(terms)
            # Average embeddings to get concept representation
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                concept_embedding = np.mean(term_embeddings, axis=0).tolist()
            else:
                # Fallback averaging
                concept_embedding = [sum(emb[i] for emb in term_embeddings) / len(term_embeddings) 
                                   for i in range(len(term_embeddings[0]))]
            concept_embeddings[concept] = concept_embedding
            
        return concept_embeddings
    
    def enhance_clauses(self, clauses: List[Clause]) -> List[SemanticClause]:
        """Enhance clauses with neural semantic analysis."""
        logger.info(f"Enhancing {len(clauses)} clauses with semantic analysis")
        
        enhanced_clauses = []
        
        # Get embeddings for all clause texts
        clause_texts = [clause.text for clause in clauses]
        clause_embeddings = self.model.encode(clause_texts)
        
        for i, clause in enumerate(clauses):
            # Create enhanced clause
            enhanced_clause = SemanticClause(
                id=clause.id,
                text=clause.text,
                category=clause.category,
                obligations=clause.obligations.copy() if clause.obligations else [],
                parties=clause.parties.copy() if clause.parties else [],
                conditions=clause.conditions.copy() if clause.conditions else [],
                confidence=clause.confidence,
                embedding=clause_embeddings[i].tolist() if hasattr(clause_embeddings[i], 'tolist') else clause_embeddings[i]
            )
            
            # Semantic type classification
            enhanced_clause.semantic_type = self._classify_semantic_type(enhanced_clause)
            
            # Extract legal entities
            enhanced_clause.legal_entities = self._extract_legal_entities(enhanced_clause.text)
            
            # Extract temporal expressions
            enhanced_clause.temporal_expressions = self._extract_temporal_expressions(enhanced_clause.text)
            
            # Extract conditional expressions
            enhanced_clause.conditional_expressions = self._extract_conditional_expressions(enhanced_clause.text)
            
            # Compute obligation strength
            enhanced_clause.obligation_strength = self._compute_obligation_strength(enhanced_clause.text)
            
            enhanced_clauses.append(enhanced_clause)
        
        logger.info(f"Enhanced clauses with semantic types: {[c.semantic_type for c in enhanced_clauses]}")
        return enhanced_clauses
    
    def _classify_semantic_type(self, clause: SemanticClause) -> str:
        """Classify clause semantic type using embedding similarity."""
        if not clause.embedding:
            return "unknown"
        
        best_concept = "unknown"
        best_similarity = -1.0
        
        for concept, concept_embedding in self.concept_embeddings.items():
            similarity = self._compute_cosine_similarity(clause.embedding, concept_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_concept = concept
        
        # Only assign if similarity is above threshold
        if best_similarity > 0.5:  # Adjust threshold as needed
            return best_concept
        else:
            return "unknown"
    
    def _compute_cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                # Use numpy for efficient computation
                vec1 = np.array(embedding1)
                vec2 = np.array(embedding2)
                dot_product = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                return float(dot_product / (norm1 * norm2))
            else:
                # Fallback implementation
                dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
                norm1 = sum(a * a for a in embedding1) ** 0.5
                norm2 = sum(b * b for b in embedding2) ** 0.5
                return dot_product / (norm1 * norm2)
        except:
            return 0.0
    
    def _extract_legal_entities(self, text: str) -> List[str]:
        """Extract legal entities from text using pattern matching."""
        entities = []
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity_text = match.group(0)
                if len(entity_text) > 2:  # Filter out very short matches
                    entities.append(f"{entity_type}:{entity_text}")
        
        return list(set(entities))  # Remove duplicates
    
    def _extract_temporal_expressions(self, text: str) -> List[str]:
        """Extract temporal expressions from text."""
        temporal_expressions = []
        
        for pattern in self.temporal_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                temporal_expressions.append(match.group(0))
        
        return list(set(temporal_expressions))
    
    def _extract_conditional_expressions(self, text: str) -> List[str]:
        """Extract conditional expressions from text."""
        conditional_expressions = []
        
        for pattern in self.conditional_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                conditional_expressions.append(match.group(0))
        
        return list(set(conditional_expressions))
    
    def _compute_obligation_strength(self, text: str) -> float:
        """Compute obligation strength based on modal verbs and legal language."""
        text_lower = text.lower()
        
        mandatory_count = sum(1 for keyword in self.obligation_keywords['mandatory'] if keyword in text_lower)
        permissive_count = sum(1 for keyword in self.obligation_keywords['permissive'] if keyword in text_lower)
        prohibited_count = sum(1 for keyword in self.obligation_keywords['prohibited'] if keyword in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        # Normalize by text length and weight different types
        mandatory_strength = (mandatory_count / total_words) * 1.0
        permissive_strength = (permissive_count / total_words) * 0.5
        prohibited_strength = (prohibited_count / total_words) * 1.0
        
        return min(1.0, mandatory_strength + permissive_strength + prohibited_strength)
    
    def find_similar_clauses(self, query_clause: SemanticClause, clauses: List[SemanticClause], 
                           similarity_threshold: float = 0.7) -> List[Tuple[SemanticClause, float]]:
        """Find clauses similar to query clause based on semantic embedding."""
        if not query_clause.embedding:
            return []
        
        similar_clauses = []
        
        for clause in clauses:
            if clause.id == query_clause.id or not clause.embedding:
                continue
                
            similarity = self._compute_cosine_similarity(query_clause.embedding, clause.embedding)
            
            if similarity >= similarity_threshold:
                similar_clauses.append((clause, similarity))
        
        # Sort by similarity descending
        similar_clauses.sort(key=lambda x: x[1], reverse=True)
        
        return similar_clauses
    
    def extract_contract_graph_data(self, enhanced_clauses: List[SemanticClause]) -> Dict[str, Any]:
        """Extract data for contract graph construction (Generation 2 preparation)."""
        
        nodes = []
        edges = []
        
        # Create nodes for each clause
        for clause in enhanced_clauses:
            node = {
                'id': clause.id,
                'text': clause.text[:200],  # Truncate for graph visualization
                'type': 'clause',
                'semantic_type': clause.semantic_type,
                'obligation_strength': clause.obligation_strength,
                'embedding': clause.embedding
            }
            nodes.append(node)
            
            # Create nodes for entities
            for entity in clause.legal_entities:
                entity_type, entity_name = entity.split(':', 1)
                entity_node = {
                    'id': f"entity_{entity_name}",
                    'text': entity_name,
                    'type': 'entity',
                    'entity_type': entity_type
                }
                nodes.append(entity_node)
                
                # Create edge from clause to entity
                edges.append({
                    'source': clause.id,
                    'target': f"entity_{entity_name}",
                    'type': 'mentions'
                })
        
        # Find semantic similarity edges between clauses
        for i, clause1 in enumerate(enhanced_clauses):
            for j, clause2 in enumerate(enhanced_clauses[i+1:], i+1):
                if clause1.embedding and clause2.embedding:
                    similarity = self._compute_cosine_similarity(clause1.embedding, clause2.embedding)
                    if similarity > 0.7:  # High similarity threshold
                        edges.append({
                            'source': clause1.id,
                            'target': clause2.id,
                            'type': 'similar_to',
                            'weight': similarity
                        })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'total_clauses': len(enhanced_clauses),
                'semantic_types': list(set(c.semantic_type for c in enhanced_clauses)),
                'entity_types': list(set(e.split(':', 1)[0] for c in enhanced_clauses for e in c.legal_entities))
            }
        }
    
    def generate_clause_summary(self, clause: SemanticClause) -> str:
        """Generate a natural language summary of the clause."""
        summary_parts = []
        
        # Semantic type
        if clause.semantic_type and clause.semantic_type != "unknown":
            summary_parts.append(f"This is a {clause.semantic_type.replace('_', ' ')} clause")
        
        # Obligation strength
        if clause.obligation_strength > 0.7:
            summary_parts.append("with strong mandatory obligations")
        elif clause.obligation_strength > 0.3:
            summary_parts.append("with moderate obligations")
        elif clause.obligation_strength > 0:
            summary_parts.append("with permissive provisions")
        
        # Parties
        if clause.parties:
            parties_str = ", ".join(clause.parties)
            summary_parts.append(f"involving {parties_str}")
        
        # Entities
        if clause.legal_entities:
            entity_types = list(set(e.split(':', 1)[0] for e in clause.legal_entities))
            summary_parts.append(f"referencing {', '.join(entity_types)}")
        
        # Temporal aspects
        if clause.temporal_expressions:
            summary_parts.append(f"with time constraints: {', '.join(clause.temporal_expressions)}")
        
        # Conditional aspects
        if clause.conditional_expressions:
            summary_parts.append("containing conditional provisions")
        
        if not summary_parts:
            return f"General contract clause (ID: {clause.id})"
        
        return ". ".join(summary_parts) + "."