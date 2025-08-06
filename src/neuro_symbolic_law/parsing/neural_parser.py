"""
Neural contract parsing with advanced NLP capabilities.
"""

from typing import Dict, List, Optional, Any, Tuple, Set
import logging
from dataclasses import dataclass, field
import re
from collections import defaultdict

# NLP imports with fallbacks for deployment flexibility
try:
    import torch
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available - using rule-based fallbacks")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from .contract_parser import ContractParser, ParsedContract, Clause, ContractParty

logger = logging.getLogger(__name__)


@dataclass
class SemanticRole:
    """Semantic role extracted from legal text."""
    actor: str
    action: str
    object: str
    purpose: Optional[str] = None
    condition: Optional[str] = None
    modality: str = "obligation"  # obligation, permission, prohibition
    confidence: float = 1.0


@dataclass
class LegalEntity:
    """Legal entity extracted from text."""
    name: str
    entity_type: str  # person, organization, system, etc.
    roles: Set[str] = field(default_factory=set)
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class ContractGraph:
    """Graph representation of contract relationships."""
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    node_features: Optional[Any] = None  # Tensor for GNN
    edge_index: Optional[Any] = None     # Tensor for GNN


class NeuralContractParser(ContractParser):
    """
    Advanced neural contract parser with NLP and graph neural networks.
    
    Features:
    - BERT-based clause classification
    - Semantic role labeling
    - Graph neural network contract representation
    - Advanced entity recognition
    - Cross-reference resolution
    """
    
    def __init__(
        self,
        model: str = 'legal-bert-base',
        enable_gpu: bool = False,
        debug: bool = False
    ):
        """
        Initialize neural parser.
        
        Args:
            model: Pre-trained model to use
            enable_gpu: Use GPU if available
            debug: Enable debug logging
        """
        super().__init__(model=model, debug=debug)
        
        self.enable_gpu = enable_gpu and torch.cuda.is_available() if TRANSFORMERS_AVAILABLE else False
        self.device = 'cuda' if self.enable_gpu else 'cpu'
        
        # Load NLP models
        self.nlp_model = self._load_nlp_model()
        self.bert_model = self._load_bert_model(model)
        
        # Semantic role patterns (enhanced)
        self.srl_patterns = self._initialize_srl_patterns()
        
        # Legal entity patterns
        self.entity_patterns = self._initialize_entity_patterns()
        
        # Contract graph builder
        self.graph_builder = ContractGraphBuilder()
    
    def parse_enhanced(
        self,
        contract_text: str,
        contract_id: Optional[str] = None,
        extract_semantics: bool = True,
        build_graph: bool = True
    ) -> Tuple[ParsedContract, Optional[ContractGraph]]:
        """
        Enhanced parsing with semantic analysis and graph construction.
        
        Args:
            contract_text: Raw contract text
            contract_id: Contract identifier
            extract_semantics: Extract semantic roles
            build_graph: Build contract graph
            
        Returns:
            (parsed_contract, contract_graph)
        """
        logger.info(f"Enhanced parsing with semantics={extract_semantics}, graph={build_graph}")
        
        # Basic parsing
        parsed_contract = self.parse(contract_text, contract_id)
        
        # Enhanced clause analysis
        if extract_semantics:
            self._enhance_clauses_with_semantics(parsed_contract)
        
        # Build contract graph
        contract_graph = None
        if build_graph:
            contract_graph = self.graph_builder.build_graph(parsed_contract)
        
        return parsed_contract, contract_graph
    
    def extract_semantic_roles(self, text: str) -> List[SemanticRole]:
        """Extract semantic roles from legal text using NLP."""
        
        logger.debug(f"Extracting semantic roles from: {text[:100]}...")
        
        roles = []
        
        try:
            # Use spaCy if available
            if SPACY_AVAILABLE and self.nlp_model:
                doc = self.nlp_model(text)
                roles.extend(self._extract_roles_with_spacy(doc))
            
            # Fallback to pattern matching
            roles.extend(self._extract_roles_with_patterns(text))
            
            # Remove duplicates and low-confidence roles
            roles = self._deduplicate_roles(roles)
            
        except Exception as e:
            logger.error(f"Error extracting semantic roles: {e}")
        
        return roles
    
    def classify_clause_neural(self, clause_text: str) -> Tuple[str, float]:
        """Classify clause using neural model."""
        
        if not TRANSFORMERS_AVAILABLE or not self.bert_model:
            # Fallback to rule-based classification
            category = self._classify_clause(clause_text)
            return category or "general", 0.6
        
        try:
            # Tokenize and encode
            inputs = self.bert_model['tokenizer'](
                clause_text,
                return_tensors='pt',
                max_length=512,
                truncation=True,
                padding=True
            )
            
            if self.enable_gpu:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.bert_model['model'](**inputs)
                predictions = torch.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
                confidence = predictions[0][predicted_class].item()
            
            # Map to category
            categories = [
                'data_processing', 'liability', 'termination', 
                'confidentiality', 'compliance', 'security',
                'intellectual_property', 'payment'
            ]
            
            category = categories[predicted_class] if predicted_class < len(categories) else 'general'
            return category, confidence
            
        except Exception as e:
            logger.error(f"Error in neural classification: {e}")
            category = self._classify_clause(clause_text)
            return category or "general", 0.5
    
    def extract_legal_entities(self, text: str) -> List[LegalEntity]:
        """Extract legal entities using advanced NLP."""
        
        entities = []
        
        try:
            # Use spaCy NER if available
            if SPACY_AVAILABLE and self.nlp_model:
                doc = self.nlp_model(text)
                entities.extend(self._extract_entities_with_spacy(doc))
            
            # Pattern-based entity extraction
            entities.extend(self._extract_entities_with_patterns(text))
            
            # Deduplicate and enhance
            entities = self._deduplicate_entities(entities)
            entities = self._enhance_entities_with_context(entities, text)
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
        
        return entities
    
    def resolve_cross_references(self, parsed_contract: ParsedContract) -> None:
        """Resolve cross-references between clauses."""
        
        logger.info("Resolving cross-references")
        
        try:
            # Find reference patterns
            reference_patterns = [
                r'section\s+(\d+(?:\.\d+)*)',
                r'clause\s+(\d+(?:\.\d+)*)',
                r'paragraph\s+(\d+(?:\.\d+)*)',
                r'article\s+(\d+(?:\.\d+)*)',
                r'as\s+defined\s+(?:in|above|below)',
                r'pursuant\s+to\s+(?:section|clause)\s+(\d+(?:\.\d+)*)',
            ]
            
            for clause in parsed_contract.clauses:
                references = []
                
                for pattern in reference_patterns:
                    matches = re.finditer(pattern, clause.text, re.IGNORECASE)
                    for match in matches:
                        ref_text = match.group(0)
                        if match.groups():
                            ref_number = match.group(1)
                            references.append({
                                'type': 'section_reference',
                                'text': ref_text,
                                'number': ref_number,
                                'position': match.span()
                            })
                        else:
                            references.append({
                                'type': 'contextual_reference',
                                'text': ref_text,
                                'position': match.span()
                            })
                
                # Store references in clause metadata
                if not hasattr(clause, 'metadata'):
                    clause.metadata = {}
                clause.metadata['cross_references'] = references
                
        except Exception as e:
            logger.error(f"Error resolving cross-references: {e}")
    
    def _enhance_clauses_with_semantics(self, contract: ParsedContract) -> None:
        """Enhance clauses with semantic analysis."""
        
        logger.info("Enhancing clauses with semantic analysis")
        
        for clause in contract.clauses:
            try:
                # Extract semantic roles
                semantic_roles = self.extract_semantic_roles(clause.text)
                
                # Neural classification
                category, confidence = self.classify_clause_neural(clause.text)
                clause.category = category
                clause.confidence = confidence
                
                # Store semantic information
                if not hasattr(clause, 'semantic_roles'):
                    clause.semantic_roles = semantic_roles
                
                # Extract entities
                entities = self.extract_legal_entities(clause.text)
                if not hasattr(clause, 'entities'):
                    clause.entities = entities
                
                # Enhanced obligations extraction
                obligations = self._extract_enhanced_obligations(clause.text, semantic_roles)
                clause.obligations = obligations
                
            except Exception as e:
                logger.error(f"Error enhancing clause {clause.id}: {e}")
    
    def _load_nlp_model(self):
        """Load spaCy NLP model."""
        
        if not SPACY_AVAILABLE:
            logger.warning("spaCy not available")
            return None
        
        try:
            # Try to load legal-specific model first, fallback to general
            model_names = ['en_legal_ner_trf', 'en_core_web_sm', 'en_core_web_md']
            
            for model_name in model_names:
                try:
                    nlp = spacy.load(model_name)
                    logger.info(f"Loaded spaCy model: {model_name}")
                    return nlp
                except OSError:
                    continue
            
            logger.warning("No spaCy models available")
            return None
            
        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")
            return None
    
    def _load_bert_model(self, model_name: str):
        """Load BERT-based model for classification."""
        
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available")
            return None
        
        try:
            # Try legal-specific BERT models first
            legal_models = [
                'nlpaueb/legal-bert-base-uncased',
                'law-ai/InLegalBERT',
                'bert-base-uncased'  # Fallback
            ]
            
            for model_path in legal_models:
                try:
                    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
                    model = transformers.AutoModelForSequenceClassification.from_pretrained(
                        model_path,
                        num_labels=8  # Number of clause categories
                    )
                    
                    if self.enable_gpu:
                        model = model.to(self.device)
                    
                    logger.info(f"Loaded BERT model: {model_path}")
                    return {'tokenizer': tokenizer, 'model': model}
                    
                except Exception as e:
                    logger.debug(f"Could not load {model_path}: {e}")
                    continue
            
            logger.warning("No BERT models available")
            return None
            
        except Exception as e:
            logger.error(f"Error loading BERT model: {e}")
            return None
    
    def _initialize_srl_patterns(self) -> List[Dict[str, Any]]:
        """Initialize semantic role labeling patterns."""
        
        return [
            {
                'pattern': r'(?P<actor>[\w\s]+?)\s+shall\s+(?P<action>[\w\s]+?)(?:\s+(?P<object>[\w\s]+?))?(?:\s+for\s+(?P<purpose>[\w\s]+?))?',
                'modality': 'obligation',
                'confidence': 0.9
            },
            {
                'pattern': r'(?P<actor>[\w\s]+?)\s+must\s+(?P<action>[\w\s]+?)(?:\s+(?P<object>[\w\s]+?))?',
                'modality': 'obligation', 
                'confidence': 0.9
            },
            {
                'pattern': r'(?P<actor>[\w\s]+?)\s+may\s+(?P<action>[\w\s]+?)(?:\s+(?P<object>[\w\s]+?))?',
                'modality': 'permission',
                'confidence': 0.8
            },
            {
                'pattern': r'(?P<actor>[\w\s]+?)\s+(?:is|are)\s+(?:prohibited|forbidden)\s+from\s+(?P<action>[\w\s]+?)(?:\s+(?P<object>[\w\s]+?))?',
                'modality': 'prohibition',
                'confidence': 0.9
            },
            {
                'pattern': r'(?P<actor>[\w\s]+?)\s+agrees?\s+to\s+(?P<action>[\w\s]+?)(?:\s+(?P<object>[\w\s]+?))?',
                'modality': 'obligation',
                'confidence': 0.8
            }
        ]
    
    def _initialize_entity_patterns(self) -> List[Dict[str, Any]]:
        """Initialize legal entity recognition patterns."""
        
        return [
            {
                'pattern': r'(?P<entity>[\w\s]+?)\s+(?:Inc\.|Corp\.|LLC|Ltd\.|Corporation|Company)',
                'type': 'corporation',
                'confidence': 0.9
            },
            {
                'pattern': r'(?P<entity>[\w\s]+?)\s+(?:University|College|Institute)',
                'type': 'educational',
                'confidence': 0.9
            },
            {
                'pattern': r'(?P<entity>[\w\s]+?)\s+(?:Government|Agency|Department|Authority)',
                'type': 'government',
                'confidence': 0.9
            },
            {
                'pattern': r'data\s+(?:controller|processor|subject)',
                'type': 'data_role',
                'confidence': 0.8
            },
            {
                'pattern': r'(?:service\s+)?provider|vendor|contractor|supplier',
                'type': 'service_provider',
                'confidence': 0.7
            }
        ]
    
    def _extract_roles_with_spacy(self, doc) -> List[SemanticRole]:
        """Extract semantic roles using spaCy."""
        
        roles = []
        
        try:
            for sent in doc.sents:
                # Find verbs and their arguments
                for token in sent:
                    if token.pos_ == 'VERB' and token.dep_ == 'ROOT':
                        # Extract subject (actor)
                        subjects = [child for child in token.children if child.dep_ in ['nsubj', 'nsubjpass']]
                        
                        # Extract object
                        objects = [child for child in token.children if child.dep_ in ['dobj', 'pobj']]
                        
                        # Extract purpose/condition
                        purposes = [child for child in token.children if child.dep_ == 'advcl']
                        
                        if subjects:
                            actor = subjects[0].text
                            action = token.lemma_
                            obj = objects[0].text if objects else ""
                            purpose = purposes[0].text if purposes else None
                            
                            # Determine modality
                            modality = self._determine_modality(sent.text)
                            
                            roles.append(SemanticRole(
                                actor=actor,
                                action=action,
                                object=obj,
                                purpose=purpose,
                                modality=modality,
                                confidence=0.7
                            ))
        
        except Exception as e:
            logger.error(f"Error extracting roles with spaCy: {e}")
        
        return roles
    
    def _extract_roles_with_patterns(self, text: str) -> List[SemanticRole]:
        """Extract semantic roles using patterns."""
        
        roles = []
        
        for pattern_info in self.srl_patterns:
            pattern = pattern_info['pattern']
            modality = pattern_info['modality']
            confidence = pattern_info['confidence']
            
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groupdict()
                
                role = SemanticRole(
                    actor=groups.get('actor', '').strip(),
                    action=groups.get('action', '').strip(),
                    object=groups.get('object', '').strip(),
                    purpose=groups.get('purpose', ''),
                    modality=modality,
                    confidence=confidence
                )
                
                if role.actor and role.action:
                    roles.append(role)
        
        return roles
    
    def _extract_entities_with_spacy(self, doc) -> List[LegalEntity]:
        """Extract entities using spaCy NER."""
        
        entities = []
        
        try:
            for ent in doc.ents:
                entity_type = self._map_spacy_label_to_legal(ent.label_)
                
                if entity_type:
                    entity = LegalEntity(
                        name=ent.text,
                        entity_type=entity_type,
                        confidence=0.8
                    )
                    entities.append(entity)
        
        except Exception as e:
            logger.error(f"Error extracting entities with spaCy: {e}")
        
        return entities
    
    def _extract_entities_with_patterns(self, text: str) -> List[LegalEntity]:
        """Extract entities using patterns."""
        
        entities = []
        
        for pattern_info in self.entity_patterns:
            pattern = pattern_info['pattern']
            entity_type = pattern_info['type']
            confidence = pattern_info['confidence']
            
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if 'entity' in match.groupdict():
                    entity_name = match.groupdict()['entity'].strip()
                else:
                    entity_name = match.group(0).strip()
                
                if entity_name:
                    entity = LegalEntity(
                        name=entity_name,
                        entity_type=entity_type,
                        confidence=confidence
                    )
                    entities.append(entity)
        
        return entities
    
    def _determine_modality(self, text: str) -> str:
        """Determine modality from text."""
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['shall', 'must', 'required', 'obligation']):
            return 'obligation'
        elif any(word in text_lower for word in ['may', 'can', 'permitted', 'allowed']):
            return 'permission'
        elif any(word in text_lower for word in ['prohibited', 'forbidden', 'shall not', 'must not']):
            return 'prohibition'
        else:
            return 'statement'
    
    def _map_spacy_label_to_legal(self, spacy_label: str) -> Optional[str]:
        """Map spaCy entity labels to legal entity types."""
        
        mapping = {
            'PERSON': 'person',
            'ORG': 'organization',
            'GPE': 'jurisdiction',
            'LAW': 'legal_reference',
            'MONEY': 'monetary_amount',
            'DATE': 'date',
            'TIME': 'time',
            'PERCENT': 'percentage'
        }
        
        return mapping.get(spacy_label)
    
    def _deduplicate_roles(self, roles: List[SemanticRole]) -> List[SemanticRole]:
        """Remove duplicate semantic roles."""
        
        unique_roles = []
        seen = set()
        
        for role in roles:
            key = (role.actor.lower(), role.action.lower(), role.object.lower())
            if key not in seen:
                seen.add(key)
                unique_roles.append(role)
        
        return unique_roles
    
    def _deduplicate_entities(self, entities: List[LegalEntity]) -> List[LegalEntity]:
        """Remove duplicate entities."""
        
        unique_entities = []
        seen = set()
        
        for entity in entities:
            key = entity.name.lower()
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _enhance_entities_with_context(self, entities: List[LegalEntity], text: str) -> List[LegalEntity]:
        """Enhance entities with contextual information."""
        
        # Add role information based on context
        for entity in entities:
            entity_context = self._get_entity_context(entity.name, text)
            
            if 'controller' in entity_context.lower():
                entity.roles.add('data_controller')
            if 'processor' in entity_context.lower():
                entity.roles.add('data_processor')
            if 'provider' in entity_context.lower():
                entity.roles.add('service_provider')
            if 'client' in entity_context.lower() or 'customer' in entity_context.lower():
                entity.roles.add('client')
        
        return entities
    
    def _get_entity_context(self, entity_name: str, text: str, window: int = 100) -> str:
        """Get text context around entity mention."""
        
        try:
            pos = text.lower().find(entity_name.lower())
            if pos != -1:
                start = max(0, pos - window)
                end = min(len(text), pos + len(entity_name) + window)
                return text[start:end]
        except Exception:
            pass
        
        return ""
    
    def _extract_enhanced_obligations(self, text: str, semantic_roles: List[SemanticRole]) -> List[str]:
        """Extract enhanced obligations using semantic roles."""
        
        obligations = []
        
        # Extract from semantic roles
        for role in semantic_roles:
            if role.modality == 'obligation':
                obligation = f"{role.actor} {role.action}"
                if role.object:
                    obligation += f" {role.object}"
                if role.purpose:
                    obligation += f" for {role.purpose}"
                obligations.append(obligation)
        
        # Fallback to pattern-based extraction
        obligations.extend(self._extract_obligations(text))
        
        return list(set(obligations))  # Remove duplicates


class ContractGraphBuilder:
    """Builds graph representations of contracts for GNN processing."""
    
    def __init__(self):
        """Initialize graph builder."""
        self.node_types = {
            'clause': 0,
            'entity': 1,
            'obligation': 2,
            'right': 3,
            'condition': 4
        }
        
        self.edge_types = {
            'references': 0,
            'obligates': 1,
            'permits': 2,
            'restricts': 3,
            'defines': 4
        }
    
    def build_graph(self, contract: ParsedContract) -> ContractGraph:
        """Build graph representation of contract."""
        
        logger.info(f"Building contract graph for {contract.id}")
        
        nodes = []
        edges = []
        node_id_map = {}
        current_node_id = 0
        
        try:
            # Add clause nodes
            for clause in contract.clauses:
                node = {
                    'id': current_node_id,
                    'type': 'clause',
                    'text': clause.text,
                    'category': clause.category or 'general',
                    'features': self._extract_node_features(clause.text)
                }
                nodes.append(node)
                node_id_map[f"clause_{clause.id}"] = current_node_id
                current_node_id += 1
            
            # Add entity nodes
            for party in contract.parties:
                node = {
                    'id': current_node_id,
                    'type': 'entity',
                    'text': party.name,
                    'category': party.role,
                    'features': self._extract_node_features(party.name)
                }
                nodes.append(node)
                node_id_map[f"entity_{party.name}"] = current_node_id
                current_node_id += 1
            
            # Add edges based on relationships
            edges.extend(self._create_clause_relationships(contract, node_id_map))
            edges.extend(self._create_entity_relationships(contract, node_id_map))
            
        except Exception as e:
            logger.error(f"Error building contract graph: {e}")
        
        return ContractGraph(nodes=nodes, edges=edges)
    
    def _extract_node_features(self, text: str) -> List[float]:
        """Extract numerical features for graph nodes."""
        
        # Simple features for now - Generation 3 would use embeddings
        features = [
            len(text),                    # Text length
            len(text.split()),           # Word count
            text.count(','),             # Comma count (complexity)
            text.count('shall'),         # Obligation indicators
            text.count('may'),           # Permission indicators
            text.count('not'),           # Negation count
            len(re.findall(r'\d+', text)), # Number count
            1.0 if any(word in text.lower() for word in ['personal', 'data']) else 0.0  # Data processing indicator
        ]
        
        return features
    
    def _create_clause_relationships(self, contract: ParsedContract, node_id_map: Dict[str, int]) -> List[Dict[str, Any]]:
        """Create edges between clauses."""
        
        edges = []
        
        try:
            # Create sequential edges between clauses
            clause_ids = [f"clause_{clause.id}" for clause in contract.clauses]
            
            for i in range(len(clause_ids) - 1):
                if clause_ids[i] in node_id_map and clause_ids[i + 1] in node_id_map:
                    edges.append({
                        'source': node_id_map[clause_ids[i]],
                        'target': node_id_map[clause_ids[i + 1]],
                        'type': 'references',
                        'weight': 1.0
                    })
            
            # Add cross-reference edges if available
            for clause in contract.clauses:
                if hasattr(clause, 'metadata') and 'cross_references' in clause.metadata:
                    for ref in clause.metadata['cross_references']:
                        # Find referenced clause and create edge
                        # This would be more sophisticated in a full implementation
                        pass
        
        except Exception as e:
            logger.error(f"Error creating clause relationships: {e}")
        
        return edges
    
    def _create_entity_relationships(self, contract: ParsedContract, node_id_map: Dict[str, int]) -> List[Dict[str, Any]]:
        """Create edges between entities and clauses."""
        
        edges = []
        
        try:
            # Connect entities to clauses where they are mentioned
            for party in contract.parties:
                entity_key = f"entity_{party.name}"
                if entity_key in node_id_map:
                    entity_id = node_id_map[entity_key]
                    
                    for clause in contract.clauses:
                        if party.name.lower() in clause.text.lower():
                            clause_key = f"clause_{clause.id}"
                            if clause_key in node_id_map:
                                clause_id = node_id_map[clause_key]
                                
                                # Determine relationship type
                                edge_type = self._determine_entity_clause_relationship(
                                    party, clause
                                )
                                
                                edges.append({
                                    'source': entity_id,
                                    'target': clause_id,
                                    'type': edge_type,
                                    'weight': 1.0
                                })
        
        except Exception as e:
            logger.error(f"Error creating entity relationships: {e}")
        
        return edges
    
    def _determine_entity_clause_relationship(self, party: ContractParty, clause: Clause) -> str:
        """Determine relationship type between entity and clause."""
        
        clause_text = clause.text.lower()
        
        if 'shall' in clause_text or 'must' in clause_text:
            return 'obligates'
        elif 'may' in clause_text or 'permitted' in clause_text:
            return 'permits'
        elif 'prohibited' in clause_text or 'forbidden' in clause_text:
            return 'restricts'
        else:
            return 'references'