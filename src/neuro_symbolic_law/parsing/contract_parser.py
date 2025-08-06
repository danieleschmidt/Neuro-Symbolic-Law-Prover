"""
Contract parsing and understanding using neural models.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class Clause:
    """Represents a single contract clause."""
    id: str
    text: str
    category: Optional[str] = None
    obligations: List[str] = field(default_factory=list)
    parties: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class ContractParty:
    """Represents a party to the contract."""
    name: str
    role: str
    entity_type: Optional[str] = None


@dataclass
class ParsedContract:
    """Represents a fully parsed contract."""
    id: str
    title: str
    text: str
    clauses: List[Clause]
    parties: List[ContractParty]
    contract_type: Optional[str] = None
    effective_date: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_clauses_by_category(self, category: str) -> List[Clause]:
        """Get all clauses of a specific category."""
        return [clause for clause in self.clauses if clause.category == category]
    
    def get_clauses_containing(self, keywords: List[str]) -> List[Clause]:
        """Get clauses containing any of the specified keywords."""
        relevant_clauses = []
        for clause in self.clauses:
            clause_text = clause.text.lower()
            if any(keyword.lower() in clause_text for keyword in keywords):
                relevant_clauses.append(clause)
        return relevant_clauses
    
    def extract_field(self, field_name: str) -> Optional[str]:
        """Extract a specific field from contract metadata or text."""
        # Check metadata first
        if field_name in self.metadata:
            return self.metadata[field_name]
        
        # Basic text extraction patterns
        patterns = {
            'application_area': r'(?:application|service|system).*?(?:area|domain|sector):\s*([^\n.]+)',
            'deployment_context': r'(?:deployment|operating|production).*?(?:context|environment):\s*([^\n.]+)',
            'data_retention': r'(?:data|information).*?(?:retention|kept|stored).*?(?:for|period|duration).*?([^\n.]+)',
        }
        
        if field_name in patterns:
            pattern = patterns[field_name]
            match = re.search(pattern, self.text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None


class ContractParser:
    """
    Neural contract parser that extracts structured information from legal text.
    
    Generation 1: Basic rule-based parsing
    Generation 2: Will add BERT-based NLP and GNN graph construction
    Generation 3: Will add advanced semantic role labeling
    """
    
    def __init__(self, model: str = 'basic', debug: bool = False):
        """
        Initialize contract parser.
        
        Args:
            model: Model to use ('basic', 'legal-bert-base', etc.)
            debug: Enable debug logging
        """
        self.model = model
        self.debug = debug
        
        if debug:
            logging.basicConfig(level=logging.DEBUG)
        
        # Generation 1: Basic patterns for clause extraction
        self.clause_patterns = [
            r'\d+\.\s+([^.]+\.)',  # Numbered clauses
            r'[A-Z][^.]+\.',       # Sentence-based clauses
            r'(?:WHEREAS|THEREFORE|PROVIDED|SUBJECT TO)[^.]+\.',  # Legal connectors
        ]
        
        self.party_patterns = [
            r'(?:Company|Corporation|Inc\.|LLC|Ltd\.|Party|Client|Vendor|Provider|Contractor|Service Provider)(?:\s+[A-Z][a-zA-Z\s]+)?',
            r'"([^"]+)"(?:\s+\([^)]+\))?',  # Quoted names
        ]
        
        # Common legal categories
        self.clause_categories = {
            'data_processing': ['data', 'processing', 'personal information', 'collection'],
            'liability': ['liable', 'liability', 'damages', 'indemnify', 'indemnification'],
            'termination': ['terminate', 'termination', 'end', 'expire', 'expiry'],
            'confidentiality': ['confidential', 'non-disclosure', 'proprietary', 'secret'],
            'compliance': ['comply', 'compliance', 'regulation', 'law', 'legal'],
            'security': ['security', 'secure', 'protection', 'safeguard', 'encrypt'],
            'intellectual_property': ['intellectual property', 'copyright', 'patent', 'trademark'],
            'payment': ['payment', 'fee', 'cost', 'price', 'invoice', 'billing'],
        }
    
    def parse(self, contract_text: str, contract_id: Optional[str] = None) -> ParsedContract:
        """
        Parse contract text into structured format.
        
        Args:
            contract_text: Raw contract text
            contract_id: Optional contract identifier
            
        Returns:
            Parsed contract object
        """
        if contract_id is None:
            contract_id = f"contract_{hash(contract_text) % 100000}"
        
        logger.info(f"Parsing contract {contract_id} ({len(contract_text)} characters)")
        
        # Extract basic information
        title = self._extract_title(contract_text)
        parties = self._extract_parties(contract_text)
        clauses = self._extract_clauses(contract_text)
        contract_type = self._infer_contract_type(contract_text)
        
        # Classify clauses
        for clause in clauses:
            clause.category = self._classify_clause(clause.text)
            clause.parties = self._extract_clause_parties(clause.text, parties)
            clause.obligations = self._extract_obligations(clause.text)
        
        logger.info(f"Extracted {len(clauses)} clauses and {len(parties)} parties")
        
        return ParsedContract(
            id=contract_id,
            title=title,
            text=contract_text,
            clauses=clauses,
            parties=parties,
            contract_type=contract_type
        )
    
    def _extract_title(self, text: str) -> str:
        """Extract contract title."""
        lines = text.strip().split('\n')
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if line and len(line) < 200:  # Reasonable title length
                if any(word in line.upper() for word in ['AGREEMENT', 'CONTRACT', 'TERMS']):
                    return line
        
        # Fallback
        first_line = lines[0].strip() if lines else "Untitled Contract"
        return first_line[:100] + "..." if len(first_line) > 100 else first_line
    
    def _extract_parties(self, text: str) -> List[ContractParty]:
        """Extract contract parties."""
        parties = []
        
        for pattern in self.party_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                party_name = match.group(1) if match.lastindex else match.group(0)
                party_name = party_name.strip('"').strip()
                
                # Determine role
                role = "party"
                context = text[max(0, match.start()-50):match.end()+50].lower()
                if any(word in context for word in ['client', 'customer', 'buyer']):
                    role = "client"
                elif any(word in context for word in ['vendor', 'provider', 'seller', 'contractor']):
                    role = "provider"
                
                parties.append(ContractParty(
                    name=party_name,
                    role=role,
                    entity_type=self._infer_entity_type(party_name)
                ))
        
        # Remove duplicates
        seen = set()
        unique_parties = []
        for party in parties:
            if party.name not in seen and len(party.name) > 2:
                seen.add(party.name)
                unique_parties.append(party)
        
        # If no parties found, create default ones from common patterns
        if not unique_parties and any(word in text.lower() for word in ['agreement', 'contract', 'between']):
            # Try to extract from "between X and Y" patterns
            between_pattern = r'between\s+([^,\n]+?)\s+and\s+([^,\n.]+)'
            between_match = re.search(between_pattern, text, re.IGNORECASE)
            if between_match:
                party1 = between_match.group(1).strip()
                party2 = between_match.group(2).strip()
                unique_parties.extend([
                    ContractParty(name=party1, role="party"),
                    ContractParty(name=party2, role="party")
                ])
        
        return unique_parties[:10]  # Limit to reasonable number
    
    def _extract_clauses(self, text: str) -> List[Clause]:
        """Extract individual clauses from contract text."""
        clauses = []
        clause_id = 1
        
        # Split text into potential clauses using various patterns
        for pattern in self.clause_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)
            for match in matches:
                clause_text = match.group(1) if match.lastindex else match.group(0)
                clause_text = clause_text.strip()
                
                if len(clause_text) > 20 and len(clause_text) < 2000:  # Reasonable clause length
                    clauses.append(Clause(
                        id=f"clause_{clause_id}",
                        text=clause_text,
                        confidence=0.8
                    ))
                    clause_id += 1
        
        # If no clauses found with patterns, split by paragraphs/sentences
        if not clauses:
            sentences = re.split(r'[.!?]+\s+', text)
            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if len(sentence) > 20:
                    clauses.append(Clause(
                        id=f"sentence_{i+1}",
                        text=sentence,
                        confidence=0.6
                    ))
        
        return clauses[:50]  # Limit to reasonable number
    
    def _classify_clause(self, clause_text: str) -> Optional[str]:
        """Classify clause into legal category."""
        clause_lower = clause_text.lower()
        
        for category, keywords in self.clause_categories.items():
            if any(keyword in clause_lower for keyword in keywords):
                return category
        
        return None
    
    def _extract_clause_parties(self, clause_text: str, all_parties: List[ContractParty]) -> List[str]:
        """Extract parties mentioned in a specific clause."""
        mentioned_parties = []
        clause_lower = clause_text.lower()
        
        for party in all_parties:
            if party.name.lower() in clause_lower:
                mentioned_parties.append(party.name)
        
        return mentioned_parties
    
    def _extract_obligations(self, clause_text: str) -> List[str]:
        """Extract obligations from clause text."""
        obligations = []
        
        # Look for obligation patterns
        obligation_patterns = [
            r'shall\s+([^.]+)',
            r'must\s+([^.]+)',
            r'agrees?\s+to\s+([^.]+)',
            r'undertakes?\s+to\s+([^.]+)',
            r'responsible\s+for\s+([^.]+)',
        ]
        
        for pattern in obligation_patterns:
            matches = re.finditer(pattern, clause_text, re.IGNORECASE)
            for match in matches:
                obligation = match.group(1).strip()
                if len(obligation) > 5 and len(obligation) < 200:
                    obligations.append(obligation)
        
        return obligations[:5]  # Limit obligations per clause
    
    def _infer_contract_type(self, text: str) -> Optional[str]:
        """Infer the type of contract."""
        text_lower = text.lower()
        
        contract_types = {
            'data_processing_agreement': ['data processing', 'dpa', 'personal data'],
            'service_agreement': ['service agreement', 'services', 'sla'],
            'employment': ['employment', 'employee', 'job'],
            'nda': ['non-disclosure', 'confidentiality', 'nda'],
            'license': ['license', 'licensing', 'intellectual property'],
            'purchase': ['purchase', 'sale', 'buy', 'sell'],
        }
        
        for contract_type, keywords in contract_types.items():
            if any(keyword in text_lower for keyword in keywords):
                return contract_type
        
        return None
    
    def _infer_entity_type(self, party_name: str) -> Optional[str]:
        """Infer entity type from party name."""
        if any(suffix in party_name for suffix in ['Inc.', 'Corp.', 'LLC', 'Ltd.', 'Corporation']):
            return "corporation"
        elif 'University' in party_name or 'College' in party_name:
            return "educational"
        elif 'Government' in party_name or 'State' in party_name:
            return "government"
        else:
            return "individual"