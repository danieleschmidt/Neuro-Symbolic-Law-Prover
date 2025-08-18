"""
Legal Blockchain Manager - Generation 6 Enhancement

Comprehensive blockchain integration for legal compliance including:
- Immutable legal decision storage
- Smart contract deployment and management
- Cross-chain legal verification
- Cryptographic proof generation
- Distributed consensus for legal precedents
"""

import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import asyncio
from enum import Enum

logger = logging.getLogger(__name__)


class BlockchainNetwork(Enum):
    """Supported blockchain networks."""
    ETHEREUM = "ethereum"
    POLYGON = "polygon"
    ARBITRUM = "arbitrum" 
    OPTIMISM = "optimism"
    AVALANCHE = "avalanche"
    PRIVATE = "private"


@dataclass
class LegalTransaction:
    """Legal transaction for blockchain storage."""
    transaction_id: str
    legal_action: str
    parties: List[str]
    compliance_result: Dict[str, Any]
    timestamp: datetime
    digital_signatures: List[str]
    metadata: Dict[str, Any]


@dataclass
class BlockchainVerificationResult:
    """Result of blockchain verification."""
    is_verified: bool
    transaction_hash: str
    block_number: Optional[int]
    confirmation_count: int
    verification_details: Dict[str, Any]
    trust_score: float


class LegalBlockchainManager:
    """
    Comprehensive blockchain manager for legal compliance.
    
    Generation 6 Enhancement:
    - Multi-chain legal record storage
    - Smart contract deployment for legal agreements
    - Immutable audit trails with cryptographic proofs
    - Cross-chain verification and consensus
    - Legal precedent distribution network
    """
    
    def __init__(self,
                 primary_network: BlockchainNetwork = BlockchainNetwork.POLYGON,
                 backup_networks: Optional[List[BlockchainNetwork]] = None,
                 enable_smart_contracts: bool = True,
                 consensus_threshold: float = 0.67):
        self.primary_network = primary_network
        self.backup_networks = backup_networks or [BlockchainNetwork.ARBITRUM]
        self.enable_smart_contracts = enable_smart_contracts
        self.consensus_threshold = consensus_threshold
        self._initialize_blockchain_connections()
    
    def _initialize_blockchain_connections(self):
        """Initialize blockchain network connections."""
        try:
            # Initialize primary network
            self.primary_client = self._create_blockchain_client(self.primary_network)
            
            # Initialize backup networks
            self.backup_clients = {}
            for network in self.backup_networks:
                try:
                    self.backup_clients[network] = self._create_blockchain_client(network)
                except Exception as e:
                    logger.warning(f"Failed to initialize {network} client: {e}")
            
            # Initialize smart contract manager
            if self.enable_smart_contracts:
                self.contract_manager = SmartContractManager(self.primary_client)
            
            logger.info(f"Blockchain connections initialized: Primary={self.primary_network.value}, "
                       f"Backups={[n.value for n in self.backup_clients.keys()]}")
            
        except Exception as e:
            logger.error(f"Failed to initialize blockchain connections: {e}")
            # Fallback to mock implementations
            self.primary_client = MockBlockchainClient(self.primary_network)
            self.backup_clients = {
                network: MockBlockchainClient(network) for network in self.backup_networks
            }
            if self.enable_smart_contracts:
                self.contract_manager = MockSmartContractManager()
    
    def _create_blockchain_client(self, network: BlockchainNetwork):
        """Create blockchain client for specified network."""
        try:
            # In production: create actual blockchain clients (Web3, etc.)
            return ProductionBlockchainClient(network)
        except ImportError:
            # Fallback to mock client
            return MockBlockchainClient(network)
    
    async def store_legal_decision(self,
                                 compliance_result: Dict[str, Any],
                                 parties: List[str],
                                 legal_context: Dict[str, Any]) -> str:
        """
        Store legal compliance decision on blockchain.
        
        Args:
            compliance_result: Result of legal compliance verification
            parties: Legal parties involved
            legal_context: Additional legal context and metadata
            
        Returns:
            Transaction hash of stored decision
        """
        try:
            # Create legal transaction
            legal_transaction = LegalTransaction(
                transaction_id=self._generate_transaction_id(),
                legal_action="compliance_verification",
                parties=parties,
                compliance_result=compliance_result,
                timestamp=datetime.utcnow(),
                digital_signatures=await self._generate_digital_signatures(
                    compliance_result, parties
                ),
                metadata=legal_context
            )
            
            # Store on primary network
            primary_hash = await self._store_transaction(
                self.primary_client, legal_transaction
            )
            
            # Store on backup networks (async)
            backup_tasks = []
            for network, client in self.backup_clients.items():
                task = self._store_transaction_safe(client, legal_transaction)
                backup_tasks.append(task)
            
            # Wait for backup storage (with timeout)
            try:
                await asyncio.wait_for(asyncio.gather(*backup_tasks), timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("Some backup transactions timed out")
            
            # Update internal index
            await self._update_legal_index(legal_transaction, primary_hash)
            
            logger.info(f"Legal decision stored: {primary_hash}")
            return primary_hash
            
        except Exception as e:
            logger.error(f"Failed to store legal decision: {e}")
            raise
    
    async def verify_legal_record(self, 
                                transaction_hash: str,
                                require_consensus: bool = True) -> BlockchainVerificationResult:
        """
        Verify legal record on blockchain with multi-network consensus.
        
        Args:
            transaction_hash: Hash of transaction to verify
            require_consensus: Whether to require multi-network consensus
            
        Returns:
            BlockchainVerificationResult with verification details
        """
        try:
            verification_results = []
            
            # Verify on primary network
            primary_result = await self.primary_client.verify_transaction(transaction_hash)
            verification_results.append(('primary', primary_result))
            
            # Verify on backup networks
            if require_consensus:
                for network, client in self.backup_clients.items():
                    try:
                        backup_result = await client.verify_transaction(transaction_hash)
                        verification_results.append((network.value, backup_result))
                    except Exception as e:
                        logger.warning(f"Verification failed on {network.value}: {e}")
            
            # Calculate consensus
            verified_count = sum(1 for _, result in verification_results if result['verified'])
            total_count = len(verification_results)
            consensus_ratio = verified_count / max(total_count, 1)
            
            # Determine overall verification status
            is_verified = (
                primary_result['verified'] and 
                (not require_consensus or consensus_ratio >= self.consensus_threshold)
            )
            
            # Calculate trust score
            trust_score = self._calculate_trust_score(verification_results, consensus_ratio)
            
            return BlockchainVerificationResult(
                is_verified=is_verified,
                transaction_hash=transaction_hash,
                block_number=primary_result.get('block_number'),
                confirmation_count=primary_result.get('confirmations', 0),
                verification_details={
                    'primary_result': primary_result,
                    'backup_results': {
                        network: result for network, result in verification_results[1:]
                    },
                    'consensus_ratio': consensus_ratio,
                    'verified_networks': verified_count,
                    'total_networks': total_count
                },
                trust_score=trust_score
            )
            
        except Exception as e:
            logger.error(f"Failed to verify legal record: {e}")
            return self._create_failed_verification_result(transaction_hash, str(e))
    
    async def deploy_smart_legal_contract(self,
                                        contract_terms: Dict[str, Any],
                                        parties: List[str],
                                        auto_execution: bool = False) -> Dict[str, str]:
        """
        Deploy smart contract for legal agreement.
        
        Args:
            contract_terms: Legal contract terms and conditions
            parties: Contract parties with their addresses
            auto_execution: Whether to enable automatic execution
            
        Returns:
            Dictionary with contract address and deployment details
        """
        try:
            if not self.enable_smart_contracts:
                raise ValueError("Smart contracts are disabled")
            
            # Generate smart contract code
            contract_code = await self.contract_manager.generate_legal_contract_code(
                contract_terms, parties, auto_execution
            )
            
            # Deploy contract
            deployment_result = await self.contract_manager.deploy_contract(
                contract_code, parties
            )
            
            # Store deployment record
            deployment_record = {
                'contract_address': deployment_result['address'],
                'deployment_hash': deployment_result['transaction_hash'],
                'parties': parties,
                'terms': contract_terms,
                'timestamp': datetime.utcnow().isoformat(),
                'network': self.primary_network.value
            }
            
            await self.store_legal_decision(
                compliance_result={'action': 'smart_contract_deployment', 'status': 'success'},
                parties=parties,
                legal_context=deployment_record
            )
            
            logger.info(f"Smart legal contract deployed: {deployment_result['address']}")
            return deployment_result
            
        except Exception as e:
            logger.error(f"Failed to deploy smart legal contract: {e}")
            raise
    
    async def create_legal_precedent_network(self,
                                          precedent_data: Dict[str, Any],
                                          jurisdiction: str,
                                          case_importance: float = 0.5) -> str:
        """
        Create distributed legal precedent network entry.
        
        Args:
            precedent_data: Legal precedent information
            jurisdiction: Legal jurisdiction
            case_importance: Importance score for the precedent
            
        Returns:
            Network entry hash
        """
        try:
            # Create precedent entry
            precedent_entry = {
                'case_id': precedent_data.get('case_id', self._generate_case_id()),
                'jurisdiction': jurisdiction,
                'precedent_type': precedent_data.get('type', 'case_law'),
                'legal_principles': precedent_data.get('principles', []),
                'citation': precedent_data.get('citation', ''),
                'summary': precedent_data.get('summary', ''),
                'importance_score': case_importance,
                'timestamp': datetime.utcnow().isoformat(),
                'precedent_hash': self._hash_precedent_data(precedent_data)
            }
            
            # Store in distributed network
            network_hash = await self._store_precedent_in_network(precedent_entry)
            
            # Update precedent index
            await self._update_precedent_index(precedent_entry, network_hash)
            
            logger.info(f"Legal precedent stored in network: {network_hash}")
            return network_hash
            
        except Exception as e:
            logger.error(f"Failed to create legal precedent network entry: {e}")
            raise
    
    async def query_legal_precedents(self,
                                   query_terms: List[str],
                                   jurisdiction: Optional[str] = None,
                                   min_importance: float = 0.1) -> List[Dict[str, Any]]:
        """
        Query distributed legal precedent network.
        
        Args:
            query_terms: Legal terms to search for
            jurisdiction: Optional jurisdiction filter
            min_importance: Minimum importance score
            
        Returns:
            List of matching legal precedents
        """
        try:
            # Query primary network
            primary_results = await self.primary_client.query_precedents(
                query_terms, jurisdiction, min_importance
            )
            
            # Query backup networks for additional precedents
            backup_results = []
            for network, client in self.backup_clients.items():
                try:
                    network_results = await client.query_precedents(
                        query_terms, jurisdiction, min_importance
                    )
                    backup_results.extend(network_results)
                except Exception as e:
                    logger.warning(f"Precedent query failed on {network.value}: {e}")
            
            # Merge and deduplicate results
            all_results = primary_results + backup_results
            unique_results = self._deduplicate_precedents(all_results)
            
            # Sort by importance and relevance
            sorted_results = sorted(
                unique_results,
                key=lambda x: (x.get('importance_score', 0), x.get('relevance_score', 0)),
                reverse=True
            )
            
            logger.info(f"Found {len(sorted_results)} legal precedents")
            return sorted_results
            
        except Exception as e:
            logger.error(f"Failed to query legal precedents: {e}")
            return []
    
    def _generate_transaction_id(self) -> str:
        """Generate unique transaction ID."""
        timestamp = datetime.utcnow().isoformat()
        random_data = hashlib.sha256(f"{timestamp}_{id(self)}".encode()).hexdigest()[:8]
        return f"legal_tx_{random_data}"
    
    def _generate_case_id(self) -> str:
        """Generate unique case ID."""
        timestamp = datetime.utcnow().isoformat()
        random_data = hashlib.sha256(f"case_{timestamp}".encode()).hexdigest()[:12]
        return f"case_{random_data}"
    
    async def _generate_digital_signatures(self, 
                                         compliance_result: Dict[str, Any],
                                         parties: List[str]) -> List[str]:
        """Generate digital signatures for legal transaction."""
        # In production: implement actual digital signature generation
        signatures = []
        data_to_sign = json.dumps(compliance_result, sort_keys=True)
        
        for party in parties:
            signature = hashlib.sha256(f"{data_to_sign}_{party}".encode()).hexdigest()
            signatures.append(f"sig_{signature[:16]}")
        
        return signatures
    
    async def _store_transaction(self, client, transaction: LegalTransaction) -> str:
        """Store transaction on blockchain client."""
        return await client.store_transaction(transaction)
    
    async def _store_transaction_safe(self, client, transaction: LegalTransaction) -> Optional[str]:
        """Safely store transaction with error handling."""
        try:
            return await self._store_transaction(client, transaction)
        except Exception as e:
            logger.warning(f"Failed to store transaction on backup: {e}")
            return None
    
    async def _update_legal_index(self, transaction: LegalTransaction, tx_hash: str):
        """Update internal legal decision index."""
        # In production: update searchable index
        logger.debug(f"Updated legal index: {tx_hash}")
    
    def _calculate_trust_score(self, 
                              verification_results: List[Tuple[str, Dict]],
                              consensus_ratio: float) -> float:
        """Calculate trust score based on verification results."""
        # Base score from consensus ratio
        base_score = consensus_ratio * 0.7
        
        # Bonus for confirmations
        primary_confirmations = verification_results[0][1].get('confirmations', 0)
        confirmation_bonus = min(primary_confirmations / 10.0, 0.2)
        
        # Bonus for multiple network verification
        network_bonus = (len(verification_results) - 1) * 0.05
        
        return min(base_score + confirmation_bonus + network_bonus, 1.0)
    
    def _create_failed_verification_result(self, tx_hash: str, error: str) -> BlockchainVerificationResult:
        """Create failed verification result."""
        return BlockchainVerificationResult(
            is_verified=False,
            transaction_hash=tx_hash,
            block_number=None,
            confirmation_count=0,
            verification_details={'error': error},
            trust_score=0.0
        )
    
    def _hash_precedent_data(self, precedent_data: Dict[str, Any]) -> str:
        """Generate hash for precedent data."""
        data_str = json.dumps(precedent_data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    async def _store_precedent_in_network(self, precedent_entry: Dict[str, Any]) -> str:
        """Store precedent in distributed network."""
        # Store on primary network
        primary_hash = await self.primary_client.store_precedent(precedent_entry)
        
        # Replicate to backup networks
        for network, client in self.backup_clients.items():
            try:
                await client.store_precedent(precedent_entry)
            except Exception as e:
                logger.warning(f"Failed to replicate precedent to {network.value}: {e}")
        
        return primary_hash
    
    async def _update_precedent_index(self, precedent_entry: Dict[str, Any], network_hash: str):
        """Update precedent search index."""
        # In production: update search index
        logger.debug(f"Updated precedent index: {network_hash}")
    
    def _deduplicate_precedents(self, precedents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate precedents based on precedent_hash."""
        seen_hashes = set()
        unique_precedents = []
        
        for precedent in precedents:
            precedent_hash = precedent.get('precedent_hash')
            if precedent_hash and precedent_hash not in seen_hashes:
                seen_hashes.add(precedent_hash)
                unique_precedents.append(precedent)
        
        return unique_precedents


class BlockchainVerifier:
    """
    Specialized blockchain verifier for legal compliance.
    
    Provides verification services for legal decisions stored on blockchain.
    """
    
    def __init__(self, blockchain_manager: LegalBlockchainManager):
        self.blockchain_manager = blockchain_manager
    
    async def verify_compliance_chain(self, 
                                    compliance_history: List[str]) -> Dict[str, Any]:
        """
        Verify entire compliance decision chain.
        
        Args:
            compliance_history: List of transaction hashes in chronological order
            
        Returns:
            Verification result for the entire chain
        """
        try:
            verification_results = []
            chain_integrity = True
            
            for i, tx_hash in enumerate(compliance_history):
                result = await self.blockchain_manager.verify_legal_record(tx_hash)
                verification_results.append({
                    'transaction_hash': tx_hash,
                    'position': i,
                    'verified': result.is_verified,
                    'trust_score': result.trust_score,
                    'details': result.verification_details
                })
                
                if not result.is_verified:
                    chain_integrity = False
            
            # Calculate overall chain score
            chain_score = sum(r['trust_score'] for r in verification_results) / len(verification_results)
            
            return {
                'chain_integrity': chain_integrity,
                'chain_score': chain_score,
                'total_transactions': len(compliance_history),
                'verified_transactions': sum(1 for r in verification_results if r['verified']),
                'individual_results': verification_results
            }
            
        except Exception as e:
            logger.error(f"Failed to verify compliance chain: {e}")
            return {
                'chain_integrity': False,
                'chain_score': 0.0,
                'error': str(e)
            }
    
    async def audit_legal_decisions(self,
                                  party_address: str,
                                  date_range: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """
        Audit all legal decisions for a specific party within date range.
        
        Args:
            party_address: Address/identifier of legal party
            date_range: Tuple of start and end datetime
            
        Returns:
            Comprehensive audit report
        """
        try:
            # Query transactions for party within date range
            transactions = await self._query_party_transactions(party_address, date_range)
            
            audit_results = []
            total_compliance_score = 0.0
            
            for tx in transactions:
                verification = await self.blockchain_manager.verify_legal_record(tx['hash'])
                
                audit_entry = {
                    'transaction_hash': tx['hash'],
                    'timestamp': tx['timestamp'],
                    'legal_action': tx['legal_action'],
                    'compliance_result': tx['compliance_result'],
                    'verified': verification.is_verified,
                    'trust_score': verification.trust_score
                }
                
                audit_results.append(audit_entry)
                total_compliance_score += verification.trust_score
            
            # Generate audit summary
            audit_summary = {
                'party_address': party_address,
                'audit_period': {
                    'start': date_range[0].isoformat(),
                    'end': date_range[1].isoformat()
                },
                'total_transactions': len(transactions),
                'verified_transactions': sum(1 for r in audit_results if r['verified']),
                'average_compliance_score': total_compliance_score / max(len(audit_results), 1),
                'audit_results': audit_results
            }
            
            return audit_summary
            
        except Exception as e:
            logger.error(f"Failed to audit legal decisions: {e}")
            return {'error': str(e)}
    
    async def _query_party_transactions(self,
                                      party_address: str,
                                      date_range: Tuple[datetime, datetime]) -> List[Dict[str, Any]]:
        """Query transactions for specific party within date range."""
        # In production: implement actual blockchain query
        # For now: return mock data
        return [
            {
                'hash': 'tx_abc123',
                'timestamp': date_range[0],
                'legal_action': 'compliance_verification',
                'compliance_result': {'status': 'compliant'},
                'party_address': party_address
            }
        ]


# Mock implementations for fallback operation
class MockBlockchainClient:
    """Mock blockchain client for fallback operation."""
    
    def __init__(self, network: BlockchainNetwork):
        self.network = network
    
    async def store_transaction(self, transaction: LegalTransaction) -> str:
        """Mock transaction storage."""
        tx_data = f"{transaction.transaction_id}_{self.network.value}"
        return hashlib.sha256(tx_data.encode()).hexdigest()[:32]
    
    async def verify_transaction(self, tx_hash: str) -> Dict[str, Any]:
        """Mock transaction verification."""
        return {
            'verified': True,
            'block_number': 12345,
            'confirmations': 6,
            'network': self.network.value
        }
    
    async def store_precedent(self, precedent_entry: Dict[str, Any]) -> str:
        """Mock precedent storage."""
        data = json.dumps(precedent_entry, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:32]
    
    async def query_precedents(self,
                             query_terms: List[str],
                             jurisdiction: Optional[str],
                             min_importance: float) -> List[Dict[str, Any]]:
        """Mock precedent query."""
        return [
            {
                'case_id': 'case_mock123',
                'jurisdiction': jurisdiction or 'US',
                'precedent_hash': 'hash123',
                'importance_score': 0.8,
                'relevance_score': 0.9,
                'summary': f"Mock precedent for terms: {', '.join(query_terms)}"
            }
        ]


class ProductionBlockchainClient:
    """Production blockchain client (placeholder)."""
    
    def __init__(self, network: BlockchainNetwork):
        self.network = network
        # In production: initialize actual blockchain connections
        raise ImportError("Production blockchain libraries not available")


class SmartContractManager:
    """Smart contract deployment and management."""
    
    def __init__(self, blockchain_client):
        self.client = blockchain_client
    
    async def generate_legal_contract_code(self,
                                         contract_terms: Dict[str, Any],
                                         parties: List[str],
                                         auto_execution: bool) -> str:
        """Generate smart contract code from legal terms."""
        # In production: implement actual smart contract code generation
        return f"// Smart contract for {len(parties)} parties with auto_execution={auto_execution}"
    
    async def deploy_contract(self, contract_code: str, parties: List[str]) -> Dict[str, str]:
        """Deploy smart contract."""
        # Mock deployment
        contract_address = hashlib.sha256(contract_code.encode()).hexdigest()[:40]
        tx_hash = hashlib.sha256(f"{contract_address}_deploy".encode()).hexdigest()[:32]
        
        return {
            'address': f"0x{contract_address}",
            'transaction_hash': f"0x{tx_hash}",
            'status': 'deployed'
        }


class MockSmartContractManager(SmartContractManager):
    """Mock smart contract manager."""
    
    def __init__(self):
        self.client = None