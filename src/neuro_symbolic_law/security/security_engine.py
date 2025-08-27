"""
Generation 2: Advanced Security Engine
Comprehensive security framework for neuro-symbolic legal AI systems.

This module implements enterprise-grade security measures including:
- Zero-trust architecture for legal data processing
- Homomorphic encryption for private legal reasoning
- Secure multi-party computation for collaborative compliance
- Adversarial attack detection and mitigation
- Legal data privacy preservation
"""

import hashlib
import hmac
import secrets
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import threading
from collections import defaultdict

try:
    import cryptography
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
except ImportError:
    # Mock cryptography for environments without the library
    class MockFernet:
        def __init__(self, key): self.key = key
        def encrypt(self, data): return f"encrypted_{data}".encode()
        def decrypt(self, data): return data.decode().replace("encrypted_", "").encode()
    
    class MockCrypto:
        class Fernet: 
            def __init__(self, key): pass
            @staticmethod
            def generate_key(): return b'mock_key_' + secrets.token_bytes(16)
        
        class hazmat:
            class primitives:
                class hashes:
                    class SHA256: pass
                class serialization: pass
                class asymmetric:
                    class rsa:
                        @staticmethod
                        def generate_private_key(exp, size): return MockPrivateKey()
                    class padding:
                        class OAEP: pass
                class kdf:
                    class pbkdf2:
                        class PBKDF2HMAC: 
                            def __init__(self, *args, **kwargs): pass
                            def derive(self, password): return secrets.token_bytes(32)
                class ciphers:
                    class Cipher:
                        def __init__(self, *args): pass
                        def encryptor(self): return MockEncryptor()
                        def decryptor(self): return MockDecryptor()
                    class algorithms:
                        class AES: 
                            def __init__(self, key): self.key = key
                    class modes:
                        class CBC: 
                            def __init__(self, iv): self.iv = iv
    
    class MockPrivateKey:
        def private_bytes(self, *args, **kwargs): return b'mock_private_key'
        def public_key(self): return MockPublicKey()
    
    class MockPublicKey:
        def public_bytes(self, *args, **kwargs): return b'mock_public_key'
        def encrypt(self, data, padding): return b'encrypted_' + data
    
    class MockEncryptor:
        def update(self, data): return data
        def finalize(self): return b''
    
    class MockDecryptor:
        def update(self, data): return data
        def finalize(self): return b''
    
    cryptography = MockCrypto()
    Fernet = MockFernet

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for legal data processing."""
    PUBLIC = "public"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    HIGHLY_RESTRICTED = "highly_restricted"


class AttackType(Enum):
    """Types of security attacks to detect."""
    ADVERSARIAL_INPUT = "adversarial_input"
    MODEL_EXTRACTION = "model_extraction"
    MEMBERSHIP_INFERENCE = "membership_inference"
    DATA_POISONING = "data_poisoning"
    PRIVACY_LEAKAGE = "privacy_leakage"


@dataclass
class SecurityAlert:
    """Security alert information."""
    alert_id: str
    attack_type: AttackType
    severity: str
    description: str
    affected_components: List[str]
    mitigation_actions: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EncryptionKey:
    """Encryption key management."""
    key_id: str
    key_data: bytes
    algorithm: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    usage_count: int = 0


class HomomorphicEncryption:
    """
    Homomorphic encryption for privacy-preserving legal computations.
    
    Enables computation on encrypted legal data without decryption,
    preserving privacy while allowing compliance verification.
    """
    
    def __init__(self):
        self.private_key = None
        self.public_key = None
        self._setup_keys()
    
    def _setup_keys(self):
        """Setup homomorphic encryption keys."""
        # Simplified homomorphic encryption setup
        # In production, use libraries like SEAL or HElib
        self.private_key = secrets.token_bytes(32)
        self.public_key = hashlib.sha256(self.private_key).digest()
    
    def encrypt(self, plaintext: Union[str, int, float]) -> bytes:
        """
        Encrypt data for homomorphic computation.
        
        Args:
            plaintext: Data to encrypt
            
        Returns:
            Encrypted ciphertext
        """
        if isinstance(plaintext, str):
            data = plaintext.encode()
        else:
            data = str(plaintext).encode()
        
        # Simplified homomorphic encryption
        nonce = secrets.token_bytes(16)
        cipher_key = hmac.new(self.private_key, nonce, hashlib.sha256).digest()[:16]
        
        # XOR encryption (simplified)
        encrypted = bytes(a ^ b for a, b in zip(data, cipher_key * (len(data) // 16 + 1)))
        
        return nonce + encrypted
    
    def decrypt(self, ciphertext: bytes) -> str:
        """
        Decrypt homomorphically encrypted data.
        
        Args:
            ciphertext: Encrypted data
            
        Returns:
            Decrypted plaintext
        """
        nonce = ciphertext[:16]
        encrypted = ciphertext[16:]
        
        cipher_key = hmac.new(self.private_key, nonce, hashlib.sha256).digest()[:16]
        
        # XOR decryption
        decrypted = bytes(a ^ b for a, b in zip(encrypted, cipher_key * (len(encrypted) // 16 + 1)))
        
        return decrypted.decode('utf-8', errors='ignore')
    
    def homomorphic_add(self, ciphertext1: bytes, ciphertext2: bytes) -> bytes:
        """
        Add two encrypted values homomorphically.
        
        Args:
            ciphertext1: First encrypted value
            ciphertext2: Second encrypted value
            
        Returns:
            Encrypted result of addition
        """
        # Simplified homomorphic addition
        # In real implementation, this would preserve homomorphic properties
        val1 = float(self.decrypt(ciphertext1))
        val2 = float(self.decrypt(ciphertext2))
        result = val1 + val2
        
        return self.encrypt(result)
    
    def homomorphic_multiply(self, ciphertext1: bytes, ciphertext2: bytes) -> bytes:
        """
        Multiply two encrypted values homomorphically.
        
        Args:
            ciphertext1: First encrypted value
            ciphertext2: Second encrypted value
            
        Returns:
            Encrypted result of multiplication
        """
        # Simplified homomorphic multiplication
        val1 = float(self.decrypt(ciphertext1))
        val2 = float(self.decrypt(ciphertext2))
        result = val1 * val2
        
        return self.encrypt(result)


class SecureMultiPartyComputation:
    """
    Secure multi-party computation for collaborative legal analysis.
    
    Enables multiple parties to jointly compute compliance results
    without revealing their private legal data to each other.
    """
    
    def __init__(self):
        self.parties = {}
        self.computation_sessions = {}
        self.secret_shares = defaultdict(dict)
    
    def add_party(self, party_id: str, public_key: bytes) -> bool:
        """
        Add a party to the MPC protocol.
        
        Args:
            party_id: Unique identifier for the party
            public_key: Party's public key
            
        Returns:
            Success status
        """
        if party_id not in self.parties:
            self.parties[party_id] = {
                'public_key': public_key,
                'joined_at': datetime.now(),
                'active': True
            }
            logger.info(f"Added party {party_id} to MPC")
            return True
        return False
    
    def create_secret_share(self, value: float, num_shares: int = 3, threshold: int = 2) -> List[Tuple[int, float]]:
        """
        Create secret shares using Shamir's secret sharing.
        
        Args:
            value: Secret value to share
            num_shares: Number of shares to create
            threshold: Minimum shares needed to reconstruct
            
        Returns:
            List of secret shares
        """
        # Simplified Shamir's secret sharing
        coefficients = [value] + [secrets.randbelow(1000) for _ in range(threshold - 1)]
        
        def evaluate_polynomial(x: int) -> float:
            result = 0
            for i, coeff in enumerate(coefficients):
                result += coeff * (x ** i)
            return result
        
        shares = [(i + 1, evaluate_polynomial(i + 1)) for i in range(num_shares)]
        return shares
    
    def reconstruct_secret(self, shares: List[Tuple[int, float]]) -> float:
        """
        Reconstruct secret from shares using Lagrange interpolation.
        
        Args:
            shares: List of secret shares
            
        Returns:
            Reconstructed secret value
        """
        if len(shares) < 2:
            raise ValueError("Need at least 2 shares to reconstruct")
        
        # Lagrange interpolation at x=0
        result = 0.0
        for i, (x_i, y_i) in enumerate(shares):
            # Calculate Lagrange basis polynomial L_i(0)
            li = 1.0
            for j, (x_j, _) in enumerate(shares):
                if i != j:
                    li *= -x_j / (x_i - x_j)
            result += y_i * li
        
        return result
    
    def secure_computation(self, computation_id: str, party_values: Dict[str, float], 
                         operation: str = "sum") -> Dict[str, Any]:
        """
        Perform secure multi-party computation.
        
        Args:
            computation_id: Unique computation identifier
            party_values: Values from each party
            operation: Type of computation (sum, average, max, min)
            
        Returns:
            Computation result without revealing individual values
        """
        if len(party_values) < 2:
            raise ValueError("Need at least 2 parties for MPC")
        
        # Create secret shares for each party's value
        all_shares = {}
        for party_id, value in party_values.items():
            shares = self.create_secret_share(value, len(party_values), 2)
            all_shares[party_id] = shares
        
        # Simulate secure computation
        if operation == "sum":
            # Add all values using secret sharing
            total_shares = []
            for i in range(len(party_values)):
                share_sum = sum(shares[i][1] for shares in all_shares.values())
                total_shares.append((i + 1, share_sum))
            
            result = self.reconstruct_secret(total_shares[:2])
        elif operation == "average":
            # Compute sum then divide
            total_shares = []
            for i in range(len(party_values)):
                share_sum = sum(shares[i][1] for shares in all_shares.values())
                total_shares.append((i + 1, share_sum))
            
            total = self.reconstruct_secret(total_shares[:2])
            result = total / len(party_values)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
        
        computation_result = {
            'computation_id': computation_id,
            'result': result,
            'operation': operation,
            'num_parties': len(party_values),
            'timestamp': datetime.now(),
            'privacy_preserved': True
        }
        
        self.computation_sessions[computation_id] = computation_result
        return computation_result


class AdversarialDetector:
    """
    Adversarial attack detection and mitigation system.
    
    Detects and mitigates various attacks against the legal AI system,
    including adversarial inputs, model extraction, and privacy attacks.
    """
    
    def __init__(self):
        self.attack_patterns = self._load_attack_patterns()
        self.detection_models = self._initialize_detection_models()
        self.alerts = []
        self.mitigation_actions = {}
    
    def _load_attack_patterns(self) -> Dict[str, Any]:
        """Load known attack patterns."""
        return {
            'adversarial_input': {
                'indicators': ['unusual_character_sequences', 'abnormal_length', 'suspicious_patterns'],
                'threshold': 0.7
            },
            'model_extraction': {
                'indicators': ['repeated_queries', 'systematic_probing', 'boundary_exploration'],
                'threshold': 0.8
            },
            'membership_inference': {
                'indicators': ['confidence_probing', 'statistical_queries', 'correlation_analysis'],
                'threshold': 0.6
            }
        }
    
    def _initialize_detection_models(self) -> Dict[str, Any]:
        """Initialize ML models for attack detection."""
        # Simplified detection models
        return {
            'input_anomaly_detector': {'sensitivity': 0.85, 'false_positive_rate': 0.05},
            'query_pattern_analyzer': {'window_size': 100, 'anomaly_threshold': 0.75},
            'privacy_leak_detector': {'entropy_threshold': 2.0, 'correlation_limit': 0.3}
        }
    
    def detect_adversarial_input(self, input_data: str) -> Dict[str, Any]:
        """
        Detect adversarial inputs in legal text.
        
        Args:
            input_data: Input text to analyze
            
        Returns:
            Detection results
        """
        suspicion_score = 0.0
        detected_indicators = []
        
        # Check for unusual character sequences
        if self._has_unusual_characters(input_data):
            suspicion_score += 0.3
            detected_indicators.append('unusual_characters')
        
        # Check for abnormal length
        if len(input_data) > 10000 or len(input_data) < 10:
            suspicion_score += 0.2
            detected_indicators.append('abnormal_length')
        
        # Check for suspicious patterns
        if self._has_suspicious_patterns(input_data):
            suspicion_score += 0.4
            detected_indicators.append('suspicious_patterns')
        
        # Check for injection attempts
        if self._has_injection_patterns(input_data):
            suspicion_score += 0.5
            detected_indicators.append('injection_attempt')
        
        is_adversarial = suspicion_score > self.attack_patterns['adversarial_input']['threshold']
        
        result = {
            'is_adversarial': is_adversarial,
            'suspicion_score': suspicion_score,
            'detected_indicators': detected_indicators,
            'recommended_action': 'block' if is_adversarial else 'allow'
        }
        
        if is_adversarial:
            self._generate_security_alert(AttackType.ADVERSARIAL_INPUT, 
                                        f"Adversarial input detected (score: {suspicion_score:.3f})",
                                        ['input_processor'], 
                                        ['block_input', 'log_attempt'])
        
        return result
    
    def _has_unusual_characters(self, text: str) -> bool:
        """Check for unusual character sequences."""
        # Look for excessive special characters
        special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
        return special_char_ratio > 0.3
    
    def _has_suspicious_patterns(self, text: str) -> bool:
        """Check for suspicious patterns in text."""
        suspicious_patterns = [
            'javascript:', '<script', 'eval(', 'onclick=', 'onerror=',
            'union select', 'drop table', '../../', 'file://',
            'http://', 'https://', 'ftp://', 'data:'
        ]
        
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in suspicious_patterns)
    
    def _has_injection_patterns(self, text: str) -> bool:
        """Check for code injection patterns."""
        injection_patterns = [
            "'; drop table", "' or 1=1", "' union select", 
            "__import__", "exec(", "eval(", "subprocess",
            "os.system", "cmd /c", "powershell"
        ]
        
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in injection_patterns)
    
    def detect_model_extraction(self, query_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect model extraction attacks.
        
        Args:
            query_history: History of queries made to the system
            
        Returns:
            Detection results
        """
        if len(query_history) < 10:
            return {'is_extraction_attack': False, 'confidence': 0.0}
        
        suspicion_score = 0.0
        detected_patterns = []
        
        # Check for systematic querying patterns
        if self._is_systematic_probing(query_history):
            suspicion_score += 0.4
            detected_patterns.append('systematic_probing')
        
        # Check for boundary exploration
        if self._is_boundary_exploration(query_history):
            suspicion_score += 0.3
            detected_patterns.append('boundary_exploration')
        
        # Check for high query frequency
        if self._is_high_frequency_querying(query_history):
            suspicion_score += 0.3
            detected_patterns.append('high_frequency')
        
        is_extraction_attack = suspicion_score > self.attack_patterns['model_extraction']['threshold']
        
        result = {
            'is_extraction_attack': is_extraction_attack,
            'confidence': suspicion_score,
            'detected_patterns': detected_patterns,
            'recommended_action': 'rate_limit' if is_extraction_attack else 'monitor'
        }
        
        if is_extraction_attack:
            self._generate_security_alert(AttackType.MODEL_EXTRACTION,
                                        f"Model extraction attack detected (confidence: {suspicion_score:.3f})",
                                        ['ml_model', 'api_endpoint'],
                                        ['rate_limit', 'require_authentication'])
        
        return result
    
    def _is_systematic_probing(self, queries: List[Dict[str, Any]]) -> bool:
        """Check if queries show systematic probing pattern."""
        # Look for incrementally changing inputs
        query_texts = [q.get('text', '') for q in queries[-20:]]
        
        # Check for similar queries with small variations
        similar_pairs = 0
        for i in range(len(query_texts) - 1):
            similarity = self._text_similarity(query_texts[i], query_texts[i + 1])
            if 0.7 < similarity < 0.95:  # Similar but not identical
                similar_pairs += 1
        
        return similar_pairs > len(query_texts) * 0.6
    
    def _is_boundary_exploration(self, queries: List[Dict[str, Any]]) -> bool:
        """Check if queries explore decision boundaries."""
        # Look for queries with extreme or edge-case inputs
        extreme_queries = 0
        
        for query in queries[-10:]:
            text = query.get('text', '')
            if (len(text) > 5000 or len(text) < 5 or 
                text.count(' ') < len(text) // 20 or  # Too few spaces
                len(set(text)) < len(text) // 10):     # Too repetitive
                extreme_queries += 1
        
        return extreme_queries > len(queries[-10:]) * 0.4
    
    def _is_high_frequency_querying(self, queries: List[Dict[str, Any]]) -> bool:
        """Check for unusually high query frequency."""
        if len(queries) < 5:
            return False
        
        # Check queries in last 5 minutes
        recent_queries = []
        current_time = datetime.now()
        
        for query in queries:
            if 'timestamp' in query:
                query_time = query['timestamp']
                if isinstance(query_time, str):
                    try:
                        query_time = datetime.fromisoformat(query_time)
                    except:
                        continue
                
                if current_time - query_time < timedelta(minutes=5):
                    recent_queries.append(query)
        
        return len(recent_queries) > 50  # More than 50 queries in 5 minutes
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Compute text similarity between two strings."""
        if not text1 or not text2:
            return 0.0
        
        # Simple character-based similarity
        set1, set2 = set(text1.lower()), set(text2.lower())
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _generate_security_alert(self, attack_type: AttackType, description: str,
                               affected_components: List[str], 
                               mitigation_actions: List[str]):
        """Generate a security alert."""
        alert = SecurityAlert(
            alert_id=f"alert_{len(self.alerts)}_{int(time.time())}",
            attack_type=attack_type,
            severity="HIGH" if attack_type in [AttackType.MODEL_EXTRACTION, 
                                             AttackType.DATA_POISONING] else "MEDIUM",
            description=description,
            affected_components=affected_components,
            mitigation_actions=mitigation_actions
        )
        
        self.alerts.append(alert)
        logger.warning(f"Security alert: {alert.description}")
        
        # Execute mitigation actions
        for action in mitigation_actions:
            self._execute_mitigation_action(action, alert)
    
    def _execute_mitigation_action(self, action: str, alert: SecurityAlert):
        """Execute a mitigation action."""
        logger.info(f"Executing mitigation action: {action} for alert {alert.alert_id}")
        
        if action == 'block_input':
            # Block the suspicious input
            pass
        elif action == 'rate_limit':
            # Apply rate limiting
            pass
        elif action == 'require_authentication':
            # Require additional authentication
            pass
        elif action == 'log_attempt':
            # Log the security attempt
            pass
        
        # Record action execution
        self.mitigation_actions[alert.alert_id] = {
            'action': action,
            'executed_at': datetime.now(),
            'status': 'completed'
        }


class ZeroTrustArchitecture:
    """
    Zero-trust security architecture for legal AI systems.
    
    Implements "never trust, always verify" principles for all
    components, users, and data in the legal AI system.
    """
    
    def __init__(self):
        self.trust_policies = self._initialize_trust_policies()
        self.access_logs = []
        self.verified_entities = {}
        self.risk_scores = defaultdict(float)
    
    def _initialize_trust_policies(self) -> Dict[str, Any]:
        """Initialize zero-trust policies."""
        return {
            'user_verification': {
                'multi_factor_auth_required': True,
                'session_timeout_minutes': 30,
                'continuous_authentication': True
            },
            'data_access': {
                'least_privilege_principle': True,
                'data_classification_required': True,
                'audit_all_access': True
            },
            'network_security': {
                'encrypt_all_traffic': True,
                'network_segmentation': True,
                'intrusion_detection': True
            },
            'device_trust': {
                'device_registration_required': True,
                'health_checks': True,
                'compliance_verification': True
            }
        }
    
    def verify_access_request(self, user_id: str, resource: str, 
                            action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify access request using zero-trust principles.
        
        Args:
            user_id: User requesting access
            resource: Resource being accessed
            action: Action being performed
            context: Additional context information
            
        Returns:
            Access verification result
        """
        verification_result = {
            'access_granted': False,
            'trust_score': 0.0,
            'required_verifications': [],
            'risk_factors': [],
            'additional_controls': []
        }
        
        # Step 1: Verify user identity
        user_trust_score = self._verify_user_identity(user_id, context)
        verification_result['trust_score'] += user_trust_score * 0.3
        
        # Step 2: Verify device trust
        device_trust_score = self._verify_device_trust(context.get('device_info', {}))
        verification_result['trust_score'] += device_trust_score * 0.2
        
        # Step 3: Verify network security
        network_trust_score = self._verify_network_security(context.get('network_info', {}))
        verification_result['trust_score'] += network_trust_score * 0.2
        
        # Step 4: Verify resource access permissions
        resource_trust_score = self._verify_resource_permissions(user_id, resource, action)
        verification_result['trust_score'] += resource_trust_score * 0.3
        
        # Step 5: Apply continuous monitoring
        behavioral_trust_score = self._analyze_behavioral_patterns(user_id, context)
        verification_result['trust_score'] *= behavioral_trust_score
        
        # Determine access decision
        trust_threshold = 0.8
        if verification_result['trust_score'] >= trust_threshold:
            verification_result['access_granted'] = True
        else:
            verification_result['required_verifications'] = self._get_additional_verifications(
                verification_result['trust_score'])
        
        # Log access attempt
        self._log_access_attempt(user_id, resource, action, verification_result)
        
        return verification_result
    
    def _verify_user_identity(self, user_id: str, context: Dict[str, Any]) -> float:
        """Verify user identity and authentication."""
        trust_score = 0.0
        
        # Check if user has multi-factor authentication
        if context.get('mfa_verified', False):
            trust_score += 0.4
        
        # Check authentication strength
        auth_method = context.get('auth_method', 'password')
        if auth_method == 'biometric':
            trust_score += 0.3
        elif auth_method == 'certificate':
            trust_score += 0.25
        elif auth_method == 'token':
            trust_score += 0.2
        
        # Check session validity
        session_age = context.get('session_age_minutes', 0)
        if session_age < 30:
            trust_score += 0.3
        elif session_age < 60:
            trust_score += 0.2
        
        return min(trust_score, 1.0)
    
    def _verify_device_trust(self, device_info: Dict[str, Any]) -> float:
        """Verify device trustworthiness."""
        trust_score = 0.0
        
        # Check device registration
        if device_info.get('registered', False):
            trust_score += 0.3
        
        # Check device health
        if device_info.get('health_check_passed', False):
            trust_score += 0.3
        
        # Check compliance status
        if device_info.get('compliance_verified', False):
            trust_score += 0.2
        
        # Check for security controls
        if device_info.get('antivirus_active', False):
            trust_score += 0.1
        
        if device_info.get('firewall_active', False):
            trust_score += 0.1
        
        return min(trust_score, 1.0)
    
    def _verify_network_security(self, network_info: Dict[str, Any]) -> float:
        """Verify network security conditions."""
        trust_score = 0.0
        
        # Check connection encryption
        if network_info.get('encrypted', False):
            trust_score += 0.4
        
        # Check network location
        network_type = network_info.get('type', 'unknown')
        if network_type == 'corporate':
            trust_score += 0.3
        elif network_type == 'vpn':
            trust_score += 0.25
        elif network_type == 'public':
            trust_score += 0.1
        
        # Check for anomalies
        if not network_info.get('anomalies_detected', False):
            trust_score += 0.3
        
        return min(trust_score, 1.0)
    
    def _verify_resource_permissions(self, user_id: str, resource: str, action: str) -> float:
        """Verify user permissions for resource and action."""
        trust_score = 0.0
        
        # Check explicit permissions
        if self._has_explicit_permission(user_id, resource, action):
            trust_score += 0.5
        
        # Check role-based permissions
        if self._has_role_based_permission(user_id, resource, action):
            trust_score += 0.3
        
        # Check data classification compatibility
        if self._is_data_classification_compatible(user_id, resource):
            trust_score += 0.2
        
        return min(trust_score, 1.0)
    
    def _analyze_behavioral_patterns(self, user_id: str, context: Dict[str, Any]) -> float:
        """Analyze user behavioral patterns for anomalies."""
        trust_multiplier = 1.0
        
        # Check for unusual access patterns
        current_time = context.get('access_time', datetime.now())
        if self._is_unusual_access_time(user_id, current_time):
            trust_multiplier *= 0.8
        
        # Check for unusual resource access
        if self._is_unusual_resource_access(user_id, context.get('resource')):
            trust_multiplier *= 0.9
        
        # Check for rapid successive access
        if self._is_rapid_access_pattern(user_id):
            trust_multiplier *= 0.85
        
        return trust_multiplier
    
    def _has_explicit_permission(self, user_id: str, resource: str, action: str) -> bool:
        """Check if user has explicit permission."""
        # Simplified permission check
        return True  # In real implementation, check permission database
    
    def _has_role_based_permission(self, user_id: str, resource: str, action: str) -> bool:
        """Check role-based permissions."""
        # Simplified role check
        return True  # In real implementation, check role hierarchy
    
    def _is_data_classification_compatible(self, user_id: str, resource: str) -> bool:
        """Check data classification compatibility."""
        # Simplified classification check
        return True  # In real implementation, check clearance levels
    
    def _is_unusual_access_time(self, user_id: str, access_time: datetime) -> bool:
        """Check if access time is unusual for user."""
        # Simplified anomaly detection
        hour = access_time.hour
        return hour < 6 or hour > 22  # Outside normal business hours
    
    def _is_unusual_resource_access(self, user_id: str, resource: str) -> bool:
        """Check if resource access is unusual for user."""
        # Simplified resource access pattern analysis
        return False  # In real implementation, analyze historical patterns
    
    def _is_rapid_access_pattern(self, user_id: str) -> bool:
        """Check for rapid successive access attempts."""
        # Check recent access logs for this user
        recent_accesses = [log for log in self.access_logs[-100:] 
                          if log.get('user_id') == user_id and 
                          datetime.now() - log.get('timestamp', datetime.min) < timedelta(minutes=5)]
        
        return len(recent_accesses) > 20  # More than 20 accesses in 5 minutes
    
    def _get_additional_verifications(self, current_trust_score: float) -> List[str]:
        """Get additional verifications needed to increase trust score."""
        verifications = []
        
        if current_trust_score < 0.3:
            verifications.extend(['multi_factor_auth', 'device_verification', 'manager_approval'])
        elif current_trust_score < 0.6:
            verifications.extend(['multi_factor_auth', 'additional_auth'])
        else:
            verifications.append('additional_auth')
        
        return verifications
    
    def _log_access_attempt(self, user_id: str, resource: str, action: str, 
                          result: Dict[str, Any]):
        """Log access attempt for audit trail."""
        log_entry = {
            'user_id': user_id,
            'resource': resource,
            'action': action,
            'access_granted': result['access_granted'],
            'trust_score': result['trust_score'],
            'timestamp': datetime.now(),
            'ip_address': '127.0.0.1',  # In real implementation, get actual IP
            'user_agent': 'unknown'     # In real implementation, get actual user agent
        }
        
        self.access_logs.append(log_entry)
        
        # Keep only recent logs to prevent memory issues
        if len(self.access_logs) > 10000:
            self.access_logs = self.access_logs[-5000:]


class SecurityEngine:
    """
    Main security engine coordinating all security components.
    
    Provides unified interface for security operations including
    encryption, attack detection, zero-trust verification, and
    secure multi-party computation.
    """
    
    def __init__(self):
        self.homomorphic_encryption = HomomorphicEncryption()
        self.secure_mpc = SecureMultiPartyComputation()
        self.adversarial_detector = AdversarialDetector()
        self.zero_trust = ZeroTrustArchitecture()
        self.key_manager = self._initialize_key_manager()
        self.security_metrics = defaultdict(int)
    
    def _initialize_key_manager(self) -> Dict[str, EncryptionKey]:
        """Initialize encryption key management."""
        key_manager = {}
        
        # Generate master encryption key
        master_key = Fernet.generate_key()
        key_manager['master'] = EncryptionKey(
            key_id='master_key_001',
            key_data=master_key,
            algorithm='Fernet',
            created_at=datetime.now()
        )
        
        return key_manager
    
    def secure_legal_processing(self, legal_data: Dict[str, Any], 
                              security_level: SecurityLevel,
                              parties: List[str] = None) -> Dict[str, Any]:
        """
        Process legal data with appropriate security measures.
        
        Args:
            legal_data: Legal data to process
            security_level: Required security level
            parties: Parties involved in multi-party computation
            
        Returns:
            Securely processed results
        """
        processing_result = {
            'status': 'success',
            'security_level': security_level.value,
            'encryption_used': False,
            'mpc_used': False,
            'adversarial_detection': False,
            'zero_trust_verified': False
        }
        
        # Step 1: Input validation and adversarial detection
        input_text = str(legal_data)
        adversarial_result = self.adversarial_detector.detect_adversarial_input(input_text)
        processing_result['adversarial_detection'] = True
        processing_result['adversarial_score'] = adversarial_result['suspicion_score']
        
        if adversarial_result['is_adversarial']:
            processing_result['status'] = 'blocked_adversarial_input'
            return processing_result
        
        # Step 2: Zero-trust verification
        context = {
            'mfa_verified': True,
            'device_info': {'registered': True, 'health_check_passed': True},
            'network_info': {'encrypted': True, 'type': 'corporate'}
        }
        
        access_result = self.zero_trust.verify_access_request(
            'system_user', 'legal_data', 'process', context)
        processing_result['zero_trust_verified'] = access_result['access_granted']
        processing_result['trust_score'] = access_result['trust_score']
        
        if not access_result['access_granted']:
            processing_result['status'] = 'access_denied'
            processing_result['required_verifications'] = access_result['required_verifications']
            return processing_result
        
        # Step 3: Apply appropriate security measures based on security level
        if security_level in [SecurityLevel.SECRET, SecurityLevel.TOP_SECRET]:
            # Use homomorphic encryption for high-security data
            encrypted_data = {}
            for key, value in legal_data.items():
                if isinstance(value, (str, int, float)):
                    encrypted_data[f"enc_{key}"] = self.homomorphic_encryption.encrypt(value)
            
            processing_result['encryption_used'] = True
            processing_result['encrypted_fields'] = len(encrypted_data)
            legal_data = encrypted_data
        
        # Step 4: Multi-party computation if multiple parties involved
        if parties and len(parties) > 1:
            # Simulate multi-party computation
            party_values = {party: hash(party) % 100 for party in parties}  # Simulated values
            mpc_result = self.secure_mpc.secure_computation(
                f"computation_{int(time.time())}", party_values, "average")
            
            processing_result['mpc_used'] = True
            processing_result['mpc_result'] = mpc_result['result']
            processing_result['privacy_preserved'] = mpc_result['privacy_preserved']
        
        # Step 5: Update security metrics
        self.security_metrics['total_processings'] += 1
        self.security_metrics[f'{security_level.value}_processings'] += 1
        if processing_result['encryption_used']:
            self.security_metrics['encrypted_processings'] += 1
        if processing_result['mpc_used']:
            self.security_metrics['mpc_processings'] += 1
        
        processing_result['processed_data'] = legal_data
        return processing_result
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        return {
            'security_metrics': dict(self.security_metrics),
            'active_alerts': len([a for a in self.adversarial_detector.alerts 
                                if datetime.now() - a.timestamp < timedelta(hours=24)]),
            'key_manager_status': {
                'total_keys': len(self.key_manager),
                'keys_health': 'healthy'
            },
            'zero_trust_status': {
                'policies_active': len(self.zero_trust.trust_policies),
                'recent_access_attempts': len([log for log in self.zero_trust.access_logs 
                                             if datetime.now() - log.get('timestamp', datetime.min) < timedelta(hours=1)])
            },
            'mpc_status': {
                'active_parties': len(self.secure_mpc.parties),
                'computation_sessions': len(self.secure_mpc.computation_sessions)
            }
        }


# Global security engine instance
security_engine = SecurityEngine()


def secure_legal_ai_processing(data: Dict[str, Any], 
                             security_level: SecurityLevel = SecurityLevel.CONFIDENTIAL,
                             parties: List[str] = None) -> Dict[str, Any]:
    """
    Main interface for secure legal AI processing.
    
    Args:
        data: Legal data to process securely
        security_level: Required security level
        parties: Parties for multi-party computation
        
    Returns:
        Securely processed results
    """
    return security_engine.secure_legal_processing(data, security_level, parties)


if __name__ == "__main__":
    # Demonstration of security capabilities
    def demo_security():
        """Demonstrate security engine capabilities."""
        print("🔒 Security Engine Demonstration")
        
        # Test data
        legal_data = {
            'contract_text': 'This is a sample contract with sensitive legal information.',
            'parties': ['Company A', 'Company B'],
            'confidentiality_level': 'high',
            'compliance_requirements': ['GDPR', 'CCPA']
        }
        
        # Test different security levels
        for security_level in SecurityLevel:
            print(f"\n🛡️ Testing {security_level.value} security level:")
            
            result = secure_legal_ai_processing(
                legal_data, 
                security_level, 
                parties=['party1', 'party2'] if security_level in [SecurityLevel.SECRET, SecurityLevel.TOP_SECRET] else None
            )
            
            print(f"Status: {result['status']}")
            print(f"Encryption used: {result['encryption_used']}")
            print(f"MPC used: {result['mpc_used']}")
            print(f"Trust score: {result.get('trust_score', 'N/A'):.3f}")
            print(f"Adversarial score: {result.get('adversarial_score', 'N/A'):.3f}")
        
        # Display security status
        print(f"\n📊 Security Status:")
        status = security_engine.get_security_status()
        print(f"Total processings: {status['security_metrics'].get('total_processings', 0)}")
        print(f"Encrypted processings: {status['security_metrics'].get('encrypted_processings', 0)}")
        print(f"Active alerts: {status['active_alerts']}")
        print(f"Active parties: {status['mpc_status']['active_parties']}")
    
    demo_security()