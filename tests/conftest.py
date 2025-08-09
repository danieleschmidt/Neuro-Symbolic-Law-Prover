"""
Pytest configuration and fixtures for neuro-symbolic law prover tests.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neuro_symbolic_law.core.legal_prover import LegalProver
from neuro_symbolic_law.core.enhanced_prover import EnhancedLegalProver
from neuro_symbolic_law.parsing.contract_parser import ContractParser, ParsedContract
from neuro_symbolic_law.regulations.gdpr import GDPR
from neuro_symbolic_law.regulations.ai_act import AIAct
from neuro_symbolic_law.regulations.ccpa import CCPA
from neuro_symbolic_law.performance.cache_manager import CacheManager
from neuro_symbolic_law.performance.resource_pool import ResourcePool, WorkerPool
from neuro_symbolic_law.performance.load_balancer import LoadBalancer


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_contract_text():
    """Sample contract text for testing."""
    return """
    DATA PROCESSING AGREEMENT
    
    This Data Processing Agreement ("Agreement") is entered into between Company A ("Data Controller") 
    and Company B ("Data Processor").
    
    1. PURPOSE AND SCOPE
    The Data Processor shall process personal data on behalf of the Data Controller in accordance 
    with this Agreement and applicable data protection laws.
    
    2. DATA SECURITY
    The Data Processor shall implement appropriate technical and organizational measures to ensure 
    a level of security appropriate to the risk, including encryption of personal data.
    
    3. DATA SUBJECT RIGHTS
    The Data Processor shall assist the Data Controller in responding to requests from data subjects 
    to exercise their rights, including rights of access, rectification, erasure, and portability.
    
    4. DATA RETENTION
    Personal data shall be retained only for as long as necessary for the purposes specified in 
    this Agreement and shall be securely deleted upon termination.
    
    5. INTERNATIONAL TRANSFERS
    The Data Processor may transfer personal data outside the EU only with appropriate safeguards 
    and subject to enforceable data subject rights.
    
    6. BREACH NOTIFICATION
    The Data Processor shall notify the Data Controller of any personal data breach without undue delay 
    and in any event within 72 hours of becoming aware of the breach.
    
    7. AUDIT AND COMPLIANCE
    The Data Controller may conduct audits to verify compliance with this Agreement and applicable 
    data protection laws.
    
    8. LIABILITY
    Each party acknowledges its liability under applicable data protection laws and agrees to 
    indemnify the other party for any damages resulting from non-compliance.
    """


@pytest.fixture
def sample_ai_contract_text():
    """Sample AI system contract for testing."""
    return """
    AI SYSTEM DEPLOYMENT AGREEMENT
    
    This Agreement governs the deployment and operation of AI System v2.0 ("System") by 
    TechCorp ("Provider") for Customer Inc ("Customer").
    
    1. AI SYSTEM DESCRIPTION
    The System is a high-risk AI system for automated decision-making in credit scoring 
    and financial risk assessment.
    
    2. TRANSPARENCY AND DOCUMENTATION
    Provider shall maintain comprehensive technical documentation describing the System's 
    functionality, training data, and decision-making logic. Users shall be informed 
    when interacting with the AI system.
    
    3. HUMAN OVERSIGHT
    Customer shall ensure effective human oversight of the System, including the ability 
    for humans to intervene, override decisions, and stop the system when necessary.
    
    4. RISK MANAGEMENT
    Provider has implemented a risk management system to identify, assess, and mitigate 
    risks throughout the System's lifecycle.
    
    5. DATA GOVERNANCE
    Training data used for the System is representative, relevant, and free from bias. 
    Data quality measures are implemented to prevent discriminatory outcomes.
    
    6. ACCURACY AND ROBUSTNESS
    The System meets appropriate accuracy requirements and is resilient against adversarial 
    attacks and security threats.
    
    7. LOGGING AND MONITORING
    The System automatically logs all operations to enable audit trails and performance monitoring.
    
    8. COMPLIANCE MONITORING
    Customer shall implement post-market monitoring to track the System's performance 
    and compliance with regulatory requirements.
    """


@pytest.fixture
def sample_ccpa_contract():
    """Sample CCPA-related contract for testing."""
    return """
    CONSUMER DATA PROCESSING AGREEMENT
    
    Agreement between DataCorp ("Business") and ConsumerApp ("Service Provider") for 
    processing California consumer personal information.
    
    1. PERSONAL INFORMATION CATEGORIES
    Business collects and processes the following categories of consumer personal information:
    - Identifiers (name, email, phone)
    - Commercial information (purchase history)
    - Internet activity (browsing behavior)
    - Geolocation data
    
    2. COLLECTION NOTICE
    Consumers are informed at the time of collection about the categories of personal 
    information collected and the purposes for which it will be used.
    
    3. CONSUMER RIGHTS
    Business provides consumers with the following rights:
    - Right to know what personal information is collected
    - Right to delete personal information
    - Right to opt-out of sale of personal information
    - Right to non-discrimination for exercising privacy rights
    
    4. VERIFICATION PROCEDURES
    Business has implemented procedures to verify consumer identity before responding 
    to requests to exercise privacy rights.
    
    5. RESPONSE TIMEFRAMES
    Business responds to consumer requests within 45 days of receipt.
    
    6. THIRD PARTY SHARING
    Personal information may be shared with service providers under contractual 
    obligations to protect consumer privacy.
    
    7. RETENTION POLICY
    Personal information is retained only as long as necessary for business purposes 
    and is securely deleted when no longer needed.
    
    8. RECORD KEEPING
    Business maintains records of consumer requests and responses for compliance purposes.
    """


@pytest.fixture
def contract_parser():
    """Contract parser instance for testing."""
    return ContractParser(debug=False)


@pytest.fixture
def parsed_contract(contract_parser, sample_contract_text):
    """Pre-parsed contract for testing."""
    return contract_parser.parse(sample_contract_text, "test_contract")


@pytest.fixture
def legal_prover():
    """Basic legal prover instance for testing."""
    return LegalProver(cache_enabled=False, debug=False)


@pytest.fixture
def enhanced_prover():
    """Enhanced legal prover instance for testing."""
    return EnhancedLegalProver(
        max_workers=2,
        cache_enabled=False,
        enable_formal_verification=True,
        debug=False
    )


@pytest.fixture
def gdpr_regulation():
    """GDPR regulation instance."""
    return GDPR()


@pytest.fixture
def ai_act_regulation():
    """AI Act regulation instance."""
    return AIAct()


@pytest.fixture
def ccpa_regulation():
    """CCPA regulation instance."""
    return CCPA()


@pytest.fixture
def cache_manager():
    """Cache manager for testing."""
    return CacheManager(
        l1_max_size=100,
        l1_ttl=60,
        l2_max_size=500,
        l2_ttl=300,
        enable_compression=False,
        enable_encryption=False
    )


@pytest.fixture
def worker_pool():
    """Worker pool for testing."""
    return WorkerPool(
        cpu_workers=2,
        io_workers=4,
        process_workers=1
    )


@pytest.fixture
def load_balancer():
    """Load balancer for testing."""
    from neuro_symbolic_law.performance.load_balancer import LoadBalancingAlgorithm
    
    lb = LoadBalancer(algorithm=LoadBalancingAlgorithm.ROUND_ROBIN)
    
    # Add test servers
    lb.add_server("localhost", 8001, weight=1.0)
    lb.add_server("localhost", 8002, weight=1.5)
    lb.add_server("localhost", 8003, weight=0.5)
    
    return lb


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    class MockRedis:
        def __init__(self):
            self.data = {}
            self.connected = True
        
        def ping(self):
            if not self.connected:
                raise Exception("Not connected")
            return True
        
        def get(self, key):
            return self.data.get(key)
        
        def set(self, key, value):
            self.data[key] = value
            return True
        
        def setex(self, key, ttl, value):
            self.data[key] = value
            return True
        
        def delete(self, *keys):
            deleted = 0
            for key in keys:
                if key in self.data:
                    del self.data[key]
                    deleted += 1
            return deleted
        
        def keys(self, pattern):
            import fnmatch
            return [key for key in self.data.keys() if fnmatch.fnmatch(key, pattern)]
        
        def info(self, section=None):
            return {
                'used_memory': 1024 * 1024,
                'used_memory_peak': 2 * 1024 * 1024
            }
        
        def close(self):
            self.connected = False
    
    return MockRedis()


@pytest.fixture
def performance_test_data():
    """Generate performance test data."""
    return {
        'small_contracts': [
            "Simple contract with basic terms.",
            "Short agreement for testing performance.",
            "Minimal contract with few clauses."
        ],
        'medium_contracts': [
            sample_contract_text() for _ in range(10)
        ],
        'large_contracts': [
            sample_contract_text() * 5 for _ in range(5)
        ]
    }


@pytest.fixture(autouse=True)
def cleanup_resources():
    """Cleanup resources after each test."""
    yield
    
    # Cleanup any remaining resources
    import gc
    gc.collect()


# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_redis: mark test as requiring Redis"
    )
    config.addinivalue_line(
        "markers", "requires_z3: mark test as requiring Z3 SMT solver"
    )


# Custom assertions
def assert_compliance_result(result, expected_status=None, min_confidence=None):
    """Assert compliance result properties."""
    assert result is not None, "Compliance result should not be None"
    assert hasattr(result, 'status'), "Result should have status attribute"
    assert hasattr(result, 'confidence'), "Result should have confidence attribute"
    assert hasattr(result, 'requirement_id'), "Result should have requirement_id"
    
    if expected_status:
        assert result.status == expected_status, f"Expected status {expected_status}, got {result.status}"
    
    if min_confidence is not None:
        assert result.confidence >= min_confidence, f"Expected confidence >= {min_confidence}, got {result.confidence}"


def assert_performance_metrics(metrics, max_response_time=None, min_success_rate=None):
    """Assert performance metrics."""
    assert isinstance(metrics, dict), "Metrics should be a dictionary"
    
    if max_response_time and 'avg_response_time' in metrics:
        assert metrics['avg_response_time'] <= max_response_time, \
            f"Response time {metrics['avg_response_time']} exceeds maximum {max_response_time}"
    
    if min_success_rate and 'success_rate' in metrics:
        assert metrics['success_rate'] >= min_success_rate, \
            f"Success rate {metrics['success_rate']} below minimum {min_success_rate}"


# Test data generators
def generate_test_contracts(count: int = 10) -> List[str]:
    """Generate test contracts for performance testing."""
    base_contract = sample_contract_text()
    
    contracts = []
    for i in range(count):
        # Add variation to each contract
        contract = base_contract.replace("Company A", f"Company A{i}")
        contract = contract.replace("Company B", f"Company B{i}")
        contracts.append(contract)
    
    return contracts


def generate_compliance_scenarios() -> List[Dict[str, Any]]:
    """Generate compliance test scenarios."""
    return [
        {
            'name': 'GDPR Data Minimization',
            'regulation': 'gdpr',
            'focus_areas': ['data_minimization'],
            'expected_violations': 0
        },
        {
            'name': 'AI Act Transparency',
            'regulation': 'ai_act',
            'focus_areas': ['transparency'],
            'expected_violations': 1
        },
        {
            'name': 'CCPA Consumer Rights',
            'regulation': 'ccpa',
            'focus_areas': ['consumer_rights'],
            'expected_violations': 0
        }
    ]