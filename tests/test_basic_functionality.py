"""
Basic functionality tests for neuro-symbolic law prover.
"""

import pytest
from neuro_symbolic_law import LegalProver, ContractParser
from neuro_symbolic_law.regulations import GDPR, AIAct, CCPA
from neuro_symbolic_law.core.compliance_result import ComplianceStatus


class TestContractParser:
    """Test contract parsing functionality."""
    
    def test_parser_initialization(self):
        """Test parser can be initialized."""
        parser = ContractParser()
        assert parser is not None
        assert parser.model == 'basic'
    
    def test_parse_simple_contract(self):
        """Test parsing a simple contract."""
        parser = ContractParser()
        
        contract_text = """
        DATA PROCESSING AGREEMENT
        
        This agreement is between ACME Corp and CloudTech Inc.
        
        1. The data controller shall implement appropriate technical measures to ensure data security.
        2. Personal data shall be processed only for specified purposes.
        3. Data subjects have the right to access their personal data.
        """
        
        parsed = parser.parse(contract_text, "test_contract")
        
        assert parsed.id == "test_contract"
        assert "DATA PROCESSING AGREEMENT" in parsed.title
        assert len(parsed.clauses) > 0
        assert len(parsed.parties) >= 2
        
        # Check some parties were extracted
        party_names = [p.name for p in parsed.parties]
        assert any("ACME" in name for name in party_names)
        assert any("CloudTech" in name for name in party_names)
    
    def test_clause_classification(self):
        """Test that clauses are properly classified."""
        parser = ContractParser()
        
        contract_text = """
        1. Personal data must be processed securely with encryption.
        2. The company is liable for any damages up to $1,000,000.
        3. This agreement terminates after 2 years.
        4. All confidential information must remain secret.
        """
        
        parsed = parser.parse(contract_text)
        
        categories = [c.category for c in parsed.clauses if c.category]
        assert 'security' in categories or 'data_processing' in categories
        assert 'liability' in categories
        assert 'termination' in categories
        assert 'confidentiality' in categories


class TestRegulations:
    """Test regulation models."""
    
    def test_gdpr_requirements(self):
        """Test GDPR regulation model."""
        gdpr = GDPR()
        
        assert len(gdpr) > 0
        assert gdpr.name == "GDPR"
        
        # Check some key requirements exist
        requirements = gdpr.get_requirements()
        req_ids = set(requirements.keys())
        
        assert "GDPR-5.1.c" in req_ids  # Data minimization
        assert "GDPR-15.1" in req_ids   # Right of access
        assert "GDPR-32.1" in req_ids   # Security measures
        
        # Check categories
        categories = gdpr.get_categories()
        assert 'data_minimization' in categories
        assert 'security' in categories
        assert 'data_subject_rights' in categories
    
    def test_ai_act_requirements(self):
        """Test AI Act regulation model."""
        ai_act = AIAct()
        
        assert len(ai_act) > 0
        assert ai_act.name == "EU AI Act"
        
        # Check some key requirements exist
        requirements = ai_act.get_requirements()
        req_ids = set(requirements.keys())
        
        assert "AI-ACT-9.1" in req_ids   # Risk management
        assert "AI-ACT-13.1" in req_ids  # Transparency
        assert "AI-ACT-50.1" in req_ids  # AI disclosure
        
        # Check categories
        categories = ai_act.get_categories()
        assert 'transparency' in categories
        assert 'risk_management' in categories
        assert 'human_oversight' in categories
    
    def test_ccpa_requirements(self):
        """Test CCPA regulation model."""
        ccpa = CCPA()
        
        assert len(ccpa) > 0
        assert ccpa.name == "CCPA"
        
        # Check some key requirements exist
        requirements = ccpa.get_requirements()
        req_ids = set(requirements.keys())
        
        assert "CCPA-1798.100.a" in req_ids  # Right to know
        assert "CCPA-1798.105.a" in req_ids  # Right to delete
        assert "CCPA-1798.120.a" in req_ids  # Right to opt-out
        
        # Check categories
        categories = ccpa.get_categories()
        assert 'consumer_rights' in categories
        assert 'opt_out' in categories
        assert 'disclosure' in categories


class TestLegalProver:
    """Test legal prover functionality."""
    
    def test_prover_initialization(self):
        """Test prover can be initialized."""
        prover = LegalProver()
        assert prover is not None
        assert prover.cache_enabled is True
    
    def test_basic_compliance_verification(self):
        """Test basic compliance verification."""
        parser = ContractParser()
        prover = LegalProver()
        gdpr = GDPR()
        
        # Create a simple contract with some GDPR-relevant content
        contract_text = """
        DATA PROCESSING AGREEMENT
        
        1. Personal data shall be processed lawfully and transparently.
        2. Data collection is limited to what is necessary for specified purposes.
        3. Personal data will be kept secure with appropriate technical measures.
        4. Data subjects have the right to access their personal data.
        5. Personal data will be deleted when no longer necessary.
        """
        
        parsed_contract = parser.parse(contract_text, "gdpr_test")
        
        # Verify compliance
        results = prover.verify_compliance(
            contract=parsed_contract,
            regulation=gdpr,
            focus_areas=['data_minimization', 'security', 'data_subject_rights']
        )
        
        assert len(results) > 0
        
        # Check that we got some results
        for req_id, result in results.items():
            assert result.requirement_id == req_id
            assert result.status in [ComplianceStatus.COMPLIANT, ComplianceStatus.NON_COMPLIANT, ComplianceStatus.UNKNOWN]
            assert 0.0 <= result.confidence <= 1.0
        
        # Generate report
        report = prover.generate_compliance_report(
            contract=parsed_contract,
            regulation=gdpr,
            results=results
        )
        
        assert report.contract_id == "gdpr_test"
        assert report.regulation_name == "GDPR"
        assert report.total_requirements == len(results)
        assert 0.0 <= report.compliance_rate <= 100.0
    
    def test_cache_functionality(self):
        """Test caching functionality."""
        prover = LegalProver(cache_enabled=True)
        
        # Check cache starts empty
        stats = prover.get_cache_stats()
        assert stats['cached_results'] == 0
        assert stats['cache_enabled'] is True
        
        # Clear cache should work
        prover.clear_cache()
        stats = prover.get_cache_stats()
        assert stats['cached_results'] == 0


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_end_to_end_gdpr_compliance(self):
        """Test complete GDPR compliance checking workflow."""
        # Initialize components
        parser = ContractParser()
        prover = LegalProver()
        gdpr = GDPR()
        
        # Sample GDPR-relevant contract
        contract_text = """
        PRIVACY POLICY AND DATA PROCESSING AGREEMENT
        
        1. We collect personal data including name, email, and usage analytics.
        2. Data is processed for service provision and legitimate business interests.
        3. We implement encryption and access controls to protect your data.
        4. You have the right to access, rectify, and delete your personal data.
        5. Data is retained for 24 months after account closure.
        6. We share data with service providers under appropriate contracts.
        7. Data breaches will be reported within 72 hours to authorities.
        """
        
        # Parse and verify
        parsed_contract = parser.parse(contract_text, "privacy_policy")
        results = prover.verify_compliance(parsed_contract, gdpr)
        
        # Generate comprehensive report
        report = prover.generate_compliance_report(parsed_contract, gdpr, results)
        
        # Verify report structure
        assert report.contract_id == "privacy_policy"
        assert report.regulation_name == "GDPR"
        assert report.total_requirements > 0
        assert isinstance(report.compliance_rate, float)
        assert report.overall_status in [
            ComplianceStatus.COMPLIANT,
            ComplianceStatus.NON_COMPLIANT, 
            ComplianceStatus.PARTIAL
        ]
        
        # Check that some requirements were found compliant due to relevant keywords
        compliant_results = [r for r in results.values() if r.compliant]
        assert len(compliant_results) > 0, "Should find at least some compliant requirements"
        
        # Verify supporting clauses were identified
        results_with_clauses = [r for r in results.values() if r.supporting_clauses]
        assert len(results_with_clauses) > 0, "Should identify supporting clauses"