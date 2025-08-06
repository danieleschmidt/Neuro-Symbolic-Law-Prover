"""
Data sharing agreement analysis and compliance verification.
"""

from typing import Dict, List, Optional, Any, Set, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

from ..core.enhanced_prover import EnhancedLegalProver
from ..parsing.neural_parser import NeuralContractParser
from ..regulations import GDPR, CCPA
from ..core.compliance_result import ComplianceResult

logger = logging.getLogger(__name__)


class DataTransferMechanism(Enum):
    """Data transfer mechanisms for international transfers."""
    ADEQUACY_DECISION = "adequacy_decision"
    STANDARD_CONTRACTUAL_CLAUSES = "standard_contractual_clauses"
    BINDING_CORPORATE_RULES = "binding_corporate_rules"
    CERTIFICATION = "certification"
    APPROVED_CODE_OF_CONDUCT = "approved_code_of_conduct"
    DEROGATIONS = "derogations"
    UNKNOWN = "unknown"


@dataclass
class DataFlow:
    """Represents a data flow between parties."""
    source: str
    destination: str
    data_categories: List[str]
    purposes: List[str]
    transfer_mechanism: DataTransferMechanism
    jurisdictions: List[str]
    safeguards: List[str]


@dataclass
class DataSharingRisk:
    """Risk associated with data sharing arrangement."""
    risk_type: str
    severity: str
    description: str
    affected_data: List[str]
    mitigation_required: bool
    suggested_controls: List[str]


class DataSharingAgreementAnalyzer:
    """
    Specialized analyzer for data sharing agreements.
    
    Features:
    - Data flow mapping and analysis
    - International transfer compliance
    - Cross-border data protection assessment
    - Data subject rights verification
    - Purpose limitation analysis
    """
    
    def __init__(self, debug: bool = False):
        """Initialize data sharing agreement analyzer."""
        self.prover = EnhancedLegalProver(debug=debug)
        self.parser = NeuralContractParser(debug=debug)
        self.debug = debug
        
        # Data categories and sensitivity levels
        self.data_categories = self._initialize_data_categories()
        
        # Transfer mechanism requirements
        self.transfer_mechanisms = self._initialize_transfer_mechanisms()
        
        # Jurisdiction adequacy status
        self.adequacy_decisions = self._initialize_adequacy_decisions()
    
    def analyze_data_sharing_agreement(
        self,
        contract_text: str,
        contract_id: Optional[str] = None,
        source_jurisdiction: str = "EU",
        destination_jurisdictions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive data sharing agreement analysis.
        
        Args:
            contract_text: Data sharing agreement text
            contract_id: Contract identifier  
            source_jurisdiction: Source jurisdiction for data
            destination_jurisdictions: Destination jurisdictions
            
        Returns:
            Comprehensive data sharing analysis
        """
        logger.info(f"Analyzing data sharing agreement {contract_id}")
        
        try:
            # Enhanced parsing
            parsed_contract, contract_graph = self.parser.parse_enhanced(
                contract_text, contract_id, extract_semantics=True, build_graph=True
            )
            
            # Extract data flows
            data_flows = self._extract_data_flows(
                parsed_contract, source_jurisdiction, destination_jurisdictions
            )
            
            # Compliance verification
            compliance_results = self._verify_data_sharing_compliance(
                parsed_contract, data_flows, source_jurisdiction
            )
            
            # Transfer mechanism analysis
            transfer_analysis = self._analyze_transfer_mechanisms(
                parsed_contract, data_flows
            )
            
            # Data subject rights analysis
            rights_analysis = self._analyze_data_subject_rights(
                parsed_contract, data_flows
            )
            
            # Purpose limitation analysis
            purpose_analysis = self._analyze_purpose_limitations(
                parsed_contract, data_flows
            )
            
            # Risk assessment
            risk_assessment = self._assess_data_sharing_risks(
                data_flows, transfer_analysis, compliance_results
            )
            
            # Generate recommendations
            recommendations = self._generate_data_sharing_recommendations(
                data_flows, compliance_results, risk_assessment
            )
            
            return {
                'contract_id': parsed_contract.id,
                'data_flows': [flow.__dict__ for flow in data_flows],
                'compliance_results': compliance_results,
                'transfer_analysis': transfer_analysis,
                'rights_analysis': rights_analysis,
                'purpose_analysis': purpose_analysis,
                'risk_assessment': risk_assessment,
                'recommendations': recommendations,
                'contract_graph': contract_graph
            }
            
        except Exception as e:
            logger.error(f"Error analyzing data sharing agreement: {e}")
            return {'error': str(e)}
    
    def map_data_flows(self, contract) -> List[DataFlow]:
        """Map data flows described in the contract."""
        
        logger.info("Mapping data flows from contract")
        
        data_flows = []
        
        try:
            # Extract parties
            parties = [party.name for party in contract.parties]
            
            # Find data sharing clauses
            sharing_keywords = [
                'share', 'transfer', 'disclose', 'provide', 'transmit',
                'send', 'access', 'process', 'data flow'
            ]
            
            sharing_clauses = contract.get_clauses_containing(sharing_keywords)
            
            # Extract flows from each clause
            for clause in sharing_clauses:
                flows = self._extract_flows_from_clause(clause, parties)
                data_flows.extend(flows)
            
            # Deduplicate and enhance flows
            data_flows = self._deduplicate_data_flows(data_flows)
            data_flows = self._enhance_data_flows_with_context(data_flows, contract)
            
        except Exception as e:
            logger.error(f"Error mapping data flows: {e}")
        
        return data_flows
    
    def assess_international_transfers(
        self,
        data_flows: List[DataFlow]
    ) -> Dict[str, Any]:
        """Assess compliance for international data transfers."""
        
        logger.info("Assessing international transfer compliance")
        
        assessment = {
            'international_transfers': [],
            'adequacy_compliance': {},
            'transfer_mechanisms': {},
            'high_risk_transfers': [],
            'compliance_status': 'unknown'
        }
        
        try:
            for flow in data_flows:
                # Check if transfer is international
                if self._is_international_transfer(flow):
                    assessment['international_transfers'].append(flow)
                    
                    # Check adequacy decision status
                    adequacy_status = self._check_adequacy_decision(flow.jurisdictions)
                    assessment['adequacy_compliance'][f"{flow.source}->{flow.destination}"] = adequacy_status
                    
                    # Analyze transfer mechanism
                    mechanism_analysis = self._analyze_transfer_mechanism(flow)
                    assessment['transfer_mechanisms'][f"{flow.source}->{flow.destination}"] = mechanism_analysis
                    
                    # Identify high-risk transfers
                    if self._is_high_risk_transfer(flow, adequacy_status):
                        assessment['high_risk_transfers'].append(flow)
            
            # Overall compliance status
            assessment['compliance_status'] = self._determine_transfer_compliance_status(assessment)
            
        except Exception as e:
            logger.error(f"Error assessing international transfers: {e}")
        
        return assessment
    
    def verify_purpose_limitation(
        self,
        contract,
        specified_purposes: List[str]
    ) -> Dict[str, Any]:
        """Verify purpose limitation compliance."""
        
        logger.info("Verifying purpose limitation compliance")
        
        # Extract actual uses from contract
        actual_uses = self._extract_actual_data_uses(contract)
        
        # Extract stated purposes
        stated_purposes = self._extract_stated_purposes(contract)
        
        verification = {
            'specified_purposes': specified_purposes,
            'stated_purposes': stated_purposes,
            'actual_uses': actual_uses,
            'purpose_compatibility': {},
            'unauthorized_uses': [],
            'compliance_status': 'compliant'
        }
        
        try:
            # Check compatibility of each actual use
            for use in actual_uses:
                compatible = self._check_purpose_compatibility(use, stated_purposes + specified_purposes)
                verification['purpose_compatibility'][use] = {
                    'compatible': compatible,
                    'matching_purposes': [p for p in stated_purposes + specified_purposes if self._purposes_compatible(use, p)]
                }
                
                if not compatible:
                    verification['unauthorized_uses'].append(use)
                    verification['compliance_status'] = 'non_compliant'
            
            # Additional checks for purpose specification
            if not stated_purposes and not specified_purposes:
                verification['compliance_status'] = 'non_compliant'
                verification['issues'] = ['No purposes specified']
            elif self._purposes_too_broad(stated_purposes):
                verification['compliance_status'] = 'partial'
                verification['issues'] = ['Purposes too broad or vague']
        
        except Exception as e:
            logger.error(f"Error verifying purpose limitation: {e}")
            verification['error'] = str(e)
        
        return verification
    
    def _extract_data_flows(
        self,
        contract,
        source_jurisdiction: str,
        destination_jurisdictions: Optional[List[str]]
    ) -> List[DataFlow]:
        """Extract data flows from contract."""
        
        # Start with basic data flow mapping
        data_flows = self.map_data_flows(contract)
        
        # Enhance with jurisdiction information
        for flow in data_flows:
            if not flow.jurisdictions:
                # Infer jurisdictions from context
                flow.jurisdictions = self._infer_jurisdictions(
                    flow, source_jurisdiction, destination_jurisdictions
                )
            
            # Determine transfer mechanism
            if flow.transfer_mechanism == DataTransferMechanism.UNKNOWN:
                flow.transfer_mechanism = self._determine_transfer_mechanism(flow, contract)
            
            # Extract safeguards
            flow.safeguards = self._extract_safeguards_for_flow(flow, contract)
        
        return data_flows
    
    def _verify_data_sharing_compliance(
        self,
        contract,
        data_flows: List[DataFlow],
        source_jurisdiction: str
    ) -> Dict[str, Dict[str, ComplianceResult]]:
        """Verify compliance with data sharing regulations."""
        
        compliance_results = {}
        
        # Select regulations based on jurisdiction
        if source_jurisdiction.upper() in ['EU', 'EEA', 'EUROPE']:
            gdpr = GDPR()
            gdpr_results = self.prover.verify_compliance(
                contract, gdpr,
                focus_areas=['data_minimization', 'purpose_limitation', 'data_subject_rights', 'security']
            )
            compliance_results['GDPR'] = gdpr_results
        
        if source_jurisdiction.upper() in ['US', 'CA', 'CALIFORNIA']:
            ccpa = CCPA()
            ccpa_results = self.prover.verify_compliance(
                contract, ccpa,
                focus_areas=['consumer_rights', 'disclosure', 'opt_out']
            )
            compliance_results['CCPA'] = ccpa_results
        
        # Add data sharing specific checks
        sharing_compliance = self._verify_sharing_specific_requirements(contract, data_flows)
        compliance_results['DATA_SHARING'] = sharing_compliance
        
        return compliance_results
    
    def _analyze_transfer_mechanisms(
        self,
        contract,
        data_flows: List[DataFlow]
    ) -> Dict[str, Any]:
        """Analyze data transfer mechanisms used."""
        
        analysis = {
            'mechanisms_identified': [],
            'adequacy_reliance': 0,
            'scc_usage': 0,
            'bcr_usage': 0,
            'derogations_usage': 0,
            'unknown_mechanisms': 0,
            'compliance_assessment': {}
        }
        
        for flow in data_flows:
            mechanism = flow.transfer_mechanism
            analysis['mechanisms_identified'].append(mechanism.value)
            
            if mechanism == DataTransferMechanism.ADEQUACY_DECISION:
                analysis['adequacy_reliance'] += 1
            elif mechanism == DataTransferMechanism.STANDARD_CONTRACTUAL_CLAUSES:
                analysis['scc_usage'] += 1
            elif mechanism == DataTransferMechanism.BINDING_CORPORATE_RULES:
                analysis['bcr_usage'] += 1
            elif mechanism == DataTransferMechanism.DEROGATIONS:
                analysis['derogations_usage'] += 1
            elif mechanism == DataTransferMechanism.UNKNOWN:
                analysis['unknown_mechanisms'] += 1
            
            # Assess compliance for each mechanism
            flow_key = f"{flow.source}->{flow.destination}"
            analysis['compliance_assessment'][flow_key] = self._assess_mechanism_compliance(
                flow, contract
            )
        
        return analysis
    
    def _analyze_data_subject_rights(
        self,
        contract,
        data_flows: List[DataFlow]
    ) -> Dict[str, Any]:
        """Analyze data subject rights provisions."""
        
        rights_keywords = [
            'access', 'rectification', 'erasure', 'deletion',
            'portability', 'object', 'restrict processing',
            'data subject', 'individual rights'
        ]
        
        rights_clauses = contract.get_clauses_containing(rights_keywords)
        
        analysis = {
            'rights_clauses_count': len(rights_clauses),
            'rights_addressed': [],
            'cross_border_rights': False,
            'response_mechanisms': False,
            'compliance_score': 0.0
        }
        
        gdpr_rights = [
            'access', 'rectification', 'erasure', 'portability',
            'restriction', 'objection', 'automated_decision_making'
        ]
        
        for right in gdpr_rights:
            if any(right in clause.text.lower() for clause in rights_clauses):
                analysis['rights_addressed'].append(right)
        
        # Check for cross-border rights handling
        cross_border_keywords = ['cross-border', 'international', 'third country']
        if any(any(keyword in clause.text.lower() for keyword in cross_border_keywords) 
               for clause in rights_clauses):
            analysis['cross_border_rights'] = True
        
        # Check for response mechanisms
        response_keywords = ['respond', 'response time', 'within', 'days']
        if any(any(keyword in clause.text.lower() for keyword in response_keywords)
               for clause in rights_clauses):
            analysis['response_mechanisms'] = True
        
        # Calculate compliance score
        score_components = [
            len(analysis['rights_addressed']) / len(gdpr_rights),  # Rights coverage
            0.2 if analysis['cross_border_rights'] else 0.0,      # Cross-border handling
            0.2 if analysis['response_mechanisms'] else 0.0       # Response mechanisms
        ]
        
        analysis['compliance_score'] = min(1.0, sum(score_components))
        
        return analysis
    
    def _initialize_data_categories(self) -> Dict[str, Dict[str, Any]]:
        """Initialize data categories and sensitivity levels."""
        
        return {
            'personal_identifiers': {
                'sensitivity': 'high',
                'examples': ['name', 'address', 'phone', 'email', 'id_number'],
                'special_protections': True
            },
            'financial_data': {
                'sensitivity': 'high',
                'examples': ['bank_account', 'credit_card', 'financial_records'],
                'special_protections': True
            },
            'health_data': {
                'sensitivity': 'special_category',
                'examples': ['medical_records', 'health_status', 'genetic_data'],
                'special_protections': True
            },
            'behavioral_data': {
                'sensitivity': 'medium',
                'examples': ['browsing_history', 'preferences', 'usage_patterns'],
                'special_protections': False
            },
            'location_data': {
                'sensitivity': 'high',
                'examples': ['gps_coordinates', 'geolocation', 'tracking_data'],
                'special_protections': True
            },
            'biometric_data': {
                'sensitivity': 'special_category',
                'examples': ['fingerprints', 'facial_recognition', 'voice_prints'],
                'special_protections': True
            }
        }
    
    def _initialize_transfer_mechanisms(self) -> Dict[str, Dict[str, Any]]:
        """Initialize data transfer mechanisms and requirements."""
        
        return {
            'adequacy_decision': {
                'requirements': ['adequate_protection_level'],
                'compliance_check': 'adequacy_status'
            },
            'standard_contractual_clauses': {
                'requirements': ['approved_sccs', 'additional_safeguards', 'impact_assessment'],
                'compliance_check': 'scc_compliance'
            },
            'binding_corporate_rules': {
                'requirements': ['approved_bcrs', 'group_companies', 'supervisory_authority_approval'],
                'compliance_check': 'bcr_compliance'
            },
            'certification': {
                'requirements': ['approved_certification', 'binding_enforceable_commitments'],
                'compliance_check': 'certification_compliance'
            },
            'derogations': {
                'requirements': ['specific_situation', 'occasional_transfer', 'limited_data'],
                'compliance_check': 'derogation_compliance'
            }
        }
    
    def _initialize_adequacy_decisions(self) -> Dict[str, str]:
        """Initialize jurisdiction adequacy decision status."""
        
        return {
            'AD': 'adequate',  # Andorra
            'AR': 'adequate',  # Argentina
            'CA': 'adequate',  # Canada (commercial organizations)
            'FO': 'adequate',  # Faroe Islands
            'GG': 'adequate',  # Guernsey
            'IL': 'adequate',  # Israel
            'IM': 'adequate',  # Isle of Man
            'JP': 'adequate',  # Japan
            'JE': 'adequate',  # Jersey
            'NZ': 'adequate',  # New Zealand
            'KR': 'adequate',  # South Korea
            'CH': 'adequate',  # Switzerland
            'UY': 'adequate',  # Uruguay
            'GB': 'adequate',  # United Kingdom
            'US': 'partial',   # United States (framework dependent)
            'CN': 'not_adequate',  # China
            'RU': 'not_adequate',  # Russia
            'IN': 'not_adequate',  # India
        }
    
    def _extract_flows_from_clause(self, clause, parties: List[str]) -> List[DataFlow]:
        """Extract data flows from individual clause."""
        
        flows = []
        clause_text = clause.text.lower()
        
        try:
            # Simple pattern matching for data flows
            # In a full implementation, this would use advanced NLP
            
            for source_party in parties:
                for dest_party in parties:
                    if source_party != dest_party:
                        if source_party.lower() in clause_text and dest_party.lower() in clause_text:
                            # Check for data sharing indicators
                            if any(verb in clause_text for verb in ['share', 'transfer', 'provide', 'send']):
                                
                                # Extract data categories
                                data_categories = self._extract_data_categories_from_text(clause_text)
                                
                                # Extract purposes  
                                purposes = self._extract_purposes_from_text(clause_text)
                                
                                flow = DataFlow(
                                    source=source_party,
                                    destination=dest_party,
                                    data_categories=data_categories,
                                    purposes=purposes,
                                    transfer_mechanism=DataTransferMechanism.UNKNOWN,
                                    jurisdictions=[],
                                    safeguards=[]
                                )
                                flows.append(flow)
        
        except Exception as e:
            logger.error(f"Error extracting flows from clause: {e}")
        
        return flows
    
    def _extract_data_categories_from_text(self, text: str) -> List[str]:
        """Extract data categories from text."""
        
        categories = []
        text_lower = text.lower()
        
        category_keywords = {
            'personal_identifiers': ['name', 'address', 'phone', 'email', 'identifier'],
            'financial_data': ['financial', 'payment', 'bank', 'credit'],
            'health_data': ['health', 'medical', 'genetic'],
            'behavioral_data': ['behavior', 'usage', 'activity', 'preference'],
            'location_data': ['location', 'gps', 'geolocation'],
            'biometric_data': ['biometric', 'fingerprint', 'facial']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                categories.append(category)
        
        return categories or ['personal_data']
    
    def _extract_purposes_from_text(self, text: str) -> List[str]:
        """Extract processing purposes from text."""
        
        purposes = []
        text_lower = text.lower()
        
        purpose_patterns = [
            r'for\s+([\w\s]+?)(?:\s+and|\s+or|,|\.|$)',
            r'purpose\s+of\s+([\w\s]+?)(?:\s+and|\s+or|,|\.|$)',
            r'in\s+order\s+to\s+([\w\s]+?)(?:\s+and|\s+or|,|\.|$)'
        ]
        
        import re
        for pattern in purpose_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                purpose = match.group(1).strip()
                if len(purpose) > 3 and len(purpose) < 50:  # Reasonable purpose length
                    purposes.append(purpose)
        
        return purposes or ['unspecified']
    
    def _deduplicate_data_flows(self, flows: List[DataFlow]) -> List[DataFlow]:
        """Remove duplicate data flows."""
        
        unique_flows = []
        seen = set()
        
        for flow in flows:
            key = (flow.source, flow.destination, tuple(flow.data_categories))
            if key not in seen:
                seen.add(key)
                unique_flows.append(flow)
        
        return unique_flows
    
    def _enhance_data_flows_with_context(self, flows: List[DataFlow], contract) -> List[DataFlow]:
        """Enhance data flows with additional context."""
        
        for flow in flows:
            # Add jurisdiction information
            flow.jurisdictions = self._infer_jurisdictions_from_parties(
                flow.source, flow.destination, contract
            )
            
            # Determine transfer mechanism
            flow.transfer_mechanism = self._determine_transfer_mechanism(flow, contract)
            
            # Extract safeguards
            flow.safeguards = self._extract_safeguards_for_flow(flow, contract)
        
        return flows
    
    def _is_international_transfer(self, flow: DataFlow) -> bool:
        """Check if flow represents international transfer."""
        
        if len(flow.jurisdictions) > 1:
            return True
        
        # Additional checks based on party names or context
        international_indicators = [
            'international', 'cross-border', 'global', 'worldwide',
            'third country', 'outside', 'abroad'
        ]
        
        flow_context = f"{flow.source} {flow.destination}".lower()
        return any(indicator in flow_context for indicator in international_indicators)
    
    def _check_adequacy_decision(self, jurisdictions: List[str]) -> Dict[str, str]:
        """Check adequacy decision status for jurisdictions."""
        
        adequacy_status = {}
        
        for jurisdiction in jurisdictions:
            # Try to match with known adequacy decisions
            jurisdiction_code = jurisdiction.upper()[:2]  # Get country code
            status = self.adequacy_decisions.get(jurisdiction_code, 'unknown')
            adequacy_status[jurisdiction] = status
        
        return adequacy_status
    
    def _analyze_transfer_mechanism(self, flow: DataFlow) -> Dict[str, Any]:
        """Analyze transfer mechanism for specific flow."""
        
        mechanism = flow.transfer_mechanism
        
        analysis = {
            'mechanism': mechanism.value,
            'compliant': False,
            'requirements_met': [],
            'requirements_missing': [],
            'recommendations': []
        }
        
        if mechanism in self.transfer_mechanisms:
            requirements = self.transfer_mechanisms[mechanism.value]['requirements']
            
            # Check each requirement
            for requirement in requirements:
                if self._check_mechanism_requirement(flow, requirement):
                    analysis['requirements_met'].append(requirement)
                else:
                    analysis['requirements_missing'].append(requirement)
            
            analysis['compliant'] = len(analysis['requirements_missing']) == 0
            
            # Generate recommendations for missing requirements
            for missing in analysis['requirements_missing']:
                analysis['recommendations'].append(
                    f"Implement {missing.replace('_', ' ')} for compliant transfer"
                )
        
        return analysis
    
    def _is_high_risk_transfer(self, flow: DataFlow, adequacy_status: Dict[str, str]) -> bool:
        """Determine if transfer is high-risk."""
        
        # Check for non-adequate destinations
        if any(status in ['not_adequate', 'unknown'] for status in adequacy_status.values()):
            return True
        
        # Check for sensitive data categories
        sensitive_categories = ['health_data', 'biometric_data', 'financial_data']
        if any(cat in flow.data_categories for cat in sensitive_categories):
            return True
        
        # Check for inadequate safeguards
        if not flow.safeguards:
            return True
        
        return False
    
    def _determine_transfer_compliance_status(self, assessment: Dict[str, Any]) -> str:
        """Determine overall transfer compliance status."""
        
        if assessment['high_risk_transfers']:
            return 'high_risk'
        elif assessment['unknown_mechanisms'] > 0:
            return 'needs_review'
        elif len(assessment['international_transfers']) == 0:
            return 'domestic_only'
        else:
            return 'compliant'
    
    def _extract_actual_data_uses(self, contract) -> List[str]:
        """Extract actual data uses from contract."""
        
        use_keywords = [
            'use', 'process', 'analyze', 'share', 'disclose',
            'store', 'retain', 'access', 'collect'
        ]
        
        use_clauses = contract.get_clauses_containing(use_keywords)
        
        uses = []
        for clause in use_clauses:
            # Extract uses using pattern matching
            clause_text = clause.text.lower()
            
            use_patterns = [
                r'(?:use|process|analyze)\s+(?:for|to)\s+([\w\s]+?)(?:\s+and|\s+or|,|\.|$)',
                r'(?:share|disclose)\s+(?:for|to)\s+([\w\s]+?)(?:\s+and|\s+or|,|\.|$)'
            ]
            
            import re
            for pattern in use_patterns:
                matches = re.finditer(pattern, clause_text)
                for match in matches:
                    use = match.group(1).strip()
                    if len(use) > 3 and len(use) < 50:
                        uses.append(use)
        
        return uses
    
    def _extract_stated_purposes(self, contract) -> List[str]:
        """Extract stated purposes from contract."""
        
        purpose_keywords = ['purpose', 'objective', 'goal', 'reason']
        purpose_clauses = contract.get_clauses_containing(purpose_keywords)
        
        purposes = []
        for clause in purpose_clauses:
            purposes.extend(self._extract_purposes_from_text(clause.text))
        
        return purposes
    
    def _check_purpose_compatibility(self, use: str, purposes: List[str]) -> bool:
        """Check if data use is compatible with stated purposes."""
        
        use_lower = use.lower()
        
        for purpose in purposes:
            if self._purposes_compatible(use_lower, purpose.lower()):
                return True
        
        return False
    
    def _purposes_compatible(self, use: str, purpose: str) -> bool:
        """Check if use and purpose are compatible."""
        
        # Simple keyword matching - in practice would use semantic similarity
        use_words = set(use.split())
        purpose_words = set(purpose.split())
        
        # Check for direct overlap
        if use_words & purpose_words:
            return True
        
        # Check for semantic compatibility
        compatible_mappings = {
            'marketing': ['promotion', 'advertising', 'commercial'],
            'analytics': ['analysis', 'insights', 'reporting'],
            'support': ['service', 'assistance', 'help'],
            'research': ['study', 'investigation', 'development']
        }
        
        for use_word in use_words:
            if use_word in compatible_mappings:
                if any(compat in purpose_words for compat in compatible_mappings[use_word]):
                    return True
        
        return False
    
    def _purposes_too_broad(self, purposes: List[str]) -> bool:
        """Check if purposes are too broad or vague."""
        
        broad_terms = [
            'business purposes', 'legitimate interests', 'any purpose',
            'commercial purposes', 'operational purposes'
        ]
        
        return any(any(broad in purpose.lower() for broad in broad_terms) 
                  for purpose in purposes)
    
    def _assess_data_sharing_risks(
        self,
        data_flows: List[DataFlow],
        transfer_analysis: Dict[str, Any],
        compliance_results: Dict[str, Dict[str, ComplianceResult]]
    ) -> Dict[str, Any]:
        """Assess risks associated with data sharing arrangement."""
        
        risks = []
        risk_score = 0
        
        # Assess risks for each flow
        for flow in data_flows:
            flow_risks = self._assess_flow_risks(flow, transfer_analysis)
            risks.extend(flow_risks)
            risk_score += sum(10 if risk.severity == 'high' else 5 if risk.severity == 'medium' else 2 
                             for risk in flow_risks)
        
        # Add compliance-based risks
        compliance_violations = 0
        for reg_results in compliance_results.values():
            for result in reg_results.values():
                if not result.compliant:
                    compliance_violations += 1
                    risk_score += 8
        
        # Determine overall risk level
        if risk_score >= 50:
            risk_level = 'HIGH'
        elif risk_score >= 30:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        return {
            'risk_level': risk_level,
            'risk_score': min(100, risk_score),
            'total_risks': len(risks),
            'high_risks': len([r for r in risks if r.severity == 'high']),
            'compliance_violations': compliance_violations,
            'detailed_risks': [risk.__dict__ for risk in risks]
        }
    
    def _assess_flow_risks(self, flow: DataFlow, transfer_analysis: Dict[str, Any]) -> List[DataSharingRisk]:
        """Assess risks for individual data flow."""
        
        risks = []
        
        # International transfer risks
        if self._is_international_transfer(flow):
            if flow.transfer_mechanism == DataTransferMechanism.UNKNOWN:
                risks.append(DataSharingRisk(
                    risk_type='unknown_transfer_mechanism',
                    severity='high',
                    description='International transfer without identified legal mechanism',
                    affected_data=flow.data_categories,
                    mitigation_required=True,
                    suggested_controls=['Implement Standard Contractual Clauses', 'Conduct transfer impact assessment']
                ))
            
            if not flow.safeguards:
                risks.append(DataSharingRisk(
                    risk_type='insufficient_safeguards',
                    severity='medium',
                    description='International transfer without adequate safeguards',
                    affected_data=flow.data_categories,
                    mitigation_required=True,
                    suggested_controls=['Implement additional safeguards', 'Add security measures']
                ))
        
        # Sensitive data risks
        sensitive_categories = ['health_data', 'biometric_data', 'financial_data']
        sensitive_data = [cat for cat in flow.data_categories if cat in sensitive_categories]
        
        if sensitive_data:
            risks.append(DataSharingRisk(
                risk_type='sensitive_data_sharing',
                severity='high',
                description=f'Sharing of sensitive data categories: {", ".join(sensitive_data)}',
                affected_data=sensitive_data,
                mitigation_required=True,
                suggested_controls=['Enhanced security measures', 'Additional consent requirements']
            ))
        
        # Purpose limitation risks
        if 'unspecified' in flow.purposes:
            risks.append(DataSharingRisk(
                risk_type='unspecified_purposes',
                severity='medium',
                description='Data sharing without clearly specified purposes',
                affected_data=flow.data_categories,
                mitigation_required=True,
                suggested_controls=['Specify clear processing purposes', 'Update agreement with purpose limitations']
            ))
        
        return risks
    
    def _generate_data_sharing_recommendations(
        self,
        data_flows: List[DataFlow],
        compliance_results: Dict[str, Dict[str, ComplianceResult]],
        risk_assessment: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for data sharing agreement."""
        
        recommendations = []
        
        # High priority risk-based recommendations
        if risk_assessment['risk_level'] == 'HIGH':
            recommendations.append("URGENT: Address high-risk data sharing arrangements immediately")
        
        high_risks = [risk for risk in risk_assessment['detailed_risks'] 
                     if isinstance(risk, dict) and risk.get('severity') == 'high']
        
        for risk in high_risks[:3]:  # Top 3 high risks
            if risk.get('suggested_controls'):
                recommendations.extend([f"HIGH RISK: {control}" for control in risk['suggested_controls']])
        
        # International transfer recommendations
        international_flows = [flow for flow in data_flows if self._is_international_transfer(flow)]
        
        if international_flows:
            unknown_mechanisms = [flow for flow in international_flows 
                                 if flow.transfer_mechanism == DataTransferMechanism.UNKNOWN]
            
            if unknown_mechanisms:
                recommendations.append("Implement appropriate transfer mechanisms for international data flows")
        
        # Compliance-based recommendations
        for reg_name, results in compliance_results.items():
            non_compliant = [r for r in results.values() if not r.compliant]
            for result in non_compliant[:2]:  # Top 2 per regulation
                if result.suggestion:
                    recommendations.append(f"{reg_name}: {result.suggestion}")
        
        # General data sharing best practices
        recommendations.extend([
            "Conduct regular reviews of data sharing arrangements",
            "Implement monitoring for data sharing activities",
            "Establish clear data retention and deletion procedures",
            "Provide training on data sharing compliance requirements"
        ])
        
        return recommendations[:10]  # Limit to top 10 recommendations