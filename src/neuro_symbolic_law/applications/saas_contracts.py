"""
SaaS contract analysis and compliance checking.
"""

from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass

from ..core.enhanced_prover import EnhancedLegalProver
from ..parsing.neural_parser import NeuralContractParser
from ..core.compliance_result import ComplianceResult, ComplianceReport
from ..regulations import GDPR, CCPA

logger = logging.getLogger(__name__)


@dataclass
class SaaSComplianceIssue:
    """SaaS-specific compliance issue."""
    issue_type: str
    severity: str
    description: str
    clause_reference: Optional[str] = None
    recommended_action: Optional[str] = None


class SaaSContractAnalyzer:
    """
    Specialized analyzer for Software-as-a-Service contracts.
    
    Focuses on SaaS-specific compliance requirements:
    - Data processing and privacy
    - Service level agreements
    - Data portability and export
    - Vendor security responsibilities
    - Liability and indemnification
    """
    
    def __init__(self, debug: bool = False):
        """Initialize SaaS contract analyzer."""
        self.prover = EnhancedLegalProver(debug=debug)
        self.parser = NeuralContractParser(debug=debug)
        self.debug = debug
        
        # SaaS-specific requirement patterns
        self.saas_requirements = self._initialize_saas_requirements()
    
    def analyze_saas_contract(
        self,
        contract_text: str,
        contract_id: Optional[str] = None,
        jurisdiction: str = "EU"
    ) -> Dict[str, Any]:
        """
        Comprehensive SaaS contract analysis.
        
        Args:
            contract_text: SaaS contract text
            contract_id: Contract identifier
            jurisdiction: Legal jurisdiction (EU, US, etc.)
            
        Returns:
            Comprehensive SaaS compliance analysis
        """
        logger.info(f"Analyzing SaaS contract for jurisdiction: {jurisdiction}")
        
        try:
            # Parse contract with enhanced features
            parsed_contract, contract_graph = self.parser.parse_enhanced(
                contract_text, contract_id, extract_semantics=True, build_graph=True
            )
            
            # Select appropriate regulations
            regulations = self._select_regulations_for_jurisdiction(jurisdiction)
            
            # Perform compliance verification
            compliance_results = {}
            for reg_name, regulation in regulations.items():
                results = self.prover.verify_compliance(parsed_contract, regulation)
                compliance_results[reg_name] = results
            
            # SaaS-specific analysis
            saas_issues = self._analyze_saas_specific_requirements(parsed_contract)
            
            # Service level analysis
            sla_analysis = self._analyze_service_levels(parsed_contract)
            
            # Data security assessment
            security_assessment = self._assess_data_security(parsed_contract)
            
            # Liability analysis
            liability_analysis = self._analyze_liability_clauses(parsed_contract)
            
            # Generate comprehensive report
            return {
                'contract_id': parsed_contract.id,
                'jurisdiction': jurisdiction,
                'compliance_results': compliance_results,
                'saas_issues': saas_issues,
                'sla_analysis': sla_analysis,
                'security_assessment': security_assessment,
                'liability_analysis': liability_analysis,
                'overall_risk_score': self._calculate_overall_risk_score(
                    compliance_results, saas_issues
                ),
                'contract_graph': contract_graph,
                'recommendations': self._generate_saas_recommendations(
                    compliance_results, saas_issues
                )
            }
            
        except Exception as e:
            logger.error(f"Error analyzing SaaS contract: {e}")
            return {'error': str(e)}
    
    def check_data_processing_compliance(
        self,
        contract_text: str,
        data_categories: List[str],
        processing_purposes: List[str]
    ) -> Dict[str, Any]:
        """
        Check data processing compliance for specific data categories and purposes.
        
        Args:
            contract_text: Contract text
            data_categories: Categories of data processed
            processing_purposes: Purposes for processing
            
        Returns:
            Data processing compliance analysis
        """
        logger.info("Checking data processing compliance")
        
        try:
            parsed_contract, _ = self.parser.parse_enhanced(contract_text)
            
            # Extract data processing clauses
            data_clauses = self._extract_data_processing_clauses(parsed_contract)
            
            # Analyze coverage for each data category
            coverage_analysis = {}
            for category in data_categories:
                coverage_analysis[category] = self._analyze_data_category_coverage(
                    data_clauses, category
                )
            
            # Analyze purpose limitation compliance
            purpose_analysis = {}
            for purpose in processing_purposes:
                purpose_analysis[purpose] = self._analyze_purpose_coverage(
                    data_clauses, purpose
                )
            
            # GDPR-specific checks
            gdpr = GDPR()
            gdpr_results = self.prover.verify_compliance(
                parsed_contract, gdpr, focus_areas=['data_minimization', 'purpose_limitation']
            )
            
            return {
                'data_categories_coverage': coverage_analysis,
                'purpose_analysis': purpose_analysis,
                'gdpr_compliance': gdpr_results,
                'recommendations': self._generate_data_processing_recommendations(
                    coverage_analysis, purpose_analysis, gdpr_results
                )
            }
            
        except Exception as e:
            logger.error(f"Error in data processing compliance check: {e}")
            return {'error': str(e)}
    
    def _initialize_saas_requirements(self) -> Dict[str, Dict[str, Any]]:
        """Initialize SaaS-specific requirements."""
        
        return {
            'data_portability': {
                'description': 'Contract must provide for data export/portability',
                'keywords': ['export', 'portability', 'data transfer', 'migration'],
                'mandatory': True,
                'severity': 'high'
            },
            'service_availability': {
                'description': 'Service level agreements must specify availability targets',
                'keywords': ['uptime', 'availability', 'sla', 'service level'],
                'mandatory': True,
                'severity': 'medium'
            },
            'data_backup': {
                'description': 'Data backup and recovery procedures must be specified',
                'keywords': ['backup', 'recovery', 'disaster recovery', 'data loss'],
                'mandatory': True,
                'severity': 'high'
            },
            'vendor_security': {
                'description': 'Vendor security responsibilities must be clearly defined',
                'keywords': ['security', 'encryption', 'access controls', 'monitoring'],
                'mandatory': True,
                'severity': 'critical'
            },
            'data_location': {
                'description': 'Data storage and processing location must be specified',
                'keywords': ['data location', 'jurisdiction', 'cross-border', 'storage'],
                'mandatory': True,
                'severity': 'high'
            },
            'vendor_access': {
                'description': 'Limitations on vendor access to customer data',
                'keywords': ['vendor access', 'customer data', 'access restrictions'],
                'mandatory': True,
                'severity': 'high'
            }
        }
    
    def _select_regulations_for_jurisdiction(self, jurisdiction: str) -> Dict[str, Any]:
        """Select appropriate regulations based on jurisdiction."""
        
        regulations = {}
        
        if jurisdiction.upper() in ['EU', 'EEA', 'EUROPE']:
            regulations['GDPR'] = GDPR()
        
        if jurisdiction.upper() in ['US', 'CA', 'CALIFORNIA']:
            regulations['CCPA'] = CCPA()
        
        # Default to GDPR if jurisdiction not recognized
        if not regulations:
            regulations['GDPR'] = GDPR()
        
        return regulations
    
    def _analyze_saas_specific_requirements(self, contract) -> List[SaaSComplianceIssue]:
        """Analyze SaaS-specific compliance requirements."""
        
        issues = []
        
        for req_id, requirement in self.saas_requirements.items():
            keywords = requirement['keywords']
            mandatory = requirement['mandatory']
            severity = requirement['severity']
            description = requirement['description']
            
            # Check if requirement is addressed in contract
            relevant_clauses = contract.get_clauses_containing(keywords)
            
            if mandatory and not relevant_clauses:
                issues.append(SaaSComplianceIssue(
                    issue_type=req_id,
                    severity=severity,
                    description=f"Missing requirement: {description}",
                    recommended_action=f"Add clause addressing {description.lower()}"
                ))
            elif relevant_clauses:
                # Analyze adequacy of coverage
                adequacy_score = self._assess_requirement_adequacy(
                    relevant_clauses, requirement
                )
                
                if adequacy_score < 0.7:  # Threshold for adequate coverage
                    issues.append(SaaSComplianceIssue(
                        issue_type=req_id,
                        severity='medium',
                        description=f"Inadequate coverage: {description}",
                        clause_reference=relevant_clauses[0].id if relevant_clauses else None,
                        recommended_action=f"Enhance clause addressing {description.lower()}"
                    ))
        
        return issues
    
    def _analyze_service_levels(self, contract) -> Dict[str, Any]:
        """Analyze service level agreements."""
        
        sla_keywords = [
            'uptime', 'availability', 'service level', 'performance',
            'response time', 'resolution time', 'maintenance window'
        ]
        
        sla_clauses = contract.get_clauses_containing(sla_keywords)
        
        analysis = {
            'has_sla': len(sla_clauses) > 0,
            'sla_clauses_count': len(sla_clauses),
            'metrics_specified': [],
            'penalties_specified': False,
            'maintenance_windows': False
        }
        
        # Extract specific SLA metrics
        for clause in sla_clauses:
            clause_text = clause.text.lower()
            
            if any(metric in clause_text for metric in ['99%', '99.9%', '99.99%']):
                analysis['metrics_specified'].append('uptime_percentage')
            
            if 'response time' in clause_text:
                analysis['metrics_specified'].append('response_time')
            
            if any(word in clause_text for word in ['penalty', 'credit', 'compensation']):
                analysis['penalties_specified'] = True
            
            if 'maintenance' in clause_text:
                analysis['maintenance_windows'] = True
        
        return analysis
    
    def _assess_data_security(self, contract) -> Dict[str, Any]:
        """Assess data security provisions."""
        
        security_keywords = [
            'encryption', 'ssl', 'tls', 'https',
            'access control', 'authentication', 'authorization',
            'security audit', 'penetration test', 'vulnerability',
            'incident response', 'data breach'
        ]
        
        security_clauses = contract.get_clauses_containing(security_keywords)
        
        assessment = {
            'security_clauses_count': len(security_clauses),
            'encryption_specified': False,
            'access_controls': False,
            'security_audits': False,
            'incident_response': False,
            'security_score': 0.0
        }
        
        score_components = []
        
        for clause in security_clauses:
            clause_text = clause.text.lower()
            
            if any(enc in clause_text for enc in ['encrypt', 'ssl', 'tls']):
                assessment['encryption_specified'] = True
                score_components.append(0.3)
            
            if any(ac in clause_text for ac in ['access control', 'authentication']):
                assessment['access_controls'] = True
                score_components.append(0.2)
            
            if any(audit in clause_text for audit in ['security audit', 'penetration test']):
                assessment['security_audits'] = True
                score_components.append(0.2)
            
            if any(ir in clause_text for ir in ['incident response', 'data breach']):
                assessment['incident_response'] = True
                score_components.append(0.3)
        
        assessment['security_score'] = min(1.0, sum(score_components))
        
        return assessment
    
    def _analyze_liability_clauses(self, contract) -> Dict[str, Any]:
        """Analyze liability and indemnification clauses."""
        
        liability_keywords = [
            'liable', 'liability', 'damages', 'indemnify', 'indemnification',
            'limitation of liability', 'consequential damages', 'direct damages'
        ]
        
        liability_clauses = contract.get_clauses_containing(liability_keywords)
        
        analysis = {
            'liability_clauses_count': len(liability_clauses),
            'liability_limited': False,
            'indemnification_provided': False,
            'mutual_indemnification': False,
            'exclusions_specified': False
        }
        
        for clause in liability_clauses:
            clause_text = clause.text.lower()
            
            if 'limitation' in clause_text and 'liability' in clause_text:
                analysis['liability_limited'] = True
            
            if 'indemnify' in clause_text or 'indemnification' in clause_text:
                analysis['indemnification_provided'] = True
                
                if 'mutual' in clause_text:
                    analysis['mutual_indemnification'] = True
            
            if any(exclusion in clause_text for exclusion in ['exclude', 'not liable', 'no liability']):
                analysis['exclusions_specified'] = True
        
        return analysis
    
    def _extract_data_processing_clauses(self, contract) -> List:
        """Extract clauses related to data processing."""
        
        data_keywords = [
            'personal data', 'personal information', 'customer data',
            'process', 'processing', 'collect', 'collection',
            'use', 'usage', 'store', 'storage'
        ]
        
        return contract.get_clauses_containing(data_keywords)
    
    def _analyze_data_category_coverage(self, data_clauses: List, category: str) -> Dict[str, Any]:
        """Analyze coverage for specific data category."""
        
        coverage = {
            'category': category,
            'mentioned': False,
            'processing_specified': False,
            'purpose_specified': False,
            'retention_specified': False
        }
        
        category_lower = category.lower()
        
        for clause in data_clauses:
            clause_text = clause.text.lower()
            
            if category_lower in clause_text:
                coverage['mentioned'] = True
                
                if any(process in clause_text for process in ['process', 'collect', 'use']):
                    coverage['processing_specified'] = True
                
                if any(purpose in clause_text for purpose in ['purpose', 'for', 'to']):
                    coverage['purpose_specified'] = True
                
                if any(retention in clause_text for retention in ['retain', 'keep', 'store']):
                    coverage['retention_specified'] = True
        
        return coverage
    
    def _analyze_purpose_coverage(self, data_clauses: List, purpose: str) -> Dict[str, Any]:
        """Analyze coverage for specific processing purpose."""
        
        coverage = {
            'purpose': purpose,
            'mentioned': False,
            'data_specified': False,
            'lawful_basis': False
        }
        
        purpose_lower = purpose.lower()
        
        for clause in data_clauses:
            clause_text = clause.text.lower()
            
            if purpose_lower in clause_text:
                coverage['mentioned'] = True
                
                if 'personal data' in clause_text or 'information' in clause_text:
                    coverage['data_specified'] = True
                
                if any(basis in clause_text for basis in ['consent', 'contract', 'legitimate']):
                    coverage['lawful_basis'] = True
        
        return coverage
    
    def _assess_requirement_adequacy(self, clauses: List, requirement: Dict[str, Any]) -> float:
        """Assess how adequately a requirement is covered."""
        
        if not clauses:
            return 0.0
        
        keywords = requirement['keywords']
        total_score = 0.0
        
        for clause in clauses:
            clause_text = clause.text.lower()
            keyword_matches = sum(1 for keyword in keywords if keyword in clause_text)
            clause_score = min(1.0, keyword_matches / len(keywords))
            total_score += clause_score
        
        return min(1.0, total_score / len(clauses))
    
    def _calculate_overall_risk_score(
        self,
        compliance_results: Dict[str, Dict[str, ComplianceResult]],
        saas_issues: List[SaaSComplianceIssue]
    ) -> Dict[str, Any]:
        """Calculate overall risk score for SaaS contract."""
        
        total_violations = 0
        critical_violations = 0
        
        # Count compliance violations
        for reg_name, results in compliance_results.items():
            for result in results.values():
                if not result.compliant:
                    total_violations += 1
                    if any(v.severity.value == 'critical' for v in result.violations):
                        critical_violations += 1
        
        # Count SaaS-specific issues
        saas_critical = len([issue for issue in saas_issues if issue.severity == 'critical'])
        saas_high = len([issue for issue in saas_issues if issue.severity == 'high'])
        
        # Calculate risk score (0-100)
        risk_score = min(100, (
            critical_violations * 20 +
            saas_critical * 15 +
            total_violations * 5 +
            saas_high * 10 +
            len(saas_issues) * 2
        ))
        
        # Risk level categorization
        if risk_score >= 70:
            risk_level = 'CRITICAL'
        elif risk_score >= 50:
            risk_level = 'HIGH'
        elif risk_score >= 30:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'total_violations': total_violations,
            'critical_violations': critical_violations + saas_critical,
            'saas_issues': len(saas_issues)
        }
    
    def _generate_saas_recommendations(
        self,
        compliance_results: Dict[str, Dict[str, ComplianceResult]],
        saas_issues: List[SaaSComplianceIssue]
    ) -> List[str]:
        """Generate recommendations for SaaS contract improvement."""
        
        recommendations = []
        
        # Priority: Critical issues first
        critical_issues = [issue for issue in saas_issues if issue.severity == 'critical']
        for issue in critical_issues:
            if issue.recommended_action:
                recommendations.append(f"CRITICAL: {issue.recommended_action}")
        
        # High priority issues
        high_issues = [issue for issue in saas_issues if issue.severity == 'high']
        for issue in high_issues[:3]:  # Top 3
            if issue.recommended_action:
                recommendations.append(f"HIGH: {issue.recommended_action}")
        
        # Compliance violations
        for reg_name, results in compliance_results.items():
            non_compliant = [r for r in results.values() if not r.compliant]
            for result in non_compliant[:2]:  # Top 2 per regulation
                if result.suggestion:
                    recommendations.append(f"{reg_name}: {result.suggestion}")
        
        # General SaaS best practices
        if not any('data portability' in rec.lower() for rec in recommendations):
            recommendations.append("Ensure data portability and export capabilities are clearly defined")
        
        if not any('security' in rec.lower() for rec in recommendations):
            recommendations.append("Specify comprehensive security measures and vendor responsibilities")
        
        return recommendations
    
    def _generate_data_processing_recommendations(
        self,
        coverage_analysis: Dict[str, Dict[str, Any]],
        purpose_analysis: Dict[str, Dict[str, Any]], 
        gdpr_results: Dict[str, ComplianceResult]
    ) -> List[str]:
        """Generate data processing specific recommendations."""
        
        recommendations = []
        
        # Check data category coverage
        for category, coverage in coverage_analysis.items():
            if not coverage['mentioned']:
                recommendations.append(f"Add explicit provisions for processing {category} data")
            elif not coverage['purpose_specified']:
                recommendations.append(f"Specify processing purposes for {category} data")
            elif not coverage['retention_specified']:
                recommendations.append(f"Define retention period for {category} data")
        
        # Check purpose coverage
        for purpose, coverage in purpose_analysis.items():
            if not coverage['lawful_basis']:
                recommendations.append(f"Specify lawful basis for {purpose} processing")
        
        # GDPR-specific recommendations
        for result in gdpr_results.values():
            if not result.compliant and result.suggestion:
                recommendations.append(f"GDPR: {result.suggestion}")
        
        return recommendations