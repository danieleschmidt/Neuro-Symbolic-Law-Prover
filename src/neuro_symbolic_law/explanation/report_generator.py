"""
Comprehensive compliance report generation.
"""

from typing import Dict, List, Optional, Any, Set
import logging
from dataclasses import dataclass, field
from datetime import datetime
import json

from ..parsing.contract_parser import ParsedContract
from ..core.compliance_result import ComplianceReport, ComplianceResult, ComplianceStatus, ViolationSeverity
from .explainer import ExplainabilityEngine, Explanation

logger = logging.getLogger(__name__)


@dataclass 
class ReportSection:
    """Individual section of compliance report."""
    title: str
    content: str
    subsections: List['ReportSection'] = field(default_factory=list)
    charts: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ComprehensiveReport:
    """Complete compliance report with all sections."""
    executive_summary: ReportSection
    detailed_findings: ReportSection
    risk_assessment: ReportSection
    remediation_plan: ReportSection
    technical_appendix: Optional[ReportSection] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ComplianceReporter:
    """
    Generates comprehensive compliance reports from verification results.
    
    Creates professional reports suitable for legal teams, executives,
    and technical teams with visualizations and actionable recommendations.
    """
    
    def __init__(self, explainer: Optional[ExplainabilityEngine] = None):
        """Initialize compliance reporter."""
        self.explainer = explainer or ExplainabilityEngine()
    
    def generate_report(
        self,
        contract: ParsedContract,
        compliance_results: Dict[str, ComplianceResult],
        regulation_name: str,
        include_sections: List[str] = None
    ) -> ComprehensiveReport:
        """
        Generate comprehensive compliance report.
        
        Args:
            contract: Contract that was analyzed
            compliance_results: Results from compliance verification
            regulation_name: Name of regulation checked
            include_sections: Sections to include in report
            
        Returns:
            Complete compliance report
        """
        logger.info(f"Generating compliance report for {contract.id}")
        
        if include_sections is None:
            include_sections = [
                'executive_summary',
                'detailed_findings', 
                'risk_assessment',
                'remediation_plan'
            ]
        
        # Create compliance report object
        compliance_report = self._create_compliance_report(
            contract, compliance_results, regulation_name
        )
        
        # Generate report sections
        report = ComprehensiveReport(
            executive_summary=self._generate_executive_summary(compliance_report),
            detailed_findings=self._generate_detailed_findings(compliance_report),
            risk_assessment=self._generate_risk_assessment(compliance_report),
            remediation_plan=self._generate_remediation_plan(compliance_report),
            metadata={
                'generated_at': datetime.now().isoformat(),
                'contract_id': contract.id,
                'regulation': regulation_name,
                'total_requirements': len(compliance_results),
                'compliance_rate': compliance_report.compliance_rate
            }
        )
        
        # Add technical appendix if requested
        if 'technical_appendix' in include_sections:
            report.technical_appendix = self._generate_technical_appendix(compliance_report)
        
        return report
    
    def create_dashboard(
        self,
        contracts: List[ParsedContract],
        regulations: List[Any],
        update_frequency: str = "daily"
    ) -> Dict[str, Any]:
        """
        Create interactive compliance dashboard.
        
        Args:
            contracts: List of contracts to monitor
            regulations: List of regulations to check
            update_frequency: How often to update dashboard
            
        Returns:
            Dashboard configuration
        """
        logger.info(f"Creating dashboard for {len(contracts)} contracts")
        
        dashboard_config = {
            "title": "Legal Compliance Dashboard",
            "update_frequency": update_frequency,
            "contracts": [
                {
                    "id": contract.id,
                    "title": contract.title,
                    "type": contract.contract_type,
                    "parties": len(contract.parties)
                } for contract in contracts
            ],
            "regulations": [reg.name for reg in regulations],
            "widgets": [
                {
                    "type": "compliance_overview",
                    "title": "Overall Compliance Status",
                    "size": "large"
                },
                {
                    "type": "violation_trends", 
                    "title": "Compliance Trends",
                    "size": "medium"
                },
                {
                    "type": "risk_heatmap",
                    "title": "Risk Assessment",
                    "size": "medium"
                },
                {
                    "type": "action_items",
                    "title": "Required Actions",
                    "size": "small"
                }
            ],
            "alerts": {
                "critical_violations": True,
                "new_regulations": True,
                "compliance_changes": True
            }
        }
        
        return dashboard_config
    
    def _create_compliance_report(
        self,
        contract: ParsedContract,
        results: Dict[str, ComplianceResult],
        regulation_name: str
    ) -> ComplianceReport:
        """Create ComplianceReport object from results."""
        
        # Determine overall status
        compliant_count = sum(1 for r in results.values() if r.compliant)
        total_count = len(results)
        
        if compliant_count == total_count and total_count > 0:
            overall_status = ComplianceStatus.COMPLIANT
        elif compliant_count == 0:
            overall_status = ComplianceStatus.NON_COMPLIANT
        else:
            overall_status = ComplianceStatus.PARTIAL
        
        return ComplianceReport(
            contract_id=contract.id,
            regulation_name=regulation_name,
            results=results,
            overall_status=overall_status,
            timestamp=datetime.now().isoformat()
        )
    
    def _generate_executive_summary(self, report: ComplianceReport) -> ReportSection:
        """Generate executive summary section."""
        
        # Key metrics
        compliance_rate = report.compliance_rate
        violation_count = report.violation_count
        critical_violations = len(report.get_critical_violations())
        
        # Status assessment
        if compliance_rate >= 90:
            status_assessment = "EXCELLENT"
            status_color = "green"
        elif compliance_rate >= 75:
            status_assessment = "GOOD" 
            status_color = "yellow"
        elif compliance_rate >= 50:
            status_assessment = "NEEDS IMPROVEMENT"
            status_color = "orange"
        else:
            status_assessment = "CRITICAL"
            status_color = "red"
        
        # Generate summary content
        content = f"""
**Contract:** {report.contract_id}
**Regulation:** {report.regulation_name}
**Overall Status:** {status_assessment} ({compliance_rate:.1f}% compliant)
**Assessment Date:** {report.timestamp[:10]}

## Key Findings

- **Total Requirements Assessed:** {report.total_requirements}
- **Compliant Requirements:** {report.compliant_count}
- **Non-Compliant Requirements:** {report.total_requirements - report.compliant_count}
- **Total Violations:** {violation_count}
- **Critical Violations:** {critical_violations}

## Risk Level

{self._assess_overall_risk(report)}

## Immediate Action Required

{self._generate_immediate_actions(report)}
        """.strip()
        
        # Add chart data
        charts = [
            {
                "type": "pie",
                "title": "Compliance Overview",
                "data": {
                    "compliant": report.compliant_count,
                    "non_compliant": report.total_requirements - report.compliant_count
                }
            },
            {
                "type": "bar",
                "title": "Violations by Severity", 
                "data": self._get_violations_by_severity_data(report)
            }
        ]
        
        return ReportSection(
            title="Executive Summary",
            content=content,
            charts=charts
        )
    
    def _generate_detailed_findings(self, report: ComplianceReport) -> ReportSection:
        """Generate detailed findings section."""
        
        content_parts = ["## Detailed Compliance Analysis\n"]
        
        # Group results by status
        compliant_results = [r for r in report.results.values() if r.compliant]
        non_compliant_results = [r for r in report.results.values() if not r.compliant]
        
        # Non-compliant findings first
        if non_compliant_results:
            content_parts.append("### ❌ Non-Compliant Requirements\n")
            
            for result in non_compliant_results:
                content_parts.append(f"#### {result.requirement_id}")
                content_parts.append(f"**Requirement:** {result.requirement_description}")
                content_parts.append(f"**Confidence:** {result.confidence:.1%}")
                
                if result.issue:
                    content_parts.append(f"**Issue:** {result.issue}")
                
                if result.suggestion:
                    content_parts.append(f"**Suggested Fix:** {result.suggestion}")
                
                # Add explanation
                if result.violations:
                    for violation in result.violations:
                        explanation = self.explainer.explain_violation(
                            violation, 
                            None,  # Contract context not needed for basic explanation
                            "legal_team"
                        )
                        content_parts.append(f"**Legal Analysis:** {explanation.legal_explanation}")
                
                content_parts.append("")  # Empty line
        
        # Compliant findings
        if compliant_results:
            content_parts.append("### ✅ Compliant Requirements\n")
            
            for result in compliant_results:
                content_parts.append(f"#### {result.requirement_id}")
                content_parts.append(f"**Requirement:** {result.requirement_description}")
                content_parts.append(f"**Confidence:** {result.confidence:.1%}")
                
                if result.supporting_clauses:
                    content_parts.append(f"**Supporting Evidence:** {len(result.supporting_clauses)} clause(s)")
                
                if result.formal_proof:
                    content_parts.append(f"**Verification:** {result.formal_proof}")
                
                content_parts.append("")  # Empty line
        
        return ReportSection(
            title="Detailed Findings",
            content="\n".join(content_parts)
        )
    
    def _generate_risk_assessment(self, report: ComplianceReport) -> ReportSection:
        """Generate risk assessment section."""
        
        # Analyze risks by severity and impact
        violations = report.get_violations()
        
        risk_levels = {
            "Critical": len([v for v in violations if v.severity == ViolationSeverity.CRITICAL]),
            "High": len([v for v in violations if v.severity == ViolationSeverity.HIGH]), 
            "Medium": len([v for v in violations if v.severity == ViolationSeverity.MEDIUM]),
            "Low": len([v for v in violations if v.severity == ViolationSeverity.LOW])
        }
        
        # Calculate risk score
        risk_score = (
            risk_levels["Critical"] * 10 +
            risk_levels["High"] * 5 +
            risk_levels["Medium"] * 2 +
            risk_levels["Low"] * 1
        )
        
        # Risk categorization
        if risk_score >= 20:
            overall_risk = "CRITICAL"
        elif risk_score >= 10:
            overall_risk = "HIGH"
        elif risk_score >= 5:
            overall_risk = "MEDIUM"
        else:
            overall_risk = "LOW"
        
        content = f"""
## Risk Assessment Summary

**Overall Risk Level:** {overall_risk}
**Risk Score:** {risk_score}/100

### Risk Breakdown by Severity

- **Critical Risks:** {risk_levels["Critical"]}
- **High Risks:** {risk_levels["High"]}
- **Medium Risks:** {risk_levels["Medium"]}
- **Low Risks:** {risk_levels["Low"]}

### Potential Business Impact

{self._assess_business_impact(report)}

### Regulatory Exposure

{self._assess_regulatory_exposure(report)}

### Recommended Risk Mitigation Priority

{self._prioritize_risk_mitigation(violations)}
        """.strip()
        
        charts = [
            {
                "type": "bar",
                "title": "Risk Distribution",
                "data": risk_levels
            }
        ]
        
        return ReportSection(
            title="Risk Assessment",
            content=content,
            charts=charts
        )
    
    def _generate_remediation_plan(self, report: ComplianceReport) -> ReportSection:
        """Generate remediation plan section."""
        
        violations = report.get_violations()
        
        # Prioritize violations by severity and impact
        prioritized_violations = sorted(
            violations,
            key=lambda v: (v.severity.value, v.rule_id),
            reverse=True
        )
        
        content_parts = ["## Compliance Remediation Plan\n"]
        
        # Immediate actions (Critical/High severity)
        immediate_actions = [
            v for v in prioritized_violations 
            if v.severity in [ViolationSeverity.CRITICAL, ViolationSeverity.HIGH]
        ]
        
        if immediate_actions:
            content_parts.append("### 🚨 Immediate Actions Required (0-30 days)\n")
            
            for i, violation in enumerate(immediate_actions, 1):
                content_parts.append(f"#### {i}. {violation.rule_id}")
                content_parts.append(f"**Priority:** {violation.severity.value.upper()}")
                content_parts.append(f"**Issue:** {violation.violation_text}")
                
                if violation.suggested_fix:
                    content_parts.append(f"**Action:** {violation.suggested_fix}")
                
                # Get detailed remediation steps
                explanation = self.explainer.explain_violation(violation, None)
                content_parts.append("**Steps:**")
                for step in explanation.remediation_steps[:3]:  # Top 3 steps
                    content_parts.append(f"- {step}")
                
                content_parts.append("")
        
        # Short-term actions (Medium severity)
        medium_actions = [
            v for v in prioritized_violations 
            if v.severity == ViolationSeverity.MEDIUM
        ]
        
        if medium_actions:
            content_parts.append("### 📅 Short-term Actions (30-90 days)\n")
            
            for violation in medium_actions:
                content_parts.append(f"- **{violation.rule_id}:** {violation.violation_text}")
                if violation.suggested_fix:
                    content_parts.append(f"  *Fix:* {violation.suggested_fix}")
            
            content_parts.append("")
        
        # Long-term improvements
        content_parts.append("### 📈 Long-term Improvements (90+ days)\n")
        content_parts.append("- Implement automated compliance monitoring")
        content_parts.append("- Establish regular compliance audits")
        content_parts.append("- Update contract templates with compliance provisions")
        content_parts.append("- Train staff on regulatory requirements")
        
        # Success metrics
        content_parts.append("\n### Success Metrics\n")
        content_parts.append("- Achieve 95%+ compliance rate")
        content_parts.append("- Zero critical violations")
        content_parts.append("- Reduce overall risk score to LOW")
        content_parts.append("- Implement continuous monitoring")
        
        return ReportSection(
            title="Remediation Plan",
            content="\n".join(content_parts)
        )
    
    def _generate_technical_appendix(self, report: ComplianceReport) -> ReportSection:
        """Generate technical appendix section."""
        
        content_parts = ["## Technical Implementation Details\n"]
        
        # Formal verification details
        formal_results = [
            r for r in report.results.values() 
            if r.formal_proof
        ]
        
        if formal_results:
            content_parts.append("### Formal Verification Results\n")
            
            for result in formal_results:
                content_parts.append(f"#### {result.requirement_id}")
                content_parts.append(f"**Proof:** {result.formal_proof}")
                content_parts.append(f"**Confidence:** {result.confidence:.3f}")
                content_parts.append("")
        
        # System recommendations
        content_parts.append("### Technical Recommendations\n")
        content_parts.append("- Implement automated compliance checking in CI/CD pipeline")
        content_parts.append("- Add compliance validation to contract management system")
        content_parts.append("- Create compliance dashboard for ongoing monitoring")
        content_parts.append("- Establish audit logging for compliance activities")
        
        # Data export
        content_parts.append("### Raw Data Export\n")
        content_parts.append("```json")
        content_parts.append(json.dumps({
            "contract_id": report.contract_id,
            "regulation": report.regulation_name, 
            "compliance_rate": report.compliance_rate,
            "results_summary": {
                "total": report.total_requirements,
                "compliant": report.compliant_count,
                "violations": report.violation_count
            }
        }, indent=2))
        content_parts.append("```")
        
        return ReportSection(
            title="Technical Appendix",
            content="\n".join(content_parts)
        )
    
    def _assess_overall_risk(self, report: ComplianceReport) -> str:
        """Assess overall compliance risk."""
        
        compliance_rate = report.compliance_rate
        critical_violations = len(report.get_critical_violations())
        
        if critical_violations > 0:
            return "**CRITICAL RISK** - Immediate action required to address critical compliance violations."
        elif compliance_rate < 50:
            return "**HIGH RISK** - Significant compliance gaps require urgent attention."
        elif compliance_rate < 75:
            return "**MEDIUM RISK** - Several compliance issues should be addressed."
        else:
            return "**LOW RISK** - Minor compliance improvements recommended."
    
    def _generate_immediate_actions(self, report: ComplianceReport) -> str:
        """Generate immediate action items."""
        
        critical_violations = report.get_critical_violations()
        
        if critical_violations:
            actions = []
            for violation in critical_violations[:3]:  # Top 3
                actions.append(f"- Address {violation.rule_id}: {violation.violation_text}")
            return "\n".join(actions)
        
        elif report.violation_count > 0:
            return "- Review and address identified compliance violations"
        
        else:
            return "- Monitor for regulatory changes\n- Maintain current compliance practices"
    
    def _get_violations_by_severity_data(self, report: ComplianceReport) -> Dict[str, int]:
        """Get violations grouped by severity for charts."""
        
        violations = report.get_violations()
        
        return {
            "Critical": len([v for v in violations if v.severity == ViolationSeverity.CRITICAL]),
            "High": len([v for v in violations if v.severity == ViolationSeverity.HIGH]),
            "Medium": len([v for v in violations if v.severity == ViolationSeverity.MEDIUM]),
            "Low": len([v for v in violations if v.severity == ViolationSeverity.LOW])
        }
    
    def _assess_business_impact(self, report: ComplianceReport) -> str:
        """Assess potential business impact."""
        
        if "GDPR" in report.regulation_name:
            return "Potential GDPR fines up to 4% of annual global turnover or €20 million, whichever is higher."
        elif "AI Act" in report.regulation_name:
            return "Potential AI Act fines up to €35 million or 7% of annual turnover, plus market access restrictions."
        else:
            return "Potential regulatory enforcement, fines, and reputational damage."
    
    def _assess_regulatory_exposure(self, report: ComplianceReport) -> str:
        """Assess regulatory exposure."""
        
        violation_count = report.violation_count
        
        if violation_count == 0:
            return "Minimal regulatory exposure with current compliance posture."
        elif violation_count <= 3:
            return "Limited regulatory exposure - address violations to minimize risk."
        else:
            return "Significant regulatory exposure - comprehensive remediation needed."
    
    def _prioritize_risk_mitigation(self, violations: List) -> str:
        """Prioritize risk mitigation efforts."""
        
        if not violations:
            return "Maintain current compliance practices and monitor for changes."
        
        priority_actions = [
            "1. Address all critical and high-severity violations immediately",
            "2. Develop systematic approach to medium-severity issues", 
            "3. Implement ongoing monitoring and compliance processes",
            "4. Regular legal and technical compliance reviews"
        ]
        
        return "\n".join(priority_actions)