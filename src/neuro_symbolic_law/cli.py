"""
Command-line interface for neuro-symbolic law prover.
"""

import click
from typing import Optional
import sys
import json
from pathlib import Path

from .core.legal_prover import LegalProver
from .parsing.contract_parser import ContractParser
from .regulations import GDPR, AIAct, CCPA


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Neuro-Symbolic Law Prover - AI-powered legal compliance verification."""
    pass


@main.command()
@click.argument('contract_file', type=click.Path(exists=True))
@click.option('--regulation', '-r', 
              type=click.Choice(['gdpr', 'ai-act', 'ccpa']), 
              default='gdpr',
              help='Regulation to check compliance against')
@click.option('--focus-areas', '-f',
              multiple=True,
              help='Specific areas to focus verification on')
@click.option('--output', '-o',
              type=click.Path(),
              help='Output file for results (JSON format)')
@click.option('--debug', '-d',
              is_flag=True,
              help='Enable debug output')
def verify(contract_file: str, regulation: str, focus_areas: tuple, output: Optional[str], debug: bool):
    """Verify contract compliance against regulations."""
    
    # Initialize components
    parser = ContractParser(debug=debug)
    prover = LegalProver(debug=debug)
    
    # Select regulation
    reg_map = {
        'gdpr': GDPR(),
        'ai-act': AIAct(), 
        'ccpa': CCPA()
    }
    selected_regulation = reg_map[regulation]
    
    try:
        # Parse contract
        click.echo(f"Parsing contract: {contract_file}")
        with open(contract_file, 'r', encoding='utf-8') as f:
            contract_text = f.read()
        
        parsed_contract = parser.parse(contract_text, contract_id=Path(contract_file).stem)
        click.echo(f"✓ Parsed {len(parsed_contract.clauses)} clauses")
        
        # Verify compliance
        click.echo(f"Verifying compliance against {selected_regulation.name}")
        focus_list = list(focus_areas) if focus_areas else None
        results = prover.verify_compliance(
            contract=parsed_contract,
            regulation=selected_regulation,
            focus_areas=focus_list
        )
        
        # Generate report
        report = prover.generate_compliance_report(
            contract=parsed_contract,
            regulation=selected_regulation,
            results=results
        )
        
        # Display results
        click.echo(f"\n📊 Compliance Report")
        click.echo(f"Contract: {report.contract_id}")
        click.echo(f"Regulation: {report.regulation_name}")
        click.echo(f"Status: {report.overall_status.value}")
        click.echo(f"Compliance Rate: {report.compliance_rate:.1f}%")
        click.echo(f"Requirements Checked: {report.total_requirements}")
        click.echo(f"Violations Found: {report.violation_count}")
        
        # Show violations
        violations = report.get_violations()
        if violations:
            click.echo(f"\n🚨 Violations:")
            for violation in violations:
                click.echo(f"  • {violation.rule_id}: {violation.violation_text}")
                if violation.suggested_fix:
                    click.echo(f"    💡 Suggestion: {violation.suggested_fix}")
        
        # Show compliant requirements
        compliant = [r for r in results.values() if r.compliant]
        if compliant:
            click.echo(f"\n✅ Compliant Requirements: {len(compliant)}")
            for result in compliant[:5]:  # Show first 5
                click.echo(f"  • {result.requirement_id}: {result.requirement_description}")
            if len(compliant) > 5:
                click.echo(f"  ... and {len(compliant) - 5} more")
        
        # Save output
        if output:
            report_data = {
                'contract_id': report.contract_id,
                'regulation': report.regulation_name,
                'status': report.overall_status.value,
                'compliance_rate': report.compliance_rate,
                'timestamp': report.timestamp,
                'results': {
                    req_id: {
                        'status': result.status.value,
                        'confidence': result.confidence,
                        'compliant': result.compliant,
                        'violations': [
                            {
                                'rule_id': v.rule_id,
                                'description': v.rule_description,
                                'severity': v.severity.value,
                                'text': v.violation_text
                            } for v in result.violations
                        ]
                    } for req_id, result in results.items()
                }
            }
            
            with open(output, 'w') as f:
                json.dump(report_data, f, indent=2)
            click.echo(f"\n📄 Report saved to: {output}")
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)
        if debug:
            raise
        sys.exit(1)


@main.command()
@click.argument('contract_file', type=click.Path(exists=True))
@click.option('--format', '-f',
              type=click.Choice(['text', 'json']),
              default='text',
              help='Output format')
def parse(contract_file: str, format: str):
    """Parse contract and extract structured information."""
    
    parser = ContractParser()
    
    try:
        # Parse contract
        with open(contract_file, 'r', encoding='utf-8') as f:
            contract_text = f.read()
        
        parsed_contract = parser.parse(contract_text, contract_id=Path(contract_file).stem)
        
        if format == 'json':
            # JSON output
            contract_data = {
                'id': parsed_contract.id,
                'title': parsed_contract.title,
                'contract_type': parsed_contract.contract_type,
                'parties': [
                    {
                        'name': p.name,
                        'role': p.role,
                        'entity_type': p.entity_type
                    } for p in parsed_contract.parties
                ],
                'clauses': [
                    {
                        'id': c.id,
                        'text': c.text,
                        'category': c.category,
                        'obligations': c.obligations,
                        'confidence': c.confidence
                    } for c in parsed_contract.clauses
                ]
            }
            click.echo(json.dumps(contract_data, indent=2))
        
        else:
            # Text output
            click.echo(f"📋 Contract Analysis")
            click.echo(f"ID: {parsed_contract.id}")
            click.echo(f"Title: {parsed_contract.title}")
            click.echo(f"Type: {parsed_contract.contract_type or 'Unknown'}")
            
            click.echo(f"\n👥 Parties ({len(parsed_contract.parties)}):")
            for party in parsed_contract.parties:
                click.echo(f"  • {party.name} ({party.role})")
            
            click.echo(f"\n📝 Clauses ({len(parsed_contract.clauses)}):")
            for clause in parsed_contract.clauses:
                category = f" [{clause.category}]" if clause.category else ""
                click.echo(f"  {clause.id}{category}: {clause.text[:100]}...")
                if clause.obligations:
                    click.echo(f"    Obligations: {', '.join(clause.obligations[:3])}")
    
    except Exception as e:
        click.echo(f"❌ Error parsing contract: {str(e)}", err=True)
        sys.exit(1)


@main.command()
@click.option('--regulation', '-r',
              type=click.Choice(['gdpr', 'ai-act', 'ccpa']),
              help='Show requirements for specific regulation')
def requirements(regulation: Optional[str]):
    """List available compliance requirements."""
    
    regulations = {
        'gdpr': GDPR(),
        'ai-act': AIAct(),
        'ccpa': CCPA()
    }
    
    if regulation:
        reg = regulations[regulation]
        click.echo(f"📋 {reg.name} Requirements ({len(reg)} total)")
        
        requirements_dict = reg.get_requirements()
        for req_id, req in requirements_dict.items():
            mandatory = "🔴" if req.mandatory else "🟡"
            click.echo(f"  {mandatory} {req_id}: {req.description}")
            click.echo(f"    Reference: {req.article_reference}")
            click.echo(f"    Categories: {', '.join(req.categories)}")
            click.echo()
    
    else:
        # Show all regulations
        click.echo("📚 Available Regulations:")
        for name, reg in regulations.items():
            click.echo(f"  • {name}: {reg.name} ({len(reg)} requirements)")
        
        click.echo("\nUse --regulation to see specific requirements")


if __name__ == '__main__':
    main()