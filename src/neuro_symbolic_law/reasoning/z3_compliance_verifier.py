"""
Z3ComplianceVerifier: verifies legal compliance rules using Z3 SMT solver.

Consumes EntityProperties objects (produced by GNNPropertyExtractor) and
encodes concrete GDPR / AI Act rules as Z3 formulas, then checks them.

Each rule returns a VerificationResult describing:
  - whether the rule is satisfied
  - a counter-example if violated (the minimal assignment that causes the violation)
  - the Z3 formula used
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from z3 import (
        Bool, BoolVal, And, Or, Not, Implies, Solver, sat, unsat, unknown, is_true, is_false,
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False

from ..gnn.property_extractor import EntityProperties

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    rule_id: str
    rule_description: str
    satisfied: bool
    confidence: float = 1.0
    counter_example: Optional[Dict[str, bool]] = None
    z3_formula: str = ""
    error: Optional[str] = None


@dataclass
class ComplianceReport:
    entity_id: str
    entity_name: str
    results: List[VerificationResult] = field(default_factory=list)

    @property
    def compliant(self) -> bool:
        return all(r.satisfied for r in self.results)

    @property
    def violations(self) -> List[VerificationResult]:
        return [r for r in self.results if not r.satisfied]

    def summary(self) -> str:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.satisfied)
        return (
            f"Entity '{self.entity_name}': {passed}/{total} rules passed"
            + (" ✓" if self.compliant else f" ✗ — {len(self.violations)} violation(s)")
        )


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

class Z3ComplianceVerifier:
    """
    Verifies GDPR and AI Act rules for a set of entities using Z3.

    Usage::

        verifier = Z3ComplianceVerifier()
        reports = verifier.verify_all(entity_properties_list)
        for report in reports:
            print(report.summary())
    """

    THRESHOLD = 0.5  # confidence threshold for treating a float score as True

    def verify_all(self, entities: List[EntityProperties]) -> List[ComplianceReport]:
        return [self.verify_entity(e) for e in entities]

    def verify_entity(self, props: EntityProperties) -> ComplianceReport:
        report = ComplianceReport(entity_id=props.entity_id, entity_name=props.entity_name)

        # Only apply data-related rules to data_category / party nodes
        if props.entity_type in ("data_category", "party"):
            report.results += [
                self._gdpr_consent_or_lawful_basis(props),
                self._gdpr_data_retention(props),
                self._gdpr_security_measures(props),
            ]
            if props.as_bool("is_personal_data"):
                report.results.append(self._gdpr_cross_border_transfer(props))
            if props.as_bool("is_sensitive_data"):
                report.results.append(self._gdpr_sensitive_data_explicit_consent(props))

        # AI Act rules for system / party nodes that may be AI systems
        if props.entity_type in ("system", "party"):
            report.results.append(self._ai_act_risk_classification(props))
            if props.as_bool("is_high_risk_ai"):
                report.results += [
                    self._ai_act_human_oversight(props),
                    self._ai_act_transparency(props),
                ]

        return report

    # -----------------------------------------------------------------------
    # GDPR rules
    # -----------------------------------------------------------------------

    def _gdpr_consent_or_lawful_basis(self, props: EntityProperties) -> VerificationResult:
        """
        GDPR Art. 6 — Processing must have a lawful basis.
        Rule: is_personal_data → has_consent
        (simplified: consent stands in for any lawful basis)
        """
        rule_id = "GDPR-Art6-LawfulBasis"
        desc = "Personal data processing requires a lawful basis (GDPR Art. 6)"

        if not Z3_AVAILABLE:
            return self._fallback_rule(rule_id, desc, props,
                                       lambda p: not p.as_bool("is_personal_data") or p.as_bool("has_consent"))

        is_pd = BoolVal(props.as_bool("is_personal_data"))
        has_c = BoolVal(props.as_bool("has_consent"))
        formula = Implies(is_pd, has_c)
        formula_str = f"Implies(is_personal_data={props.as_bool('is_personal_data')}, has_consent={props.as_bool('has_consent')})"

        return self._check_formula(rule_id, desc, formula, formula_str, props)

    def _gdpr_data_retention(self, props: EntityProperties) -> VerificationResult:
        """
        GDPR Art. 5(1)(e) — Storage limitation.
        Rule: is_personal_data → has_retention_limit
        """
        rule_id = "GDPR-Art5e-Retention"
        desc = "Personal data must not be kept longer than necessary (GDPR Art. 5(1)(e))"

        if not Z3_AVAILABLE:
            return self._fallback_rule(rule_id, desc, props,
                                       lambda p: not p.as_bool("is_personal_data") or p.as_bool("has_retention_limit"))

        is_pd = BoolVal(props.as_bool("is_personal_data"))
        has_r = BoolVal(props.as_bool("has_retention_limit"))
        formula = Implies(is_pd, has_r)
        formula_str = f"Implies(is_personal_data={props.as_bool('is_personal_data')}, has_retention_limit={props.as_bool('has_retention_limit')})"

        return self._check_formula(rule_id, desc, formula, formula_str, props)

    def _gdpr_security_measures(self, props: EntityProperties) -> VerificationResult:
        """
        GDPR Art. 32 — Security of processing.
        Rule: is_personal_data → has_security_measure
        """
        rule_id = "GDPR-Art32-Security"
        desc = "Processing of personal data requires appropriate security measures (GDPR Art. 32)"

        if not Z3_AVAILABLE:
            return self._fallback_rule(rule_id, desc, props,
                                       lambda p: not p.as_bool("is_personal_data") or p.as_bool("has_security_measure"))

        is_pd = BoolVal(props.as_bool("is_personal_data"))
        has_s = BoolVal(props.as_bool("has_security_measure"))
        formula = Implies(is_pd, has_s)
        formula_str = f"Implies(is_personal_data={props.as_bool('is_personal_data')}, has_security_measure={props.as_bool('has_security_measure')})"

        return self._check_formula(rule_id, desc, formula, formula_str, props)

    def _gdpr_cross_border_transfer(self, props: EntityProperties) -> VerificationResult:
        """
        GDPR Art. 46 — Transfers to third countries.
        Rule: is_personal_data ∧ ¬has_transfer_safeguard → VIOLATION
        i.e. is_personal_data → has_transfer_safeguard
        """
        rule_id = "GDPR-Art46-CrossBorderTransfer"
        desc = "International transfers of personal data require adequate safeguards (GDPR Art. 46)"

        if not Z3_AVAILABLE:
            return self._fallback_rule(rule_id, desc, props,
                                       lambda p: not p.as_bool("is_personal_data") or p.as_bool("has_transfer_safeguard"))

        is_pd = BoolVal(props.as_bool("is_personal_data"))
        has_ts = BoolVal(props.as_bool("has_transfer_safeguard"))
        formula = Implies(is_pd, has_ts)
        formula_str = f"Implies(is_personal_data={props.as_bool('is_personal_data')}, has_transfer_safeguard={props.as_bool('has_transfer_safeguard')})"

        return self._check_formula(rule_id, desc, formula, formula_str, props)

    def _gdpr_sensitive_data_explicit_consent(self, props: EntityProperties) -> VerificationResult:
        """
        GDPR Art. 9 — Special categories of personal data.
        Rule: is_sensitive_data → has_consent (explicit consent required)
        """
        rule_id = "GDPR-Art9-SensitiveData"
        desc = "Processing of special category data requires explicit consent (GDPR Art. 9)"

        if not Z3_AVAILABLE:
            return self._fallback_rule(rule_id, desc, props,
                                       lambda p: not p.as_bool("is_sensitive_data") or p.as_bool("has_consent"))

        is_s = BoolVal(props.as_bool("is_sensitive_data"))
        has_c = BoolVal(props.as_bool("has_consent"))
        formula = Implies(is_s, has_c)
        formula_str = f"Implies(is_sensitive_data={props.as_bool('is_sensitive_data')}, has_consent={props.as_bool('has_consent')})"

        return self._check_formula(rule_id, desc, formula, formula_str, props)

    # -----------------------------------------------------------------------
    # AI Act rules
    # -----------------------------------------------------------------------

    def _ai_act_risk_classification(self, props: EntityProperties) -> VerificationResult:
        """
        EU AI Act Art. 6 — High-risk classification.
        Rule: is_high_risk_ai → (has_human_oversight ∧ has_transparency)
        If an entity is classified as high-risk AI, it must have both
        human oversight and transparency mechanisms.
        """
        rule_id = "AIAct-Art6-RiskClassification"
        desc = (
            "High-risk AI systems must have human oversight and transparency "
            "(EU AI Act Art. 6, 13, 14)"
        )

        if not Z3_AVAILABLE:
            return self._fallback_rule(
                rule_id, desc, props,
                lambda p: not p.as_bool("is_high_risk_ai") or (
                    p.as_bool("has_human_oversight") and p.as_bool("has_transparency")
                ),
            )

        is_hr = BoolVal(props.as_bool("is_high_risk_ai"))
        has_ho = BoolVal(props.as_bool("has_human_oversight"))
        has_tr = BoolVal(props.as_bool("has_transparency"))
        formula = Implies(is_hr, And(has_ho, has_tr))
        formula_str = (
            f"Implies(is_high_risk_ai={props.as_bool('is_high_risk_ai')}, "
            f"And(has_human_oversight={props.as_bool('has_human_oversight')}, "
            f"has_transparency={props.as_bool('has_transparency')}))"
        )

        return self._check_formula(rule_id, desc, formula, formula_str, props)

    def _ai_act_human_oversight(self, props: EntityProperties) -> VerificationResult:
        """
        EU AI Act Art. 14 — Human oversight for high-risk AI.
        Rule (separate from classification): is_high_risk_ai → has_human_oversight
        """
        rule_id = "AIAct-Art14-HumanOversight"
        desc = "High-risk AI systems must allow effective human oversight (EU AI Act Art. 14)"

        if not Z3_AVAILABLE:
            return self._fallback_rule(rule_id, desc, props,
                                       lambda p: not p.as_bool("is_high_risk_ai") or p.as_bool("has_human_oversight"))

        is_hr = BoolVal(props.as_bool("is_high_risk_ai"))
        has_ho = BoolVal(props.as_bool("has_human_oversight"))
        formula = Implies(is_hr, has_ho)
        formula_str = f"Implies(is_high_risk_ai={props.as_bool('is_high_risk_ai')}, has_human_oversight={props.as_bool('has_human_oversight')})"

        return self._check_formula(rule_id, desc, formula, formula_str, props)

    def _ai_act_transparency(self, props: EntityProperties) -> VerificationResult:
        """
        EU AI Act Art. 13 — Transparency for high-risk AI.
        Rule: is_high_risk_ai → has_transparency
        """
        rule_id = "AIAct-Art13-Transparency"
        desc = "High-risk AI systems must be transparent to users (EU AI Act Art. 13)"

        if not Z3_AVAILABLE:
            return self._fallback_rule(rule_id, desc, props,
                                       lambda p: not p.as_bool("is_high_risk_ai") or p.as_bool("has_transparency"))

        is_hr = BoolVal(props.as_bool("is_high_risk_ai"))
        has_tr = BoolVal(props.as_bool("has_transparency"))
        formula = Implies(is_hr, has_tr)
        formula_str = f"Implies(is_high_risk_ai={props.as_bool('is_high_risk_ai')}, has_transparency={props.as_bool('has_transparency')})"

        return self._check_formula(rule_id, desc, formula, formula_str, props)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _check_formula(
        self,
        rule_id: str,
        desc: str,
        formula,
        formula_str: str,
        props: EntityProperties,
    ) -> VerificationResult:
        """Check a Z3 formula. Satisfied = formula is a tautology (always true)."""
        try:
            solver = Solver()
            # Check if the negation is satisfiable (i.e. there's a counter-example)
            solver.add(Not(formula))
            result = solver.check()

            if result == unsat:
                # No counter-example found → formula is always true
                return VerificationResult(
                    rule_id=rule_id,
                    rule_description=desc,
                    satisfied=True,
                    confidence=1.0,
                    z3_formula=formula_str,
                )
            elif result == sat:
                # Counter-example exists — the formula can be violated
                model = solver.model()
                ce = {}
                for d in model.decls():
                    v = model[d]
                    ce[str(d.name())] = is_true(v)
                return VerificationResult(
                    rule_id=rule_id,
                    rule_description=desc,
                    satisfied=False,
                    confidence=1.0,
                    counter_example=ce,
                    z3_formula=formula_str,
                )
            else:
                return VerificationResult(
                    rule_id=rule_id, rule_description=desc,
                    satisfied=False, confidence=0.5,
                    error="Z3 returned unknown", z3_formula=formula_str,
                )
        except Exception as exc:
            logger.exception("Z3 error for rule %s", rule_id)
            return VerificationResult(
                rule_id=rule_id, rule_description=desc,
                satisfied=False, confidence=0.0,
                error=str(exc), z3_formula=formula_str,
            )

    @staticmethod
    def _fallback_rule(rule_id: str, desc: str, props: EntityProperties, check_fn) -> VerificationResult:
        """Pure Python fallback when Z3 is not available."""
        satisfied = check_fn(props)
        return VerificationResult(
            rule_id=rule_id,
            rule_description=desc,
            satisfied=satisfied,
            confidence=0.9,
            z3_formula="(pure-python fallback — Z3 not available)",
        )
