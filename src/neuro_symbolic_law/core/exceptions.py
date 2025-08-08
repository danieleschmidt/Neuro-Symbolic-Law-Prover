"""
Custom exceptions for Neuro-Symbolic Law Prover.
Provides specific error types for different failure modes.
"""

from typing import Optional, Dict, Any, List


class NeuroSymbolicLawError(Exception):
    """Base exception for all Neuro-Symbolic Law Prover errors."""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "NSL_GENERAL_ERROR"
        self.details = details or {}


class ContractParsingError(NeuroSymbolicLawError):
    """Raised when contract parsing fails."""
    
    def __init__(self, message: str, contract_text: str = None, parsing_stage: str = None):
        super().__init__(message, "NSL_PARSING_ERROR", {
            "contract_length": len(contract_text) if contract_text else 0,
            "parsing_stage": parsing_stage
        })
        self.contract_text = contract_text
        self.parsing_stage = parsing_stage


class ComplianceVerificationError(NeuroSymbolicLawError):
    """Raised when compliance verification fails."""
    
    def __init__(self, message: str, regulation_name: str = None, requirement_id: str = None):
        super().__init__(message, "NSL_VERIFICATION_ERROR", {
            "regulation_name": regulation_name,
            "requirement_id": requirement_id
        })
        self.regulation_name = regulation_name
        self.requirement_id = requirement_id


class RegulationError(NeuroSymbolicLawError):
    """Raised when regulation model has issues."""
    
    def __init__(self, message: str, regulation_name: str = None):
        super().__init__(message, "NSL_REGULATION_ERROR", {
            "regulation_name": regulation_name
        })
        self.regulation_name = regulation_name


class ModelLoadingError(NeuroSymbolicLawError):
    """Raised when neural models fail to load."""
    
    def __init__(self, message: str, model_name: str = None, model_type: str = None):
        super().__init__(message, "NSL_MODEL_ERROR", {
            "model_name": model_name,
            "model_type": model_type
        })
        self.model_name = model_name
        self.model_type = model_type


class Z3EncodingError(NeuroSymbolicLawError):
    """Raised when Z3 encoding fails."""
    
    def __init__(self, message: str, constraint_name: str = None):
        super().__init__(message, "NSL_Z3_ERROR", {
            "constraint_name": constraint_name
        })
        self.constraint_name = constraint_name


class ValidationError(NeuroSymbolicLawError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field_name: str = None, field_value: Any = None):
        super().__init__(message, "NSL_VALIDATION_ERROR", {
            "field_name": field_name,
            "field_value": str(field_value) if field_value is not None else None
        })
        self.field_name = field_name
        self.field_value = field_value


class SecurityError(NeuroSymbolicLawError):
    """Raised when security validation fails."""
    
    def __init__(self, message: str, security_type: str = None):
        super().__init__(message, "NSL_SECURITY_ERROR", {
            "security_type": security_type
        })
        self.security_type = security_type


class ConfigurationError(NeuroSymbolicLawError):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, config_key: str = None):
        super().__init__(message, "NSL_CONFIG_ERROR", {
            "config_key": config_key
        })
        self.config_key = config_key


class ResourceError(NeuroSymbolicLawError):
    """Raised when system resources are insufficient."""
    
    def __init__(self, message: str, resource_type: str = None):
        super().__init__(message, "NSL_RESOURCE_ERROR", {
            "resource_type": resource_type
        })
        self.resource_type = resource_type


class ExplanationError(NeuroSymbolicLawError):
    """Raised when explanation generation fails."""
    
    def __init__(self, message: str, explanation_type: str = None):
        super().__init__(message, "NSL_EXPLANATION_ERROR", {
            "explanation_type": explanation_type
        })
        self.explanation_type = explanation_type


def handle_exception_gracefully(func):
    """Decorator to handle exceptions gracefully with fallbacks."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NeuroSymbolicLawError:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            # Wrap unknown exceptions
            raise NeuroSymbolicLawError(
                f"Unexpected error in {func.__name__}: {str(e)}",
                "NSL_UNEXPECTED_ERROR",
                {"function": func.__name__, "original_error": str(e)}
            )
    
    return wrapper


def validate_contract_text(contract_text: str) -> None:
    """Validate contract text input."""
    if not isinstance(contract_text, str):
        raise ValidationError("Contract text must be a string", "contract_text", type(contract_text).__name__)
    
    if not contract_text.strip():
        raise ValidationError("Contract text cannot be empty", "contract_text", "empty")
    
    if len(contract_text) < 10:
        raise ValidationError("Contract text is too short", "contract_text", len(contract_text))
    
    if len(contract_text) > 10_000_000:  # 10MB limit
        raise ValidationError("Contract text is too large", "contract_text", len(contract_text))
    
    # Check for potential security issues
    suspicious_patterns = [
        '<?php', '<%', '<script', 'javascript:', 'vbscript:',
        'onload=', 'onerror=', 'eval(', 'exec('
    ]
    
    contract_lower = contract_text.lower()
    for pattern in suspicious_patterns:
        if pattern in contract_lower:
            raise SecurityError(
                f"Suspicious pattern detected in contract text: {pattern}", 
                "malicious_content"
            )


def validate_contract_id(contract_id: str) -> None:
    """Validate contract ID."""
    if not isinstance(contract_id, str):
        raise ValidationError("Contract ID must be a string", "contract_id", type(contract_id).__name__)
    
    if not contract_id.strip():
        raise ValidationError("Contract ID cannot be empty", "contract_id", "empty")
    
    if len(contract_id) > 255:
        raise ValidationError("Contract ID is too long", "contract_id", len(contract_id))
    
    # Only allow alphanumeric, underscore, hyphen, and period
    import re
    if not re.match(r'^[a-zA-Z0-9_.-]+$', contract_id):
        raise ValidationError(
            "Contract ID contains invalid characters", 
            "contract_id", 
            contract_id
        )


def validate_focus_areas(focus_areas: List[str]) -> None:
    """Validate focus areas list."""
    if focus_areas is None:
        return  # Optional parameter
    
    if not isinstance(focus_areas, list):
        raise ValidationError("Focus areas must be a list", "focus_areas", type(focus_areas).__name__)
    
    if len(focus_areas) > 20:
        raise ValidationError("Too many focus areas", "focus_areas", len(focus_areas))
    
    valid_areas = {
        'data_processing', 'data_minimization', 'purpose_limitation', 'storage_limitation',
        'data_subject_rights', 'access_rights', 'deletion', 'rectification', 'portability',
        'security', 'technical_measures', 'organizational_measures', 'encryption',
        'transparency', 'consent', 'lawful_basis', 'privacy_by_design',
        'breach_notification', 'dpia', 'dpo', 'processor_agreements',
        'international_transfers', 'adequacy_decisions', 'safeguards'
    }
    
    for area in focus_areas:
        if not isinstance(area, str):
            raise ValidationError(f"Focus area must be string: {area}", "focus_areas", type(area).__name__)
        
        if area not in valid_areas:
            raise ValidationError(f"Invalid focus area: {area}", "focus_areas", area)


def sanitize_text_input(text: str) -> str:
    """Sanitize text input by removing dangerous patterns."""
    if not isinstance(text, str):
        return str(text)
    
    # Remove null bytes and control characters
    sanitized = ''.join(char for char in text if ord(char) >= 32 or char in ['\n', '\r', '\t'])
    
    # Limit length
    if len(sanitized) > 10_000_000:
        sanitized = sanitized[:10_000_000]
    
    return sanitized


def log_security_event(event_type: str, details: Dict[str, Any], severity: str = "INFO") -> None:
    """Log security-related events."""
    import logging
    import json
    
    security_logger = logging.getLogger("neuro_symbolic_law.security")
    
    log_entry = {
        "event_type": event_type,
        "timestamp": __import__('datetime').datetime.now().isoformat(),
        "severity": severity,
        "details": details
    }
    
    if severity == "CRITICAL":
        security_logger.critical(json.dumps(log_entry))
    elif severity == "ERROR":
        security_logger.error(json.dumps(log_entry))
    elif severity == "WARNING":
        security_logger.warning(json.dumps(log_entry))
    else:
        security_logger.info(json.dumps(log_entry))