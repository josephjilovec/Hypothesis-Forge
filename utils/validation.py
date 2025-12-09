"""
Input validation utilities for Hypothesis Forge.
"""
from typing import Any, Dict, List, Optional
import re


def validate_hypothesis_text(text: str, max_length: int = 1000) -> tuple[bool, Optional[str]]:
    """
    Validate hypothesis text.

    Args:
        text: Hypothesis text to validate
        max_length: Maximum allowed length

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not text or not isinstance(text, str):
        return False, "Hypothesis text must be a non-empty string"

    if len(text) > max_length:
        return False, f"Hypothesis text exceeds maximum length of {max_length} characters"

    if len(text.strip()) < 10:
        return False, "Hypothesis text is too short (minimum 10 characters)"

    # Check for potentially malicious content
    dangerous_patterns = [
        r'<script',
        r'javascript:',
        r'onerror=',
        r'onload=',
    ]
    for pattern in dangerous_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False, f"Hypothesis text contains potentially dangerous content: {pattern}"

    return True, None


def validate_simulation_params(params: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate simulation parameters.

    Args:
        params: Simulation parameters dictionary

    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = ["simulation_type"]
    for field in required_fields:
        if field not in params:
            return False, f"Missing required field: {field}"

    sim_type = params.get("simulation_type")
    if sim_type not in ("protein_folding", "gravitational_dynamics"):
        return False, f"Invalid simulation_type: {sim_type}"

    return True, None


def validate_hypothesis_params(params: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate hypothesis parameters.

    Args:
        params: Hypothesis parameters dictionary

    Returns:
        Tuple of (is_valid, error_message)
    """
    if "parameters" not in params:
        return False, "Missing 'parameters' field"

    hyp_params = params.get("parameters", {})
    if not isinstance(hyp_params, dict):
        return False, "Parameters must be a dictionary"

    # Validate parameter values are in valid range
    for key, value in hyp_params.items():
        if not isinstance(value, (int, float)):
            return False, f"Parameter {key} must be numeric"
        if abs(value) > 10.0:  # Reasonable limit
            return False, f"Parameter {key} value {value} is out of range"

    return True, None


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent path traversal and other issues.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove path components
    filename = filename.replace("/", "_").replace("\\", "_")
    # Remove dangerous characters
    filename = re.sub(r'[<>:"|?*]', '', filename)
    # Limit length
    if len(filename) > 255:
        filename = filename[:255]
    return filename

