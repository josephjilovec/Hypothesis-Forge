"""
Configuration validation for production deployment.
"""
import os
from typing import List, Dict, Any
from utils.logging_config import logger


class ConfigValidator:
    """Validate configuration for production."""

    @staticmethod
    def validate() -> tuple[bool, List[str]]:
        """
        Validate all configuration.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check required directories
        from config.config import (
            RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR,
            HYPOTHESIS_DIR, MODEL_DIR, CHECKPOINT_DIR
        )

        directories = {
            "RAW_DATA_DIR": RAW_DATA_DIR,
            "PROCESSED_DATA_DIR": PROCESSED_DATA_DIR,
            "RESULTS_DIR": RESULTS_DIR,
            "HYPOTHESIS_DIR": HYPOTHESIS_DIR,
            "MODEL_DIR": MODEL_DIR,
            "CHECKPOINT_DIR": CHECKPOINT_DIR,
        }

        for name, path in directories.items():
            try:
                path.mkdir(parents=True, exist_ok=True)
                # Test write
                test_file = path / ".write_test"
                test_file.write_text("test")
                test_file.unlink()
            except Exception as e:
                errors.append(f"Cannot write to {name} ({path}): {e}")

        # Validate environment variables
        optional_vars = [
            "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD",
            "PUBMED_API_KEY", "LOG_LEVEL"
        ]

        for var in optional_vars:
            value = os.getenv(var)
            if var == "NEO4J_PASSWORD" and os.getenv("NEO4J_URI") and not value:
                errors.append(f"{var} is required when NEO4J_URI is set")
            if var == "PUBMED_API_KEY" and not value:
                logger.warning(f"{var} not set - PubMed features will be limited")

        # Validate numeric configs
        from config.config import (
            SIMULATION_TIMEOUT, MAX_SIMULATION_RUNTIME,
            RL_LEARNING_RATE, RL_BATCH_SIZE, RL_GAMMA
        )

        if SIMULATION_TIMEOUT <= 0:
            errors.append("SIMULATION_TIMEOUT must be positive")
        if MAX_SIMULATION_RUNTIME <= 0:
            errors.append("MAX_SIMULATION_RUNTIME must be positive")
        if not 0 < RL_LEARNING_RATE < 1:
            errors.append("RL_LEARNING_RATE must be between 0 and 1")
        if RL_BATCH_SIZE <= 0:
            errors.append("RL_BATCH_SIZE must be positive")
        if not 0 < RL_GAMMA <= 1:
            errors.append("RL_GAMMA must be between 0 and 1")

        is_valid = len(errors) == 0
        return is_valid, errors

    @staticmethod
    def print_validation_report():
        """Print configuration validation report."""
        is_valid, errors = ConfigValidator.validate()

        if is_valid:
            logger.info("✓ Configuration validation passed")
        else:
            logger.error("✗ Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")

        return is_valid


def validate_config():
    """Validate configuration and exit if invalid."""
    is_valid, errors = ConfigValidator.validate()
    if not is_valid:
        logger.error("Configuration validation failed. Please fix the errors above.")
        for error in errors:
            print(f"ERROR: {error}")
        exit(1)
    return True

