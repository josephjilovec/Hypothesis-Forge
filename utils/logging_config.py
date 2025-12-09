"""
Logging configuration for Hypothesis Forge.
Provides structured logging with file and console handlers.
"""
import logging
import sys
import os
from pathlib import Path
from typing import Optional
from loguru import logger


def setup_logging(log_level: Optional[str] = None, log_file: Optional[str] = None):
    """
    Configure logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
    """
    # Get defaults from environment if not provided
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO")
    if log_file is None:
        log_file = os.getenv("LOG_FILE")

    # Determine base directory
    base_dir = Path(__file__).parent.parent

    # Remove default handler
    logger.remove()

    # Add console handler with color
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True,
    )

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=log_level,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
        )
    else:
        # Default log file in logs directory
        logs_dir = base_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        logger.add(
            logs_dir / "hypothesis_forge.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=log_level,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
        )

    return logger


# Initialize logging with safe defaults
try:
    setup_logging()
except Exception:
    # Fallback to basic logging if setup fails
    import logging
    logging.basicConfig(level=logging.INFO)

