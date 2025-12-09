"""
Error handling utilities for Hypothesis Forge.
Provides decorators and context managers for robust error handling.
"""
import logging
import functools
from typing import Callable, Any, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Retry decorator for functions that may fail.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Backoff multiplier for delay
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        import time
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")

            raise last_exception

        return wrapper
    return decorator


@contextmanager
def handle_errors(operation_name: str, default_return: Any = None, log_level: str = "error"):
    """
    Context manager for error handling.

    Args:
        operation_name: Name of the operation for logging
        default_return: Default value to return on error
        log_level: Logging level ('error', 'warning', 'info')
    """
    try:
        yield
    except Exception as e:
        log_func = getattr(logger, log_level, logger.error)
        log_func(f"Error in {operation_name}: {e}", exc_info=True)
        if default_return is not None:
            return default_return
        raise


def safe_execute(func: Callable, *args, default_return: Any = None, **kwargs) -> Any:
    """
    Safely execute a function with error handling.

    Args:
        func: Function to execute
        *args: Positional arguments
        default_return: Default value to return on error
        **kwargs: Keyword arguments

    Returns:
        Function result or default_return on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error executing {func.__name__}: {e}", exc_info=True)
        return default_return

