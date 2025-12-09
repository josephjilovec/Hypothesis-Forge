"""
Monitoring utilities for production.
"""
import time
import functools
from typing import Callable, Any
from utils.logging_config import logger


def monitor_performance(func: Callable) -> Callable:
    """
    Decorator to monitor function performance.

    Args:
        func: Function to monitor

    Returns:
        Wrapped function with performance monitoring
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{func.__name__} failed after {elapsed:.2f}s: {e}")
            raise
    return wrapper


class PerformanceMonitor:
    """Performance monitoring context manager."""

    def __init__(self, operation_name: str):
        """
        Initialize performance monitor.

        Args:
            operation_name: Name of the operation being monitored
        """
        self.operation_name = operation_name
        self.start_time = None

    def __enter__(self):
        """Start monitoring."""
        self.start_time = time.time()
        logger.debug(f"Starting {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End monitoring and log results."""
        elapsed = time.time() - self.start_time
        if exc_type is None:
            logger.info(f"{self.operation_name} completed in {elapsed:.2f}s")
        else:
            logger.error(f"{self.operation_name} failed after {elapsed:.2f}s: {exc_val}")
        return False  # Don't suppress exceptions


def track_memory_usage(func: Callable) -> Callable:
    """
    Decorator to track memory usage (if psutil available).

    Args:
        func: Function to monitor

    Returns:
        Wrapped function with memory tracking
    """
    try:
        import psutil
        import os
        PSUTIL_AVAILABLE = True
    except ImportError:
        PSUTIL_AVAILABLE = False

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            result = func(*args, **kwargs)
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_delta = mem_after - mem_before
            logger.debug(f"{func.__name__} memory: {mem_before:.2f}MB -> {mem_after:.2f}MB (Δ{mem_delta:+.2f}MB)")
        else:
            result = func(*args, **kwargs)
        return result
    return wrapper

