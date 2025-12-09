"""
Graceful shutdown utilities for production.
"""
import signal
import sys
import threading
from typing import List, Callable
from utils.logging_config import logger


class GracefulShutdown:
    """Handle graceful shutdown of the application."""

    def __init__(self):
        """Initialize graceful shutdown handler."""
        self.shutdown_handlers: List[Callable] = []
        self.shutdown_event = threading.Event()
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()
        self.shutdown()

    def register_handler(self, handler: Callable):
        """
        Register a shutdown handler.

        Args:
            handler: Function to call during shutdown
        """
        self.shutdown_handlers.append(handler)

    def shutdown(self):
        """Execute all shutdown handlers."""
        logger.info("Executing shutdown handlers...")
        for handler in reversed(self.shutdown_handlers):  # Reverse order
            try:
                handler()
            except Exception as e:
                logger.error(f"Error in shutdown handler: {e}")

        logger.info("Shutdown complete")
        sys.exit(0)

    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self.shutdown_event.is_set()


# Global shutdown handler
_shutdown_handler = None


def get_shutdown_handler() -> GracefulShutdown:
    """Get or create global shutdown handler."""
    global _shutdown_handler
    if _shutdown_handler is None:
        _shutdown_handler = GracefulShutdown()
    return _shutdown_handler

