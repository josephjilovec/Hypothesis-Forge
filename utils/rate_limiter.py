"""
Rate limiting utilities for API calls.
"""
import time
from typing import Dict, Optional
from collections import defaultdict
from threading import Lock


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, max_calls: int = 10, time_window: float = 60.0):
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum number of calls allowed
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls: Dict[str, list] = defaultdict(list)
        self.lock = Lock()

    def is_allowed(self, key: str = "default") -> bool:
        """
        Check if a call is allowed.

        Args:
            key: Identifier for the rate limit (e.g., API endpoint)

        Returns:
            True if allowed, False otherwise
        """
        with self.lock:
            now = time.time()
            # Remove old calls outside time window
            self.calls[key] = [
                call_time for call_time in self.calls[key]
                if now - call_time < self.time_window
            ]

            # Check if under limit
            if len(self.calls[key]) < self.max_calls:
                self.calls[key].append(now)
                return True
            return False

    def wait_if_needed(self, key: str = "default") -> float:
        """
        Wait if rate limit is exceeded.

        Args:
            key: Identifier for the rate limit

        Returns:
            Wait time in seconds (0 if no wait needed)
        """
        if self.is_allowed(key):
            return 0.0

        with self.lock:
            if self.calls[key]:
                oldest_call = min(self.calls[key])
                wait_time = self.time_window - (time.time() - oldest_call)
                if wait_time > 0:
                    time.sleep(wait_time)
                    return wait_time
        return 0.0

    def reset(self, key: Optional[str] = None):
        """Reset rate limiter for a key or all keys."""
        with self.lock:
            if key:
                self.calls.pop(key, None)
            else:
                self.calls.clear()


# Global rate limiters
pubmed_limiter = RateLimiter(max_calls=3, time_window=1.0)  # 3 calls per second
arxiv_limiter = RateLimiter(max_calls=1, time_window=3.0)  # 1 call per 3 seconds

