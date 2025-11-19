"""
Rate limiting implementations for controlling API request rates.

Provides thread-safe rate limiters to ensure compliance with API
rate limits and avoid overwhelming services.
"""

from __future__ import annotations

import time
import random
import threading
from typing import Optional

from .base import RateLimiter


class TokenBucket(RateLimiter):
    """
    Token bucket rate limiter.
    
    Maintains a bucket of "tokens" that refill at a specified rate
    (requests per second). Each request consumes one token. When the
    bucket is empty, requests wait until tokens become available.
    
    Thread-safe implementation using a lock.
    """
    
    def __init__(
        self,
        rate: float,
        capacity: Optional[int] = None,
        jitter: bool = False,
    ):
        """
        Initialize token bucket.
        
        Args:
            rate: Tokens per second (i.e., requests per second allowed)
            capacity: Maximum bucket size (defaults to 2x rate)
            jitter: If True, add small random delays to avoid synchronized requests
        """
        if rate <= 0:
            raise ValueError("rate must be > 0")
        
        self.rate = float(rate)
        self.capacity = capacity or max(1, int(rate * 2))
        self.tokens = float(self.capacity)
        self.last_refill = time.perf_counter()
        self.lock = threading.Lock()
        self.jitter = jitter
    
    def wait(self) -> None:
        """Block until it's safe to make a request."""
        self.acquire(1)
    
    def acquire(self, count: int = 1) -> None:
        """
        Acquire one or more tokens.
        
        Blocks until the requested number of tokens are available.
        
        Args:
            count: Number of tokens to acquire
        """
        if count <= 0:
            raise ValueError("count must be > 0")
        
        while True:
            with self.lock:
                now = time.perf_counter()
                elapsed = now - self.last_refill
                self.last_refill = now
                
                # Add new tokens
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                
                # Check if we have enough
                if self.tokens >= count:
                    self.tokens -= count
                    if self.jitter:
                        jitter = random.random() * 0.002
                        time.sleep(jitter)
                    return
            
            # Sleep briefly before retrying
            time.sleep(0.005)


class SimpleRateGate(RateLimiter):
    """
    Simple rate gate with fixed inter-request delay.
    
    Less sophisticated than token bucket but simpler and useful
    for low-volume rate limiting.
    """
    
    def __init__(self, requests_per_second: float):
        """
        Initialize rate gate.
        
        Args:
            requests_per_second: Target rate (requests per second)
        """
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be > 0")
        
        self.dt = 1.0 / float(requests_per_second)
        self.next_time = time.perf_counter()
        self.lock = threading.Lock()
    
    def wait(self) -> None:
        """Block until it's safe to make a request."""
        self.acquire(1)
    
    def acquire(self, count: int = 1) -> None:
        """
        Acquire one or more request slots.
        
        Args:
            count: Number of requests to acquire
        """
        with self.lock:
            now = time.perf_counter()
            delay_needed = self.next_time - now
            if delay_needed > 0:
                time.sleep(delay_needed)
                now = time.perf_counter()
            self.next_time = now + (self.dt * count)


class NoOpRateLimiter(RateLimiter):
    """
    Rate limiter that does nothing (for testing/development).
    
    Useful when you want to disable rate limiting without changing code.
    """
    
    def wait(self) -> None:
        """Do nothing."""
        pass
    
    def acquire(self, count: int = 1) -> None:
        """Do nothing."""
        pass


class AdaptiveRateLimiter(RateLimiter):
    """
    Adaptive rate limiter that adjusts based on response times.
    
    If responses are slow, automatically reduces request rate.
    If responses are fast, gradually increases request rate.
    """
    
    def __init__(
        self,
        initial_rate: float,
        min_rate: float = 0.5,
        max_rate: float = 100.0,
    ):
        """
        Initialize adaptive rate limiter.
        
        Args:
            initial_rate: Starting requests per second
            min_rate: Minimum requests per second to allow
            max_rate: Maximum requests per second to allow
        """
        self.current_rate = float(initial_rate)
        self.min_rate = float(min_rate)
        self.max_rate = float(max_rate)
        self.bucket = TokenBucket(self.current_rate)
        self.last_response_time = time.perf_counter()
        self.lock = threading.Lock()
    
    def wait(self) -> None:
        """Wait and potentially adjust rate based on response time."""
        now = time.perf_counter()
        response_time = now - self.last_response_time
        self.last_response_time = now
        
        # Adjust rate based on response time
        # If responses take > 1 second, reduce rate
        # If responses are < 100ms, increase rate
        with self.lock:
            if response_time > 1.0:
                self.current_rate = max(self.min_rate, self.current_rate * 0.9)
                self.bucket = TokenBucket(self.current_rate)
            elif response_time < 0.1:
                self.current_rate = min(self.max_rate, self.current_rate * 1.1)
                self.bucket = TokenBucket(self.current_rate)
        
        self.bucket.wait()
    
    def acquire(self, count: int = 1) -> None:
        """Acquire tokens with rate adjustment."""
        self.bucket.acquire(count)
