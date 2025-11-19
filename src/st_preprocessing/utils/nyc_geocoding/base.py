"""
Abstract base classes for the geocoding system.

These define the interfaces that all concrete implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from pathlib import Path

from .models import (
    GeocodeRequest, GeocodeResult, 
    NormalizationRequest, NormalizationResult
)


class Normalizer(ABC):
    """
    Abstract base for string normalizers.
    
    Normalizers are responsible for transforming input strings into
    a normalized form (e.g., "Manhattan" → "MANHATTAN").
    """
    
    @abstractmethod
    def normalize(self, value: str, context: Optional[str] = None) -> str:
        """
        Normalize a string value.
        
        Args:
            value: String to normalize
            context: Optional context (e.g., borough name for street)
            
        Returns:
            Normalized string
        """
        pass
    
    def normalize_batch(self, values: List[str]) -> List[str]:
        """
        Normalize multiple values. Default implementation calls normalize()
        for each value, but subclasses can override for efficiency.
        
        Args:
            values: List of strings to normalize
            
        Returns:
            List of normalized strings
        """
        return [self.normalize(v) for v in values]


class Geocoder(ABC):
    """
    Abstract base for geocoders.
    
    Geocoders are responsible for finding coordinates for a given
    street intersection or address.
    """
    
    @abstractmethod
    def geocode(self, request: GeocodeRequest) -> GeocodeResult:
        """
        Geocode a single request.
        
        Args:
            request: The geocoding request
            
        Returns:
            GeocodeResult with coordinates and metadata
        """
        pass
    
    def geocode_batch(self, requests: List[GeocodeRequest]) -> List[GeocodeResult]:
        """
        Geocode multiple requests. Default implementation calls geocode()
        for each request, but subclasses can override for efficiency.
        
        Args:
            requests: List of geocoding requests
            
        Returns:
            List of GeocodeResult objects
        """
        return [self.geocode(req) for req in requests]


class StorageBackend(ABC):
    """
    Abstract base for persistence storage.
    
    Storage backends handle reading/writing geocoding results,
    caches, and error logs.
    """
    
    @abstractmethod
    def write_success(self, result: GeocodeResult) -> None:
        """Store a successful geocoding result."""
        pass
    
    @abstractmethod
    def write_batch_success(self, results: List[GeocodeResult]) -> None:
        """Store multiple successful results. Can be overridden for efficiency."""
        for result in results:
            self.write_success(result)
    
    @abstractmethod
    def write_error(self, result: GeocodeResult) -> None:
        """Store an error/failed geocoding result."""
        pass
    
    @abstractmethod
    def write_batch_error(self, results: List[GeocodeResult]) -> None:
        """Store multiple error results. Can be overridden for efficiency."""
        for result in results:
            self.write_error(result)
    
    @abstractmethod
    def read(self, unique_key: str) -> Optional[GeocodeResult]:
        """Retrieve a stored result by unique key."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close connections/cleanup resources."""
        pass


class RateLimiter(ABC):
    """
    Abstract base for rate limiters.
    
    Rate limiters control the rate at which requests are made,
    useful for respecting API rate limits.
    """
    
    @abstractmethod
    def wait(self) -> None:
        """Block until it's safe to make another request."""
        pass
    
    @abstractmethod
    def acquire(self, count: int = 1) -> None:
        """
        Acquire one or more request slots.
        
        Args:
            count: Number of requests to acquire (for bulk operations)
        """
        pass


class NormalizationCache(ABC):
    """
    Abstract base for caching normalized values.
    
    Reduces redundant API calls by caching normalization results.
    """
    
    @abstractmethod
    def get(self, key: str) -> Optional[str]:
        """Get a cached normalized value."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: str) -> None:
        """Cache a normalized value."""
        pass
    
    @abstractmethod
    def get_batch(self, keys: List[str]) -> Dict[str, Optional[str]]:
        """Get multiple cached values."""
        return {k: self.get(k) for k in keys}
    
    @abstractmethod
    def set_batch(self, items: Dict[str, str]) -> None:
        """Cache multiple values."""
        for k, v in items.items():
            self.set(k, v)
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cached values."""
        pass
