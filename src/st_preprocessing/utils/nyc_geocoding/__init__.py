"""
- Models: Data structures (GeocodeRequest, GeocodeResult, ...)
- Base classes: Abstract interfaces
- Normalizers: Street and borough name normalization
- Geocoders: Coordinate lookup implementations
- Throttling: Rate limiting for API calls
- Storage: Backends for storing geocoding results
"""

from .models import (
    GeocodeSource,
    GeocodeStatus,
    GeocodeRequest,
    GeocodeResult,
    GeocodeError,
    NormalizationRequest,
    NormalizationResult,
    GeocodingPipelineConfig,
    compute_intersection_hash,
)

from .base import (
    Normalizer,
    Geocoder,
    StorageBackend,
    RateLimiter,
    NormalizationCache,
)

from .normalizers import (
    StreetNormalizer,
    BoroughNormalizer,
    StandardTokenizer,
    CompositeNormalizer,
    DataPreparer,
    BOROUGH_ALIASES,
    VALID_BOROUGHS,
)

from .throttling import (
    TokenBucket,
    SimpleRateGate,
    NoOpRateLimiter,
    AdaptiveRateLimiter,
)

from .geocoders import (
    NYCGeoClient,
)

from .storage import (
    DuckDBStorageBackend,
    CSVStorageBackend,
    CompositeStorageBackend,
)

__all__ = [
    # Models
    "GeocodeSource",
    "GeocodeStatus",
    "GeocodeRequest",
    "GeocodeResult",
    "GeocodeError",
    "NormalizationRequest",
    "NormalizationResult",
    "GeocodingPipelineConfig",
    "compute_intersection_hash",
    # Base classes
    "Normalizer",
    "Geocoder",
    "StorageBackend",
    "RateLimiter",
    "NormalizationCache",
    # Normalizers
    "StreetNormalizer",
    "BoroughNormalizer",
    "StandardTokenizer",
    "CompositeNormalizer",
    "DataPreparer",
    "BOROUGH_ALIASES",
    "VALID_BOROUGHS",
    # Throttling
    "TokenBucket",
    "SimpleRateGate",
    "NoOpRateLimiter",
    "AdaptiveRateLimiter",
    # Geocoders
    "NYCGeoClient",
    # Storage
    "DuckDBStorageBackend",
    "CSVStorageBackend",
    "CompositeStorageBackend",
]

