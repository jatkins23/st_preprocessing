"""
Core data models for geocoding and normalization operations.

These immutable, frozen dataclasses serve as the contract between
different components of the geocoding pipeline.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, List
from enum import StrEnum
from pathlib import Path
import os
from typing import Dict
from hashlib import sha256


def _try_import(module_name: str):
    try:
        return __import__(module_name)
    except Exception:
        return None


def compute_intersection_hash(streets: List[str], borough: str) -> str:
    """
    Compute normalized intersection hash from streets and borough.
    
    Streets are first normalized (spelling, abbreviations, ordinals, directionals)
    and then sorted alphabetically for determinism. All variations of the same
    intersection produce the same intersection_hash.
    
    Args:
        streets: List of street names (any case, any order)
        borough: Borough name
        
    Returns:
        Normalized intersection hash (string starting with "loc_")
        
    Examples:
        compute_intersection_hash(["Broadway", "5th Ave"], "Manhattan")
        → "loc_a3f2b8c9e1d5..."
        
        compute_intersection_hash(["5th Ave", "Broadway"], "Manhattan")  # same result!
        → "loc_a3f2b8c9e1d5..."
    """
    # Lazy imports to avoid circular dependencies at module import time
    from .normalizers import StandardTokenizer, BoroughNormalizer

    street_normalizer = StandardTokenizer()
    borough_normalizer = BoroughNormalizer()

    normalized_streets = []
    for s in streets:
        if not s:
            continue
        norm = street_normalizer.normalize(s)
        if norm:
            normalized_streets.append(norm)

    normalized_borough = borough_normalizer.normalize(borough) or borough

    normalized = sorted([s.upper().strip() for s in normalized_streets if s.strip()])
    boro_key = normalized_borough.upper().strip()
    key = "|".join(normalized) + "|" + boro_key
    hash_value = sha256(key.encode()).hexdigest()[:12]
    return f"loc_{hash_value}"


# using beautiful composition pattern instead of hierachical inheritance
class GeocodeSource(StrEnum):
    """Source of a geocoded result."""
    INTERSECTION = "intersection"
    SEARCH = "search"
    ADDRESS = "address"
    CACHE = "cache"
    NONE = "none"


class GeocodeStatus(StrEnum):
    """Status of a geocode request."""
    OK = "ok"
    NOT_FOUND = "not_found"
    INVALID_INPUT = "invalid_input"
    API_ERROR = "api_error"
    EXCEPTION = "exception"
    CACHED = "cached"


@dataclass(frozen=True)
class GeocodeRequest:
    """
    A single geocoding request.
    
    Represents a request to find coordinates for a street intersection
    or address in a specific borough.
    
    Design for multi-street intersections:
    - Store ALL streets (all_streets) representing the actual intersection
    - Primary streets (street1, street2) used for API queries
    - Database row links all variations to single normalized intersection_hash
    - All variations of same intersection share same intersection_hash
    
    Example:
    - "Broadway & 5th Ave"          → intersection_hash=loc_123
    - "5th Ave & Park Ave"          → intersection_hash=loc_123  
    - "Broadway & Park Ave"         → intersection_hash=loc_123
    All map to same (lat, lon) via primary streets "Broadway & 5th Ave"
    """
    unique_key: str
    street1: str  
    street2: str  
    borough: str
    intersection_hash: str  # Normalized hash linking all variations
    all_streets: List[str] = field(default_factory=list)  # All streets in intersection
    
    def is_valid(self) -> bool:
        """Check if request has minimum required fields."""
        return bool(
            self.unique_key.strip() and
            self.street1.strip() and
            self.street2.strip() and
            self.borough.strip() and
            self.intersection_hash.strip()
        )
    
    def streets_for_query(self) -> tuple[str, str]:
        """Return the primary pair for API geocoding query."""
        return (self.street1, self.street2)
    
    def all_streets_normalized(self) -> List[str]:
        """Return all streets (useful for logging/audit)."""
        if self.all_streets:
            return self.all_streets
        return [self.street1, self.street2]


@dataclass(frozen=True)
class GeocodeError:
    """Details of an API error during geocoding."""
    endpoint: str
    http_status: Optional[int] = None
    params_json: str = ""
    body_snippet: str = ""
    error_label: str = ""
    api_message: Optional[str] = None


@dataclass(frozen=True)
class GeocodeResult:
    """
    The result of a geocoding operation.
    
    Immutable result containing coordinates, metadata, and any errors
    encountered during the geocoding process.
    """
    unique_key: str
    intersection_hash: str
    street1: str = ""  # Store request details for error logging
    street2: str = ""
    borough: str = ""
    lon: Optional[float] = None
    lat: Optional[float] = None
    source: GeocodeSource = GeocodeSource.NONE
    status: GeocodeStatus = GeocodeStatus.NOT_FOUND
    errors: List[GeocodeError] = field(default_factory=list)
    
    def is_success(self) -> bool:
        """
        Check if geocoding was successful.
        
        Validates that both coordinates exist and are within valid geographic ranges.
        """
        if self.lon is None or self.lat is None:
            return False
        try:
            return -90 <= self.lat <= 90 and -180 <= self.lon <= 180
        except (TypeError, ValueError):
            return False
    
    def distance_to(self, other: "GeocodeResult", use_km: bool = False) -> Optional[float]:
        """
        Calculate distance to another result using Haversine formula.
        
        Args:
            other: Another GeocodeResult to measure distance to
            use_km: If True, return distance in km; else meters
            
        Returns:
            Distance in meters (or km if use_km=True), or None if coordinates missing
        """
        if not (self.is_success() and other.is_success()):
            return None
        
        # At this point, both lat/lon are not None
        assert self.lat is not None and self.lon is not None
        assert other.lat is not None and other.lon is not None
        
        # Haversine formula
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371000  # Earth radius in meters
        
        lat1, lon1 = radians(self.lat), radians(self.lon)
        lat2, lon2 = radians(other.lat), radians(other.lon)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        distance = R * c  # in meters
        return distance / 1000 if use_km else distance
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "unique_key": self.unique_key,
            "intersection_hash": self.intersection_hash,
            "lon": self.lon,
            "lat": self.lat,
            "source": self.source.value,
            "status": self.status.value,
            "errors": [
                {
                    "endpoint": e.endpoint,
                    "http_status": e.http_status,
                    "params_json": e.params_json,
                    "body_snippet": e.body_snippet,
                    "error_label": e.error_label,
                    "api_message": e.api_message,
                }
                for e in self.errors
            ],
        }


@dataclass(frozen=True)
class NormalizationRequest:
    """A request to normalize street or borough names."""
    value: str
    context: Optional[str] = None  # e.g., borough name for street context


@dataclass(frozen=True)
class NormalizationResult:
    """Result of normalization."""
    original: str
    normalized: str
    source: str = "local"  # "local" or "api"
    confidence: float = 1.0


@dataclass
class GeocodingPipelineConfig:
    """Configuration for the geocoding pipeline."""
    # API settings
    api_key: Optional[str] = None
    api_base_url: str = "https://api.nyc.gov/geoclient/v2"
    api_timeout: float = 10.0
    
    # Rate limiting
    requests_per_second: Optional[float] = None
    max_workers: int = 16
    
    # Persistence
    ok_db_path: Optional[Path] = None
    err_db_path: Optional[Path] = None
    cache_path: Optional[Path] = None
    
    # Caching
    cache_invalid: bool = False
    
    # Batch processing
    batch_size: int = 500
    
    # Normalization
    use_api_normalization: str = "auto"  # "auto", "always", "never"
    
    # Error handling
    fail_on_invalid: bool = False
    invalid_sample_size: int = 8

    # --- Environment helpers -------------------------------------------------
    @staticmethod
    def _env_var_candidates() -> List[str]:
        """Return prioritized list of environment variable names to check for API key."""
        return [
            "STREETTRANSFORMER_GEOCODING_API_KEY",
            "NYC_GEOCODING_API_KEY",
            "GEOCODING_API_KEY",
            "GECLIENT_API_KEY",
        ]

    @staticmethod
    def _read_key_from_file(path: str) -> Optional[str]:
        try:
            with open(path, "r", encoding="utf8") as fh:
                return fh.read().strip()
        except Exception:
            return None

    @staticmethod
    def _try_keyring(service: str = "streetTransformer", username: str = "geocoding") -> Optional[str]:
        """Try to read the secret from the system keyring if `keyring` package is available."""
        keyring = _try_import("keyring")
        if keyring is None:
            return None
        try:
            # keyring.get_password may raise if backend misconfigured
            return keyring.get_password(service, username)
        except Exception:
            return None

    @classmethod
    def from_env(cls, **overrides: Any) -> "GeocodingPipelineConfig":
        """Create a config instance using environment secrets and optional overrides.

        Priority for api_key resolution:
        1. explicit `api_key` passed in `overrides`
        2. environment variables (candidates returned by `_env_var_candidates`)
        3. file path in env `GEOCODING_API_KEY_FILE`
        4. system keyring (if available)
        5. None

        The method does not create external dependencies: it only uses `keyring` or
        `python-dotenv` if they are present; failure to import those packages is
        handled gracefully.
        """
        cfg_values: Dict[str, Any] = {}
        # Start from defaults declared on the dataclass
        for f in cls.__dataclass_fields__:  # type: ignore[attr-defined]
            cfg_values[f] = getattr(cls(), f)

        # Apply any overrides provided programmatically
        cfg_values.update(overrides)

        # 1) explicit override
        if cfg_values.get("api_key"):
            return cls(**cfg_values)

        # 2) environment variables
        for name in cls._env_var_candidates():
            val = os.getenv(name)
            if val:
                cfg_values["api_key"] = val.strip()
                return cls(**cfg_values)

        # 3) file path specified in env
        key_file = os.getenv("GEOCODING_API_KEY_FILE")
        if key_file:
            key_from_file = cls._read_key_from_file(key_file)
            if key_from_file:
                cfg_values["api_key"] = key_from_file
                return cls(**cfg_values)

        # 4) system keyring
        kr = cls._try_keyring()
        if kr:
            cfg_values["api_key"] = kr
            return cls(**cfg_values)

        # 5) attempt .env via python-dotenv if present
        dotenv = _try_import("dotenv")
        if dotenv is not None:
            try:
                # load_dotenv returns True/False but also populates os.environ
                dotenv.load_dotenv(override=False)
                for name in cls._env_var_candidates():
                    val = os.getenv(name)
                    if val:
                        cfg_values["api_key"] = val.strip()
                        return cls(**cfg_values)
            except Exception:
                # Not critical - continue
                pass

        # Nothing found — return config with api_key left as None
        return cls(**cfg_values)

    def get_api_key(self) -> Optional[str]:
        """Return the resolved api_key for runtime use.

        If `api_key` is already set on the instance that value is returned. If not,
        the method will attempt to read env variables, file, or keyring as a
        last resort. This mirrors `from_env` resolution but is instance-bound.
        """
        if self.api_key:
            return self.api_key

        # check envs
        for name in self._env_var_candidates():
            val = os.getenv(name)
            if val:
                return val.strip()

        key_file = os.getenv("GEOCODING_API_KEY_FILE")
        if key_file:
            val = self._read_key_from_file(key_file)
            if val:
                return val

        kr = self._try_keyring()
        if kr:
            return kr

        # attempt to use dotenv if available
        dotenv = _try_import("dotenv")
        if dotenv is not None:
            try:
                dotenv.load_dotenv(override=False)
                for name in self._env_var_candidates():
                    val = os.getenv(name)
                    if val:
                        return val.strip()
            except Exception:
                pass

        return None
