"""
NYC Geoclient API wrapper implementing the Geocoder interface.

Use NYC Department of City Planning's Geoclient API v2 for
geocoding street intersections and addresses in New York City.

Reference: https://api.cityofnewyork.us/geoclient/v2/
"""

import json
import time
from typing import Optional, Any, List, Tuple
from urllib.parse import urljoin
import logging
import requests

from .base import Geocoder, RateLimiter
from .models import GeocodeRequest, GeocodeResult, GeocodeError, GeocodeSource, GeocodeStatus
from .normalizers import StreetNormalizer, BoroughNormalizer

from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class GeocodeRegion(ABC):
    """
    Abstract base class for geocode regions.
    
    Defines interface for region-specific geocoding behavior.
    Not finished, just a placeholder for future extension.
    """
    
    # @abstractmethod
    # def validate_request(self, request: GeocodeRequest) -> bool:
    #     """
    #     Validate if the request is valid for this region.
        
    #     Args:
    #         request: GeocodeRequest to validate
            
    #     Returns:
    #         True if valid, False otherwise
    #     """
    #     pass
    
    @abstractmethod
    def geocode(self, request: GeocodeRequest) -> GeocodeResult:
        """
        Geocode a single request for this region.
        
        Args:
            request: GeocodeRequest to geocode

        Returns:
            GeocodeResult containing the geocoded information
        """
        pass




class NYCGeoClient(Geocoder):
    """
    NYC Geoclient API wrapper.
    
    Implements the Geocoder interface for NYC Geoclient v2 API.
    Handles street intersections and addresses.
    """
    
    def __init__(
        self,
        api_key: str,
        api_base_url: str = "https://api.nyc.gov/geoclient/v2",
        timeout: float = 10.0,
        rate_limiter: Optional[RateLimiter] = None,
        max_retries: int = 3,
        retry_delay_s: float = 1.0,
    ):
        """
        Initialize NYC Geoclient wrapper.
        
        Args:
            api_key: NYC Geoclient API key
            api_base_url: Base URL for API (default: NYC production)
            timeout: HTTP request timeout in seconds
            rate_limiter: Optional rate limiter (defaults to no limit)
            max_retries: Number of retries on transient errors
            retry_delay_s: Delay between retries in seconds
        """
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.timeout = timeout
        self.rate_limiter = rate_limiter
        self.max_retries = max_retries
        self.retry_delay_s = retry_delay_s
        
        # Initialize normalizers for API parameters
        self.street_normalizer = StreetNormalizer()
        self.borough_normalizer = BoroughNormalizer()
        
        logger.info(
            f"Initialized NYCGeoClient: {api_base_url}, "
            f"retries={max_retries}, timeout={timeout}s"
        )
    
    def _extract_from_intersection(
            self, 
            response: dict[str, Any]
            ) -> Tuple[Optional[float], Optional[float], GeocodeSource, Optional[dict[str, Any]]]:
        """
        Extract coordinates from intersection response.
        
        Returns:
            Tuple of (latitude, longitude, source, all_fields)
            latitude/longitude are None if not found
        """
        # Check for intersection response (could be 'intersection' or 'intersectionResponse')
        intersection = response.get("intersection") or response.get("intersectionResponse")
        
        if not intersection:
            return None, None, GeocodeSource.NONE, None
        
        # Check for geosupport error message first
        geosupport_message = intersection.get("message")
        geosupport_return_code = intersection.get("geosupportReturnCode") or intersection.get("returnCode")
        
        # NYC returns coordinates as strings sometimes
        try:
            lat = float(intersection.get("latitude"))
            lon = float(intersection.get("longitude"))
            return lat, lon, GeocodeSource.INTERSECTION, intersection
        except (ValueError, TypeError):
            # If coordinates not available and we have a geosupport error, include it
            error_dict = {}
            if geosupport_return_code and geosupport_return_code != "00":
                if geosupport_message:
                    error_dict = {"error": geosupport_message, "returnCode": geosupport_return_code}
                else:
                    error_dict = {"returnCode": geosupport_return_code}
            elif geosupport_message:
                error_dict = {"error": geosupport_message}
            
            return None, None, GeocodeSource.NONE, error_dict if error_dict else None
    
    def _extract_from_address(self, response: dict[str, Any]) -> Tuple[Optional[float], Optional[float], GeocodeSource, Optional[dict[str, Any]]]:
        """
        Extract coordinates from address response.
        
        Returns:
            Tuple of (latitude, longitude, source, all_fields)
            latitude/longitude are None if not found
        """
        if "addressResponse" not in response:
            return None, None, GeocodeSource.NONE, None
        
        address = response.get("addressResponse", {})
        if address:
            try:
                lat = float(address.get("latitude"))
                lon = float(address.get("longitude"))
                return lat, lon, GeocodeSource.ADDRESS, address
            except (ValueError, TypeError):
                return None, None, GeocodeSource.NONE, None
        else:
            return None, None, GeocodeSource.NONE, None
    
    def _extract_error_details(
        self,
        response: dict[str, Any],
        endpoint: str,
        http_status: int,
        request_params: dict[str, str]
    ) -> List[GeocodeError]:
        """
        Extract error details from API response.
        
        Args:
            response: API response dict
            endpoint: API endpoint used
            http_status: HTTP status code
            request_params: Parameters sent to API
            
        Returns:
            List of GeocodeError objects
        """
        errors = []
        
        # Check for API-level error message
        if "message" in response:
            errors.append(GeocodeError(
                endpoint=endpoint,
                http_status=http_status,
                error_label="api_message",
                api_message=response.get("message"),
                params_json=json.dumps(request_params)[:900],  # truncate
            ))
        
        # Check for status code in response
        if "status" in response:
            status = response.get("status")
            if status not in [200, 201]:
                errors.append(GeocodeError(
                    endpoint=endpoint,
                    http_status=http_status,
                    error_label="response_status",
                    body_snippet=json.dumps(response)[:200],
                    params_json=json.dumps(request_params)[:500],
                ))
        
        return errors if errors else [GeocodeError(
            endpoint=endpoint,
            http_status=http_status,
            error_label="no_result",
            api_message="No coordinates found in response",
            params_json=json.dumps(request_params)[:500],
        )]
    
    def geocode(
        self,
        request: GeocodeRequest,
        **kwargs: Any
    ) -> GeocodeResult:
        """
        Geocode a single request.
        
        Queries NYC Geoclient API using street1 & street2 (primary pair).
        Handles retries and rate limiting.
        
        Args:
            request: GeocodeRequest with street1, street2, borough
            **kwargs: Additional arguments (unused, for interface compatibility)
            
        Returns:
            GeocodeResult with coordinates or error details
        """
        if not request.is_valid():
            return GeocodeResult(
                unique_key=request.unique_key,
                intersection_hash=request.intersection_hash,
                street1=request.street1,
                street2=request.street2,
                borough=request.borough,
                status=GeocodeStatus.INVALID_INPUT,
                errors=[GeocodeError(
                    endpoint="validation",
                    error_label="invalid_request",
                    api_message="Request missing required fields"
                )]
            )
        
        # Apply rate limiting if configured
        if self.rate_limiter:
            self.rate_limiter.wait()
        
        # Try to geocode with retries
        for attempt in range(self.max_retries):
            try:
                response = self._query_intersection(
                    street1=request.street1,
                    street2=request.street2,
                    borough=request.borough,
                )
                
                # Extract coordinates
                lat, lon, source, all_fields = self._extract_from_intersection(response)
                
                # Check if coordinates are valid using GeocodeResult's validation
                if lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180:
                    return GeocodeResult(
                        unique_key=request.unique_key,
                        intersection_hash=request.intersection_hash,
                        street1=request.street1,
                        street2=request.street2,
                        borough=request.borough,
                        lat=lat,
                        lon=lon,
                        source=source,
                        status=GeocodeStatus.OK,
                    )
                else:
                    # Extract the actual geosupport error message from response
                    api_message = "No coordinates found in response"
                    error_label = "no_result"
                    
                    if all_fields and "error" in all_fields:
                        # This is a geosupport error like "intersects more than twice"
                        api_message = all_fields["error"]
                        error_label = "geosupport_error"
                    elif isinstance(response, dict):
                        # Try to extract from intersection response
                        intersection = response.get("intersection") or response.get("intersectionResponse")
                        if intersection and "message" in intersection:
                            api_message = intersection["message"]
                            error_label = "geosupport_message"
                    
                    # Don't try address fallback for geosupport errors
                    # Just return the error immediately
                    return GeocodeResult(
                        unique_key=request.unique_key,
                        intersection_hash=request.intersection_hash,
                        street1=request.street1,
                        street2=request.street2,
                        borough=request.borough,
                        status=GeocodeStatus.NOT_FOUND,
                        errors=[GeocodeError(
                            endpoint="intersection",
                            http_status=200,  # API returned 200 but with geosupport error
                            error_label=error_label,
                            api_message=api_message,
                            params_json=json.dumps({
                                "crossStreetOne": request.street1,
                                "crossStreetTwo": request.street2,
                                "borough": request.borough,
                            })[:500],
                        )]
                    )
                    
            except Exception as e:
                logger.warning(
                    f"Attempt {attempt+1}/{self.max_retries} failed for "
                    f"{request.street1} & {request.street2}, {request.borough}: {e}"
                )
                
                # Extract HTTP status and error details from HTTPError if available
                http_status = None
                error_label = "exception"
                api_message = str(e)[:500]
                
                if isinstance(e, requests.exceptions.HTTPError) and hasattr(e, 'response'):
                    response = e.response
                    http_status = response.status_code
                    error_label = f"http_{http_status}"
                    
                    # Try to extract meaningful error from response
                    try:
                        resp_json = response.json()
                        if isinstance(resp_json, dict):
                            api_message = resp_json.get("message") or resp_json.get("error") or str(resp_json)
                    except Exception:
                        api_message = response.text[:500] if response.text else api_message
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay_s * (2 ** attempt))  # exponential backoff
                    continue
                else:
                    # All retries exhausted
                    return GeocodeResult(
                        unique_key=request.unique_key,
                        intersection_hash=request.intersection_hash,
                        street1=request.street1,
                        street2=request.street2,
                        borough=request.borough,
                        status=GeocodeStatus.EXCEPTION,
                        errors=[GeocodeError(
                            endpoint="intersection",
                            http_status=http_status,
                            error_label=error_label,
                            api_message=str(api_message)[:500]
                        )]
                    )
        
        # No result found
        return GeocodeResult(
            unique_key=request.unique_key,
            intersection_hash=request.intersection_hash,
            street1=request.street1,
            street2=request.street2,
            borough=request.borough,
            status=GeocodeStatus.NOT_FOUND,
        )
    
    def geocode_batch(
        self,
        requests: List[GeocodeRequest],
        **kwargs: Any
    ) -> List[GeocodeResult]:
        """
        Geocode multiple requests sequentially.
        
        Each request is geocoded individually with rate limiting applied.
        
        Args:
            requests: List of GeocodeRequest objects
            **kwargs: Additional arguments
            
        Returns:
            List of GeocodeResult objects (same order as input)
        """
        return [self.geocode(req, **kwargs) for req in requests]
    
    def _query_intersection(
        self,
        street1: str,
        street2: str,
        borough: str,
    ) -> dict[str, Any]:
        """
        Query NYC Geoclient API for intersection.
        
        Normalizes street names before querying API. Returns raw API response.
        
        Args:
            street1: First street name
            street2: Second street name
            borough: Borough name
            
        Returns:
            API response dict with intersectionResponse or error info
            
        Raises:
            requests.RequestException on network/connection errors
            ValueError if API returns invalid JSON
        """
        # Normalize input for API
        normalized_street1 = self.street_normalizer.normalize(street1, None)
        normalized_street2 = self.street_normalizer.normalize(street2, None)
        normalized_borough = self.borough_normalizer.normalize(borough, None)
        
        # Build API endpoint URL (avoid urljoin stripping the /v2 segment)
        endpoint = f"{self.api_base_url.rstrip('/')}/intersection.json"
        
        # Prepare request parameters
        params = {
            "crossStreetOne": normalized_street1,
            "crossStreetTwo": normalized_street2,
            "borough": normalized_borough,
        }
        headers = {"Accept": "application/json"}
        params["app_key"] = self.api_key
        headers["Ocp-Apim-Subscription-Key"] = self.api_key
        
        logger.debug(
            f"Querying intersection: {normalized_street1} & {normalized_street2}, "
            f"{normalized_borough}"
        )
        
        # HTTP GET request
        response = requests.get(
            endpoint,
            params=params,
            timeout=self.timeout,
            headers=headers,
        )
        
        # Log request details for debugging
        logger.debug(f"Intersection query status: {response.status_code}")
        
        # Try to parse JSON response regardless of status
        try:
            response_json = response.json()
        except Exception:
            response_json = None
        
        # Raise for HTTP errors (4xx, 5xx) - this will be caught by the retry logic
        # but we can inspect response_json if needed
        if not response.ok:
            response.raise_for_status()
        
        # Parse and return JSON response
        return response_json if response_json is not None else response.json()
    
    def _fallback_to_address(self, request: GeocodeRequest) -> GeocodeResult:
        """
        Fallback to address geocoding if intersection query fails.
        
        Queries the /address endpoint using street1 as the street address.
        Only attempted if intersection query returns no valid coordinates.
        
        Args:
            request: Original request
            
        Returns:
            GeocodeResult from address query or NOT_FOUND
        """
        try:
            # Apply rate limiting
            if self.rate_limiter:
                self.rate_limiter.wait()
            
            # Query address endpoint (use street1 as the street)
            endpoint = f"{self.api_base_url.rstrip('/')}/address.json"
            
            normalized_street = self.street_normalizer.normalize(request.street1, None)
            normalized_borough = self.borough_normalizer.normalize(request.borough, None)
            
            params = {
                "street": normalized_street,
                "borough": normalized_borough,
            }
            headers = {"Accept": "application/json"}
            
            params["app_key"] = self.api_key
            headers["Ocp-Apim-Subscription-Key"] = self.api_key
            
            logger.debug(
                f"Fallback address query: {normalized_street}, {normalized_borough}"
            )
            
            response = requests.get(
                endpoint,
                params=params,
                timeout=self.timeout,
                headers=headers,
            )
            response.raise_for_status()
            
            response_json = response.json()
            lat, lon, source, all_fields = self._extract_from_address(response_json)
            
            # Check if coordinates are valid using same validation as GeocodeResult.is_success()
            if lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180:
                return GeocodeResult(
                    unique_key=request.unique_key,
                    intersection_hash=request.intersection_hash,
                    street1=request.street1,
                    street2=request.street2,
                    borough=request.borough,
                    lat=lat,
                    lon=lon,
                    source=source,
                    status=GeocodeStatus.OK,
                )
            else:
                # Check if we have a geosupport error message from the response
                api_message = "No coordinates found in response"
                if all_fields and "error" in all_fields:
                    api_message = all_fields["error"]
                elif "message" in response_json:
                    api_message = response_json["message"]
                
                errors = [GeocodeError(
                    endpoint="address",
                    http_status=response.status_code,
                    error_label="no_result",
                    api_message=api_message,
                    params_json=json.dumps(params)[:500],
                )]
                return GeocodeResult(
                    unique_key=request.unique_key,
                    intersection_hash=request.intersection_hash,
                    street1=request.street1,
                    street2=request.street2,
                    borough=request.borough,
                    status=GeocodeStatus.NOT_FOUND,
                    errors=errors,
                )
                
        except Exception as e:
            logger.warning(f"Address fallback failed: {e}")
            return GeocodeResult(
                unique_key=request.unique_key,
                intersection_hash=request.intersection_hash,
                street1=request.street1,
                street2=request.street2,
                borough=request.borough,
                status=GeocodeStatus.NOT_FOUND,
                errors=[GeocodeError(
                    endpoint="address",
                    error_label="fallback_exception",
                    api_message=str(e)
                )]
            )
    
    def geocode_csv(
        self,
        csv_path: str,
        storage_backend=None,
        limit: Optional[int] = None,
    ) -> tuple[int, int]:
        """
        Geocode all intersections in a CSV file and optionally persist results.
        
        Args:
            csv_path: Path to CSV with columns: street1, street2, borough, unique_key
            storage_backend: Optional storage backend to persist results (DuckDB, CSV, etc.)
            limit: Optional limit on number of rows to process
            
        Returns:
            Tuple of (successes, failures) counts
            
        Example:
            from src.streetTransformer.geocoding import NYCGeoClient
            from src.streetTransformer.geocoding.storage import DuckDBStorageBackend
            
            client = NYCGeoClient(api_key="your_key", rate_limiter=limiter)
            storage = DuckDBStorageBackend("results.db")
            success_count, fail_count = client.geocode_csv("data.csv", storage)
            print(f"Results: {success_count} success, {fail_count} failed")
        """
        from .normalizers import DataPreparer
        
        # Prepare/clean data (drop empty rows, etc.)
        preparer = DataPreparer()
        df = preparer.prepare_csv(csv_path, limit=limit)
        
        # Build requests
        requests = []
        for _, row in df.iterrows():
            s1 = self.street_normalizer.normalize(str(row["street1"]))
            s2 = self.street_normalizer.normalize(str(row["street2"]))
            b = self.borough_normalizer.normalize(str(row["borough"]))
            if not s1 or not s2 or not b:
                continue
            
            from .models import compute_intersection_hash
            h = compute_intersection_hash([s1, s2], b)
            requests.append(GeocodeRequest(str(row["unique_key"]), s1, s2, b, h))
        
        logger.info(f"Prepared {len(requests)} requests from {len(df)} cleaned rows")
        
        # Geocode all requests
        results = self.geocode_batch(requests)
        
        # Optionally persist results
        if storage_backend:
            for result in results:
                if result.is_success():
                    storage_backend.write_success(result)
                else:
                    storage_backend.write_error(result)
            storage_backend.close()
        
        # Count results
        success_count = sum(1 for r in results if r.is_success())
        fail_count = len(results) - success_count
        
        logger.info(f"Geocoding complete: {success_count} success, {fail_count} failed")
        return success_count, fail_count

