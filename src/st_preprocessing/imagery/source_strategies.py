"""Strategy pattern implementations for loading imagery from different sources.

This module defines the ImagerySource interface and concrete strategies
for loading imagery from local files, APIs, or cached sources.
"""

from __future__ import annotations

from typing import Any, Iterable
from abc import ABC, abstractmethod
from pathlib import Path
import logging
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import mercantile
import requests
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger('ImagerySource')


class ImagerySource(ABC):
    """Abstract strategy interface for loading imagery.

    Concrete strategies implement different ways to fetch imagery
    (local files, remote APIs, cached sources, etc.).
    """

    @abstractmethod
    def fetch(self, location: tuple[float, float], **kwargs: Any) -> Any:
        """Fetch imagery for a specific location.

        Args:
            location: Tuple of (latitude, longitude)
            **kwargs: Additional parameters specific to the source

        Returns:
            Image data (format depends on implementation)
        """
        ...

    @abstractmethod
    def fetch_batch(self, locations: Iterable[tuple[float, float]], **kwargs: Any) -> list[Any]:
        """Fetch imagery for multiple locations.

        Args:
            locations: Iterable of (latitude, longitude) tuples
            **kwargs: Additional parameters specific to the source

        Returns:
            List of image data
        """
        ...


class LocalFileSource(ImagerySource):
    """Strategy for loading imagery from local files.

    Args:
        directory: Directory containing image files
        file_pattern: Pattern for matching files (e.g., '{lat}_{lon}.png')
        file_extension: File extension to use (default: '.png')
    """

    def __init__(self, directory: Path | str, file_pattern: str = '{id}', file_extension: str = '.png'):
        self.directory = Path(directory)
        self.file_pattern = file_pattern
        self.file_extension = file_extension

        if not self.directory.exists():
            raise ValueError(f"Directory does not exist: {self.directory}")

    def fetch(self, location: tuple[float, float], **kwargs: Any) -> Path:
        """Fetch image file path for a location.

        Args:
            location: (latitude, longitude) tuple
            **kwargs: May include 'id' for custom file naming

        Returns:
            Path to the image file
        """
        lat, lon = location
        location_id = kwargs.get('id', f"{lat}_{lon}")

        file_name = self.file_pattern.format(
            id=location_id,
            lat=lat,
            lon=lon
        ) + self.file_extension

        file_path = self.directory / file_name

        if not file_path.exists():
            logger.warning(f"Image file not found: {file_path}")
            return None

        return file_path

    def fetch_batch(self, locations: Iterable[tuple[float, float]], n_workers: int = 1, show_progress: bool = True, **kwargs: Any) -> list[Path]:
        """Fetch multiple image file paths.

        Args:
            locations: Iterable of (latitude, longitude) tuples
            n_workers: Number of worker threads for parallel fetching (default: 1)
            show_progress: Whether to show progress bar (default: True)
            **kwargs: Additional parameters passed to fetch()

        Returns:
            List of Path objects
        """
        locations_list = list(locations)

        if n_workers == 1:
            iterator = tqdm(locations_list, desc="Fetching images", disable=not show_progress) if show_progress else locations_list
            return [self.fetch(loc, **kwargs) for loc in iterator]

        results = [None] * len(locations_list)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_idx = {
                executor.submit(self.fetch, loc, **kwargs): idx
                for idx, loc in enumerate(locations_list)
            }

            with tqdm(total=len(future_to_idx), desc="Fetching images", disable=not show_progress) as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        logger.error(f"Error fetching location {locations_list[idx]}: {e}")
                        results[idx] = None
                    pbar.update(1)

        return results


class APISource(ImagerySource):
    """Base strategy for loading imagery from an API.

    This is a base class that specific API implementations can extend.

    Args:
        api_key: API key for authentication
        base_url: Base URL for the API
        params: Default parameters for API requests
    """

    def __init__(self, api_key: str, base_url: str, params: dict[str, Any] | None = None):
        self.api_key = api_key
        self.base_url = base_url
        self.default_params = params or {}

    def fetch(self, location: tuple[float, float], **kwargs: Any) -> Any:
        """Fetch imagery from API for a location.

        Args:
            location: (latitude, longitude) tuple
            **kwargs: Additional API parameters

        Returns:
            Image data from API
        """
        raise NotImplementedError(
            "Subclasses must implement fetch() with specific API logic"
        )

    def fetch_batch(self, locations: Iterable[tuple[float, float]], n_workers: int = 1, show_progress: bool = True, **kwargs: Any) -> list[Any]:
        """Fetch imagery for multiple locations.

        Args:
            locations: Iterable of (latitude, longitude) tuples
            n_workers: Number of worker threads for parallel fetching (default: 1)
            show_progress: Whether to show progress bar (default: True)
            **kwargs: Additional parameters passed to fetch()

        Returns:
            List of image data
        """
        locations_list = list(locations)

        if n_workers == 1:
            iterator = tqdm(locations_list, desc="Fetching images", disable=not show_progress) if show_progress else locations_list
            return [self.fetch(loc, **kwargs) for loc in iterator]

        results = [None] * len(locations_list)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_idx = {
                executor.submit(self.fetch, loc, **kwargs): idx
                for idx, loc in enumerate(locations_list)
            }

            with tqdm(total=len(future_to_idx), desc="Fetching images", disable=not show_progress) as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        logger.error(f"Error fetching location {locations_list[idx]}: {e}")
                        results[idx] = None
                    pbar.update(1)

        return results


class GoogleMapsAPISource(APISource):
    """Strategy for loading imagery from Google Maps API.

    Args:
        api_key: Google Maps API key
        zoom: Zoom level (default: 18)
        size: Image size as "widthxheight" (default: "640x640")
        maptype: Map type ('satellite', 'roadmap', etc.)
    """

    def __init__(
        self,
        api_key: str,
        zoom: int = 18,
        size: str = "640x640",
        maptype: str = "satellite"
    ):
        super().__init__(
            api_key=api_key,
            base_url="https://maps.googleapis.com/maps/api/staticmap",
            params={
                'zoom': zoom,
                'size': size,
                'maptype': maptype,
            }
        )

    def fetch(self, location: tuple[float, float], **kwargs: Any) -> bytes:
        """Fetch satellite imagery from Google Maps.

        Args:
            location: (latitude, longitude) tuple
            **kwargs: Additional parameters (zoom, size, maptype, etc.)

        Returns:
            Image data as bytes
        """
        import requests

        lat, lon = location
        params = {
            **self.default_params,
            **kwargs,
            'center': f"{lat},{lon}",
            'key': self.api_key,
        }

        logger.debug(f"Fetching imagery for ({lat}, {lon}) from Google Maps")
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()

        return response.content


class CachedSource(ImagerySource):
    """Strategy that caches imagery from another source.

    This strategy checks a local cache before fetching from the underlying source.
    If the image exists in cache, it's loaded from there. Otherwise, it's fetched
    from the source and saved to cache.

    Args:
        source: Underlying imagery source to fetch from if not cached
        cache_dir: Directory to store cached images
        cache_pattern: Pattern for cache file names (default: '{id}.png')
    """

    def __init__(
        self,
        source: ImagerySource,
        cache_dir: Path | str,
        cache_pattern: str = '{id}.png'
    ):
        self.source = source
        self.cache_dir = Path(cache_dir)
        self.cache_pattern = cache_pattern

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, location: tuple[float, float], **kwargs: Any) -> Path:
        """Get the cache file path for a location."""
        lat, lon = location
        location_id = kwargs.get('id', f"{lat}_{lon}")

        file_name = self.cache_pattern.format(
            id=location_id,
            lat=lat,
            lon=lon
        )

        return self.cache_dir / file_name

    def fetch(self, location: tuple[float, float], **kwargs: Any) -> Any:
        """Fetch imagery, using cache if available.

        Args:
            location: (latitude, longitude) tuple
            **kwargs: Additional parameters

        Returns:
            Image data (from cache or source)
        """
        cache_path = self._get_cache_path(location, **kwargs)

        # Check cache first
        if cache_path.exists():
            logger.debug(f"Loading from cache: {cache_path}")
            return cache_path

        # Fetch from source
        logger.debug(f"Cache miss, fetching from source: {location}")
        data = self.source.fetch(location, **kwargs)

        # Save to cache
        if data is not None:
            if isinstance(data, bytes):
                cache_path.write_bytes(data)
                logger.debug(f"Cached image to: {cache_path}")
                return cache_path
            elif isinstance(data, Path):
                # Source returned a file path, copy it
                import shutil
                shutil.copy(data, cache_path)
                logger.debug(f"Cached image to: {cache_path}")
                return cache_path

        return data

    def fetch_batch(self, locations: Iterable[tuple[float, float]], n_workers: int = 1, show_progress: bool = True, **kwargs: Any) -> list[Any]:
        """Fetch multiple images with caching.

        Args:
            locations: Iterable of (latitude, longitude) tuples
            n_workers: Number of worker threads for parallel fetching (default: 1)
            show_progress: Whether to show progress bar (default: True)
            **kwargs: Additional parameters passed to fetch()

        Returns:
            List of image data (cached or fetched)
        """
        locations_list = list(locations)

        if n_workers == 1:
            iterator = tqdm(locations_list, desc="Fetching images", disable=not show_progress) if show_progress else locations_list
            return [self.fetch(loc, **kwargs) for loc in iterator]

        results = [None] * len(locations_list)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_idx = {
                executor.submit(self.fetch, loc, **kwargs): idx
                for idx, loc in enumerate(locations_list)
            }

            with tqdm(total=len(future_to_idx), desc="Fetching images", disable=not show_progress) as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        logger.error(f"Error fetching location {locations_list[idx]}: {e}")
                        results[idx] = None
                    pbar.update(1)

        return results


class TileStitchingSource(ImagerySource):
    """Fetch and stitch map tiles based on LocationGeometry tile grids.

    This strategy:
    1. Fetches individual tiles from the tile grid
    2. Stitches them together into a single image
    3. Centers on the intersection centroid
    4. Crops to the specified bounding box

    Args:
        tile_provider: URL template for tile provider (e.g., OSM, Mapbox)
                      Use {z}/{x}/{y} placeholders
        tile_size: Size of individual tiles in pixels (default: 256)
        cache_dir: Optional directory for caching tiles
        headers: Optional headers for tile requests (e.g., for Mapbox token)

    Example:
        # OpenStreetMap
        source = TileStitchingSource(
            tile_provider='https://tile.openstreetmap.org/{z}/{x}/{y}.png',
            cache_dir='./tile_cache'
        )

        # Mapbox Satellite
        source = TileStitchingSource(
            tile_provider='https://api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}.png',
            headers={'access_token': 'YOUR_TOKEN'},
            cache_dir='./tile_cache'
        )
    """

    def __init__(
        self,
        tile_provider: str = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
        tile_size: int = 256,
        cache_dir: Path | str | None = None,
        headers: dict[str, str] | None = None
    ):
        self.tile_provider = tile_provider
        self.tile_size = tile_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.headers = headers or {}

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create persistent session for connection pooling
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        # Set connection pool size (large enough for parallel workers)
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=20,
            pool_maxsize=200,  
            max_retries=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

    def __del__(self):
        """Cleanup session on deletion to avoid resource leaks."""
        if hasattr(self, 'session'):
            self.session.close()

    def cleanup(self):
        """Explicitly cleanup session and connection pools to avoid semaphore leaks.

        This method clears the connection pool and closes the session more aggressively
        than just calling session.close().
        """
        if hasattr(self, 'session') and self.session is not None:
            # Clear all adapters to cleanup their connection pools
            for adapter in self.session.adapters.values():
                adapter.close()
            # Close the session itself
            self.session.close()

    def _get_tile_url(self, tile: mercantile.Tile) -> str:
        """Generate URL for a specific tile."""
        return self.tile_provider.format(z=tile.z, x=tile.x, y=tile.y)

    def _get_tile_cache_path(self, tile: mercantile.Tile) -> Path | None:
        """Get cache path for a tile."""
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{tile.z}_{tile.x}_{tile.y}.png"

    def _fetch_single_tile(self, tile: mercantile.Tile, silent: bool = False) -> tuple[Image.Image, bool]:
        """Fetch a single tile, using cache if available.

        Args:
            tile: Tile to fetch
            silent: If True, suppress warning messages (for tqdm compatibility)

        Returns:
            Tuple of (image, success) where success indicates if tile was fetched successfully
        """
        cache_path = self._get_tile_cache_path(tile)

        # Check cache
        if cache_path and cache_path.exists():
            try:
                # Verify it's a valid file, not a directory
                if cache_path.is_dir():
                    if not silent:
                        logger.warning(f"Cache path is a directory, removing: {cache_path}")
                    cache_path.rmdir()
                else:
                    # Try to open cached image
                    logger.debug(f"Loading tile from cache: {cache_path}")
                    img = Image.open(cache_path)
                    img.load()  # Force load to detect corrupt files
                    return img, True
            except (OSError, IOError):
                # Cache file is corrupt or unreadable, delete it and re-fetch
                # Don't log during parallel operations to avoid messing up progress bars
                try:
                    cache_path.unlink()
                except Exception:
                    pass

        # Fetch from provider
        url = self._get_tile_url(tile)
        logger.debug(f"Fetching tile: {url}")

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            # Verify response has content
            if not response.content or len(response.content) == 0:
                raise ValueError(f"Empty response from {url}")

            img = Image.open(BytesIO(response.content))
            img.load()  # Force load to verify it's valid

            # Save to cache
            if cache_path:
                try:
                    # Ensure parent directory exists
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    img.save(cache_path)
                    logger.debug(f"Cached tile to: {cache_path}")
                except Exception:
                    # Silently continue if caching fails
                    pass

            return img, True

        except Exception as e:
            if not silent:
                logger.error(f"Failed to fetch tile {tile} from {url}: {e}")
            # Return blank tile on error
            return Image.new('RGB', (self.tile_size, self.tile_size), color='gray'), False

    def _get_tiles_for_bounds(self, bounds: tuple[float, float, float, float], zlevel: int) -> list[mercantile.Tile]:
        """Calculate which tiles cover the given bounding box.

        Args:
            bounds: (west, south, east, north) in WGS84 degrees
            zlevel: Zoom level

        Returns:
            List of tiles that cover the bounding box
        """
        west, south, east, north = bounds

        # Get tiles at corners
        ul_tile = mercantile.tile(west, north, zlevel)
        lr_tile = mercantile.tile(east, south, zlevel)

        # Get all tiles in the range
        tiles = []
        for x in range(ul_tile.x, lr_tile.x + 1):
            for y in range(ul_tile.y, lr_tile.y + 1):
                tiles.append(mercantile.Tile(x=x, y=y, z=zlevel))

        logger.debug(f"Calculated {len(tiles)} tiles for bounds {bounds} at zoom {zlevel}")
        return tiles

    def _stitch_tiles(self, tiles: list[tuple[mercantile.Tile, Image.Image]]) -> Image.Image:
        """Stitch tiles together into a single image.

        Args:
            tiles: List of (tile, image) tuples

        Returns:
            Stitched image
        """
        if not tiles:
            raise ValueError("No tiles to stitch")

        # Get the min and max x,y to calculate dimensions
        min_x = min(t[0].x for t in tiles)
        max_x = max(t[0].x for t in tiles)
        min_y = min(t[0].y for t in tiles)
        max_y = max(t[0].y for t in tiles)

        # Calculate output dimensions
        width_tiles = max_x - min_x + 1
        height_tiles = max_y - min_y + 1
        output_width = width_tiles * self.tile_size
        output_height = height_tiles * self.tile_size

        # Create blank canvas
        stitched = Image.new('RGB', (output_width, output_height))

        # Paste each tile
        for tile, img in tiles:
            # Calculate pixel position
            x_offset = (tile.x - min_x) * self.tile_size
            y_offset = (tile.y - min_y) * self.tile_size

            stitched.paste(img, (x_offset, y_offset))
            logger.debug(f"Placed tile {tile} at ({x_offset}, {y_offset})")

        return stitched

    def _crop_to_bounds(
        self,
        img: Image.Image,
        stitched_tiles: list[mercantile.Tile],
        target_bounds: tuple[float, float, float, float]
    ) -> Image.Image:
        """Crop stitched image to the target bounding box.

        Args:
            img: Stitched image
            stitched_tiles: List of tiles that were stitched
            target_bounds: (west, south, east, north) to crop to

        Returns:
            Cropped image
        """
        if not stitched_tiles:
            return img

        # Calculate the geographic bounds of the stitched image
        min_x = min(t.x for t in stitched_tiles)
        max_x = max(t.x for t in stitched_tiles)
        min_y = min(t.y for t in stitched_tiles)
        max_y = max(t.y for t in stitched_tiles)
        zlevel = stitched_tiles[0].z

        # Get geographic bounds from tiles
        ul_tile = mercantile.Tile(x=min_x, y=min_y, z=zlevel)
        lr_tile = mercantile.Tile(x=max_x, y=max_y, z=zlevel)

        ul_bounds = mercantile.bounds(ul_tile)
        lr_bounds = mercantile.bounds(lr_tile)

        # Full stitched image bounds
        full_west = ul_bounds.west
        full_north = ul_bounds.north
        full_east = lr_bounds.east
        full_south = lr_bounds.south

        # Target crop bounds
        crop_west, crop_south, crop_east, crop_north = target_bounds

        # Convert geographic bounds to pixel coordinates
        img_width, img_height = img.size

        # Calculate pixel positions
        x_min = int(((crop_west - full_west) / (full_east - full_west)) * img_width)
        x_max = int(((crop_east - full_west) / (full_east - full_west)) * img_width)
        y_min = int(((full_north - crop_north) / (full_north - full_south)) * img_height)
        y_max = int(((full_north - crop_south) / (full_north - full_south)) * img_height)

        # Ensure bounds are within image
        x_min = max(0, min(x_min, img_width))
        x_max = max(0, min(x_max, img_width))
        y_min = max(0, min(y_min, img_height))
        y_max = max(0, min(y_max, img_height))

        logger.debug(f"Cropping to: ({x_min}, {y_min}, {x_max}, {y_max})")

        return img.crop((x_min, y_min, x_max, y_max))

    def fetch(
        self,
        bounds: tuple[float, float, float, float],
        zlevel: int = 18,
        crop: bool = True,
        tile_workers: int = 8,
        **kwargs: Any
    ) -> Image.Image:
        """Fetch and stitch tiles for a bounding box.

        Args:
            bounds: (west, south, east, north) in WGS84 degrees
            zlevel: Zoom level for tiles (default: 18)
            crop: Whether to crop to exact bounds (default: True)
            tile_workers: Number of threads for parallel tile fetching (default: 8)
            **kwargs: Additional parameters (reserved for future use)

        Returns:
            PIL Image object (stitched and optionally cropped)

        Example:
            # Fetch imagery for Times Square area
            source = TileStitchingSource()
            bounds = (-74.0020, 40.7570, -73.9980, 40.7590)  # (W, S, E, N)
            img = source.fetch(bounds, zlevel=18)
            img.save('times_square.png')
        """
        # Get tiles needed for this bounding box
        tiles = self._get_tiles_for_bounds(bounds, zlevel)

        # Fetch all tiles in parallel
        tiles_with_images = []
        tile_errors = 0

        if tile_workers == 1 or len(tiles) == 1:
            # Serial fetching
            for tile in tiles:
                img, success = self._fetch_single_tile(tile, silent=False)
                tiles_with_images.append((tile, img))
                if not success:
                    tile_errors += 1
        else:
            # Parallel tile fetching with error tracking
            with ThreadPoolExecutor(max_workers=tile_workers) as executor:
                future_to_tile = {
                    executor.submit(self._fetch_single_tile, tile, True): tile
                    for tile in tiles
                }

                for future in as_completed(future_to_tile):
                    tile = future_to_tile[future]
                    try:
                        img, success = future.result()
                        tiles_with_images.append((tile, img))
                        if not success:
                            tile_errors += 1
                    except Exception as e:
                        logger.error(f"Error fetching tile {tile}: {e}")
                        tile_errors += 1
                        # Add blank tile on error
                        img = Image.new('RGB', (self.tile_size, self.tile_size), color='gray')
                        tiles_with_images.append((tile, img))

        # Log tile errors if any occurred (without messing up progress bars)
        if tile_errors > 0:
            tqdm.write(f"Warning: {tile_errors}/{len(tiles)} tiles failed to fetch (replaced with gray)")

        # Stitch tiles together
        stitched = self._stitch_tiles(tiles_with_images)

        # Optionally crop to exact bounds
        if crop:
            stitched = self._crop_to_bounds(stitched, tiles, bounds)

        return stitched

    def fetch_from_location_geometry(self, location_geometry: 'LocationGeometry', **kwargs: Any) -> Image.Image:
        """Convenience method to fetch using a LocationGeometry object.

        Args:
            location_geometry: LocationGeometry object with bounds and zlevel
            **kwargs: Additional parameters passed to fetch()

        Returns:
            PIL Image object
        """
        return self.fetch(
            bounds=location_geometry.bounds_gcs,
            zlevel=location_geometry.zlevel,
            **kwargs
        )

    def fetch_batch(
        self,
        bounds_list: Iterable[tuple[float, float, float, float]],
        zlevel: int = 18,
        n_workers: int = 1,
        show_progress: bool = True,
        **kwargs: Any
    ) -> list[Image.Image]:
        """Fetch and stitch tiles for multiple bounding boxes.

        Args:
            bounds_list: List of (west, south, east, north) bounding boxes
            zlevel: Zoom level for tiles
            n_workers: Number of worker threads for parallel fetching (default: 1)
            show_progress: Whether to show progress bar (default: True)
            **kwargs: Additional parameters passed to fetch()

        Returns:
            List of PIL Images
        """
        bounds_list = list(bounds_list)

        if n_workers == 1:
            iterator = tqdm(bounds_list, desc="Stitching tiles", disable=not show_progress) if show_progress else bounds_list
            return [self.fetch(bounds, zlevel=zlevel, **kwargs) for bounds in iterator]

        results = [None] * len(bounds_list)
        failed_images = 0

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_idx = {
                executor.submit(self.fetch, bounds, zlevel=zlevel, **kwargs): idx
                for idx, bounds in enumerate(bounds_list)
            }

            with tqdm(total=len(future_to_idx), desc="Stitching tiles", disable=not show_progress) as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        tqdm.write(f"Error fetching image {idx}: {e}")
                        results[idx] = None
                        failed_images += 1

                    # Update progress bar with error count
                    pbar.set_postfix({'failed': failed_images})
                    pbar.update(1)

        return results
