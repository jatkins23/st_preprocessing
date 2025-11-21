# Load imagery using the Strategy pattern
from __future__ import annotations

from typing import Iterable, Any, ClassVar, TYPE_CHECKING
from pathlib import Path
import logging

from tqdm import tqdm

from .source_strategies import ImagerySource, LocalFileSource, CachedSource, TileStitchingSource
from ..data_loader import DataLoader

if TYPE_CHECKING:
    from ..locations.universe import Universe

logger = logging.getLogger('ImageryLoader')


class ImageryLoader(DataLoader):
    """Load imagery using a pluggable source strategy.

    Registers as 'imagery' modality in DataLoader.
    The ImageryLoader uses the Strategy pattern to allow different imagery
    sources (local files, APIs, cached sources) to be used interchangeably.

    Note: Unlike other modalities, ImageryLoader doesn't use the registry pattern
    since it uses strategies instead. Use the convenience methods from_local()
    and from_api() for common use cases.

    Args:
        source_strategy: Strategy for fetching imagery
        region: Region identifier for the imagery
        locations: Locations to fetch imagery for (lat, lon tuples)

    Example:
        # Load from local files
        loader = ImageryLoader.from_local(
            file_dir='./images',
            region='manhattan',
            locations=[(40.7128, -74.0060)]
        )

        # Load from Google Maps API with caching
        from .source_strategies import GoogleMapsAPISource

        api_source = GoogleMapsAPISource(api_key='YOUR_KEY')
        loader = ImageryLoader.from_api(
            api_source=api_source,
            region='manhattan',
            locations=[(40.7128, -74.0060)],
            cache_dir='./cache'
        )

        images = loader.load()
    """

    # Register this loader as the 'imagery' modality
    MODALITY: ClassVar[str] = "imagery"

    def __init__(
        self,
        source_strategy: ImagerySource,
        region: str,
        locations: Iterable[tuple[float, float]]
    ):
        self.source = source_strategy
        self.region = region
        self.locations = list(locations)

    def _load_raw(self) -> list[Any]:
        """Load imagery using the configured strategy.

        Returns:
            List of image data (format depends on strategy)
        """
        logger.info(f"Loading imagery for region '{self.region}' with {len(self.locations)} locations")

        # If using TileStitchingSource with bounds, pass zlevel
        if isinstance(self.source, TileStitchingSource) and hasattr(self, 'zlevel'):
            return self.source.fetch_batch(self.locations, zlevel=self.zlevel)
        else:
            return self.source.fetch_batch(self.locations)

    def load(self) -> list[Any]:
        """Load and return imagery.

        Returns:
            List of image data
        """
        return self._load_raw()

    @classmethod
    def from_local(
        cls,
        file_dir: Path | str,
        region: str,
        locations: Iterable[tuple[float, float]],
        **kwargs: Any
    ) -> ImageryLoader:
        """Convenience method to create a loader with LocalFileSource.

        Args:
            file_dir: Directory containing image files
            region: Region identifier
            locations: Locations to load imagery for
            **kwargs: Additional arguments for LocalFileSource

        Returns:
            ImageryLoader configured with LocalFileSource
        """
        source = LocalFileSource(directory=file_dir, **kwargs)
        return cls(source_strategy=source, region=region, locations=locations)

    @classmethod
    def from_api(
        cls,
        api_source: ImagerySource,
        region: str,
        locations: Iterable[tuple[float, float]],
        cache_dir: Path | str | None = None,
        **kwargs: Any
    ) -> ImageryLoader:
        """Convenience method to create a loader with an API source.

        Args:
            api_source: API source strategy (e.g., GoogleMapsAPISource)
            region: Region identifier
            locations: Locations to load imagery for
            cache_dir: Optional directory for caching (wraps source in CachedSource)
            **kwargs: Additional arguments

        Returns:
            ImageryLoader configured with API source (optionally cached)
        """
        source = api_source

        # Wrap in caching strategy if cache_dir provided
        if cache_dir is not None:
            source = CachedSource(source=api_source, cache_dir=cache_dir)

        return cls(source_strategy=source, region=region, locations=locations)

    @classmethod
    def from_tiles(
        cls,
        region: str,
        bounds_list: Iterable[tuple[float, float, float, float]],
        source: ImagerySource,
        zlevel: int = 18,
        **kwargs: Any
    ) -> ImageryLoader:
        """Convenience method to create a loader with an ImagerySource for tile-based loading.

        This method creates an imagery loader that fetches imagery for bounding boxes
        using the provided ImagerySource strategy.

        Args:
            region: Region identifier
            bounds_list: List of (west, south, east, north) bounding boxes in WGS84
            source: ImagerySource strategy for fetching imagery
            zlevel: Zoom level for tiles (default: 18)
            **kwargs: Additional arguments

        Returns:
            ImageryLoader configured with the provided source

        Example:
            # Using TileStitchingSource
            from st_preprocessing.imagery.source_strategies import TileStitchingSource

            source = TileStitchingSource(
                tile_provider='https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                cache_dir='./tile_cache'
            )

            imagery = ImageryLoader.from_tiles(
                region='nyc',
                bounds_list=bounds,
                source=source,
                zlevel=18
            )

            images = imagery.load()
        """
        # Store bounds and zlevel for loading
        loader = cls(source_strategy=source, region=region, locations=list(bounds_list))
        loader.zlevel = zlevel  # Store zlevel for batch loading
        return loader

    @classmethod
    def load_from_universe(
        cls,
        universe: 'Universe',
        source: ImagerySource,
        save_dir: Path | str,
        zlevel: int | None = None,
        n_workers: int = 4,
        **kwargs: Any
    ) -> dict[int, Path]:
        """Load and save imagery for all locations in a universe.

        This wrapper function iterates over universe.location_geometries,
        loads imagery for each bounds, and saves it to disk. Uses batch
        fetching with multithreading for improved performance.

        Args:
            universe: Universe object with location_geometries
            source: ImagerySource strategy for fetching imagery
            save_dir: Directory to save images
            zlevel: Optional zoom level override (uses universe's zlevel if None)
            n_workers: Number of worker threads for parallel fetching (default: 4)
            **kwargs: Additional arguments passed to the source

        Returns:
            Dictionary mapping location_id to saved image path

        Example:
            from st_preprocessing.locations import UniverseLoader
            from st_preprocessing.imagery.source_strategies import TileStitchingSource

            # Load universe with geometries
            universe = UniverseLoader.from_source('lion', with_geometries=True)

            # Create imagery source
            source = TileStitchingSource(
                tile_provider='https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                cache_dir='./tile_cache'
            )

            # Load and save imagery for all locations (using 4 threads)
            image_paths = ImageryLoader.load_from_universe(
                universe=universe,
                source=source,
                save_dir='./images/lion',
                n_workers=4
            )
        """
        if universe.location_geometries is None:
            raise ValueError(
                "Universe has no location_geometries. "
                "Load with with_geometries=True or use load_with_geometries()"
            )

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        geoms_df = universe.location_geometries

        logger.info(f"Loading imagery for {len(geoms_df)} locations from universe '{universe.name}'")

        # Prepare data for batch fetching
        location_ids = []
        bounds_list = []

        for _, row in geoms_df.iterrows():
            location_ids.append(row['location_id'])
            bounds_gcs = row['bounds_gcs']

            # Convert bounds_gcs to tuple if it's a list
            if isinstance(bounds_gcs, list):
                bounds_gcs = tuple(bounds_gcs)

            bounds_list.append(bounds_gcs)

        # Determine zlevel to use
        if zlevel is None:
            # Use the first zlevel from the dataframe (assuming all are the same)
            zlevel = geoms_df.iloc[0]['zlevel']

        # Batch fetch all images using multithreading
        logger.info(f"Fetching {len(bounds_list)} images with {n_workers} workers")
        images = source.fetch_batch(
            bounds_list=bounds_list,
            zlevel=zlevel,
            n_workers=n_workers,
            show_progress=True,
            **kwargs
        )

        # Save all images with progress bar
        image_paths = {}
        failed_count = 0

        logger.info(f"Saving {len(images)} images to {save_dir}")
        with tqdm(total=len(images), desc="Saving images", unit="img") as pbar:
            for location_id, img in zip(location_ids, images):
                if img is None:
                    failed_count += 1
                    logger.warning(f"Skipping location_id={location_id} (fetch failed)")
                    pbar.update(1)
                    continue

                try:
                    image_path = save_dir / f"{location_id}.png"
                    img.save(image_path)
                    image_paths[location_id] = image_path
                    pbar.set_postfix({'saved': len(image_paths), 'failed': failed_count})
                except Exception as e:
                    failed_count += 1
                    logger.error(f"Failed to save image for location_id={location_id}: {e}")
                    pbar.set_postfix({'saved': len(image_paths), 'failed': failed_count})
                finally:
                    pbar.update(1)

        logger.info(
            f"Successfully loaded and saved {len(image_paths)} images to {save_dir} "
            f"({failed_count} failed)"
        )

        return image_paths

    def _validate(self):
        """Validate loaded imagery data.

        Default implementation does no validation.
        Subclasses can override to add specific validation logic.
        """
        pass

    def _to_database(self):
        """Save imagery to database.

        Default implementation does nothing.
        Subclasses can override to implement database persistence.
        """
        pass