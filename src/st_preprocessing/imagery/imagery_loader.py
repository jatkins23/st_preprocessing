# Load imagery using the Strategy pattern
from __future__ import annotations

from typing import Iterable, Any, ClassVar, TYPE_CHECKING
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from tqdm import tqdm
import pandas as pd

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
        zlevel: int = 20,
        n_workers: int = 4,
        update_db_metadata: bool = True,
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
            update_db_metadata: Whether to register files in database (default: True)
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
        location_ids = geoms_df['location_id'].tolist()
        bounds_list = geoms_df['bounds_gcs'].tolist()

        def _fetch_and_save(location_id: int, bounds: tuple[float, float, float, float]) -> tuple[int, Path | None, Exception | None]:
            """Fetch a single image and save to disk to avoid retaining all images in memory."""
            img = None
            try:
                img = source.fetch(bounds=bounds, zlevel=zlevel, **kwargs)
                if img is None:
                    return location_id, None, None

                image_path = save_dir / f"{location_id}.png"
                img.save(image_path)
                return location_id, image_path, None
            except Exception as exc:  # noqa: BLE001
                return location_id, None, exc
            finally:
                if hasattr(img, "close"):
                    try:
                        img.close()
                    except Exception:
                        pass

        logger.info(f"Fetching {len(bounds_list)} images with {n_workers} workers (streaming saves to limit memory)")

        # Save all images with progress bar
        image_paths = {}
        failed_count = 0

        logger.info(f"Saving {len(bounds_list)} images to {save_dir}")
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_fetch_and_save, loc_id, bounds): loc_id
                for loc_id, bounds in zip(location_ids, bounds_list)
            }

            with tqdm(total=len(futures), desc="Fetching+Saving images", unit="img") as pbar:
                for future in as_completed(futures):
                    location_id, image_path, error = future.result()

                    if error is not None:
                        failed_count += 1
                        logger.error(f"Failed to save image for location_id={location_id}: {error}")
                    elif image_path is None:
                        failed_count += 1
                        logger.warning(f"Skipping location_id={location_id} (fetch returned None)")
                    else:
                        image_paths[location_id] = image_path

                    pbar.set_postfix({'saved': len(image_paths), 'failed': failed_count})
                    pbar.update(1)

        logger.info(
            f"Successfully loaded and saved {len(image_paths)} images to {save_dir} "
            f"({failed_count} failed)"
        )

        # Register files to database if requested
        if update_db_metadata:
            # Extract year from source
            if hasattr(source, 'year'):
                cls._register_files_to_db(
                    universe_name=universe.name,
                    image_paths=image_paths,
                    year=source.year,
                    save_dir=save_dir
                )
            else:
                logger.warning(
                    "Source does not have a 'year' attribute. "
                    "Skipping database metadata registration. "
                    "Use a source with year attribute (e.g., NewYork(year=2024))"
                )

        return image_paths

    @classmethod
    def _create_metadata_table(cls, universe_name: str) -> None:
        """Create location_year_files table in universe schema if it doesn't exist.

        Args:
            universe_name: Name of the universe (used as schema name)
        """
        from ..db.db import duckdb_connection
        from ..settings import settings

        with duckdb_connection() as db_con:
            # Create schema if it doesn't exist
            db_con.execute(f"CREATE SCHEMA IF NOT EXISTS {universe_name}")

            # Create location_year_files table
            db_con.execute(f"""
                CREATE TABLE IF NOT EXISTS {universe_name}.location_year_files (
                    location_id INTEGER NOT NULL,
                    universe_name VARCHAR NOT NULL,
                    year INTEGER NOT NULL,
                    file_type VARCHAR NOT NULL,
                    file_path_abs VARCHAR NOT NULL,
                    file_path_rel VARCHAR NOT NULL,
                    file_size INTEGER,
                    exists BOOLEAN DEFAULT TRUE,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_date TIMESTAMP,
                    PRIMARY KEY (location_id, year, file_type)
                )
            """)

            logger.debug(f"Created/verified location_year_files table in schema {universe_name}")

    @classmethod
    def _register_files_to_db(
        cls,
        universe_name: str,
        image_paths: dict[int, Path],
        year: int,
        save_dir: Path | str
    ) -> None:
        """Register image files to database metadata table.

        Only adds new files (doesn't update existing records).

        Args:
            universe_name: Name of the universe
            image_paths: Dictionary mapping location_id to image path
            year: Year of imagery
            save_dir: Base directory where images are saved
        """
        from ..db.db import duckdb_connection
        from ..settings import settings

        # Create table if needed
        cls._create_metadata_table(universe_name)

        save_dir = Path(save_dir)
        export_path = settings.export_path

        # Build metadata DataFrame
        records = []
        for location_id, file_path in image_paths.items():
            file_path = Path(file_path)

            # Calculate relative path from export_path
            try:
                file_path_rel = file_path.relative_to(export_path / 'imagery')
            except ValueError:
                # If file is not under export_path, use relative to save_dir
                file_path_rel = file_path.relative_to(save_dir.parent)

            # Get file metadata
            if file_path.exists():
                file_size = file_path.stat().st_size
                updated_date = datetime.fromtimestamp(file_path.stat().st_mtime)
                exists = True
            else:
                file_size = None
                updated_date = None
                exists = False

            records.append({
                'location_id': location_id,
                'universe_name': universe_name,
                'year': year,
                'file_type': 'image',
                'file_path_abs': str(file_path.absolute()),
                'file_path_rel': str(file_path_rel),
                'file_size': file_size,
                'exists': exists,
                'updated_date': updated_date
            })

        if not records:
            logger.info("No image files to register in database")
            return

        metadata_df = pd.DataFrame(records)

        with duckdb_connection() as db_con:
            # Upsert: Insert new records or update if anything changed
            db_con.register('_tmp_metadata', metadata_df)
            try:
                # Count existing records before upsert to calculate inserts
                existing_count = db_con.execute(f"""
                    SELECT COUNT(*)
                    FROM {universe_name}.location_year_files t
                    JOIN _tmp_metadata m
                        ON t.location_id = m.location_id
                        AND t.year = m.year
                        AND t.file_type = m.file_type
                """).fetchone()[0]

                db_con.execute(f"""
                    INSERT INTO {universe_name}.location_year_files
                        (location_id, universe_name, year, file_type, file_path_abs,
                         file_path_rel, file_size, exists, updated_date)
                    SELECT location_id, universe_name, year, file_type, file_path_abs,
                           file_path_rel, file_size, exists, updated_date
                    FROM _tmp_metadata
                    ON CONFLICT (location_id, year, file_type)
                    DO UPDATE SET
                        file_path_abs = excluded.file_path_abs,
                        file_path_rel = excluded.file_path_rel,
                        file_size = excluded.file_size,
                        exists = excluded.exists,
                        updated_date = excluded.updated_date
                    WHERE
                        {universe_name}.location_year_files.file_path_abs != excluded.file_path_abs OR
                        {universe_name}.location_year_files.file_size != excluded.file_size OR
                        {universe_name}.location_year_files.exists != excluded.exists OR
                        {universe_name}.location_year_files.updated_date != excluded.updated_date
                """)

                # Log statistics
                new_records = len(metadata_df) - existing_count
                logger.info(
                    f"Processed {len(metadata_df)} files: "
                    f"~{new_records} new, ~{existing_count} existing (may have updated)"
                )
            finally:
                db_con.unregister('_tmp_metadata')

    @classmethod
    def sync_metadata_from_disk(
        cls,
        universe_name: str,
        base_path: Path | str,
        years: list[int] | None = None,
        skip_download: bool = True
    ) -> pd.DataFrame:
        """Sync database metadata from existing files on disk.

        This method scans disk for existing image files and updates the database
        metadata table without downloading any new images.

        Args:
            universe_name: Name of the universe
            base_path: Base path to imagery directory (e.g., EXPORT_PATH/imagery/lion)
            years: List of years to sync (defaults to all subdirectories that look like years)
            skip_download: If True, only register existing files (default: True)

        Returns:
            DataFrame of registered file metadata

        Example:
            # Sync all years found on disk
            ImageryLoader.sync_metadata_from_disk(
                universe_name='lion',
                base_path=settings.export_path / 'imagery' / 'lion'
            )

            # Sync specific years
            ImageryLoader.sync_metadata_from_disk(
                universe_name='lion',
                base_path=settings.export_path / 'imagery' / 'lion',
                years=[2020, 2022, 2024]
            )
        """
        from ..settings import settings

        base_path = Path(base_path)

        if not base_path.exists():
            raise FileNotFoundError(f"Base path does not exist: {base_path}")

        # Auto-detect years from directory structure if not provided
        if years is None:
            years = []
            for item in base_path.iterdir():
                if item.is_dir() and item.name.isdigit():
                    years.append(int(item.name))
            years.sort()
            logger.info(f"Auto-detected years: {years}")

        if not years:
            logger.warning(f"No years to sync in {base_path}")
            return pd.DataFrame()

        # Scan each year directory for image files
        all_records = []

        with tqdm(total=len(years), desc="Syncing years", unit="year") as year_pbar:
            for year in years:
                year_dir = base_path / str(year)

                if not year_dir.exists():
                    logger.warning(f"Year directory does not exist: {year_dir}")
                    year_pbar.update(1)
                    continue

                # Find all PNG files (assumes {location_id}.png naming)
                image_files = list(year_dir.glob('*.png'))
                logger.info(f"Found {len(image_files)} images for year {year}")

                # Build image_paths dict with progress bar
                image_paths = {}
                with tqdm(total=len(image_files), desc=f"Scanning {year}", unit="file", leave=False) as file_pbar:
                    for img_file in image_files:
                        # Extract location_id from filename
                        try:
                            location_id = int(img_file.stem)
                            image_paths[location_id] = img_file
                        except ValueError:
                            logger.warning(f"Skipping file with non-numeric name: {img_file.name}")
                        finally:
                            file_pbar.update(1)

                # Register files to database
                if image_paths:
                    cls._register_files_to_db(
                        universe_name=universe_name,
                        image_paths=image_paths,
                        year=year,
                        save_dir=year_dir
                    )
                    all_records.extend(image_paths.keys())

                year_pbar.set_postfix({'files': len(image_paths), 'total': len(all_records)})
                year_pbar.update(1)

        logger.info(f"Synced {len(all_records)} total files across {len(years)} years")

        # Return summary of registered files
        from ..db.db import duckdb_connection
        with duckdb_connection() as db_con:
            result_df = db_con.execute(f"""
                SELECT *
                FROM {universe_name}.location_year_files
                WHERE year IN ({','.join(map(str, years))})
                AND file_type = 'image'
                ORDER BY year, location_id
            """).df()

        return result_df
