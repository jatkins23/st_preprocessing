# Script that loads universe for NYC and pulls imagery for various years
from argparse import ArgumentParser
import logging
import gc

from st_preprocessing.data_loader import DataLoader
from st_preprocessing.locations import UniverseLoader, sources
from st_preprocessing.imagery.imagery_loader import ImageryLoader
from st_preprocessing.imagery.source import NewYork
from st_preprocessing.imagery.source_strategies import TileStitchingSource
from st_preprocessing.settings import settings

from pathlib import Path

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename='./logs/load_nyc.log')

    parser = ArgumentParser()
    parser.add_argument('--years', '-y', nargs='+')
    parser.add_argument('--reload-locations', '-r', action='store_true')
    parser.add_argument('--sync-only', '-s', action='store_true')
    parser.add_argument('--universe','-u', type=str, default='lion')
    parser.add_argument('--outdir', '-o', type=Path, default = settings.export_path / 'imagery' / 'lion')
    args = parser.parse_args()

    #universe = DataLoader.load(modality='universe', source='lion', progress=True, with_geometries=True, from_db=True)
    if args.reload_locations:
        universe = DataLoader.load(modality='universe', source='lion', progress=True, with_geometries=True)
    else:
        universe = DataLoader.load(modality='universe', source='lion', from_db=True)

    if not args.sync_only:
        for year in args.years:
            year = int(year)
            source = None

            try:
                nyc_source = NewYork(year=year)
                tile_url = nyc_source.get_tile_url()

                print(f'Pulling images for {year}..')
                source = TileStitchingSource(
                    tile_provider=tile_url,
                    cache_dir=args.outdir / str(nyc_source.year) / '.tile_cache'
                )

                image_paths = ImageryLoader.load_from_universe(
                    universe=universe,
                    source=source,
                    save_dir=args.outdir / str(nyc_source.year),
                    n_workers=8
                )
            except Exception as e:
                print(e)
            finally:
                # Explicitly cleanup session and connection pools to avoid semaphore leaks
                if source is not None:
                    source.cleanup()
                    del source  # Force deletion to trigger __del__
                    gc.collect()  # Force garbage collection to cleanup semaphores

        
    ImageryLoader.sync_metadata_from_disk(
        universe_name='lion',
        base_path=args.outdir,
        years=args.years
    )