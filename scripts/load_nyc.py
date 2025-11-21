# Script that loads universe for NYC and pulls imagery for various years
from argparse import ArgumentParser

from st_preprocessing.data_loader import DataLoader
from st_preprocessing.locations import UniverseLoader, sources
from st_preprocessing.imagery.imagery_loader import ImageryLoader
from st_preprocessing.imagery.source import NewYork
from st_preprocessing.imagery.source_strategies import TileStitchingSource



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--years', '-y', nargs='+')
    args = parser.parse_args()

    universe = DataLoader.load(modality='universe', source='lion', progress=True, with_geometries=True, from_db=True)
    print(universe.locations)
    print(universe.location_geometries)

    # for year in args.years:
    #     year = int(year)
        
    #     try:
    #         nyc_source = NewYork(year=year)
    #         tile_url = nyc_source.get_tile_url()

    #         print('Pulling images for {year}..')
    #         source = TileStitchingSource(
    #             tile_provider=tile_url,
    #             cache_dir='../data/.tile_cache'
    #         )

    #         image_paths = ImageryLoader.load_from_universe(
    #             universe=universe,
    #             source=source,
    #             save_dir=f'../data/{universe.name}/{nyc_source.year}',
    #             n_workers=16
    #         )
    #     except Exception as e:
    #         print(e)