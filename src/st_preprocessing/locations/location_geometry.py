from pydantic import BaseModel, field_serializer, field_validator

import mercantile
from pyproj import Transformer


class LocationGeometry(BaseModel):
    location_id: int
    centroid: tuple[float, float]
    tile_width: int                           = 3
    zlevel: int                               = 20 
    proj_crs: str                             = 'EPSG:2263'
    
    # Derived & Cacheable
    bounds_gcs:  list[float] | None           = None
    bounds_proj: list[float] | None           = None
    center_tile: mercantile.Tile | None       = None
    centroid_p: tuple[float, float] | None    = None
    tile_grid: list[mercantile.Tile] | None   = None
    
    # Post-init computes the derived fields
    def model_post_init(self, __context):
        # Only compute if missing so that I can load from the cache when necessary

        # Tiles
        self.center_tile = self.center_tile or mercantile.tile(self.centroid[0], self.centroid[1], self.zlevel)
        self.tile_grid = self.tile_grid or self._generate_tile_grid_from_center_tile(self.center_tile, self.tile_width)
        
        # generate_bounds
        self.bounds_gcs = self.bounds_gcs or self._get_geometric_bounds_from_tile_grid(self.tile_grid) 
        self.bounds_proj = self.bounds_proj or self._get_projected_bounds_from_geometric_bounds(self.bounds_gcs, self.proj_crs)

        if self.centroid_p is None:
            # project centroid
            transformer = Transformer.from_crs("EPSG:4326", self.proj_crs, always_xy=True)
            self.centroid_p = transformer.transform(self.centroid[0], self.centroid[1])


    @staticmethod
    def _generate_tile_grid_from_center_tile(center_tile:mercantile.Tile, tile_width: int) -> list[mercantile.Tile]:
        def _centered_range(n:int) -> list[int]:
            if n % 2 == 0:
                raise ValueError(f"n must be odd, got {n}")
            start = -(n // 2)
            end = n // 2 + 1
            return list(range(start, end))
        return [
            mercantile.Tile(x=center_tile.x + dx, y=center_tile.y + dy, z=center_tile.z)
            for dy in _centered_range(tile_width)
            for dx in _centered_range(tile_width)
        ]

    @staticmethod
    def _get_geometric_bounds_from_tile_grid(tile_grid):
        def _get_bounds_from_gridboxes(grid_bboxes):
            # Merge the bounding boxes for each tile one bounding box
            west   = min(b.west for b in grid_bboxes)
            south  = min(b.south for b in grid_bboxes)
            east   = max(b.east for b in grid_bboxes)
            north  = max(b.north for b in grid_bboxes)

            return [west, south, east, north]

        grid_bboxes = [mercantile.bounds(t) for t in tile_grid]

        bbox = _get_bounds_from_gridboxes(grid_bboxes)
        return bbox
        
        
    def _get_projected_bounds_from_geometric_bounds(bbox:list[float], output_crs):
        transformer = Transformer.from_crs("EPSG:4326", output_crs, always_xy=True)
        minx, miny = transformer.transform(bbox[0], bbox[1])
        maxx, maxy = transformer.transform(bbox[2], bbox[3])
        
        return [minx, miny, maxx, maxy]

    # Serializers - for saving to database
    @field_serializer('center_tile')
    def _serialize_center_tile(self, t: mercantile.Tile|None, _info):
        if t is None: 
            return None
        return {'x': t.x, 'y': t.y, 'z': t.z}
    
    @field_serializer('tile_grid')
    def _serialize_tile_grid(self, tiles: list[mercantile.Tile]|None, _info):
        if tiles is None: 
            return None
        return [{'x': t.x, 'y': t.y, 'z': t.z} for t in tiles]
    
    # Validators can allow loading back from dicts
    @field_validator('center_tile', mode='before')
    @classmethod
    def _validate_center_tile(cls, v):
        if v is None or isinstance(v, mercantile.Tile): 
            return v
        return mercantile.Tile(x=int(v['x']), y=int(v['y']), z=int(v['z']))
    
    @field_validator('tile_grid', mode='before')
    @classmethod
    def _validate_tile_grid(cls, v):
        if v is None or (v and isinstance(v[0], mercantile.Tile)):
            return v
        return [mercantile.Tile(x=int(d['x']), y=int(d['y']), z=int(d['z'])) for d in v]

    

