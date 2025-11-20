from contextlib import contextmanager
import duckdb
from shapely import wkt
import geopandas as gpd

from ..settings import settings

def get_duckdb_path() -> str:
    return str(settings.ddb_path)

@contextmanager
def duckdb_connection(read_only: bool = False):
    con = duckdb.connect(get_duckdb_path(), read_only=read_only)
    try:
        con.execute("INSTALL spatial;")
        con.execute("LOAD spatial;")  # if you always want spatial
        yield con
    finally:
        con.close()

def load_wkt_gdf(con: duckdb.DuckDBPyConnection, table_name: str, geom_col: str = "geometry", crs: str = "EPSG:4326", table_schema:str|None = None):
    if table_schema:
        full_name = f'{table_schema}.{table_name}'
    else:
        full_name = table_name
    
    # Query DuckDB table into a pandas DataFrame
    df = con.sql(f"SELECT *, {geom_col} FROM {full_name}").df()

    # Convert WKT → shapely geometry
    df[geom_col] = df[geom_col].apply(wkt.loads)

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=geom_col, crs=crs)
    return gdf