from contextlib import contextmanager
import duckdb

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