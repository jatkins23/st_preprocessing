from pathlib import Path
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    ddb_path: Path = Path(str(os.getenv('DDB_PATH')))
    data_path: Path | None = None
    proj_crs: int | None = None

    class Config: 
        env_prefix = ""
        env_file   = ".env"

settings = Settings()