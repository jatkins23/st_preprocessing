"""
Storage backends for geocoding results.

Implements DuckDB and CSV storage for successful results and errors
with full error message details.
"""


import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

import pandas as pd
import duckdb

from .base import StorageBackend
from .models import GeocodeResult, GeocodeError


logger = logging.getLogger(__name__)


class DuckDBStorageBackend(StorageBackend):
    """
    DuckDB storage backend for geocoding results.
    
    Stores successful results and errors in separate tables with full details.
    """
    
    DDL_SUCCESS = """
    CREATE TABLE IF NOT EXISTS geocode_results (
        unique_key TEXT PRIMARY KEY,
        intersection_hash TEXT,
        street1 TEXT,
        street2 TEXT,
        borough TEXT,
        lon DOUBLE,
        lat DOUBLE,
        source TEXT,
        status TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    DDL_ERRORS = """
    CREATE TABLE IF NOT EXISTS geocode_errors (
        unique_key TEXT,
        intersection_hash TEXT,
        street1 TEXT,
        street2 TEXT,
        borough TEXT,
        endpoint TEXT,
        http_status INTEGER,
        params_json TEXT,
        body_snippet TEXT,
        error_label TEXT,
        api_message TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    def __init__(self, db_path: Path | str):
        """
        Initialize DuckDB storage.
        
        Args:
            db_path: Path to DuckDB database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.db_path))
        self.con.execute(self.DDL_SUCCESS)
        self.con.execute(self.DDL_ERRORS)
        logger.info(f"Initialized DuckDB storage: {self.db_path}")
    
    def write_success(self, result: GeocodeResult) -> None:
        """Store a successful geocoding result."""
        if not result.is_success():
            raise ValueError(f"Result {result.unique_key} is not successful")
        
        self.con.execute(
            """
            INSERT INTO geocode_results 
            (unique_key, intersection_hash, street1, street2, borough, lon, lat, source, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(unique_key) DO UPDATE SET
                lon = excluded.lon,
                lat = excluded.lat,
                source = excluded.source,
                status = excluded.status
            """,
            [
                result.unique_key,
                result.intersection_hash,
                result.street1,
                result.street2,
                result.borough,
                result.lon,
                result.lat,
                result.source.value,
                result.status.value,
            ]
        )
    
    def write_batch_success(self, results: List[GeocodeResult]) -> None:
        """Store multiple successful results efficiently."""
        if not results:
            return
        
        rows = []
        for result in results:
            if result.is_success():
                rows.append((
                    result.unique_key,
                    result.intersection_hash,
                    result.street1,
                    result.street2,
                    result.borough,
                    result.lon,
                    result.lat,
                    result.source.value,
                    result.status.value,
                ))
        
        if rows:
            df = pd.DataFrame(rows, columns=[
                'unique_key', 'intersection_hash', 'street1', 'street2', 'borough',
                'lon', 'lat', 'source', 'status'
            ])
            self.con.register('batch_success', df)
            self.con.execute(
                """
                INSERT INTO geocode_results 
                SELECT * FROM batch_success
                ON CONFLICT(unique_key) DO UPDATE SET
                    lon = excluded.lon,
                    lat = excluded.lat,
                    source = excluded.source,
                    status = excluded.status,
                    created_at = CURRENT_TIMESTAMP
                """
            )
            self.con.unregister('batch_success')
    
    def write_error(self, result: GeocodeResult) -> None:
        """Store an error/failed geocoding result."""
        if result.is_success():
            raise ValueError(f"Result {result.unique_key} is successful, use write_success")
        
        errors_to_write = result.errors if result.errors else [GeocodeError(
            endpoint="unknown",
            error_label=result.status.value,
            api_message=None
        )]
        
        for error in errors_to_write:
            self.con.execute(
                """
                INSERT INTO geocode_errors 
                (unique_key, intersection_hash, street1, street2, borough,
                 endpoint, http_status, params_json, body_snippet, error_label, api_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    result.unique_key,
                    result.intersection_hash,
                    result.street1,
                    result.street2,
                    result.borough,
                    error.endpoint,
                    error.http_status,
                    error.params_json,
                    error.body_snippet,
                    error.error_label,
                    error.api_message,
                ]
            )
    
    def write_batch_error(self, results: List[GeocodeResult]) -> None:
        """Store multiple error results efficiently."""
        if not results:
            return
        
        rows = []
        for result in results:
            if not result.is_success():
                errors_to_write = result.errors if result.errors else [GeocodeError(
                    endpoint="unknown",
                    error_label=result.status.value
                )]
                for error in errors_to_write:
                    rows.append((
                        result.unique_key,
                        result.intersection_hash,
                        result.street1,
                        result.street2,
                        result.borough,
                        error.endpoint,
                        error.http_status,
                        error.params_json,
                        error.body_snippet,
                        error.error_label,
                        error.api_message,
                    ))
        
        if rows:
            df = pd.DataFrame(rows, columns=[
                'unique_key', 'intersection_hash', 'street1', 'street2', 'borough',
                'endpoint', 'http_status', 'params_json', 'body_snippet',
                'error_label', 'api_message'
            ])
            self.con.register('batch_errors', df)
            self.con.execute(
                "INSERT INTO geocode_errors SELECT * FROM batch_errors"
            )
            self.con.unregister('batch_errors')
    
    def read(self, unique_key: str) -> Optional[GeocodeResult]:
        """Retrieve a stored result by unique key."""
        result = self.con.execute(
            "SELECT * FROM geocode_results WHERE unique_key = ?",
            [unique_key]
        ).fetchall()
        
        if result:
            row = result[0]
            # This is a simplified version - full reconstruction would be more complex
            return GeocodeResult(
                unique_key=row[0],
                intersection_hash=row[1],
                lon=row[4],
                lat=row[5],
            )
        return None
    
    def close(self) -> None:
        """Close database connection."""
        if self.con:
            self.con.close()
            logger.info("Closed DuckDB connection")
    
    def export_errors_to_csv(self, csv_path: Path | str) -> int:
        """
        Export all errors to CSV.
        
        Args:
            csv_path: Path to output CSV file
            
        Returns:
            Number of errors exported
        """
        csv_path = Path(csv_path)
        errors_df = self.con.execute(
            "SELECT * FROM geocode_errors ORDER BY created_at DESC"
        ).df()
        
        if len(errors_df) > 0:
            errors_df.to_csv(csv_path, index=False)
            logger.info(f"Exported {len(errors_df)} errors to {csv_path}")
        
        return len(errors_df)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        result = self.con.execute(
            "SELECT COUNT(*) FROM geocode_results"
        ).fetchone()
        success_count = result[0] if result else 0
        
        result = self.con.execute(
            "SELECT COUNT(DISTINCT unique_key) FROM geocode_errors"
        ).fetchone()
        error_count = result[0] if result else 0
        
        return {
            'successful': success_count,
            'failed': error_count,
            'total_unique_keys': success_count + error_count,
        }


class CSVStorageBackend(StorageBackend):
    """
    CSV storage backend for geocoding results.
    
    Stores results in CSV files for easy inspection.
    """
    
    def __init__(self, success_csv: Path | str, error_csv: Path | str):
        """
        Initialize CSV storage.
        
        Args:
            success_csv: Path to successful results CSV
            error_csv: Path to error results CSV
        """
        self.success_csv = Path(success_csv)
        self.error_csv = Path(error_csv)
        self.success_csv.parent.mkdir(parents=True, exist_ok=True)
        self.error_csv.parent.mkdir(parents=True, exist_ok=True)
        
        self.success_rows: List[Dict[str, Any]] = []
        self.error_rows: List[Dict[str, Any]] = []
        logger.info(f"Initialized CSV storage: {self.success_csv}, {self.error_csv}")
    
    def write_success(self, result: GeocodeResult) -> None:
        """Store a successful geocoding result."""
        if not result.is_success():
            raise ValueError(f"Result {result.unique_key} is not successful")
        
        self.success_rows.append({
            'unique_key': result.unique_key,
            'intersection_hash': result.intersection_hash,
            'street1': result.street1,
            'street2': result.street2,
            'borough': result.borough,
            'lon': result.lon,
            'lat': result.lat,
            'source': result.source.value,
            'status': result.status.value,
            'created_at': datetime.now().isoformat(),
        })
    
    def write_batch_success(self, results: List[GeocodeResult]) -> None:
        """Store multiple successful results."""
        for result in results:
            if result.is_success():
                self.write_success(result)
    
    def write_error(self, result: GeocodeResult) -> None:
        """Store an error/failed geocoding result."""
        if result.is_success():
            raise ValueError(f"Result {result.unique_key} is successful")
        
        errors_to_write = result.errors if result.errors else [GeocodeError(
            endpoint="unknown",
            error_label=result.status.value
        )]
        
        for error in errors_to_write:
            self.error_rows.append({
                'unique_key': result.unique_key,
                'intersection_hash': result.intersection_hash,
                'street1': result.street1,
                'street2': result.street2,
                'borough': result.borough,
                'endpoint': error.endpoint,
                'http_status': error.http_status,
                'error_label': error.error_label,
                'api_message': error.api_message,
                'params_json': error.params_json,
                'body_snippet': error.body_snippet,
                'created_at': datetime.now().isoformat(),
            })
    
    def write_batch_error(self, results: List[GeocodeResult]) -> None:
        """Store multiple error results."""
        for result in results:
            if not result.is_success():
                self.write_error(result)
    
    def read(self, unique_key: str) -> Optional[GeocodeResult]:
        """Retrieve a stored result by unique key."""
        # CSV backend doesn't support efficient reading
        return None
    
    def close(self) -> None:
        """Flush CSV files to disk."""
        if self.success_rows:
            df = pd.DataFrame(self.success_rows)
            df.to_csv(self.success_csv, index=False)
            logger.info(f"Wrote {len(self.success_rows)} successful results to {self.success_csv}")
        
        if self.error_rows:
            df = pd.DataFrame(self.error_rows)
            df.to_csv(self.error_csv, index=False)
            logger.info(f"Wrote {len(self.error_rows)} error results to {self.error_csv}")


class CompositeStorageBackend(StorageBackend):
    """
    Composite storage backend that writes to multiple backends.
    
    Useful for persisting to both DuckDB and CSV simultaneously.
    """
    
    def __init__(self, backends: List[StorageBackend]):
        """
        Initialize composite backend.
        
        Args:
            backends: List of storage backends to write to
        """
        self.backends = backends
        logger.info(f"Initialized composite storage with {len(backends)} backends")
    
    def write_success(self, result: GeocodeResult) -> None:
        """Store result to all backends."""
        for backend in self.backends:
            backend.write_success(result)
    
    def write_batch_success(self, results: List[GeocodeResult]) -> None:
        """Store batch to all backends."""
        for backend in self.backends:
            backend.write_batch_success(results)
    
    def write_error(self, result: GeocodeResult) -> None:
        """Store error to all backends."""
        for backend in self.backends:
            backend.write_error(result)
    
    def write_batch_error(self, results: List[GeocodeResult]) -> None:
        """Store error batch to all backends."""
        for backend in self.backends:
            backend.write_batch_error(results)
    
    def read(self, unique_key: str) -> Optional[GeocodeResult]:
        """Read from first backend that supports it."""
        for backend in self.backends:
            result = backend.read(unique_key)
            if result:
                return result
        return None
    
    def close(self) -> None:
        """Close all backends."""
        for backend in self.backends:
            backend.close()
