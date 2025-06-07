#!/usr/bin/env python3
"""
Data Storage and Backup Module

Implements Phase 3.6: Data Storage and Backup

This module provides comprehensive data storage and backup capabilities including:
- Automated backup systems
- Version control for datasets
- Data integrity verification
- Multiple storage format support
- Disaster recovery procedures
- Data archiving and compression
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import pickle
import gzip
import shutil
import hashlib
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime, timedelta
import os
import tempfile
import warnings
from dataclasses import dataclass
from enum import Enum

# Handle optional parquet dependencies
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    warnings.warn("Parquet support not available. Install pyarrow for parquet functionality.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StorageFormat(Enum):
    """Supported storage formats."""
    CSV = "csv"
    PARQUET = "parquet"
    PICKLE = "pickle"
    JSON = "json"
    SQLITE = "sqlite"
    EXCEL = "excel"
    HDF5 = "hdf5"

class CompressionType(Enum):
    """Supported compression types."""
    NONE = "none"
    GZIP = "gzip"
    ZIP = "zip"
    BZIP2 = "bz2"
    XZ = "xz"

@dataclass
class BackupMetadata:
    """Metadata for backup files."""
    timestamp: datetime
    file_path: str
    file_size: int
    checksum: str
    compression: str
    format: str
    description: str
    version: str
    tags: List[str]

class DataStorageManager:
    """
    Manages data storage operations with multiple format support.
    
    Provides functionality for:
    - Saving data in various formats
    - Loading data with format detection
    - Data compression and decompression
    - Metadata management
    """
    
    def __init__(self, base_storage_path: str = "data/storage"):
        self.base_path = Path(base_storage_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.base_path / "storage_metadata.json"
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict[str, Any]:
        """Load storage metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"files": {}, "last_updated": None}
    
    def _save_metadata(self) -> None:
        """Save storage metadata."""
        self.metadata["last_updated"] = datetime.now().isoformat()
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def save_data(self, data: pd.DataFrame, filename: str, 
                  storage_format: StorageFormat = StorageFormat.PARQUET,
                  compression: CompressionType = CompressionType.GZIP,
                  description: str = "", tags: List[str] = None,
                  version: str = "1.0") -> str:
        """
        Save data in specified format with optional compression.
        
        Args:
            data: DataFrame to save
            filename: Base filename (without extension)
            storage_format: Storage format
            compression: Compression type
            description: Description of the data
            tags: Tags for categorization
            version: Version string
            
        Returns:
            Path to saved file
        """
        if tags is None:
            tags = []
        
        # Create filename with extension
        file_extension = storage_format.value
        if storage_format == StorageFormat.PICKLE:
            file_extension = "pkl"
        elif storage_format == StorageFormat.HDF5:
            file_extension = "h5"
        
        base_filename = f"{filename}.{file_extension}"
        file_path = self.base_path / base_filename
        
        # Save data in specified format
        try:
            if storage_format == StorageFormat.CSV:
                data.to_csv(file_path, index=False)
            elif storage_format == StorageFormat.PARQUET:
                if not PARQUET_AVAILABLE:
                    raise ImportError("Parquet support not available. Install pyarrow for parquet functionality.")
                data.to_parquet(file_path, index=False)
            elif storage_format == StorageFormat.PICKLE:
                data.to_pickle(file_path)
            elif storage_format == StorageFormat.JSON:
                data.to_json(file_path, orient='records', indent=2)
            elif storage_format == StorageFormat.EXCEL:
                data.to_excel(file_path, index=False)
            elif storage_format == StorageFormat.SQLITE:
                conn = sqlite3.connect(file_path)
                data.to_sql('data', conn, if_exists='replace', index=False)
                conn.close()
            elif storage_format == StorageFormat.HDF5:
                data.to_hdf(file_path, key='data', mode='w')
            else:
                raise ValueError(f"Unsupported storage format: {storage_format}")
            
            # Apply compression if specified
            final_path = file_path
            if compression != CompressionType.NONE:
                final_path = self._compress_file(file_path, compression)
                if final_path != file_path:
                    os.remove(file_path)  # Remove uncompressed file
            
            # Calculate metadata
            file_size = final_path.stat().st_size
            checksum = self._calculate_checksum(final_path)
            
            # Store metadata
            metadata = BackupMetadata(
                timestamp=datetime.now(),
                file_path=str(final_path),
                file_size=file_size,
                checksum=checksum,
                compression=compression.value,
                format=storage_format.value,
                description=description,
                version=version,
                tags=tags
            )
            
            self.metadata["files"][str(final_path)] = {
                "timestamp": metadata.timestamp.isoformat(),
                "file_size": metadata.file_size,
                "checksum": metadata.checksum,
                "compression": metadata.compression,
                "format": metadata.format,
                "description": metadata.description,
                "version": metadata.version,
                "tags": metadata.tags
            }
            
            self._save_metadata()
            
            logger.info(f"Data saved successfully to {final_path}")
            return str(final_path)
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise
    
    def _compress_file(self, file_path: Path, compression: CompressionType) -> Path:
        """Compress a file using specified compression type."""
        if compression == CompressionType.GZIP:
            compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return compressed_path
        
        elif compression == CompressionType.ZIP:
            compressed_path = file_path.with_suffix(file_path.suffix + '.zip')
            with zipfile.ZipFile(compressed_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(file_path, file_path.name)
            return compressed_path
        
        elif compression == CompressionType.BZIP2:
            compressed_path = file_path.with_suffix(file_path.suffix + '.bz2')
            with open(file_path, 'rb') as f_in:
                with open(compressed_path, 'wb') as f_out:
                    import bz2
                    f_out.write(bz2.compress(f_in.read()))
            return compressed_path
        
        else:
            return file_path
    
    def load_data(self, file_path: str, verify_checksum: bool = True) -> pd.DataFrame:
        """
        Load data from file with automatic format detection.
        
        Args:
            file_path: Path to file
            verify_checksum: Whether to verify file integrity
            
        Returns:
            Loaded DataFrame
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Verify checksum if requested
        if verify_checksum and str(file_path) in self.metadata["files"]:
            stored_checksum = self.metadata["files"][str(file_path)]["checksum"]
            current_checksum = self._calculate_checksum(file_path)
            if stored_checksum != current_checksum:
                logger.warning(f"Checksum mismatch for {file_path}. File may be corrupted.")
        
        # Decompress if necessary
        temp_file = None
        actual_file_path = file_path
        
        if file_path.suffix == '.gz':
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            with gzip.open(file_path, 'rb') as f_in:
                shutil.copyfileobj(f_in, temp_file)
            temp_file.close()
            actual_file_path = Path(temp_file.name)
        
        elif file_path.suffix == '.zip':
            temp_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(file_path, 'r') as zipf:
                zipf.extractall(temp_dir)
                # Assume single file in zip
                extracted_files = list(Path(temp_dir).iterdir())
                if extracted_files:
                    actual_file_path = extracted_files[0]
        
        try:
            # Detect format and load
            # For compressed files, get the original extension
            if file_path.suffix in ['.gz', '.zip', '.bz2']:
                # Get the extension before compression
                original_name = file_path.stem
                suffix = Path(original_name).suffix.lower()
            else:
                suffix = actual_file_path.suffix.lower()
            
            if suffix == '.csv':
                data = pd.read_csv(actual_file_path)
            elif suffix == '.parquet':
                if not PARQUET_AVAILABLE:
                    raise ImportError("Parquet support not available. Install pyarrow for parquet functionality.")
                data = pd.read_parquet(actual_file_path)
            elif suffix in ['.pkl', '.pickle']:
                data = pd.read_pickle(actual_file_path)
            elif suffix == '.json':
                data = pd.read_json(actual_file_path)
            elif suffix in ['.xlsx', '.xls']:
                data = pd.read_excel(actual_file_path)
            elif suffix in ['.db', '.sqlite']:
                conn = sqlite3.connect(actual_file_path)
                data = pd.read_sql_query("SELECT * FROM data", conn)
                conn.close()
            elif suffix == '.h5':
                data = pd.read_hdf(actual_file_path, key='data')
            else:
                raise ValueError(f"Unsupported file format: {suffix}")
            
            logger.info(f"Data loaded successfully from {file_path}")
            return data
            
        finally:
            # Clean up temporary files
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
    
    def list_stored_files(self, tags: List[str] = None, format_filter: str = None) -> List[Dict[str, Any]]:
        """
        List stored files with optional filtering.
        
        Args:
            tags: Filter by tags
            format_filter: Filter by format
            
        Returns:
            List of file metadata
        """
        files = []
        
        for file_path, metadata in self.metadata["files"].items():
            # Apply filters
            if tags and not any(tag in metadata.get("tags", []) for tag in tags):
                continue
            
            if format_filter and metadata.get("format") != format_filter:
                continue
            
            files.append({
                "file_path": file_path,
                "timestamp": metadata["timestamp"],
                "file_size": metadata["file_size"],
                "format": metadata["format"],
                "compression": metadata["compression"],
                "description": metadata["description"],
                "version": metadata["version"],
                "tags": metadata["tags"]
            })
        
        return sorted(files, key=lambda x: x["timestamp"], reverse=True)


class DataBackupManager:
    """
    Manages automated data backup operations.
    
    Provides functionality for:
    - Scheduled backups
    - Incremental backups
    - Backup rotation and cleanup
    - Disaster recovery
    """
    
    def __init__(self, backup_base_path: str = "data/backups", 
                 storage_manager: DataStorageManager = None):
        self.backup_path = Path(backup_base_path)
        self.backup_path.mkdir(parents=True, exist_ok=True)
        self.storage_manager = storage_manager or DataStorageManager()
        self.backup_log_file = self.backup_path / "backup_log.json"
        self.backup_log = self._load_backup_log()
        
    def _load_backup_log(self) -> Dict[str, Any]:
        """Load backup log."""
        if self.backup_log_file.exists():
            with open(self.backup_log_file, 'r') as f:
                return json.load(f)
        return {"backups": [], "last_backup": None, "backup_count": 0}
    
    def _save_backup_log(self) -> None:
        """Save backup log."""
        with open(self.backup_log_file, 'w') as f:
            json.dump(self.backup_log, f, indent=2, default=str)
    
    def create_backup(self, data: pd.DataFrame, backup_name: str, 
                     description: str = "", tags: List[str] = None,
                     compression: CompressionType = CompressionType.GZIP) -> str:
        """
        Create a backup of the data.
        
        Args:
            data: DataFrame to backup
            backup_name: Name for the backup
            description: Description of the backup
            tags: Tags for categorization
            compression: Compression type
            
        Returns:
            Path to backup file
        """
        if tags is None:
            tags = ["backup"]
        else:
            tags = tags + ["backup"]
        
        # Create timestamped backup name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{backup_name}_{timestamp}"
        
        # Save backup
        backup_path = self.storage_manager.save_data(
            data=data,
            filename=backup_filename,
            storage_format=StorageFormat.CSV,
            compression=compression,
            description=f"Backup: {description}",
            tags=tags,
            version="backup"
        )
        
        # Update backup log
        backup_entry = {
            "backup_name": backup_name,
            "timestamp": datetime.now().isoformat(),
            "file_path": backup_path,
            "description": description,
            "tags": tags,
            "data_shape": list(data.shape),
            "data_size_mb": round(data.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        }
        
        self.backup_log["backups"].append(backup_entry)
        self.backup_log["last_backup"] = datetime.now().isoformat()
        self.backup_log["backup_count"] += 1
        
        self._save_backup_log()
        
        logger.info(f"Backup created successfully: {backup_path}")
        return backup_path
    
    def create_incremental_backup(self, current_data: pd.DataFrame, 
                                 reference_backup_name: str,
                                 backup_name: str, description: str = "") -> Optional[str]:
        """
        Create incremental backup (only if data has changed).
        
        Args:
            current_data: Current data to backup
            reference_backup_name: Name of reference backup to compare against
            backup_name: Name for new backup
            description: Description of the backup
            
        Returns:
            Path to backup file if created, None if no changes detected
        """
        # Find latest backup with reference name
        reference_backup = None
        for backup in reversed(self.backup_log["backups"]):
            if backup["backup_name"] == reference_backup_name:
                reference_backup = backup
                break
        
        if reference_backup is None:
            logger.info(f"No reference backup found for {reference_backup_name}. Creating full backup.")
            return self.create_backup(current_data, backup_name, description, ["incremental"])
        
        # Load reference data
        try:
            reference_data = self.storage_manager.load_data(reference_backup["file_path"])
            
            # Compare data
            if current_data.equals(reference_data):
                logger.info("No changes detected. Skipping incremental backup.")
                return None
            
            # Data has changed, create backup
            logger.info("Changes detected. Creating incremental backup.")
            return self.create_backup(current_data, backup_name, 
                                    f"Incremental: {description}", ["incremental"])
            
        except Exception as e:
            logger.error(f"Error during incremental backup: {str(e)}")
            logger.info("Creating full backup instead.")
            return self.create_backup(current_data, backup_name, description, ["incremental", "fallback"])
    
    def restore_backup(self, backup_name: str, timestamp: str = None) -> pd.DataFrame:
        """
        Restore data from backup.
        
        Args:
            backup_name: Name of backup to restore
            timestamp: Specific timestamp (if None, uses latest)
            
        Returns:
            Restored DataFrame
        """
        # Find matching backup
        matching_backups = [
            backup for backup in self.backup_log["backups"]
            if backup["backup_name"] == backup_name
        ]
        
        if not matching_backups:
            raise ValueError(f"No backup found with name: {backup_name}")
        
        if timestamp:
            # Find backup with specific timestamp
            target_backup = None
            for backup in matching_backups:
                if backup["timestamp"].startswith(timestamp):
                    target_backup = backup
                    break
            
            if target_backup is None:
                raise ValueError(f"No backup found with timestamp: {timestamp}")
        else:
            # Use latest backup
            target_backup = max(matching_backups, key=lambda x: x["timestamp"])
        
        # Load and return data
        logger.info(f"Restoring backup: {target_backup['file_path']}")
        return self.storage_manager.load_data(target_backup["file_path"])
    
    def cleanup_old_backups(self, retention_days: int = 30, max_backups: int = 50) -> None:
        """
        Clean up old backups based on retention policy.
        
        Args:
            retention_days: Number of days to retain backups
            max_backups: Maximum number of backups to keep
        """
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        # Group backups by name
        backup_groups = {}
        for backup in self.backup_log["backups"]:
            name = backup["backup_name"]
            if name not in backup_groups:
                backup_groups[name] = []
            backup_groups[name].append(backup)
        
        backups_to_remove = []
        
        for backup_name, backups in backup_groups.items():
            # Sort by timestamp (newest first)
            backups.sort(key=lambda x: x["timestamp"], reverse=True)
            
            # Keep recent backups and limit total count
            for i, backup in enumerate(backups):
                backup_date = datetime.fromisoformat(backup["timestamp"])
                
                # Remove if too old or exceeds max count
                if backup_date < cutoff_date or i >= max_backups:
                    backups_to_remove.append(backup)
        
        # Remove old backup files and log entries
        for backup in backups_to_remove:
            try:
                file_path = Path(backup["file_path"])
                if file_path.exists():
                    os.remove(file_path)
                
                self.backup_log["backups"].remove(backup)
                logger.info(f"Removed old backup: {backup['file_path']}")
                
            except Exception as e:
                logger.error(f"Error removing backup {backup['file_path']}: {str(e)}")
        
        self._save_backup_log()
        logger.info(f"Cleanup completed. Removed {len(backups_to_remove)} old backups.")
    
    def get_backup_status(self) -> Dict[str, Any]:
        """
        Get comprehensive backup status.
        
        Returns:
            Dictionary with backup status information
        """
        total_size = 0
        backup_names = set()
        
        for backup in self.backup_log["backups"]:
            backup_names.add(backup["backup_name"])
            try:
                file_path = Path(backup["file_path"])
                if file_path.exists():
                    total_size += file_path.stat().st_size
            except Exception:
                pass
        
        return {
            "total_backups": len(self.backup_log["backups"]),
            "unique_backup_names": len(backup_names),
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "last_backup": self.backup_log.get("last_backup"),
            "backup_names": list(backup_names)
        }


class DataVersionControl:
    """
    Provides version control functionality for datasets.
    
    Tracks changes, maintains version history, and enables
    rollback to previous versions.
    """
    
    def __init__(self, version_base_path: str = "data/versions"):
        self.version_path = Path(version_base_path)
        self.version_path.mkdir(parents=True, exist_ok=True)
        self.version_log_file = self.version_path / "version_log.json"
        self.version_log = self._load_version_log()
        self.storage_manager = DataStorageManager(str(self.version_path))
        
    def _load_version_log(self) -> Dict[str, Any]:
        """Load version log."""
        if self.version_log_file.exists():
            with open(self.version_log_file, 'r') as f:
                return json.load(f)
        return {"datasets": {}, "last_updated": None}
    
    def _save_version_log(self) -> None:
        """Save version log."""
        self.version_log["last_updated"] = datetime.now().isoformat()
        with open(self.version_log_file, 'w') as f:
            json.dump(self.version_log, f, indent=2, default=str)
    
    def commit_version(self, data: pd.DataFrame, dataset_name: str, 
                      commit_message: str, author: str = "system") -> str:
        """
        Commit a new version of the dataset.
        
        Args:
            data: DataFrame to version
            dataset_name: Name of the dataset
            commit_message: Description of changes
            author: Author of the changes
            
        Returns:
            Version identifier
        """
        # Initialize dataset if not exists
        if dataset_name not in self.version_log["datasets"]:
            self.version_log["datasets"][dataset_name] = {
                "versions": [],
                "current_version": None
            }
        
        dataset_info = self.version_log["datasets"][dataset_name]
        
        # Generate version number
        version_number = len(dataset_info["versions"]) + 1
        version_id = f"v{version_number:03d}"
        
        # Save versioned data
        version_filename = f"{dataset_name}_{version_id}"
        file_path = self.storage_manager.save_data(
            data=data,
            filename=version_filename,
            storage_format=StorageFormat.PARQUET,
            compression=CompressionType.GZIP,
            description=f"Version {version_id}: {commit_message}",
            tags=["version", dataset_name],
            version=version_id
        )
        
        # Create version entry
        version_entry = {
            "version_id": version_id,
            "timestamp": datetime.now().isoformat(),
            "file_path": file_path,
            "commit_message": commit_message,
            "author": author,
            "data_shape": list(data.shape),
            "checksum": self.storage_manager._calculate_checksum(Path(file_path))
        }
        
        # Update version log
        dataset_info["versions"].append(version_entry)
        dataset_info["current_version"] = version_id
        
        self._save_version_log()
        
        logger.info(f"Version {version_id} committed for dataset {dataset_name}")
        return version_id
    
    def get_version(self, dataset_name: str, version_id: str = None) -> pd.DataFrame:
        """
        Get specific version of dataset.
        
        Args:
            dataset_name: Name of the dataset
            version_id: Version identifier (if None, gets current version)
            
        Returns:
            DataFrame for the specified version
        """
        if dataset_name not in self.version_log["datasets"]:
            raise ValueError(f"Dataset not found: {dataset_name}")
        
        dataset_info = self.version_log["datasets"][dataset_name]
        
        if version_id is None:
            version_id = dataset_info["current_version"]
        
        # Find version
        version_entry = None
        for version in dataset_info["versions"]:
            if version["version_id"] == version_id:
                version_entry = version
                break
        
        if version_entry is None:
            raise ValueError(f"Version not found: {version_id}")
        
        # Load and return data
        return self.storage_manager.load_data(version_entry["file_path"])
    
    def list_versions(self, dataset_name: str) -> List[Dict[str, Any]]:
        """
        List all versions of a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            List of version information
        """
        if dataset_name not in self.version_log["datasets"]:
            return []
        
        return self.version_log["datasets"][dataset_name]["versions"]
    
    def diff_versions(self, dataset_name: str, version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare two versions of a dataset.
        
        Args:
            dataset_name: Name of the dataset
            version1: First version to compare
            version2: Second version to compare
            
        Returns:
            Dictionary with comparison results
        """
        data1 = self.get_version(dataset_name, version1)
        data2 = self.get_version(dataset_name, version2)
        
        comparison = {
            "shape_changes": {
                "version1_shape": list(data1.shape),
                "version2_shape": list(data2.shape),
                "rows_added": data2.shape[0] - data1.shape[0],
                "columns_added": data2.shape[1] - data1.shape[1]
            },
            "column_changes": {
                "added_columns": list(set(data2.columns) - set(data1.columns)),
                "removed_columns": list(set(data1.columns) - set(data2.columns)),
                "common_columns": list(set(data1.columns) & set(data2.columns))
            },
            "data_changes": {}
        }
        
        # Compare common columns
        common_cols = comparison["column_changes"]["common_columns"]
        if common_cols and data1.shape[0] == data2.shape[0]:
            for col in common_cols:
                if pd.api.types.is_numeric_dtype(data1[col]) and pd.api.types.is_numeric_dtype(data2[col]):
                    comparison["data_changes"][col] = {
                        "values_changed": not data1[col].equals(data2[col]),
                        "mean_change": float(data2[col].mean() - data1[col].mean()) if data1[col].notna().any() and data2[col].notna().any() else None
                    }
        
        return comparison


def main():
    """
    Main function for testing storage and backup functionality.
    """
    # Example usage
    db_path = "data/raw/score.db"
    
    # Load data
    conn = sqlite3.connect(db_path)
    data = pd.read_sql_query("SELECT * FROM student_scores", conn)
    conn.close()
    
    # Initialize managers
    storage_manager = DataStorageManager("data/storage")
    backup_manager = DataBackupManager("data/backups", storage_manager)
    version_control = DataVersionControl("data/versions")
    
    # Test storage
    logger.info("Testing data storage...")
    storage_path = storage_manager.save_data(
        data, "student_scores_processed", 
        StorageFormat.PARQUET, CompressionType.GZIP,
        "Processed student scores data", ["processed", "scores"]
    )
    
    # Test backup
    logger.info("Testing data backup...")
    backup_path = backup_manager.create_backup(
        data, "student_scores", "Initial backup of student scores"
    )
    
    # Test version control
    logger.info("Testing version control...")
    version_id = version_control.commit_version(
        data, "student_scores", "Initial version", "data_processor"
    )
    
    # Print status
    backup_status = backup_manager.get_backup_status()
    stored_files = storage_manager.list_stored_files()
    
    print("\nStorage and Backup completed successfully!")
    print(f"Stored files: {len(stored_files)}")
    print(f"Total backups: {backup_status['total_backups']}")
    print(f"Version committed: {version_id}")


if __name__ == "__main__":
    main()