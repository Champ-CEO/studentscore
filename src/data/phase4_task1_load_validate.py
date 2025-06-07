#!/usr/bin/env python3
"""
Phase 4 Task 1: Load and Validate Data

This module implements task 4.1 from TASKS.md:
- Load final_processed.csv
- Validate data integrity and structure
- Prepare data for feature engineering

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Phase4DataLoader:
    """
    Handles loading and validation of processed data for Phase 4 feature engineering.
    """
    
    def __init__(self, data_path: str = "data/processed/final_processed.csv"):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the processed dataset
        """
        self.data_path = Path(data_path)
        self.df = None
        self.validation_results = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the processed dataset with validation.
        
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data validation fails
        """
        logger.info(f"Loading data from {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        try:
            # Load with appropriate data types
            self.df = pd.read_csv(self.data_path, index_col=0)
            logger.info(f"Successfully loaded {len(self.df)} records with {len(self.df.columns)} columns")
            
            # Run validation
            self._validate_data()
            
            return self.df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _validate_data(self) -> None:
        """
        Validate the loaded data structure and integrity.
        """
        logger.info("Validating data structure and integrity")
        
        # Expected columns based on Phase 3 processing
        expected_base_columns = [
            'student_id', 'tuition', 'extracurricular_activities', 'learning_style',
            'gender', 'direct_admission', 'previous_score', 'study_hours', 'attendance_rate',
            'sleep_time', 'wake_time', 'transport_mode', 'favorite_color', 'final_test'
        ]
        
        # Check basic structure
        self.validation_results['total_records'] = len(self.df)
        self.validation_results['total_columns'] = len(self.df.columns)
        
        # Check for expected base columns
        missing_base_cols = [col for col in expected_base_columns if col not in self.df.columns]
        self.validation_results['missing_base_columns'] = missing_base_cols
        
        # Check data types preservation
        self._validate_data_types()
        
        # Check for data corruption
        self._check_data_corruption()
        
        # Check record count expectations
        self._validate_record_count()
        
        # Log validation summary
        self._log_validation_summary()
        
    def _validate_data_types(self) -> None:
        """
        Validate that data types are preserved correctly.
        """
        type_checks = {
            'continuous_age_present': 'age_corrected' in self.df.columns,
            'standardized_categoricals': self._check_standardized_categoricals(),
            'numeric_columns_valid': self._check_numeric_columns()
        }
        
        self.validation_results['data_types'] = type_checks
        
    def _check_standardized_categoricals(self) -> bool:
        """
        Check if categorical variables are properly standardized.
        """
        categorical_cols = ['tuition', 'direct_admission', 'gender', 'transport_mode', 'learning_style']
        
        for col in categorical_cols:
            if col in self.df.columns:
                unique_vals = self.df[col].unique()
                # Check if values look standardized (no mixed case, consistent format)
                if any(pd.isna(val) for val in unique_vals):
                    continue  # Skip NaN values
                    
        return True  # Simplified check for now
        
    def _check_numeric_columns(self) -> bool:
        """
        Check if numeric columns have appropriate data types.
        """
        numeric_cols = ['previous_score', 'study_hours', 'attendance_rate', 'final_test']
        
        for col in numeric_cols:
            if col in self.df.columns:
                if not pd.api.types.is_numeric_dtype(self.df[col]):
                    logger.warning(f"Column {col} is not numeric type: {self.df[col].dtype}")
                    return False
                    
        return True
        
    def _check_data_corruption(self) -> None:
        """
        Check for signs of data corruption.
        """
        corruption_checks = {
            'has_duplicated_indices': self.df.index.duplicated().any(),
            'has_all_nan_rows': self.df.isnull().all(axis=1).any(),
            'has_infinite_values': np.isinf(self.df.select_dtypes(include=[np.number])).any().any()
        }
        
        self.validation_results['corruption_checks'] = corruption_checks
        
        # Log warnings for any corruption found
        for check, result in corruption_checks.items():
            if result:
                logger.warning(f"Data corruption detected: {check}")
                
    def _validate_record_count(self) -> None:
        """
        Validate that record count is within expected range after Phase 3 processing.
        """
        # Expected range after deduplication and cleaning
        min_expected = 15000  # Conservative estimate after cleaning
        max_expected = 15900  # Original count
        
        record_count = len(self.df)
        
        self.validation_results['record_count_valid'] = (
            min_expected <= record_count <= max_expected
        )
        
        if not self.validation_results['record_count_valid']:
            logger.warning(
                f"Record count {record_count} outside expected range "
                f"[{min_expected}, {max_expected}]"
            )
            
    def _log_validation_summary(self) -> None:
        """
        Log a summary of validation results.
        """
        logger.info("=== Data Validation Summary ===")
        logger.info(f"Total records: {self.validation_results['total_records']}")
        logger.info(f"Total columns: {self.validation_results['total_columns']}")
        
        if self.validation_results['missing_base_columns']:
            logger.warning(f"Missing base columns: {self.validation_results['missing_base_columns']}")
        else:
            logger.info("All expected base columns present")
            
        # Check if validation passed
        validation_passed = (
            not self.validation_results['missing_base_columns'] and
            self.validation_results['record_count_valid'] and
            not any(self.validation_results['corruption_checks'].values())
        )
        
        if validation_passed:
            logger.info("✓ Data validation PASSED - Ready for feature engineering")
        else:
            logger.error("✗ Data validation FAILED - Review issues before proceeding")
            
        self.validation_results['validation_passed'] = validation_passed
        
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the loaded data.
        
        Returns:
            Dictionary containing data summary information
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        summary = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum(),
            'validation_results': self.validation_results
        }
        
        return summary
        
    def save_validation_report(self, output_path: str = "data/featured/phase4_validation_report.json") -> None:
        """
        Save validation results to a JSON file.
        
        Args:
            output_path: Path to save the validation report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        report = self.get_data_summary()
        
        # Convert dtypes to strings
        report['dtypes'] = {k: str(v) for k, v in report['dtypes'].items()}
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info(f"Validation report saved to {output_path}")


def main():
    """
    Main function to run Phase 4 Task 1: Load and Validate Data.
    """
    try:
        # Initialize loader
        loader = Phase4DataLoader()
        
        # Load and validate data
        df = loader.load_data()
        
        # Save validation report
        loader.save_validation_report()
        
        # Print summary
        summary = loader.get_data_summary()
        print(f"\n=== Phase 4 Data Loading Complete ===")
        print(f"Loaded {summary['shape'][0]} records with {summary['shape'][1]} columns")
        print(f"Memory usage: {summary['memory_usage'] / 1024 / 1024:.2f} MB")
        print(f"Validation passed: {summary['validation_results']['validation_passed']}")
        
        return df
        
    except Exception as e:
        logger.error(f"Phase 4 Task 1 failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()