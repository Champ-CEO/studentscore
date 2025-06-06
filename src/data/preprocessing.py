import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Handles missing data and preprocessing for the student score dataset.
    
    This class implements the requirements for task 2.1.1:
    - Median imputation for attendance_rate by subgroups
    - Missing indicator variables where appropriate
    - Exclude final_test missing values from training set
    """
    
    def __init__(self, db_path: str):
        """
        Initialize the DataPreprocessor.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.data = None
        self.processed_data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from SQLite database.
        
        Returns:
            DataFrame containing the raw data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            self.data = pd.read_sql_query("SELECT * FROM score", conn)
            conn.close()
            logger.info(f"Loaded {len(self.data)} records from database")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def detect_missing_values(self) -> Dict[str, int]:
        """
        Detect missing values in the dataset.
        
        Returns:
            Dictionary with column names and missing value counts
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        missing_counts = self.data.isnull().sum()
        missing_dict = {col: count for col, count in missing_counts.items() if count > 0}
        
        logger.info(f"Missing values detected: {missing_dict}")
        return missing_dict
    
    def create_missing_indicators(self, columns: List[str]) -> pd.DataFrame:
        """
        Create missing indicator variables for specified columns.
        
        Args:
            columns: List of column names to create indicators for
            
        Returns:
            DataFrame with missing indicator columns
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        indicators = pd.DataFrame()
        for col in columns:
            if col in self.data.columns:
                indicators[f'{col}_missing'] = self.data[col].isnull().astype(int)
                logger.info(f"Created missing indicator for {col}: {indicators[f'{col}_missing'].sum()} missing values")
        
        return indicators
    
    def impute_attendance_rate(self, groupby_columns: List[str] = None) -> pd.Series:
        """
        Impute missing attendance_rate values using median by subgroups.
        
        Args:
            groupby_columns: Columns to group by for imputation. 
                           Defaults to ['gender', 'CCA', 'learning_style']
                           
        Returns:
            Series with imputed attendance_rate values
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if groupby_columns is None:
            groupby_columns = ['gender', 'CCA', 'learning_style']
        
        # Create a copy of attendance_rate for imputation
        attendance_imputed = self.data['attendance_rate'].copy()
        
        # Get rows with missing attendance_rate
        missing_mask = attendance_imputed.isnull()
        
        if missing_mask.sum() == 0:
            logger.info("No missing values in attendance_rate")
            return attendance_imputed
        
        # Calculate median by subgroups
        group_medians = self.data.groupby(groupby_columns)['attendance_rate'].median()
        
        # Impute missing values
        for idx in self.data[missing_mask].index:
            row = self.data.loc[idx]
            group_key = tuple(row[col] for col in groupby_columns)
            
            if group_key in group_medians and not pd.isna(group_medians[group_key]):
                attendance_imputed.loc[idx] = group_medians[group_key]
            else:
                # Fallback to overall median if group median is not available
                overall_median = self.data['attendance_rate'].median()
                attendance_imputed.loc[idx] = overall_median
                logger.warning(f"Used overall median for group {group_key}")
        
        imputed_count = missing_mask.sum()
        logger.info(f"Imputed {imputed_count} missing attendance_rate values using group medians")
        
        return attendance_imputed
    
    def exclude_missing_final_test(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Separate data into training set (excluding missing final_test) and prediction set.
        
        Args:
            data: DataFrame to split
            
        Returns:
            Tuple of (training_data, prediction_data)
        """
        training_mask = data['final_test'].notna()
        training_data = data[training_mask].copy()
        prediction_data = data[~training_mask].copy()
        
        logger.info(f"Training set: {len(training_data)} records")
        logger.info(f"Prediction set: {len(prediction_data)} records (missing final_test)")
        
        return training_data, prediction_data
    
    def process_missing_data(self, 
                           create_indicators: bool = True,
                           impute_attendance: bool = True,
                           exclude_missing_target: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Complete missing data processing pipeline.
        
        Args:
            create_indicators: Whether to create missing indicator variables
            impute_attendance: Whether to impute missing attendance_rate values
            exclude_missing_target: Whether to exclude missing final_test from training
            
        Returns:
            Dictionary containing processed datasets
        """
        if self.data is None:
            self.load_data()
        
        # Start with original data
        processed = self.data.copy()
        
        # 1. Create missing indicators
        missing_indicators = pd.DataFrame()
        if create_indicators:
            columns_with_missing = ['final_test', 'attendance_rate']
            missing_indicators = self.create_missing_indicators(columns_with_missing)
        
        # 2. Impute attendance_rate
        if impute_attendance:
            processed['attendance_rate'] = self.impute_attendance_rate()
        
        # 3. Add missing indicators to processed data
        if not missing_indicators.empty:
            processed = pd.concat([processed, missing_indicators], axis=1)
        
        # 4. Split into training and prediction sets
        result = {'full_processed': processed}
        
        if exclude_missing_target:
            training_data, prediction_data = self.exclude_missing_final_test(processed)
            result['training'] = training_data
            result['prediction'] = prediction_data
        
        self.processed_data = result
        logger.info("Missing data processing completed")
        
        return result
    
    def save_processed_data(self, output_dir: str = "data/processed") -> None:
        """
        Save processed data to CSV files.
        
        Args:
            output_dir: Directory to save processed files
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Run process_missing_data() first.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for dataset_name, dataset in self.processed_data.items():
            file_path = output_path / f"{dataset_name}.csv"
            dataset.to_csv(file_path, index=False)
            logger.info(f"Saved {dataset_name} dataset to {file_path}")
    
    def get_missing_summary(self) -> Dict[str, any]:
        """
        Get summary of missing data handling.
        
        Returns:
            Dictionary with missing data statistics
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        original_missing = self.detect_missing_values()
        
        summary = {
            'original_record_count': len(self.data),
            'original_missing_values': original_missing,
            'missing_percentages': {
                col: (count / len(self.data)) * 100 
                for col, count in original_missing.items()
            }
        }
        
        if self.processed_data:
            if 'training' in self.processed_data:
                summary['training_record_count'] = len(self.processed_data['training'])
            if 'prediction' in self.processed_data:
                summary['prediction_record_count'] = len(self.processed_data['prediction'])
        
        return summary


def main():
    """
    Example usage of the DataPreprocessor.
    """
    # Initialize preprocessor
    db_path = "data/raw/score.db"
    preprocessor = DataPreprocessor(db_path)
    
    # Load and process data
    preprocessor.load_data()
    
    # Get missing data summary
    summary = preprocessor.get_missing_summary()
    print("Missing Data Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Process missing data
    processed_datasets = preprocessor.process_missing_data()
    
    # Save processed data
    preprocessor.save_processed_data()
    
    # Final summary
    final_summary = preprocessor.get_missing_summary()
    print("\nFinal Summary:")
    for key, value in final_summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()