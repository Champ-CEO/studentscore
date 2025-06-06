import pandas as pd
import numpy as np
import sqlite3
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataDeduplicator:
    """
    Handles duplicate record detection and removal for the student score dataset.
    
    Based on db-structure.md, there are 139 duplicate records that need to be addressed.
    This class provides comprehensive duplicate analysis and removal strategies.
    """
    
    def __init__(self, db_path: Optional[str] = None, data: Optional[pd.DataFrame] = None):
        """
        Initialize the DataDeduplicator.
        
        Args:
            db_path: Path to SQLite database file
            data: Pre-loaded DataFrame (alternative to db_path)
        """
        self.db_path = db_path
        self.data = data
        self.original_data = None
        self.duplicates_info = {}
        self.removal_strategy = 'keep_first'  # Default strategy
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from database or return provided DataFrame.
        
        Returns:
            DataFrame with the loaded data
        """
        if self.data is not None:
            logger.info(f"Using provided DataFrame with {len(self.data)} records")
            return self.data.copy()
        
        if self.db_path is None:
            raise ValueError("Either db_path or data must be provided")
        
        try:
            conn = sqlite3.connect(self.db_path)
            query = "SELECT * FROM score"
            data = pd.read_sql_query(query, conn)
            conn.close()
            
            logger.info(f"Loaded {len(data)} records from database")
            self.data = data
            return data.copy()
            
        except Exception as e:
            logger.error(f"Error loading data from database: {e}")
            raise
    
    def detect_duplicates(self, subset_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect duplicate records in the dataset.
        
        Args:
            subset_columns: Columns to consider for duplicate detection.
                          If None, uses all columns except index.
        
        Returns:
            Dictionary with duplicate analysis results
        """
        if self.data is None:
            self.data = self.load_data()
        
        # Store original data for comparison
        if self.original_data is None:
            self.original_data = self.data.copy()
        
        # Define columns to check for duplicates
        if subset_columns is None:
            # Exclude index column from duplicate detection
            subset_columns = [col for col in self.data.columns if col != 'index']
        
        # Find duplicates
        duplicate_mask = self.data.duplicated(subset=subset_columns, keep=False)
        duplicate_records = self.data[duplicate_mask]
        
        # Analyze duplicate patterns
        duplicate_groups = self.data[duplicate_mask].groupby(
            subset_columns, dropna=False
        ).size().reset_index(name='count')
        
        # Get first occurrence duplicates
        first_duplicates = self.data.duplicated(subset=subset_columns, keep='first')
        
        # Get last occurrence duplicates
        last_duplicates = self.data.duplicated(subset=subset_columns, keep='last')
        
        analysis = {
            'total_records': len(self.data),
            'duplicate_records': len(duplicate_records),
            'unique_records': len(self.data) - duplicate_records.duplicated(subset=subset_columns).sum(),
            'duplicate_groups': len(duplicate_groups),
            'duplicate_mask': duplicate_mask,
            'first_duplicates_mask': first_duplicates,
            'last_duplicates_mask': last_duplicates,
            'duplicate_groups_info': duplicate_groups,
            'subset_columns_used': subset_columns
        }
        
        self.duplicates_info = analysis
        
        logger.info(f"Duplicate Analysis Results:")
        logger.info(f"  Total records: {analysis['total_records']}")
        logger.info(f"  Duplicate records: {analysis['duplicate_records']}")
        logger.info(f"  Unique records after deduplication: {analysis['unique_records']}")
        logger.info(f"  Number of duplicate groups: {analysis['duplicate_groups']}")
        
        return analysis
    
    def analyze_duplicate_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in duplicate records to understand their nature.
        
        Returns:
            Dictionary with pattern analysis results
        """
        if not self.duplicates_info:
            self.detect_duplicates()
        
        duplicate_mask = self.duplicates_info['duplicate_mask']
        duplicate_records = self.data[duplicate_mask]
        
        if len(duplicate_records) == 0:
            return {'message': 'No duplicates found'}
        
        # Analyze duplicate characteristics
        patterns = {
            'student_id_patterns': {},
            'age_patterns': {},
            'gender_patterns': {},
            'final_test_patterns': {},
            'exact_duplicates': 0,
            'partial_duplicates': 0
        }
        
        # Group duplicates by student_id to see if same students have multiple records
        if 'student_id' in duplicate_records.columns:
            student_id_counts = duplicate_records['student_id'].value_counts()
            patterns['student_id_patterns'] = {
                'unique_student_ids_in_duplicates': len(student_id_counts),
                'max_records_per_student': student_id_counts.max(),
                'avg_records_per_student': student_id_counts.mean()
            }
        
        # Analyze age patterns in duplicates
        if 'age' in duplicate_records.columns:
            age_stats = duplicate_records['age'].describe()
            patterns['age_patterns'] = {
                'age_range': (age_stats['min'], age_stats['max']),
                'age_mean': age_stats['mean'],
                'age_std': age_stats['std']
            }
        
        # Analyze gender distribution in duplicates
        if 'gender' in duplicate_records.columns:
            gender_counts = duplicate_records['gender'].value_counts()
            patterns['gender_patterns'] = gender_counts.to_dict()
        
        # Analyze final_test score patterns
        if 'final_test' in duplicate_records.columns:
            final_test_stats = duplicate_records['final_test'].describe()
            patterns['final_test_patterns'] = {
                'score_range': (final_test_stats['min'], final_test_stats['max']),
                'score_mean': final_test_stats['mean'],
                'missing_scores': duplicate_records['final_test'].isna().sum()
            }
        
        # Check for exact vs partial duplicates
        subset_columns = self.duplicates_info['subset_columns_used']
        exact_duplicates = self.data.duplicated(subset=subset_columns, keep=False)
        patterns['exact_duplicates'] = exact_duplicates.sum()
        
        logger.info(f"Duplicate Pattern Analysis:")
        logger.info(f"  Exact duplicates: {patterns['exact_duplicates']}")
        if 'student_id_patterns' in patterns:
            logger.info(f"  Unique student IDs in duplicates: {patterns['student_id_patterns'].get('unique_student_ids_in_duplicates', 'N/A')}")
        
        return patterns
    
    def remove_duplicates(self, strategy: str = 'keep_first', 
                         subset_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Remove duplicate records using specified strategy.
        
        Args:
            strategy: Removal strategy ('keep_first', 'keep_last', 'keep_best')
            subset_columns: Columns to consider for duplicate detection
        
        Returns:
            DataFrame with duplicates removed
        """
        if self.data is None:
            self.data = self.load_data()
        
        # Detect duplicates if not already done
        if not self.duplicates_info:
            self.detect_duplicates(subset_columns)
        
        if subset_columns is None:
            subset_columns = self.duplicates_info['subset_columns_used']
        
        self.removal_strategy = strategy
        
        if strategy == 'keep_first':
            deduplicated_data = self.data.drop_duplicates(subset=subset_columns, keep='first')
            logger.info("Removed duplicates keeping first occurrence")
            
        elif strategy == 'keep_last':
            deduplicated_data = self.data.drop_duplicates(subset=subset_columns, keep='last')
            logger.info("Removed duplicates keeping last occurrence")
            
        elif strategy == 'keep_best':
            deduplicated_data = self._remove_duplicates_keep_best(subset_columns)
            logger.info("Removed duplicates keeping best record based on data quality")
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'keep_first', 'keep_last', or 'keep_best'")
        
        # Reset index
        deduplicated_data = deduplicated_data.reset_index(drop=True)
        
        logger.info(f"Deduplication complete:")
        logger.info(f"  Original records: {len(self.data)}")
        logger.info(f"  After deduplication: {len(deduplicated_data)}")
        logger.info(f"  Records removed: {len(self.data) - len(deduplicated_data)}")
        
        return deduplicated_data
    
    def _remove_duplicates_keep_best(self, subset_columns: List[str]) -> pd.DataFrame:
        """
        Remove duplicates by keeping the "best" record based on data quality criteria.
        
        Args:
            subset_columns: Columns to consider for duplicate detection
        
        Returns:
            DataFrame with duplicates removed, keeping best records
        """
        # Create a copy to work with
        data_copy = self.data.copy()
        
        # Add a quality score for each record
        data_copy['_quality_score'] = self._calculate_quality_score(data_copy)
        
        # For each group of duplicates, keep the one with highest quality score
        deduplicated_data = (
            data_copy
            .sort_values('_quality_score', ascending=False)
            .drop_duplicates(subset=subset_columns, keep='first')
            .drop(columns=['_quality_score'])
        )
        
        return deduplicated_data
    
    def _calculate_quality_score(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate a quality score for each record to determine which duplicate to keep.
        
        Args:
            data: DataFrame to calculate quality scores for
        
        Returns:
            Series with quality scores
        """
        quality_score = pd.Series(0.0, index=data.index)
        
        # Higher score for records with fewer missing values
        missing_count = data.isnull().sum(axis=1)
        quality_score += (data.shape[1] - missing_count) / data.shape[1]
        
        # Higher score for records with final_test scores (if available)
        if 'final_test' in data.columns:
            quality_score += data['final_test'].notna().astype(float) * 0.5
        
        # Higher score for records with attendance_rate (if available)
        if 'attendance_rate' in data.columns:
            quality_score += data['attendance_rate'].notna().astype(float) * 0.3
        
        # Higher score for records with reasonable age values
        if 'age' in data.columns:
            reasonable_age = ((data['age'] >= 10) & (data['age'] <= 25)).astype(float)
            quality_score += reasonable_age * 0.2
        
        return quality_score
    
    def get_deduplication_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report on the deduplication process.
        
        Returns:
            Dictionary with deduplication report
        """
        if not self.duplicates_info:
            return {'error': 'No deduplication analysis performed yet'}
        
        report = {
            'original_record_count': self.duplicates_info['total_records'],
            'duplicate_record_count': self.duplicates_info['duplicate_records'],
            'unique_record_count': self.duplicates_info['unique_records'],
            'duplicate_groups': self.duplicates_info['duplicate_groups'],
            'removal_strategy_used': getattr(self, 'removal_strategy', 'Not applied'),
            'columns_used_for_detection': self.duplicates_info['subset_columns_used']
        }
        
        # Add pattern analysis if available
        try:
            patterns = self.analyze_duplicate_patterns()
            report['duplicate_patterns'] = patterns
        except Exception as e:
            report['pattern_analysis_error'] = str(e)
        
        return report
    
    def save_deduplicated_data(self, output_path: str, deduplicated_data: pd.DataFrame) -> None:
        """
        Save deduplicated data to CSV file.
        
        Args:
            output_path: Path to save the deduplicated data
            deduplicated_data: DataFrame with duplicates removed
        """
        try:
            # Ensure output directory exists
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            deduplicated_data.to_csv(output_path, index=False)
            logger.info(f"Deduplicated data saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving deduplicated data: {e}")
            raise
    
    def run_full_deduplication_pipeline(self, 
                                       output_path: Optional[str] = None,
                                       strategy: str = 'keep_first',
                                       subset_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Run the complete deduplication pipeline.
        
        Args:
            output_path: Path to save deduplicated data (optional)
            strategy: Removal strategy to use
            subset_columns: Columns to consider for duplicate detection
        
        Returns:
            DataFrame with duplicates removed
        """
        logger.info("Starting full deduplication pipeline")
        
        # Load data
        self.load_data()
        
        # Detect duplicates
        self.detect_duplicates(subset_columns)
        
        # Analyze patterns
        self.analyze_duplicate_patterns()
        
        # Remove duplicates
        deduplicated_data = self.remove_duplicates(strategy, subset_columns)
        
        # Save if output path provided
        if output_path:
            self.save_deduplicated_data(output_path, deduplicated_data)
        
        # Generate report
        report = self.get_deduplication_report()
        logger.info(f"Deduplication pipeline complete. Report: {report}")
        
        return deduplicated_data


# Example usage
if __name__ == "__main__":
    # Example with database
    db_path = "data/raw/score.db"
    deduplicator = DataDeduplicator(db_path=db_path)
    
    # Run full pipeline
    cleaned_data = deduplicator.run_full_deduplication_pipeline(
        output_path="data/processed/deduplicated.csv",
        strategy="keep_first"
    )
    
    print(f"Deduplication complete. Final dataset has {len(cleaned_data)} records.")