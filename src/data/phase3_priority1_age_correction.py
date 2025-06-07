#!/usr/bin/env python3
"""
Phase 3.1.1: Age Data Correction

Implements Priority 1 of Phase 3 data preprocessing:
- Identifies and corrects 5 records with negative age values (-5)
- Implements age validation rules (0 ≤ age ≤ 100)
- Maintains audit trail of corrections
- Preserves data integrity

Follows TASKS.md Phase 3.1.1 specifications exactly.
"""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgeDataCorrector:
    """
    Handles age data correction for Phase 3.1.1.
    
    Implements the requirements for task 3.1.1:
    - Identify all 5 records with negative age values (-5)
    - Implement age correction/removal strategy
    - Apply age validation rules (0 ≤ age ≤ 100)
    - Maintain audit trail of corrections
    """
    
    def __init__(self, db_path: str):
        """
        Initialize the AgeDataCorrector.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.data = None
        self.corrected_data = None
        self.audit_trail = []
        
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
    
    def identify_negative_ages(self) -> pd.DataFrame:
        """
        Identify all records with negative age values.
        
        Returns:
            DataFrame containing records with negative ages
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        negative_age_records = self.data[self.data['age'] < 0]
        
        logger.info(f"Found {len(negative_age_records)} records with negative ages")
        
        if len(negative_age_records) > 0:
            age_values = negative_age_records['age'].value_counts()
            logger.info(f"Negative age distribution: {age_values.to_dict()}")
            
            # Log details for audit trail
            for idx, row in negative_age_records.iterrows():
                self.audit_trail.append({
                    'timestamp': datetime.now().isoformat(),
                    'action': 'identified_negative_age',
                    'record_index': idx,
                    'student_id': row.get('student_id', 'N/A'),
                    'original_age': row['age'],
                    'details': 'Record identified with negative age value'
                })
        
        return negative_age_records
    
    def validate_age_range(self, age_series: pd.Series) -> Dict[str, int]:
        """
        Validate age values against acceptable range (0 ≤ age ≤ 100).
        
        Args:
            age_series: Series containing age values
            
        Returns:
            Dictionary with validation statistics
        """
        validation_stats = {
            'total_records': len(age_series),
            'valid_ages': ((age_series >= 0) & (age_series <= 100)).sum(),
            'negative_ages': (age_series < 0).sum(),
            'excessive_ages': (age_series > 100).sum(),
            'null_ages': age_series.isnull().sum()
        }
        
        validation_stats['invalid_total'] = (
            validation_stats['negative_ages'] + 
            validation_stats['excessive_ages']
        )
        
        logger.info(f"Age validation results: {validation_stats}")
        return validation_stats
    
    def investigate_negative_age_patterns(self) -> Dict[str, any]:
        """
        Investigate patterns in negative age records to determine if systematic issue.
        
        Returns:
            Dictionary with investigation findings
        """
        negative_records = self.identify_negative_ages()
        
        if len(negative_records) == 0:
            return {'pattern_analysis': 'No negative ages found'}
        
        investigation = {
            'total_negative_records': len(negative_records),
            'unique_negative_values': negative_records['age'].unique().tolist(),
            'student_id_patterns': negative_records['student_id'].tolist(),
            'other_features_analysis': {}
        }
        
        # Analyze other features for patterns
        categorical_cols = ['gender', 'CCA', 'learning_style', 'tuition']
        for col in categorical_cols:
            if col in negative_records.columns:
                investigation['other_features_analysis'][col] = (
                    negative_records[col].value_counts().to_dict()
                )
        
        # Check if negative ages cluster in specific ranges of other numeric features
        numeric_cols = ['attendance_rate', 'final_test']
        for col in numeric_cols:
            if col in negative_records.columns:
                investigation['other_features_analysis'][f'{col}_stats'] = {
                    'mean': negative_records[col].mean(),
                    'median': negative_records[col].median(),
                    'std': negative_records[col].std()
                }
        
        logger.info(f"Negative age investigation: {investigation}")
        return investigation
    
    def correct_negative_ages(self, strategy: str = 'remove') -> pd.DataFrame:
        """
        Correct negative age values using specified strategy.
        
        Args:
            strategy: Correction strategy ('remove', 'median_impute', 'manual_correct')
            
        Returns:
            DataFrame with corrected age values
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        self.corrected_data = self.data.copy()
        negative_mask = self.corrected_data['age'] < 0
        negative_count = negative_mask.sum()
        
        if negative_count == 0:
            logger.info("No negative ages found to correct")
            return self.corrected_data
        
        logger.info(f"Correcting {negative_count} negative age records using strategy: {strategy}")
        
        if strategy == 'remove':
            # Remove records with negative ages
            removed_indices = self.corrected_data[negative_mask].index.tolist()
            self.corrected_data = self.corrected_data[~negative_mask].reset_index(drop=True)
            
            # Log removal for audit trail
            for idx in removed_indices:
                self.audit_trail.append({
                    'timestamp': datetime.now().isoformat(),
                    'action': 'removed_negative_age_record',
                    'original_index': idx,
                    'student_id': self.data.loc[idx, 'student_id'] if 'student_id' in self.data.columns else 'N/A',
                    'original_age': self.data.loc[idx, 'age'],
                    'strategy': strategy,
                    'details': 'Record removed due to negative age value'
                })
            
            logger.info(f"Removed {len(removed_indices)} records with negative ages")
            
        elif strategy == 'median_impute':
            # Replace with median age of valid records
            valid_ages = self.corrected_data[self.corrected_data['age'] >= 0]['age']
            median_age = valid_ages.median()
            
            original_ages = self.corrected_data.loc[negative_mask, 'age'].tolist()
            self.corrected_data.loc[negative_mask, 'age'] = median_age
            
            # Log imputation for audit trail
            for idx, orig_age in zip(self.corrected_data[negative_mask].index, original_ages):
                self.audit_trail.append({
                    'timestamp': datetime.now().isoformat(),
                    'action': 'imputed_negative_age',
                    'record_index': idx,
                    'student_id': self.corrected_data.loc[idx, 'student_id'] if 'student_id' in self.corrected_data.columns else 'N/A',
                    'original_age': orig_age,
                    'corrected_age': median_age,
                    'strategy': strategy,
                    'details': f'Negative age replaced with median: {median_age}'
                })
            
            logger.info(f"Imputed {negative_count} negative ages with median: {median_age}")
            
        else:
            raise ValueError(f"Unknown correction strategy: {strategy}")
        
        return self.corrected_data
    
    def apply_age_validation_rules(self) -> pd.DataFrame:
        """
        Apply age validation rules (0 ≤ age ≤ 100) to the corrected data.
        
        Returns:
            DataFrame with validated age values
        """
        if self.corrected_data is None:
            raise ValueError("No corrected data available. Run correct_negative_ages() first.")
        
        # Check for any remaining invalid ages
        invalid_mask = (self.corrected_data['age'] < 0) | (self.corrected_data['age'] > 100)
        invalid_count = invalid_mask.sum()
        
        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} records with ages outside valid range (0-100)")
            
            # Log validation failures
            for idx in self.corrected_data[invalid_mask].index:
                self.audit_trail.append({
                    'timestamp': datetime.now().isoformat(),
                    'action': 'validation_failure',
                    'record_index': idx,
                    'age_value': self.corrected_data.loc[idx, 'age'],
                    'details': 'Age value outside valid range (0-100)'
                })
        else:
            logger.info("All age values pass validation rules (0 ≤ age ≤ 100)")
        
        return self.corrected_data
    
    def get_correction_summary(self) -> Dict[str, any]:
        """
        Get summary of age correction process.
        
        Returns:
            Dictionary with correction statistics and audit information
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        summary = {
            'original_record_count': len(self.data),
            'original_age_stats': {
                'min': self.data['age'].min(),
                'max': self.data['age'].max(),
                'mean': self.data['age'].mean(),
                'median': self.data['age'].median(),
                'negative_count': (self.data['age'] < 0).sum()
            },
            'audit_trail_entries': len(self.audit_trail)
        }
        
        if self.corrected_data is not None:
            summary.update({
                'corrected_record_count': len(self.corrected_data),
                'corrected_age_stats': {
                    'min': self.corrected_data['age'].min(),
                    'max': self.corrected_data['age'].max(),
                    'mean': self.corrected_data['age'].mean(),
                    'median': self.corrected_data['age'].median(),
                    'negative_count': (self.corrected_data['age'] < 0).sum()
                },
                'records_removed': len(self.data) - len(self.corrected_data)
            })
        
        return summary
    
    def save_corrected_data(self, output_path: str = "data/processed/age_corrected.csv") -> None:
        """
        Save corrected data to CSV file.
        
        Args:
            output_path: Path to save the corrected data
        """
        if self.corrected_data is None:
            raise ValueError("No corrected data available. Run correction process first.")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.corrected_data.to_csv(output_file, index=False)
        logger.info(f"Saved corrected data to {output_file}")
    
    def save_audit_trail(self, output_path: str = "data/processed/age_correction_audit.json") -> None:
        """
        Save audit trail to JSON file.
        
        Args:
            output_path: Path to save the audit trail
        """
        import json
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.audit_trail, f, indent=2)
        
        logger.info(f"Saved audit trail to {output_file}")
    
    def run_complete_correction(self, strategy: str = 'remove') -> Dict[str, any]:
        """
        Run the complete age correction process.
        
        Args:
            strategy: Correction strategy for negative ages
            
        Returns:
            Dictionary with process summary
        """
        logger.info("Starting Phase 3.1.1: Age Data Correction")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Identify negative ages
        negative_records = self.identify_negative_ages()
        
        # Step 3: Investigate patterns
        investigation = self.investigate_negative_age_patterns()
        
        # Step 4: Validate original age range
        original_validation = self.validate_age_range(self.data['age'])
        
        # Step 5: Correct negative ages
        self.correct_negative_ages(strategy=strategy)
        
        # Step 6: Apply validation rules
        self.apply_age_validation_rules()
        
        # Step 7: Final validation
        final_validation = self.validate_age_range(self.corrected_data['age'])
        
        # Step 8: Save results
        self.save_corrected_data()
        self.save_audit_trail()
        
        # Step 9: Generate summary
        summary = self.get_correction_summary()
        summary.update({
            'investigation_findings': investigation,
            'original_validation': original_validation,
            'final_validation': final_validation,
            'correction_strategy': strategy
        })
        
        logger.info("Phase 3.1.1: Age Data Correction completed successfully")
        return summary


def main():
    """
    Main execution function for Phase 3.1.1: Age Data Correction.
    """
    # Initialize corrector
    db_path = "data/raw/score.db"
    corrector = AgeDataCorrector(db_path)
    
    # Run complete correction process
    summary = corrector.run_complete_correction(strategy='remove')
    
    # Print summary
    print("\n=== Phase 3.1.1: Age Data Correction Summary ===")
    print(f"Original records: {summary['original_record_count']}")
    print(f"Corrected records: {summary['corrected_record_count']}")
    print(f"Records removed: {summary['records_removed']}")
    print(f"Original negative ages: {summary['original_validation']['negative_ages']}")
    print(f"Final negative ages: {summary['final_validation']['negative_ages']}")
    print(f"Audit trail entries: {summary['audit_trail_entries']}")
    
    print("\nAge correction completed successfully!")
    print("Output files:")
    print("- data/processed/age_corrected.csv")
    print("- data/processed/age_correction_audit.json")


if __name__ == "__main__":
    main()