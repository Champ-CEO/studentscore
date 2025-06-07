#!/usr/bin/env python3
"""
Phase 3.1.2: Categorical Data Standardization

Implements Priority 1 of Phase 3 data preprocessing:
- Standardizes tuition 'Y' to 'Yes' for consistency
- Applies case normalization ('CLUBS' → 'Clubs')
- Ensures consistent formatting across all categorical features
- Creates standardization mapping dictionaries
- Validates no new inconsistencies are introduced

Follows TASKS.md Phase 3.1.2 specifications exactly.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CategoricalDataStandardizer:
    """
    Handles categorical data standardization for Phase 3.1.2.
    
    Implements the requirements for task 3.1.2:
    - Standardize tuition 'Y' to 'Yes' for consistency
    - Apply case normalization across categorical features
    - Create standardization mapping dictionaries
    - Validate consistency and prevent new inconsistencies
    """
    
    def __init__(self, input_path: str = "data/processed/age_corrected.csv"):
        """
        Initialize the CategoricalDataStandardizer.
        
        Args:
            input_path: Path to the age-corrected CSV file from Phase 3.1.1
        """
        self.input_path = input_path
        self.data = None
        self.standardized_data = None
        self.standardization_mappings = {}
        self.audit_trail = []
        
    def load_data(self) -> pd.DataFrame:
        """
        Load age-corrected data from Phase 3.1.1.
        
        Returns:
            DataFrame containing the age-corrected data
        """
        try:
            self.data = pd.read_csv(self.input_path)
            logger.info(f"Loaded {len(self.data)} records from {self.input_path}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def identify_categorical_columns(self) -> List[str]:
        """
        Identify categorical columns in the dataset.
        
        Returns:
            List of categorical column names
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Based on EDA findings, these are the categorical columns
        categorical_cols = ['gender', 'CCA', 'learning_style', 'tuition']
        
        # Verify columns exist in data
        existing_cols = [col for col in categorical_cols if col in self.data.columns]
        missing_cols = [col for col in categorical_cols if col not in self.data.columns]
        
        if missing_cols:
            logger.warning(f"Expected categorical columns not found: {missing_cols}")
        
        logger.info(f"Identified categorical columns: {existing_cols}")
        return existing_cols
    
    def analyze_categorical_values(self) -> Dict[str, Dict[str, int]]:
        """
        Analyze current values in categorical columns.
        
        Returns:
            Dictionary with value counts for each categorical column
        """
        categorical_cols = self.identify_categorical_columns()
        analysis = {}
        
        for col in categorical_cols:
            value_counts = self.data[col].value_counts(dropna=False)
            analysis[col] = value_counts.to_dict()
            
            logger.info(f"Column '{col}' values: {dict(value_counts)}")
            
            # Check for potential standardization issues
            values = [str(v) for v in value_counts.index if pd.notna(v)]
            
            # Check for case inconsistencies
            case_issues = []
            for val in values:
                if val != val.title() and val.upper() != val:
                    case_issues.append(val)
            
            if case_issues:
                logger.info(f"Column '{col}' has potential case issues: {case_issues}")
        
        return analysis
    
    def create_tuition_standardization_mapping(self) -> Dict[str, str]:
        """
        Create standardization mapping for tuition column.
        
        Returns:
            Dictionary mapping original values to standardized values
        """
        if 'tuition' not in self.data.columns:
            logger.warning("Tuition column not found in data")
            return {}
        
        # Get unique tuition values
        unique_values = self.data['tuition'].dropna().unique()
        logger.info(f"Original tuition values: {list(unique_values)}")
        
        # Create mapping: 'Y' -> 'Yes', keep others as-is but ensure proper case
        mapping = {}
        for val in unique_values:
            val_str = str(val)
            if val_str.upper() == 'Y':
                mapping[val_str] = 'Yes'
            elif val_str.upper() == 'YES':
                mapping[val_str] = 'Yes'
            elif val_str.upper() == 'N':
                mapping[val_str] = 'No'
            elif val_str.upper() == 'NO':
                mapping[val_str] = 'No'
            else:
                # Apply title case for consistency
                mapping[val_str] = val_str.title()
        
        logger.info(f"Tuition standardization mapping: {mapping}")
        return mapping
    
    def create_case_normalization_mapping(self, column: str) -> Dict[str, str]:
        """
        Create case normalization mapping for a categorical column.
        
        Args:
            column: Name of the categorical column
            
        Returns:
            Dictionary mapping original values to normalized values
        """
        if column not in self.data.columns:
            logger.warning(f"Column '{column}' not found in data")
            return {}
        
        unique_values = self.data[column].dropna().unique()
        mapping = {}
        
        for val in unique_values:
            val_str = str(val)
            
            # Special handling for specific known cases
            if column == 'CCA':
                if val_str.upper() == 'CLUBS':
                    mapping[val_str] = 'Clubs'
                elif val_str.upper() == 'SPORTS':
                    mapping[val_str] = 'Sports'
                else:
                    mapping[val_str] = val_str.title()
            else:
                # General title case normalization
                mapping[val_str] = val_str.title()
        
        logger.info(f"Case normalization mapping for '{column}': {mapping}")
        return mapping
    
    def apply_standardization(self) -> pd.DataFrame:
        """
        Apply standardization to all categorical columns.
        
        Returns:
            DataFrame with standardized categorical values
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        self.standardized_data = self.data.copy()
        categorical_cols = self.identify_categorical_columns()
        
        # Track changes for audit trail
        total_changes = 0
        
        for col in categorical_cols:
            logger.info(f"Standardizing column: {col}")
            
            # Create appropriate mapping
            if col == 'tuition':
                mapping = self.create_tuition_standardization_mapping()
            else:
                mapping = self.create_case_normalization_mapping(col)
            
            # Store mapping for reproducibility
            self.standardization_mappings[col] = mapping
            
            # Apply mapping
            if mapping:
                original_values = self.standardized_data[col].copy()
                self.standardized_data[col] = self.standardized_data[col].map(mapping).fillna(self.standardized_data[col])
                
                # Count changes
                changes = (original_values != self.standardized_data[col]).sum()
                total_changes += changes
                
                logger.info(f"Applied {len(mapping)} mappings to '{col}', changed {changes} values")
                
                # Log significant changes for audit trail
                if changes > 0:
                    self.audit_trail.append({
                        'timestamp': datetime.now().isoformat(),
                        'action': 'categorical_standardization',
                        'column': col,
                        'mapping_applied': mapping,
                        'values_changed': int(changes),
                        'details': f'Standardized {changes} values in column {col}'
                    })
        
        logger.info(f"Total standardization changes applied: {total_changes}")
        return self.standardized_data
    
    def validate_standardization(self) -> Dict[str, Any]:
        """
        Validate that standardization was applied correctly and no new inconsistencies introduced.
        
        Returns:
            Dictionary with validation results
        """
        if self.standardized_data is None:
            raise ValueError("No standardized data available. Run apply_standardization() first.")
        
        validation_results = {
            'validation_passed': True,
            'issues_found': [],
            'column_analysis': {}
        }
        
        categorical_cols = self.identify_categorical_columns()
        
        for col in categorical_cols:
            col_analysis = {
                'unique_values_before': len(self.data[col].dropna().unique()),
                'unique_values_after': len(self.standardized_data[col].dropna().unique()),
                'values_after_standardization': self.standardized_data[col].value_counts(dropna=False).to_dict()
            }
            
            # Check for specific validation rules
            if col == 'tuition':
                # Validate tuition only has 'Yes', 'No', or NaN
                valid_tuition_values = {'Yes', 'No'}
                actual_values = set(self.standardized_data[col].dropna().unique())
                invalid_values = actual_values - valid_tuition_values
                
                if invalid_values:
                    validation_results['validation_passed'] = False
                    validation_results['issues_found'].append(
                        f"Tuition column has invalid values: {invalid_values}"
                    )
                    col_analysis['validation_issue'] = f"Invalid values: {invalid_values}"
                else:
                    col_analysis['validation_status'] = 'PASSED'
                    logger.info(f"Tuition standardization validation PASSED")
            
            # Check for case consistency
            values = [str(v) for v in self.standardized_data[col].dropna().unique()]
            case_inconsistent = []
            for val in values:
                if val != val.title() and val not in ['Yes', 'No']:  # Special cases
                    case_inconsistent.append(val)
            
            if case_inconsistent:
                validation_results['issues_found'].append(
                    f"Column '{col}' has case inconsistencies: {case_inconsistent}"
                )
                col_analysis['case_issues'] = case_inconsistent
            else:
                col_analysis['case_consistency'] = 'PASSED'
            
            validation_results['column_analysis'][col] = col_analysis
        
        # Overall validation status
        if validation_results['issues_found']:
            validation_results['validation_passed'] = False
            logger.warning(f"Validation issues found: {validation_results['issues_found']}")
        else:
            logger.info("All categorical standardization validations PASSED")
        
        return validation_results
    
    def get_standardization_summary(self) -> Dict[str, Any]:
        """
        Get summary of categorical standardization process.
        
        Returns:
            Dictionary with standardization statistics
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        summary = {
            'record_count': len(self.data),
            'categorical_columns_processed': list(self.standardization_mappings.keys()),
            'mappings_applied': self.standardization_mappings,
            'audit_trail_entries': len(self.audit_trail)
        }
        
        if self.standardized_data is not None:
            # Calculate total changes
            total_changes = 0
            for entry in self.audit_trail:
                if entry['action'] == 'categorical_standardization':
                    total_changes += entry['values_changed']
            
            summary['total_values_changed'] = total_changes
            
            # Before/after analysis
            categorical_cols = self.identify_categorical_columns()
            summary['before_after_analysis'] = {}
            
            for col in categorical_cols:
                if col in self.data.columns:
                    summary['before_after_analysis'][col] = {
                        'before': self.data[col].value_counts(dropna=False).to_dict(),
                        'after': self.standardized_data[col].value_counts(dropna=False).to_dict()
                    }
        
        return summary
    
    def save_standardized_data(self, output_path: str = "data/processed/standardized.csv") -> None:
        """
        Save standardized data to CSV file.
        
        Args:
            output_path: Path to save the standardized data
        """
        if self.standardized_data is None:
            raise ValueError("No standardized data available. Run standardization process first.")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.standardized_data.to_csv(output_file, index=False)
        logger.info(f"Saved standardized data to {output_file}")
    
    def save_standardization_mappings(self, output_path: str = "data/processed/standardization_mappings.json") -> None:
        """
        Save standardization mappings to JSON file for reproducibility.
        
        Args:
            output_path: Path to save the mappings
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.standardization_mappings, f, indent=2)
        
        logger.info(f"Saved standardization mappings to {output_file}")
    
    def save_audit_trail(self, output_path: str = "data/processed/categorical_standardization_audit.json") -> None:
        """
        Save audit trail to JSON file.
        
        Args:
            output_path: Path to save the audit trail
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.audit_trail, f, indent=2)
        
        logger.info(f"Saved audit trail to {output_file}")
    
    def run_complete_standardization(self) -> Dict[str, Any]:
        """
        Run the complete categorical standardization process.
        
        Returns:
            Dictionary with process summary
        """
        logger.info("Starting Phase 3.1.2: Categorical Data Standardization")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Analyze current categorical values
        original_analysis = self.analyze_categorical_values()
        
        # Step 3: Apply standardization
        self.apply_standardization()
        
        # Step 4: Validate standardization
        validation_results = self.validate_standardization()
        
        # Step 5: Save results
        self.save_standardized_data()
        self.save_standardization_mappings()
        self.save_audit_trail()
        
        # Step 6: Generate summary
        summary = self.get_standardization_summary()
        summary.update({
            'original_analysis': original_analysis,
            'validation_results': validation_results
        })
        
        logger.info("Phase 3.1.2: Categorical Data Standardization completed successfully")
        return summary


def main():
    """
    Main execution function for Phase 3.1.2: Categorical Data Standardization.
    """
    # Initialize standardizer
    standardizer = CategoricalDataStandardizer()
    
    # Run complete standardization process
    summary = standardizer.run_complete_standardization()
    
    # Print summary
    print("\n=== Phase 3.1.2: Categorical Data Standardization Summary ===")
    print(f"Records processed: {summary['record_count']}")
    print(f"Categorical columns processed: {summary['categorical_columns_processed']}")
    print(f"Total values changed: {summary.get('total_values_changed', 0)}")
    print(f"Validation passed: {summary['validation_results']['validation_passed']}")
    
    if summary['validation_results']['issues_found']:
        print(f"Issues found: {summary['validation_results']['issues_found']}")
    
    print(f"Audit trail entries: {summary['audit_trail_entries']}")
    
    # Show key mappings applied
    print("\nKey standardization mappings applied:")
    for col, mapping in summary['mappings_applied'].items():
        if mapping:  # Only show non-empty mappings
            print(f"  {col}: {mapping}")
    
    print("\nCategorical standardization completed successfully!")
    print("Output files:")
    print("- data/processed/standardized.csv")
    print("- data/processed/standardization_mappings.json")
    print("- data/processed/categorical_standardization_audit.json")


if __name__ == "__main__":
    main()