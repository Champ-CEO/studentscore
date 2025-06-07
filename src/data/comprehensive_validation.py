#!/usr/bin/env python3
"""
Comprehensive Data Validation and Quality Checks Module

Implements Phase 3.1.4: Data validation and quality checks

This module provides comprehensive validation of data quality,
including missing values, inconsistencies, out-of-range values,
and logical inconsistencies across features.
"""

import pandas as pd
import numpy as np
import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime, time
import re
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveDataValidator:
    """
    Comprehensive data validation and quality assessment.
    
    Performs extensive validation checks including:
    - Missing value analysis
    - Range validation
    - Logical consistency checks
    - Data type validation
    - Cross-feature validation
    - Outlier detection
    """
    
    def __init__(self, db_path: Optional[str] = None, data: Optional[pd.DataFrame] = None):
        """
        Initialize the Comprehensive Data Validator.
        
        Args:
            db_path: Path to SQLite database file
            data: Pre-loaded DataFrame (alternative to db_path)
        """
        self.db_path = db_path
        self.data = data
        self.validation_results = {}
        self.validation_rules = self._define_validation_rules()
        
    def _define_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """
        Define validation rules for each feature.
        
        Returns:
            Dictionary with validation rules for each column
        """
        rules = {
            'student_id': {
                'required': True,
                'unique': True,
                'data_type': 'string',
                'pattern': r'^[A-Za-z0-9]+$',  # Alphanumeric only
                'min_length': 1,
                'max_length': 50
            },
            'gender': {
                'required': True,
                'data_type': 'categorical',
                'allowed_values': ['Male', 'Female', 'M', 'F'],
                'case_sensitive': False
            },
            'age': {
                'required': True,
                'data_type': 'numeric',
                'min_value': 10,
                'max_value': 25,
                'integer_only': False
            },
            'hours_per_week': {
                'required': True,
                'data_type': 'numeric',
                'min_value': 0,
                'max_value': 168,  # Maximum hours in a week
                'integer_only': False
            },
            'attendance_rate': {
                'required': False,  # Known to have missing values
                'data_type': 'numeric',
                'min_value': 0,
                'max_value': 100,
                'integer_only': False
            },
            'sleep_time': {
                'required': True,
                'data_type': 'time',
                'format': 'HH:MM',
                'valid_range': ('00:00', '23:59')
            },
            'wake_time': {
                'required': True,
                'data_type': 'time',
                'format': 'HH:MM',
                'valid_range': ('00:00', '23:59')
            },
            'family_income': {
                'required': True,
                'data_type': 'categorical',
                'allowed_values': ['Low', 'Medium', 'High'],
                'case_sensitive': False
            },
            'internet_access': {
                'required': True,
                'data_type': 'categorical',
                'allowed_values': ['Yes', 'No', 'Y', 'N', 'True', 'False', '1', '0'],
                'case_sensitive': False
            },
            'transport_mode': {
                'required': True,
                'data_type': 'categorical',
                'allowed_values': ['Bus', 'Car', 'Walk', 'Bike', 'Train', 'Other'],
                'case_sensitive': False
            },
            'study_location': {
                'required': True,
                'data_type': 'categorical',
                'allowed_values': ['Home', 'Library', 'School', 'Other'],
                'case_sensitive': False
            },
            'learning_style': {
                'required': True,
                'data_type': 'categorical',
                'allowed_values': ['Visual', 'Auditory', 'Kinesthetic', 'Reading'],
                'case_sensitive': False
            },
            'CCA': {
                'required': True,
                'data_type': 'categorical',
                'allowed_values': ['Yes', 'No', 'Y', 'N', 'True', 'False', '1', '0'],
                'case_sensitive': False
            },
            'tuition': {
                'required': True,
                'data_type': 'categorical',
                'allowed_values': ['Yes', 'No', 'Y', 'N', 'True', 'False', '1', '0'],
                'case_sensitive': False
            },
            'direct_admission': {
                'required': True,
                'data_type': 'categorical',
                'allowed_values': ['Yes', 'No', 'Y', 'N', 'True', 'False', '1', '0'],
                'case_sensitive': False
            },
            'final_test': {
                'required': False,  # Target variable, may have missing values
                'data_type': 'numeric',
                'min_value': 0,
                'max_value': 100,
                'integer_only': False
            }
        }
        
        return rules
    
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
    
    def validate_all(self) -> Dict[str, Any]:
        """
        Run all validation checks.
        
        Returns:
            Dictionary with comprehensive validation results
        """
        if self.data is None:
            self.data = self.load_data()
        
        logger.info("Starting comprehensive data validation")
        
        validation_results = {
            'basic_validation': self.validate_basic_requirements(),
            'missing_values': self.validate_missing_values(),
            'data_types': self.validate_data_types(),
            'range_validation': self.validate_ranges(),
            'categorical_validation': self.validate_categorical_values(),
            'logical_consistency': self.validate_logical_consistency(),
            'cross_feature_validation': self.validate_cross_features(),
            'outlier_detection': self.detect_outliers(),
            'duplicate_analysis': self.analyze_duplicates(),
            'data_quality_score': None  # Will be calculated
        }
        
        # Calculate overall data quality score
        validation_results['data_quality_score'] = self._calculate_quality_score(validation_results)
        
        self.validation_results = validation_results
        logger.info(f"Validation completed. Quality score: {validation_results['data_quality_score']:.2f}")
        
        return validation_results
    
    def validate_basic_requirements(self) -> Dict[str, Any]:
        """
        Validate basic data requirements.
        
        Returns:
            Dictionary with basic validation results
        """
        results = {
            'total_records': len(self.data),
            'total_columns': len(self.data.columns),
            'expected_columns': list(self.validation_rules.keys()),
            'missing_columns': [],
            'extra_columns': [],
            'column_match': True
        }
        
        # Check for missing columns
        expected_cols = set(self.validation_rules.keys())
        actual_cols = set(self.data.columns)
        
        results['missing_columns'] = list(expected_cols - actual_cols)
        results['extra_columns'] = list(actual_cols - expected_cols)
        results['column_match'] = len(results['missing_columns']) == 0
        
        return results
    
    def validate_missing_values(self) -> Dict[str, Any]:
        """
        Validate missing values against requirements.
        
        Returns:
            Dictionary with missing value validation results
        """
        results = {
            'missing_counts': {},
            'missing_percentages': {},
            'required_field_violations': [],
            'missing_patterns': {}
        }
        
        for column in self.data.columns:
            if column in self.validation_rules:
                missing_count = self.data[column].isnull().sum()
                missing_pct = (missing_count / len(self.data)) * 100
                
                results['missing_counts'][column] = missing_count
                results['missing_percentages'][column] = missing_pct
                
                # Check if required field has missing values
                if self.validation_rules[column].get('required', False) and missing_count > 0:
                    results['required_field_violations'].append({
                        'column': column,
                        'missing_count': missing_count,
                        'missing_percentage': missing_pct
                    })
        
        # Analyze missing patterns
        results['missing_patterns'] = self._analyze_missing_patterns()
        
        return results
    
    def validate_data_types(self) -> Dict[str, Any]:
        """
        Validate data types for each column.
        
        Returns:
            Dictionary with data type validation results
        """
        results = {
            'type_validation': {},
            'type_conversion_issues': [],
            'invalid_values': {}
        }
        
        for column, rules in self.validation_rules.items():
            if column not in self.data.columns:
                continue
            
            expected_type = rules.get('data_type')
            column_data = self.data[column].dropna()
            
            if expected_type == 'numeric':
                validation_result = self._validate_numeric_type(column, column_data, rules)
            elif expected_type == 'categorical':
                validation_result = self._validate_categorical_type(column, column_data, rules)
            elif expected_type == 'time':
                validation_result = self._validate_time_type(column, column_data, rules)
            elif expected_type == 'string':
                validation_result = self._validate_string_type(column, column_data, rules)
            else:
                validation_result = {'valid': True, 'issues': []}
            
            results['type_validation'][column] = validation_result
            
            if not validation_result['valid']:
                results['type_conversion_issues'].append({
                    'column': column,
                    'expected_type': expected_type,
                    'issues': validation_result['issues']
                })
        
        return results
    
    def validate_ranges(self) -> Dict[str, Any]:
        """
        Validate numeric ranges for applicable columns.
        
        Returns:
            Dictionary with range validation results
        """
        results = {
            'range_violations': {},
            'out_of_range_counts': {},
            'extreme_values': {}
        }
        
        for column, rules in self.validation_rules.items():
            if column not in self.data.columns or rules.get('data_type') != 'numeric':
                continue
            
            column_data = pd.to_numeric(self.data[column], errors='coerce')
            min_val = rules.get('min_value')
            max_val = rules.get('max_value')
            
            violations = []
            
            if min_val is not None:
                below_min = column_data < min_val
                if below_min.any():
                    violations.append({
                        'type': 'below_minimum',
                        'count': below_min.sum(),
                        'min_allowed': min_val,
                        'actual_min': column_data.min()
                    })
            
            if max_val is not None:
                above_max = column_data > max_val
                if above_max.any():
                    violations.append({
                        'type': 'above_maximum',
                        'count': above_max.sum(),
                        'max_allowed': max_val,
                        'actual_max': column_data.max()
                    })
            
            if violations:
                results['range_violations'][column] = violations
                results['out_of_range_counts'][column] = sum(v['count'] for v in violations)
            
            # Identify extreme values (beyond 3 standard deviations)
            if len(column_data.dropna()) > 0:
                mean_val = column_data.mean()
                std_val = column_data.std()
                if std_val > 0:
                    extreme_mask = np.abs(column_data - mean_val) > 3 * std_val
                    if extreme_mask.any():
                        results['extreme_values'][column] = {
                            'count': extreme_mask.sum(),
                            'values': column_data[extreme_mask].tolist()
                        }
        
        return results
    
    def validate_categorical_values(self) -> Dict[str, Any]:
        """
        Validate categorical values against allowed values.
        
        Returns:
            Dictionary with categorical validation results
        """
        results = {
            'invalid_categories': {},
            'case_issues': {},
            'unexpected_values': {}
        }
        
        for column, rules in self.validation_rules.items():
            if column not in self.data.columns or rules.get('data_type') != 'categorical':
                continue
            
            allowed_values = rules.get('allowed_values', [])
            case_sensitive = rules.get('case_sensitive', True)
            
            column_data = self.data[column].dropna().astype(str)
            
            if not case_sensitive:
                allowed_values_lower = [str(v).lower() for v in allowed_values]
                invalid_mask = ~column_data.str.lower().isin(allowed_values_lower)
            else:
                invalid_mask = ~column_data.isin([str(v) for v in allowed_values])
            
            if invalid_mask.any():
                invalid_values = column_data[invalid_mask].value_counts().to_dict()
                results['invalid_categories'][column] = {
                    'count': invalid_mask.sum(),
                    'values': invalid_values,
                    'allowed_values': allowed_values
                }
            
            # Check for case issues if case insensitive
            if not case_sensitive:
                case_issues = []
                for value in column_data.unique():
                    if value.lower() in allowed_values_lower:
                        expected_case = next((v for v in allowed_values if str(v).lower() == value.lower()), value)
                        if str(expected_case) != value:
                            case_issues.append({
                                'found': value,
                                'expected': str(expected_case)
                            })
                
                if case_issues:
                    results['case_issues'][column] = case_issues
        
        return results
    
    def validate_logical_consistency(self) -> Dict[str, Any]:
        """
        Validate logical consistency across features.
        
        Returns:
            Dictionary with logical consistency results
        """
        results = {
            'sleep_wake_consistency': self._validate_sleep_wake_times(),
            'age_consistency': self._validate_age_consistency(),
            'study_time_consistency': self._validate_study_time_consistency(),
            'income_access_consistency': self._validate_income_access_consistency()
        }
        
        return results
    
    def validate_cross_features(self) -> Dict[str, Any]:
        """
        Validate relationships between features.
        
        Returns:
            Dictionary with cross-feature validation results
        """
        results = {
            'correlation_anomalies': self._detect_correlation_anomalies(),
            'impossible_combinations': self._detect_impossible_combinations(),
            'suspicious_patterns': self._detect_suspicious_patterns()
        }
        
        return results
    
    def detect_outliers(self) -> Dict[str, Any]:
        """
        Detect outliers in numeric columns.
        
        Returns:
            Dictionary with outlier detection results
        """
        results = {
            'statistical_outliers': {},
            'iqr_outliers': {},
            'isolation_forest_outliers': {}
        }
        
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column in self.validation_rules:
                column_data = self.data[column].dropna()
                
                if len(column_data) > 0:
                    # Statistical outliers (Z-score > 3)
                    z_scores = np.abs((column_data - column_data.mean()) / column_data.std())
                    stat_outliers = z_scores > 3
                    
                    if stat_outliers.any():
                        results['statistical_outliers'][column] = {
                            'count': stat_outliers.sum(),
                            'indices': column_data[stat_outliers].index.tolist(),
                            'values': column_data[stat_outliers].tolist()
                        }
                    
                    # IQR outliers
                    Q1 = column_data.quantile(0.25)
                    Q3 = column_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    iqr_outliers = (column_data < lower_bound) | (column_data > upper_bound)
                    
                    if iqr_outliers.any():
                        results['iqr_outliers'][column] = {
                            'count': iqr_outliers.sum(),
                            'lower_bound': lower_bound,
                            'upper_bound': upper_bound,
                            'indices': column_data[iqr_outliers].index.tolist(),
                            'values': column_data[iqr_outliers].tolist()
                        }
        
        return results
    
    def analyze_duplicates(self) -> Dict[str, Any]:
        """
        Analyze duplicate records.
        
        Returns:
            Dictionary with duplicate analysis results
        """
        results = {
            'exact_duplicates': 0,
            'duplicate_student_ids': 0,
            'near_duplicates': 0,
            'duplicate_patterns': {}
        }
        
        # Exact duplicates
        exact_dups = self.data.duplicated().sum()
        results['exact_duplicates'] = exact_dups
        
        # Duplicate student IDs
        if 'student_id' in self.data.columns:
            dup_ids = self.data['student_id'].duplicated().sum()
            results['duplicate_student_ids'] = dup_ids
        
        # Near duplicates (same values except for one or two columns)
        results['near_duplicates'] = self._detect_near_duplicates()
        
        return results
    
    def _validate_numeric_type(self, column: str, data: pd.Series, rules: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate numeric data type.
        
        Args:
            column: Column name
            data: Column data
            rules: Validation rules
            
        Returns:
            Validation result dictionary
        """
        issues = []
        
        # Try to convert to numeric
        numeric_data = pd.to_numeric(data, errors='coerce')
        conversion_failures = numeric_data.isnull() & data.notnull()
        
        if conversion_failures.any():
            issues.append(f"Cannot convert {conversion_failures.sum()} values to numeric")
        
        # Check integer requirement
        if rules.get('integer_only', False):
            if not numeric_data.dropna().apply(lambda x: x == int(x)).all():
                issues.append("Non-integer values found where integers required")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'conversion_failures': conversion_failures.sum()
        }
    
    def _validate_categorical_type(self, column: str, data: pd.Series, rules: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate categorical data type.
        
        Args:
            column: Column name
            data: Column data
            rules: Validation rules
            
        Returns:
            Validation result dictionary
        """
        issues = []
        
        # Check if all values are strings
        non_string_count = sum(not isinstance(val, str) for val in data)
        if non_string_count > 0:
            issues.append(f"{non_string_count} non-string values found")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }
    
    def _validate_time_type(self, column: str, data: pd.Series, rules: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate time data type.
        
        Args:
            column: Column name
            data: Column data
            rules: Validation rules
            
        Returns:
            Validation result dictionary
        """
        issues = []
        time_format = rules.get('format', 'HH:MM')
        
        invalid_times = []
        for value in data:
            try:
                if time_format == 'HH:MM':
                    datetime.strptime(str(value), '%H:%M')
            except ValueError:
                invalid_times.append(value)
        
        if invalid_times:
            issues.append(f"{len(invalid_times)} invalid time formats found")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'invalid_times': invalid_times[:10]  # Sample of invalid times
        }
    
    def _validate_string_type(self, column: str, data: pd.Series, rules: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate string data type.
        
        Args:
            column: Column name
            data: Column data
            rules: Validation rules
            
        Returns:
            Validation result dictionary
        """
        issues = []
        
        # Check length constraints
        min_length = rules.get('min_length')
        max_length = rules.get('max_length')
        
        if min_length is not None:
            too_short = data.str.len() < min_length
            if too_short.any():
                issues.append(f"{too_short.sum()} values shorter than minimum length {min_length}")
        
        if max_length is not None:
            too_long = data.str.len() > max_length
            if too_long.any():
                issues.append(f"{too_long.sum()} values longer than maximum length {max_length}")
        
        # Check pattern
        pattern = rules.get('pattern')
        if pattern:
            pattern_match = data.str.match(pattern)
            if not pattern_match.all():
                issues.append(f"{(~pattern_match).sum()} values don't match required pattern")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }
    
    def _analyze_missing_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in missing data.
        
        Returns:
            Dictionary with missing pattern analysis
        """
        missing_matrix = self.data.isnull()
        
        # Find columns with missing values
        missing_cols = missing_matrix.columns[missing_matrix.any()].tolist()
        
        patterns = {}
        
        if len(missing_cols) > 1:
            # Analyze co-occurrence of missing values
            for i, col1 in enumerate(missing_cols):
                for col2 in missing_cols[i+1:]:
                    both_missing = (missing_matrix[col1] & missing_matrix[col2]).sum()
                    if both_missing > 0:
                        patterns[f"{col1}_and_{col2}"] = both_missing
        
        return {
            'columns_with_missing': missing_cols,
            'co_missing_patterns': patterns,
            'total_complete_cases': (~missing_matrix.any(axis=1)).sum()
        }
    
    def _validate_sleep_wake_times(self) -> Dict[str, Any]:
        """
        Validate logical consistency between sleep and wake times.
        
        Returns:
            Dictionary with sleep/wake time validation results
        """
        if 'sleep_time' not in self.data.columns or 'wake_time' not in self.data.columns:
            return {'error': 'Sleep or wake time columns not found'}
        
        issues = []
        
        # Convert times to datetime for comparison
        try:
            sleep_times = pd.to_datetime(self.data['sleep_time'], format='%H:%M', errors='coerce')
            wake_times = pd.to_datetime(self.data['wake_time'], format='%H:%M', errors='coerce')
            
            # Check for impossible sleep durations (e.g., wake time before sleep time on same day)
            # This is complex due to overnight sleep, so we'll check for obvious issues
            
            # Sleep time after 6 AM is unusual
            late_sleep = sleep_times.dt.hour >= 6
            if late_sleep.any():
                issues.append(f"{late_sleep.sum()} records with sleep time after 6 AM")
            
            # Wake time before 4 AM is unusual
            early_wake = wake_times.dt.hour < 4
            if early_wake.any():
                issues.append(f"{early_wake.sum()} records with wake time before 4 AM")
            
        except Exception as e:
            issues.append(f"Error processing sleep/wake times: {str(e)}")
        
        return {
            'issues': issues,
            'valid': len(issues) == 0
        }
    
    def _validate_age_consistency(self) -> Dict[str, Any]:
        """
        Validate age consistency.
        
        Returns:
            Dictionary with age validation results
        """
        if 'age' not in self.data.columns:
            return {'error': 'Age column not found'}
        
        issues = []
        age_data = pd.to_numeric(self.data['age'], errors='coerce')
        
        # Check for negative ages
        negative_ages = age_data < 0
        if negative_ages.any():
            issues.append(f"{negative_ages.sum()} records with negative age")
        
        # Check for unrealistic ages for students
        too_young = age_data < 10
        too_old = age_data > 25
        
        if too_young.any():
            issues.append(f"{too_young.sum()} records with age below 10")
        
        if too_old.any():
            issues.append(f"{too_old.sum()} records with age above 25")
        
        return {
            'issues': issues,
            'valid': len(issues) == 0,
            'age_range': (age_data.min(), age_data.max()) if not age_data.empty else None
        }
    
    def _validate_study_time_consistency(self) -> Dict[str, Any]:
        """
        Validate study time consistency.
        
        Returns:
            Dictionary with study time validation results
        """
        if 'hours_per_week' not in self.data.columns:
            return {'error': 'Hours per week column not found'}
        
        issues = []
        hours_data = pd.to_numeric(self.data['hours_per_week'], errors='coerce')
        
        # Check for impossible study hours
        too_many_hours = hours_data > 168  # More than hours in a week
        if too_many_hours.any():
            issues.append(f"{too_many_hours.sum()} records with more than 168 hours per week")
        
        # Check for negative hours
        negative_hours = hours_data < 0
        if negative_hours.any():
            issues.append(f"{negative_hours.sum()} records with negative study hours")
        
        return {
            'issues': issues,
            'valid': len(issues) == 0
        }
    
    def _validate_income_access_consistency(self) -> Dict[str, Any]:
        """
        Validate consistency between family income and internet access.
        
        Returns:
            Dictionary with income/access validation results
        """
        if 'family_income' not in self.data.columns or 'internet_access' not in self.data.columns:
            return {'error': 'Required columns not found'}
        
        issues = []
        
        # Check for potential inconsistencies
        # High income families without internet access might be unusual
        high_income_no_internet = (
            (self.data['family_income'].str.lower() == 'high') & 
            (self.data['internet_access'].str.lower().isin(['no', 'n', 'false', '0']))
        )
        
        if high_income_no_internet.any():
            issues.append(f"{high_income_no_internet.sum()} high-income families without internet access")
        
        return {
            'issues': issues,
            'valid': len(issues) == 0,
            'potential_inconsistencies': high_income_no_internet.sum()
        }
    
    def _detect_correlation_anomalies(self) -> Dict[str, Any]:
        """
        Detect anomalies in feature correlations.
        
        Returns:
            Dictionary with correlation anomaly results
        """
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            return {'error': 'Insufficient numeric columns for correlation analysis'}
        
        correlation_matrix = numeric_data.corr()
        
        # Find unusually high correlations (potential data leakage)
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.95:  # Very high correlation threshold
                    high_correlations.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        return {
            'high_correlations': high_correlations,
            'correlation_matrix_shape': correlation_matrix.shape
        }
    
    def _detect_impossible_combinations(self) -> Dict[str, Any]:
        """
        Detect impossible feature combinations.
        
        Returns:
            Dictionary with impossible combination results
        """
        impossible_combinations = []
        
        # Example: Students with 0 study hours but high attendance
        if 'hours_per_week' in self.data.columns and 'attendance_rate' in self.data.columns:
            zero_hours_high_attendance = (
                (pd.to_numeric(self.data['hours_per_week'], errors='coerce') == 0) &
                (pd.to_numeric(self.data['attendance_rate'], errors='coerce') > 80)
            )
            
            if zero_hours_high_attendance.any():
                impossible_combinations.append({
                    'type': 'zero_study_hours_high_attendance',
                    'count': zero_hours_high_attendance.sum(),
                    'description': 'Students with 0 study hours but high attendance rate'
                })
        
        return {
            'impossible_combinations': impossible_combinations
        }
    
    def _detect_suspicious_patterns(self) -> Dict[str, Any]:
        """
        Detect suspicious data patterns.
        
        Returns:
            Dictionary with suspicious pattern results
        """
        patterns = []
        
        # Check for repeated values across multiple columns (potential copy-paste errors)
        for column in self.data.columns:
            if self.data[column].dtype == 'object':
                value_counts = self.data[column].value_counts()
                if len(value_counts) > 0:
                    most_common_pct = value_counts.iloc[0] / len(self.data) * 100
                    if most_common_pct > 50:  # More than 50% same value
                        patterns.append({
                            'type': 'excessive_repetition',
                            'column': column,
                            'value': value_counts.index[0],
                            'percentage': most_common_pct
                        })
        
        return {
            'suspicious_patterns': patterns
        }
    
    def _detect_near_duplicates(self) -> int:
        """
        Detect near-duplicate records.
        
        Returns:
            Count of near-duplicate records
        """
        # Simple near-duplicate detection: records that differ in only 1-2 columns
        near_duplicates = 0
        
        # This is computationally expensive, so we'll sample
        sample_size = min(1000, len(self.data))
        sample_data = self.data.sample(n=sample_size, random_state=42)
        
        for i in range(len(sample_data)):
            for j in range(i+1, len(sample_data)):
                row1 = sample_data.iloc[i]
                row2 = sample_data.iloc[j]
                
                differences = (row1 != row2).sum()
                if 1 <= differences <= 2:  # Differ in 1-2 columns
                    near_duplicates += 1
        
        # Extrapolate to full dataset
        if sample_size < len(self.data):
            scaling_factor = (len(self.data) / sample_size) ** 2
            near_duplicates = int(near_duplicates * scaling_factor)
        
        return near_duplicates
    
    def _calculate_quality_score(self, validation_results: Dict[str, Any]) -> float:
        """
        Calculate overall data quality score.
        
        Args:
            validation_results: Dictionary with all validation results
            
        Returns:
            Quality score between 0 and 100
        """
        score_components = []
        
        # Basic requirements (20% weight)
        basic_score = 100 if validation_results['basic_validation']['column_match'] else 50
        score_components.append(('basic', basic_score, 0.2))
        
        # Missing values (15% weight)
        required_violations = len(validation_results['missing_values']['required_field_violations'])
        missing_score = max(0, 100 - required_violations * 20)
        score_components.append(('missing', missing_score, 0.15))
        
        # Data types (15% weight)
        type_issues = len(validation_results['data_types']['type_conversion_issues'])
        type_score = max(0, 100 - type_issues * 15)
        score_components.append(('types', type_score, 0.15))
        
        # Range validation (15% weight)
        range_violations = len(validation_results['range_validation']['range_violations'])
        range_score = max(0, 100 - range_violations * 20)
        score_components.append(('ranges', range_score, 0.15))
        
        # Categorical validation (10% weight)
        cat_issues = len(validation_results['categorical_validation']['invalid_categories'])
        cat_score = max(0, 100 - cat_issues * 25)
        score_components.append(('categorical', cat_score, 0.1))
        
        # Logical consistency (15% weight)
        logic_issues = sum(1 for check in validation_results['logical_consistency'].values() 
                          if isinstance(check, dict) and not check.get('valid', True))
        logic_score = max(0, 100 - logic_issues * 25)
        score_components.append(('logic', logic_score, 0.15))
        
        # Duplicates (10% weight)
        dup_score = 100 if validation_results['duplicate_analysis']['exact_duplicates'] == 0 else 80
        score_components.append(('duplicates', dup_score, 0.1))
        
        # Calculate weighted average
        total_score = sum(score * weight for _, score, weight in score_components)
        
        return round(total_score, 2)
    
    def save_validation_results(self, output_dir: str) -> None:
        """
        Save validation results to files.
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save validation results
        results_file = output_path / 'comprehensive_validation_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        logger.info(f"Validation results saved to {results_file}")
    
    def generate_validation_report(self) -> str:
        """
        Generate a comprehensive validation report.
        
        Returns:
            Validation report string
        """
        if not self.validation_results:
            return "No validation results available. Run validate_all() first."
        
        report = []
        report.append("=== Comprehensive Data Validation Report ===")
        report.append("")
        
        # Overall quality score
        quality_score = self.validation_results.get('data_quality_score', 0)
        report.append(f"Overall Data Quality Score: {quality_score}/100")
        report.append("")
        
        # Basic validation
        basic = self.validation_results['basic_validation']
        report.append("## Basic Validation")
        report.append(f"- Total records: {basic['total_records']}")
        report.append(f"- Total columns: {basic['total_columns']}")
        report.append(f"- Column match: {basic['column_match']}")
        if basic['missing_columns']:
            report.append(f"- Missing columns: {', '.join(basic['missing_columns'])}")
        if basic['extra_columns']:
            report.append(f"- Extra columns: {', '.join(basic['extra_columns'])}")
        report.append("")
        
        # Missing values
        missing = self.validation_results['missing_values']
        report.append("## Missing Values")
        if missing['required_field_violations']:
            report.append("Required field violations:")
            for violation in missing['required_field_violations']:
                report.append(f"- {violation['column']}: {violation['missing_count']} missing ({violation['missing_percentage']:.1f}%)")
        else:
            report.append("- No required field violations")
        report.append("")
        
        # Data type issues
        types = self.validation_results['data_types']
        report.append("## Data Type Issues")
        if types['type_conversion_issues']:
            for issue in types['type_conversion_issues']:
                report.append(f"- {issue['column']}: {', '.join(issue['issues'])}")
        else:
            report.append("- No data type issues")
        report.append("")
        
        # Range violations
        ranges = self.validation_results['range_validation']
        report.append("## Range Violations")
        if ranges['range_violations']:
            for column, violations in ranges['range_violations'].items():
                for violation in violations:
                    report.append(f"- {column}: {violation['count']} values {violation['type']}")
        else:
            report.append("- No range violations")
        report.append("")
        
        # Logical consistency
        logic = self.validation_results['logical_consistency']
        report.append("## Logical Consistency")
        for check_name, result in logic.items():
            if isinstance(result, dict) and 'valid' in result:
                status = "✓" if result['valid'] else "✗"
                report.append(f"- {check_name}: {status}")
                if not result['valid'] and 'issues' in result:
                    for issue in result['issues']:
                        report.append(f"  - {issue}")
        report.append("")
        
        # Duplicates
        dups = self.validation_results['duplicate_analysis']
        report.append("## Duplicate Analysis")
        report.append(f"- Exact duplicates: {dups['exact_duplicates']}")
        report.append(f"- Duplicate student IDs: {dups['duplicate_student_ids']}")
        report.append(f"- Near duplicates: {dups['near_duplicates']}")
        report.append("")
        
        return "\n".join(report)


def main():
    """
    Main function for testing the Comprehensive Data Validator.
    """
    # Example usage
    db_path = "data/raw/score.db"
    
    validator = ComprehensiveDataValidator(db_path=db_path)
    
    # Run comprehensive validation
    results = validator.validate_all()
    print(f"Validation completed. Quality score: {results['data_quality_score']}")
    
    # Generate report
    report = validator.generate_validation_report()
    print(report)
    
    # Save results
    validator.save_validation_results("data/processed")
    print("Results saved")


if __name__ == "__main__":
    main()