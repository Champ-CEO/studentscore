#!/usr/bin/env python3
"""
Comprehensive Data Entry Consistency Check Module

Implements Phase 3.1.6: Comprehensive data entry consistency check

This module provides comprehensive consistency checking across all data fields,
including format consistency, value consistency, cross-field validation,
and pattern detection for data entry errors.
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Set
import logging
from datetime import datetime, time
from collections import defaultdict, Counter
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveConsistencyChecker:
    """
    Comprehensive data entry consistency checker.
    
    Performs extensive consistency checks including:
    - Format consistency across similar fields
    - Value consistency and standardization
    - Cross-field logical consistency
    - Pattern detection for data entry errors
    - Encoding and case consistency
    - Temporal consistency
    """
    
    def __init__(self, db_path: Optional[str] = None, data: Optional[pd.DataFrame] = None):
        """
        Initialize the Comprehensive Consistency Checker.
        
        Args:
            db_path: Path to SQLite database file
            data: Pre-loaded DataFrame (alternative to db_path)
        """
        self.db_path = db_path
        self.data = data
        self.consistency_rules = self._define_consistency_rules()
        self.consistency_results = {}
        
    def _define_consistency_rules(self) -> Dict[str, Dict[str, Any]]:
        """
        Define consistency rules for data validation.
        
        Returns:
            Dictionary with consistency rules
        """
        rules = {
            'format_consistency': {
                'time_fields': {
                    'fields': ['sleep_time', 'wake_time'],
                    'expected_format': 'HH:MM',
                    'pattern': r'^([01]?[0-9]|2[0-3]):[0-5][0-9]$'
                },
                'boolean_fields': {
                    'fields': ['internet_access', 'CCA', 'tuition', 'direct_admission'],
                    'expected_values': ['Yes', 'No', 'Y', 'N', 'True', 'False', '1', '0'],
                    'standardized_values': {'Yes': ['yes', 'y', 'true', '1'], 'No': ['no', 'n', 'false', '0']}
                },
                'categorical_fields': {
                    'gender': {
                        'expected_values': ['Male', 'Female', 'M', 'F'],
                        'standardized_values': {'Male': ['male', 'm'], 'Female': ['female', 'f']}
                    },
                    'family_income': {
                        'expected_values': ['Low', 'Medium', 'High'],
                        'standardized_values': {'Low': ['low'], 'Medium': ['medium', 'med'], 'High': ['high']}
                    },
                    'transport_mode': {
                        'expected_values': ['Bus', 'Car', 'Walk', 'Bike', 'Train', 'Other'],
                        'standardized_values': {
                            'Bus': ['bus'], 'Car': ['car'], 'Walk': ['walk', 'walking'],
                            'Bike': ['bike', 'bicycle'], 'Train': ['train'], 'Other': ['other']
                        }
                    },
                    'study_location': {
                        'expected_values': ['Home', 'Library', 'School', 'Other'],
                        'standardized_values': {
                            'Home': ['home'], 'Library': ['library', 'lib'],
                            'School': ['school'], 'Other': ['other']
                        }
                    },
                    'learning_style': {
                        'expected_values': ['Visual', 'Auditory', 'Kinesthetic', 'Reading'],
                        'standardized_values': {
                            'Visual': ['visual'], 'Auditory': ['auditory', 'audio'],
                            'Kinesthetic': ['kinesthetic', 'kinetic'], 'Reading': ['reading', 'read']
                        }
                    }
                }
            },
            'value_consistency': {
                'numeric_ranges': {
                    'age': {'min': 10, 'max': 25, 'type': 'int'},
                    'hours_per_week': {'min': 0, 'max': 168, 'type': 'float'},
                    'attendance_rate': {'min': 0, 'max': 100, 'type': 'float'},
                    'final_test': {'min': 0, 'max': 100, 'type': 'float'}
                },
                'string_patterns': {
                    'student_id': {
                        'pattern': r'^[A-Za-z0-9]+$',
                        'min_length': 1,
                        'max_length': 50
                    }
                }
            },
            'cross_field_consistency': {
                'logical_relationships': [
                    {
                        'name': 'sleep_wake_consistency',
                        'fields': ['sleep_time', 'wake_time'],
                        'rule': 'wake_time_after_sleep_time'
                    },
                    {
                        'name': 'income_internet_consistency',
                        'fields': ['family_income', 'internet_access'],
                        'rule': 'high_income_likely_internet'
                    },
                    {
                        'name': 'age_study_hours_consistency',
                        'fields': ['age', 'hours_per_week'],
                        'rule': 'reasonable_study_hours_for_age'
                    },
                    {
                        'name': 'attendance_performance_consistency',
                        'fields': ['attendance_rate', 'final_test'],
                        'rule': 'attendance_performance_correlation'
                    }
                ]
            },
            'pattern_detection': {
                'suspicious_patterns': [
                    'repeated_values_across_records',
                    'sequential_patterns',
                    'copy_paste_errors',
                    'default_value_overuse',
                    'impossible_combinations'
                ]
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
    
    def check_all_consistency(self) -> Dict[str, Any]:
        """
        Run all consistency checks.
        
        Returns:
            Dictionary with comprehensive consistency results
        """
        if self.data is None:
            self.data = self.load_data()
        
        logger.info("Starting comprehensive consistency checking")
        
        consistency_results = {
            'format_consistency': self.check_format_consistency(),
            'value_consistency': self.check_value_consistency(),
            'cross_field_consistency': self.check_cross_field_consistency(),
            'pattern_detection': self.detect_suspicious_patterns(),
            'encoding_consistency': self.check_encoding_consistency(),
            'case_consistency': self.check_case_consistency(),
            'whitespace_consistency': self.check_whitespace_consistency(),
            'duplicate_consistency': self.check_duplicate_consistency(),
            'overall_consistency_score': None  # Will be calculated
        }
        
        # Calculate overall consistency score
        consistency_results['overall_consistency_score'] = self._calculate_consistency_score(consistency_results)
        
        self.consistency_results = consistency_results
        logger.info(f"Consistency checking completed. Score: {consistency_results['overall_consistency_score']:.2f}")
        
        return consistency_results
    
    def check_format_consistency(self) -> Dict[str, Any]:
        """
        Check format consistency across fields.
        
        Returns:
            Dictionary with format consistency results
        """
        results = {
            'time_format_issues': {},
            'boolean_format_issues': {},
            'categorical_format_issues': {},
            'format_standardization_suggestions': {}
        }
        
        # Check time format consistency
        time_fields = self.consistency_rules['format_consistency']['time_fields']
        pattern = re.compile(time_fields['pattern'])
        
        for field in time_fields['fields']:
            if field in self.data.columns:
                field_data = self.data[field].dropna().astype(str)
                invalid_formats = []
                
                for idx, value in field_data.items():
                    if not pattern.match(value):
                        invalid_formats.append({'index': idx, 'value': value})
                
                if invalid_formats:
                    results['time_format_issues'][field] = {
                        'count': len(invalid_formats),
                        'percentage': (len(invalid_formats) / len(field_data)) * 100,
                        'examples': invalid_formats[:5]  # First 5 examples
                    }
        
        # Check boolean format consistency
        boolean_fields = self.consistency_rules['format_consistency']['boolean_fields']
        expected_values = set(v.lower() for v in boolean_fields['expected_values'])
        
        for field in boolean_fields['fields']:
            if field in self.data.columns:
                field_data = self.data[field].dropna().astype(str)
                invalid_values = []
                
                for idx, value in field_data.items():
                    if value.lower() not in expected_values:
                        invalid_values.append({'index': idx, 'value': value})
                
                if invalid_values:
                    results['boolean_format_issues'][field] = {
                        'count': len(invalid_values),
                        'percentage': (len(invalid_values) / len(field_data)) * 100,
                        'examples': invalid_values[:5]
                    }
        
        # Check categorical format consistency
        categorical_fields = self.consistency_rules['format_consistency']['categorical_fields']
        
        for field, field_rules in categorical_fields.items():
            if field in self.data.columns:
                field_data = self.data[field].dropna().astype(str)
                expected_values = set(v.lower() for v in field_rules['expected_values'])
                invalid_values = []
                standardization_needed = []
                
                for idx, value in field_data.items():
                    value_lower = value.lower()
                    if value_lower not in expected_values:
                        # Check if it can be standardized
                        can_standardize = False
                        for standard_val, variants in field_rules['standardized_values'].items():
                            if value_lower in [v.lower() for v in variants]:
                                standardization_needed.append({
                                    'index': idx,
                                    'current_value': value,
                                    'suggested_value': standard_val
                                })
                                can_standardize = True
                                break
                        
                        if not can_standardize:
                            invalid_values.append({'index': idx, 'value': value})
                
                if invalid_values or standardization_needed:
                    results['categorical_format_issues'][field] = {
                        'invalid_count': len(invalid_values),
                        'standardization_count': len(standardization_needed),
                        'invalid_examples': invalid_values[:5],
                        'standardization_examples': standardization_needed[:5]
                    }
                
                if standardization_needed:
                    results['format_standardization_suggestions'][field] = standardization_needed
        
        return results
    
    def check_value_consistency(self) -> Dict[str, Any]:
        """
        Check value consistency and ranges.
        
        Returns:
            Dictionary with value consistency results
        """
        results = {
            'numeric_range_violations': {},
            'string_pattern_violations': {},
            'data_type_inconsistencies': {},
            'value_distribution_anomalies': {}
        }
        
        # Check numeric ranges
        numeric_ranges = self.consistency_rules['value_consistency']['numeric_ranges']
        
        for field, range_rules in numeric_ranges.items():
            if field in self.data.columns:
                field_data = pd.to_numeric(self.data[field], errors='coerce')
                violations = []
                
                # Check minimum values
                below_min = field_data < range_rules['min']
                if below_min.any():
                    violations.extend([
                        {'index': idx, 'value': val, 'violation': 'below_minimum', 'min_allowed': range_rules['min']}
                        for idx, val in field_data[below_min].items()
                    ])
                
                # Check maximum values
                above_max = field_data > range_rules['max']
                if above_max.any():
                    violations.extend([
                        {'index': idx, 'value': val, 'violation': 'above_maximum', 'max_allowed': range_rules['max']}
                        for idx, val in field_data[above_max].items()
                    ])
                
                # Check data type consistency
                if range_rules['type'] == 'int':
                    non_integer = field_data.dropna() != field_data.dropna().astype(int)
                    if non_integer.any():
                        violations.extend([
                            {'index': idx, 'value': val, 'violation': 'non_integer'}
                            for idx, val in field_data[non_integer].items()
                        ])
                
                if violations:
                    results['numeric_range_violations'][field] = {
                        'total_violations': len(violations),
                        'violation_percentage': (len(violations) / len(field_data.dropna())) * 100,
                        'violations': violations[:10]  # First 10 examples
                    }
        
        # Check string patterns
        string_patterns = self.consistency_rules['value_consistency']['string_patterns']
        
        for field, pattern_rules in string_patterns.items():
            if field in self.data.columns:
                field_data = self.data[field].dropna().astype(str)
                violations = []
                
                pattern = re.compile(pattern_rules['pattern'])
                
                for idx, value in field_data.items():
                    # Check pattern
                    if not pattern.match(value):
                        violations.append({
                            'index': idx,
                            'value': value,
                            'violation': 'pattern_mismatch'
                        })
                    
                    # Check length
                    if len(value) < pattern_rules['min_length']:
                        violations.append({
                            'index': idx,
                            'value': value,
                            'violation': 'too_short',
                            'min_length': pattern_rules['min_length']
                        })
                    
                    if len(value) > pattern_rules['max_length']:
                        violations.append({
                            'index': idx,
                            'value': value,
                            'violation': 'too_long',
                            'max_length': pattern_rules['max_length']
                        })
                
                if violations:
                    results['string_pattern_violations'][field] = {
                        'total_violations': len(violations),
                        'violations': violations[:10]
                    }
        
        # Check value distribution anomalies
        for column in self.data.columns:
            if self.data[column].dtype == 'object':
                value_counts = self.data[column].value_counts()
                total_count = len(self.data[column].dropna())
                
                # Check for excessive repetition of single values
                if len(value_counts) > 0:
                    most_common_pct = (value_counts.iloc[0] / total_count) * 100
                    if most_common_pct > 80:  # More than 80% same value
                        results['value_distribution_anomalies'][column] = {
                            'type': 'excessive_repetition',
                            'most_common_value': value_counts.index[0],
                            'percentage': most_common_pct,
                            'count': value_counts.iloc[0]
                        }
        
        return results
    
    def check_cross_field_consistency(self) -> Dict[str, Any]:
        """
        Check consistency across multiple fields.
        
        Returns:
            Dictionary with cross-field consistency results
        """
        results = {
            'logical_inconsistencies': {},
            'correlation_anomalies': {},
            'impossible_combinations': []
        }
        
        logical_relationships = self.consistency_rules['cross_field_consistency']['logical_relationships']
        
        for relationship in logical_relationships:
            rule_name = relationship['name']
            fields = relationship['fields']
            rule_type = relationship['rule']
            
            # Check if all required fields exist
            if all(field in self.data.columns for field in fields):
                inconsistencies = self._check_logical_rule(rule_type, fields)
                
                if inconsistencies:
                    results['logical_inconsistencies'][rule_name] = {
                        'rule_type': rule_type,
                        'fields': fields,
                        'inconsistency_count': len(inconsistencies),
                        'inconsistencies': inconsistencies[:10]  # First 10 examples
                    }
        
        # Check for impossible combinations
        impossible_combinations = self._detect_impossible_combinations()
        results['impossible_combinations'] = impossible_combinations
        
        return results
    
    def detect_suspicious_patterns(self) -> Dict[str, Any]:
        """
        Detect suspicious data entry patterns.
        
        Returns:
            Dictionary with suspicious pattern detection results
        """
        results = {
            'repeated_values_across_records': {},
            'sequential_patterns': {},
            'copy_paste_errors': {},
            'default_value_overuse': {},
            'suspicious_uniformity': {}
        }
        
        # Detect repeated values across records
        results['repeated_values_across_records'] = self._detect_repeated_values()
        
        # Detect sequential patterns
        results['sequential_patterns'] = self._detect_sequential_patterns()
        
        # Detect copy-paste errors
        results['copy_paste_errors'] = self._detect_copy_paste_errors()
        
        # Detect default value overuse
        results['default_value_overuse'] = self._detect_default_value_overuse()
        
        # Detect suspicious uniformity
        results['suspicious_uniformity'] = self._detect_suspicious_uniformity()
        
        return results
    
    def check_encoding_consistency(self) -> Dict[str, Any]:
        """
        Check character encoding consistency.
        
        Returns:
            Dictionary with encoding consistency results
        """
        results = {
            'encoding_issues': {},
            'special_characters': {},
            'unicode_anomalies': []
        }
        
        for column in self.data.columns:
            if self.data[column].dtype == 'object':
                column_data = self.data[column].dropna().astype(str)
                
                encoding_issues = []
                special_chars = set()
                
                for idx, value in column_data.items():
                    # Check for non-ASCII characters
                    try:
                        value.encode('ascii')
                    except UnicodeEncodeError:
                        encoding_issues.append({
                            'index': idx,
                            'value': value,
                            'issue': 'non_ascii_characters'
                        })
                    
                    # Collect special characters
                    for char in value:
                        if not char.isalnum() and not char.isspace() and char not in '.,!?-_()[]{}"\'':
                            special_chars.add(char)
                
                if encoding_issues:
                    results['encoding_issues'][column] = {
                        'count': len(encoding_issues),
                        'examples': encoding_issues[:5]
                    }
                
                if special_chars:
                    results['special_characters'][column] = list(special_chars)
        
        return results
    
    def check_case_consistency(self) -> Dict[str, Any]:
        """
        Check case consistency in text fields.
        
        Returns:
            Dictionary with case consistency results
        """
        results = {
            'case_inconsistencies': {},
            'mixed_case_issues': {},
            'standardization_suggestions': {}
        }
        
        for column in self.data.columns:
            if self.data[column].dtype == 'object':
                column_data = self.data[column].dropna().astype(str)
                
                if len(column_data) == 0:
                    continue
                
                # Group by lowercase version to find case inconsistencies
                case_groups = defaultdict(list)
                for idx, value in column_data.items():
                    case_groups[value.lower()].append((idx, value))
                
                inconsistencies = []
                for lower_value, variants in case_groups.items():
                    if len(set(variant[1] for variant in variants)) > 1:
                        # Multiple case variants exist
                        variant_counts = Counter(variant[1] for variant in variants)
                        most_common = variant_counts.most_common(1)[0][0]
                        
                        inconsistencies.append({
                            'base_value': lower_value,
                            'variants': list(variant_counts.keys()),
                            'counts': dict(variant_counts),
                            'suggested_standard': most_common
                        })
                
                if inconsistencies:
                    results['case_inconsistencies'][column] = inconsistencies
        
        return results
    
    def check_whitespace_consistency(self) -> Dict[str, Any]:
        """
        Check whitespace consistency in text fields.
        
        Returns:
            Dictionary with whitespace consistency results
        """
        results = {
            'leading_whitespace': {},
            'trailing_whitespace': {},
            'multiple_spaces': {},
            'tab_characters': {}
        }
        
        for column in self.data.columns:
            if self.data[column].dtype == 'object':
                column_data = self.data[column].dropna().astype(str)
                
                leading_ws = []
                trailing_ws = []
                multiple_spaces = []
                tab_chars = []
                
                for idx, value in column_data.items():
                    # Check leading whitespace
                    if value != value.lstrip():
                        leading_ws.append({'index': idx, 'value': repr(value)})
                    
                    # Check trailing whitespace
                    if value != value.rstrip():
                        trailing_ws.append({'index': idx, 'value': repr(value)})
                    
                    # Check multiple consecutive spaces
                    if '  ' in value:  # Two or more spaces
                        multiple_spaces.append({'index': idx, 'value': repr(value)})
                    
                    # Check tab characters
                    if '\t' in value:
                        tab_chars.append({'index': idx, 'value': repr(value)})
                
                if leading_ws:
                    results['leading_whitespace'][column] = {
                        'count': len(leading_ws),
                        'examples': leading_ws[:5]
                    }
                
                if trailing_ws:
                    results['trailing_whitespace'][column] = {
                        'count': len(trailing_ws),
                        'examples': trailing_ws[:5]
                    }
                
                if multiple_spaces:
                    results['multiple_spaces'][column] = {
                        'count': len(multiple_spaces),
                        'examples': multiple_spaces[:5]
                    }
                
                if tab_chars:
                    results['tab_characters'][column] = {
                        'count': len(tab_chars),
                        'examples': tab_chars[:5]
                    }
        
        return results
    
    def check_duplicate_consistency(self) -> Dict[str, Any]:
        """
        Check for duplicate-related consistency issues.
        
        Returns:
            Dictionary with duplicate consistency results
        """
        results = {
            'exact_duplicates': 0,
            'near_duplicates': [],
            'duplicate_key_fields': {},
            'suspicious_similarities': []
        }
        
        # Check exact duplicates
        exact_dups = self.data.duplicated().sum()
        results['exact_duplicates'] = exact_dups
        
        # Check duplicate key fields
        key_fields = ['student_id']  # Add other key fields as needed
        
        for field in key_fields:
            if field in self.data.columns:
                duplicates = self.data[field].duplicated().sum()
                if duplicates > 0:
                    dup_values = self.data[self.data[field].duplicated(keep=False)][field].value_counts()
                    results['duplicate_key_fields'][field] = {
                        'count': duplicates,
                        'duplicate_values': dup_values.to_dict()
                    }
        
        # Check near duplicates (records that are very similar)
        near_duplicates = self._detect_near_duplicates()
        results['near_duplicates'] = near_duplicates
        
        return results
    
    def _check_logical_rule(self, rule_type: str, fields: List[str]) -> List[Dict[str, Any]]:
        """
        Check a specific logical rule.
        
        Args:
            rule_type: Type of logical rule to check
            fields: Fields involved in the rule
            
        Returns:
            List of inconsistencies found
        """
        inconsistencies = []
        
        if rule_type == 'wake_time_after_sleep_time':
            # This is complex due to overnight sleep patterns
            # For now, we'll check for obvious issues
            sleep_col, wake_col = fields
            
            for idx, row in self.data.iterrows():
                sleep_time = row[sleep_col]
                wake_time = row[wake_col]
                
                if pd.notnull(sleep_time) and pd.notnull(wake_time):
                    try:
                        sleep_dt = datetime.strptime(str(sleep_time), '%H:%M')
                        wake_dt = datetime.strptime(str(wake_time), '%H:%M')
                        
                        # Check for same time (suspicious)
                        if sleep_dt == wake_dt:
                            inconsistencies.append({
                                'index': idx,
                                'issue': 'same_sleep_wake_time',
                                'sleep_time': sleep_time,
                                'wake_time': wake_time
                            })
                    except ValueError:
                        # Invalid time format - will be caught by format checking
                        pass
        
        elif rule_type == 'high_income_likely_internet':
            income_col, internet_col = fields
            
            high_income_no_internet = (
                (self.data[income_col].str.lower() == 'high') &
                (self.data[internet_col].str.lower().isin(['no', 'n', 'false', '0']))
            )
            
            for idx in self.data[high_income_no_internet].index:
                inconsistencies.append({
                    'index': idx,
                    'issue': 'high_income_no_internet',
                    'family_income': self.data.loc[idx, income_col],
                    'internet_access': self.data.loc[idx, internet_col]
                })
        
        elif rule_type == 'reasonable_study_hours_for_age':
            age_col, hours_col = fields
            
            for idx, row in self.data.iterrows():
                age = pd.to_numeric(row[age_col], errors='coerce')
                hours = pd.to_numeric(row[hours_col], errors='coerce')
                
                if pd.notnull(age) and pd.notnull(hours):
                    # Very young students with excessive study hours
                    if age <= 14 and hours > 60:
                        inconsistencies.append({
                            'index': idx,
                            'issue': 'young_student_excessive_hours',
                            'age': age,
                            'hours_per_week': hours
                        })
                    
                    # Any student with more hours than possible
                    if hours > 168:  # More than hours in a week
                        inconsistencies.append({
                            'index': idx,
                            'issue': 'impossible_study_hours',
                            'age': age,
                            'hours_per_week': hours
                        })
        
        elif rule_type == 'attendance_performance_correlation':
            attendance_col, performance_col = fields
            
            # Check for very low attendance with very high performance (suspicious)
            for idx, row in self.data.iterrows():
                attendance = pd.to_numeric(row[attendance_col], errors='coerce')
                performance = pd.to_numeric(row[performance_col], errors='coerce')
                
                if pd.notnull(attendance) and pd.notnull(performance):
                    if attendance < 30 and performance > 90:
                        inconsistencies.append({
                            'index': idx,
                            'issue': 'low_attendance_high_performance',
                            'attendance_rate': attendance,
                            'final_test': performance
                        })
        
        return inconsistencies
    
    def _detect_impossible_combinations(self) -> List[Dict[str, Any]]:
        """
        Detect impossible feature combinations.
        
        Returns:
            List of impossible combinations found
        """
        impossible_combinations = []
        
        # Example: Students claiming to study 0 hours but having high attendance
        if 'hours_per_week' in self.data.columns and 'attendance_rate' in self.data.columns:
            zero_hours_high_attendance = (
                (pd.to_numeric(self.data['hours_per_week'], errors='coerce') == 0) &
                (pd.to_numeric(self.data['attendance_rate'], errors='coerce') > 80)
            )
            
            for idx in self.data[zero_hours_high_attendance].index:
                impossible_combinations.append({
                    'index': idx,
                    'type': 'zero_study_hours_high_attendance',
                    'description': 'Student claims 0 study hours but has high attendance',
                    'hours_per_week': self.data.loc[idx, 'hours_per_week'],
                    'attendance_rate': self.data.loc[idx, 'attendance_rate']
                })
        
        return impossible_combinations
    
    def _detect_repeated_values(self) -> Dict[str, Any]:
        """
        Detect suspicious repeated values across records.
        
        Returns:
            Dictionary with repeated value analysis
        """
        repeated_analysis = {}
        
        for column in self.data.columns:
            if column != 'student_id':  # Skip ID column
                value_counts = self.data[column].value_counts()
                total_records = len(self.data[column].dropna())
                
                if len(value_counts) > 0:
                    most_common_count = value_counts.iloc[0]
                    most_common_pct = (most_common_count / total_records) * 100
                    
                    # Flag if more than 50% of records have the same value
                    if most_common_pct > 50 and most_common_count > 10:
                        repeated_analysis[column] = {
                            'most_common_value': value_counts.index[0],
                            'count': most_common_count,
                            'percentage': most_common_pct,
                            'total_unique_values': len(value_counts)
                        }
        
        return repeated_analysis
    
    def _detect_sequential_patterns(self) -> Dict[str, Any]:
        """
        Detect sequential patterns that might indicate data entry errors.
        
        Returns:
            Dictionary with sequential pattern analysis
        """
        sequential_patterns = {}
        
        # Check numeric columns for sequential patterns
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column in ['student_id']:  # Skip ID-like columns
                continue
            
            column_data = self.data[column].dropna().sort_values()
            
            if len(column_data) > 5:
                # Check for arithmetic sequences
                diffs = column_data.diff().dropna()
                
                # If most differences are the same, it might be sequential
                if len(diffs.unique()) <= 3 and len(diffs) > 10:
                    most_common_diff = diffs.mode().iloc[0] if not diffs.mode().empty else None
                    
                    if most_common_diff and most_common_diff != 0:
                        sequential_patterns[column] = {
                            'type': 'arithmetic_sequence',
                            'common_difference': most_common_diff,
                            'sequence_length': len(diffs[diffs == most_common_diff])
                        }
        
        return sequential_patterns
    
    def _detect_copy_paste_errors(self) -> Dict[str, Any]:
        """
        Detect potential copy-paste errors.
        
        Returns:
            Dictionary with copy-paste error analysis
        """
        copy_paste_errors = {}
        
        # Look for identical rows (excluding student_id)
        comparison_columns = [col for col in self.data.columns if col != 'student_id']
        
        if comparison_columns:
            # Group by all columns except student_id
            grouped = self.data.groupby(comparison_columns).size()
            duplicated_patterns = grouped[grouped > 1]
            
            if len(duplicated_patterns) > 0:
                copy_paste_errors['identical_records'] = {
                    'count': len(duplicated_patterns),
                    'total_affected_records': duplicated_patterns.sum(),
                    'examples': duplicated_patterns.head().to_dict()
                }
        
        # Look for repeated sequences of values across multiple columns
        for i in range(len(self.data) - 1):
            row1 = self.data.iloc[i]
            row2 = self.data.iloc[i + 1]
            
            # Count how many fields are identical
            identical_fields = sum(1 for col in comparison_columns 
                                 if pd.notnull(row1[col]) and pd.notnull(row2[col]) and row1[col] == row2[col])
            
            # If most fields are identical, it might be copy-paste
            if identical_fields > len(comparison_columns) * 0.8 and identical_fields > 5:
                if 'consecutive_similar_records' not in copy_paste_errors:
                    copy_paste_errors['consecutive_similar_records'] = []
                
                copy_paste_errors['consecutive_similar_records'].append({
                    'row1_index': i,
                    'row2_index': i + 1,
                    'identical_fields': identical_fields,
                    'total_fields': len(comparison_columns)
                })
        
        return copy_paste_errors
    
    def _detect_default_value_overuse(self) -> Dict[str, Any]:
        """
        Detect overuse of default values.
        
        Returns:
            Dictionary with default value analysis
        """
        default_value_analysis = {}
        
        # Common default values to check for
        common_defaults = {
            'numeric': [0, 1, -1, 99, 999],
            'text': ['N/A', 'NA', 'NULL', 'null', 'None', 'none', '', 'Unknown', 'unknown', 'Default', 'default'],
            'boolean': ['No', 'False', '0']
        }
        
        for column in self.data.columns:
            column_data = self.data[column].dropna()
            
            if len(column_data) == 0:
                continue
            
            # Check for numeric defaults
            if pd.api.types.is_numeric_dtype(column_data):
                for default_val in common_defaults['numeric']:
                    count = (column_data == default_val).sum()
                    percentage = (count / len(column_data)) * 100
                    
                    if percentage > 30:  # More than 30% default values
                        if column not in default_value_analysis:
                            default_value_analysis[column] = []
                        
                        default_value_analysis[column].append({
                            'default_value': default_val,
                            'count': count,
                            'percentage': percentage
                        })
            
            # Check for text defaults
            else:
                column_str = column_data.astype(str)
                for default_val in common_defaults['text']:
                    count = (column_str.str.lower() == default_val.lower()).sum()
                    percentage = (count / len(column_data)) * 100
                    
                    if percentage > 30:
                        if column not in default_value_analysis:
                            default_value_analysis[column] = []
                        
                        default_value_analysis[column].append({
                            'default_value': default_val,
                            'count': count,
                            'percentage': percentage
                        })
        
        return default_value_analysis
    
    def _detect_suspicious_uniformity(self) -> Dict[str, Any]:
        """
        Detect suspicious uniformity in data.
        
        Returns:
            Dictionary with uniformity analysis
        """
        uniformity_analysis = {}
        
        for column in self.data.columns:
            if self.data[column].dtype == 'object':
                unique_values = self.data[column].nunique()
                total_values = len(self.data[column].dropna())
                
                if total_values > 0:
                    uniqueness_ratio = unique_values / total_values
                    
                    # Very low uniqueness might indicate data entry issues
                    if uniqueness_ratio < 0.1 and unique_values > 1:
                        uniformity_analysis[column] = {
                            'unique_values': unique_values,
                            'total_values': total_values,
                            'uniqueness_ratio': uniqueness_ratio,
                            'top_values': self.data[column].value_counts().head().to_dict()
                        }
        
        return uniformity_analysis
    
    def _detect_near_duplicates(self) -> List[Dict[str, Any]]:
        """
        Detect near-duplicate records.
        
        Returns:
            List of near-duplicate record pairs
        """
        near_duplicates = []
        
        # Sample for performance (checking all pairs is expensive)
        sample_size = min(500, len(self.data))
        sample_data = self.data.sample(n=sample_size, random_state=42)
        
        comparison_columns = [col for col in sample_data.columns if col != 'student_id']
        
        for i in range(len(sample_data)):
            for j in range(i + 1, len(sample_data)):
                row1 = sample_data.iloc[i]
                row2 = sample_data.iloc[j]
                
                # Count matching fields
                matches = 0
                total_comparable = 0
                
                for col in comparison_columns:
                    val1, val2 = row1[col], row2[col]
                    
                    if pd.notnull(val1) and pd.notnull(val2):
                        total_comparable += 1
                        if val1 == val2:
                            matches += 1
                
                # If most fields match, consider it a near duplicate
                if total_comparable > 0:
                    similarity = matches / total_comparable
                    
                    if 0.8 <= similarity < 1.0:  # 80-99% similar
                        near_duplicates.append({
                            'index1': sample_data.index[i],
                            'index2': sample_data.index[j],
                            'similarity': similarity,
                            'matching_fields': matches,
                            'total_fields': total_comparable
                        })
        
        return near_duplicates
    
    def _calculate_consistency_score(self, results: Dict[str, Any]) -> float:
        """
        Calculate overall consistency score.
        
        Args:
            results: Dictionary with all consistency check results
            
        Returns:
            Consistency score between 0 and 100
        """
        score_components = []
        
        # Format consistency (20% weight)
        format_issues = (
            len(results['format_consistency']['time_format_issues']) +
            len(results['format_consistency']['boolean_format_issues']) +
            len(results['format_consistency']['categorical_format_issues'])
        )
        format_score = max(0, 100 - format_issues * 10)
        score_components.append(('format', format_score, 0.2))
        
        # Value consistency (20% weight)
        value_issues = (
            len(results['value_consistency']['numeric_range_violations']) +
            len(results['value_consistency']['string_pattern_violations'])
        )
        value_score = max(0, 100 - value_issues * 15)
        score_components.append(('value', value_score, 0.2))
        
        # Cross-field consistency (15% weight)
        cross_field_issues = len(results['cross_field_consistency']['logical_inconsistencies'])
        cross_field_score = max(0, 100 - cross_field_issues * 20)
        score_components.append(('cross_field', cross_field_score, 0.15))
        
        # Pattern detection (15% weight)
        pattern_issues = (
            len(results['pattern_detection']['repeated_values_across_records']) +
            len(results['pattern_detection']['copy_paste_errors'])
        )
        pattern_score = max(0, 100 - pattern_issues * 10)
        score_components.append(('patterns', pattern_score, 0.15))
        
        # Encoding consistency (10% weight)
        encoding_issues = len(results['encoding_consistency']['encoding_issues'])
        encoding_score = max(0, 100 - encoding_issues * 20)
        score_components.append(('encoding', encoding_score, 0.1))
        
        # Case consistency (10% weight)
        case_issues = len(results['case_consistency']['case_inconsistencies'])
        case_score = max(0, 100 - case_issues * 15)
        score_components.append(('case', case_score, 0.1))
        
        # Whitespace consistency (10% weight)
        whitespace_issues = (
            len(results['whitespace_consistency']['leading_whitespace']) +
            len(results['whitespace_consistency']['trailing_whitespace'])
        )
        whitespace_score = max(0, 100 - whitespace_issues * 10)
        score_components.append(('whitespace', whitespace_score, 0.1))
        
        # Calculate weighted average
        total_score = sum(score * weight for _, score, weight in score_components)
        
        return round(total_score, 2)
    
    def save_consistency_results(self, output_dir: str) -> None:
        """
        Save consistency check results to files.
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save consistency results
        results_file = output_path / 'consistency_check_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.consistency_results, f, indent=2, default=str)
        
        logger.info(f"Consistency check results saved to {results_file}")
    
    def generate_consistency_report(self) -> str:
        """
        Generate a comprehensive consistency report.
        
        Returns:
            Consistency report string
        """
        if not self.consistency_results:
            return "No consistency results available. Run check_all_consistency() first."
        
        report = []
        report.append("=== Comprehensive Data Consistency Report ===")
        report.append("")
        
        # Overall score
        score = self.consistency_results.get('overall_consistency_score', 0)
        report.append(f"Overall Consistency Score: {score}/100")
        report.append("")
        
        # Format consistency
        format_results = self.consistency_results['format_consistency']
        report.append("## Format Consistency")
        
        time_issues = len(format_results['time_format_issues'])
        boolean_issues = len(format_results['boolean_format_issues'])
        categorical_issues = len(format_results['categorical_format_issues'])
        
        report.append(f"- Time format issues: {time_issues} fields")
        report.append(f"- Boolean format issues: {boolean_issues} fields")
        report.append(f"- Categorical format issues: {categorical_issues} fields")
        report.append("")
        
        # Value consistency
        value_results = self.consistency_results['value_consistency']
        report.append("## Value Consistency")
        
        range_violations = len(value_results['numeric_range_violations'])
        pattern_violations = len(value_results['string_pattern_violations'])
        
        report.append(f"- Numeric range violations: {range_violations} fields")
        report.append(f"- String pattern violations: {pattern_violations} fields")
        report.append("")
        
        # Cross-field consistency
        cross_results = self.consistency_results['cross_field_consistency']
        report.append("## Cross-Field Consistency")
        
        logical_issues = len(cross_results['logical_inconsistencies'])
        impossible_combinations = len(cross_results['impossible_combinations'])
        
        report.append(f"- Logical inconsistencies: {logical_issues} types")
        report.append(f"- Impossible combinations: {impossible_combinations} cases")
        report.append("")
        
        # Pattern detection
        pattern_results = self.consistency_results['pattern_detection']
        report.append("## Suspicious Patterns")
        
        repeated_values = len(pattern_results['repeated_values_across_records'])
        copy_paste = len(pattern_results['copy_paste_errors'])
        default_overuse = len(pattern_results['default_value_overuse'])
        
        report.append(f"- Fields with repeated values: {repeated_values}")
        report.append(f"- Copy-paste error types: {copy_paste}")
        report.append(f"- Default value overuse: {default_overuse} fields")
        report.append("")
        
        # Encoding and formatting
        encoding_results = self.consistency_results['encoding_consistency']
        case_results = self.consistency_results['case_consistency']
        whitespace_results = self.consistency_results['whitespace_consistency']
        
        report.append("## Encoding and Formatting")
        
        encoding_issues = len(encoding_results['encoding_issues'])
        case_issues = len(case_results['case_inconsistencies'])
        whitespace_issues = (
            len(whitespace_results['leading_whitespace']) +
            len(whitespace_results['trailing_whitespace'])
        )
        
        report.append(f"- Encoding issues: {encoding_issues} fields")
        report.append(f"- Case inconsistencies: {case_issues} fields")
        report.append(f"- Whitespace issues: {whitespace_issues} fields")
        report.append("")
        
        return "\n".join(report)


def main():
    """
    Main function for testing the Comprehensive Consistency Checker.
    """
    # Example usage
    db_path = "data/raw/score.db"
    
    checker = ComprehensiveConsistencyChecker(db_path=db_path)
    
    # Run comprehensive consistency checking
    results = checker.check_all_consistency()
    print(f"Consistency checking completed. Score: {results['overall_consistency_score']}")
    
    # Generate report
    report = checker.generate_consistency_report()
    print(report)
    
    # Save results
    checker.save_consistency_results("data/processed")
    print("Results saved")


if __name__ == "__main__":
    main()