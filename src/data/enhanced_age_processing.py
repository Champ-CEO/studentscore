#!/usr/bin/env python3
"""
Enhanced Age Processing and Feature Engineering Module

Implements Phase 3.1.5: Enhanced age processing and feature engineering

This module provides advanced age processing capabilities including:
- Age validation and correction
- Age-based feature engineering
- Age group categorization
- Age-related outlier detection
- Age consistency checks
"""

import pandas as pd
import numpy as np
import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedAgeProcessor:
    """
    Enhanced age processing and feature engineering.
    
    Provides comprehensive age-related data processing including:
    - Age validation and correction
    - Feature engineering from age data
    - Age group categorization
    - Age-based outlier detection
    - Statistical age analysis
    """
    
    def __init__(self, db_path: Optional[str] = None, data: Optional[pd.DataFrame] = None):
        """
        Initialize the Enhanced Age Processor.
        
        Args:
            db_path: Path to SQLite database file
            data: Pre-loaded DataFrame (alternative to db_path)
        """
        self.db_path = db_path
        self.data = data
        self.age_statistics = {}
        self.age_rules = self._define_age_rules()
        self.processing_results = {}
        
    def _define_age_rules(self) -> Dict[str, Any]:
        """
        Define age validation and processing rules.
        
        Returns:
            Dictionary with age processing rules
        """
        rules = {
            'valid_range': {
                'min_age': 10,  # Minimum reasonable age for students
                'max_age': 25,  # Maximum reasonable age for students
                'typical_min': 12,  # Typical minimum age
                'typical_max': 22   # Typical maximum age
            },
            'correction_rules': {
                'negative_age_action': 'flag_and_impute',  # Options: flag_and_impute, remove, manual_review
                'extreme_age_action': 'flag_and_review',   # Options: flag_and_review, cap, remove
                'decimal_handling': 'round',               # Options: round, floor, ceil, keep
                'missing_imputation': 'median_by_group'    # Options: median, mean, mode, median_by_group
            },
            'feature_engineering': {
                'create_age_groups': True,
                'create_age_bins': True,
                'create_age_percentiles': True,
                'create_age_zscore': True,
                'create_age_categories': True
            },
            'outlier_detection': {
                'methods': ['iqr', 'zscore', 'isolation_forest'],
                'zscore_threshold': 3.0,
                'iqr_multiplier': 1.5
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
    
    def process_age_comprehensive(self) -> Dict[str, Any]:
        """
        Run comprehensive age processing.
        
        Returns:
            Dictionary with processing results and enhanced data
        """
        if self.data is None:
            self.data = self.load_data()
        
        logger.info("Starting comprehensive age processing")
        
        # Make a copy for processing
        processed_data = self.data.copy()
        
        # Step 1: Validate and analyze current age data
        validation_results = self.validate_age_data(processed_data)
        
        # Step 2: Clean and correct age data
        processed_data, correction_results = self.clean_and_correct_ages(processed_data)
        
        # Step 3: Generate age statistics
        age_stats = self.generate_age_statistics(processed_data)
        
        # Step 4: Detect age outliers
        outlier_results = self.detect_age_outliers(processed_data)
        
        # Step 5: Create age-based features
        processed_data, feature_results = self.engineer_age_features(processed_data)
        
        # Step 6: Perform age group analysis
        group_analysis = self.analyze_age_groups(processed_data)
        
        # Step 7: Age consistency checks
        consistency_results = self.check_age_consistency(processed_data)
        
        # Compile results
        processing_results = {
            'validation': validation_results,
            'correction': correction_results,
            'statistics': age_stats,
            'outliers': outlier_results,
            'features': feature_results,
            'group_analysis': group_analysis,
            'consistency': consistency_results,
            'processed_data': processed_data
        }
        
        self.processing_results = processing_results
        logger.info("Age processing completed successfully")
        
        return processing_results
    
    def validate_age_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate age data quality and identify issues.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        if 'age' not in data.columns:
            return {'error': 'Age column not found in data'}
        
        age_series = data['age']
        validation_results = {
            'total_records': len(data),
            'missing_ages': age_series.isnull().sum(),
            'missing_percentage': (age_series.isnull().sum() / len(data)) * 100,
            'data_type_issues': [],
            'range_violations': {},
            'decimal_ages': 0,
            'negative_ages': 0,
            'zero_ages': 0,
            'extreme_ages': []
        }
        
        # Check data types
        non_numeric_mask = pd.to_numeric(age_series, errors='coerce').isnull() & age_series.notnull()
        if non_numeric_mask.any():
            validation_results['data_type_issues'] = age_series[non_numeric_mask].tolist()
        
        # Convert to numeric for further analysis
        numeric_ages = pd.to_numeric(age_series, errors='coerce')
        valid_ages = numeric_ages.dropna()
        
        if len(valid_ages) > 0:
            # Check for decimal ages
            decimal_mask = valid_ages != valid_ages.astype(int)
            validation_results['decimal_ages'] = decimal_mask.sum()
            
            # Check for negative ages
            negative_mask = valid_ages < 0
            validation_results['negative_ages'] = negative_mask.sum()
            
            # Check for zero ages
            zero_mask = valid_ages == 0
            validation_results['zero_ages'] = zero_mask.sum()
            
            # Check range violations
            min_age = self.age_rules['valid_range']['min_age']
            max_age = self.age_rules['valid_range']['max_age']
            
            below_min = valid_ages < min_age
            above_max = valid_ages > max_age
            
            validation_results['range_violations'] = {
                'below_minimum': {
                    'count': below_min.sum(),
                    'values': valid_ages[below_min].tolist()
                },
                'above_maximum': {
                    'count': above_max.sum(),
                    'values': valid_ages[above_max].tolist()
                }
            }
            
            # Identify extreme ages
            typical_min = self.age_rules['valid_range']['typical_min']
            typical_max = self.age_rules['valid_range']['typical_max']
            
            extreme_mask = (valid_ages < typical_min) | (valid_ages > typical_max)
            if extreme_mask.any():
                validation_results['extreme_ages'] = valid_ages[extreme_mask].tolist()
        
        return validation_results
    
    def clean_and_correct_ages(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Clean and correct age data based on defined rules.
        
        Args:
            data: DataFrame to clean
            
        Returns:
            Tuple of (cleaned DataFrame, correction results)
        """
        if 'age' not in data.columns:
            return data, {'error': 'Age column not found'}
        
        cleaned_data = data.copy()
        correction_results = {
            'corrections_made': [],
            'flagged_records': [],
            'imputed_values': [],
            'removed_records': []
        }
        
        # Convert age to numeric
        original_ages = cleaned_data['age'].copy()
        cleaned_data['age'] = pd.to_numeric(cleaned_data['age'], errors='coerce')
        
        # Track conversion issues
        conversion_issues = original_ages.notnull() & cleaned_data['age'].isnull()
        if conversion_issues.any():
            correction_results['flagged_records'].extend([
                {'index': idx, 'issue': 'non_numeric_age', 'original_value': original_ages.iloc[idx]}
                for idx in conversion_issues[conversion_issues].index
            ])
        
        # Handle negative ages
        negative_mask = cleaned_data['age'] < 0
        if negative_mask.any():
            if self.age_rules['correction_rules']['negative_age_action'] == 'flag_and_impute':
                # Flag negative ages and impute with median
                median_age = cleaned_data['age'][cleaned_data['age'] >= 0].median()
                
                for idx in negative_mask[negative_mask].index:
                    correction_results['flagged_records'].append({
                        'index': idx,
                        'issue': 'negative_age',
                        'original_value': cleaned_data.loc[idx, 'age'],
                        'corrected_value': median_age
                    })
                
                cleaned_data.loc[negative_mask, 'age'] = median_age
                correction_results['corrections_made'].append(f"Corrected {negative_mask.sum()} negative ages")
        
        # Handle decimal ages
        if self.age_rules['correction_rules']['decimal_handling'] == 'round':
            decimal_mask = cleaned_data['age'] != cleaned_data['age'].round()
            if decimal_mask.any():
                original_decimals = cleaned_data.loc[decimal_mask, 'age'].copy()
                cleaned_data.loc[decimal_mask, 'age'] = cleaned_data.loc[decimal_mask, 'age'].round()
                
                correction_results['corrections_made'].append(f"Rounded {decimal_mask.sum()} decimal ages")
        
        # Handle extreme ages
        min_age = self.age_rules['valid_range']['min_age']
        max_age = self.age_rules['valid_range']['max_age']
        
        extreme_mask = (cleaned_data['age'] < min_age) | (cleaned_data['age'] > max_age)
        if extreme_mask.any():
            if self.age_rules['correction_rules']['extreme_age_action'] == 'flag_and_review':
                for idx in extreme_mask[extreme_mask].index:
                    correction_results['flagged_records'].append({
                        'index': idx,
                        'issue': 'extreme_age',
                        'value': cleaned_data.loc[idx, 'age']
                    })
        
        # Handle missing ages
        missing_mask = cleaned_data['age'].isnull()
        if missing_mask.any():
            if self.age_rules['correction_rules']['missing_imputation'] == 'median_by_group':
                # Impute by gender if available, otherwise overall median
                if 'gender' in cleaned_data.columns:
                    for gender in cleaned_data['gender'].unique():
                        if pd.notnull(gender):
                            gender_mask = cleaned_data['gender'] == gender
                            gender_missing = missing_mask & gender_mask
                            
                            if gender_missing.any():
                                gender_median = cleaned_data.loc[gender_mask & ~missing_mask, 'age'].median()
                                if pd.notnull(gender_median):
                                    cleaned_data.loc[gender_missing, 'age'] = gender_median
                                    
                                    correction_results['imputed_values'].extend([
                                        {'index': idx, 'imputed_value': gender_median, 'method': f'median_by_gender_{gender}'}
                                        for idx in gender_missing[gender_missing].index
                                    ])
                
                # Impute remaining missing values with overall median
                still_missing = cleaned_data['age'].isnull()
                if still_missing.any():
                    overall_median = cleaned_data['age'].median()
                    cleaned_data.loc[still_missing, 'age'] = overall_median
                    
                    correction_results['imputed_values'].extend([
                        {'index': idx, 'imputed_value': overall_median, 'method': 'overall_median'}
                        for idx in still_missing[still_missing].index
                    ])
                
                correction_results['corrections_made'].append(f"Imputed {missing_mask.sum()} missing ages")
        
        return cleaned_data, correction_results
    
    def generate_age_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive age statistics.
        
        Args:
            data: DataFrame with cleaned age data
            
        Returns:
            Dictionary with age statistics
        """
        if 'age' not in data.columns:
            return {'error': 'Age column not found'}
        
        age_series = data['age'].dropna()
        
        if len(age_series) == 0:
            return {'error': 'No valid age data found'}
        
        statistics = {
            'basic_stats': {
                'count': len(age_series),
                'mean': float(age_series.mean()),
                'median': float(age_series.median()),
                'mode': float(age_series.mode().iloc[0]) if not age_series.mode().empty else None,
                'std': float(age_series.std()),
                'variance': float(age_series.var()),
                'min': float(age_series.min()),
                'max': float(age_series.max()),
                'range': float(age_series.max() - age_series.min())
            },
            'percentiles': {
                'p5': float(age_series.quantile(0.05)),
                'p10': float(age_series.quantile(0.10)),
                'p25': float(age_series.quantile(0.25)),
                'p50': float(age_series.quantile(0.50)),
                'p75': float(age_series.quantile(0.75)),
                'p90': float(age_series.quantile(0.90)),
                'p95': float(age_series.quantile(0.95))
            },
            'distribution': {
                'skewness': float(age_series.skew()),
                'kurtosis': float(age_series.kurtosis()),
                'is_normal': self._test_normality(age_series)
            },
            'value_counts': age_series.value_counts().to_dict(),
            'age_groups': self._analyze_age_distribution(age_series)
        }
        
        # Add gender-based statistics if available
        if 'gender' in data.columns:
            statistics['by_gender'] = {}
            for gender in data['gender'].unique():
                if pd.notnull(gender):
                    gender_ages = data[data['gender'] == gender]['age'].dropna()
                    if len(gender_ages) > 0:
                        statistics['by_gender'][gender] = {
                            'count': len(gender_ages),
                            'mean': float(gender_ages.mean()),
                            'median': float(gender_ages.median()),
                            'std': float(gender_ages.std())
                        }
        
        self.age_statistics = statistics
        return statistics
    
    def detect_age_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect age outliers using multiple methods.
        
        Args:
            data: DataFrame with age data
            
        Returns:
            Dictionary with outlier detection results
        """
        if 'age' not in data.columns:
            return {'error': 'Age column not found'}
        
        age_series = data['age'].dropna()
        
        if len(age_series) == 0:
            return {'error': 'No valid age data found'}
        
        outlier_results = {
            'methods_used': self.age_rules['outlier_detection']['methods'],
            'outliers_by_method': {},
            'consensus_outliers': [],
            'outlier_summary': {}
        }
        
        # Z-score method
        if 'zscore' in self.age_rules['outlier_detection']['methods']:
            z_scores = np.abs((age_series - age_series.mean()) / age_series.std())
            z_threshold = self.age_rules['outlier_detection']['zscore_threshold']
            z_outliers = age_series[z_scores > z_threshold]
            
            outlier_results['outliers_by_method']['zscore'] = {
                'threshold': z_threshold,
                'count': len(z_outliers),
                'indices': z_outliers.index.tolist(),
                'values': z_outliers.tolist(),
                'z_scores': z_scores[z_scores > z_threshold].tolist()
            }
        
        # IQR method
        if 'iqr' in self.age_rules['outlier_detection']['methods']:
            Q1 = age_series.quantile(0.25)
            Q3 = age_series.quantile(0.75)
            IQR = Q3 - Q1
            multiplier = self.age_rules['outlier_detection']['iqr_multiplier']
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            iqr_outliers = age_series[(age_series < lower_bound) | (age_series > upper_bound)]
            
            outlier_results['outliers_by_method']['iqr'] = {
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'count': len(iqr_outliers),
                'indices': iqr_outliers.index.tolist(),
                'values': iqr_outliers.tolist()
            }
        
        # Isolation Forest method
        if 'isolation_forest' in self.age_rules['outlier_detection']['methods']:
            try:
                from sklearn.ensemble import IsolationForest
                
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = iso_forest.fit_predict(age_series.values.reshape(-1, 1))
                
                iso_outliers = age_series[outlier_labels == -1]
                
                outlier_results['outliers_by_method']['isolation_forest'] = {
                    'count': len(iso_outliers),
                    'indices': iso_outliers.index.tolist(),
                    'values': iso_outliers.tolist()
                }
            except ImportError:
                logger.warning("Scikit-learn not available for Isolation Forest")
        
        # Find consensus outliers (detected by multiple methods)
        all_outlier_indices = set()
        method_counts = {}
        
        for method, results in outlier_results['outliers_by_method'].items():
            indices = set(results['indices'])
            all_outlier_indices.update(indices)
            
            for idx in indices:
                method_counts[idx] = method_counts.get(idx, 0) + 1
        
        # Consensus outliers are those detected by at least 2 methods
        consensus_outliers = [idx for idx, count in method_counts.items() if count >= 2]
        outlier_results['consensus_outliers'] = consensus_outliers
        
        # Summary
        outlier_results['outlier_summary'] = {
            'total_unique_outliers': len(all_outlier_indices),
            'consensus_outliers': len(consensus_outliers),
            'outlier_percentage': (len(all_outlier_indices) / len(age_series)) * 100
        }
        
        return outlier_results
    
    def engineer_age_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Engineer age-based features.
        
        Args:
            data: DataFrame with age data
            
        Returns:
            Tuple of (DataFrame with new features, feature engineering results)
        """
        if 'age' not in data.columns:
            return data, {'error': 'Age column not found'}
        
        enhanced_data = data.copy()
        feature_results = {
            'features_created': [],
            'feature_descriptions': {}
        }
        
        age_series = enhanced_data['age']
        
        # Age groups
        if self.age_rules['feature_engineering']['create_age_groups']:
            enhanced_data['age_group'] = self._create_age_groups(age_series)
            feature_results['features_created'].append('age_group')
            feature_results['feature_descriptions']['age_group'] = 'Categorical age groups (Young, Middle, Older)'
        
        # Age bins
        if self.age_rules['feature_engineering']['create_age_bins']:
            enhanced_data['age_bin'] = self._create_age_bins(age_series)
            feature_results['features_created'].append('age_bin')
            feature_results['feature_descriptions']['age_bin'] = 'Age ranges in 2-year bins'
        
        # Age percentiles
        if self.age_rules['feature_engineering']['create_age_percentiles']:
            enhanced_data['age_percentile'] = self._create_age_percentiles(age_series)
            feature_results['features_created'].append('age_percentile')
            feature_results['feature_descriptions']['age_percentile'] = 'Age percentile rank (0-100)'
        
        # Age z-score
        if self.age_rules['feature_engineering']['create_age_zscore']:
            enhanced_data['age_zscore'] = self._create_age_zscore(age_series)
            feature_results['features_created'].append('age_zscore')
            feature_results['feature_descriptions']['age_zscore'] = 'Standardized age (z-score)'
        
        # Age categories
        if self.age_rules['feature_engineering']['create_age_categories']:
            enhanced_data['age_category'] = self._create_age_categories(age_series)
            feature_results['features_created'].append('age_category')
            feature_results['feature_descriptions']['age_category'] = 'Detailed age categories'
        
        # Age relative to median
        enhanced_data['age_relative_to_median'] = age_series - age_series.median()
        feature_results['features_created'].append('age_relative_to_median')
        feature_results['feature_descriptions']['age_relative_to_median'] = 'Age difference from median age'
        
        # Age squared (for potential non-linear relationships)
        enhanced_data['age_squared'] = age_series ** 2
        feature_results['features_created'].append('age_squared')
        feature_results['feature_descriptions']['age_squared'] = 'Age squared for non-linear modeling'
        
        return enhanced_data, feature_results
    
    def analyze_age_groups(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze age group distributions and characteristics.
        
        Args:
            data: DataFrame with age data and features
            
        Returns:
            Dictionary with age group analysis
        """
        if 'age' not in data.columns:
            return {'error': 'Age column not found'}
        
        analysis = {
            'age_distribution': {},
            'group_statistics': {},
            'group_comparisons': {},
            'performance_by_age': {}
        }
        
        # Basic age distribution
        age_counts = data['age'].value_counts().sort_index()
        analysis['age_distribution'] = age_counts.to_dict()
        
        # Group statistics if age groups exist
        if 'age_group' in data.columns:
            for group in data['age_group'].unique():
                if pd.notnull(group):
                    group_data = data[data['age_group'] == group]
                    analysis['group_statistics'][group] = {
                        'count': len(group_data),
                        'percentage': (len(group_data) / len(data)) * 100,
                        'age_range': (group_data['age'].min(), group_data['age'].max()),
                        'mean_age': float(group_data['age'].mean())
                    }
        
        # Performance by age (if final_test is available)
        if 'final_test' in data.columns:
            age_performance = data.groupby('age')['final_test'].agg([
                'count', 'mean', 'std', 'median'
            ]).round(2)
            
            analysis['performance_by_age'] = age_performance.to_dict('index')
        
        return analysis
    
    def check_age_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check age consistency with other features.
        
        Args:
            data: DataFrame with age and other features
            
        Returns:
            Dictionary with consistency check results
        """
        consistency_results = {
            'checks_performed': [],
            'inconsistencies_found': [],
            'consistency_score': 100
        }
        
        # Check age vs study hours consistency
        if 'hours_per_week' in data.columns:
            # Younger students might have different study patterns
            young_students = data[data['age'] <= 15]
            if len(young_students) > 0:
                high_hours_young = young_students[young_students['hours_per_week'] > 50]
                if len(high_hours_young) > 0:
                    consistency_results['inconsistencies_found'].append({
                        'type': 'young_students_high_study_hours',
                        'count': len(high_hours_young),
                        'description': 'Students aged 15 or younger with >50 study hours per week'
                    })
            
            consistency_results['checks_performed'].append('age_vs_study_hours')
        
        # Check age vs sleep patterns
        if 'sleep_time' in data.columns and 'wake_time' in data.columns:
            # This is a placeholder for more complex sleep pattern analysis
            consistency_results['checks_performed'].append('age_vs_sleep_patterns')
        
        # Calculate consistency score
        total_inconsistencies = sum(inc['count'] for inc in consistency_results['inconsistencies_found'])
        if len(data) > 0:
            inconsistency_rate = (total_inconsistencies / len(data)) * 100
            consistency_results['consistency_score'] = max(0, 100 - inconsistency_rate * 10)
        
        return consistency_results
    
    def _test_normality(self, series: pd.Series) -> bool:
        """
        Test if age distribution is approximately normal.
        
        Args:
            series: Age data series
            
        Returns:
            Boolean indicating if distribution is approximately normal
        """
        try:
            from scipy import stats
            _, p_value = stats.normaltest(series)
            return p_value > 0.05  # Not significantly different from normal
        except ImportError:
            # Fallback: simple skewness and kurtosis check
            skewness = abs(series.skew())
            kurtosis = abs(series.kurtosis())
            return skewness < 1 and kurtosis < 3
    
    def _analyze_age_distribution(self, series: pd.Series) -> Dict[str, Any]:
        """
        Analyze age distribution patterns.
        
        Args:
            series: Age data series
            
        Returns:
            Dictionary with distribution analysis
        """
        distribution_analysis = {
            'most_common_age': int(series.mode().iloc[0]) if not series.mode().empty else None,
            'age_spread': float(series.max() - series.min()),
            'concentration': {},
            'gaps': []
        }
        
        # Analyze concentration in different ranges
        total_count = len(series)
        
        ranges = {
            '10-15': (10, 15),
            '16-18': (16, 18),
            '19-22': (19, 22),
            '23+': (23, float('inf'))
        }
        
        for range_name, (min_age, max_age) in ranges.items():
            if max_age == float('inf'):
                count = ((series >= min_age)).sum()
            else:
                count = ((series >= min_age) & (series <= max_age)).sum()
            
            distribution_analysis['concentration'][range_name] = {
                'count': int(count),
                'percentage': float((count / total_count) * 100)
            }
        
        # Find gaps in age distribution
        age_range = range(int(series.min()), int(series.max()) + 1)
        present_ages = set(series.astype(int))
        
        for age in age_range:
            if age not in present_ages:
                distribution_analysis['gaps'].append(age)
        
        return distribution_analysis
    
    def _create_age_groups(self, age_series: pd.Series) -> pd.Series:
        """
        Create age groups.
        
        Args:
            age_series: Age data series
            
        Returns:
            Series with age group labels
        """
        def categorize_age(age):
            if pd.isnull(age):
                return 'Unknown'
            elif age <= 16:
                return 'Young'
            elif age <= 20:
                return 'Middle'
            else:
                return 'Older'
        
        return age_series.apply(categorize_age)
    
    def _create_age_bins(self, age_series: pd.Series) -> pd.Series:
        """
        Create age bins.
        
        Args:
            age_series: Age data series
            
        Returns:
            Series with age bin labels
        """
        bins = [10, 13, 15, 17, 19, 21, 25]
        labels = ['10-12', '13-14', '15-16', '17-18', '19-20', '21+']
        
        return pd.cut(age_series, bins=bins, labels=labels, include_lowest=True)
    
    def _create_age_percentiles(self, age_series: pd.Series) -> pd.Series:
        """
        Create age percentiles.
        
        Args:
            age_series: Age data series
            
        Returns:
            Series with age percentile ranks
        """
        return age_series.rank(pct=True) * 100
    
    def _create_age_zscore(self, age_series: pd.Series) -> pd.Series:
        """
        Create age z-scores.
        
        Args:
            age_series: Age data series
            
        Returns:
            Series with age z-scores
        """
        return (age_series - age_series.mean()) / age_series.std()
    
    def _create_age_categories(self, age_series: pd.Series) -> pd.Series:
        """
        Create detailed age categories.
        
        Args:
            age_series: Age data series
            
        Returns:
            Series with detailed age category labels
        """
        def categorize_detailed(age):
            if pd.isnull(age):
                return 'Unknown'
            elif age <= 14:
                return 'Early_Teen'
            elif age <= 16:
                return 'Mid_Teen'
            elif age <= 18:
                return 'Late_Teen'
            elif age <= 20:
                return 'Young_Adult'
            else:
                return 'Adult'
        
        return age_series.apply(categorize_detailed)
    
    def save_processing_results(self, output_dir: str) -> None:
        """
        Save age processing results to files.
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save processing results (excluding processed_data)
        results_to_save = {k: v for k, v in self.processing_results.items() if k != 'processed_data'}
        
        results_file = output_path / 'enhanced_age_processing_results.json'
        with open(results_file, 'w') as f:
            json.dump(results_to_save, f, indent=2, default=str)
        
        # Save processed data
        if 'processed_data' in self.processing_results:
            data_file = output_path / 'age_processed_data.csv'
            self.processing_results['processed_data'].to_csv(data_file, index=False)
        
        logger.info(f"Age processing results saved to {output_path}")
    
    def generate_age_report(self) -> str:
        """
        Generate a comprehensive age processing report.
        
        Returns:
            Age processing report string
        """
        if not self.processing_results:
            return "No processing results available. Run process_age_comprehensive() first."
        
        report = []
        report.append("=== Enhanced Age Processing Report ===")
        report.append("")
        
        # Validation summary
        validation = self.processing_results.get('validation', {})
        if validation:
            report.append("## Age Data Validation")
            report.append(f"- Total records: {validation.get('total_records', 'N/A')}")
            report.append(f"- Missing ages: {validation.get('missing_ages', 'N/A')} ({validation.get('missing_percentage', 0):.1f}%)")
            report.append(f"- Negative ages: {validation.get('negative_ages', 'N/A')}")
            report.append(f"- Decimal ages: {validation.get('decimal_ages', 'N/A')}")
            
            range_violations = validation.get('range_violations', {})
            if range_violations:
                below_min = range_violations.get('below_minimum', {}).get('count', 0)
                above_max = range_violations.get('above_maximum', {}).get('count', 0)
                report.append(f"- Ages below minimum: {below_min}")
                report.append(f"- Ages above maximum: {above_max}")
            
            report.append("")
        
        # Correction summary
        correction = self.processing_results.get('correction', {})
        if correction:
            report.append("## Age Data Corrections")
            corrections = correction.get('corrections_made', [])
            if corrections:
                for correction_msg in corrections:
                    report.append(f"- {correction_msg}")
            else:
                report.append("- No corrections needed")
            
            flagged = len(correction.get('flagged_records', []))
            if flagged > 0:
                report.append(f"- Flagged records: {flagged}")
            
            report.append("")
        
        # Statistics summary
        statistics = self.processing_results.get('statistics', {})
        if statistics:
            basic_stats = statistics.get('basic_stats', {})
            if basic_stats:
                report.append("## Age Statistics")
                report.append(f"- Count: {basic_stats.get('count', 'N/A')}")
                report.append(f"- Mean: {basic_stats.get('mean', 0):.2f}")
                report.append(f"- Median: {basic_stats.get('median', 0):.2f}")
                report.append(f"- Standard deviation: {basic_stats.get('std', 0):.2f}")
                report.append(f"- Range: {basic_stats.get('min', 0):.1f} - {basic_stats.get('max', 0):.1f}")
                report.append("")
        
        # Outlier summary
        outliers = self.processing_results.get('outliers', {})
        if outliers:
            report.append("## Age Outliers")
            outlier_summary = outliers.get('outlier_summary', {})
            report.append(f"- Total unique outliers: {outlier_summary.get('total_unique_outliers', 'N/A')}")
            report.append(f"- Consensus outliers: {outlier_summary.get('consensus_outliers', 'N/A')}")
            report.append(f"- Outlier percentage: {outlier_summary.get('outlier_percentage', 0):.2f}%")
            report.append("")
        
        # Feature engineering summary
        features = self.processing_results.get('features', {})
        if features:
            report.append("## Age Feature Engineering")
            created_features = features.get('features_created', [])
            if created_features:
                report.append(f"- Features created: {len(created_features)}")
                for feature in created_features:
                    description = features.get('feature_descriptions', {}).get(feature, 'No description')
                    report.append(f"  - {feature}: {description}")
            report.append("")
        
        # Consistency summary
        consistency = self.processing_results.get('consistency', {})
        if consistency:
            report.append("## Age Consistency")
            score = consistency.get('consistency_score', 0)
            report.append(f"- Consistency score: {score:.1f}/100")
            
            inconsistencies = consistency.get('inconsistencies_found', [])
            if inconsistencies:
                report.append("- Inconsistencies found:")
                for inc in inconsistencies:
                    report.append(f"  - {inc.get('description', 'Unknown')}: {inc.get('count', 0)} cases")
            else:
                report.append("- No significant inconsistencies found")
            
            report.append("")
        
        return "\n".join(report)


def main():
    """
    Main function for testing the Enhanced Age Processor.
    """
    # Example usage
    db_path = "data/raw/score.db"
    
    processor = EnhancedAgeProcessor(db_path=db_path)
    
    # Run comprehensive age processing
    results = processor.process_age_comprehensive()
    print(f"Age processing completed. Features created: {len(results['features']['features_created'])}")
    
    # Generate report
    report = processor.generate_age_report()
    print(report)
    
    # Save results
    processor.save_processing_results("data/processed")
    print("Results saved")


if __name__ == "__main__":
    main()