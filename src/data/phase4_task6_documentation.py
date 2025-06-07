#!/usr/bin/env python3
"""
Phase 4 Task 6.1: Documentation and Validation (High Priority)

This module implements task 4.6.1 from TASKS.md:
- Generate comprehensive feature engineering documentation
- Create validation reports for all Phase 4 tasks
- Document feature transformations and their rationale
- Create model-ready dataset documentation
- Generate feature dictionary and metadata

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from datetime import datetime
import warnings
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Phase4Documentation:
    """
    Generates comprehensive documentation and validation for Phase 4 feature engineering.
    """
    
    def __init__(self, original_df: pd.DataFrame, processed_df: pd.DataFrame, 
                 target_column: str = 'final_test'):
        """
        Initialize with original and processed datasets.
        
        Args:
            original_df: Original dataset before Phase 4 processing
            processed_df: Final processed dataset after Phase 4
            target_column: Name of the target variable column
        """
        self.original_df = original_df.copy()
        self.processed_df = processed_df.copy()
        self.target_column = target_column
        self.documentation = {}
        self.validation_results = {}
        self.feature_metadata = {}
        
    def generate_feature_dictionary(self) -> Dict[str, Any]:
        """
        Generate comprehensive feature dictionary.
        
        Returns:
            Dictionary containing feature metadata
        """
        logger.info("Generating feature dictionary")
        
        feature_dict = {
            'creation_timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'original_shape': self.original_df.shape,
                'processed_shape': self.processed_df.shape,
                'target_column': self.target_column
            },
            'features': {}
        }
        
        # Analyze each feature in processed dataset
        for column in self.processed_df.columns:
            feature_info = self._analyze_feature(column)
            feature_dict['features'][column] = feature_info
            
        # Add feature categories
        feature_dict['feature_categories'] = self._categorize_features()
        
        # Add transformation summary
        feature_dict['transformations_applied'] = self._document_transformations()
        
        self.feature_metadata = feature_dict
        
        logger.info(f"Feature dictionary generated for {len(feature_dict['features'])} features")
        return feature_dict
        
    def _analyze_feature(self, column: str) -> Dict[str, Any]:
        """
        Analyze individual feature characteristics.
        
        Args:
            column: Feature column name
            
        Returns:
            Dictionary containing feature analysis
        """
        feature_data = self.processed_df[column]
        
        analysis = {
            'name': column,
            'data_type': str(feature_data.dtype),
            'description': self._get_feature_description(column),
            'statistics': {},
            'quality_metrics': {},
            'transformation_info': {},
            'relationship_with_target': {}
        }
        
        # Basic statistics
        analysis['statistics'] = {
            'count': int(feature_data.count()),
            'missing_count': int(feature_data.isnull().sum()),
            'missing_percentage': float(feature_data.isnull().sum() / len(feature_data) * 100),
            'unique_count': int(feature_data.nunique())
        }
        
        # Type-specific statistics
        if pd.api.types.is_numeric_dtype(feature_data):
            numeric_stats = {
                'mean': float(feature_data.mean()) if not feature_data.isnull().all() else None,
                'std': float(feature_data.std()) if not feature_data.isnull().all() else None,
                'min': float(feature_data.min()) if not feature_data.isnull().all() else None,
                'max': float(feature_data.max()) if not feature_data.isnull().all() else None,
                'median': float(feature_data.median()) if not feature_data.isnull().all() else None,
                'q25': float(feature_data.quantile(0.25)) if not feature_data.isnull().all() else None,
                'q75': float(feature_data.quantile(0.75)) if not feature_data.isnull().all() else None
            }
            
            # Distribution characteristics
            if not feature_data.isnull().all() and len(feature_data.dropna()) > 3:
                try:
                    numeric_stats['skewness'] = float(stats.skew(feature_data.dropna()))
                    numeric_stats['kurtosis'] = float(stats.kurtosis(feature_data.dropna()))
                except:
                    numeric_stats['skewness'] = None
                    numeric_stats['kurtosis'] = None
                    
            analysis['statistics'].update(numeric_stats)
            
        else:
            # Categorical statistics
            categorical_stats = {
                'most_frequent': feature_data.mode().iloc[0] if len(feature_data.mode()) > 0 else None,
                'most_frequent_count': int(feature_data.value_counts().iloc[0]) if len(feature_data.value_counts()) > 0 else 0,
                'categories': feature_data.unique().tolist()[:10]  # Limit to first 10
            }
            analysis['statistics'].update(categorical_stats)
            
        # Quality metrics
        analysis['quality_metrics'] = {
            'completeness_score': float(1 - (feature_data.isnull().sum() / len(feature_data))),
            'uniqueness_score': float(feature_data.nunique() / len(feature_data)) if len(feature_data) > 0 else 0,
            'consistency_score': self._calculate_consistency_score(feature_data)
        }
        
        # Relationship with target
        if column != self.target_column and self.target_column in self.processed_df.columns:
            analysis['relationship_with_target'] = self._analyze_target_relationship(column)
            
        # Transformation info
        analysis['transformation_info'] = self._get_transformation_info(column)
        
        return analysis
        
    def _get_feature_description(self, column: str) -> str:
        """
        Get human-readable description for feature.
        
        Args:
            column: Feature column name
            
        Returns:
            Feature description
        """
        descriptions = {
            'final_test': 'Final test score (target variable)',
            'previous_score': 'Previous academic performance score',
            'study_hours': 'Weekly study hours',
            'attendance': 'Class attendance percentage',
            'age': 'Student age',
            'parental_education_level': 'Highest education level of parents',
            'distance_from_home': 'Distance category from home to school',
            'internet_access': 'Internet access availability',
            'family_income': 'Family income level',
            'school_type': 'Type of school (public/private)',
            'peer_influence': 'Peer influence level',
            'physical_activity': 'Physical activity level',
            'learning_disabilities': 'Presence of learning disabilities',
            'parental_support': 'Level of parental support',
            'extracurricular_activities': 'Participation in extracurricular activities',
            'motivation_level': 'Student motivation level',
            'teacher_quality': 'Quality of teaching',
            'school_type_encoded': 'Encoded school type',
            'tutoring_sessions': 'Number of tutoring sessions',
            'gender_encoded': 'Encoded gender',
            'study_efficiency_score': 'Derived efficiency metric combining study hours and performance',
            'academic_support_index': 'Composite index of academic support factors',
            'study_attendance_interaction': 'Interaction between study hours and attendance'
        }
        
        # Handle transformed features
        if 'log_' in column:
            base_feature = column.replace('log_', '')
            return f"Log-transformed {descriptions.get(base_feature, base_feature)}"
        elif 'boxcox_' in column:
            base_feature = column.replace('boxcox_', '')
            return f"Box-Cox transformed {descriptions.get(base_feature, base_feature)}"
        elif 'scaled_' in column:
            base_feature = column.replace('scaled_', '')
            return f"Scaled {descriptions.get(base_feature, base_feature)}"
        elif 'encoded_' in column:
            base_feature = column.replace('encoded_', '')
            return f"Encoded {descriptions.get(base_feature, base_feature)}"
            
        return descriptions.get(column, f"Feature: {column}")
        
    def _calculate_consistency_score(self, feature_data: pd.Series) -> float:
        """
        Calculate consistency score for a feature.
        
        Args:
            feature_data: Feature data series
            
        Returns:
            Consistency score (0-1)
        """
        if feature_data.isnull().all():
            return 0.0
            
        # For numeric data, check for outliers
        if pd.api.types.is_numeric_dtype(feature_data):
            clean_data = feature_data.dropna()
            if len(clean_data) < 4:
                return 1.0
                
            q1 = clean_data.quantile(0.25)
            q3 = clean_data.quantile(0.75)
            iqr = q3 - q1
            
            if iqr == 0:
                return 1.0
                
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = ((clean_data < lower_bound) | (clean_data > upper_bound)).sum()
            consistency_score = 1 - (outliers / len(clean_data))
            
        else:
            # For categorical data, check distribution balance
            value_counts = feature_data.value_counts()
            if len(value_counts) == 0:
                return 0.0
                
            # Calculate entropy-based consistency
            proportions = value_counts / value_counts.sum()
            entropy = -np.sum(proportions * np.log2(proportions + 1e-10))
            max_entropy = np.log2(len(value_counts))
            
            consistency_score = entropy / max_entropy if max_entropy > 0 else 1.0
            
        return float(max(0, min(1, consistency_score)))
        
    def _analyze_target_relationship(self, column: str) -> Dict[str, Any]:
        """
        Analyze relationship between feature and target variable.
        
        Args:
            column: Feature column name
            
        Returns:
            Dictionary containing relationship analysis
        """
        feature_data = self.processed_df[column]
        target_data = self.processed_df[self.target_column]
        
        # Remove rows where either feature or target is missing
        valid_mask = ~(feature_data.isnull() | target_data.isnull())
        clean_feature = feature_data[valid_mask]
        clean_target = target_data[valid_mask]
        
        relationship = {
            'correlation': None,
            'mutual_information': None,
            'statistical_significance': None,
            'relationship_strength': 'unknown'
        }
        
        if len(clean_feature) < 10:
            return relationship
            
        try:
            # Correlation (for numeric features)
            if pd.api.types.is_numeric_dtype(clean_feature):
                correlation = clean_feature.corr(clean_target)
                relationship['correlation'] = float(correlation) if not np.isnan(correlation) else None
                
                # Statistical significance test
                if len(clean_feature) > 3:
                    try:
                        _, p_value = stats.pearsonr(clean_feature, clean_target)
                        relationship['statistical_significance'] = float(p_value)
                    except:
                        pass
                        
            # Mutual information
            try:
                # Encode categorical features for mutual information
                if pd.api.types.is_numeric_dtype(clean_feature):
                    mi_feature = clean_feature.values.reshape(-1, 1)
                else:
                    le = LabelEncoder()
                    mi_feature = le.fit_transform(clean_feature.astype(str)).reshape(-1, 1)
                    
                mi_score = mutual_info_regression(mi_feature, clean_target)[0]
                relationship['mutual_information'] = float(mi_score)
                
            except Exception as e:
                logger.warning(f"Mutual information calculation failed for {column}: {str(e)}")
                
            # Determine relationship strength
            if relationship['correlation'] is not None:
                abs_corr = abs(relationship['correlation'])
                if abs_corr > 0.7:
                    relationship['relationship_strength'] = 'strong'
                elif abs_corr > 0.3:
                    relationship['relationship_strength'] = 'moderate'
                elif abs_corr > 0.1:
                    relationship['relationship_strength'] = 'weak'
                else:
                    relationship['relationship_strength'] = 'very_weak'
                    
        except Exception as e:
            logger.warning(f"Relationship analysis failed for {column}: {str(e)}")
            
        return relationship
        
    def _get_transformation_info(self, column: str) -> Dict[str, Any]:
        """
        Get transformation information for a feature.
        
        Args:
            column: Feature column name
            
        Returns:
            Dictionary containing transformation details
        """
        transformation_info = {
            'original_feature': None,
            'transformation_type': 'none',
            'transformation_parameters': {},
            'rationale': ''
        }
        
        # Detect transformation type from column name
        if 'log_' in column:
            transformation_info['original_feature'] = column.replace('log_', '')
            transformation_info['transformation_type'] = 'logarithmic'
            transformation_info['rationale'] = 'Applied to reduce right skewness and normalize distribution'
            
        elif 'boxcox_' in column:
            transformation_info['original_feature'] = column.replace('boxcox_', '')
            transformation_info['transformation_type'] = 'box_cox'
            transformation_info['rationale'] = 'Applied to achieve normality and reduce skewness'
            
        elif 'scaled_' in column:
            transformation_info['original_feature'] = column.replace('scaled_', '')
            transformation_info['transformation_type'] = 'scaling'
            transformation_info['rationale'] = 'Applied for feature normalization and model compatibility'
            
        elif 'encoded_' in column:
            transformation_info['original_feature'] = column.replace('encoded_', '')
            transformation_info['transformation_type'] = 'encoding'
            transformation_info['rationale'] = 'Applied to convert categorical data to numerical format'
            
        elif 'interaction' in column.lower():
            transformation_info['transformation_type'] = 'interaction'
            transformation_info['rationale'] = 'Created to capture combined effect of multiple features'
            
        elif any(keyword in column.lower() for keyword in ['efficiency', 'index', 'score']):
            if column not in ['final_test', 'previous_score']:  # Exclude original scores
                transformation_info['transformation_type'] = 'derived_feature'
                transformation_info['rationale'] = 'Engineered feature combining multiple data points'
                
        return transformation_info
        
    def _categorize_features(self) -> Dict[str, List[str]]:
        """
        Categorize features by type and purpose.
        
        Returns:
            Dictionary containing feature categories
        """
        categories = {
            'target': [],
            'original_numerical': [],
            'original_categorical': [],
            'transformed_numerical': [],
            'encoded_categorical': [],
            'derived_features': [],
            'interaction_features': [],
            'scaled_features': []
        }
        
        for column in self.processed_df.columns:
            if column == self.target_column:
                categories['target'].append(column)
            elif any(prefix in column for prefix in ['log_', 'boxcox_']):
                categories['transformed_numerical'].append(column)
            elif 'encoded_' in column or column.endswith('_encoded'):
                categories['encoded_categorical'].append(column)
            elif 'scaled_' in column:
                categories['scaled_features'].append(column)
            elif 'interaction' in column.lower():
                categories['interaction_features'].append(column)
            elif any(keyword in column.lower() for keyword in ['efficiency', 'index', 'score']) and column != self.target_column:
                categories['derived_features'].append(column)
            elif pd.api.types.is_numeric_dtype(self.processed_df[column]):
                categories['original_numerical'].append(column)
            else:
                categories['original_categorical'].append(column)
                
        return categories
        
    def _document_transformations(self) -> Dict[str, Any]:
        """
        Document all transformations applied during Phase 4.
        
        Returns:
            Dictionary containing transformation documentation
        """
        transformations = {
            'summary': {
                'total_features_original': self.original_df.shape[1],
                'total_features_processed': self.processed_df.shape[1],
                'features_added': self.processed_df.shape[1] - self.original_df.shape[1],
                'transformation_types_applied': []
            },
            'by_type': {
                'logarithmic': [],
                'box_cox': [],
                'scaling': [],
                'encoding': [],
                'derived_features': [],
                'interaction_features': [],
                'feature_selection': []
            },
            'rationale': {}
        }
        
        # Identify transformations
        for column in self.processed_df.columns:
            if 'log_' in column:
                transformations['by_type']['logarithmic'].append(column)
                transformations['rationale'][column] = "Applied to reduce right skewness"
            elif 'boxcox_' in column:
                transformations['by_type']['box_cox'].append(column)
                transformations['rationale'][column] = "Applied to achieve normality"
            elif 'scaled_' in column:
                transformations['by_type']['scaling'].append(column)
                transformations['rationale'][column] = "Applied for feature normalization"
            elif 'encoded_' in column or column.endswith('_encoded'):
                transformations['by_type']['encoding'].append(column)
                transformations['rationale'][column] = "Applied to convert categorical to numerical"
            elif 'interaction' in column.lower():
                transformations['by_type']['interaction_features'].append(column)
                transformations['rationale'][column] = "Created to capture feature interactions"
            elif any(keyword in column.lower() for keyword in ['efficiency', 'index', 'score']) and column != self.target_column:
                transformations['by_type']['derived_features'].append(column)
                transformations['rationale'][column] = "Engineered from multiple features"
                
        # Update summary
        transformation_types = [t_type for t_type, features in transformations['by_type'].items() if features]
        transformations['summary']['transformation_types_applied'] = transformation_types
        
        return transformations
        
    def generate_validation_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive validation report for Phase 4.
        
        Returns:
            Dictionary containing validation results
        """
        logger.info("Generating validation report")
        
        validation_report = {
            'validation_timestamp': datetime.now().isoformat(),
            'dataset_comparison': self._compare_datasets(),
            'feature_engineering_validation': self._validate_feature_engineering(),
            'data_quality_validation': self._validate_data_quality(),
            'model_readiness_validation': self._validate_model_readiness(),
            'overall_validation_status': 'pending'
        }
        
        # Determine overall validation status
        validation_checks = [
            validation_report['dataset_comparison']['validation_passed'],
            validation_report['feature_engineering_validation']['validation_passed'],
            validation_report['data_quality_validation']['validation_passed'],
            validation_report['model_readiness_validation']['validation_passed']
        ]
        
        if all(validation_checks):
            validation_report['overall_validation_status'] = 'passed'
        elif any(validation_checks):
            validation_report['overall_validation_status'] = 'partial'
        else:
            validation_report['overall_validation_status'] = 'failed'
            
        self.validation_results = validation_report
        
        logger.info(f"Validation report generated - Status: {validation_report['overall_validation_status']}")
        return validation_report
        
    def _compare_datasets(self) -> Dict[str, Any]:
        """
        Compare original and processed datasets.
        
        Returns:
            Dictionary containing dataset comparison
        """
        comparison = {
            'shape_comparison': {
                'original_shape': self.original_df.shape,
                'processed_shape': self.processed_df.shape,
                'rows_preserved': self.processed_df.shape[0] == self.original_df.shape[0],
                'features_added': self.processed_df.shape[1] - self.original_df.shape[1]
            },
            'data_preservation': {},
            'validation_passed': True,
            'issues': []
        }
        
        # Check data preservation for common columns
        common_columns = set(self.original_df.columns) & set(self.processed_df.columns)
        
        for column in common_columns:
            orig_data = self.original_df[column]
            proc_data = self.processed_df[column]
            
            # Check if data is preserved (allowing for minor floating point differences)
            if pd.api.types.is_numeric_dtype(orig_data) and pd.api.types.is_numeric_dtype(proc_data):
                data_preserved = np.allclose(orig_data.fillna(0), proc_data.fillna(0), rtol=1e-10, atol=1e-10)
            else:
                data_preserved = orig_data.equals(proc_data)
                
            comparison['data_preservation'][column] = {
                'preserved': data_preserved,
                'original_nulls': int(orig_data.isnull().sum()),
                'processed_nulls': int(proc_data.isnull().sum())
            }
            
            if not data_preserved:
                comparison['issues'].append(f"Data not preserved in column: {column}")
                
        # Check for critical issues
        if not comparison['shape_comparison']['rows_preserved']:
            comparison['issues'].append("Row count not preserved")
            comparison['validation_passed'] = False
            
        if comparison['shape_comparison']['features_added'] < 0:
            comparison['issues'].append("Features were removed unexpectedly")
            comparison['validation_passed'] = False
            
        return comparison
        
    def _validate_feature_engineering(self) -> Dict[str, Any]:
        """
        Validate feature engineering results.
        
        Returns:
            Dictionary containing feature engineering validation
        """
        validation = {
            'feature_count_validation': {},
            'transformation_validation': {},
            'derived_feature_validation': {},
            'validation_passed': True,
            'issues': []
        }
        
        # Feature count validation
        expected_min_features = self.original_df.shape[1]  # At least original features
        actual_features = self.processed_df.shape[1]
        
        validation['feature_count_validation'] = {
            'expected_minimum': expected_min_features,
            'actual_count': actual_features,
            'meets_minimum': actual_features >= expected_min_features
        }
        
        if not validation['feature_count_validation']['meets_minimum']:
            validation['issues'].append("Feature count below expected minimum")
            validation['validation_passed'] = False
            
        # Transformation validation
        transformed_features = [col for col in self.processed_df.columns 
                              if any(prefix in col for prefix in ['log_', 'boxcox_', 'scaled_', 'encoded_'])]
        
        validation['transformation_validation'] = {
            'transformed_features_count': len(transformed_features),
            'transformations_applied': len(transformed_features) > 0,
            'transformed_features': transformed_features
        }
        
        # Derived feature validation
        derived_features = [col for col in self.processed_df.columns 
                          if any(keyword in col.lower() for keyword in ['efficiency', 'index', 'interaction'])]
        
        validation['derived_feature_validation'] = {
            'derived_features_count': len(derived_features),
            'derived_features_created': len(derived_features) > 0,
            'derived_features': derived_features
        }
        
        # Check for expected derived features
        expected_derived = ['study_efficiency_score', 'academic_support_index', 'study_attendance_interaction']
        missing_derived = [feat for feat in expected_derived if feat not in self.processed_df.columns]
        
        if missing_derived:
            validation['issues'].append(f"Missing expected derived features: {missing_derived}")
            
        return validation
        
    def _validate_data_quality(self) -> Dict[str, Any]:
        """
        Validate data quality of processed dataset.
        
        Returns:
            Dictionary containing data quality validation
        """
        validation = {
            'missing_data_validation': {},
            'data_type_validation': {},
            'range_validation': {},
            'validation_passed': True,
            'issues': []
        }
        
        # Missing data validation
        missing_percentages = (self.processed_df.isnull().sum() / len(self.processed_df) * 100)
        high_missing_features = missing_percentages[missing_percentages > 20].index.tolist()
        
        validation['missing_data_validation'] = {
            'max_missing_percentage': float(missing_percentages.max()),
            'features_with_high_missing': high_missing_features,
            'acceptable_missing_levels': len(high_missing_features) == 0
        }
        
        if high_missing_features:
            validation['issues'].append(f"High missing data in features: {high_missing_features}")
            validation['validation_passed'] = False
            
        # Data type validation
        numeric_features = self.processed_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = self.processed_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        validation['data_type_validation'] = {
            'numeric_features_count': len(numeric_features),
            'categorical_features_count': len(categorical_features),
            'mixed_types_detected': False
        }
        
        # Range validation for key features
        range_issues = []
        if 'final_test' in self.processed_df.columns:
            final_test_data = self.processed_df['final_test'].dropna()
            if len(final_test_data) > 0 and (final_test_data.min() < 0 or final_test_data.max() > 100):
                range_issues.append('final_test values outside expected range [0, 100]')
                
        validation['range_validation'] = {
            'range_issues': range_issues,
            'ranges_valid': len(range_issues) == 0
        }
        
        if range_issues:
            validation['issues'].extend(range_issues)
            validation['validation_passed'] = False
            
        return validation
        
    def _validate_model_readiness(self) -> Dict[str, Any]:
        """
        Validate model readiness of processed dataset.
        
        Returns:
            Dictionary containing model readiness validation
        """
        validation = {
            'feature_scaling_validation': {},
            'categorical_encoding_validation': {},
            'target_variable_validation': {},
            'validation_passed': True,
            'issues': []
        }
        
        # Feature scaling validation
        numeric_features = self.processed_df.select_dtypes(include=[np.number]).columns
        numeric_features = [col for col in numeric_features if col != self.target_column]
        
        if numeric_features:
            # Check if features are on similar scales
            feature_ranges = {}
            for feature in numeric_features:
                data = self.processed_df[feature].dropna()
                if len(data) > 0:
                    feature_ranges[feature] = data.max() - data.min()
                    
            if feature_ranges:
                max_range = max(feature_ranges.values())
                min_range = min(feature_ranges.values())
                scale_ratio = max_range / min_range if min_range > 0 else float('inf')
                
                validation['feature_scaling_validation'] = {
                    'scale_ratio': float(scale_ratio),
                    'features_similarly_scaled': scale_ratio < 100,  # Reasonable threshold
                    'feature_ranges': {k: float(v) for k, v in feature_ranges.items()}
                }
                
                if scale_ratio >= 100:
                    validation['issues'].append("Features have very different scales - consider additional scaling")
                    
        # Categorical encoding validation
        categorical_features = self.processed_df.select_dtypes(include=['object', 'category']).columns
        
        validation['categorical_encoding_validation'] = {
            'categorical_features_remaining': len(categorical_features),
            'all_features_encoded': len(categorical_features) == 0
        }
        
        if len(categorical_features) > 0:
            validation['issues'].append(f"Categorical features not encoded: {categorical_features.tolist()}")
            validation['validation_passed'] = False
            
        # Target variable validation
        if self.target_column in self.processed_df.columns:
            target_data = self.processed_df[self.target_column].dropna()
            
            validation['target_variable_validation'] = {
                'target_present': True,
                'target_numeric': pd.api.types.is_numeric_dtype(target_data),
                'target_missing_count': int(self.processed_df[self.target_column].isnull().sum()),
                'target_range': (float(target_data.min()), float(target_data.max())) if len(target_data) > 0 else None
            }
            
            if not validation['target_variable_validation']['target_numeric']:
                validation['issues'].append("Target variable is not numeric")
                validation['validation_passed'] = False
                
        else:
            validation['target_variable_validation'] = {'target_present': False}
            validation['issues'].append("Target variable not found")
            validation['validation_passed'] = False
            
        return validation
        
    def save_documentation(self, output_dir: str = "data/featured") -> None:
        """
        Save all documentation files.
        
        Args:
            output_dir: Directory to save documentation
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save feature dictionary
        if not self.feature_metadata:
            self.generate_feature_dictionary()
            
        feature_dict_path = output_path / "feature_dictionary.json"
        with open(feature_dict_path, 'w') as f:
            json.dump(self.feature_metadata, f, indent=2, default=str)
            
        # Save validation report
        if not self.validation_results:
            self.generate_validation_report()
            
        validation_path = output_path / "phase4_validation_report.json"
        with open(validation_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
            
        # Generate and save markdown documentation
        markdown_doc = self._generate_markdown_documentation()
        markdown_path = output_path / "phase4_documentation.md"
        with open(markdown_path, 'w') as f:
            f.write(markdown_doc)
            
        logger.info(f"Documentation saved to {output_path}")
        
    def _generate_markdown_documentation(self) -> str:
        """
        Generate markdown documentation.
        
        Returns:
            Markdown documentation string
        """
        if not self.feature_metadata:
            self.generate_feature_dictionary()
        if not self.validation_results:
            self.generate_validation_report()
            
        doc = f"""# Phase 4 Feature Engineering Documentation

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

This document provides comprehensive documentation for Phase 4 Feature Engineering tasks completed on the student performance dataset.

### Dataset Summary
- **Original Dataset Shape:** {self.feature_metadata['dataset_info']['original_shape']}
- **Processed Dataset Shape:** {self.feature_metadata['dataset_info']['processed_shape']}
- **Target Variable:** {self.feature_metadata['dataset_info']['target_column']}
- **Features Added:** {self.feature_metadata['dataset_info']['processed_shape'][1] - self.feature_metadata['dataset_info']['original_shape'][1]}

## Feature Categories

"""
        
        # Add feature categories
        categories = self.feature_metadata['feature_categories']
        for category, features in categories.items():
            if features:
                doc += f"### {category.replace('_', ' ').title()}\n"
                for feature in features:
                    doc += f"- `{feature}`\n"
                doc += "\n"
                
        # Add transformations summary
        doc += "## Transformations Applied\n\n"
        transformations = self.feature_metadata['transformations_applied']
        
        for transform_type, features in transformations['by_type'].items():
            if features:
                doc += f"### {transform_type.replace('_', ' ').title()}\n"
                for feature in features:
                    rationale = transformations['rationale'].get(feature, 'No rationale provided')
                    doc += f"- `{feature}`: {rationale}\n"
                doc += "\n"
                
        # Add validation summary
        doc += "## Validation Summary\n\n"
        doc += f"**Overall Status:** {self.validation_results['overall_validation_status'].upper()}\n\n"
        
        validation_sections = [
            ('dataset_comparison', 'Dataset Comparison'),
            ('feature_engineering_validation', 'Feature Engineering'),
            ('data_quality_validation', 'Data Quality'),
            ('model_readiness_validation', 'Model Readiness')
        ]
        
        for section_key, section_name in validation_sections:
            section_data = self.validation_results[section_key]
            status = "✅ PASSED" if section_data['validation_passed'] else "❌ FAILED"
            doc += f"- **{section_name}:** {status}\n"
            
            if section_data.get('issues'):
                doc += f"  - Issues: {', '.join(section_data['issues'])}\n"
                
        doc += "\n## Feature Details\n\n"
        
        # Add top features by target correlation
        features_with_correlation = []
        for feature_name, feature_info in self.feature_metadata['features'].items():
            if feature_name != self.target_column:
                correlation = feature_info.get('relationship_with_target', {}).get('correlation')
                if correlation is not None:
                    features_with_correlation.append((feature_name, abs(correlation), correlation))
                    
        features_with_correlation.sort(key=lambda x: x[1], reverse=True)
        
        if features_with_correlation:
            doc += "### Top Features by Target Correlation\n\n"
            for feature_name, abs_corr, corr in features_with_correlation[:10]:
                doc += f"- `{feature_name}`: {corr:.3f}\n"
            doc += "\n"
            
        return doc


def main():
    """
    Main function to run Phase 4 Task 6.1: Documentation and Validation.
    """
    try:
        # Load original processed data
        original_path = "data/processed/final_processed.csv"
        if not Path(original_path).exists():
            raise FileNotFoundError(f"Original processed file not found: {original_path}")
            
        original_df = pd.read_csv(original_path, index_col=0)
        logger.info(f"Loaded original dataset with shape: {original_df.shape}")
        
        # Load final processed data (after Phase 4)
        processed_path = "data/featured/selected_features_dataset.csv"
        if not Path(processed_path).exists():
            # Try alternative path
            processed_path = "data/featured/model_ready_dataset.csv"
            if not Path(processed_path).exists():
                raise FileNotFoundError(
                    f"Processed dataset not found. Please run previous Phase 4 tasks first."
                )
                
        processed_df = pd.read_csv(processed_path, index_col=0)
        logger.info(f"Loaded processed dataset with shape: {processed_df.shape}")
        
        # Create documentation generator
        doc_generator = Phase4Documentation(original_df, processed_df)
        
        # Generate feature dictionary
        feature_dict = doc_generator.generate_feature_dictionary()
        
        # Generate validation report
        validation_report = doc_generator.generate_validation_report()
        
        # Save documentation
        doc_generator.save_documentation()
        
        # Print summary
        print(f"\n=== Phase 4 Task 6.1 Complete ===")
        print(f"Features documented: {len(feature_dict['features'])}")
        print(f"Validation status: {validation_report['overall_validation_status'].upper()}")
        print(f"Transformations applied: {len(feature_dict['transformations_applied']['summary']['transformation_types_applied'])}")
        
        # Print validation summary
        validation_checks = [
            ('Dataset Comparison', validation_report['dataset_comparison']['validation_passed']),
            ('Feature Engineering', validation_report['feature_engineering_validation']['validation_passed']),
            ('Data Quality', validation_report['data_quality_validation']['validation_passed']),
            ('Model Readiness', validation_report['model_readiness_validation']['validation_passed'])
        ]
        
        print("\nValidation Results:")
        for check_name, passed in validation_checks:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"  {check_name}: {status}")
            
        # Print feature categories summary
        categories = feature_dict['feature_categories']
        print("\nFeature Categories:")
        for category, features in categories.items():
            if features:
                print(f"  {category.replace('_', ' ').title()}: {len(features)} features")
        
        return {
            'feature_dictionary': feature_dict,
            'validation_report': validation_report
        }
        
    except Exception as e:
        logger.error(f"Phase 4 Task 6.1 failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()