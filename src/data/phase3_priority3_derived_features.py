#!/usr/bin/env python3
"""
Phase 3.3.1: Derived Features Creation

Implements Priority 3 of Phase 3 data preprocessing:
- Creates meaningful derived features from existing data
- Implements age groups, attendance categories, and performance ratios
- Adds temporal and behavioral indicators
- Validates feature quality and distributions
- Documents feature engineering decisions

Follows TASKS.md Phase 3.3.1 specifications exactly.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DerivedFeaturesCreator:
    """
    Creates derived features for Phase 3.3.1.
    
    Implements the requirements for task 3.3.1:
    - Create age groups and attendance categories
    - Generate performance ratios and behavioral indicators
    - Add temporal and academic performance features
    - Validate feature quality and distributions
    - Document feature engineering decisions
    """
    
    def __init__(self, input_path: str = "data/processed/missing_handled.csv"):
        """
        Initialize the DerivedFeaturesCreator.
        
        Args:
            input_path: Path to the processed CSV file from Phase 3.2.2
        """
        self.input_path = input_path
        self.data = None
        self.derived_features = {}
        self.feature_definitions = {}
        self.feature_statistics = {}
        self.audit_trail = []
        
    def load_data(self) -> pd.DataFrame:
        """
        Load processed data from Phase 3.2.2.
        
        Returns:
            DataFrame containing the processed data
        """
        try:
            self.data = pd.read_csv(self.input_path)
            logger.info(f"Loaded {len(self.data)} records from {self.input_path}")
            
            # Log available columns
            logger.info(f"Available columns: {list(self.data.columns)}")
            
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def create_age_groups(self) -> pd.Series:
        """
        Create age group categories from age values.
        
        Returns:
            Series with age group categories
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        logger.info("Creating age group features")
        
        # Define age group bins and labels
        age_bins = [0, 12, 15, 18, 21, 100]
        age_labels = ['Child', 'Early_Teen', 'Late_Teen', 'Young_Adult', 'Adult']
        
        # Create age groups
        age_groups = pd.cut(
            self.data['age'], 
            bins=age_bins, 
            labels=age_labels, 
            include_lowest=True
        )
        
        # Handle any potential NaN values by converting to string first
        age_groups = age_groups.astype(str)
        age_groups = age_groups.replace('nan', 'Unknown')
        
        # Store feature definition
        self.feature_definitions['age_group'] = {
            'description': 'Categorical age groups based on developmental stages',
            'bins': age_bins,
            'labels': age_labels,
            'method': 'pd.cut with predefined bins',
            'rationale': 'Groups students by developmental/educational stages'
        }
        
        # Calculate statistics
        age_group_stats = age_groups.value_counts().to_dict()
        age_group_percentages = age_groups.value_counts(normalize=True).round(3).to_dict()
        
        self.feature_statistics['age_group'] = {
            'counts': age_group_stats,
            'percentages': age_group_percentages,
            'unique_values': len(age_group_stats)
        }
        
        logger.info(f"Age groups created: {age_group_stats}")
        
        self.audit_trail.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'created_age_groups',
            'feature': 'age_group',
            'bins': age_bins,
            'labels': age_labels,
            'statistics': age_group_stats,
            'details': 'Created categorical age groups from continuous age values'
        })
        
        return age_groups
    
    def create_attendance_categories(self) -> pd.Series:
        """
        Create attendance rate categories.
        
        Returns:
            Series with attendance categories
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        logger.info("Creating attendance category features")
        
        # Define attendance bins and labels based on common educational standards
        attendance_bins = [0, 0.6, 0.8, 0.9, 1.0]
        attendance_labels = ['Poor', 'Below_Average', 'Good', 'Excellent']
        
        # Create attendance categories
        attendance_categories = pd.cut(
            self.data['attendance_rate'], 
            bins=attendance_bins, 
            labels=attendance_labels, 
            include_lowest=True
        )
        
        # Handle any potential NaN values by converting to string first
        attendance_categories = attendance_categories.astype(str)
        attendance_categories = attendance_categories.replace('nan', 'Unknown')
        
        # Store feature definition
        self.feature_definitions['attendance_category'] = {
            'description': 'Categorical attendance levels based on educational standards',
            'bins': attendance_bins,
            'labels': attendance_labels,
            'method': 'pd.cut with educational standard bins',
            'rationale': 'Categorizes attendance based on common educational thresholds'
        }
        
        # Calculate statistics
        attendance_stats = attendance_categories.value_counts().to_dict()
        attendance_percentages = attendance_categories.value_counts(normalize=True).round(3).to_dict()
        
        self.feature_statistics['attendance_category'] = {
            'counts': attendance_stats,
            'percentages': attendance_percentages,
            'unique_values': len(attendance_stats)
        }
        
        logger.info(f"Attendance categories created: {attendance_stats}")
        
        self.audit_trail.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'created_attendance_categories',
            'feature': 'attendance_category',
            'bins': attendance_bins,
            'labels': attendance_labels,
            'statistics': attendance_stats,
            'details': 'Created categorical attendance levels from continuous attendance_rate'
        })
        
        return attendance_categories
    
    def create_performance_indicators(self) -> Dict[str, pd.Series]:
        """
        Create performance-related derived features.
        
        Returns:
            Dictionary with performance indicator series
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        logger.info("Creating performance indicator features")
        
        performance_features = {}
        
        # 1. High Performer Indicator (based on final_test if available)
        if 'final_test' in self.data.columns:
            # Calculate percentiles for non-missing values
            non_missing_final = self.data['final_test'].dropna()
            if len(non_missing_final) > 0:
                high_threshold = non_missing_final.quantile(0.75)
                low_threshold = non_missing_final.quantile(0.25)
                
                # Create performance categories
                performance_level = pd.Series('Unknown', index=self.data.index)
                
                # Only categorize non-missing values
                non_missing_mask = self.data['final_test'].notna()
                performance_level.loc[
                    non_missing_mask & (self.data['final_test'] >= high_threshold)
                ] = 'High'
                performance_level.loc[
                    non_missing_mask & (self.data['final_test'] < high_threshold) & 
                    (self.data['final_test'] >= low_threshold)
                ] = 'Medium'
                performance_level.loc[
                    non_missing_mask & (self.data['final_test'] < low_threshold)
                ] = 'Low'
                
                performance_features['performance_level'] = performance_level
                
                self.feature_definitions['performance_level'] = {
                    'description': 'Performance level based on final_test quartiles',
                    'high_threshold': high_threshold,
                    'low_threshold': low_threshold,
                    'method': 'Quartile-based categorization',
                    'rationale': 'Identifies high, medium, and low performers based on test scores'
                }
        
        # 2. Attendance-Age Interaction
        if 'age' in self.data.columns and 'attendance_rate' in self.data.columns:
            # Create normalized attendance by age (older students might have different patterns)
            age_mean = self.data['age'].mean()
            attendance_age_ratio = self.data['attendance_rate'] * (self.data['age'] / age_mean)
            
            performance_features['attendance_age_ratio'] = attendance_age_ratio
            
            self.feature_definitions['attendance_age_ratio'] = {
                'description': 'Attendance rate adjusted by age relative to mean age',
                'formula': 'attendance_rate * (age / mean_age)',
                'mean_age': age_mean,
                'method': 'Ratio calculation',
                'rationale': 'Captures interaction between age and attendance patterns'
            }
        
        # 3. Engagement Score (composite of available factors)
        engagement_components = []
        
        # Add attendance component
        if 'attendance_rate' in self.data.columns:
            engagement_components.append(self.data['attendance_rate'])
        
        # Add CCA participation component
        if 'CCA' in self.data.columns:
            cca_engagement = (self.data['CCA'] == 'Yes').astype(float)
            engagement_components.append(cca_engagement)
        
        # Add tuition component (indicates additional academic engagement)
        if 'tuition' in self.data.columns:
            tuition_engagement = (self.data['tuition'] == 'Yes').astype(float)
            engagement_components.append(tuition_engagement)
        
        if engagement_components:
            # Calculate weighted engagement score
            weights = [0.6, 0.25, 0.15][:len(engagement_components)]  # Prioritize attendance
            engagement_score = sum(w * comp for w, comp in zip(weights, engagement_components))
            engagement_score = engagement_score / sum(weights)  # Normalize
            
            performance_features['engagement_score'] = engagement_score
            
            self.feature_definitions['engagement_score'] = {
                'description': 'Composite engagement score from attendance, CCA, and tuition',
                'components': ['attendance_rate', 'CCA', 'tuition'][:len(engagement_components)],
                'weights': weights,
                'method': 'Weighted average of engagement indicators',
                'rationale': 'Measures overall student engagement across multiple dimensions'
            }
        
        # Calculate statistics for all performance features
        for feature_name, feature_series in performance_features.items():
            if feature_series.dtype in ['float64', 'int64']:
                # Numeric feature statistics
                self.feature_statistics[feature_name] = {
                    'mean': feature_series.mean(),
                    'median': feature_series.median(),
                    'std': feature_series.std(),
                    'min': feature_series.min(),
                    'max': feature_series.max(),
                    'missing_count': feature_series.isnull().sum()
                }
            else:
                # Categorical feature statistics
                value_counts = feature_series.value_counts().to_dict()
                self.feature_statistics[feature_name] = {
                    'counts': value_counts,
                    'percentages': feature_series.value_counts(normalize=True).round(3).to_dict(),
                    'unique_values': len(value_counts)
                }
        
        logger.info(f"Created {len(performance_features)} performance indicator features")
        
        self.audit_trail.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'created_performance_indicators',
            'features': list(performance_features.keys()),
            'feature_count': len(performance_features),
            'details': 'Created performance-related derived features'
        })
        
        return performance_features
    
    def create_behavioral_indicators(self) -> Dict[str, pd.Series]:
        """
        Create behavioral and preference-related derived features.
        
        Returns:
            Dictionary with behavioral indicator series
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        logger.info("Creating behavioral indicator features")
        
        behavioral_features = {}
        
        # 1. Learning Style Preference Strength
        if 'learning_style' in self.data.columns:
            # Create binary indicators for each learning style
            learning_styles = self.data['learning_style'].dropna().unique()
            
            for style in learning_styles:
                if pd.notna(style):
                    feature_name = f'learning_style_{style.lower()}'
                    behavioral_features[feature_name] = (
                        self.data['learning_style'] == style
                    ).astype(int)
                    
                    self.feature_definitions[feature_name] = {
                        'description': f'Binary indicator for {style} learning style',
                        'method': 'One-hot encoding of learning_style',
                        'rationale': f'Captures preference for {style} learning approach'
                    }
        
        # 2. Support Seeking Behavior
        support_components = []
        
        # Tuition indicates seeking additional academic support
        if 'tuition' in self.data.columns:
            tuition_support = (self.data['tuition'] == 'Yes').astype(float)
            support_components.append(tuition_support)
        
        # CCA indicates seeking extracurricular engagement
        if 'CCA' in self.data.columns:
            cca_support = (self.data['CCA'] == 'Yes').astype(float)
            support_components.append(cca_support)
        
        if support_components:
            support_seeking = sum(support_components) / len(support_components)
            behavioral_features['support_seeking_score'] = support_seeking
            
            self.feature_definitions['support_seeking_score'] = {
                'description': 'Score indicating tendency to seek additional support/activities',
                'components': ['tuition', 'CCA'][:len(support_components)],
                'method': 'Average of binary support indicators',
                'rationale': 'Measures proactive behavior in seeking additional support'
            }
        
        # 3. Gender-based Learning Patterns (if gender is available)
        if 'gender' in self.data.columns:
            # Create gender-specific features that might be relevant
            gender_values = self.data['gender'].dropna().unique()
            
            for gender in gender_values:
                if pd.notna(gender):
                    feature_name = f'gender_{gender.lower()}'
                    behavioral_features[feature_name] = (
                        self.data['gender'] == gender
                    ).astype(int)
                    
                    self.feature_definitions[feature_name] = {
                        'description': f'Binary indicator for {gender} gender',
                        'method': 'One-hot encoding of gender',
                        'rationale': f'Captures potential gender-specific learning patterns'
                    }
        
        # 4. Consistency Indicator (based on attendance patterns)
        if 'attendance_rate' in self.data.columns:
            # High attendance rate indicates consistent behavior
            consistency_threshold = 0.85
            consistency_indicator = (
                self.data['attendance_rate'] >= consistency_threshold
            ).astype(int)
            
            behavioral_features['high_consistency'] = consistency_indicator
            
            self.feature_definitions['high_consistency'] = {
                'description': f'Binary indicator for consistent attendance (>= {consistency_threshold})',
                'threshold': consistency_threshold,
                'method': 'Binary threshold on attendance_rate',
                'rationale': 'Identifies students with consistent behavioral patterns'
            }
        
        # Calculate statistics for all behavioral features
        for feature_name, feature_series in behavioral_features.items():
            if feature_series.dtype in ['float64', 'int64']:
                # Numeric feature statistics
                self.feature_statistics[feature_name] = {
                    'mean': feature_series.mean(),
                    'median': feature_series.median(),
                    'std': feature_series.std(),
                    'min': feature_series.min(),
                    'max': feature_series.max(),
                    'missing_count': feature_series.isnull().sum()
                }
            else:
                # Categorical feature statistics
                value_counts = feature_series.value_counts().to_dict()
                self.feature_statistics[feature_name] = {
                    'counts': value_counts,
                    'percentages': feature_series.value_counts(normalize=True).round(3).to_dict(),
                    'unique_values': len(value_counts)
                }
        
        logger.info(f"Created {len(behavioral_features)} behavioral indicator features")
        
        self.audit_trail.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'created_behavioral_indicators',
            'features': list(behavioral_features.keys()),
            'feature_count': len(behavioral_features),
            'details': 'Created behavioral and preference-related derived features'
        })
        
        return behavioral_features
    
    def create_risk_indicators(self) -> Dict[str, pd.Series]:
        """
        Create risk and early warning indicator features.
        
        Returns:
            Dictionary with risk indicator series
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        logger.info("Creating risk indicator features")
        
        risk_features = {}
        
        # 1. Low Attendance Risk
        if 'attendance_rate' in self.data.columns:
            low_attendance_threshold = 0.7
            risk_features['low_attendance_risk'] = (
                self.data['attendance_rate'] < low_attendance_threshold
            ).astype(int)
            
            self.feature_definitions['low_attendance_risk'] = {
                'description': f'Risk indicator for low attendance (< {low_attendance_threshold})',
                'threshold': low_attendance_threshold,
                'method': 'Binary threshold on attendance_rate',
                'rationale': 'Early warning indicator for academic risk'
            }
        
        # 2. Age-Grade Mismatch Risk (if age suggests grade retention or acceleration)
        if 'age' in self.data.columns:
            # Typical age ranges for different educational levels
            # Assuming secondary education context (ages 12-18 typical)
            typical_age_range = (12, 18)
            
            age_risk = (
                (self.data['age'] < typical_age_range[0]) | 
                (self.data['age'] > typical_age_range[1])
            ).astype(int)
            
            risk_features['age_mismatch_risk'] = age_risk
            
            self.feature_definitions['age_mismatch_risk'] = {
                'description': 'Risk indicator for age outside typical range',
                'typical_range': typical_age_range,
                'method': 'Binary indicator for age outside typical range',
                'rationale': 'Identifies potential grade retention or acceleration cases'
            }
        
        # 3. Low Engagement Risk
        if 'engagement_score' in self.derived_features:
            low_engagement_threshold = 0.4
            risk_features['low_engagement_risk'] = (
                self.derived_features['engagement_score'] < low_engagement_threshold
            ).astype(int)
            
            self.feature_definitions['low_engagement_risk'] = {
                'description': f'Risk indicator for low engagement (< {low_engagement_threshold})',
                'threshold': low_engagement_threshold,
                'method': 'Binary threshold on engagement_score',
                'rationale': 'Early warning for students at risk of disengagement'
            }
        
        # 4. Multiple Risk Factors
        risk_components = []
        risk_names = []
        
        for risk_name, risk_series in risk_features.items():
            if risk_name != 'multiple_risk_factors':  # Avoid circular reference
                risk_components.append(risk_series)
                risk_names.append(risk_name)
        
        if risk_components:
            multiple_risk = sum(risk_components)
            risk_features['multiple_risk_factors'] = multiple_risk
            
            self.feature_definitions['multiple_risk_factors'] = {
                'description': 'Count of concurrent risk factors',
                'components': risk_names,
                'method': 'Sum of individual risk indicators',
                'rationale': 'Identifies students with multiple concurrent risk factors'
            }
        
        # Calculate statistics for all risk features
        for feature_name, feature_series in risk_features.items():
            if feature_series.dtype in ['float64', 'int64']:
                # Numeric feature statistics
                self.feature_statistics[feature_name] = {
                    'mean': feature_series.mean(),
                    'median': feature_series.median(),
                    'std': feature_series.std(),
                    'min': feature_series.min(),
                    'max': feature_series.max(),
                    'missing_count': feature_series.isnull().sum()
                }
            else:
                # Categorical feature statistics
                value_counts = feature_series.value_counts().to_dict()
                self.feature_statistics[feature_name] = {
                    'counts': value_counts,
                    'percentages': feature_series.value_counts(normalize=True).round(3).to_dict(),
                    'unique_values': len(value_counts)
                }
        
        logger.info(f"Created {len(risk_features)} risk indicator features")
        
        self.audit_trail.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'created_risk_indicators',
            'features': list(risk_features.keys()),
            'feature_count': len(risk_features),
            'details': 'Created risk and early warning indicator features'
        })
        
        return risk_features
    
    def validate_derived_features(self) -> Dict[str, Any]:
        """
        Validate the quality and distributions of derived features.
        
        Returns:
            Dictionary with validation results
        """
        if not self.derived_features:
            raise ValueError("No derived features available. Create features first.")
        
        logger.info("Validating derived features")
        
        validation_results = {
            'total_features_created': len(self.derived_features),
            'feature_types': {},
            'quality_checks': {},
            'distribution_analysis': {},
            'correlation_analysis': {},
            'issues_found': []
        }
        
        # Analyze feature types
        for feature_name, feature_series in self.derived_features.items():
            if feature_series.dtype in ['float64', 'int64']:
                validation_results['feature_types'][feature_name] = 'numeric'
            else:
                validation_results['feature_types'][feature_name] = 'categorical'
        
        # Quality checks
        for feature_name, feature_series in self.derived_features.items():
            quality_check = {
                'missing_count': feature_series.isnull().sum(),
                'missing_percentage': (feature_series.isnull().sum() / len(feature_series)) * 100,
                'unique_values': feature_series.nunique(),
                'data_type': str(feature_series.dtype)
            }
            
            # Check for potential issues
            if quality_check['missing_percentage'] > 50:
                validation_results['issues_found'].append(
                    f"{feature_name}: High missing percentage ({quality_check['missing_percentage']:.1f}%)"
                )
            
            if quality_check['unique_values'] == 1:
                validation_results['issues_found'].append(
                    f"{feature_name}: Constant feature (only one unique value)"
                )
            
            # For numeric features, check for extreme values
            if feature_series.dtype in ['float64', 'int64']:
                q1 = feature_series.quantile(0.25)
                q3 = feature_series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = ((feature_series < lower_bound) | (feature_series > upper_bound)).sum()
                quality_check['outlier_count'] = outliers
                quality_check['outlier_percentage'] = (outliers / len(feature_series)) * 100
                
                if quality_check['outlier_percentage'] > 10:
                    validation_results['issues_found'].append(
                        f"{feature_name}: High outlier percentage ({quality_check['outlier_percentage']:.1f}%)"
                    )
            
            validation_results['quality_checks'][feature_name] = quality_check
        
        # Distribution analysis for numeric features
        numeric_features = [
            name for name, series in self.derived_features.items() 
            if series.dtype in ['float64', 'int64']
        ]
        
        for feature_name in numeric_features:
            feature_series = self.derived_features[feature_name]
            validation_results['distribution_analysis'][feature_name] = {
                'mean': feature_series.mean(),
                'median': feature_series.median(),
                'std': feature_series.std(),
                'skewness': feature_series.skew(),
                'kurtosis': feature_series.kurtosis(),
                'min': feature_series.min(),
                'max': feature_series.max()
            }
        
        # Correlation analysis between numeric derived features
        if len(numeric_features) > 1:
            numeric_df = pd.DataFrame({name: self.derived_features[name] for name in numeric_features})
            correlation_matrix = numeric_df.corr()
            
            # Find high correlations (> 0.8)
            high_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.8:
                        high_correlations.append({
                            'feature1': correlation_matrix.columns[i],
                            'feature2': correlation_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            validation_results['correlation_analysis'] = {
                'high_correlations': high_correlations,
                'correlation_matrix': correlation_matrix.to_dict()
            }
            
            if high_correlations:
                for corr in high_correlations:
                    validation_results['issues_found'].append(
                        f"High correlation between {corr['feature1']} and {corr['feature2']}: {corr['correlation']:.3f}"
                    )
        
        logger.info(f"Validation completed: {len(validation_results['issues_found'])} issues found")
        
        if validation_results['issues_found']:
            logger.warning("Issues found during validation:")
            for issue in validation_results['issues_found']:
                logger.warning(f"  - {issue}")
        
        self.audit_trail.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'validated_derived_features',
            'features_validated': len(self.derived_features),
            'issues_found': len(validation_results['issues_found']),
            'details': 'Validated quality and distributions of derived features'
        })
        
        return validation_results
    
    def create_feature_dataset(self) -> pd.DataFrame:
        """
        Create final dataset with original and derived features.
        
        Returns:
            DataFrame with original and derived features combined
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if not self.derived_features:
            raise ValueError("No derived features available. Create features first.")
        
        logger.info("Creating final feature dataset")
        
        # Start with original data
        feature_dataset = self.data.copy()
        
        # Add all derived features
        for feature_name, feature_series in self.derived_features.items():
            feature_dataset[feature_name] = feature_series
        
        logger.info(f"Final dataset created with {len(feature_dataset.columns)} total features")
        logger.info(f"Original features: {len(self.data.columns)}")
        logger.info(f"Derived features: {len(self.derived_features)}")
        
        return feature_dataset
    
    def save_derived_features(self, output_path: str = "data/featured/derived_features.csv") -> None:
        """
        Save dataset with derived features.
        
        Args:
            output_path: Path to save the featured dataset
        """
        feature_dataset = self.create_feature_dataset()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        feature_dataset.to_csv(output_file, index=False)
        logger.info(f"Saved featured dataset to {output_file}")
    
    def save_feature_documentation(self, output_path: str = "data/featured/feature_definitions.json") -> None:
        """
        Save feature definitions and statistics.
        
        Args:
            output_path: Path to save the feature documentation
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        documentation = {
            'feature_definitions': self.feature_definitions,
            'feature_statistics': self.feature_statistics,
            'creation_summary': {
                'total_derived_features': len(self.derived_features),
                'feature_categories': {
                    'demographic': ['age_group'],
                    'behavioral': [name for name in self.derived_features.keys() if 'learning_style' in name or 'gender' in name or 'consistency' in name or 'support' in name],
                    'performance': [name for name in self.derived_features.keys() if 'performance' in name or 'engagement' in name or 'attendance_age' in name],
                    'risk': [name for name in self.derived_features.keys() if 'risk' in name],
                    'categorical': ['attendance_category']
                }
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(documentation, f, indent=2, default=str)
        
        logger.info(f"Saved feature documentation to {output_file}")
    
    def save_audit_trail(self, output_path: str = "data/featured/derived_features_audit.json") -> None:
        """
        Save audit trail to JSON file.
        
        Args:
            output_path: Path to save the audit trail
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.audit_trail, f, indent=2, default=str)
        
        logger.info(f"Saved audit trail to {output_file}")
    
    def run_complete_feature_creation(self) -> Dict[str, Any]:
        """
        Run the complete derived features creation process.
        
        Returns:
            Dictionary with complete feature creation summary
        """
        logger.info("Starting Phase 3.3.1: Derived Features Creation")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Create age groups
        self.derived_features['age_group'] = self.create_age_groups()
        
        # Step 3: Create attendance categories
        self.derived_features['attendance_category'] = self.create_attendance_categories()
        
        # Step 4: Create performance indicators
        performance_features = self.create_performance_indicators()
        self.derived_features.update(performance_features)
        
        # Step 5: Create behavioral indicators
        behavioral_features = self.create_behavioral_indicators()
        self.derived_features.update(behavioral_features)
        
        # Step 6: Create risk indicators
        risk_features = self.create_risk_indicators()
        self.derived_features.update(risk_features)
        
        # Step 7: Validate derived features
        validation_results = self.validate_derived_features()
        
        # Step 8: Save results
        self.save_derived_features()
        self.save_feature_documentation()
        self.save_audit_trail()
        
        # Step 9: Generate summary
        summary = {
            'input_file': self.input_path,
            'total_records': len(self.data),
            'original_features': len(self.data.columns),
            'derived_features_created': len(self.derived_features),
            'total_features': len(self.data.columns) + len(self.derived_features),
            'feature_categories': {
                'demographic': 1,  # age_group
                'categorical': 1,  # attendance_category
                'performance': len([f for f in self.derived_features.keys() if 'performance' in f or 'engagement' in f or 'attendance_age' in f]),
                'behavioral': len([f for f in self.derived_features.keys() if 'learning_style' in f or 'gender' in f or 'consistency' in f or 'support' in f]),
                'risk': len([f for f in self.derived_features.keys() if 'risk' in f])
            },
            'validation_results': validation_results,
            'audit_trail_entries': len(self.audit_trail)
        }
        
        logger.info("Phase 3.3.1: Derived Features Creation completed successfully")
        return summary


def main():
    """
    Main execution function for Phase 3.3.1: Derived Features Creation.
    """
    # Initialize creator
    creator = DerivedFeaturesCreator()
    
    # Run complete feature creation
    summary = creator.run_complete_feature_creation()
    
    # Print summary
    print("\n=== Phase 3.3.1: Derived Features Creation Summary ===")
    print(f"Total records: {summary['total_records']}")
    print(f"Original features: {summary['original_features']}")
    print(f"Derived features created: {summary['derived_features_created']}")
    print(f"Total features: {summary['total_features']}")
    
    print("\nFeature categories:")
    for category, count in summary['feature_categories'].items():
        print(f"  {category}: {count} features")
    
    print(f"\nValidation issues found: {len(summary['validation_results']['issues_found'])}")
    if summary['validation_results']['issues_found']:
        print("Issues:")
        for issue in summary['validation_results']['issues_found']:
            print(f"  - {issue}")
    
    print(f"\nAudit trail entries: {summary['audit_trail_entries']}")
    
    print("\nDerived features creation completed successfully!")
    print("Output files:")
    print("- data/featured/derived_features.csv")
    print("- data/featured/feature_definitions.json")
    print("- data/featured/derived_features_audit.json")


if __name__ == "__main__":
    main()