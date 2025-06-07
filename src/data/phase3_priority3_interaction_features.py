#!/usr/bin/env python3
"""
Phase 3.3.2: Interaction Features Creation

Implements Priority 3 of Phase 3 data preprocessing:
- Creates meaningful interaction features between variables
- Implements statistical and domain-knowledge based interactions
- Validates interaction feature quality and significance
- Documents interaction rationale and impact
- Ensures computational efficiency

Follows TASKS.md Phase 3.3.2 specifications exactly.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import json
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InteractionFeaturesCreator:
    """
    Creates interaction features for Phase 3.3.2.
    
    Implements the requirements for task 3.3.2:
    - Create meaningful interaction features between variables
    - Implement statistical and domain-knowledge based interactions
    - Validate interaction feature quality and significance
    - Document interaction rationale and impact
    - Ensure computational efficiency
    """
    
    def __init__(self, input_path: str = "data/featured/derived_features.csv"):
        """
        Initialize the InteractionFeaturesCreator.
        
        Args:
            input_path: Path to the derived features CSV file from Phase 3.3.1
        """
        self.input_path = input_path
        self.data = None
        self.interaction_features = {}
        self.interaction_definitions = {}
        self.interaction_statistics = {}
        self.audit_trail = []
        
    def load_data(self) -> pd.DataFrame:
        """
        Load derived features data from Phase 3.3.1.
        
        Returns:
            DataFrame containing the derived features data
        """
        try:
            self.data = pd.read_csv(self.input_path)
            logger.info(f"Loaded {len(self.data)} records from {self.input_path}")
            
            # Log available columns
            logger.info(f"Available columns: {len(self.data.columns)} features")
            
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def identify_interaction_candidates(self) -> Dict[str, List[str]]:
        """
        Identify candidate features for interaction creation.
        
        Returns:
            Dictionary with categorized interaction candidates
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        logger.info("Identifying interaction candidates")
        
        candidates = {
            'numeric_features': [],
            'categorical_features': [],
            'binary_features': [],
            'high_priority_pairs': [],
            'domain_knowledge_pairs': []
        }
        
        # Categorize features by type
        for col in self.data.columns:
            if col in ['student_id']:  # Skip ID columns
                continue
                
            if self.data[col].dtype in ['float64', 'int64']:
                # Check if it's binary (0/1 values only)
                unique_vals = self.data[col].dropna().unique()
                if len(unique_vals) <= 2 and all(val in [0, 1] for val in unique_vals):
                    candidates['binary_features'].append(col)
                else:
                    candidates['numeric_features'].append(col)
            else:
                candidates['categorical_features'].append(col)
        
        # Define high-priority interaction pairs based on domain knowledge
        domain_pairs = [
            ('age', 'attendance_rate'),  # Age-attendance interaction
            ('attendance_rate', 'final_test'),  # Attendance-performance
            ('age', 'final_test'),  # Age-performance
            ('gender', 'learning_style'),  # Gender-learning style
            ('tuition', 'final_test'),  # Tuition-performance
            ('CCA', 'attendance_rate'),  # CCA-attendance
            ('age_group', 'attendance_category'),  # Derived feature interactions
            ('engagement_score', 'performance_level'),  # Engagement-performance
            ('support_seeking_score', 'final_test'),  # Support seeking-performance
        ]
        
        # Filter pairs that exist in the data
        for pair in domain_pairs:
            if pair[0] in self.data.columns and pair[1] in self.data.columns:
                candidates['domain_knowledge_pairs'].append(pair)
                candidates['high_priority_pairs'].append(pair)
        
        logger.info(f"Identified candidates:")
        logger.info(f"  Numeric features: {len(candidates['numeric_features'])}")
        logger.info(f"  Categorical features: {len(candidates['categorical_features'])}")
        logger.info(f"  Binary features: {len(candidates['binary_features'])}")
        logger.info(f"  High priority pairs: {len(candidates['high_priority_pairs'])}")
        
        return candidates
    
    def create_numeric_interactions(self, candidates: Dict[str, List[str]]) -> Dict[str, pd.Series]:
        """
        Create interaction features between numeric variables.
        
        Args:
            candidates: Dictionary with categorized feature candidates
            
        Returns:
            Dictionary with numeric interaction features
        """
        logger.info("Creating numeric interaction features")
        
        numeric_interactions = {}
        numeric_features = candidates['numeric_features']
        
        # High-priority numeric interactions
        priority_numeric_pairs = [
            ('age', 'attendance_rate'),
            ('attendance_rate', 'final_test'),
            ('age', 'final_test'),
            ('attendance_age_ratio', 'engagement_score'),
        ]
        
        for pair in priority_numeric_pairs:
            if pair[0] in numeric_features and pair[1] in numeric_features:
                feat1, feat2 = pair
                
                # Skip if either feature has too many missing values
                if (self.data[feat1].isnull().sum() / len(self.data) > 0.5 or 
                    self.data[feat2].isnull().sum() / len(self.data) > 0.5):
                    continue
                
                # 1. Multiplicative interaction
                interaction_name = f"{feat1}_x_{feat2}"
                numeric_interactions[interaction_name] = self.data[feat1] * self.data[feat2]
                
                self.interaction_definitions[interaction_name] = {
                    'type': 'multiplicative',
                    'features': [feat1, feat2],
                    'formula': f'{feat1} * {feat2}',
                    'rationale': f'Captures multiplicative effect between {feat1} and {feat2}'
                }
                
                # 2. Ratio interaction (if feat2 is not zero)
                if (self.data[feat2] != 0).all():
                    ratio_name = f"{feat1}_div_{feat2}"
                    numeric_interactions[ratio_name] = self.data[feat1] / self.data[feat2]
                    
                    self.interaction_definitions[ratio_name] = {
                        'type': 'ratio',
                        'features': [feat1, feat2],
                        'formula': f'{feat1} / {feat2}',
                        'rationale': f'Captures ratio relationship between {feat1} and {feat2}'
                    }
                
                # 3. Difference interaction
                diff_name = f"{feat1}_minus_{feat2}"
                numeric_interactions[diff_name] = self.data[feat1] - self.data[feat2]
                
                self.interaction_definitions[diff_name] = {
                    'type': 'difference',
                    'features': [feat1, feat2],
                    'formula': f'{feat1} - {feat2}',
                    'rationale': f'Captures difference between {feat1} and {feat2}'
                }
        
        # Polynomial features for key variables
        key_numeric_features = ['age', 'attendance_rate']
        for feat in key_numeric_features:
            if feat in numeric_features:
                # Quadratic term
                quad_name = f"{feat}_squared"
                numeric_interactions[quad_name] = self.data[feat] ** 2
                
                self.interaction_definitions[quad_name] = {
                    'type': 'polynomial',
                    'features': [feat],
                    'formula': f'{feat}^2',
                    'rationale': f'Captures non-linear relationship in {feat}'
                }
        
        logger.info(f"Created {len(numeric_interactions)} numeric interaction features")
        
        return numeric_interactions
    
    def create_categorical_interactions(self, candidates: Dict[str, List[str]]) -> Dict[str, pd.Series]:
        """
        Create interaction features between categorical variables.
        
        Args:
            candidates: Dictionary with categorized feature candidates
            
        Returns:
            Dictionary with categorical interaction features
        """
        logger.info("Creating categorical interaction features")
        
        categorical_interactions = {}
        categorical_features = candidates['categorical_features']
        
        # High-priority categorical interactions
        priority_categorical_pairs = [
            ('gender', 'learning_style'),
            ('age_group', 'attendance_category'),
            ('gender', 'CCA'),
            ('learning_style', 'tuition'),
        ]
        
        for pair in priority_categorical_pairs:
            if pair[0] in categorical_features and pair[1] in categorical_features:
                feat1, feat2 = pair
                
                # Create combined categorical feature
                interaction_name = f"{feat1}_x_{feat2}"
                
                # Combine categories with separator
                combined_values = (
                    self.data[feat1].astype(str) + "_" + self.data[feat2].astype(str)
                )
                
                # Handle missing values
                combined_values = combined_values.replace('nan_nan', 'Unknown')
                combined_values = combined_values.replace('nan_', 'Unknown')
                combined_values = combined_values.replace('_nan', 'Unknown')
                
                categorical_interactions[interaction_name] = combined_values
                
                self.interaction_definitions[interaction_name] = {
                    'type': 'categorical_combination',
                    'features': [feat1, feat2],
                    'formula': f'{feat1} + "_" + {feat2}',
                    'rationale': f'Captures combined effect of {feat1} and {feat2} categories'
                }
        
        logger.info(f"Created {len(categorical_interactions)} categorical interaction features")
        
        return categorical_interactions
    
    def create_mixed_interactions(self, candidates: Dict[str, List[str]]) -> Dict[str, pd.Series]:
        """
        Create interaction features between numeric and categorical variables.
        
        Args:
            candidates: Dictionary with categorized feature candidates
            
        Returns:
            Dictionary with mixed interaction features
        """
        logger.info("Creating mixed (numeric-categorical) interaction features")
        
        mixed_interactions = {}
        numeric_features = candidates['numeric_features']
        categorical_features = candidates['categorical_features']
        
        # High-priority mixed interactions
        priority_mixed_pairs = [
            ('attendance_rate', 'gender'),
            ('age', 'learning_style'),
            ('final_test', 'tuition'),
            ('engagement_score', 'age_group'),
        ]
        
        for pair in priority_mixed_pairs:
            numeric_feat, categorical_feat = pair
            
            if (numeric_feat in numeric_features and categorical_feat in categorical_features):
                
                # Create group-based statistics
                grouped_stats = self.data.groupby(categorical_feat)[numeric_feat].agg([
                    'mean', 'median', 'std'
                ]).round(3)
                
                # Map back to original data
                for stat in ['mean', 'median', 'std']:
                    interaction_name = f"{numeric_feat}_{stat}_by_{categorical_feat}"
                    
                    # Create mapping dictionary
                    stat_mapping = grouped_stats[stat].to_dict()
                    
                    # Map values
                    mapped_values = self.data[categorical_feat].map(stat_mapping)
                    
                    mixed_interactions[interaction_name] = mapped_values
                    
                    self.interaction_definitions[interaction_name] = {
                        'type': 'group_statistic',
                        'features': [numeric_feat, categorical_feat],
                        'formula': f'{stat}({numeric_feat}) grouped by {categorical_feat}',
                        'rationale': f'Captures {stat} of {numeric_feat} within {categorical_feat} groups'
                    }
                
                # Create deviation from group mean
                group_means = self.data.groupby(categorical_feat)[numeric_feat].transform('mean')
                deviation_name = f"{numeric_feat}_deviation_from_{categorical_feat}_mean"
                mixed_interactions[deviation_name] = self.data[numeric_feat] - group_means
                
                self.interaction_definitions[deviation_name] = {
                    'type': 'group_deviation',
                    'features': [numeric_feat, categorical_feat],
                    'formula': f'{numeric_feat} - mean({numeric_feat}) by {categorical_feat}',
                    'rationale': f'Captures how {numeric_feat} deviates from its group mean in {categorical_feat}'
                }
        
        logger.info(f"Created {len(mixed_interactions)} mixed interaction features")
        
        return mixed_interactions
    
    def create_domain_specific_interactions(self) -> Dict[str, pd.Series]:
        """
        Create domain-specific interaction features based on educational context.
        
        Returns:
            Dictionary with domain-specific interaction features
        """
        logger.info("Creating domain-specific interaction features")
        
        domain_interactions = {}
        
        # 1. Academic Support Index
        support_components = []
        if 'tuition' in self.data.columns:
            support_components.append((self.data['tuition'] == 'Yes').astype(float))
        if 'CCA' in self.data.columns:
            support_components.append((self.data['CCA'] == 'Yes').astype(float))
        if 'attendance_rate' in self.data.columns:
            support_components.append(self.data['attendance_rate'])
        
        if len(support_components) >= 2:
            academic_support_index = sum(support_components) / len(support_components)
            domain_interactions['academic_support_index'] = academic_support_index
            
            self.interaction_definitions['academic_support_index'] = {
                'type': 'domain_composite',
                'features': ['tuition', 'CCA', 'attendance_rate'][:len(support_components)],
                'formula': 'average of normalized support indicators',
                'rationale': 'Comprehensive measure of academic support utilization'
            }
        
        # 2. Age-Performance Expectation
        if 'age' in self.data.columns and 'final_test' in self.data.columns:
            # Calculate expected performance by age group
            age_performance_expected = self.data.groupby('age')['final_test'].transform('median')
            
            # Performance relative to age peers
            performance_vs_age_peers = self.data['final_test'] - age_performance_expected
            domain_interactions['performance_vs_age_peers'] = performance_vs_age_peers
            
            self.interaction_definitions['performance_vs_age_peers'] = {
                'type': 'domain_relative',
                'features': ['final_test', 'age'],
                'formula': 'final_test - median(final_test) by age',
                'rationale': 'Performance relative to age-matched peers'
            }
        
        # 3. Learning Style Effectiveness
        if ('learning_style' in self.data.columns and 
            'final_test' in self.data.columns and 
            'attendance_rate' in self.data.columns):
            
            # Calculate learning style effectiveness
            style_effectiveness = self.data.groupby('learning_style').agg({
                'final_test': 'median',
                'attendance_rate': 'median'
            })
            
            # Create composite effectiveness score
            style_effectiveness['composite'] = (
                style_effectiveness['final_test'] * 0.7 + 
                style_effectiveness['attendance_rate'] * 100 * 0.3
            )
            
            # Map back to students
            effectiveness_mapping = style_effectiveness['composite'].to_dict()
            learning_style_effectiveness = self.data['learning_style'].map(effectiveness_mapping)
            
            domain_interactions['learning_style_effectiveness'] = learning_style_effectiveness
            
            self.interaction_definitions['learning_style_effectiveness'] = {
                'type': 'domain_effectiveness',
                'features': ['learning_style', 'final_test', 'attendance_rate'],
                'formula': '0.7 * median(final_test) + 0.3 * median(attendance_rate) by learning_style',
                'rationale': 'Effectiveness score of each learning style based on outcomes'
            }
        
        # 4. Risk Amplification Score
        risk_features = [col for col in self.data.columns if 'risk' in col.lower()]
        if len(risk_features) >= 2:
            # Calculate risk amplification (risks compound)
            risk_sum = sum(self.data[feat] for feat in risk_features)
            risk_amplification = risk_sum ** 1.5  # Non-linear amplification
            
            domain_interactions['risk_amplification_score'] = risk_amplification
            
            self.interaction_definitions['risk_amplification_score'] = {
                'type': 'domain_amplification',
                'features': risk_features,
                'formula': '(sum of risk indicators)^1.5',
                'rationale': 'Non-linear amplification of multiple concurrent risks'
            }
        
        logger.info(f"Created {len(domain_interactions)} domain-specific interaction features")
        
        return domain_interactions
    
    def validate_interaction_features(self) -> Dict[str, Any]:
        """
        Validate the quality and significance of interaction features.
        
        Returns:
            Dictionary with validation results
        """
        if not self.interaction_features:
            raise ValueError("No interaction features available. Create features first.")
        
        logger.info("Validating interaction features")
        
        validation_results = {
            'total_interactions_created': len(self.interaction_features),
            'interaction_types': {},
            'quality_checks': {},
            'significance_analysis': {},
            'correlation_with_target': {},
            'issues_found': []
        }
        
        # Analyze interaction types
        for feature_name in self.interaction_features.keys():
            if feature_name in self.interaction_definitions:
                interaction_type = self.interaction_definitions[feature_name]['type']
                if interaction_type not in validation_results['interaction_types']:
                    validation_results['interaction_types'][interaction_type] = 0
                validation_results['interaction_types'][interaction_type] += 1
        
        # Quality checks for each interaction
        for feature_name, feature_series in self.interaction_features.items():
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
            
            # For numeric features, check for extreme values and infinite values
            if feature_series.dtype in ['float64', 'int64']:
                # Check for infinite values
                inf_count = np.isinf(feature_series).sum()
                if inf_count > 0:
                    validation_results['issues_found'].append(
                        f"{feature_name}: Contains {inf_count} infinite values"
                    )
                
                # Check for extreme outliers
                finite_values = feature_series[np.isfinite(feature_series)]
                if len(finite_values) > 0:
                    q1 = finite_values.quantile(0.25)
                    q3 = finite_values.quantile(0.75)
                    iqr = q3 - q1
                    
                    if iqr > 0:
                        lower_bound = q1 - 3 * iqr  # More conservative than 1.5
                        upper_bound = q3 + 3 * iqr
                        
                        extreme_outliers = (
                            (finite_values < lower_bound) | (finite_values > upper_bound)
                        ).sum()
                        
                        quality_check['extreme_outlier_count'] = extreme_outliers
                        quality_check['extreme_outlier_percentage'] = (extreme_outliers / len(finite_values)) * 100
                        
                        if quality_check['extreme_outlier_percentage'] > 5:
                            validation_results['issues_found'].append(
                                f"{feature_name}: High extreme outlier percentage ({quality_check['extreme_outlier_percentage']:.1f}%)"
                            )
            
            validation_results['quality_checks'][feature_name] = quality_check
        
        # Correlation with target variable (if available)
        if 'final_test' in self.data.columns:
            target_correlations = {}
            
            for feature_name, feature_series in self.interaction_features.items():
                if feature_series.dtype in ['float64', 'int64']:
                    # Calculate correlation with target
                    correlation = feature_series.corr(self.data['final_test'])
                    if not np.isnan(correlation):
                        target_correlations[feature_name] = correlation
            
            # Sort by absolute correlation
            sorted_correlations = sorted(
                target_correlations.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )
            
            validation_results['correlation_with_target'] = {
                'correlations': target_correlations,
                'top_5_correlations': sorted_correlations[:5],
                'significant_correlations': [
                    (name, corr) for name, corr in target_correlations.items() 
                    if abs(corr) > 0.1
                ]
            }
        
        # Check for high correlations between interaction features
        numeric_interactions = {
            name: series for name, series in self.interaction_features.items()
            if series.dtype in ['float64', 'int64']
        }
        
        if len(numeric_interactions) > 1:
            interaction_df = pd.DataFrame(numeric_interactions)
            correlation_matrix = interaction_df.corr()
            
            high_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.9 and not np.isnan(corr_value):
                        high_correlations.append({
                            'feature1': correlation_matrix.columns[i],
                            'feature2': correlation_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            validation_results['high_correlations_between_interactions'] = high_correlations
            
            if high_correlations:
                for corr in high_correlations:
                    validation_results['issues_found'].append(
                        f"High correlation between interactions {corr['feature1']} and {corr['feature2']}: {corr['correlation']:.3f}"
                    )
        
        logger.info(f"Validation completed: {len(validation_results['issues_found'])} issues found")
        
        if validation_results['issues_found']:
            logger.warning("Issues found during validation:")
            for issue in validation_results['issues_found'][:5]:  # Show first 5 issues
                logger.warning(f"  - {issue}")
            if len(validation_results['issues_found']) > 5:
                logger.warning(f"  ... and {len(validation_results['issues_found']) - 5} more issues")
        
        return validation_results
    
    def calculate_interaction_statistics(self) -> None:
        """
        Calculate statistics for all interaction features.
        """
        logger.info("Calculating interaction feature statistics")
        
        for feature_name, feature_series in self.interaction_features.items():
            if feature_series.dtype in ['float64', 'int64']:
                # Numeric feature statistics
                finite_values = feature_series[np.isfinite(feature_series)]
                
                self.interaction_statistics[feature_name] = {
                    'mean': finite_values.mean() if len(finite_values) > 0 else np.nan,
                    'median': finite_values.median() if len(finite_values) > 0 else np.nan,
                    'std': finite_values.std() if len(finite_values) > 0 else np.nan,
                    'min': finite_values.min() if len(finite_values) > 0 else np.nan,
                    'max': finite_values.max() if len(finite_values) > 0 else np.nan,
                    'missing_count': feature_series.isnull().sum(),
                    'infinite_count': np.isinf(feature_series).sum()
                }
            else:
                # Categorical feature statistics
                value_counts = feature_series.value_counts().to_dict()
                self.interaction_statistics[feature_name] = {
                    'counts': value_counts,
                    'percentages': feature_series.value_counts(normalize=True).round(3).to_dict(),
                    'unique_values': len(value_counts),
                    'missing_count': feature_series.isnull().sum()
                }
    
    def create_final_dataset(self) -> pd.DataFrame:
        """
        Create final dataset with original, derived, and interaction features.
        
        Returns:
            DataFrame with all features combined
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if not self.interaction_features:
            raise ValueError("No interaction features available. Create features first.")
        
        logger.info("Creating final interaction dataset")
        
        # Start with original data
        final_dataset = self.data.copy()
        
        # Add all interaction features
        for feature_name, feature_series in self.interaction_features.items():
            final_dataset[feature_name] = feature_series
        
        logger.info(f"Final dataset created with {len(final_dataset.columns)} total features")
        logger.info(f"Original + derived features: {len(self.data.columns)}")
        logger.info(f"Interaction features: {len(self.interaction_features)}")
        
        return final_dataset
    
    def save_interaction_features(self, output_path: str = "data/featured/interaction_features.csv") -> None:
        """
        Save dataset with interaction features.
        
        Args:
            output_path: Path to save the interaction dataset
        """
        final_dataset = self.create_final_dataset()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        final_dataset.to_csv(output_file, index=False)
        logger.info(f"Saved interaction dataset to {output_file}")
    
    def save_interaction_documentation(self, output_path: str = "data/featured/interaction_definitions.json") -> None:
        """
        Save interaction definitions and statistics.
        
        Args:
            output_path: Path to save the interaction documentation
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        documentation = {
            'interaction_definitions': self.interaction_definitions,
            'interaction_statistics': self.interaction_statistics,
            'creation_summary': {
                'total_interaction_features': len(self.interaction_features),
                'interaction_categories': {
                    'numeric': len([name for name, defn in self.interaction_definitions.items() 
                                  if defn['type'] in ['multiplicative', 'ratio', 'difference', 'polynomial']]),
                    'categorical': len([name for name, defn in self.interaction_definitions.items() 
                                     if defn['type'] == 'categorical_combination']),
                    'mixed': len([name for name, defn in self.interaction_definitions.items() 
                                if defn['type'] in ['group_statistic', 'group_deviation']]),
                    'domain_specific': len([name for name, defn in self.interaction_definitions.items() 
                                          if defn['type'].startswith('domain')])
                }
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(documentation, f, indent=2, default=str)
        
        logger.info(f"Saved interaction documentation to {output_file}")
    
    def save_audit_trail(self, output_path: str = "data/featured/interaction_features_audit.json") -> None:
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
    
    def run_complete_interaction_creation(self) -> Dict[str, Any]:
        """
        Run the complete interaction features creation process.
        
        Returns:
            Dictionary with complete interaction creation summary
        """
        logger.info("Starting Phase 3.3.2: Interaction Features Creation")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Identify interaction candidates
        candidates = self.identify_interaction_candidates()
        
        # Step 3: Create numeric interactions
        numeric_interactions = self.create_numeric_interactions(candidates)
        self.interaction_features.update(numeric_interactions)
        
        # Step 4: Create categorical interactions
        categorical_interactions = self.create_categorical_interactions(candidates)
        self.interaction_features.update(categorical_interactions)
        
        # Step 5: Create mixed interactions
        mixed_interactions = self.create_mixed_interactions(candidates)
        self.interaction_features.update(mixed_interactions)
        
        # Step 6: Create domain-specific interactions
        domain_interactions = self.create_domain_specific_interactions()
        self.interaction_features.update(domain_interactions)
        
        # Step 7: Calculate statistics
        self.calculate_interaction_statistics()
        
        # Step 8: Validate interaction features
        validation_results = self.validate_interaction_features()
        
        # Step 9: Save results
        self.save_interaction_features()
        self.save_interaction_documentation()
        self.save_audit_trail()
        
        # Step 10: Generate summary
        summary = {
            'input_file': self.input_path,
            'total_records': len(self.data),
            'original_features': len(self.data.columns),
            'interaction_features_created': len(self.interaction_features),
            'total_features': len(self.data.columns) + len(self.interaction_features),
            'interaction_categories': {
                'numeric': len(numeric_interactions),
                'categorical': len(categorical_interactions),
                'mixed': len(mixed_interactions),
                'domain_specific': len(domain_interactions)
            },
            'validation_results': validation_results,
            'candidates_identified': candidates,
            'audit_trail_entries': len(self.audit_trail)
        }
        
        self.audit_trail.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'completed_interaction_creation',
            'total_interactions': len(self.interaction_features),
            'categories': summary['interaction_categories'],
            'validation_issues': len(validation_results['issues_found']),
            'details': 'Completed Phase 3.3.2: Interaction Features Creation'
        })
        
        logger.info("Phase 3.3.2: Interaction Features Creation completed successfully")
        return summary


def main():
    """
    Main execution function for Phase 3.3.2: Interaction Features Creation.
    """
    # Initialize creator
    creator = InteractionFeaturesCreator()
    
    # Run complete interaction creation
    summary = creator.run_complete_interaction_creation()
    
    # Print summary
    print("\n=== Phase 3.3.2: Interaction Features Creation Summary ===")
    print(f"Total records: {summary['total_records']}")
    print(f"Original + derived features: {summary['original_features']}")
    print(f"Interaction features created: {summary['interaction_features_created']}")
    print(f"Total features: {summary['total_features']}")
    
    print("\nInteraction categories:")
    for category, count in summary['interaction_categories'].items():
        print(f"  {category}: {count} features")
    
    print(f"\nValidation issues found: {len(summary['validation_results']['issues_found'])}")
    if summary['validation_results']['issues_found']:
        print("Top issues:")
        for issue in summary['validation_results']['issues_found'][:3]:
            print(f"  - {issue}")
    
    # Show top correlations with target if available
    if 'correlation_with_target' in summary['validation_results']:
        top_corrs = summary['validation_results']['correlation_with_target'].get('top_5_correlations', [])
        if top_corrs:
            print("\nTop 3 correlations with final_test:")
            for name, corr in top_corrs[:3]:
                print(f"  {name}: {corr:.3f}")
    
    print(f"\nAudit trail entries: {summary['audit_trail_entries']}")
    
    print("\nInteraction features creation completed successfully!")
    print("Output files:")
    print("- data/featured/interaction_features.csv")
    print("- data/featured/interaction_definitions.json")
    print("- data/featured/interaction_features_audit.json")


if __name__ == "__main__":
    main()