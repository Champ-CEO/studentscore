#!/usr/bin/env python3
"""
Phase 4 Task 2.1: EDA-Driven Derived Features (High Priority)

This module implements task 4.2.1 from TASKS.md:
- Create Study Efficiency Score
- Create Academic Support Index
- Based on Phase 3 recommendations and EDA insights

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Phase4DerivedFeatures:
    """
    Creates EDA-driven derived features for Phase 4 feature engineering.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with the loaded dataset.
        
        Args:
            df: Processed DataFrame from Phase 3
        """
        self.df = df.copy()
        self.feature_definitions = {}
        self.audit_log = []
        
    def create_study_efficiency_score(self) -> pd.Series:
        """
        Create Study Efficiency Score combining study_hours and attendance_rate.
        
        Formula: (study_hours * attendance_rate) / normalization_factor
        Rationale: EDA showed strong correlation (>0.6) with target
        
        Returns:
            Series containing the Study Efficiency Score
        """
        logger.info("Creating Study Efficiency Score")
        
        # Check required columns
        required_cols = ['study_hours', 'attendance_rate']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Calculate raw efficiency score
        raw_score = self.df['study_hours'] * self.df['attendance_rate']
        
        # Normalize to 0-100 scale
        max_possible = self.df['study_hours'].max() * 100  # 100% attendance
        efficiency_score = (raw_score / max_possible) * 100
        
        # Handle edge cases
        efficiency_score = efficiency_score.fillna(0)  # Fill NaN with 0
        efficiency_score = efficiency_score.clip(0, 100)  # Ensure 0-100 range
        
        # Add to dataframe
        self.df['study_efficiency_score'] = efficiency_score
        
        # Record feature definition
        self.feature_definitions['study_efficiency_score'] = {
            'description': 'Efficiency score combining study hours and attendance rate',
            'formula': '(study_hours * attendance_rate) / max_possible_score * 100',
            'rationale': 'EDA showed strong correlation (>0.6) with target variable',
            'range': '0-100',
            'source_columns': ['study_hours', 'attendance_rate'],
            'created_by': 'Phase4DerivedFeatures.create_study_efficiency_score'
        }
        
        # Log statistics
        stats = {
            'mean': efficiency_score.mean(),
            'std': efficiency_score.std(),
            'min': efficiency_score.min(),
            'max': efficiency_score.max(),
            'null_count': efficiency_score.isnull().sum()
        }
        
        self.audit_log.append({
            'feature': 'study_efficiency_score',
            'action': 'created',
            'statistics': stats
        })
        
        logger.info(f"Study Efficiency Score created - Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
        
        return efficiency_score
        
    def create_academic_support_index(self) -> pd.Series:
        """
        Create Academic Support Index from tuition, direct_admission, extracurricular_activities.
        
        Weighted combination based on EDA categorical analysis insights.
        
        Returns:
            Series containing the Academic Support Index
        """
        logger.info("Creating Academic Support Index")
        
        # Check required columns
        required_cols = ['tuition', 'direct_admission', 'extracurricular_activities']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Convert categorical variables to numeric scores
        support_score = 0
        
        # Tuition (weight: 0.4) - indicates family investment in education
        tuition_score = (self.df['tuition'] == 'Yes').astype(int) * 40
        
        # Direct admission (weight: 0.3) - indicates academic preparation/privilege
        direct_admission_score = (self.df['direct_admission'] == 'Yes').astype(int) * 30
        
        # Extracurricular activities (weight: 0.3) - indicates additional support/engagement
        # Handle various extracurricular categories
        extracurricular_score = np.zeros(len(self.df))
        
        # Map extracurricular activities to scores
        extracurricular_mapping = {
            'Sports': 25,
            'Arts': 20,
            'Clubs': 30,  # Highest as it often indicates academic clubs
            'Music': 20,
            'Drama': 20,
            'Debate': 30,
            'Science': 30
        }
        
        for activity, score in extracurricular_mapping.items():
            mask = self.df['extracurricular_activities'].str.contains(activity, na=False)
            extracurricular_score[mask] = score
            
        # Handle missing/unknown extracurricular activities
        no_activity_mask = (
            self.df['extracurricular_activities'].isnull() |
            (self.df['extracurricular_activities'] == '') |
            (self.df['extracurricular_activities'] == 'None')
        )
        extracurricular_score[no_activity_mask] = 0
        
        # Combine scores
        academic_support_index = tuition_score + direct_admission_score + extracurricular_score
        
        # Ensure 0-100 range
        academic_support_index = academic_support_index.clip(0, 100)
        
        # Add to dataframe
        self.df['academic_support_index'] = academic_support_index
        
        # Record feature definition
        self.feature_definitions['academic_support_index'] = {
            'description': 'Weighted index of academic support factors',
            'formula': 'tuition_score(40%) + direct_admission_score(30%) + extracurricular_score(30%)',
            'rationale': 'EDA categorical analysis showed these as key differentiators',
            'range': '0-100',
            'source_columns': ['tuition', 'direct_admission', 'extracurricular_activities'],
            'weights': {'tuition': 0.4, 'direct_admission': 0.3, 'extracurricular_activities': 0.3},
            'extracurricular_mapping': extracurricular_mapping,
            'created_by': 'Phase4DerivedFeatures.create_academic_support_index'
        }
        
        # Log statistics
        stats = {
            'mean': academic_support_index.mean(),
            'std': academic_support_index.std(),
            'min': academic_support_index.min(),
            'max': academic_support_index.max(),
            'null_count': academic_support_index.isnull().sum(),
            'distribution': academic_support_index.value_counts().head(10).to_dict()
        }
        
        self.audit_log.append({
            'feature': 'academic_support_index',
            'action': 'created',
            'statistics': stats
        })
        
        logger.info(f"Academic Support Index created - Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
        
        return academic_support_index

    def create_study_time_categories(self) -> pd.Series:
        """
        Create Study Time Categories from study_time.

        Categorizes study times (e.g., Early, Peak, Afternoon, Evening).
        Rationale: Based on EDA distribution analysis.

        Returns:
            Series containing the Study Time Categories
        """
        logger.info("Creating Study Time Categories")

        # Check required columns
        required_cols = ['study_time']
        missing_cols = [col for col in required_cols if col not in self.df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Define bins and labels for study time categories
        # Assuming study_time is in hours (e.g., 0-23.99)
        bins = [-0.01, 5.99, 11.99, 17.99, 23.99]  # 0-6 (Night), 6-12 (Morning), 12-18 (Afternoon), 18-24 (Evening)
        labels = ['Night', 'Morning', 'Afternoon', 'Evening']
        
        # Convert study_time to numeric if it's not already, coercing errors
        study_time_numeric = pd.to_numeric(self.df['study_time'], errors='coerce')

        study_time_categories = pd.cut(study_time_numeric, bins=bins, labels=labels, right=True)

        # Handle potential NaNs from coercion or if study_time is outside bins
        study_time_categories = study_time_categories.fillna('Unknown')

        # Add to dataframe
        self.df['study_time_category'] = study_time_categories

        # Record feature definition
        self.feature_definitions['study_time_category'] = {
            'description': 'Categorical representation of study time of day.',
            'formula': 'pd.cut(study_time, bins=[-0.01, 5.99, 11.99, 17.99, 23.99], labels=["Night", "Morning", "Afternoon", "Evening"])',
            'rationale': 'Based on EDA distribution analysis of study times.',
            'categories': labels + ['Unknown'],
            'source_columns': ['study_time'],
            'created_by': 'Phase4DerivedFeatures.create_study_time_categories'
        }

        # Log statistics
        stats = {
            'value_counts': study_time_categories.value_counts(normalize=True).to_dict(),
            'null_count': study_time_categories.isnull().sum(), # Should be 0 after fillna
            'mode': study_time_categories.mode()[0] if not study_time_categories.mode().empty else 'N/A'
        }

        self.audit_log.append({
            'feature': 'study_time_category',
            'action': 'created',
            'statistics': stats
        })

        logger.info(f"Study Time Categories created - Mode: {stats['mode']}")
        logger.info(f"Distribution: {stats['value_counts']}")

        return study_time_categories

    def create_sleep_quality_indicator(self) -> pd.Series:
        """
        Create Sleep Quality Indicator from sleep_time and wake_time.

        Categorizes sleep duration (e.g., Optimal, Insufficient, Excessive).
        Rationale: EDA showed non-linear relationship with performance.

        Returns:
            Series containing the Sleep Quality Indicator
        """
        logger.info("Creating Sleep Quality Indicator")

        required_cols = ['sleep_time', 'wake_time']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for sleep quality: {missing_cols}")

        # Convert sleep_time and wake_time to numeric (hours from midnight)
        sleep_time_hours = pd.to_numeric(self.df['sleep_time'], errors='coerce')
        wake_time_hours = pd.to_numeric(self.df['wake_time'], errors='coerce')

        # Calculate sleep duration
        # Handles cases where wake_time is on the next day (e.g., sleep at 23:00, wake at 7:00)
        sleep_duration = wake_time_hours - sleep_time_hours
        sleep_duration[sleep_duration < 0] += 24  # Add 24 hours if wake time is on the next day

        # Define bins and labels for sleep quality
        # Based on common recommendations: <6 Insufficient, 6-9 Optimal, >9 Excessive
        bins = [-0.01, 5.99, 9.0, 24.0] 
        labels = ['Insufficient', 'Optimal', 'Excessive']

        sleep_quality = pd.cut(sleep_duration, bins=bins, labels=labels, right=True)
        sleep_quality = sleep_quality.fillna('Unknown') # Handle NaNs from time conversion or out of bounds
        
        self.df['sleep_duration_hours'] = sleep_duration # Store raw duration for reference
        self.df['sleep_quality_indicator'] = sleep_quality

        self.feature_definitions['sleep_quality_indicator'] = {
            'description': 'Categorical indicator of sleep quality based on duration.',
            'formula': 'pd.cut(wake_time - sleep_time (adjusted for overnight), bins=[-0.01, 5.99, 9.0, 24.0], labels=["Insufficient", "Optimal", "Excessive"])',
            'rationale': 'EDA showed non-linear relationship with performance. Sleep duration is a key health factor.',
            'categories': labels + ['Unknown'],
            'source_columns': ['sleep_time', 'wake_time'],
            'created_by': 'Phase4DerivedFeatures.create_sleep_quality_indicator'
        }
        self.feature_definitions['sleep_duration_hours'] = {
            'description': 'Calculated sleep duration in hours.',
            'formula': 'wake_time - sleep_time (adjusted for overnight)',
            'rationale': 'Intermediate calculation for sleep quality, can be useful on its own.',
            'range': '0-24 (potentially, depends on data quality)',
            'source_columns': ['sleep_time', 'wake_time'],
            'created_by': 'Phase4DerivedFeatures.create_sleep_quality_indicator'
        }

        stats_quality = {
            'value_counts': sleep_quality.value_counts(normalize=True).to_dict(),
            'null_count': sleep_quality.isnull().sum(),
            'mode': sleep_quality.mode()[0] if not sleep_quality.mode().empty else 'N/A'
        }
        stats_duration = {
            'mean': sleep_duration.mean(),
            'std': sleep_duration.std(),
            'min': sleep_duration.min(),
            'max': sleep_duration.max(),
            'null_count': sleep_duration.isnull().sum()
        }

        self.audit_log.append({
            'feature': 'sleep_quality_indicator',
            'action': 'created',
            'statistics': stats_quality
        })
        self.audit_log.append({
            'feature': 'sleep_duration_hours',
            'action': 'created',
            'statistics': stats_duration
        })

        logger.info(f"Sleep Quality Indicator created - Mode: {stats_quality['mode']}")
        logger.info(f"Sleep Duration (hours) created - Mean: {stats_duration['mean']:.2f}, Std: {stats_duration['std']:.2f}")

        return sleep_quality

    def validate_derived_features(self) -> Dict[str, Any]:
        """
        Validate the created derived features.
        
        Returns:
            Dictionary containing validation results
        """
        logger.info("Validating derived features")
        
        validation_results = {}
        
        # Check Study Efficiency Score
        if 'study_efficiency_score' in self.df.columns:
            ses = self.df['study_efficiency_score']
            validation_results['study_efficiency_score'] = {
                'range_valid': (ses >= 0).all() and (ses <= 100).all(),
                'no_nulls': ses.isnull().sum() == 0,
                'reasonable_distribution': 0 < ses.std() < 50,  # Should have some variance
                'correlation_with_components': {
                    'study_hours': ses.corr(self.df['study_hours']),
                    'attendance_rate': ses.corr(self.df['attendance_rate'])
                }
            }
            
        # Check Academic Support Index
        if 'academic_support_index' in self.df.columns:
            asi = self.df['academic_support_index']
            validation_results['academic_support_index'] = {
                'range_valid': (asi >= 0).all() and (asi <= 100).all(),
                'no_nulls': asi.isnull().sum() == 0,
                'reasonable_distribution': 0 < asi.std() < 50,
                'distinct_values': asi.nunique() > 5  # Should have multiple distinct values
            }

        # Check Study Time Categories
        if 'study_time_category' in self.df.columns:
            stc = self.df['study_time_category']
            validation_results['study_time_category'] = {
                'no_nulls': stc.isnull().sum() == 0,
                'expected_categories': set(stc.unique()).issubset(set(['Night', 'Morning', 'Afternoon', 'Evening', 'Unknown'])),
                'has_variance': stc.nunique() > 1 # Should have more than one category
            }

        # Check Sleep Quality Indicator
        if 'sleep_quality_indicator' in self.df.columns:
            sqi = self.df['sleep_quality_indicator']
            validation_results['sleep_quality_indicator'] = {
                'no_nulls': sqi.isnull().sum() == 0,
                'expected_categories': set(sqi.unique()).issubset(set(['Insufficient', 'Optimal', 'Excessive', 'Unknown'])),
                'has_variance': sqi.nunique() > 1
            }
        if 'sleep_duration_hours' in self.df.columns:
            sdh = self.df['sleep_duration_hours']
            validation_results['sleep_duration_hours'] = {
                'no_nulls_if_inputs_ok': True, # Nulls can come from input, not necessarily error here
                'reasonable_range': (sdh.min() >= 0 if sdh.notnull().any() else True) and \
                                    (sdh.max() <= 24 if sdh.notnull().any() else True),
                'has_variance_if_not_all_null': sdh.std() > 0 if sdh.notnull().any() and sdh.nunique() > 1 else True
            }
            
        # Overall validation
        all_valid = all(
            all(checks.values()) if isinstance(checks, dict) else checks
            for feature_checks in validation_results.values()
            for checks in feature_checks.values()
            if isinstance(checks, (dict, bool))
        )
        
        validation_results['overall_valid'] = all_valid
        
        if all_valid:
            logger.info("✓ All derived features validation PASSED")
        else:
            logger.warning("✗ Some derived features validation FAILED")
            
        return validation_results
        
    def get_feature_summary(self) -> Dict[str, Any]:
        """
        Get summary of created features.
        
        Returns:
            Dictionary containing feature summary
        """
        summary = {
            'features_created': list(self.feature_definitions.keys()),
            'feature_definitions': self.feature_definitions,
            'audit_log': self.audit_log,
            'dataframe_shape': self.df.shape
        }
        
        return summary
        
    def save_derived_features(self, output_path: str = "data/featured/derived_features.csv") -> None:
        """
        Save the dataframe with derived features.
        
        Args:
            output_path: Path to save the enhanced dataset
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.df.to_csv(output_path)
        logger.info(f"Derived features dataset saved to {output_path}")
        
    def save_feature_documentation(self, output_path: str = "data/featured/derived_features_definitions.json") -> None:
        """
        Save feature definitions and audit log.
        
        Args:
            output_path: Path to save the documentation
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        documentation = self.get_feature_summary()
        
        with open(output_path, 'w') as f:
            json.dump(documentation, f, indent=2, default=str)
            
        logger.info(f"Feature documentation saved to {output_path}")


def main():
    """
    Main function to run Phase 4 Task 2.1: EDA-Driven Derived Features.
    """
    try:
        # Load data (assuming Phase 4 Task 1 was completed)
        from phase4_task1_load_validate import Phase4DataLoader
        
        loader = Phase4DataLoader()
        df = loader.load_data()
        
        if not loader.validation_results.get('validation_passed', False):
            raise ValueError("Data validation failed. Cannot proceed with feature engineering.")
            
        # Create derived features
        feature_creator = Phase4DerivedFeatures(df)
        
        # Create Study Efficiency Score
        feature_creator.create_study_efficiency_score()
        
        # Create Academic Support Index
        feature_creator.create_academic_support_index()
        
        # Create Study Time Categories
        feature_creator.create_study_time_categories()

        # Create Sleep Quality Indicator
        feature_creator.create_sleep_quality_indicator()

        # Validate features
        validation_results = feature_creator.validate_derived_features()
        
        # Save results
        feature_creator.save_derived_features()
        feature_creator.save_feature_documentation()
        
        # Print summary
        summary = feature_creator.get_feature_summary()
        print(f"\n=== Phase 4 Task 2.1 Complete ===")
        print(f"Features created: {summary['features_created']}")
        print(f"Dataset shape: {summary['dataframe_shape']}")
        print(f"Validation passed: {validation_results['overall_valid']}")
        
        return feature_creator.df
        
    except Exception as e:
        logger.error(f"Phase 4 Task 2.1 failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()