#!/usr/bin/env python3
"""
Phase 4 Task 2.2: High-Impact Interaction Features (Primary)

This module implements task 4.2.2 from TASKS.md:
- Create Study × Attendance Interaction
- Based on highest correlation pair in EDA (r = 0.67)

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

class Phase4InteractionFeatures:
    """
    Creates high-impact interaction features for Phase 4 feature engineering.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with the dataset.
        
        Args:
            df: DataFrame with derived features from previous tasks
        """
        self.df = df.copy()
        self.interaction_definitions = {}
        self.audit_log = []
        
    def create_study_attendance_interaction(self) -> pd.Series:
        """
        Create Study × Attendance Interaction feature.
        
        This is the highest correlation pair in EDA (r = 0.67) and expected primary predictor.
        
        Returns:
            Series containing the interaction term
        """
        logger.info("Creating Study × Attendance Interaction")
        
        # Check required columns
        required_cols = ['study_hours', 'attendance_rate']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Create interaction term
        interaction = self.df['study_hours'] * self.df['attendance_rate']
        
        # Handle edge cases
        interaction = interaction.fillna(0)  # Fill NaN with 0
        
        # Add to dataframe
        self.df['study_attendance_interaction'] = interaction
        
        # Record interaction definition
        self.interaction_definitions['study_attendance_interaction'] = {
            'description': 'Multiplicative interaction between study hours and attendance rate',
            'formula': 'study_hours * attendance_rate',
            'rationale': 'Highest correlation pair in EDA (r = 0.67), expected primary predictor',
            'interpretation': 'Higher values indicate both high study time AND high attendance',
            'source_columns': ['study_hours', 'attendance_rate'],
            'interaction_type': 'multiplicative',
            'created_by': 'Phase4InteractionFeatures.create_study_attendance_interaction'
        }
        
        # Calculate statistics
        stats = {
            'mean': interaction.mean(),
            'std': interaction.std(),
            'min': interaction.min(),
            'max': interaction.max(),
            'null_count': interaction.isnull().sum(),
            'zero_count': (interaction == 0).sum(),
            'correlation_with_target': None  # Will be calculated if target available
        }
        
        # Calculate correlation with target if available
        if 'final_test' in self.df.columns:
            stats['correlation_with_target'] = interaction.corr(self.df['final_test'])
            
        self.audit_log.append({
            'feature': 'study_attendance_interaction',
            'action': 'created',
            'statistics': stats
        })
        
        logger.info(f"Study × Attendance Interaction created - Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
        
        if stats['correlation_with_target'] is not None:
            logger.info(f"Correlation with target: {stats['correlation_with_target']:.3f}")
            
        return interaction
        
    def create_additional_primary_interactions(self) -> None:
        """
        Create additional high-impact interaction features based on EDA insights.
        
        These are secondary priority but still high-impact interactions.
        """
        logger.info("Creating additional primary interaction features")
        
        # Study Hours × Previous Score (academic momentum)
        if all(col in self.df.columns for col in ['study_hours', 'previous_score']):
            self.df['study_previous_interaction'] = self.df['study_hours'] * self.df['previous_score']
            
            self.interaction_definitions['study_previous_interaction'] = {
                'description': 'Interaction between current study effort and past performance',
                'formula': 'study_hours * previous_score',
                'rationale': 'Academic momentum - students with good past performance who study more',
                'interpretation': 'Higher values indicate sustained academic effort and performance',
                'source_columns': ['study_hours', 'previous_score'],
                'interaction_type': 'multiplicative',
                'created_by': 'Phase4InteractionFeatures.create_additional_primary_interactions'
            }
            
            logger.info("Created Study × Previous Score interaction")
            
        # Attendance × Previous Score (consistency effect)
        if all(col in self.df.columns for col in ['attendance_rate', 'previous_score']):
            self.df['attendance_previous_interaction'] = self.df['attendance_rate'] * self.df['previous_score']
            
            self.interaction_definitions['attendance_previous_interaction'] = {
                'description': 'Interaction between attendance consistency and past performance',
                'formula': 'attendance_rate * previous_score',
                'rationale': 'Consistency effect - reliable students with good past performance',
                'interpretation': 'Higher values indicate consistent high performers',
                'source_columns': ['attendance_rate', 'previous_score'],
                'interaction_type': 'multiplicative',
                'created_by': 'Phase4InteractionFeatures.create_additional_primary_interactions'
            }
            
            logger.info("Created Attendance × Previous Score interaction")
            
    def create_efficiency_interactions(self) -> None:
        """
        Create interactions with the derived efficiency score.
        """
        logger.info("Creating efficiency-based interactions")
        
        # Study Efficiency × Previous Score
        if all(col in self.df.columns for col in ['study_efficiency_score', 'previous_score']):
            self.df['efficiency_previous_interaction'] = (
                self.df['study_efficiency_score'] * self.df['previous_score']
            )
            
            self.interaction_definitions['efficiency_previous_interaction'] = {
                'description': 'Interaction between study efficiency and past performance',
                'formula': 'study_efficiency_score * previous_score',
                'rationale': 'Combines current efficiency with past success',
                'interpretation': 'Higher values indicate efficient students with good track record',
                'source_columns': ['study_efficiency_score', 'previous_score'],
                'interaction_type': 'multiplicative',
                'created_by': 'Phase4InteractionFeatures.create_efficiency_interactions'
            }
            
            logger.info("Created Study Efficiency × Previous Score interaction")
            
    def create_parent_education_socioeconomic_interaction(self) -> None:
        """
        Create Parent Education × Socioeconomic Status Interaction feature.
        This is a cross-categorical interaction based on EDA insights.
        """
        logger.info("Creating Parent Education × Socioeconomic Status Interaction")

        required_cols = ['parent_education_level', 'socioeconomic_status']
        missing_cols = [col for col in required_cols if col not in self.df.columns]

        if missing_cols:
            logger.warning(f"Missing required columns for Parent Education × Socioeconomic Interaction: {missing_cols}. Skipping feature creation.")
            return

        # Ensure columns are categorical for interaction
        # This might involve mapping to numerical representations if not already done
        # For simplicity, we'll assume they are numerically encoded or can be combined as strings
        # A more robust approach might involve target encoding or creating dummy variables first

        # Simple string concatenation for interaction - can be refined
        try:
            interaction_series = self.df['parent_education_level'].astype(str) + "_" + self.df['socioeconomic_status'].astype(str)
            # Convert to categorical and then to codes for a numerical representation
            self.df['parent_edu_socio_interaction'] = pd.Categorical(interaction_series).codes

            self.interaction_definitions['parent_edu_socio_interaction'] = {
                'description': 'Interaction between parent education level and socioeconomic status',
                'formula': 'parent_education_level (coded) + socioeconomic_status (coded)', # Simplified representation
                'rationale': 'EDA showed compound effect. Higher values represent combined higher status.',
                'interpretation': 'Categorical interaction representing combined parental background.',
                'source_columns': ['parent_education_level', 'socioeconomic_status'],
                'interaction_type': 'categorical_combination',
                'created_by': 'Phase4InteractionFeatures.create_parent_education_socioeconomic_interaction'
            }

            stats = {
                'unique_combinations': self.df['parent_edu_socio_interaction'].nunique(),
                'null_count': self.df['parent_edu_socio_interaction'].isnull().sum(),
                'correlation_with_target': None
            }

            if 'final_test' in self.df.columns:
                stats['correlation_with_target'] = self.df['parent_edu_socio_interaction'].corr(self.df['final_test'])

            self.audit_log.append({
                'feature': 'parent_edu_socio_interaction',
                'action': 'created',
                'statistics': stats
            })

            logger.info(f"Parent Education × Socioeconomic Interaction created - Unique combinations: {stats['unique_combinations']}")
            if stats['correlation_with_target'] is not None:
                logger.info(f"Correlation with target: {stats['correlation_with_target']:.3f}")

        except Exception as e:
            logger.error(f"Error creating Parent Education × Socioeconomic Interaction: {e}")

    def create_sleep_study_hours_interaction(self) -> None:
        """
        Create Sleep Hours × Study Hours Interaction feature.
        Rationale: Non-linear interaction for optimal study conditions.
        """
        logger.info("Creating Sleep Hours × Study Hours Interaction")

        # Assuming 'sleep_hours' is derived or available. If not, it needs to be created first.
        # For this example, let's assume 'sleep_hours' exists.
        # If 'sleep_hours' needs to be calculated from 'sleep_time' and 'wake_time', that logic
        # should ideally be in a prior step (e.g., derived features).
        # For now, we'll proceed assuming 'sleep_hours' is present.
        required_cols = ['sleep_hours', 'study_hours']
        missing_cols = [col for col in required_cols if col not in self.df.columns]

        if missing_cols:
            logger.warning(f"Missing required columns for Sleep × Study Hours Interaction: {missing_cols}. Skipping feature creation.")
            # Potentially, we could try to derive sleep_hours here if 'sleep_time' and 'wake_time' exist
            # and 'sleep_hours' is the only one missing.
            # Example: if 'sleep_hours' not in self.df.columns and all(c in self.df.columns for c in ['sleep_time', 'wake_time']):
            #   self.df['sleep_hours'] = (pd.to_datetime(self.df['wake_time'], format='%H:%M') - pd.to_datetime(self.df['sleep_time'], format='%H:%M')).dt.total_seconds() / 3600
            #   self.df['sleep_hours'] = self.df['sleep_hours'].apply(lambda x: x if x >= 0 else x + 24) # Handle overnight sleep
            #   logger.info("Derived 'sleep_hours' for interaction.")
            # else:
            #   return
            return

        try:
            interaction_series = self.df['sleep_hours'] * self.df['study_hours']
            self.df['sleep_study_interaction'] = interaction_series.fillna(0) # Fill NaN with 0

            self.interaction_definitions['sleep_study_interaction'] = {
                'description': 'Multiplicative interaction between sleep hours and study hours.',
                'formula': 'sleep_hours * study_hours',
                'rationale': 'Non-linear interaction for optimal study conditions. Captures synergy or trade-off.',
                'interpretation': 'Higher values suggest a balance or combined effect of sufficient sleep and study.',
                'source_columns': ['sleep_hours', 'study_hours'],
                'interaction_type': 'multiplicative',
                'created_by': 'Phase4InteractionFeatures.create_sleep_study_hours_interaction'
            }

            stats = {
                'mean': self.df['sleep_study_interaction'].mean(),
                'std': self.df['sleep_study_interaction'].std(),
                'min': self.df['sleep_study_interaction'].min(),
                'max': self.df['sleep_study_interaction'].max(),
                'null_count': self.df['sleep_study_interaction'].isnull().sum(),
                'correlation_with_target': None
            }

            if 'final_test' in self.df.columns:
                stats['correlation_with_target'] = self.df['sleep_study_interaction'].corr(self.df['final_test'])

            self.audit_log.append({
                'feature': 'sleep_study_interaction',
                'action': 'created',
                'statistics': stats
            })

            logger.info(f"Sleep × Study Hours Interaction created - Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
            if stats['correlation_with_target'] is not None:
                logger.info(f"Correlation with target: {stats['correlation_with_target']:.3f}")

        except Exception as e:
            logger.error(f"Error creating Sleep × Study Hours Interaction: {e}")

    def create_exercise_performance_interaction(self) -> None:
        """
        Create Exercise Hours × Previous Score Interaction feature.
        Rationale: Balance indicator from EDA insights.
        """
        logger.info("Creating Exercise Hours × Previous Score Interaction")

        required_cols = ['exercise_hours', 'previous_score']
        missing_cols = [col for col in required_cols if col not in self.df.columns]

        if missing_cols:
            logger.warning(f"Missing required columns for Exercise × Previous Score Interaction: {missing_cols}. Skipping feature creation.")
            return

        try:
            interaction_series = self.df['exercise_hours'] * self.df['previous_score']
            self.df['exercise_previous_score_interaction'] = interaction_series.fillna(0) # Fill NaN with 0

            self.interaction_definitions['exercise_previous_score_interaction'] = {
                'description': 'Multiplicative interaction between exercise hours and previous score.',
                'formula': 'exercise_hours * previous_score',
                'rationale': 'Balance indicator from EDA. Explores if exercise moderates academic performance.',
                'interpretation': 'Higher values may indicate a beneficial balance or a trade-off depending on context.',
                'source_columns': ['exercise_hours', 'previous_score'],
                'interaction_type': 'multiplicative',
                'created_by': 'Phase4InteractionFeatures.create_exercise_performance_interaction'
            }

            stats = {
                'mean': self.df['exercise_previous_score_interaction'].mean(),
                'std': self.df['exercise_previous_score_interaction'].std(),
                'min': self.df['exercise_previous_score_interaction'].min(),
                'max': self.df['exercise_previous_score_interaction'].max(),
                'null_count': self.df['exercise_previous_score_interaction'].isnull().sum(),
                'correlation_with_target': None
            }

            if 'final_test' in self.df.columns:
                stats['correlation_with_target'] = self.df['exercise_previous_score_interaction'].corr(self.df['final_test'])

            self.audit_log.append({
                'feature': 'exercise_previous_score_interaction',
                'action': 'created',
                'statistics': stats
            })

            logger.info(f"Exercise × Previous Score Interaction created - Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
            if stats['correlation_with_target'] is not None:
                logger.info(f"Correlation with target: {stats['correlation_with_target']:.3f}")

        except Exception as e:
            logger.error(f"Error creating Exercise × Previous Score Interaction: {e}")

    def create_transport_attendance_interaction(self) -> None:
        """
        Create Transport Mode × Attendance Rate Interaction feature.
        Rationale: Accessibility impact on consistent attendance.
        """
        logger.info("Creating Transport Mode × Attendance Rate Interaction")

        required_cols = ['transport_mode', 'attendance_rate']
        missing_cols = [col for col in required_cols if col not in self.df.columns]

        if missing_cols:
            logger.warning(f"Missing required columns for Transport × Attendance Interaction: {missing_cols}. Skipping feature creation.")
            return
        
        # This interaction might be more meaningful if transport_mode is numerically encoded 
        # (e.g., based on reliability or travel time if such data exists or can be inferred).
        # For now, we'll create a simple interaction. If 'transport_mode' is categorical,
        # this will result in multiple interaction terms if one-hot encoded first, or a 
        # combined categorical feature. Let's try a different approach: interaction with a 
        # binarized version of transport (e.g. Public vs Private) or by creating 
        # mean attendance per transport mode if that makes sense.
        # For simplicity, let's assume 'transport_mode' can be mapped to some ordinal scale or risk factor.
        # Example: {'Walk': 0, 'Bike': 0, 'Public': 1, 'Private': 0.5} - lower is better/less risky for attendance
        # This mapping should be based on EDA or domain knowledge.
        # If no such mapping is readily available, a simple multiplication might not be ideal if 'transport_mode' is purely nominal.

        # Let's proceed with a placeholder for a more complex encoding if needed.
        # We can create an interaction term by first converting transport_mode to numerical if it's not.
        # If 'transport_mode' is categorical, we might need to encode it first.
        # For this example, we'll assume 'transport_mode' is or can be made numeric for interaction.
        # A simple approach: if 'transport_mode' is categorical, create dummy variables and interact each with attendance.
        # Or, create a risk score from transport_mode.

        try:
            # Attempting a simple interaction. This assumes 'transport_mode' is numeric or can be treated as such.
            # If 'transport_mode' is categorical like 'Walk', 'Car', 'Bus', this direct multiplication is not meaningful.
            # It would be better to: 
            # 1. One-hot encode transport_mode: transport_Walk, transport_Car, transport_Bus
            # 2. Interact each with attendance_rate: transport_Walk_x_attendance, transport_Car_x_attendance, ...
            # For now, let's create a placeholder interaction. The user might need to refine this based on the nature of 'transport_mode'.
            
            # Let's assume 'transport_mode' has been preprocessed to be numeric (e.g. a risk score)
            # If not, this will raise an error or produce meaningless results.
            if pd.api.types.is_numeric_dtype(self.df['transport_mode']):
                interaction_series = self.df['transport_mode'] * self.df['attendance_rate']
                feature_name = 'transport_x_attendance_interaction'
                description = 'Multiplicative interaction between (numeric) transport mode and attendance rate.'
                formula = 'transport_mode_numeric * attendance_rate'
            else:
                # If transport_mode is categorical, we can create a combined categorical feature
                # or skip if a numeric interaction is strictly expected.
                # For this example, let's log a warning and skip if not numeric, as per task description 'Pandas operations'.
                logger.warning("'transport_mode' is not numeric. Skipping direct multiplicative interaction for 'transport_x_attendance_interaction'. Consider encoding 'transport_mode' first.")
                return

            self.df[feature_name] = interaction_series.fillna(0)

            self.interaction_definitions[feature_name] = {
                'description': description,
                'formula': formula,
                'rationale': 'Accessibility impact on consistent attendance. Numeric interaction assumes transport_mode is ordered/scalar.',
                'interpretation': 'Interaction effect of transport mode on attendance.',
                'source_columns': ['transport_mode', 'attendance_rate'],
                'interaction_type': 'multiplicative_conditional_numeric',
                'created_by': 'Phase4InteractionFeatures.create_transport_attendance_interaction'
            }

            stats = {
                'mean': self.df[feature_name].mean(),
                'std': self.df[feature_name].std(),
                'min': self.df[feature_name].min(),
                'max': self.df[feature_name].max(),
                'null_count': self.df[feature_name].isnull().sum(),
                'correlation_with_target': None
            }

            if 'final_test' in self.df.columns:
                stats['correlation_with_target'] = self.df[feature_name].corr(self.df['final_test'])

            self.audit_log.append({
                'feature': feature_name,
                'action': 'created',
                'statistics': stats
            })

            logger.info(f"Transport × Attendance Interaction ('{feature_name}') created - Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
            if stats['correlation_with_target'] is not None:
                logger.info(f"Correlation with target: {stats['correlation_with_target']:.3f}")

        except Exception as e:
            logger.error(f"Error creating Transport × Attendance Interaction: {e}")
            
    def validate_interaction_features(self) -> Dict[str, Any]:
        """
        Validate the created interaction features.
        
        Returns:
            Dictionary containing validation results
        """
        logger.info("Validating interaction features")
        
        validation_results = {}
        
        for feature_name in self.interaction_definitions.keys():
            if feature_name in self.df.columns:
                feature_series = self.df[feature_name]
                
                validation_results[feature_name] = {
                    'no_nulls_after_creation': feature_series.isnull().sum() == 0,
                    'has_variance': feature_series.std() > 0,
                    'no_infinite_values': not np.isinf(feature_series).any(),
                    'reasonable_range': self._check_reasonable_range(feature_series, feature_name)
                }
                
                # Check correlation with components
                definition = self.interaction_definitions[feature_name]
                source_cols = definition['source_columns']
                
                if all(col in self.df.columns for col in source_cols):
                    correlations = {}
                    for col in source_cols:
                        correlations[col] = feature_series.corr(self.df[col])
                    validation_results[feature_name]['correlations_with_components'] = correlations
                    
        # Overall validation
        all_valid = all(
            all(checks.values()) if isinstance(checks, dict) else checks
            for feature_checks in validation_results.values()
            for checks in feature_checks.values()
            if isinstance(checks, (dict, bool))
        )
        
        validation_results['overall_valid'] = all_valid
        
        if all_valid:
            logger.info("✓ All interaction features validation PASSED")
        else:
            logger.warning("✗ Some interaction features validation FAILED")
            
        return validation_results
        
    def _check_reasonable_range(self, series: pd.Series, feature_name: str) -> bool:
        """
        Check if the feature values are in a reasonable range.
        
        Args:
            series: The feature series to check
            feature_name: Name of the feature
            
        Returns:
            True if range is reasonable, False otherwise
        """
        # Basic checks for reasonable ranges
        min_val, max_val = series.min(), series.max()
        
        # Should not have extreme outliers (more than 5 standard deviations)
        mean_val, std_val = series.mean(), series.std()
        
        if std_val > 0:
            z_scores = np.abs((series - mean_val) / std_val)
            extreme_outliers = (z_scores > 5).sum()
            
            # Allow some outliers but not too many
            outlier_ratio = extreme_outliers / len(series)
            return outlier_ratio < 0.01  # Less than 1% extreme outliers
            
        return True
        
    def get_interaction_summary(self) -> Dict[str, Any]:
        """
        Get summary of created interaction features.
        
        Returns:
            Dictionary containing interaction summary
        """
        summary = {
            'interactions_created': list(self.interaction_definitions.keys()),
            'interaction_definitions': self.interaction_definitions,
            'audit_log': self.audit_log,
            'dataframe_shape': self.df.shape
        }
        
        return summary
        
    def save_interaction_features(self, output_path: str = "data/featured/interaction_features.csv") -> None:
        """
        Save the dataframe with interaction features.
        
        Args:
            output_path: Path to save the enhanced dataset
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.df.to_csv(output_path)
        logger.info(f"Interaction features dataset saved to {output_path}")
        
    def save_interaction_documentation(self, output_path: str = "data/featured/interaction_definitions.json") -> None:
        """
        Save interaction definitions and audit log.
        
        Args:
            output_path: Path to save the documentation
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        documentation = self.get_interaction_summary()
        
        with open(output_path, 'w') as f:
            json.dump(documentation, f, indent=2, default=str)
            
        logger.info(f"Interaction documentation saved to {output_path}")


def main():
    """
    Main function to run Phase 4 Task 2.2: High-Impact Interaction Features.
    """
    try:
        # Load data with derived features (assuming previous tasks completed)
        derived_features_path = "data/featured/derived_features.csv"
        
        if not Path(derived_features_path).exists():
            raise FileNotFoundError(
                f"Derived features file not found: {derived_features_path}. "
                "Please run Phase 4 Task 2.1 first."
            )
            
        df = pd.read_csv(derived_features_path, index_col=0)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        
        # Create interaction features
        interaction_creator = Phase4InteractionFeatures(df)
        
        # Create primary interaction (Study × Attendance)
        interaction_creator.create_study_attendance_interaction()
        
        # Create additional primary interactions
        interaction_creator.create_additional_primary_interactions()
        
        # Create efficiency-based interactions
        interaction_creator.create_efficiency_interactions()
        interaction_creator.create_parent_education_socioeconomic_interaction()
        interaction_creator.create_sleep_study_hours_interaction()
        interaction_creator.create_exercise_performance_interaction()
        interaction_creator.create_transport_attendance_interaction() # Add call to new method

        # Validate features
        validation_results = interaction_creator.validate_interaction_features()
        
        # Save results
        interaction_creator.save_interaction_features()
        interaction_creator.save_interaction_documentation()
        
        # Print summary
        summary = interaction_creator.get_interaction_summary()
        print(f"\n=== Phase 4 Task 2.2 Complete ===")
        print(f"Interactions created: {summary['interactions_created']}")
        print(f"Dataset shape: {summary['dataframe_shape']}")
        print(f"Validation passed: {validation_results['overall_valid']}")
        
        return interaction_creator.df
        
    except Exception as e:
        logger.error(f"Phase 4 Task 2.2 failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()