#!/usr/bin/env python3
"""
Phase 4 Task 2.3: Distribution-Based Transformations (High Priority)

This module implements task 4.2.3 from TASKS.md:
- Transform right-skewed variables (study_hours, previous_score)
- Apply Log transformation for study_hours (skewness = 1.2)
- Apply Box-Cox transformation for previous_score (skewness = 0.8)

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
from scipy import stats
from sklearn.preprocessing import PowerTransformer, FunctionTransformer
import warnings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Phase4Transformations:
    """
    Applies distribution-based transformations for Phase 4 feature engineering.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with the dataset.
        
        Args:
            df: DataFrame with interaction features from previous tasks
        """
        self.df = df.copy()
        self.transformation_definitions = {}
        self.transformers = {}  # Store fitted transformers for inverse transform if needed
        self.audit_log = []
        
    def analyze_skewness(self, columns: List[str] = None) -> Dict[str, float]:
        """
        Analyze skewness of numerical columns.
        
        Args:
            columns: List of columns to analyze. If None, analyze all numerical columns.
            
        Returns:
            Dictionary mapping column names to skewness values
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
        skewness_results = {}
        
        for col in columns:
            if col in self.df.columns:
                # Remove NaN values for skewness calculation
                clean_data = self.df[col].dropna()
                if len(clean_data) > 0:
                    skewness_results[col] = stats.skew(clean_data)
                    
        logger.info("Skewness analysis completed")
        for col, skew_val in skewness_results.items():
            if abs(skew_val) > 0.5:
                logger.info(f"{col}: {skew_val:.3f} {'(right-skewed)' if skew_val > 0 else '(left-skewed)'}")
                
        return skewness_results
        
    def transform_study_hours_log(self) -> pd.Series:
        """
        Apply log transformation to study_hours (skewness = 1.2).
        
        Returns:
            Series containing the log-transformed study_hours
        """
        logger.info("Applying log transformation to study_hours")
        
        if 'study_hours' not in self.df.columns:
            raise ValueError("study_hours column not found")
            
        original_data = self.df['study_hours'].copy()
        
        # Check for non-positive values
        non_positive_count = (original_data <= 0).sum()
        if non_positive_count > 0:
            logger.warning(f"Found {non_positive_count} non-positive values in study_hours")
            
        # Apply log1p transformation (log(1 + x)) to handle zeros
        transformed_data = np.log1p(original_data.fillna(0))
        
        # Handle NaN values in original data
        transformed_data[original_data.isnull()] = np.nan
        
        # Add transformed column
        self.df['study_hours_log'] = transformed_data
        
        # Calculate skewness before and after
        original_skew = stats.skew(original_data.dropna())
        transformed_skew = stats.skew(transformed_data.dropna())
        
        # Record transformation definition
        self.transformation_definitions['study_hours_log'] = {
            'description': 'Log transformation of study_hours to reduce right skewness',
            'transformation': 'log1p(study_hours)',
            'rationale': 'Original skewness = 1.2, identified in EDA as right-skewed',
            'original_column': 'study_hours',
            'original_skewness': original_skew,
            'transformed_skewness': transformed_skew,
            'improvement': abs(original_skew) - abs(transformed_skew),
            'method': 'numpy.log1p',
            'created_by': 'Phase4Transformations.transform_study_hours_log'
        }
        
        # Log statistics
        stats_dict = {
            'original_skewness': original_skew,
            'transformed_skewness': transformed_skew,
            'skewness_improvement': abs(original_skew) - abs(transformed_skew),
            'null_count': transformed_data.isnull().sum(),
            'mean': transformed_data.mean(),
            'std': transformed_data.std()
        }
        
        self.audit_log.append({
            'feature': 'study_hours_log',
            'action': 'log_transformation',
            'statistics': stats_dict
        })
        
        logger.info(f"Log transformation completed - Skewness: {original_skew:.3f} → {transformed_skew:.3f}")
        
        return transformed_data
        
    def transform_previous_score_boxcox(self) -> pd.Series:
        """
        Apply Box-Cox transformation to previous_score (skewness = 0.8).
        
        Returns:
            Series containing the Box-Cox transformed previous_score
        """
        logger.info("Applying Box-Cox transformation to previous_score")
        
        if 'previous_score' not in self.df.columns:
            raise ValueError("previous_score column not found")
            
        original_data = self.df['previous_score'].copy()
        
        # Remove NaN values for transformation
        clean_data = original_data.dropna()
        
        if len(clean_data) == 0:
            raise ValueError("No valid data in previous_score column")
            
        # Check for non-positive values
        non_positive_count = (clean_data <= 0).sum()
        if non_positive_count > 0:
            logger.warning(f"Found {non_positive_count} non-positive values in previous_score")
            # Add small constant to make all values positive
            min_val = clean_data.min()
            if min_val <= 0:
                shift_amount = abs(min_val) + 1
                clean_data = clean_data + shift_amount
                logger.info(f"Shifted data by {shift_amount} to ensure positive values")
            else:
                shift_amount = 0
        else:
            shift_amount = 0
            
        # Apply Box-Cox transformation
        try:
            # Use PowerTransformer which handles Box-Cox internally
            transformer = PowerTransformer(method='box-cox', standardize=False)
            
            # Fit and transform
            transformed_clean = transformer.fit_transform(clean_data.values.reshape(-1, 1)).flatten()
            
            # Store transformer for potential inverse transform
            self.transformers['previous_score_boxcox'] = {
                'transformer': transformer,
                'shift_amount': shift_amount
            }
            
            # Create full transformed series
            transformed_data = pd.Series(index=original_data.index, dtype=float)
            transformed_data[clean_data.index] = transformed_clean
            transformed_data[original_data.isnull()] = np.nan
            
        except Exception as e:
            logger.warning(f"Box-Cox transformation failed: {str(e)}. Falling back to log transformation.")
            # Fallback to log transformation
            transformed_data = np.log1p(clean_data)
            transformed_full = pd.Series(index=original_data.index, dtype=float)
            transformed_full[clean_data.index] = transformed_data
            transformed_full[original_data.isnull()] = np.nan
            transformed_data = transformed_full
            
            self.transformers['previous_score_boxcox'] = {
                'transformer': 'log1p_fallback',
                'shift_amount': shift_amount
            }
            
        # Add transformed column
        self.df['previous_score_boxcox'] = transformed_data
        
        # Calculate skewness before and after
        original_skew = stats.skew(clean_data)
        transformed_skew = stats.skew(transformed_data.dropna())
        
        # Record transformation definition
        self.transformation_definitions['previous_score_boxcox'] = {
            'description': 'Box-Cox transformation of previous_score to reduce right skewness',
            'transformation': 'box-cox(previous_score + shift)',
            'rationale': 'Original skewness = 0.8, identified in EDA as right-skewed',
            'original_column': 'previous_score',
            'original_skewness': original_skew,
            'transformed_skewness': transformed_skew,
            'improvement': abs(original_skew) - abs(transformed_skew),
            'shift_amount': shift_amount,
            'method': 'sklearn.PowerTransformer(box-cox)',
            'created_by': 'Phase4Transformations.transform_previous_score_boxcox'
        }
        
        # Log statistics
        stats_dict = {
            'original_skewness': original_skew,
            'transformed_skewness': transformed_skew,
            'skewness_improvement': abs(original_skew) - abs(transformed_skew),
            'null_count': transformed_data.isnull().sum(),
            'mean': transformed_data.mean(),
            'std': transformed_data.std(),
            'shift_amount': shift_amount
        }
        
        self.audit_log.append({
            'feature': 'previous_score_boxcox',
            'action': 'boxcox_transformation',
            'statistics': stats_dict
        })
        
        logger.info(f"Box-Cox transformation completed - Skewness: {original_skew:.3f} → {transformed_skew:.3f}")
        
        return transformed_data
        
    def apply_additional_transformations(self) -> None:
        """
        Apply transformations to other skewed features if needed.
        """
        logger.info("Checking for additional features requiring transformation")
        
        # Analyze skewness of all numerical features
        skewness_results = self.analyze_skewness()
        
        # Define threshold for transformation
        skewness_threshold = 0.75
        
        for col, skew_val in skewness_results.items():
            # Skip already transformed columns and target variable
            if col in ['study_hours', 'previous_score', 'final_test'] or '_log' in col or '_boxcox' in col:
                continue
                
            if abs(skew_val) > skewness_threshold:
                logger.info(f"Applying transformation to {col} (skewness: {skew_val:.3f})")
                
                try:
                    if skew_val > 0:  # Right-skewed
                        # Try log transformation first
                        clean_data = self.df[col].dropna()
                        if (clean_data > 0).all():
                            transformed = np.log1p(clean_data)
                            transformed_full = pd.Series(index=self.df.index, dtype=float)
                            transformed_full[clean_data.index] = transformed
                            transformed_full[self.df[col].isnull()] = np.nan
                            
                            self.df[f'{col}_log'] = transformed_full
                            
                            new_skew = stats.skew(transformed)
                            
                            self.transformation_definitions[f'{col}_log'] = {
                                'description': f'Log transformation of {col} to reduce right skewness',
                                'transformation': f'log1p({col})',
                                'original_column': col,
                                'original_skewness': skew_val,
                                'transformed_skewness': new_skew,
                                'improvement': abs(skew_val) - abs(new_skew),
                                'method': 'numpy.log1p',
                                'created_by': 'Phase4Transformations.apply_additional_transformations'
                            }
                            
                            logger.info(f"Log transformation applied to {col}: {skew_val:.3f} → {new_skew:.3f}")
                            
                except Exception as e:
                    logger.warning(f"Failed to transform {col}: {str(e)}")
                    
    def validate_transformations(self) -> Dict[str, Any]:
        """
        Validate the applied transformations.
        
        Returns:
            Dictionary containing validation results
        """
        logger.info("Validating transformations")
        
        validation_results = {}
        
        for feature_name, definition in self.transformation_definitions.items():
            if feature_name in self.df.columns:
                feature_series = self.df[feature_name]
                original_col = definition['original_column']
                
                validation_results[feature_name] = {
                    'no_infinite_values': not np.isinf(feature_series).any(),
                    'skewness_improved': definition.get('improvement', 0) > 0,
                    'reasonable_skewness': abs(definition.get('transformed_skewness', 0)) < 2.0,
                    'maintains_relationships': self._check_relationship_preservation(
                        feature_series, self.df[original_col]
                    )
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
            logger.info("✓ All transformations validation PASSED")
        else:
            logger.warning("✗ Some transformations validation FAILED")
            
        return validation_results
        
    def _check_relationship_preservation(self, transformed: pd.Series, original: pd.Series) -> bool:
        """
        Check if the transformation preserves the monotonic relationship.
        
        Args:
            transformed: Transformed feature series
            original: Original feature series
            
        Returns:
            True if relationship is preserved, False otherwise
        """
        try:
            # Remove NaN values
            mask = ~(transformed.isnull() | original.isnull())
            if mask.sum() < 10:  # Need at least 10 points
                return True
                
            clean_transformed = transformed[mask]
            clean_original = original[mask]
            
            # Check if correlation is strong and positive (monotonic relationship preserved)
            correlation = clean_transformed.corr(clean_original)
            return correlation > 0.8
            
        except Exception:
            return False
            
    def get_transformation_summary(self) -> Dict[str, Any]:
        """
        Get summary of applied transformations.
        
        Returns:
            Dictionary containing transformation summary
        """
        summary = {
            'transformations_applied': list(self.transformation_definitions.keys()),
            'transformation_definitions': self.transformation_definitions,
            'transformers': {k: str(v) for k, v in self.transformers.items()},  # Convert to string for JSON
            'audit_log': self.audit_log,
            'dataframe_shape': self.df.shape
        }
        
        return summary
        
    def save_transformed_features(self, output_path: str = "data/featured/transformed_features.csv") -> None:
        """
        Save the dataframe with transformed features.
        
        Args:
            output_path: Path to save the enhanced dataset
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.df.to_csv(output_path)
        logger.info(f"Transformed features dataset saved to {output_path}")
        
    def save_transformation_documentation(self, output_path: str = "data/featured/transformation_definitions.json") -> None:
        """
        Save transformation definitions and audit log.
        
        Args:
            output_path: Path to save the documentation
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        documentation = self.get_transformation_summary()
        
        with open(output_path, 'w') as f:
            json.dump(documentation, f, indent=2, default=str)
            
        logger.info(f"Transformation documentation saved to {output_path}")


def main():
    """
    Main function to run Phase 4 Task 2.3: Distribution-Based Transformations.
    """
    try:
        # Load data with interaction features (assuming previous tasks completed)
        interaction_features_path = "data/featured/interaction_features.csv"
        
        if not Path(interaction_features_path).exists():
            raise FileNotFoundError(
                f"Interaction features file not found: {interaction_features_path}. "
                "Please run Phase 4 Task 2.2 first."
            )
            
        df = pd.read_csv(interaction_features_path, index_col=0)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        
        # Create transformation processor
        transformer = Phase4Transformations(df)
        
        # Apply specific transformations
        transformer.transform_study_hours_log()
        transformer.transform_previous_score_boxcox()
        
        # Apply additional transformations if needed
        transformer.apply_additional_transformations()
        
        # Validate transformations
        validation_results = transformer.validate_transformations()
        
        # Save results
        transformer.save_transformed_features()
        transformer.save_transformation_documentation()
        
        # Print summary
        summary = transformer.get_transformation_summary()
        print(f"\n=== Phase 4 Task 2.3 Complete ===")
        print(f"Transformations applied: {summary['transformations_applied']}")
        print(f"Dataset shape: {summary['dataframe_shape']}")
        print(f"Validation passed: {validation_results['overall_valid']}")
        
        return transformer.df
        
    except Exception as e:
        logger.error(f"Phase 4 Task 2.3 failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()