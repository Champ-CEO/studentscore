#!/usr/bin/env python3
"""
Phase 4 Task 3.1: Advanced Preprocessing (High Priority)

This module implements task 4.3.1 from TASKS.md:
- Normalization/Standardization of numerical features
- Encoding of categorical variables
- Feature scaling for model readiness

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, OrdinalEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Phase4AdvancedPreprocessing:
    """
    Applies advanced preprocessing for Phase 4 feature engineering.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with the dataset.
        
        Args:
            df: DataFrame with transformed features from previous tasks
        """
        self.df = df.copy()
        self.preprocessing_definitions = {}
        self.fitted_preprocessors = {}  # Store fitted preprocessors
        self.audit_log = []
        self.feature_mappings = {}  # Track original to processed feature mappings
        
    def identify_feature_types(self) -> Dict[str, List[str]]:
        """
        Identify and categorize features by type.
        
        Returns:
            Dictionary mapping feature types to column lists
        """
        feature_types = {
            'numerical_continuous': [],
            'numerical_discrete': [],
            'categorical_nominal': [],
            'categorical_ordinal': [],
            'binary': [],
            'target': [],
            'id_columns': [],
            'derived_features': [],
            'interaction_features': [],
            'transformed_features': []
        }
        
        for col in self.df.columns:
            # Skip target variable
            if col in ['final_test', 'target']:
                feature_types['target'].append(col)
                continue
                
            # Skip ID columns
            if 'id' in col.lower() or col in ['student_id']:
                feature_types['id_columns'].append(col)
                continue
                
            # Identify derived features (from Phase 4 tasks)
            if any(suffix in col for suffix in ['_efficiency', '_support', '_index', '_score']):
                feature_types['derived_features'].append(col)
                continue
                
            # Identify interaction features
            if 'x' in col.lower() or '_interaction' in col:
                feature_types['interaction_features'].append(col)
                continue
                
            # Identify transformed features
            if any(suffix in col for suffix in ['_log', '_boxcox', '_sqrt']):
                feature_types['transformed_features'].append(col)
                continue
                
            # Analyze data type and values
            if self.df[col].dtype in ['object', 'category']:
                unique_values = self.df[col].nunique()
                if unique_values == 2:
                    feature_types['binary'].append(col)
                elif col in ['parental_education_level', 'distance_from_home']:
                    feature_types['categorical_ordinal'].append(col)
                else:
                    feature_types['categorical_nominal'].append(col)
            else:
                # Numerical features
                unique_values = self.df[col].nunique()
                total_values = len(self.df[col].dropna())
                
                if unique_values <= 10 or unique_values / total_values < 0.05:
                    feature_types['numerical_discrete'].append(col)
                else:
                    feature_types['numerical_continuous'].append(col)
                    
        logger.info("Feature type identification completed")
        for feature_type, columns in feature_types.items():
            if columns:
                logger.info(f"{feature_type}: {len(columns)} features")
                
        return feature_types
        
    def apply_numerical_scaling(self, feature_types: Dict[str, List[str]]) -> None:
        """
        Apply appropriate scaling to numerical features.
        
        Args:
            feature_types: Dictionary mapping feature types to column lists
        """
        logger.info("Applying numerical feature scaling")
        
        # Combine all numerical features
        numerical_features = (
            feature_types['numerical_continuous'] + 
            feature_types['numerical_discrete'] +
            feature_types['derived_features'] +
            feature_types['interaction_features'] +
            feature_types['transformed_features']
        )
        
        if not numerical_features:
            logger.info("No numerical features found for scaling")
            return
            
        # Define scaling strategies
        scaling_strategies = {
            'standard': StandardScaler(),  # For normally distributed features
            'minmax': MinMaxScaler(),      # For features with known bounds
            'robust': RobustScaler()       # For features with outliers
        }
        
        # Apply different scaling based on feature characteristics
        for feature in numerical_features:
            if feature not in self.df.columns:
                continue
                
            feature_data = self.df[feature].dropna()
            if len(feature_data) == 0:
                continue
                
            # Ensure feature data is numeric
            try:
                feature_data = pd.to_numeric(feature_data, errors='coerce')
                feature_data = feature_data.dropna()
                
                if len(feature_data) == 0:
                    logger.warning(f"Feature {feature} has no valid numeric data after conversion")
                    continue
            except Exception as e:
                logger.warning(f"Failed to convert {feature} to numeric: {str(e)}")
                continue
                
            # Choose scaling method based on feature characteristics
            scaling_method = self._choose_scaling_method(feature, feature_data)
            scaler = scaling_strategies[scaling_method]
            
            # Fit and transform
            try:
                scaled_data = scaler.fit_transform(feature_data.values.reshape(-1, 1)).flatten()
                
                # Create scaled feature column
                scaled_feature_name = f"{feature}_scaled"
                scaled_series = pd.Series(index=self.df.index, dtype=float)
                scaled_series[feature_data.index] = scaled_data
                scaled_series[self.df[feature].isnull()] = np.nan
                
                self.df[scaled_feature_name] = scaled_series
                
                # Store fitted scaler
                self.fitted_preprocessors[scaled_feature_name] = scaler
                
                # Record preprocessing definition
                self.preprocessing_definitions[scaled_feature_name] = {
                    'description': f'{scaling_method.title()} scaling of {feature}',
                    'method': scaling_method,
                    'original_feature': feature,
                    'scaler_type': type(scaler).__name__,
                    'scaler_params': scaler.get_params(),
                    'created_by': 'Phase4AdvancedPreprocessing.apply_numerical_scaling'
                }
                
                # Update feature mapping
                self.feature_mappings[feature] = scaled_feature_name
                
                # Log statistics
                self.audit_log.append({
                    'feature': scaled_feature_name,
                    'action': f'{scaling_method}_scaling',
                    'original_stats': {
                        'mean': feature_data.mean(),
                        'std': feature_data.std(),
                        'min': feature_data.min(),
                        'max': feature_data.max()
                    },
                    'scaled_stats': {
                        'mean': scaled_data.mean(),
                        'std': scaled_data.std(),
                        'min': scaled_data.min(),
                        'max': scaled_data.max()
                    }
                })
                
                logger.info(f"Applied {scaling_method} scaling to {feature}")
                
            except Exception as e:
                logger.warning(f"Failed to scale {feature}: {str(e)}")
                
    def _choose_scaling_method(self, feature_name: str, feature_data: pd.Series) -> str:
        """
        Choose appropriate scaling method based on feature characteristics.
        
        Args:
            feature_name: Name of the feature
            feature_data: Feature data series
            
        Returns:
            Scaling method name
        """
        # Ensure feature_data is numeric
        try:
            feature_data = pd.to_numeric(feature_data, errors='coerce')
            feature_data = feature_data.dropna()
            
            if len(feature_data) == 0:
                return 'standard'  # Default fallback
        except Exception:
            return 'standard'  # Default fallback
            
        # Check for outliers using IQR
        q1 = feature_data.quantile(0.25)
        q3 = feature_data.quantile(0.75)
        iqr = q3 - q1
        outlier_threshold = 1.5 * iqr
        outliers = ((feature_data < (q1 - outlier_threshold)) | 
                   (feature_data > (q3 + outlier_threshold))).sum()
        outlier_ratio = outliers / len(feature_data)
        
        # Check if feature has known bounds (0-100 for scores, etc.)
        has_known_bounds = (
            'score' in feature_name.lower() or 
            'percentage' in feature_name.lower() or
            'rate' in feature_name.lower()
        )
        
        # Decision logic
        if outlier_ratio > 0.1:  # More than 10% outliers
            return 'robust'
        elif has_known_bounds or feature_data.min() >= 0:
            return 'minmax'
        else:
            return 'standard'
            
    def apply_categorical_encoding(self, feature_types: Dict[str, List[str]]) -> None:
        """
        Apply appropriate encoding to categorical features.
        
        Args:
            feature_types: Dictionary mapping feature types to column lists
        """
        logger.info("Applying categorical feature encoding")
        
        # Handle binary features
        self._encode_binary_features(feature_types['binary'])
        
        # Handle ordinal features
        self._encode_ordinal_features(feature_types['categorical_ordinal'])
        
        # Handle nominal features
        self._encode_nominal_features(feature_types['categorical_nominal'])
        
    def _encode_binary_features(self, binary_features: List[str]) -> None:
        """
        Encode binary categorical features.
        
        Args:
            binary_features: List of binary feature names
        """
        for feature in binary_features:
            if feature not in self.df.columns:
                continue
                
            try:
                # Use LabelEncoder for binary features
                encoder = LabelEncoder()
                feature_data = self.df[feature].dropna()
                
                if len(feature_data) == 0:
                    continue
                    
                encoded_data = encoder.fit_transform(feature_data)
                
                # Create encoded feature column
                encoded_feature_name = f"{feature}_encoded"
                encoded_series = pd.Series(index=self.df.index, dtype=float)
                encoded_series[feature_data.index] = encoded_data
                encoded_series[self.df[feature].isnull()] = np.nan
                
                self.df[encoded_feature_name] = encoded_series
                
                # Store fitted encoder
                self.fitted_preprocessors[encoded_feature_name] = encoder
                
                # Record preprocessing definition
                self.preprocessing_definitions[encoded_feature_name] = {
                    'description': f'Label encoding of binary feature {feature}',
                    'method': 'label_encoding',
                    'original_feature': feature,
                    'encoder_type': 'LabelEncoder',
                    'classes': encoder.classes_.tolist(),
                    'created_by': 'Phase4AdvancedPreprocessing._encode_binary_features'
                }
                
                # Update feature mapping
                self.feature_mappings[feature] = encoded_feature_name
                
                logger.info(f"Applied label encoding to binary feature {feature}")
                
            except Exception as e:
                logger.warning(f"Failed to encode binary feature {feature}: {str(e)}")
                
    def _encode_ordinal_features(self, ordinal_features: List[str]) -> None:
        """
        Encode ordinal categorical features.
        
        Args:
            ordinal_features: List of ordinal feature names
        """
        # Define ordinal mappings
        ordinal_mappings = {
            'parental_education_level': {
                'No Education': 0,
                'Primary': 1,
                'Secondary': 2,
                'Higher Secondary': 3,
                'Bachelor': 4,
                'Master': 5,
                'PhD': 6
            },
            'distance_from_home': {
                'Near': 0,
                'Moderate': 1,
                'Far': 2
            }
        }
        
        for feature in ordinal_features:
            if feature not in self.df.columns:
                continue
                
            try:
                if feature in ordinal_mappings:
                    # Use predefined mapping
                    mapping = ordinal_mappings[feature]
                    encoded_data = self.df[feature].map(mapping)
                    
                    # Create encoded feature column
                    encoded_feature_name = f"{feature}_encoded"
                    self.df[encoded_feature_name] = encoded_data
                    
                    # Record preprocessing definition
                    self.preprocessing_definitions[encoded_feature_name] = {
                        'description': f'Ordinal encoding of {feature}',
                        'method': 'ordinal_encoding',
                        'original_feature': feature,
                        'mapping': mapping,
                        'created_by': 'Phase4AdvancedPreprocessing._encode_ordinal_features'
                    }
                    
                else:
                    # Use OrdinalEncoder for unknown ordinal features
                    encoder = OrdinalEncoder()
                    feature_data = self.df[feature].dropna()
                    
                    if len(feature_data) == 0:
                        continue
                        
                    encoded_data = encoder.fit_transform(feature_data.values.reshape(-1, 1)).flatten()
                    
                    # Create encoded feature column
                    encoded_feature_name = f"{feature}_encoded"
                    encoded_series = pd.Series(index=self.df.index, dtype=float)
                    encoded_series[feature_data.index] = encoded_data
                    encoded_series[self.df[feature].isnull()] = np.nan
                    
                    self.df[encoded_feature_name] = encoded_series
                    
                    # Store fitted encoder
                    self.fitted_preprocessors[encoded_feature_name] = encoder
                    
                    # Record preprocessing definition
                    self.preprocessing_definitions[encoded_feature_name] = {
                        'description': f'Ordinal encoding of {feature}',
                        'method': 'ordinal_encoding',
                        'original_feature': feature,
                        'encoder_type': 'OrdinalEncoder',
                        'categories': encoder.categories_[0].tolist(),
                        'created_by': 'Phase4AdvancedPreprocessing._encode_ordinal_features'
                    }
                    
                # Update feature mapping
                self.feature_mappings[feature] = encoded_feature_name
                
                logger.info(f"Applied ordinal encoding to {feature}")
                
            except Exception as e:
                logger.warning(f"Failed to encode ordinal feature {feature}: {str(e)}")
                
    def _encode_nominal_features(self, nominal_features: List[str]) -> None:
        """
        Encode nominal categorical features using one-hot encoding.
        
        Args:
            nominal_features: List of nominal feature names
        """
        for feature in nominal_features:
            if feature not in self.df.columns:
                continue
                
            try:
                # Check cardinality
                unique_values = self.df[feature].nunique()
                
                if unique_values > 10:
                    logger.warning(f"High cardinality feature {feature} ({unique_values} categories). Consider alternative encoding.")
                    continue
                    
                # Use OneHotEncoder
                encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
                feature_data = self.df[feature].dropna()
                
                if len(feature_data) == 0:
                    continue
                    
                encoded_data = encoder.fit_transform(feature_data.values.reshape(-1, 1))
                
                # Create encoded feature columns
                feature_names = [f"{feature}_{cat}" for cat in encoder.categories_[0][1:]]  # Skip first due to drop='first'
                
                for i, col_name in enumerate(feature_names):
                    encoded_series = pd.Series(index=self.df.index, dtype=float)
                    encoded_series[feature_data.index] = encoded_data[:, i]
                    encoded_series[self.df[feature].isnull()] = np.nan
                    
                    self.df[col_name] = encoded_series
                    
                # Store fitted encoder
                self.fitted_preprocessors[f"{feature}_onehot"] = encoder
                
                # Record preprocessing definition
                self.preprocessing_definitions[f"{feature}_onehot"] = {
                    'description': f'One-hot encoding of nominal feature {feature}',
                    'method': 'one_hot_encoding',
                    'original_feature': feature,
                    'encoder_type': 'OneHotEncoder',
                    'categories': encoder.categories_[0].tolist(),
                    'feature_names': feature_names,
                    'created_by': 'Phase4AdvancedPreprocessing._encode_nominal_features'
                }
                
                # Update feature mapping
                self.feature_mappings[feature] = feature_names
                
                logger.info(f"Applied one-hot encoding to {feature} (created {len(feature_names)} features)")
                
            except Exception as e:
                logger.warning(f"Failed to encode nominal feature {feature}: {str(e)}")
                
    def create_model_ready_dataset(self) -> pd.DataFrame:
        """
        Create a model-ready dataset with only processed features.
        
        Returns:
            DataFrame with processed features ready for modeling
        """
        logger.info("Creating model-ready dataset")
        
        # Identify processed features
        processed_features = []
        
        for col in self.df.columns:
            # Include scaled numerical features
            if col.endswith('_scaled'):
                processed_features.append(col)
            # Include encoded categorical features
            elif col.endswith('_encoded') or any(original in col for original in self.feature_mappings.keys()):
                processed_features.append(col)
            # Include target variable
            elif col in ['final_test', 'target']:
                processed_features.append(col)
                
        # Create model-ready dataset
        model_ready_df = self.df[processed_features].copy()
        
        logger.info(f"Model-ready dataset created with {len(processed_features)} features")
        
        return model_ready_df
        
    def validate_preprocessing(self) -> Dict[str, Any]:
        """
        Validate the applied preprocessing.
        
        Returns:
            Dictionary containing validation results
        """
        logger.info("Validating preprocessing")
        
        validation_results = {}
        
        # Check scaled features
        scaled_features = [col for col in self.df.columns if col.endswith('_scaled')]
        for feature in scaled_features:
            feature_data = self.df[feature].dropna()
            if len(feature_data) > 0:
                validation_results[feature] = {
                    'no_infinite_values': not np.isinf(feature_data).any(),
                    'reasonable_range': feature_data.std() > 0,
                    'no_extreme_outliers': (np.abs(feature_data) < 10).all()  # Scaled features should be reasonable
                }
                
        # Check encoded features
        encoded_features = [col for col in self.df.columns if col.endswith('_encoded')]
        for feature in encoded_features:
            feature_data = self.df[feature].dropna()
            if len(feature_data) > 0:
                validation_results[feature] = {
                    'integer_values': feature_data.apply(lambda x: x == int(x) if pd.notnull(x) else True).all(),
                    'reasonable_range': feature_data.min() >= 0,
                    'no_missing_after_encoding': feature_data.isnull().sum() == 0
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
            logger.info("✓ All preprocessing validation PASSED")
        else:
            logger.warning("✗ Some preprocessing validation FAILED")
            
        return validation_results
        
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Get summary of applied preprocessing.
        
        Returns:
            Dictionary containing preprocessing summary
        """
        summary = {
            'preprocessing_applied': list(self.preprocessing_definitions.keys()),
            'preprocessing_definitions': self.preprocessing_definitions,
            'feature_mappings': self.feature_mappings,
            'fitted_preprocessors': {k: str(v) for k, v in self.fitted_preprocessors.items()},
            'audit_log': self.audit_log,
            'dataframe_shape': self.df.shape
        }
        
        return summary
        
    def save_preprocessed_features(self, output_path: str = "data/featured/preprocessed_features.csv") -> None:
        """
        Save the dataframe with preprocessed features.
        
        Args:
            output_path: Path to save the enhanced dataset
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.df.to_csv(output_path)
        logger.info(f"Preprocessed features dataset saved to {output_path}")
        
    def save_model_ready_dataset(self, output_path: str = "data/featured/model_ready_dataset.csv") -> None:
        """
        Save the model-ready dataset.
        
        Args:
            output_path: Path to save the model-ready dataset
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_ready_df = self.create_model_ready_dataset()
        model_ready_df.to_csv(output_path)
        logger.info(f"Model-ready dataset saved to {output_path}")
        
    def save_preprocessing_documentation(self, output_path: str = "data/featured/preprocessing_definitions.json") -> None:
        """
        Save preprocessing definitions and audit log.
        
        Args:
            output_path: Path to save the documentation
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        documentation = self.get_preprocessing_summary()
        
        with open(output_path, 'w') as f:
            json.dump(documentation, f, indent=2, default=str)
            
        logger.info(f"Preprocessing documentation saved to {output_path}")


def main():
    """
    Main function to run Phase 4 Task 3.1: Advanced Preprocessing.
    """
    try:
        # Load data with transformed features (assuming previous tasks completed)
        transformed_features_path = "data/featured/transformed_features.csv"
        
        if not Path(transformed_features_path).exists():
            raise FileNotFoundError(
                f"Transformed features file not found: {transformed_features_path}. "
                "Please run Phase 4 Task 2.3 first."
            )
            
        df = pd.read_csv(transformed_features_path, index_col=0)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        
        # Create preprocessing processor
        preprocessor = Phase4AdvancedPreprocessing(df)
        
        # Identify feature types
        feature_types = preprocessor.identify_feature_types()
        
        # Apply preprocessing
        preprocessor.apply_numerical_scaling(feature_types)
        preprocessor.apply_categorical_encoding(feature_types)
        
        # Validate preprocessing
        validation_results = preprocessor.validate_preprocessing()
        
        # Save results
        preprocessor.save_preprocessed_features()
        preprocessor.save_model_ready_dataset()
        preprocessor.save_preprocessing_documentation()
        
        # Print summary
        summary = preprocessor.get_preprocessing_summary()
        print(f"\n=== Phase 4 Task 3.1 Complete ===")
        print(f"Preprocessing applied: {summary['preprocessing_applied']}")
        print(f"Dataset shape: {summary['dataframe_shape']}")
        print(f"Validation passed: {validation_results['overall_valid']}")
        
        return preprocessor.df
        
    except Exception as e:
        logger.error(f"Phase 4 Task 3.1 failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()