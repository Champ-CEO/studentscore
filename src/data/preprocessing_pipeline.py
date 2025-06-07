#!/usr/bin/env python3
"""
Data Preprocessing Pipeline Module

Implements Phase 3.4: Data Preprocessing Pipeline

This module provides a comprehensive data preprocessing pipeline including:
- Data splitting (train/validation/test)
- Feature scaling and normalization
- Categorical encoding
- Pipeline orchestration and automation
- Cross-validation setup
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime
import warnings

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessingPipeline:
    """
    Comprehensive data preprocessing pipeline.
    
    Handles all aspects of data preprocessing including:
    - Data loading and validation
    - Train/validation/test splitting
    - Feature scaling and encoding
    - Missing value imputation
    - Feature selection
    - Pipeline persistence
    """
    
    def __init__(self, db_path: Optional[str] = None, data: Optional[pd.DataFrame] = None):
        """
        Initialize the Data Preprocessing Pipeline.
        
        Args:
            db_path: Path to SQLite database file
            data: Pre-loaded DataFrame (alternative to db_path)
        """
        self.db_path = db_path
        self.data = data
        self.pipeline = None
        self.feature_names = []
        self.target_name = None
        self.categorical_features = []
        self.numerical_features = []
        self.preprocessing_config = {}
        self.split_indices = {}
        self.scalers = {}
        self.encoders = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from database or use provided DataFrame.
        
        Returns:
            DataFrame with loaded data
        """
        if self.data is not None:
            return self.data.copy()
        
        if self.db_path is None:
            raise ValueError("Either db_path or data must be provided")
        
        try:
            conn = sqlite3.connect(self.db_path)
            query = "SELECT * FROM student_scores"
            data = pd.read_sql_query(query, conn)
            conn.close()
            
            logger.info(f"Loaded {len(data)} records from database")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def identify_feature_types(self, data: pd.DataFrame, target_col: str = 'final_test') -> Tuple[List[str], List[str]]:
        """
        Automatically identify categorical and numerical features.
        
        Args:
            data: Input DataFrame
            target_col: Name of target column
            
        Returns:
            Tuple of (categorical_features, numerical_features)
        """
        categorical_features = []
        numerical_features = []
        
        for col in data.columns:
            if col == target_col:
                continue
                
            if data[col].dtype == 'object' or data[col].nunique() < 10:
                categorical_features.append(col)
            else:
                numerical_features.append(col)
        
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.target_name = target_col
        
        logger.info(f"Identified {len(categorical_features)} categorical and {len(numerical_features)} numerical features")
        return categorical_features, numerical_features
    
    def create_data_splits(self, data: pd.DataFrame, target_col: str = 'final_test',
                          test_size: float = 0.2, val_size: float = 0.2,
                          random_state: int = 42, stratify_col: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Create train/validation/test splits.
        
        Args:
            data: Input DataFrame
            target_col: Name of target column
            test_size: Proportion of data for test set
            val_size: Proportion of remaining data for validation set
            random_state: Random seed for reproducibility
            stratify_col: Column to use for stratified splitting
            
        Returns:
            Dictionary with train, validation, and test DataFrames
        """
        # Remove rows with missing target values
        clean_data = data.dropna(subset=[target_col]).copy()
        
        # Prepare stratification variable if specified
        stratify_var = None
        if stratify_col and stratify_col in clean_data.columns:
            stratify_var = clean_data[stratify_col]
        
        # First split: separate test set
        train_val_data, test_data = train_test_split(
            clean_data,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_var
        )
        
        # Second split: separate validation from training
        if stratify_col and stratify_col in train_val_data.columns:
            stratify_var_remaining = train_val_data[stratify_col]
        else:
            stratify_var_remaining = None
            
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=val_size,
            random_state=random_state,
            stratify=stratify_var_remaining
        )
        
        # Store split indices for reproducibility
        self.split_indices = {
            'train': train_data.index.tolist(),
            'validation': val_data.index.tolist(),
            'test': test_data.index.tolist()
        }
        
        splits = {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }
        
        logger.info(f"Created data splits - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        return splits
    
    def create_preprocessing_pipeline(self, scaling_method: str = 'standard',
                                    encoding_method: str = 'onehot',
                                    imputation_strategy: str = 'median') -> Pipeline:
        """
        Create sklearn preprocessing pipeline.
        
        Args:
            scaling_method: Method for scaling numerical features ('standard', 'minmax', 'robust')
            encoding_method: Method for encoding categorical features ('onehot', 'label')
            imputation_strategy: Strategy for missing value imputation ('mean', 'median', 'most_frequent', 'knn')
            
        Returns:
            Configured sklearn Pipeline
        """
        # Numerical pipeline
        if imputation_strategy == 'knn':
            num_imputer = KNNImputer(n_neighbors=5)
        else:
            num_imputer = SimpleImputer(strategy=imputation_strategy)
        
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")
        
        numerical_pipeline = Pipeline([
            ('imputer', num_imputer),
            ('scaler', scaler)
        ])
        
        # Categorical pipeline
        if encoding_method == 'onehot':
            encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        elif encoding_method == 'label':
            encoder = LabelEncoder()
        else:
            raise ValueError(f"Unknown encoding method: {encoding_method}")
        
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', encoder)
        ])
        
        # Combine pipelines
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, self.numerical_features),
                ('cat', categorical_pipeline, self.categorical_features)
            ],
            remainder='passthrough'
        )
        
        # Full pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor)
        ])
        
        self.pipeline = pipeline
        self.preprocessing_config = {
            'scaling_method': scaling_method,
            'encoding_method': encoding_method,
            'imputation_strategy': imputation_strategy,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features
        }
        
        logger.info(f"Created preprocessing pipeline with {scaling_method} scaling and {encoding_method} encoding")
        return pipeline
    
    def fit_pipeline(self, train_data: pd.DataFrame, target_col: str = 'final_test') -> Pipeline:
        """
        Fit the preprocessing pipeline on training data.
        
        Args:
            train_data: Training DataFrame
            target_col: Name of target column
            
        Returns:
            Fitted pipeline
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not created. Call create_preprocessing_pipeline() first.")
        
        # Separate features and target
        X_train = train_data.drop(columns=[target_col])
        
        # Fit pipeline
        self.pipeline.fit(X_train)
        
        # Store feature names after transformation
        self._update_feature_names()
        
        logger.info("Fitted preprocessing pipeline on training data")
        return self.pipeline
    
    def transform_data(self, data: pd.DataFrame, target_col: str = 'final_test') -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data using fitted pipeline.
        
        Args:
            data: Input DataFrame
            target_col: Name of target column
            
        Returns:
            Tuple of (transformed_features, target_values)
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit_pipeline() first.")
        
        # Separate features and target
        X = data.drop(columns=[target_col])
        y = data[target_col].values
        
        # Transform features
        X_transformed = self.pipeline.transform(X)
        
        return X_transformed, y
    
    def _update_feature_names(self) -> None:
        """
        Update feature names after transformation.
        """
        try:
            # Get feature names from the fitted pipeline
            preprocessor = self.pipeline.named_steps['preprocessor']
            
            # Numerical feature names (unchanged)
            num_features = self.numerical_features.copy()
            
            # Categorical feature names (may be expanded for one-hot encoding)
            cat_transformer = preprocessor.named_transformers_['cat']
            if hasattr(cat_transformer.named_steps['encoder'], 'get_feature_names_out'):
                cat_features = cat_transformer.named_steps['encoder'].get_feature_names_out(self.categorical_features).tolist()
            else:
                cat_features = self.categorical_features.copy()
            
            self.feature_names = num_features + cat_features
            
        except Exception as e:
            logger.warning(f"Could not extract feature names: {str(e)}")
            self.feature_names = self.numerical_features + self.categorical_features
    
    def create_cross_validation_splits(self, data: pd.DataFrame, target_col: str = 'final_test',
                                     n_splits: int = 5, stratify_col: Optional[str] = None,
                                     random_state: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create cross-validation splits.
        
        Args:
            data: Input DataFrame
            target_col: Name of target column
            n_splits: Number of CV folds
            stratify_col: Column to use for stratified CV
            random_state: Random seed
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        # Remove rows with missing target values
        clean_data = data.dropna(subset=[target_col])
        
        if stratify_col and stratify_col in clean_data.columns:
            # Stratified CV
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            stratify_var = clean_data[stratify_col]
            splits = list(cv.split(clean_data, stratify_var))
        else:
            # Regular CV
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            splits = list(cv.split(clean_data))
        
        logger.info(f"Created {n_splits}-fold cross-validation splits")
        return splits
    
    def save_pipeline(self, filepath: str) -> None:
        """
        Save the fitted pipeline to disk.
        
        Args:
            filepath: Path to save the pipeline
        """
        if self.pipeline is None:
            raise ValueError("No pipeline to save. Create and fit pipeline first.")
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save pipeline and metadata
        pipeline_data = {
            'pipeline': self.pipeline,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'preprocessing_config': self.preprocessing_config,
            'split_indices': self.split_indices
        }
        
        joblib.dump(pipeline_data, filepath)
        logger.info(f"Saved preprocessing pipeline to {filepath}")
    
    def load_pipeline(self, filepath: str) -> Pipeline:
        """
        Load a fitted pipeline from disk.
        
        Args:
            filepath: Path to the saved pipeline
            
        Returns:
            Loaded pipeline
        """
        pipeline_data = joblib.load(filepath)
        
        self.pipeline = pipeline_data['pipeline']
        self.feature_names = pipeline_data['feature_names']
        self.target_name = pipeline_data['target_name']
        self.categorical_features = pipeline_data['categorical_features']
        self.numerical_features = pipeline_data['numerical_features']
        self.preprocessing_config = pipeline_data['preprocessing_config']
        self.split_indices = pipeline_data['split_indices']
        
        logger.info(f"Loaded preprocessing pipeline from {filepath}")
        return self.pipeline
    
    def generate_preprocessing_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive preprocessing report.
        
        Args:
            output_path: Path to save the report
            
        Returns:
            Dictionary with preprocessing details
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'preprocessing_config': self.preprocessing_config,
            'feature_summary': {
                'total_features': len(self.feature_names),
                'numerical_features': len(self.numerical_features),
                'categorical_features': len(self.categorical_features),
                'feature_names': self.feature_names
            },
            'data_splits': {
                'train_size': len(self.split_indices.get('train', [])),
                'validation_size': len(self.split_indices.get('validation', [])),
                'test_size': len(self.split_indices.get('test', []))
            },
            'pipeline_steps': []
        }
        
        # Extract pipeline steps information
        if self.pipeline is not None:
            for step_name, step in self.pipeline.steps:
                step_info = {
                    'step_name': step_name,
                    'step_type': type(step).__name__,
                    'parameters': step.get_params() if hasattr(step, 'get_params') else {}
                }
                report['pipeline_steps'].append(step_info)
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Preprocessing report saved to {output_path}")
        
        return report
    
    def run_complete_preprocessing(self, target_col: str = 'final_test',
                                 test_size: float = 0.2, val_size: float = 0.2,
                                 scaling_method: str = 'standard',
                                 encoding_method: str = 'onehot',
                                 save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Run complete preprocessing pipeline.
        
        Args:
            target_col: Name of target column
            test_size: Proportion for test set
            val_size: Proportion for validation set
            scaling_method: Scaling method for numerical features
            encoding_method: Encoding method for categorical features
            save_dir: Directory to save results
            
        Returns:
            Dictionary with processed data and metadata
        """
        logger.info("Starting complete preprocessing pipeline")
        
        # Load data
        data = self.load_data()
        
        # Identify feature types
        self.identify_feature_types(data, target_col)
        
        # Create data splits
        splits = self.create_data_splits(data, target_col, test_size, val_size)
        
        # Create and fit preprocessing pipeline
        self.create_preprocessing_pipeline(scaling_method, encoding_method)
        self.fit_pipeline(splits['train'], target_col)
        
        # Transform all splits
        processed_data = {}
        for split_name, split_data in splits.items():
            X_transformed, y = self.transform_data(split_data, target_col)
            processed_data[split_name] = {
                'X': X_transformed,
                'y': y,
                'original_data': split_data
            }
        
        # Save results if directory specified
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save pipeline
            self.save_pipeline(save_path / 'preprocessing_pipeline.pkl')
            
            # Save processed data
            for split_name, split_info in processed_data.items():
                np.save(save_path / f'{split_name}_X.npy', split_info['X'])
                np.save(save_path / f'{split_name}_y.npy', split_info['y'])
                split_info['original_data'].to_csv(save_path / f'{split_name}_original.csv', index=False)
            
            # Save report
            self.generate_preprocessing_report(save_path / 'preprocessing_report.json')
        
        result = {
            'processed_data': processed_data,
            'pipeline': self.pipeline,
            'feature_names': self.feature_names,
            'preprocessing_config': self.preprocessing_config
        }
        
        logger.info("Completed preprocessing pipeline")
        return result


class AdvancedPreprocessingPipeline(DataPreprocessingPipeline):
    """
    Advanced preprocessing pipeline with additional features.
    
    Extends the basic pipeline with:
    - Feature selection
    - Advanced imputation strategies
    - Custom transformations
    - Automated hyperparameter tuning for preprocessing
    """
    
    def __init__(self, db_path: Optional[str] = None, data: Optional[pd.DataFrame] = None):
        super().__init__(db_path, data)
        self.feature_selector = None
        self.selected_features = []
    
    def add_feature_selection(self, method: str = 'k_best', k: int = 10,
                            score_func: str = 'f_regression') -> None:
        """
        Add feature selection to the pipeline.
        
        Args:
            method: Feature selection method ('k_best', 'percentile')
            k: Number of features to select
            score_func: Scoring function ('f_regression', 'mutual_info_regression')
        """
        if score_func == 'f_regression':
            score_function = f_regression
        elif score_func == 'mutual_info_regression':
            score_function = mutual_info_regression
        else:
            raise ValueError(f"Unknown score function: {score_func}")
        
        if method == 'k_best':
            self.feature_selector = SelectKBest(score_func=score_function, k=k)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        logger.info(f"Added feature selection: {method} with {k} features")
    
    def create_advanced_pipeline(self, scaling_method: str = 'standard',
                               encoding_method: str = 'onehot',
                               imputation_strategy: str = 'median',
                               include_feature_selection: bool = False) -> Pipeline:
        """
        Create advanced preprocessing pipeline with optional feature selection.
        
        Args:
            scaling_method: Scaling method
            encoding_method: Encoding method
            imputation_strategy: Imputation strategy
            include_feature_selection: Whether to include feature selection
            
        Returns:
            Advanced preprocessing pipeline
        """
        # Create base pipeline
        base_pipeline = self.create_preprocessing_pipeline(
            scaling_method, encoding_method, imputation_strategy
        )
        
        # Add feature selection if requested
        if include_feature_selection and self.feature_selector is not None:
            steps = base_pipeline.steps + [('feature_selection', self.feature_selector)]
            advanced_pipeline = Pipeline(steps)
        else:
            advanced_pipeline = base_pipeline
        
        self.pipeline = advanced_pipeline
        logger.info("Created advanced preprocessing pipeline")
        return advanced_pipeline
    
    def get_selected_features(self) -> List[str]:
        """
        Get names of selected features after feature selection.
        
        Returns:
            List of selected feature names
        """
        if self.feature_selector is None or not hasattr(self.feature_selector, 'get_support'):
            return self.feature_names
        
        # Get boolean mask of selected features
        selected_mask = self.feature_selector.get_support()
        selected_features = [name for name, selected in zip(self.feature_names, selected_mask) if selected]
        
        self.selected_features = selected_features
        logger.info(f"Selected {len(selected_features)} features out of {len(self.feature_names)}")
        return selected_features


def main():
    """
    Main function for testing the preprocessing pipeline.
    """
    # Example usage
    db_path = "data/raw/score.db"
    pipeline = DataPreprocessingPipeline(db_path=db_path)
    
    # Run complete preprocessing
    results = pipeline.run_complete_preprocessing(
        target_col='final_test',
        test_size=0.2,
        val_size=0.2,
        scaling_method='standard',
        encoding_method='onehot',
        save_dir='data/processed/pipeline'
    )
    
    print("Data Preprocessing Pipeline completed successfully!")
    print(f"Processed data shapes:")
    for split_name, split_data in results['processed_data'].items():
        print(f"  {split_name}: X={split_data['X'].shape}, y={split_data['y'].shape}")


if __name__ == "__main__":
    main()