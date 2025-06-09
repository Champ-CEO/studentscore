#!/usr/bin/env python3
"""
Phase 6 Task 6.1.2: SHAP Interpretability Analysis Fix

This module fixes the SHAP interpretability analysis that failed in Phase 5
due to pipeline compatibility issues. It provides proper SHAP analysis
for different model types with appropriate preprocessing handling.

Author: AI Assistant
Date: 2025-01-08
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# SHAP imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")

# XGBoost import
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

# Plotting imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: Plotting libraries not available. Install with: pip install matplotlib seaborn")

warnings.filterwarnings('ignore')

class SHAPInterpretabilityAnalyzer:
    """
    Fixes SHAP interpretability analysis for different model types.
    
    This class addresses the pipeline compatibility issues that caused
    SHAP analysis to fail in Phase 5 by:
    1. Properly handling different model types
    2. Extracting models from pipelines when needed
    3. Using appropriate SHAP explainers for each model type
    4. Handling preprocessing steps correctly
    """
    
    def __init__(self, 
                 clean_data_path: str,
                 output_path: str = "data/modeling_outputs",
                 random_state: int = 42):
        """
        Initialize the SHAP interpretability analyzer.
        
        Args:
            clean_data_path: Path to the clean dataset
            output_path: Directory to save analysis results
            random_state: Random state for reproducibility
        """
        self.clean_data_path = Path(clean_data_path)
        self.output_path = Path(output_path)
        self.random_state = random_state
        
        # Ensure output directory exists
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Model storage
        self.models = {}
        self.pipelines = {}
        self.preprocessors = {}
        
        # SHAP analysis results
        self.shap_results = {}
        
        # Feature information
        self.feature_names = None
        self.categorical_features = None
        self.numerical_features = None
    
    def load_data(self, target_column: str = 'final_test') -> bool:
        """
        Load and prepare the dataset.
        
        Args:
            target_column: Name of the target column
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"Loading data from {self.clean_data_path}...")
            
            if not self.clean_data_path.exists():
                print(f"Error: Data file not found at {self.clean_data_path}")
                return False
            
            # Load data
            self.data = pd.read_csv(self.clean_data_path)
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            
            # Check if target column exists
            if target_column not in self.data.columns:
                print(f"Error: Target column '{target_column}' not found in data")
                return False
            
            # Separate features and target
            self.X = self.data.drop(columns=[target_column])
            self.y = self.data[target_column]
            
            # Handle missing values in target
            if self.y.isnull().any():
                print(f"Warning: Found {self.y.isnull().sum()} missing values in target. Removing...")
                mask = ~self.y.isnull()
                self.X = self.X[mask]
                self.y = self.y[mask]
            
            # Identify feature types
            self.feature_names = list(self.X.columns)
            self.categorical_features = list(self.X.select_dtypes(include=['object', 'category']).columns)
            self.numerical_features = list(self.X.select_dtypes(include=[np.number]).columns)
            
            print(f"Features: {len(self.feature_names)} total, {len(self.numerical_features)} numerical, {len(self.categorical_features)} categorical")
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=self.random_state
            )
            
            print(f"Data split: Train {self.X_train.shape[0]}, Test {self.X_test.shape[0]}")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def create_preprocessor(self) -> ColumnTransformer:
        """
        Create a preprocessor for the data.
        
        Returns:
            ColumnTransformer: Preprocessor for numerical and categorical features
        """
        # Handle numerical features
        numerical_transformer = StandardScaler()
        
        # Handle categorical features
        categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='passthrough'
        )
        
        return preprocessor
    
    def train_models(self) -> Dict[str, Any]:
        """
        Train multiple models with proper preprocessing.
        
        Returns:
            Dict: Training results and model information
        """
        print("\nTraining models for SHAP analysis...")
        
        # Create preprocessor
        preprocessor = self.create_preprocessor()
        
        # Define models to train
        model_configs = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0, random_state=self.random_state),
            'Lasso': Lasso(alpha=1.0, random_state=self.random_state),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=self.random_state),
            'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=self.random_state)
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            model_configs['XGBoost'] = xgb.XGBRegressor(random_state=self.random_state)
        
        results = {
            'models_trained': [],
            'performance': {},
            'training_errors': []
        }
        
        for model_name, model in model_configs.items():
            try:
                print(f"Training {model_name}...")
                
                # Create pipeline
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('model', model)
                ])
                
                # Train model
                pipeline.fit(self.X_train, self.y_train)
                
                # Make predictions
                y_pred_train = pipeline.predict(self.X_train)
                y_pred_test = pipeline.predict(self.X_test)
                
                # Calculate metrics
                train_r2 = r2_score(self.y_train, y_pred_train)
                test_r2 = r2_score(self.y_test, y_pred_test)
                train_mae = mean_absolute_error(self.y_train, y_pred_train)
                test_mae = mean_absolute_error(self.y_test, y_pred_test)
                
                # Store results
                self.pipelines[model_name] = pipeline
                self.models[model_name] = model
                self.preprocessors[model_name] = preprocessor
                
                results['models_trained'].append(model_name)
                results['performance'][model_name] = {
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_mae': train_mae,
                    'test_mae': test_mae
                }
                
                print(f"{model_name} - Test R²: {test_r2:.4f}, Test MAE: {test_mae:.4f}")
                
            except Exception as e:
                error_msg = f"Error training {model_name}: {str(e)}"
                print(error_msg)
                results['training_errors'].append(error_msg)
        
        print(f"\nSuccessfully trained {len(results['models_trained'])} models")
        return results
    
    def get_feature_names_after_preprocessing(self, model_name: str) -> List[str]:
        """
        Get feature names after preprocessing.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List[str]: Feature names after preprocessing
        """
        try:
            preprocessor = self.pipelines[model_name].named_steps['preprocessor']
            
            # Get feature names from transformers
            feature_names = []
            
            # Numerical features (unchanged names)
            feature_names.extend(self.numerical_features)
            
            # Categorical features (one-hot encoded)
            if self.categorical_features:
                cat_transformer = preprocessor.named_transformers_['cat']
                if hasattr(cat_transformer, 'get_feature_names_out'):
                    cat_feature_names = cat_transformer.get_feature_names_out(self.categorical_features)
                    feature_names.extend(cat_feature_names)
                else:
                    # Fallback for older sklearn versions
                    for cat_feature in self.categorical_features:
                        unique_values = self.X_train[cat_feature].unique()
                        for value in unique_values[1:]:  # Skip first due to drop='first'
                            feature_names.append(f"{cat_feature}_{value}")
            
            return feature_names
            
        except Exception as e:
            print(f"Error getting feature names for {model_name}: {str(e)}")
            return [f"feature_{i}" for i in range(len(self.feature_names))]
    
    def analyze_model_with_shap(self, model_name: str, max_samples: int = 1000) -> Dict[str, Any]:
        """
        Perform SHAP analysis for a specific model.
        
        Args:
            model_name: Name of the model to analyze
            max_samples: Maximum number of samples for SHAP analysis
            
        Returns:
            Dict: SHAP analysis results
        """
        if not SHAP_AVAILABLE:
            return {'error': 'SHAP not available'}
        
        if model_name not in self.pipelines:
            return {'error': f'Model {model_name} not found'}
        
        try:
            print(f"\nPerforming SHAP analysis for {model_name}...")
            
            pipeline = self.pipelines[model_name]
            model = self.models[model_name]
            
            # Prepare data for SHAP
            X_train_processed = pipeline.named_steps['preprocessor'].transform(self.X_train)
            X_test_processed = pipeline.named_steps['preprocessor'].transform(self.X_test)
            
            # Limit samples for performance
            if X_train_processed.shape[0] > max_samples:
                indices = np.random.choice(X_train_processed.shape[0], max_samples, replace=False)
                X_train_sample = X_train_processed[indices]
            else:
                X_train_sample = X_train_processed
            
            if X_test_processed.shape[0] > max_samples:
                indices = np.random.choice(X_test_processed.shape[0], max_samples, replace=False)
                X_test_sample = X_test_processed[indices]
            else:
                X_test_sample = X_test_processed
            
            # Get feature names
            feature_names = self.get_feature_names_after_preprocessing(model_name)
            
            # Choose appropriate SHAP explainer based on model type
            explainer = None
            shap_values = None
            
            if model_name in ['LinearRegression', 'Ridge', 'Lasso']:
                # Use Linear explainer for linear models
                explainer = shap.LinearExplainer(model, X_train_sample)
                shap_values = explainer.shap_values(X_test_sample)
                
            elif model_name == 'RandomForest':
                # Use Tree explainer for tree-based models
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test_sample)
                
            elif model_name == 'XGBoost' and XGBOOST_AVAILABLE:
                # Use Tree explainer for XGBoost
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test_sample)
                
            else:
                # Use Kernel explainer as fallback (slower but works for any model)
                print(f"Using Kernel explainer for {model_name} (this may take longer)...")
                
                # Create a prediction function that uses the full pipeline
                def predict_fn(X):
                    return model.predict(X)
                
                # Use a smaller background sample for Kernel explainer
                background_size = min(100, X_train_sample.shape[0])
                background_indices = np.random.choice(X_train_sample.shape[0], background_size, replace=False)
                background_sample = X_train_sample[background_indices]
                
                explainer = shap.KernelExplainer(predict_fn, background_sample)
                
                # Use even smaller sample for explanation
                explain_size = min(50, X_test_sample.shape[0])
                explain_indices = np.random.choice(X_test_sample.shape[0], explain_size, replace=False)
                X_explain = X_test_sample[explain_indices]
                
                shap_values = explainer.shap_values(X_explain)
                X_test_sample = X_explain
            
            # Calculate feature importance
            if shap_values is not None:
                # Handle different SHAP value formats
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]  # For multi-output models
                
                # Calculate mean absolute SHAP values for feature importance
                feature_importance = np.mean(np.abs(shap_values), axis=0)
                
                # Create feature importance ranking
                importance_df = pd.DataFrame({
                    'feature': feature_names[:len(feature_importance)],
                    'importance': feature_importance
                }).sort_values('importance', ascending=False)
                
                # Calculate SHAP statistics
                shap_stats = {
                    'mean_abs_shap': np.mean(np.abs(shap_values)),
                    'std_abs_shap': np.std(np.abs(shap_values)),
                    'max_abs_shap': np.max(np.abs(shap_values)),
                    'min_abs_shap': np.min(np.abs(shap_values))
                }
                
                results = {
                    'success': True,
                    'explainer_type': type(explainer).__name__,
                    'n_samples_explained': X_test_sample.shape[0],
                    'n_features': len(feature_names),
                    'feature_importance': importance_df.to_dict('records'),
                    'top_10_features': importance_df.head(10).to_dict('records'),
                    'shap_statistics': shap_stats,
                    'shap_values_shape': shap_values.shape,
                    'feature_names': feature_names
                }
                
                # Store SHAP values and explainer for potential plotting
                self.shap_results[model_name] = {
                    'explainer': explainer,
                    'shap_values': shap_values,
                    'X_test_sample': X_test_sample,
                    'feature_names': feature_names,
                    'results': results
                }
                
                print(f"SHAP analysis completed for {model_name}")
                print(f"Top 3 features: {[f['feature'] for f in importance_df.head(3).to_dict('records')]}")
                
                return results
            
            else:
                return {'error': 'Failed to compute SHAP values'}
                
        except Exception as e:
            error_msg = f"Error in SHAP analysis for {model_name}: {str(e)}"
            print(error_msg)
            return {'error': error_msg}
    
    def create_shap_plots(self, model_name: str, save_plots: bool = True) -> Dict[str, str]:
        """
        Create SHAP visualization plots.
        
        Args:
            model_name: Name of the model
            save_plots: Whether to save plots to files
            
        Returns:
            Dict: Paths to saved plots
        """
        if not PLOTTING_AVAILABLE or not SHAP_AVAILABLE:
            return {'error': 'Plotting libraries or SHAP not available'}
        
        if model_name not in self.shap_results:
            return {'error': f'SHAP results not found for {model_name}'}
        
        try:
            shap_data = self.shap_results[model_name]
            explainer = shap_data['explainer']
            shap_values = shap_data['shap_values']
            X_test_sample = shap_data['X_test_sample']
            feature_names = shap_data['feature_names']
            
            plot_paths = {}
            
            if save_plots:
                plot_dir = self.output_path / 'shap_plots' / model_name
                plot_dir.mkdir(parents=True, exist_ok=True)
                
                # Summary plot
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, show=False)
                summary_path = plot_dir / 'summary_plot.png'
                plt.savefig(summary_path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_paths['summary'] = str(summary_path)
                
                # Feature importance plot
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, plot_type='bar', show=False)
                importance_path = plot_dir / 'feature_importance.png'
                plt.savefig(importance_path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_paths['importance'] = str(importance_path)
                
                # Waterfall plot for first prediction
                if hasattr(shap, 'waterfall_plot'):
                    plt.figure(figsize=(10, 8))
                    shap.waterfall_plot(explainer.expected_value, shap_values[0], X_test_sample[0], feature_names=feature_names, show=False)
                    waterfall_path = plot_dir / 'waterfall_plot.png'
                    plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    plot_paths['waterfall'] = str(waterfall_path)
                
                print(f"SHAP plots saved for {model_name} in {plot_dir}")
            
            return plot_paths
            
        except Exception as e:
            error_msg = f"Error creating SHAP plots for {model_name}: {str(e)}"
            print(error_msg)
            return {'error': error_msg}
    
    def run_complete_shap_analysis(self, models_to_analyze: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run complete SHAP analysis for all or specified models.
        
        Args:
            models_to_analyze: List of model names to analyze. If None, analyze all trained models.
            
        Returns:
            Dict: Complete SHAP analysis results
        """
        print("\n" + "="*80)
        print("STARTING COMPLETE SHAP INTERPRETABILITY ANALYSIS")
        print("="*80)
        
        # Load data
        if not self.load_data():
            return {'error': 'Failed to load data'}
        
        # Train models
        training_results = self.train_models()
        
        if not training_results['models_trained']:
            return {'error': 'No models were successfully trained'}
        
        # Determine which models to analyze
        if models_to_analyze is None:
            models_to_analyze = training_results['models_trained']
        else:
            models_to_analyze = [m for m in models_to_analyze if m in training_results['models_trained']]
        
        print(f"\nAnalyzing SHAP interpretability for {len(models_to_analyze)} models...")
        
        # Perform SHAP analysis for each model
        shap_analysis_results = {}
        plot_results = {}
        
        for model_name in models_to_analyze:
            print(f"\n{'-'*50}")
            print(f"Analyzing {model_name}")
            print(f"{'-'*50}")
            
            # SHAP analysis
            shap_result = self.analyze_model_with_shap(model_name)
            shap_analysis_results[model_name] = shap_result
            
            # Create plots if analysis was successful
            if shap_result.get('success', False):
                plot_result = self.create_shap_plots(model_name)
                plot_results[model_name] = plot_result
        
        # Compile final results
        final_results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'data_path': str(self.clean_data_path),
                'data_shape': list(self.data.shape),
                'features_count': len(self.feature_names),
                'numerical_features': len(self.numerical_features),
                'categorical_features': len(self.categorical_features),
                'target_name': 'final_test'
            },
            'training_results': training_results,
            'shap_analysis_results': shap_analysis_results,
            'plot_results': plot_results,
            'successful_analyses': [m for m, r in shap_analysis_results.items() if r.get('success', False)],
            'failed_analyses': [m for m, r in shap_analysis_results.items() if not r.get('success', False)],
            'overall_status': 'SUCCESS' if any(r.get('success', False) for r in shap_analysis_results.values()) else 'FAILED'
        }
        
        # Save results
        output_file = self.output_path / 'shap_interpretability_analysis.json'
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print("\n" + "="*80)
        print("SHAP INTERPRETABILITY ANALYSIS COMPLETE")
        print("="*80)
        print(f"Overall Status: {final_results['overall_status']}")
        print(f"Successful Analyses: {len(final_results['successful_analyses'])}")
        print(f"Failed Analyses: {len(final_results['failed_analyses'])}")
        
        if final_results['successful_analyses']:
            print("\nSuccessfully analyzed models:")
            for model_name in final_results['successful_analyses']:
                result = shap_analysis_results[model_name]
                print(f"- {model_name}: {result['explainer_type']} explainer, {result['n_samples_explained']} samples")
        
        if final_results['failed_analyses']:
            print("\nFailed analyses:")
            for model_name in final_results['failed_analyses']:
                error = shap_analysis_results[model_name].get('error', 'Unknown error')
                print(f"- {model_name}: {error}")
        
        print(f"\nDetailed results saved to: {output_file}")
        
        return final_results

def main():
    """
    Main function to run SHAP interpretability analysis.
    """
    # Configuration
    clean_data_path = "data/modeling_outputs/clean_dataset_no_leakage.csv"
    output_path = "data/modeling_outputs"
    
    # Create analyzer
    analyzer = SHAPInterpretabilityAnalyzer(
        clean_data_path=clean_data_path,
        output_path=output_path
    )
    
    # Run complete analysis
    results = analyzer.run_complete_shap_analysis()
    
    return results

if __name__ == "__main__":
    main()