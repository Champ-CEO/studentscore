#!/usr/bin/env python3
"""
Phase 6 Task 6.3.1: External Validation and Robustness Analysis (Simplified)

This module implements essential external validation and robustness testing
for the trained models with optimized performance.

Author: AI Assistant
Date: 2025-01-08
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Modeling imports
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimplifiedExternalValidator:
    """
    Simplified external validation and robustness analysis for machine learning models.
    Focuses on essential validation techniques with optimized performance.
    """
    
    def __init__(self, project_root: str = None):
        if project_root is None:
            current_file = Path(__file__).resolve()
            self.project_root = current_file.parents[2]
        else:
            self.project_root = Path(project_root)
        
        self.data_path = self.project_root / "data" / "modeling_outputs"
        self.output_path = self.project_root / "data" / "modeling_outputs"
        self.plots_path = self.output_path / "external_validation_plots"
        
        # Ensure output directories exist
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.plots_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Output path: {self.output_path}")
        logger.info(f"Plots path: {self.plots_path}")
    
    def load_clean_data(self) -> pd.DataFrame:
        """Load the cleaned dataset without data leakage."""
        data_files = [
            self.data_path / "clean_dataset_final_no_leakage.csv",
            self.data_path / "clean_dataset_no_leakage.csv"
        ]
        
        for data_file in data_files:
            if data_file.exists():
                logger.info(f"Loading data from {data_file}")
                df = pd.read_csv(data_file)
                logger.info(f"Data loaded: {df.shape}")
                return df
        
        raise FileNotFoundError("No clean dataset found")
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for modeling."""
        logger.info("Preparing data for external validation...")
        
        # Separate features and target
        X = df.drop(['final_test'], axis=1)
        y = df['final_test']
        
        # Remove rows with missing target
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        # Get numerical features only
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numerical = X[numerical_features]
        
        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X_filled = pd.DataFrame(
            imputer.fit_transform(X_numerical),
            columns=X_numerical.columns,
            index=X_numerical.index
        )
        
        logger.info(f"Final data shape: X={X_filled.shape}, y={y.shape}")
        return X_filled, y
    
    def load_or_train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Load existing models or train new ones if needed."""
        models = {}
        model_files = {
            'LinearRegression': 'linearregression_comparison.joblib',
            'Ridge': 'ridge_comparison.joblib',
            'Lasso': 'lasso_comparison.joblib',
            'RandomForest': 'randomforest_comparison.joblib',
            'GradientBoosting': 'gradientboosting_comparison.joblib'
        }
        
        models_dir = self.data_path / "models"
        
        # Try to load existing models
        for model_name, filename in model_files.items():
            model_file = models_dir / filename
            if model_file.exists():
                try:
                    model = joblib.load(model_file)
                    models[model_name] = model
                    logger.info(f"Loaded {model_name} from {model_file}")
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")
        
        # Train missing models
        if not models:
            logger.info("No existing models found, training new ones...")
            model_configs = {
                'LinearRegression': LinearRegression(),
                'Ridge': Ridge(alpha=1.0),
                'Lasso': Lasso(alpha=1.0, max_iter=2000),
                'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42),  # Reduced for speed
                'GradientBoosting': GradientBoostingRegressor(n_estimators=50, random_state=42)  # Reduced for speed
            }
            
            for model_name, model in model_configs.items():
                try:
                    logger.info(f"Training {model_name}...")
                    model.fit(X, y)
                    models[model_name] = model
                except Exception as e:
                    logger.error(f"Failed to train {model_name}: {e}")
        
        return models
    
    def temporal_validation(self, X: pd.DataFrame, y: pd.Series, models: Dict[str, Any]) -> Dict[str, Any]:
        """Perform temporal validation by splitting data chronologically."""
        logger.info("Performing temporal validation...")
        
        temporal_results = {}
        
        # Create temporal splits (assuming data is ordered chronologically)
        n_samples = len(X)
        split_points = [0.6, 0.8]  # 60% train, 20% validation, 20% test
        
        train_end = int(n_samples * split_points[0])
        val_end = int(n_samples * split_points[1])
        
        X_train_temp = X.iloc[:train_end]
        y_train_temp = y.iloc[:train_end]
        X_val_temp = X.iloc[train_end:val_end]
        y_val_temp = y.iloc[train_end:val_end]
        X_test_temp = X.iloc[val_end:]
        y_test_temp = y.iloc[val_end:]
        
        logger.info(f"Temporal splits - Train: {len(X_train_temp)}, Val: {len(X_val_temp)}, Test: {len(X_test_temp)}")
        
        for model_name, model in models.items():
            try:
                logger.info(f"Temporal validation for {model_name}...")
                
                # Train on temporal training set
                model_temp = type(model)(**model.get_params() if hasattr(model, 'get_params') else {})
                model_temp.fit(X_train_temp, y_train_temp)
                
                # Predict on validation and test sets
                val_pred = model_temp.predict(X_val_temp)
                test_pred = model_temp.predict(X_test_temp)
                
                temporal_results[model_name] = {
                    'validation_r2': float(r2_score(y_val_temp, val_pred)),
                    'validation_mae': float(mean_absolute_error(y_val_temp, val_pred)),
                    'validation_rmse': float(np.sqrt(mean_squared_error(y_val_temp, val_pred))),
                    'test_r2': float(r2_score(y_test_temp, test_pred)),
                    'test_mae': float(mean_absolute_error(y_test_temp, test_pred)),
                    'test_rmse': float(np.sqrt(mean_squared_error(y_test_temp, test_pred))),
                    'train_size': len(X_train_temp),
                    'val_size': len(X_val_temp),
                    'test_size': len(X_test_temp)
                }
                
            except Exception as e:
                logger.error(f"Temporal validation failed for {model_name}: {e}")
                temporal_results[model_name] = {'error': str(e)}
        
        return temporal_results
    
    def cross_validation_analysis(self, X: pd.DataFrame, y: pd.Series, models: Dict[str, Any]) -> Dict[str, Any]:
        """Perform cross-validation analysis."""
        logger.info("Performing cross-validation analysis...")
        
        cv_results = {}
        cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for model_name, model in models.items():
            try:
                logger.info(f"Cross-validation for {model_name}...")
                
                # R² scores
                r2_scores = cross_val_score(model, X, y, cv=cv_strategy, scoring='r2', n_jobs=1)
                
                # MAE scores (negative because sklearn uses negative MAE)
                mae_scores = -cross_val_score(model, X, y, cv=cv_strategy, scoring='neg_mean_absolute_error', n_jobs=1)
                
                # RMSE scores
                rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=cv_strategy, scoring='neg_mean_squared_error', n_jobs=1))
                
                cv_results[model_name] = {
                    'r2_mean': float(np.mean(r2_scores)),
                    'r2_std': float(np.std(r2_scores)),
                    'r2_scores': r2_scores.tolist(),
                    'mae_mean': float(np.mean(mae_scores)),
                    'mae_std': float(np.std(mae_scores)),
                    'mae_scores': mae_scores.tolist(),
                    'rmse_mean': float(np.mean(rmse_scores)),
                    'rmse_std': float(np.std(rmse_scores)),
                    'rmse_scores': rmse_scores.tolist()
                }
                
            except Exception as e:
                logger.error(f"Cross-validation failed for {model_name}: {e}")
                cv_results[model_name] = {'error': str(e)}
        
        return cv_results
    
    def robustness_testing(self, X: pd.DataFrame, y: pd.Series, models: Dict[str, Any]) -> Dict[str, Any]:
        """Perform robustness testing with noise injection."""
        logger.info("Performing robustness testing...")
        
        robustness_results = {}
        
        # Split data for robustness testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Noise levels to test
        noise_levels = [0.01, 0.05, 0.1, 0.2]
        
        for model_name, model in models.items():
            try:
                logger.info(f"Robustness testing for {model_name}...")
                
                # Train model on clean data
                model_robust = type(model)(**model.get_params() if hasattr(model, 'get_params') else {})
                model_robust.fit(X_train, y_train)
                
                # Baseline performance on clean test data
                baseline_pred = model_robust.predict(X_test)
                baseline_r2 = r2_score(y_test, baseline_pred)
                baseline_mae = mean_absolute_error(y_test, baseline_pred)
                
                noise_results = {
                    'baseline_r2': float(baseline_r2), 
                    'baseline_mae': float(baseline_mae)
                }
                
                # Test with different noise levels
                for noise_level in noise_levels:
                    # Add Gaussian noise to features
                    np.random.seed(42)  # For reproducibility
                    noise = np.random.normal(0, noise_level, X_test.shape)
                    X_test_noisy = X_test + noise
                    
                    # Predict on noisy data
                    noisy_pred = model_robust.predict(X_test_noisy)
                    noisy_r2 = r2_score(y_test, noisy_pred)
                    noisy_mae = mean_absolute_error(y_test, noisy_pred)
                    
                    # Calculate performance degradation
                    r2_degradation = (baseline_r2 - noisy_r2) / baseline_r2 if baseline_r2 != 0 else 0
                    mae_increase = (noisy_mae - baseline_mae) / baseline_mae if baseline_mae != 0 else 0
                    
                    noise_results[f'noise_{noise_level}'] = {
                        'r2': float(noisy_r2),
                        'mae': float(noisy_mae),
                        'r2_degradation': float(r2_degradation),
                        'mae_increase': float(mae_increase)
                    }
                
                robustness_results[model_name] = noise_results
                
            except Exception as e:
                logger.error(f"Robustness testing failed for {model_name}: {e}")
                robustness_results[model_name] = {'error': str(e)}
        
        return robustness_results
    
    def outlier_sensitivity_analysis(self, X: pd.DataFrame, y: pd.Series, models: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze model sensitivity to outliers."""
        logger.info("Performing outlier sensitivity analysis...")
        
        outlier_results = {}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        for model_name, model in models.items():
            try:
                logger.info(f"Outlier sensitivity for {model_name}...")
                
                # Train on original data
                model_orig = type(model)(**model.get_params() if hasattr(model, 'get_params') else {})
                model_orig.fit(X_train, y_train)
                orig_pred = model_orig.predict(X_test)
                orig_r2 = r2_score(y_test, orig_pred)
                orig_mae = mean_absolute_error(y_test, orig_pred)
                
                # Identify outliers using IQR method
                Q1 = y_train.quantile(0.25)
                Q3 = y_train.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (y_train < lower_bound) | (y_train > upper_bound)
                n_outliers = outlier_mask.sum()
                
                # Train without outliers
                X_train_clean = X_train[~outlier_mask]
                y_train_clean = y_train[~outlier_mask]
                
                model_clean = type(model)(**model.get_params() if hasattr(model, 'get_params') else {})
                model_clean.fit(X_train_clean, y_train_clean)
                clean_pred = model_clean.predict(X_test)
                clean_r2 = r2_score(y_test, clean_pred)
                clean_mae = mean_absolute_error(y_test, clean_pred)
                
                # Calculate sensitivity metrics
                r2_change = clean_r2 - orig_r2
                mae_change = clean_mae - orig_mae
                
                outlier_results[model_name] = {
                    'original_r2': float(orig_r2),
                    'original_mae': float(orig_mae),
                    'clean_r2': float(clean_r2),
                    'clean_mae': float(clean_mae),
                    'r2_change': float(r2_change),
                    'mae_change': float(mae_change),
                    'n_outliers': int(n_outliers),
                    'outlier_percentage': float(n_outliers / len(y_train) * 100),
                    'outlier_sensitivity': float(abs(r2_change))  # Higher means more sensitive
                }
                
            except Exception as e:
                logger.error(f"Outlier sensitivity analysis failed for {model_name}: {e}")
                outlier_results[model_name] = {'error': str(e)}
        
        return outlier_results
    
    def generate_validation_plots(self, cv_results: Dict, robustness_results: Dict, temporal_results: Dict):
        """Generate essential validation plots."""
        try:
            # Cross-validation performance comparison
            self.plot_cv_performance(cv_results)
            
            # Robustness to noise
            self.plot_robustness_analysis(robustness_results)
            
            # Temporal validation results
            self.plot_temporal_validation(temporal_results)
            
        except Exception as e:
            logger.error(f"Failed to generate validation plots: {e}")
    
    def plot_cv_performance(self, cv_results: Dict):
        """Plot cross-validation performance comparison."""
        try:
            models = [name for name in cv_results.keys() if 'error' not in cv_results[name]]
            if not models:
                return
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # R² comparison
            r2_means = [cv_results[model]['r2_mean'] for model in models]
            r2_stds = [cv_results[model]['r2_std'] for model in models]
            
            axes[0].bar(models, r2_means, yerr=r2_stds, capsize=5, alpha=0.7)
            axes[0].set_title('Cross-Validation R² Scores')
            axes[0].set_ylabel('R² Score')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(axis='y', alpha=0.3)
            
            # MAE comparison
            mae_means = [cv_results[model]['mae_mean'] for model in models]
            mae_stds = [cv_results[model]['mae_std'] for model in models]
            
            axes[1].bar(models, mae_means, yerr=mae_stds, capsize=5, alpha=0.7, color='orange')
            axes[1].set_title('Cross-Validation MAE Scores')
            axes[1].set_ylabel('MAE')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plot_file = self.plots_path / "cross_validation_performance.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Cross-validation plot saved: {plot_file}")
            
        except Exception as e:
            logger.error(f"Failed to plot CV performance: {e}")
    
    def plot_robustness_analysis(self, robustness_results: Dict):
        """Plot robustness analysis results."""
        try:
            models = [name for name in robustness_results.keys() if 'error' not in robustness_results[name]]
            if not models:
                return
            
            noise_levels = [0.01, 0.05, 0.1, 0.2]
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # R² degradation
            for model in models:
                r2_degradations = [robustness_results[model][f'noise_{noise}']['r2_degradation'] for noise in noise_levels]
                axes[0].plot(noise_levels, r2_degradations, marker='o', label=model)
            
            axes[0].set_xlabel('Noise Level')
            axes[0].set_ylabel('R² Degradation')
            axes[0].set_title('Model Robustness: R² Degradation vs Noise')
            axes[0].legend()
            axes[0].grid(alpha=0.3)
            
            # MAE increase
            for model in models:
                mae_increases = [robustness_results[model][f'noise_{noise}']['mae_increase'] for noise in noise_levels]
                axes[1].plot(noise_levels, mae_increases, marker='o', label=model)
            
            axes[1].set_xlabel('Noise Level')
            axes[1].set_ylabel('MAE Increase')
            axes[1].set_title('Model Robustness: MAE Increase vs Noise')
            axes[1].legend()
            axes[1].grid(alpha=0.3)
            
            plt.tight_layout()
            plot_file = self.plots_path / "robustness_analysis.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Robustness analysis plot saved: {plot_file}")
            
        except Exception as e:
            logger.error(f"Failed to plot robustness analysis: {e}")
    
    def plot_temporal_validation(self, temporal_results: Dict):
        """Plot temporal validation results."""
        try:
            models = [name for name in temporal_results.keys() if 'error' not in temporal_results[name]]
            if not models:
                return
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # R² comparison (validation vs test)
            val_r2 = [temporal_results[model]['validation_r2'] for model in models]
            test_r2 = [temporal_results[model]['test_r2'] for model in models]
            
            x = np.arange(len(models))
            width = 0.35
            
            axes[0].bar(x - width/2, val_r2, width, label='Validation', alpha=0.7)
            axes[0].bar(x + width/2, test_r2, width, label='Test', alpha=0.7)
            axes[0].set_xlabel('Models')
            axes[0].set_ylabel('R² Score')
            axes[0].set_title('Temporal Validation: R² Scores')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(models, rotation=45)
            axes[0].legend()
            axes[0].grid(axis='y', alpha=0.3)
            
            # MAE comparison (validation vs test)
            val_mae = [temporal_results[model]['validation_mae'] for model in models]
            test_mae = [temporal_results[model]['test_mae'] for model in models]
            
            axes[1].bar(x - width/2, val_mae, width, label='Validation', alpha=0.7, color='orange')
            axes[1].bar(x + width/2, test_mae, width, label='Test', alpha=0.7, color='red')
            axes[1].set_xlabel('Models')
            axes[1].set_ylabel('MAE')
            axes[1].set_title('Temporal Validation: MAE Scores')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(models, rotation=45)
            axes[1].legend()
            axes[1].grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plot_file = self.plots_path / "temporal_validation.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Temporal validation plot saved: {plot_file}")
            
        except Exception as e:
            logger.error(f"Failed to plot temporal validation: {e}")
    
    def run_simplified_external_validation(self) -> Dict[str, Any]:
        """Run simplified external validation and robustness analysis."""
        logger.info("Starting simplified external validation...")
        
        try:
            # Load data
            df = self.load_clean_data()
            X, y = self.prepare_data(df)
            
            # Load or train models
            models = self.load_or_train_models(X, y)
            
            if not models:
                raise ValueError("No models available for validation")
            
            # Perform validation analyses
            logger.info("Performing temporal validation...")
            temporal_results = self.temporal_validation(X, y, models)
            
            logger.info("Performing cross-validation analysis...")
            cv_results = self.cross_validation_analysis(X, y, models)
            
            logger.info("Performing robustness testing...")
            robustness_results = self.robustness_testing(X, y, models)
            
            logger.info("Performing outlier sensitivity analysis...")
            outlier_results = self.outlier_sensitivity_analysis(X, y, models)
            
            # Generate validation plots
            logger.info("Generating validation plots...")
            self.generate_validation_plots(cv_results, robustness_results, temporal_results)
            
            # Compile results
            validation_results = {
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'simplified_external_validation',
                'dataset_info': {
                    'shape': list(df.shape),
                    'features_used': list(X.columns),
                    'target_variable': 'final_test'
                },
                'models_analyzed': list(models.keys()),
                'temporal_validation': temporal_results,
                'cross_validation': cv_results,
                'robustness_testing': robustness_results,
                'outlier_sensitivity': outlier_results,
                'summary': self.generate_validation_summary(temporal_results, cv_results, robustness_results, outlier_results),
                'recommendations': self.generate_validation_recommendations(temporal_results, cv_results, robustness_results, outlier_results)
            }
            
            # Save results
            results_file = self.output_path / "simplified_external_validation.json"
            with open(results_file, 'w') as f:
                json.dump(validation_results, f, indent=2, default=str)
            
            logger.info(f"Simplified external validation completed. Results saved to {results_file}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Simplified external validation failed: {e}")
            raise
    
    def generate_validation_summary(self, temporal_results: Dict, cv_results: Dict, 
                                  robustness_results: Dict, outlier_results: Dict) -> Dict[str, Any]:
        """Generate validation summary."""
        summary = {
            'best_performing_model': None,
            'most_robust_model': None,
            'validation_consistency': {},
            'key_findings': []
        }
        
        # Find best performing model (based on CV R²)
        cv_r2_scores = {}
        for model, results in cv_results.items():
            if 'error' not in results:
                cv_r2_scores[model] = results['r2_mean']
        
        if cv_r2_scores:
            summary['best_performing_model'] = max(cv_r2_scores.items(), key=lambda x: x[1])[0]
        
        # Find most robust model (lowest average robustness degradation)
        robustness_scores = {}
        for model, results in robustness_results.items():
            if 'error' not in results:
                degradations = []
                for noise_level in [0.01, 0.05, 0.1, 0.2]:
                    if f'noise_{noise_level}' in results:
                        degradations.append(abs(results[f'noise_{noise_level}']['r2_degradation']))
                if degradations:
                    robustness_scores[model] = np.mean(degradations)
        
        if robustness_scores:
            summary['most_robust_model'] = min(robustness_scores.items(), key=lambda x: x[1])[0]
        
        # Validation consistency analysis
        for model in cv_results.keys():
            if 'error' not in cv_results.get(model, {}):
                consistency_metrics = {
                    'cv_std': cv_results[model].get('r2_std', 0)
                }
                summary['validation_consistency'][model] = consistency_metrics
        
        # Key findings
        summary['key_findings'].append(f"Best performing model: {summary['best_performing_model']}")
        summary['key_findings'].append(f"Most robust model: {summary['most_robust_model']}")
        summary['key_findings'].append(f"Models analyzed: {len(cv_results)}")
        
        return summary
    
    def generate_validation_recommendations(self, temporal_results: Dict, cv_results: Dict,
                                          robustness_results: Dict, outlier_results: Dict) -> List[str]:
        """Generate validation recommendations."""
        recommendations = []
        
        # General recommendations
        recommendations.append("Use cross-validation as primary model selection criterion")
        recommendations.append("Monitor model performance degradation under noise")
        recommendations.append("Consider temporal stability for production deployment")
        
        # Model-specific recommendations
        if cv_results:
            best_cv_model = None
            best_cv_score = -float('inf')
            
            for model, results in cv_results.items():
                if 'error' not in results:
                    score = results['r2_mean']
                    if score > best_cv_score:
                        best_cv_score = score
                        best_cv_model = model
            
            if best_cv_model:
                recommendations.append(f"Recommend {best_cv_model} based on cross-validation performance (R²={best_cv_score:.3f})")
        
        # Robustness recommendations
        if robustness_results:
            high_sensitivity_models = []
            for model, results in robustness_results.items():
                if 'error' not in results:
                    # Check if model is sensitive to noise
                    if 'noise_0.1' in results and results['noise_0.1']['r2_degradation'] > 0.2:
                        high_sensitivity_models.append(model)
            
            if high_sensitivity_models:
                recommendations.append(f"Models sensitive to noise: {', '.join(high_sensitivity_models)}")
                recommendations.append("Consider feature scaling or regularization for noise-sensitive models")
        
        # Outlier recommendations
        if outlier_results:
            outlier_sensitive_models = []
            for model, results in outlier_results.items():
                if 'error' not in results and results.get('outlier_sensitivity', 0) > 0.1:
                    outlier_sensitive_models.append(model)
            
            if outlier_sensitive_models:
                recommendations.append(f"Models sensitive to outliers: {', '.join(outlier_sensitive_models)}")
                recommendations.append("Consider outlier detection and removal for sensitive models")
        
        # Temporal validation recommendations
        if temporal_results:
            recommendations.append("Monitor temporal stability of model performance")
            recommendations.append("Consider retraining frequency based on temporal validation results")
        
        return recommendations

def main():
    """Main execution function."""
    try:
        validator = SimplifiedExternalValidator()
        results = validator.run_simplified_external_validation()
        
        print("\n=== SIMPLIFIED EXTERNAL VALIDATION COMPLETED ===")
        print(f"Timestamp: {results['timestamp']}")
        print(f"Models Analyzed: {len(results['models_analyzed'])}")
        
        print("\n=== VALIDATION SUMMARY ===")
        summary = results['summary']
        print(f"Best Performing Model: {summary['best_performing_model']}")
        print(f"Most Robust Model: {summary['most_robust_model']}")
        
        print("\n=== KEY FINDINGS ===")
        for finding in summary['key_findings']:
            print(f"- {finding}")
        
        print("\n=== RECOMMENDATIONS ===")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"{i}. {rec}")
        
        print("\n=== EXTERNAL VALIDATION COMPLETE ===")
        
    except Exception as e:
        logger.error(f"External validation failed: {e}")
        raise

if __name__ == "__main__":
    main()