#!/usr/bin/env python3
"""
Phase 6 Task 6.1.3: Model Comparison Investigation

This module conducts comprehensive comparison of all models after the data leakage fix,
including performance analysis, feature importance, learning curves, and statistical tests.

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
from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold, learning_curve,
    validation_curve, StratifiedKFold
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    explained_variance_score, max_error
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from scipy import stats
from scipy.stats import pearsonr, spearmanr, normaltest, shapiro
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelComparisonInvestigator:
    """
    Conducts comprehensive comparison and analysis of all models
    after data leakage fixes to understand performance patterns.
    """
    
    def __init__(self, project_root: str = None):
        if project_root is None:
            # Get the absolute path to the project root
            current_file = Path(__file__).resolve()
            self.project_root = current_file.parents[2]  # Go up 2 levels from src/modeling/
        else:
            self.project_root = Path(project_root)
        
        self.data_path = self.project_root / "data" / "modeling_outputs"
        self.output_path = self.project_root / "data" / "modeling_outputs"
        self.plots_path = self.output_path / "model_comparison_plots"
        
        # Ensure output directories exist
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.plots_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"Output path: {self.output_path}")
        logger.info(f"Plots path: {self.plots_path}")
    
    def load_clean_data(self) -> pd.DataFrame:
        """Load the cleaned dataset without data leakage."""
        # Try the latest cleaned dataset first
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
        logger.info("Preparing data for modeling...")
        
        # Separate features and target
        X = df.drop(['final_test'], axis=1)
        y = df['final_test']
        
        # Remove rows with missing target
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        # Get numerical features only for this analysis
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
        logger.info(f"Features: {list(X_filled.columns)}")
        
        return X_filled, y
    
    def get_model_configurations(self) -> Dict[str, Any]:
        """Get all model configurations for comparison."""
        models = {
            'LinearRegression': {
                'model': LinearRegression(),
                'params': {},
                'interpretable': True
            },
            'Ridge': {
                'model': Ridge(alpha=1.0),
                'params': {'alpha': 1.0},
                'interpretable': True
            },
            'Lasso': {
                'model': Lasso(alpha=1.0, max_iter=2000),
                'params': {'alpha': 1.0, 'max_iter': 2000},
                'interpretable': True
            },
            'RandomForest': {
                'model': RandomForestRegressor(n_estimators=100, random_state=42),
                'params': {'n_estimators': 100, 'random_state': 42},
                'interpretable': False
            },
            'GradientBoosting': {
                'model': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'params': {'n_estimators': 100, 'random_state': 42},
                'interpretable': False
            },
            'SVR': {
                'model': SVR(kernel='rbf', C=1.0),
                'params': {'kernel': 'rbf', 'C': 1.0},
                'interpretable': False
            },
            'MLPRegressor': {
                'model': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
                'params': {'hidden_layer_sizes': (100, 50), 'max_iter': 500, 'random_state': 42},
                'interpretable': False
            }
        }
        
        return models
    
    def evaluate_model_performance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Comprehensive evaluation of all models."""
        logger.info("Evaluating model performance...")
        
        models = self.get_model_configurations()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features for models that need it
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        performance_results = {
            'data_info': {
                'train_size': len(X_train),
                'test_size': len(X_test),
                'feature_count': len(X.columns),
                'target_stats': {
                    'mean': float(y.mean()),
                    'std': float(y.std()),
                    'min': float(y.min()),
                    'max': float(y.max())
                }
            },
            'model_results': {}
        }
        
        for model_name, model_config in models.items():
            logger.info(f"Evaluating {model_name}...")
            
            try:
                model = model_config['model']
                
                # Use scaled data for SVR and MLP
                if model_name in ['SVR', 'MLPRegressor']:
                    X_train_use = X_train_scaled
                    X_test_use = X_test_scaled
                else:
                    X_train_use = X_train
                    X_test_use = X_test
                
                # Train model
                model.fit(X_train_use, y_train)
                
                # Predictions
                y_pred_train = model.predict(X_train_use)
                y_pred_test = model.predict(X_test_use)
                
                # Calculate metrics
                train_metrics = self.calculate_metrics(y_train, y_pred_train)
                test_metrics = self.calculate_metrics(y_test, y_pred_test)
                
                # Cross-validation
                cv_scores = self.cross_validate_model(model, X, y, model_name)
                
                # Feature importance (if available)
                feature_importance = self.get_feature_importance(model, X_train_use, y_train, model_name)
                
                model_result = {
                    'model_config': model_config['params'],
                    'interpretable': model_config['interpretable'],
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'cv_results': cv_scores,
                    'feature_importance': feature_importance,
                    'overfitting_indicators': {
                        'mae_ratio': float(test_metrics['mae'] / train_metrics['mae']) if train_metrics['mae'] > 0 else 1.0,
                        'r2_difference': float(train_metrics['r2'] - test_metrics['r2']),
                        'rmse_ratio': float(test_metrics['rmse'] / train_metrics['rmse']) if train_metrics['rmse'] > 0 else 1.0
                    }
                }
                
                performance_results['model_results'][model_name] = model_result
                
                # Save model
                model_file = self.output_path / f"models/{model_name.lower()}_comparison.joblib"
                model_file.parent.mkdir(exist_ok=True)
                joblib.dump(model, model_file)
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                performance_results['model_results'][model_name] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        return performance_results
    
    def calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive regression metrics."""
        return {
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'mse': float(mean_squared_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'r2': float(r2_score(y_true, y_pred)),
            'explained_variance': float(explained_variance_score(y_true, y_pred)),
            'max_error': float(max_error(y_true, y_pred))
        }
    
    def cross_validate_model(self, model, X: pd.DataFrame, y: pd.Series, model_name: str) -> Dict[str, Any]:
        """Perform cross-validation analysis."""
        try:
            # Use scaled data for models that need it
            if model_name in ['SVR', 'MLPRegressor']:
                scaler = StandardScaler()
                X_scaled = pd.DataFrame(
                    scaler.fit_transform(X),
                    columns=X.columns,
                    index=X.index
                )
                X_use = X_scaled
            else:
                X_use = X
            
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            
            # Multiple scoring metrics
            scoring_metrics = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
            cv_results = {}
            
            for metric in scoring_metrics:
                scores = cross_val_score(model, X_use, y, cv=cv, scoring=metric, n_jobs=1)
                
                if metric.startswith('neg_'):
                    scores = -scores  # Convert negative scores to positive
                    metric_name = metric[4:]  # Remove 'neg_' prefix
                else:
                    metric_name = metric
                
                cv_results[metric_name] = {
                    'scores': scores.tolist(),
                    'mean': float(scores.mean()),
                    'std': float(scores.std()),
                    'min': float(scores.min()),
                    'max': float(scores.max())
                }
            
            return cv_results
            
        except Exception as e:
            logger.warning(f"Cross-validation failed for {model_name}: {e}")
            return {'error': str(e)}
    
    def get_feature_importance(self, model, X: pd.DataFrame, y: pd.Series, model_name: str) -> Dict[str, Any]:
        """Extract feature importance using various methods."""
        importance_results = {}
        
        try:
            # Method 1: Built-in feature importance (for tree-based models)
            if hasattr(model, 'feature_importances_'):
                importance_results['built_in'] = {
                    'values': model.feature_importances_.tolist(),
                    'features': X.columns.tolist()
                }
            
            # Method 2: Coefficients (for linear models)
            if hasattr(model, 'coef_'):
                importance_results['coefficients'] = {
                    'values': model.coef_.tolist() if hasattr(model.coef_, 'tolist') else [float(model.coef_)],
                    'features': X.columns.tolist(),
                    'abs_values': np.abs(model.coef_).tolist() if hasattr(model.coef_, 'tolist') else [float(abs(model.coef_))]
                }
            
            # Method 3: Permutation importance (for all models)
            perm_importance = permutation_importance(
                model, X, y, n_repeats=5, random_state=42, n_jobs=1
            )
            
            importance_results['permutation'] = {
                'importances_mean': perm_importance.importances_mean.tolist(),
                'importances_std': perm_importance.importances_std.tolist(),
                'features': X.columns.tolist()
            }
            
        except Exception as e:
            logger.warning(f"Feature importance extraction failed for {model_name}: {e}")
            importance_results['error'] = str(e)
        
        return importance_results
    
    def generate_learning_curves(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Generate learning curves for all models."""
        logger.info("Generating learning curves...")
        
        models = self.get_model_configurations()
        learning_curve_results = {}
        
        # Training sizes to test
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        for model_name, model_config in models.items():
            logger.info(f"Generating learning curve for {model_name}...")
            
            try:
                model = model_config['model']
                
                # Use scaled data for models that need it
                if model_name in ['SVR', 'MLPRegressor']:
                    scaler = StandardScaler()
                    X_scaled = pd.DataFrame(
                        scaler.fit_transform(X),
                        columns=X.columns,
                        index=X.index
                    )
                    X_use = X_scaled
                else:
                    X_use = X
                
                # Generate learning curve
                train_sizes_abs, train_scores, val_scores = learning_curve(
                    model, X_use, y,
                    train_sizes=train_sizes,
                    cv=5,
                    scoring='r2',
                    n_jobs=1,
                    random_state=42
                )
                
                learning_curve_results[model_name] = {
                    'train_sizes': train_sizes_abs.tolist(),
                    'train_scores_mean': train_scores.mean(axis=1).tolist(),
                    'train_scores_std': train_scores.std(axis=1).tolist(),
                    'val_scores_mean': val_scores.mean(axis=1).tolist(),
                    'val_scores_std': val_scores.std(axis=1).tolist()
                }
                
                # Create learning curve plot
                self.plot_learning_curve(model_name, learning_curve_results[model_name])
                
            except Exception as e:
                logger.error(f"Learning curve generation failed for {model_name}: {e}")
                learning_curve_results[model_name] = {'error': str(e)}
        
        return learning_curve_results
    
    def plot_learning_curve(self, model_name: str, curve_data: Dict[str, Any]):
        """Plot learning curve for a specific model."""
        try:
            plt.figure(figsize=(10, 6))
            
            train_sizes = curve_data['train_sizes']
            train_mean = curve_data['train_scores_mean']
            train_std = curve_data['train_scores_std']
            val_mean = curve_data['val_scores_mean']
            val_std = curve_data['val_scores_std']
            
            plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
            plt.fill_between(train_sizes, 
                           np.array(train_mean) - np.array(train_std),
                           np.array(train_mean) + np.array(train_std),
                           alpha=0.1, color='blue')
            
            plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
            plt.fill_between(train_sizes,
                           np.array(val_mean) - np.array(val_std),
                           np.array(val_mean) + np.array(val_std),
                           alpha=0.1, color='red')
            
            plt.xlabel('Training Set Size')
            plt.ylabel('R² Score')
            plt.title(f'Learning Curve - {model_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_file = self.plots_path / f"{model_name.lower()}_learning_curve.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Learning curve plot saved: {plot_file}")
            
        except Exception as e:
            logger.error(f"Failed to plot learning curve for {model_name}: {e}")
    
    def perform_statistical_tests(self, performance_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical tests on model performance."""
        logger.info("Performing statistical tests...")
        
        statistical_results = {
            'normality_tests': {},
            'performance_comparisons': {},
            'significance_tests': {}
        }
        
        # Extract R² scores for comparison
        model_r2_scores = {}
        model_mae_scores = {}
        
        for model_name, results in performance_results['model_results'].items():
            if 'cv_results' in results and 'r2' in results['cv_results']:
                model_r2_scores[model_name] = results['cv_results']['r2']['scores']
                model_mae_scores[model_name] = results['cv_results']['mean_absolute_error']['scores']
        
        # Normality tests
        for model_name, scores in model_r2_scores.items():
            try:
                shapiro_stat, shapiro_p = shapiro(scores)
                statistical_results['normality_tests'][model_name] = {
                    'shapiro_stat': float(shapiro_stat),
                    'shapiro_p': float(shapiro_p),
                    'is_normal': bool(shapiro_p > 0.05)
                }
            except Exception as e:
                logger.warning(f"Normality test failed for {model_name}: {e}")
        
        # Performance ranking
        model_performance_ranking = []
        for model_name, results in performance_results['model_results'].items():
            if 'test_metrics' in results:
                model_performance_ranking.append({
                    'model': model_name,
                    'test_r2': results['test_metrics']['r2'],
                    'test_mae': results['test_metrics']['mae'],
                    'cv_r2_mean': results.get('cv_results', {}).get('r2', {}).get('mean', 0)
                })
        
        # Sort by test R²
        model_performance_ranking.sort(key=lambda x: x['test_r2'], reverse=True)
        statistical_results['performance_ranking'] = model_performance_ranking
        
        # Pairwise comparisons (if we have enough models)
        if len(model_r2_scores) >= 2:
            model_names = list(model_r2_scores.keys())
            for i, model1 in enumerate(model_names):
                for model2 in model_names[i+1:]:
                    try:
                        # Paired t-test
                        t_stat, t_p = stats.ttest_rel(model_r2_scores[model1], model_r2_scores[model2])
                        
                        comparison_key = f"{model1}_vs_{model2}"
                        statistical_results['significance_tests'][comparison_key] = {
                            'ttest_stat': float(t_stat),
                            'ttest_p': float(t_p),
                            'significant_difference': bool(t_p < 0.05),
                            'better_model': model1 if np.mean(model_r2_scores[model1]) > np.mean(model_r2_scores[model2]) else model2
                        }
                    except Exception as e:
                        logger.warning(f"Statistical test failed for {model1} vs {model2}: {e}")
        
        return statistical_results
    
    def generate_comparison_visualizations(self, performance_results: Dict[str, Any]):
        """Generate comprehensive comparison visualizations."""
        logger.info("Generating comparison visualizations...")
        
        try:
            # Model performance comparison
            self.plot_model_performance_comparison(performance_results)
            
            # Feature importance comparison
            self.plot_feature_importance_comparison(performance_results)
            
            # Overfitting analysis
            self.plot_overfitting_analysis(performance_results)
            
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
    
    def plot_model_performance_comparison(self, performance_results: Dict[str, Any]):
        """Plot model performance comparison."""
        try:
            models = []
            test_r2 = []
            test_mae = []
            cv_r2_mean = []
            cv_r2_std = []
            
            for model_name, results in performance_results['model_results'].items():
                if 'test_metrics' in results and 'cv_results' in results:
                    models.append(model_name)
                    test_r2.append(results['test_metrics']['r2'])
                    test_mae.append(results['test_metrics']['mae'])
                    cv_r2_mean.append(results['cv_results'].get('r2', {}).get('mean', 0))
                    cv_r2_std.append(results['cv_results'].get('r2', {}).get('std', 0))
            
            if not models:
                return
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Test R² comparison
            ax1.bar(models, test_r2, color='skyblue', alpha=0.7)
            ax1.set_title('Test R² Score Comparison')
            ax1.set_ylabel('R² Score')
            ax1.tick_params(axis='x', rotation=45)
            
            # Test MAE comparison
            ax2.bar(models, test_mae, color='lightcoral', alpha=0.7)
            ax2.set_title('Test MAE Comparison')
            ax2.set_ylabel('Mean Absolute Error')
            ax2.tick_params(axis='x', rotation=45)
            
            # CV R² with error bars
            ax3.bar(models, cv_r2_mean, yerr=cv_r2_std, color='lightgreen', alpha=0.7, capsize=5)
            ax3.set_title('Cross-Validation R² Score (with std)')
            ax3.set_ylabel('R² Score')
            ax3.tick_params(axis='x', rotation=45)
            
            # Performance scatter
            ax4.scatter(test_r2, test_mae, s=100, alpha=0.7)
            for i, model in enumerate(models):
                ax4.annotate(model, (test_r2[i], test_mae[i]), xytext=(5, 5), textcoords='offset points')
            ax4.set_xlabel('Test R² Score')
            ax4.set_ylabel('Test MAE')
            ax4.set_title('Performance Trade-off (R² vs MAE)')
            
            plt.tight_layout()
            plot_file = self.plots_path / "model_performance_comparison.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Performance comparison plot saved: {plot_file}")
            
        except Exception as e:
            logger.error(f"Failed to plot performance comparison: {e}")
    
    def plot_feature_importance_comparison(self, performance_results: Dict[str, Any]):
        """Plot feature importance comparison for interpretable models."""
        try:
            interpretable_models = {}
            
            for model_name, results in performance_results['model_results'].items():
                if (results.get('interpretable', False) and 
                    'feature_importance' in results and 
                    'permutation' in results['feature_importance']):
                    interpretable_models[model_name] = results['feature_importance']['permutation']
            
            if not interpretable_models:
                logger.warning("No interpretable models with feature importance found")
                return
            
            # Get common features
            all_features = None
            for model_data in interpretable_models.values():
                features = model_data['features']
                if all_features is None:
                    all_features = set(features)
                else:
                    all_features = all_features.intersection(set(features))
            
            if not all_features:
                logger.warning("No common features found across models")
                return
            
            common_features = sorted(list(all_features))
            
            # Create comparison plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            x = np.arange(len(common_features))
            width = 0.8 / len(interpretable_models)
            
            for i, (model_name, importance_data) in enumerate(interpretable_models.items()):
                features = importance_data['features']
                importances = importance_data['importances_mean']
                
                # Align importances with common features
                aligned_importances = []
                for feature in common_features:
                    if feature in features:
                        idx = features.index(feature)
                        aligned_importances.append(importances[idx])
                    else:
                        aligned_importances.append(0)
                
                ax.bar(x + i * width, aligned_importances, width, label=model_name, alpha=0.7)
            
            ax.set_xlabel('Features')
            ax.set_ylabel('Permutation Importance')
            ax.set_title('Feature Importance Comparison (Interpretable Models)')
            ax.set_xticks(x + width * (len(interpretable_models) - 1) / 2)
            ax.set_xticklabels(common_features, rotation=45, ha='right')
            ax.legend()
            
            plt.tight_layout()
            plot_file = self.plots_path / "feature_importance_comparison.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Feature importance comparison plot saved: {plot_file}")
            
        except Exception as e:
            logger.error(f"Failed to plot feature importance comparison: {e}")
    
    def plot_overfitting_analysis(self, performance_results: Dict[str, Any]):
        """Plot overfitting analysis."""
        try:
            models = []
            mae_ratios = []
            r2_differences = []
            
            for model_name, results in performance_results['model_results'].items():
                if 'overfitting_indicators' in results:
                    models.append(model_name)
                    mae_ratios.append(results['overfitting_indicators']['mae_ratio'])
                    r2_differences.append(results['overfitting_indicators']['r2_difference'])
            
            if not models:
                return
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # MAE ratio (test/train)
            colors = ['red' if ratio > 1.5 else 'orange' if ratio > 1.2 else 'green' for ratio in mae_ratios]
            ax1.bar(models, mae_ratios, color=colors, alpha=0.7)
            ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Perfect fit line')
            ax1.axhline(y=1.2, color='orange', linestyle='--', alpha=0.5, label='Moderate overfitting')
            ax1.axhline(y=1.5, color='red', linestyle='--', alpha=0.5, label='High overfitting')
            ax1.set_title('Overfitting Analysis - MAE Ratio (Test/Train)')
            ax1.set_ylabel('MAE Ratio')
            ax1.tick_params(axis='x', rotation=45)
            ax1.legend()
            
            # R² difference (train - test)
            colors = ['red' if diff > 0.2 else 'orange' if diff > 0.1 else 'green' for diff in r2_differences]
            ax2.bar(models, r2_differences, color=colors, alpha=0.7)
            ax2.axhline(y=0.0, color='black', linestyle='--', alpha=0.5, label='No difference')
            ax2.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Moderate overfitting')
            ax2.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='High overfitting')
            ax2.set_title('Overfitting Analysis - R² Difference (Train - Test)')
            ax2.set_ylabel('R² Difference')
            ax2.tick_params(axis='x', rotation=45)
            ax2.legend()
            
            plt.tight_layout()
            plot_file = self.plots_path / "overfitting_analysis.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Overfitting analysis plot saved: {plot_file}")
            
        except Exception as e:
            logger.error(f"Failed to plot overfitting analysis: {e}")
    
    def run_comprehensive_comparison(self) -> Dict[str, Any]:
        """Run the complete model comparison analysis."""
        logger.info("Starting comprehensive model comparison...")
        
        try:
            # Load data
            df = self.load_clean_data()
            X, y = self.prepare_data(df)
            
            # Evaluate all models
            performance_results = self.evaluate_model_performance(X, y)
            
            # Generate learning curves
            learning_curves = self.generate_learning_curves(X, y)
            
            # Perform statistical tests
            statistical_results = self.perform_statistical_tests(performance_results)
            
            # Generate visualizations
            self.generate_comparison_visualizations(performance_results)
            
            # Compile final results
            comparison_results = {
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'comprehensive_model_comparison',
                'dataset_info': {
                    'shape': list(df.shape),
                    'features_used': list(X.columns),
                    'target_variable': 'final_test'
                },
                'performance_results': performance_results,
                'learning_curves': learning_curves,
                'statistical_analysis': statistical_results,
                'summary': self.generate_comparison_summary(performance_results, statistical_results),
                'recommendations': self.generate_comparison_recommendations(performance_results, statistical_results)
            }
            
            # Save results
            results_file = self.output_path / "model_comparison_investigation.json"
            with open(results_file, 'w') as f:
                json.dump(comparison_results, f, indent=2, default=str)
            
            logger.info(f"Model comparison completed. Results saved to {results_file}")
            
            return comparison_results
            
        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            raise
    
    def generate_comparison_summary(self, performance_results: Dict[str, Any], statistical_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of comparison findings."""
        summary = {
            'best_performing_models': [],
            'overfitting_concerns': [],
            'interpretability_insights': [],
            'performance_patterns': []
        }
        
        # Identify best performing models
        if 'performance_ranking' in statistical_results:
            top_3 = statistical_results['performance_ranking'][:3]
            summary['best_performing_models'] = [
                f"{model['model']}: R²={model['test_r2']:.3f}, MAE={model['test_mae']:.3f}"
                for model in top_3
            ]
        
        # Identify overfitting concerns
        for model_name, results in performance_results['model_results'].items():
            if 'overfitting_indicators' in results:
                indicators = results['overfitting_indicators']
                if indicators['mae_ratio'] > 1.5 or indicators['r2_difference'] > 0.2:
                    summary['overfitting_concerns'].append(
                        f"{model_name}: MAE ratio={indicators['mae_ratio']:.2f}, R² diff={indicators['r2_difference']:.3f}"
                    )
        
        # Interpretability insights
        interpretable_models = []
        for model_name, results in performance_results['model_results'].items():
            if results.get('interpretable', False) and 'test_metrics' in results:
                interpretable_models.append({
                    'model': model_name,
                    'r2': results['test_metrics']['r2']
                })
        
        if interpretable_models:
            best_interpretable = max(interpretable_models, key=lambda x: x['r2'])
            summary['interpretability_insights'].append(
                f"Best interpretable model: {best_interpretable['model']} (R²={best_interpretable['r2']:.3f})"
            )
        
        return summary
    
    def generate_comparison_recommendations(self, performance_results: Dict[str, Any], statistical_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on comparison."""
        recommendations = []
        
        # Performance-based recommendations
        if 'performance_ranking' in statistical_results and statistical_results['performance_ranking']:
            best_model = statistical_results['performance_ranking'][0]
            recommendations.append(f"Consider {best_model['model']} as primary model (highest R²: {best_model['test_r2']:.3f})")
        
        # Overfitting recommendations
        overfitting_models = []
        for model_name, results in performance_results['model_results'].items():
            if 'overfitting_indicators' in results:
                indicators = results['overfitting_indicators']
                if indicators['mae_ratio'] > 1.3:
                    overfitting_models.append(model_name)
        
        if overfitting_models:
            recommendations.append(f"Address overfitting in: {', '.join(overfitting_models)}")
            recommendations.append("Consider regularization, cross-validation, or ensemble methods")
        
        # Model selection recommendations
        recommendations.append("Evaluate trade-off between performance and interpretability")
        recommendations.append("Consider ensemble methods combining top performers")
        recommendations.append("Validate final model selection with additional external data")
        
        return recommendations

def main():
    """Main execution function."""
    try:
        investigator = ModelComparisonInvestigator()
        results = investigator.run_comprehensive_comparison()
        
        print("\n=== MODEL COMPARISON INVESTIGATION COMPLETED ===")
        print(f"Timestamp: {results['timestamp']}")
        print(f"Models analyzed: {len(results['performance_results']['model_results'])}")
        
        print("\n=== BEST PERFORMING MODELS ===")
        for model_info in results['summary']['best_performing_models']:
            print(f"- {model_info}")
        
        print("\n=== OVERFITTING CONCERNS ===")
        if results['summary']['overfitting_concerns']:
            for concern in results['summary']['overfitting_concerns']:
                print(f"- {concern}")
        else:
            print("- No significant overfitting detected")
        
        print("\n=== RECOMMENDATIONS ===")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"{i}. {rec}")
        
        print("\n=== COMPARISON COMPLETE ===")
        
    except Exception as e:
        logger.error(f"Model comparison failed: {e}")
        raise

if __name__ == "__main__":
    main()