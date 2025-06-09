#!/usr/bin/env python3
"""
Phase 6 Task 6.2.1: SHAP Analysis Implementation Fix

This module fixes the SHAP analysis implementation for scikit-learn models
that failed in Phase 5, providing working model interpretability analysis.

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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

# SHAP imports with error handling
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Installing...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
        import shap
        SHAP_AVAILABLE = True
    except Exception as e:
        print(f"Failed to install SHAP: {e}")
        SHAP_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SHAPAnalysisFixer:
    """
    Fixes and implements working SHAP analysis for scikit-learn models
    with proper pipeline compatibility and error handling.
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
        self.plots_path = self.output_path / "shap_analysis_plots"
        
        # Ensure output directories exist
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.plots_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"Output path: {self.output_path}")
        logger.info(f"Plots path: {self.plots_path}")
        
        if not SHAP_AVAILABLE:
            logger.warning("SHAP is not available. Will use alternative interpretability methods.")
    
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
    
    def load_trained_models(self) -> Dict[str, Any]:
        """Load previously trained models."""
        models = {}
        model_files = {
            'LinearRegression': 'linearregression_comparison.joblib',
            'Ridge': 'ridge_comparison.joblib',
            'Lasso': 'lasso_comparison.joblib',
            'RandomForest': 'randomforest_comparison.joblib',
            'GradientBoosting': 'gradientboosting_comparison.joblib'
        }
        
        models_dir = self.data_path / "models"
        
        for model_name, filename in model_files.items():
            model_file = models_dir / filename
            if model_file.exists():
                try:
                    model = joblib.load(model_file)
                    models[model_name] = model
                    logger.info(f"Loaded {model_name} from {model_file}")
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")
            else:
                logger.warning(f"Model file not found: {model_file}")
        
        return models
    
    def train_models_if_needed(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train models if they're not available."""
        logger.info("Training models for SHAP analysis...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=1.0, max_iter=2000),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        trained_models = {}
        
        for model_name, model in models.items():
            try:
                logger.info(f"Training {model_name}...")
                model.fit(X_train, y_train)
                trained_models[model_name] = {
                    'model': model,
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test
                }
                
                # Save model
                model_file = self.output_path / f"models/{model_name.lower()}_shap.joblib"
                model_file.parent.mkdir(exist_ok=True)
                joblib.dump(model, model_file)
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
        
        return trained_models
    
    def analyze_with_shap(self, models: Dict[str, Any], X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Perform SHAP analysis on trained models."""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available, skipping SHAP analysis")
            return {}
        
        logger.info("Starting SHAP analysis...")
        
        shap_results = {}
        
        # Use a subset of data for SHAP analysis to avoid memory issues
        sample_size = min(1000, len(X))
        X_sample = X.sample(n=sample_size, random_state=42)
        
        for model_name, model_data in models.items():
            logger.info(f"SHAP analysis for {model_name}...")
            
            try:
                model = model_data['model'] if isinstance(model_data, dict) else model_data
                
                # Choose appropriate SHAP explainer based on model type
                if model_name in ['LinearRegression', 'Ridge', 'Lasso']:
                    # For linear models, use LinearExplainer
                    explainer = shap.LinearExplainer(model, X_sample)
                    shap_values = explainer.shap_values(X_sample)
                    
                elif model_name in ['RandomForest', 'GradientBoosting']:
                    # For tree-based models, use TreeExplainer
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample)
                    
                else:
                    # For other models, use KernelExplainer (slower but more general)
                    explainer = shap.KernelExplainer(model.predict, X_sample.iloc[:100])  # Use smaller background
                    shap_values = explainer.shap_values(X_sample.iloc[:200])  # Analyze smaller subset
                
                # Store SHAP results
                shap_results[model_name] = {
                    'explainer': explainer,
                    'shap_values': shap_values,
                    'feature_names': list(X_sample.columns),
                    'X_sample': X_sample,
                    'expected_value': explainer.expected_value if hasattr(explainer, 'expected_value') else model.predict(X_sample).mean()
                }
                
                # Generate SHAP plots
                self.generate_shap_plots(model_name, shap_results[model_name])
                
                logger.info(f"SHAP analysis completed for {model_name}")
                
            except Exception as e:
                logger.error(f"SHAP analysis failed for {model_name}: {e}")
                shap_results[model_name] = {'error': str(e)}
        
        return shap_results
    
    def generate_shap_plots(self, model_name: str, shap_data: Dict[str, Any]):
        """Generate SHAP visualization plots."""
        try:
            shap_values = shap_data['shap_values']
            X_sample = shap_data['X_sample']
            feature_names = shap_data['feature_names']
            
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
            plt.title(f'SHAP Summary Plot - {model_name}')
            plt.tight_layout()
            summary_file = self.plots_path / f"{model_name.lower()}_shap_summary.png"
            plt.savefig(summary_file, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"SHAP summary plot saved: {summary_file}")
            
            # Feature importance plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
            plt.title(f'SHAP Feature Importance - {model_name}')
            plt.tight_layout()
            importance_file = self.plots_path / f"{model_name.lower()}_shap_importance.png"
            plt.savefig(importance_file, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"SHAP importance plot saved: {importance_file}")
            
            # Waterfall plot for first prediction
            if len(shap_values) > 0:
                plt.figure(figsize=(10, 8))
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_values[0],
                        base_values=shap_data['expected_value'],
                        data=X_sample.iloc[0].values,
                        feature_names=feature_names
                    ),
                    show=False
                )
                plt.title(f'SHAP Waterfall Plot - {model_name} (First Prediction)')
                plt.tight_layout()
                waterfall_file = self.plots_path / f"{model_name.lower()}_shap_waterfall.png"
                plt.savefig(waterfall_file, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"SHAP waterfall plot saved: {waterfall_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate SHAP plots for {model_name}: {e}")
    
    def alternative_interpretability_analysis(self, models: Dict[str, Any], X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Perform alternative interpretability analysis when SHAP fails."""
        logger.info("Performing alternative interpretability analysis...")
        
        interpretability_results = {}
        
        for model_name, model_data in models.items():
            logger.info(f"Alternative analysis for {model_name}...")
            
            try:
                model = model_data['model'] if isinstance(model_data, dict) else model_data
                
                result = {
                    'model_name': model_name,
                    'feature_names': list(X.columns)
                }
                
                # Method 1: Built-in feature importance (for tree-based models)
                if hasattr(model, 'feature_importances_'):
                    result['built_in_importance'] = {
                        'values': model.feature_importances_.tolist(),
                        'features': list(X.columns)
                    }
                    
                    # Create feature importance plot
                    self.plot_feature_importance(model_name, result['built_in_importance'])
                
                # Method 2: Coefficients (for linear models)
                if hasattr(model, 'coef_'):
                    coef_values = model.coef_.tolist() if hasattr(model.coef_, 'tolist') else [float(model.coef_)]
                    result['coefficients'] = {
                        'values': coef_values,
                        'abs_values': [abs(x) for x in coef_values],
                        'features': list(X.columns)
                    }
                    
                    # Create coefficient plot
                    self.plot_coefficients(model_name, result['coefficients'])
                
                # Method 3: Permutation importance (for all models)
                from sklearn.inspection import permutation_importance
                
                # Use a subset for faster computation
                X_subset = X.iloc[:min(500, len(X))]
                y_subset = y.iloc[:min(500, len(y))]
                
                perm_importance = permutation_importance(
                    model, X_subset, y_subset, n_repeats=5, random_state=42, n_jobs=1
                )
                
                result['permutation_importance'] = {
                    'importances_mean': perm_importance.importances_mean.tolist(),
                    'importances_std': perm_importance.importances_std.tolist(),
                    'features': list(X.columns)
                }
                
                # Create permutation importance plot
                self.plot_permutation_importance(model_name, result['permutation_importance'])
                
                interpretability_results[model_name] = result
                
            except Exception as e:
                logger.error(f"Alternative analysis failed for {model_name}: {e}")
                interpretability_results[model_name] = {'error': str(e)}
        
        return interpretability_results
    
    def plot_feature_importance(self, model_name: str, importance_data: Dict[str, Any]):
        """Plot built-in feature importance."""
        try:
            features = importance_data['features']
            values = importance_data['values']
            
            # Get top 15 features
            top_indices = np.argsort(values)[-15:]
            top_features = [features[i] for i in top_indices]
            top_values = [values[i] for i in top_indices]
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(top_features)), top_values)
            plt.yticks(range(len(top_features)), top_features)
            plt.xlabel('Feature Importance')
            plt.title(f'Built-in Feature Importance - {model_name}')
            plt.tight_layout()
            
            plot_file = self.plots_path / f"{model_name.lower()}_builtin_importance.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Feature importance plot saved: {plot_file}")
            
        except Exception as e:
            logger.error(f"Failed to plot feature importance for {model_name}: {e}")
    
    def plot_coefficients(self, model_name: str, coef_data: Dict[str, Any]):
        """Plot linear model coefficients."""
        try:
            features = coef_data['features']
            values = coef_data['values']
            abs_values = coef_data['abs_values']
            
            # Get top 15 features by absolute coefficient value
            top_indices = np.argsort(abs_values)[-15:]
            top_features = [features[i] for i in top_indices]
            top_values = [values[i] for i in top_indices]
            
            plt.figure(figsize=(10, 8))
            colors = ['red' if x < 0 else 'blue' for x in top_values]
            plt.barh(range(len(top_features)), top_values, color=colors, alpha=0.7)
            plt.yticks(range(len(top_features)), top_features)
            plt.xlabel('Coefficient Value')
            plt.title(f'Linear Model Coefficients - {model_name}')
            plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            plt.tight_layout()
            
            plot_file = self.plots_path / f"{model_name.lower()}_coefficients.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Coefficients plot saved: {plot_file}")
            
        except Exception as e:
            logger.error(f"Failed to plot coefficients for {model_name}: {e}")
    
    def plot_permutation_importance(self, model_name: str, perm_data: Dict[str, Any]):
        """Plot permutation importance."""
        try:
            features = perm_data['features']
            importances_mean = perm_data['importances_mean']
            importances_std = perm_data['importances_std']
            
            # Get top 15 features
            top_indices = np.argsort(importances_mean)[-15:]
            top_features = [features[i] for i in top_indices]
            top_means = [importances_mean[i] for i in top_indices]
            top_stds = [importances_std[i] for i in top_indices]
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(top_features)), top_means, xerr=top_stds, capsize=3, alpha=0.7)
            plt.yticks(range(len(top_features)), top_features)
            plt.xlabel('Permutation Importance')
            plt.title(f'Permutation Importance - {model_name}')
            plt.tight_layout()
            
            plot_file = self.plots_path / f"{model_name.lower()}_permutation_importance.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Permutation importance plot saved: {plot_file}")
            
        except Exception as e:
            logger.error(f"Failed to plot permutation importance for {model_name}: {e}")
    
    def run_shap_analysis_fix(self) -> Dict[str, Any]:
        """Run the complete SHAP analysis fix."""
        logger.info("Starting SHAP analysis fix...")
        
        try:
            # Load data
            df = self.load_clean_data()
            X, y = self.prepare_data(df)
            
            # Try to load existing models, or train new ones
            models = self.load_trained_models()
            if not models:
                logger.info("No existing models found, training new ones...")
                models = self.train_models_if_needed(X, y)
            
            # Perform SHAP analysis
            shap_results = {}
            if SHAP_AVAILABLE:
                shap_results = self.analyze_with_shap(models, X, y)
            
            # Perform alternative interpretability analysis
            alternative_results = self.alternative_interpretability_analysis(models, X, y)
            
            # Compile final results
            analysis_results = {
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'shap_analysis_fix',
                'shap_available': SHAP_AVAILABLE,
                'dataset_info': {
                    'shape': list(df.shape),
                    'features_used': list(X.columns),
                    'target_variable': 'final_test'
                },
                'models_analyzed': list(models.keys()),
                'shap_results': self.serialize_shap_results(shap_results),
                'alternative_interpretability': alternative_results,
                'summary': self.generate_interpretability_summary(shap_results, alternative_results),
                'recommendations': self.generate_interpretability_recommendations(shap_results, alternative_results)
            }
            
            # Save results
            results_file = self.output_path / "shap_analysis_fix_results.json"
            with open(results_file, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            
            logger.info(f"SHAP analysis fix completed. Results saved to {results_file}")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"SHAP analysis fix failed: {e}")
            raise
    
    def serialize_shap_results(self, shap_results: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize SHAP results for JSON storage."""
        serialized = {}
        
        for model_name, result in shap_results.items():
            if 'error' in result:
                serialized[model_name] = result
            else:
                try:
                    # Extract key information from SHAP results
                    shap_values = result['shap_values']
                    feature_names = result['feature_names']
                    
                    # Calculate feature importance from SHAP values
                    if isinstance(shap_values, np.ndarray):
                        feature_importance = np.abs(shap_values).mean(axis=0).tolist()
                    else:
                        feature_importance = [0] * len(feature_names)
                    
                    serialized[model_name] = {
                        'feature_names': feature_names,
                        'feature_importance': feature_importance,
                        'expected_value': float(result['expected_value']),
                        'sample_size': len(result['X_sample']),
                        'status': 'success'
                    }
                    
                except Exception as e:
                    serialized[model_name] = {'error': f"Serialization failed: {e}"}
        
        return serialized
    
    def generate_interpretability_summary(self, shap_results: Dict[str, Any], alternative_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of interpretability analysis."""
        summary = {
            'shap_analysis_status': {},
            'top_features_by_model': {},
            'interpretability_methods_used': []
        }
        
        # SHAP analysis status
        for model_name, result in shap_results.items():
            if 'error' in result:
                summary['shap_analysis_status'][model_name] = 'failed'
            else:
                summary['shap_analysis_status'][model_name] = 'success'
        
        # Top features from alternative methods
        for model_name, result in alternative_results.items():
            if 'error' not in result:
                top_features = []
                
                # From permutation importance
                if 'permutation_importance' in result:
                    perm_data = result['permutation_importance']
                    top_indices = np.argsort(perm_data['importances_mean'])[-5:]
                    top_features.extend([perm_data['features'][i] for i in top_indices])
                
                summary['top_features_by_model'][model_name] = list(set(top_features))
        
        # Methods used
        if SHAP_AVAILABLE and shap_results:
            summary['interpretability_methods_used'].append('SHAP')
        summary['interpretability_methods_used'].extend(['permutation_importance', 'built_in_importance', 'coefficients'])
        
        return summary
    
    def generate_interpretability_recommendations(self, shap_results: Dict[str, Any], alternative_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on interpretability analysis."""
        recommendations = []
        
        # SHAP-specific recommendations
        if SHAP_AVAILABLE and shap_results:
            successful_shap = [name for name, result in shap_results.items() if 'error' not in result]
            if successful_shap:
                recommendations.append(f"SHAP analysis successful for: {', '.join(successful_shap)}")
                recommendations.append("Use SHAP plots for detailed feature interaction analysis")
            else:
                recommendations.append("SHAP analysis failed for all models - rely on alternative methods")
        else:
            recommendations.append("Install SHAP library for more detailed interpretability analysis")
        
        # Alternative methods recommendations
        recommendations.append("Use permutation importance for model-agnostic feature ranking")
        recommendations.append("Compare feature importance across different methods for robustness")
        recommendations.append("Focus on consistently important features across all models")
        
        # Model-specific recommendations
        linear_models = ['LinearRegression', 'Ridge', 'Lasso']
        tree_models = ['RandomForest', 'GradientBoosting']
        
        if any(model in alternative_results for model in linear_models):
            recommendations.append("Use coefficient analysis for linear models to understand feature direction")
        
        if any(model in alternative_results for model in tree_models):
            recommendations.append("Use built-in feature importance for tree-based models")
        
        return recommendations

def main():
    """Main execution function."""
    try:
        fixer = SHAPAnalysisFixer()
        results = fixer.run_shap_analysis_fix()
        
        print("\n=== SHAP ANALYSIS FIX COMPLETED ===")
        print(f"Timestamp: {results['timestamp']}")
        print(f"SHAP Available: {results['shap_available']}")
        print(f"Models Analyzed: {len(results['models_analyzed'])}")
        
        print("\n=== SHAP ANALYSIS STATUS ===")
        for model, status in results['summary']['shap_analysis_status'].items():
            print(f"- {model}: {status}")
        
        print("\n=== TOP FEATURES BY MODEL ===")
        for model, features in results['summary']['top_features_by_model'].items():
            print(f"- {model}: {', '.join(features[:3])}...")
        
        print("\n=== INTERPRETABILITY METHODS USED ===")
        for method in results['summary']['interpretability_methods_used']:
            print(f"- {method}")
        
        print("\n=== RECOMMENDATIONS ===")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"{i}. {rec}")
        
        print("\n=== SHAP ANALYSIS FIX COMPLETE ===")
        
    except Exception as e:
        logger.error(f"SHAP analysis fix failed: {e}")
        raise

if __name__ == "__main__":
    main()