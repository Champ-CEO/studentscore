#!/usr/bin/env python3
"""
Phase 6 Task 6.2.2: Alternative Interpretability Methods Implementation

This module implements comprehensive alternative interpretability methods
for machine learning models when SHAP is not available or fails.

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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AlternativeInterpretabilityAnalyzer:
    """
    Comprehensive alternative interpretability methods for machine learning models.
    Includes permutation importance, partial dependence plots, feature interactions,
    and model-specific interpretability techniques.
    """
    
    def __init__(self, project_root: str = None):
        if project_root is None:
            current_file = Path(__file__).resolve()
            self.project_root = current_file.parents[2]
        else:
            self.project_root = Path(project_root)
        
        self.data_path = self.project_root / "data" / "modeling_outputs"
        self.output_path = self.project_root / "data" / "modeling_outputs"
        self.plots_path = self.output_path / "alternative_interpretability_plots"
        
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
        logger.info("Preparing data for interpretability analysis...")
        
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
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train models for interpretability analysis."""
        logger.info("Training models for interpretability analysis...")
        
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
                
                # Calculate performance metrics
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                trained_models[model_name] = {
                    'model': model,
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test,
                    'train_r2': r2_score(y_train, train_pred),
                    'test_r2': r2_score(y_test, test_pred),
                    'train_mae': mean_absolute_error(y_train, train_pred),
                    'test_mae': mean_absolute_error(y_test, test_pred)
                }
                
                logger.info(f"{model_name} - Test R²: {trained_models[model_name]['test_r2']:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
        
        return trained_models
    
    def analyze_permutation_importance(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze permutation importance for all models."""
        logger.info("Analyzing permutation importance...")
        
        perm_results = {}
        
        for model_name, model_data in models.items():
            try:
                logger.info(f"Permutation importance for {model_name}...")
                
                model = model_data['model']
                X_test = model_data['X_test']
                y_test = model_data['y_test']
                
                # Use subset for faster computation
                subset_size = min(500, len(X_test))
                X_subset = X_test.iloc[:subset_size]
                y_subset = y_test.iloc[:subset_size]
                
                # Calculate permutation importance
                perm_importance = permutation_importance(
                    model, X_subset, y_subset, 
                    n_repeats=10, random_state=42, n_jobs=1
                )
                
                # Store results
                perm_results[model_name] = {
                    'importances_mean': perm_importance.importances_mean.tolist(),
                    'importances_std': perm_importance.importances_std.tolist(),
                    'feature_names': list(X_test.columns)
                }
                
                # Generate plot
                self.plot_permutation_importance(model_name, perm_results[model_name])
                
            except Exception as e:
                logger.error(f"Permutation importance failed for {model_name}: {e}")
                perm_results[model_name] = {'error': str(e)}
        
        return perm_results
    
    def analyze_partial_dependence(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze partial dependence for top features."""
        logger.info("Analyzing partial dependence...")
        
        pd_results = {}
        
        for model_name, model_data in models.items():
            try:
                logger.info(f"Partial dependence for {model_name}...")
                
                model = model_data['model']
                X_train = model_data['X_train']
                
                # Get top 5 most important features (from permutation importance if available)
                feature_names = list(X_train.columns)
                
                # Use subset for faster computation
                subset_size = min(1000, len(X_train))
                X_subset = X_train.iloc[:subset_size]
                
                # Select top features for partial dependence
                if hasattr(model, 'feature_importances_'):
                    # For tree-based models, use built-in importance
                    top_indices = np.argsort(model.feature_importances_)[-5:]
                    top_features = [feature_names[i] for i in top_indices]
                else:
                    # For linear models, use top 5 features by absolute coefficient
                    if hasattr(model, 'coef_'):
                        top_indices = np.argsort(np.abs(model.coef_))[-5:]
                        top_features = [feature_names[i] for i in top_indices]
                    else:
                        # Default to first 5 features
                        top_features = feature_names[:5]
                
                pd_data = {}
                
                for feature in top_features:
                    try:
                        feature_idx = feature_names.index(feature)
                        
                        # Calculate partial dependence
                        pd_result = partial_dependence(
                            model, X_subset, [feature_idx], 
                            grid_resolution=20
                        )
                        
                        pd_data[feature] = {
                            'values': pd_result['values'][0].tolist(),
                            'grid': pd_result['grid_values'][0].tolist()
                        }
                        
                    except Exception as e:
                        logger.warning(f"Partial dependence failed for {feature}: {e}")
                
                pd_results[model_name] = pd_data
                
                # Generate partial dependence plots
                if pd_data:
                    self.plot_partial_dependence(model_name, pd_data)
                
            except Exception as e:
                logger.error(f"Partial dependence failed for {model_name}: {e}")
                pd_results[model_name] = {'error': str(e)}
        
        return pd_results
    
    def analyze_feature_interactions(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze feature interactions using correlation and model predictions."""
        logger.info("Analyzing feature interactions...")
        
        interaction_results = {}
        
        for model_name, model_data in models.items():
            try:
                logger.info(f"Feature interactions for {model_name}...")
                
                X_train = model_data['X_train']
                
                # Calculate feature correlations
                correlation_matrix = X_train.corr()
                
                # Find top correlated feature pairs
                corr_pairs = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        corr_value = correlation_matrix.iloc[i, j]
                        if abs(corr_value) > 0.3:  # Only significant correlations
                            corr_pairs.append({
                                'feature1': correlation_matrix.columns[i],
                                'feature2': correlation_matrix.columns[j],
                                'correlation': corr_value
                            })
                
                # Sort by absolute correlation
                corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
                
                interaction_results[model_name] = {
                    'correlation_matrix': correlation_matrix.to_dict(),
                    'top_correlations': corr_pairs[:20]  # Top 20 correlations
                }
                
                # Generate correlation heatmap
                self.plot_correlation_heatmap(model_name, correlation_matrix)
                
            except Exception as e:
                logger.error(f"Feature interactions failed for {model_name}: {e}")
                interaction_results[model_name] = {'error': str(e)}
        
        return interaction_results
    
    def analyze_model_specific_interpretability(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze model-specific interpretability features."""
        logger.info("Analyzing model-specific interpretability...")
        
        model_specific_results = {}
        
        for model_name, model_data in models.items():
            try:
                logger.info(f"Model-specific analysis for {model_name}...")
                
                model = model_data['model']
                X_train = model_data['X_train']
                feature_names = list(X_train.columns)
                
                result = {'model_type': model_name}
                
                # Linear models: coefficients analysis
                if model_name in ['LinearRegression', 'Ridge', 'Lasso']:
                    if hasattr(model, 'coef_'):
                        coefficients = model.coef_.tolist() if hasattr(model.coef_, 'tolist') else [float(model.coef_)]
                        
                        result['coefficients'] = {
                            'values': coefficients,
                            'abs_values': [abs(x) for x in coefficients],
                            'feature_names': feature_names,
                            'intercept': float(model.intercept_) if hasattr(model, 'intercept_') else 0.0
                        }
                        
                        # Coefficient statistics
                        result['coefficient_stats'] = {
                            'mean_abs_coef': np.mean([abs(x) for x in coefficients]),
                            'max_abs_coef': max([abs(x) for x in coefficients]),
                            'min_abs_coef': min([abs(x) for x in coefficients]),
                            'num_positive': sum(1 for x in coefficients if x > 0),
                            'num_negative': sum(1 for x in coefficients if x < 0)
                        }
                        
                        # Generate coefficient plot
                        self.plot_model_coefficients(model_name, result['coefficients'])
                
                # Tree-based models: feature importance and tree analysis
                elif model_name in ['RandomForest', 'GradientBoosting']:
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_.tolist()
                        
                        result['feature_importances'] = {
                            'values': importances,
                            'feature_names': feature_names
                        }
                        
                        # Feature importance statistics
                        result['importance_stats'] = {
                            'mean_importance': np.mean(importances),
                            'max_importance': max(importances),
                            'min_importance': min(importances),
                            'importance_concentration': max(importances) / sum(importances),
                            'num_important_features': sum(1 for x in importances if x > 0.01)
                        }
                        
                        # Generate feature importance plot
                        self.plot_feature_importance(model_name, result['feature_importances'])
                    
                    # Additional tree-specific analysis
                    if model_name == 'RandomForest':
                        result['forest_stats'] = {
                            'n_estimators': model.n_estimators,
                            'max_depth': model.max_depth,
                            'min_samples_split': model.min_samples_split,
                            'min_samples_leaf': model.min_samples_leaf
                        }
                    
                    elif model_name == 'GradientBoosting':
                        result['boosting_stats'] = {
                            'n_estimators': model.n_estimators,
                            'learning_rate': model.learning_rate,
                            'max_depth': model.max_depth,
                            'subsample': model.subsample
                        }
                
                model_specific_results[model_name] = result
                
            except Exception as e:
                logger.error(f"Model-specific analysis failed for {model_name}: {e}")
                model_specific_results[model_name] = {'error': str(e)}
        
        return model_specific_results
    
    def plot_permutation_importance(self, model_name: str, perm_data: Dict[str, Any]):
        """Plot permutation importance."""
        try:
            features = perm_data['feature_names']
            importances_mean = perm_data['importances_mean']
            importances_std = perm_data['importances_std']
            
            # Get top 15 features
            top_indices = np.argsort(importances_mean)[-15:]
            top_features = [features[i] for i in top_indices]
            top_means = [importances_mean[i] for i in top_indices]
            top_stds = [importances_std[i] for i in top_indices]
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(top_features)), top_means, xerr=top_stds, capsize=3, alpha=0.7)
            plt.yticks(range(len(top_features)), top_features)
            plt.xlabel('Permutation Importance')
            plt.title(f'Permutation Importance - {model_name}')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            plot_file = self.plots_path / f"{model_name.lower()}_permutation_importance.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Permutation importance plot saved: {plot_file}")
            
        except Exception as e:
            logger.error(f"Failed to plot permutation importance for {model_name}: {e}")
    
    def plot_partial_dependence(self, model_name: str, pd_data: Dict[str, Any]):
        """Plot partial dependence for top features."""
        try:
            n_features = len(pd_data)
            if n_features == 0:
                return
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, (feature, data) in enumerate(pd_data.items()):
                if i >= 6:  # Limit to 6 plots
                    break
                
                ax = axes[i]
                ax.plot(data['grid'], data['values'], linewidth=2)
                ax.set_xlabel(feature)
                ax.set_ylabel('Partial Dependence')
                ax.set_title(f'Partial Dependence: {feature}')
                ax.grid(alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(pd_data), 6):
                axes[i].set_visible(False)
            
            plt.suptitle(f'Partial Dependence Plots - {model_name}', fontsize=16)
            plt.tight_layout()
            
            plot_file = self.plots_path / f"{model_name.lower()}_partial_dependence.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Partial dependence plot saved: {plot_file}")
            
        except Exception as e:
            logger.error(f"Failed to plot partial dependence for {model_name}: {e}")
    
    def plot_correlation_heatmap(self, model_name: str, correlation_matrix: pd.DataFrame):
        """Plot correlation heatmap."""
        try:
            # Select top correlated features for visualization
            corr_abs = correlation_matrix.abs()
            top_features_idx = corr_abs.sum().nlargest(20).index
            corr_subset = correlation_matrix.loc[top_features_idx, top_features_idx]
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_subset, annot=True, cmap='coolwarm', center=0, 
                       square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
            plt.title(f'Feature Correlation Heatmap - {model_name}')
            plt.tight_layout()
            
            plot_file = self.plots_path / f"{model_name.lower()}_correlation_heatmap.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Correlation heatmap saved: {plot_file}")
            
        except Exception as e:
            logger.error(f"Failed to plot correlation heatmap for {model_name}: {e}")
    
    def plot_model_coefficients(self, model_name: str, coef_data: Dict[str, Any]):
        """Plot linear model coefficients."""
        try:
            features = coef_data['feature_names']
            values = coef_data['values']
            abs_values = coef_data['abs_values']
            
            # Get top 15 features by absolute coefficient value
            top_indices = np.argsort(abs_values)[-15:]
            top_features = [features[i] for i in top_indices]
            top_values = [values[i] for i in top_indices]
            
            plt.figure(figsize=(12, 8))
            colors = ['red' if x < 0 else 'blue' for x in top_values]
            plt.barh(range(len(top_features)), top_values, color=colors, alpha=0.7)
            plt.yticks(range(len(top_features)), top_features)
            plt.xlabel('Coefficient Value')
            plt.title(f'Linear Model Coefficients - {model_name}')
            plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            plot_file = self.plots_path / f"{model_name.lower()}_coefficients.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Coefficients plot saved: {plot_file}")
            
        except Exception as e:
            logger.error(f"Failed to plot coefficients for {model_name}: {e}")
    
    def plot_feature_importance(self, model_name: str, importance_data: Dict[str, Any]):
        """Plot built-in feature importance."""
        try:
            features = importance_data['feature_names']
            values = importance_data['values']
            
            # Get top 15 features
            top_indices = np.argsort(values)[-15:]
            top_features = [features[i] for i in top_indices]
            top_values = [values[i] for i in top_indices]
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(top_features)), top_values, alpha=0.7)
            plt.yticks(range(len(top_features)), top_features)
            plt.xlabel('Feature Importance')
            plt.title(f'Built-in Feature Importance - {model_name}')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            plot_file = self.plots_path / f"{model_name.lower()}_builtin_importance.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Feature importance plot saved: {plot_file}")
            
        except Exception as e:
            logger.error(f"Failed to plot feature importance for {model_name}: {e}")
    
    def run_comprehensive_interpretability_analysis(self) -> Dict[str, Any]:
        """Run comprehensive alternative interpretability analysis."""
        logger.info("Starting comprehensive interpretability analysis...")
        
        try:
            # Load data
            df = self.load_clean_data()
            X, y = self.prepare_data(df)
            
            # Train models
            models = self.train_models(X, y)
            
            # Perform different interpretability analyses
            logger.info("Performing permutation importance analysis...")
            perm_importance = self.analyze_permutation_importance(models)
            
            logger.info("Performing partial dependence analysis...")
            partial_dependence = self.analyze_partial_dependence(models)
            
            logger.info("Performing feature interaction analysis...")
            feature_interactions = self.analyze_feature_interactions(models)
            
            logger.info("Performing model-specific interpretability analysis...")
            model_specific = self.analyze_model_specific_interpretability(models)
            
            # Compile results
            analysis_results = {
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'alternative_interpretability_comprehensive',
                'dataset_info': {
                    'shape': list(df.shape),
                    'features_used': list(X.columns),
                    'target_variable': 'final_test'
                },
                'model_performance': {
                    model_name: {
                        'test_r2': model_data['test_r2'],
                        'test_mae': model_data['test_mae'],
                        'train_r2': model_data['train_r2'],
                        'train_mae': model_data['train_mae']
                    }
                    for model_name, model_data in models.items()
                },
                'permutation_importance': perm_importance,
                'partial_dependence': partial_dependence,
                'feature_interactions': feature_interactions,
                'model_specific_interpretability': model_specific,
                'summary': self.generate_comprehensive_summary(perm_importance, partial_dependence, feature_interactions, model_specific),
                'recommendations': self.generate_comprehensive_recommendations(perm_importance, partial_dependence, feature_interactions, model_specific)
            }
            
            # Save results
            results_file = self.output_path / "alternative_interpretability_comprehensive.json"
            with open(results_file, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            
            logger.info(f"Comprehensive interpretability analysis completed. Results saved to {results_file}")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Comprehensive interpretability analysis failed: {e}")
            raise
    
    def generate_comprehensive_summary(self, perm_importance: Dict, partial_dependence: Dict, 
                                     feature_interactions: Dict, model_specific: Dict) -> Dict[str, Any]:
        """Generate comprehensive summary of interpretability analysis."""
        summary = {
            'top_features_across_models': {},
            'model_interpretability_status': {},
            'key_insights': []
        }
        
        # Aggregate top features across models
        all_features = {}
        for model_name, perm_data in perm_importance.items():
            if 'error' not in perm_data:
                features = perm_data['feature_names']
                importances = perm_data['importances_mean']
                
                # Get top 5 features for this model
                top_indices = np.argsort(importances)[-5:]
                top_features = [features[i] for i in top_indices]
                
                for feature in top_features:
                    if feature not in all_features:
                        all_features[feature] = 0
                    all_features[feature] += 1
        
        # Sort features by frequency across models
        summary['top_features_across_models'] = dict(
            sorted(all_features.items(), key=lambda x: x[1], reverse=True)
        )
        
        # Model interpretability status
        for model_name in perm_importance.keys():
            status = {
                'permutation_importance': 'success' if 'error' not in perm_importance.get(model_name, {}) else 'failed',
                'partial_dependence': 'success' if 'error' not in partial_dependence.get(model_name, {}) else 'failed',
                'feature_interactions': 'success' if 'error' not in feature_interactions.get(model_name, {}) else 'failed',
                'model_specific': 'success' if 'error' not in model_specific.get(model_name, {}) else 'failed'
            }
            summary['model_interpretability_status'][model_name] = status
        
        # Key insights
        most_important_feature = max(all_features.items(), key=lambda x: x[1])[0] if all_features else "Unknown"
        summary['key_insights'].append(f"Most consistently important feature: {most_important_feature}")
        summary['key_insights'].append(f"Number of models analyzed: {len(perm_importance)}")
        summary['key_insights'].append(f"Total unique features: {len(all_features)}")
        
        return summary
    
    def generate_comprehensive_recommendations(self, perm_importance: Dict, partial_dependence: Dict,
                                             feature_interactions: Dict, model_specific: Dict) -> List[str]:
        """Generate comprehensive recommendations."""
        recommendations = []
        
        # General recommendations
        recommendations.append("Use permutation importance as the primary feature ranking method")
        recommendations.append("Compare feature importance across multiple models for robustness")
        recommendations.append("Focus on features that are consistently important across models")
        
        # Model-specific recommendations
        linear_models = ['LinearRegression', 'Ridge', 'Lasso']
        tree_models = ['RandomForest', 'GradientBoosting']
        
        if any(model in model_specific for model in linear_models):
            recommendations.append("For linear models: analyze coefficient signs and magnitudes")
            recommendations.append("Consider feature scaling for better coefficient interpretation")
        
        if any(model in model_specific for model in tree_models):
            recommendations.append("For tree models: use built-in feature importance alongside permutation importance")
            recommendations.append("Consider feature interaction effects in tree-based models")
        
        # Partial dependence recommendations
        if any('error' not in pd_data for pd_data in partial_dependence.values()):
            recommendations.append("Use partial dependence plots to understand feature-target relationships")
            recommendations.append("Look for non-linear relationships in partial dependence plots")
        
        # Feature interaction recommendations
        if any('error' not in fi_data for fi_data in feature_interactions.values()):
            recommendations.append("Investigate highly correlated features for potential redundancy")
            recommendations.append("Consider feature engineering based on correlation patterns")
        
        return recommendations

def main():
    """Main execution function."""
    try:
        analyzer = AlternativeInterpretabilityAnalyzer()
        results = analyzer.run_comprehensive_interpretability_analysis()
        
        print("\n=== ALTERNATIVE INTERPRETABILITY ANALYSIS COMPLETED ===")
        print(f"Timestamp: {results['timestamp']}")
        print(f"Models Analyzed: {len(results['model_performance'])}")
        
        print("\n=== MODEL PERFORMANCE ===")
        for model, perf in results['model_performance'].items():
            print(f"- {model}: R²={perf['test_r2']:.3f}, MAE={perf['test_mae']:.3f}")
        
        print("\n=== TOP FEATURES ACROSS MODELS ===")
        for feature, count in list(results['summary']['top_features_across_models'].items())[:10]:
            print(f"- {feature}: appears in {count} models")
        
        print("\n=== KEY INSIGHTS ===")
        for insight in results['summary']['key_insights']:
            print(f"- {insight}")
        
        print("\n=== RECOMMENDATIONS ===")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"{i}. {rec}")
        
        print("\n=== ALTERNATIVE INTERPRETABILITY ANALYSIS COMPLETE ===")
        
    except Exception as e:
        logger.error(f"Alternative interpretability analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()