#!/usr/bin/env python3
"""
Advanced Preprocessing Module

Implements Phase 3.5: Advanced Preprocessing

This module provides advanced preprocessing capabilities including:
- Advanced feature selection techniques
- Sophisticated validation frameworks
- Automated preprocessing optimization
- Custom transformation pipelines
- Advanced missing data handling
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import logging
from datetime import datetime
import warnings

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer, QuantileTransformer, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, RFE, RFECV, 
    SelectFromModel, VarianceThreshold, f_regression, 
    mutual_info_regression, chi2
)
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array
from scipy import stats
from scipy.stats import jarque_bera, shapiro, anderson
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedFeatureSelector:
    """
    Advanced feature selection with multiple methods and validation.
    
    Provides comprehensive feature selection including:
    - Statistical methods (univariate, correlation-based)
    - Model-based methods (tree-based, linear model-based)
    - Wrapper methods (RFE, forward/backward selection)
    - Dimensionality reduction methods
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.selection_results = {}
        self.selected_features = {}
        self.feature_scores = {}
        
    def univariate_selection(self, X: pd.DataFrame, y: pd.Series, 
                           method: str = 'f_regression', k: int = 10) -> List[str]:
        """
        Perform univariate feature selection.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            method: Selection method ('f_regression', 'mutual_info_regression')
            k: Number of features to select
            
        Returns:
            List of selected feature names
        """
        if method == 'f_regression':
            score_func = f_regression
        elif method == 'mutual_info_regression':
            score_func = mutual_info_regression
        else:
            raise ValueError(f"Unknown method: {method}")
        
        selector = SelectKBest(score_func=score_func, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        # Store scores
        scores = selector.scores_
        feature_scores = dict(zip(X.columns, scores))
        
        self.selected_features[f'univariate_{method}'] = selected_features
        self.feature_scores[f'univariate_{method}'] = feature_scores
        
        logger.info(f"Univariate selection ({method}): selected {len(selected_features)} features")
        return selected_features
    
    def correlation_based_selection(self, X: pd.DataFrame, y: pd.Series, 
                                  threshold: float = 0.95) -> List[str]:
        """
        Remove highly correlated features.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            threshold: Correlation threshold for removal
            
        Returns:
            List of selected feature names
        """
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find highly correlated pairs
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = [column for column in upper_triangle.columns 
                  if any(upper_triangle[column] > threshold)]
        
        # Keep features not in drop list
        selected_features = [col for col in X.columns if col not in to_drop]
        
        self.selected_features['correlation_based'] = selected_features
        
        logger.info(f"Correlation-based selection: removed {len(to_drop)} features, kept {len(selected_features)}")
        return selected_features
    
    def variance_threshold_selection(self, X: pd.DataFrame, threshold: float = 0.01) -> List[str]:
        """
        Remove low-variance features.
        
        Args:
            X: Feature DataFrame
            threshold: Variance threshold
            
        Returns:
            List of selected feature names
        """
        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(X)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        self.selected_features['variance_threshold'] = selected_features
        
        logger.info(f"Variance threshold selection: selected {len(selected_features)} features")
        return selected_features
    
    def model_based_selection(self, X: pd.DataFrame, y: pd.Series, 
                            model_type: str = 'random_forest', n_features: int = 10) -> List[str]:
        """
        Model-based feature selection.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            model_type: Type of model ('random_forest', 'lasso', 'ridge')
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        if model_type == 'random_forest':
            estimator = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        elif model_type == 'lasso':
            estimator = LassoCV(cv=5, random_state=self.random_state)
        elif model_type == 'ridge':
            estimator = RidgeCV(cv=5)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Fit model and select features
        estimator.fit(X, y)
        
        if hasattr(estimator, 'feature_importances_'):
            # Tree-based models
            importances = estimator.feature_importances_
            feature_importance = dict(zip(X.columns, importances))
            
            # Select top features
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            selected_features = [feat for feat, _ in sorted_features[:n_features]]
            
        elif hasattr(estimator, 'coef_'):
            # Linear models
            coefficients = np.abs(estimator.coef_)
            feature_importance = dict(zip(X.columns, coefficients))
            
            # Select top features
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            selected_features = [feat for feat, _ in sorted_features[:n_features]]
        else:
            raise ValueError(f"Model {model_type} does not provide feature importance")
        
        self.selected_features[f'model_based_{model_type}'] = selected_features
        self.feature_scores[f'model_based_{model_type}'] = feature_importance
        
        logger.info(f"Model-based selection ({model_type}): selected {len(selected_features)} features")
        return selected_features
    
    def recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series, 
                                    estimator_type: str = 'random_forest', 
                                    n_features: int = 10, cv: int = 5) -> List[str]:
        """
        Recursive Feature Elimination with Cross-Validation.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            estimator_type: Type of estimator
            n_features: Number of features to select
            cv: Number of CV folds
            
        Returns:
            List of selected feature names
        """
        if estimator_type == 'random_forest':
            estimator = RandomForestRegressor(n_estimators=50, random_state=self.random_state)
        elif estimator_type == 'lasso':
            estimator = LassoCV(cv=3, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown estimator type: {estimator_type}")
        
        # Perform RFECV
        selector = RFECV(estimator, min_features_to_select=n_features, cv=cv, 
                        scoring='neg_mean_squared_error', n_jobs=-1)
        selector.fit(X, y)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        self.selected_features[f'rfe_{estimator_type}'] = selected_features
        self.selection_results[f'rfe_{estimator_type}'] = {
            'n_features': selector.n_features_,
            'cv_scores': selector.cv_results_
        }
        
        logger.info(f"RFE selection ({estimator_type}): selected {len(selected_features)} features")
        return selected_features
    
    def ensemble_feature_selection(self, X: pd.DataFrame, y: pd.Series, 
                                 methods: List[str] = None, voting: str = 'majority') -> List[str]:
        """
        Ensemble feature selection combining multiple methods.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            methods: List of methods to combine
            voting: Voting strategy ('majority', 'union', 'intersection')
            
        Returns:
            List of selected feature names
        """
        if methods is None:
            methods = ['univariate_f_regression', 'model_based_random_forest', 'correlation_based']
        
        # Run individual methods if not already done
        method_features = {}
        
        for method in methods:
            if method == 'univariate_f_regression':
                features = self.univariate_selection(X, y, 'f_regression', k=min(20, X.shape[1]//2))
                method_features[method] = set(features)
            elif method == 'model_based_random_forest':
                features = self.model_based_selection(X, y, 'random_forest', n_features=min(20, X.shape[1]//2))
                method_features[method] = set(features)
            elif method == 'correlation_based':
                features = self.correlation_based_selection(X, y)
                method_features[method] = set(features)
            elif method in self.selected_features:
                method_features[method] = set(self.selected_features[method])
        
        # Combine results based on voting strategy
        all_features = set(X.columns)
        
        if voting == 'majority':
            # Features selected by majority of methods
            feature_votes = {feat: 0 for feat in all_features}
            for features in method_features.values():
                for feat in features:
                    feature_votes[feat] += 1
            
            threshold = len(methods) // 2 + 1
            selected_features = [feat for feat, votes in feature_votes.items() if votes >= threshold]
            
        elif voting == 'union':
            # Features selected by any method
            selected_features = list(set().union(*method_features.values()))
            
        elif voting == 'intersection':
            # Features selected by all methods
            selected_features = list(set.intersection(*method_features.values()))
            
        else:
            raise ValueError(f"Unknown voting strategy: {voting}")
        
        self.selected_features[f'ensemble_{voting}'] = selected_features
        
        logger.info(f"Ensemble selection ({voting}): selected {len(selected_features)} features")
        return selected_features


class AdvancedTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for advanced preprocessing operations.
    """
    
    def __init__(self, transformation_type: str = 'power', **kwargs):
        self.transformation_type = transformation_type
        self.kwargs = kwargs
        self.transformer = None
        
    def fit(self, X, y=None):
        if self.transformation_type == 'power':
            self.transformer = PowerTransformer(method='yeo-johnson', **self.kwargs)
        elif self.transformation_type == 'quantile':
            self.transformer = QuantileTransformer(output_distribution='normal', **self.kwargs)
        elif self.transformation_type == 'polynomial':
            degree = self.kwargs.get('degree', 2)
            self.transformer = PolynomialFeatures(degree=degree, include_bias=False)
        else:
            raise ValueError(f"Unknown transformation type: {self.transformation_type}")
        
        self.transformer.fit(X, y)
        return self
    
    def transform(self, X):
        return self.transformer.transform(X)


class AdvancedValidationFramework:
    """
    Advanced validation framework for preprocessing pipelines.
    
    Provides comprehensive validation including:
    - Cross-validation with multiple metrics
    - Statistical tests for preprocessing effectiveness
    - Automated hyperparameter tuning for preprocessing
    - Performance comparison across different preprocessing strategies
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.validation_results = {}
        self.best_pipeline = None
        
    def validate_preprocessing_pipeline(self, X: pd.DataFrame, y: pd.Series, 
                                      pipeline: Pipeline, cv: int = 5, 
                                      scoring: List[str] = None) -> Dict[str, Any]:
        """
        Validate preprocessing pipeline using cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            pipeline: Preprocessing pipeline
            cv: Number of CV folds
            scoring: List of scoring metrics
            
        Returns:
            Dictionary with validation results
        """
        if scoring is None:
            scoring = ['neg_mean_squared_error', 'r2', 'neg_mean_absolute_error']
        
        results = {}
        
        for metric in scoring:
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=metric, n_jobs=-1)
            results[metric] = {
                'scores': scores.tolist(),
                'mean': scores.mean(),
                'std': scores.std(),
                'min': scores.min(),
                'max': scores.max()
            }
        
        logger.info(f"Pipeline validation completed with {cv}-fold CV")
        return results
    
    def compare_preprocessing_strategies(self, X: pd.DataFrame, y: pd.Series, 
                                       strategies: Dict[str, Pipeline], 
                                       cv: int = 5) -> Dict[str, Any]:
        """
        Compare multiple preprocessing strategies.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            strategies: Dictionary of strategy_name -> pipeline
            cv: Number of CV folds
            
        Returns:
            Dictionary with comparison results
        """
        comparison_results = {}
        
        for strategy_name, pipeline in strategies.items():
            logger.info(f"Validating strategy: {strategy_name}")
            results = self.validate_preprocessing_pipeline(X, y, pipeline, cv)
            comparison_results[strategy_name] = results
        
        # Find best strategy based on R2 score
        best_strategy = max(comparison_results.keys(), 
                          key=lambda x: comparison_results[x]['r2']['mean'])
        
        self.validation_results = comparison_results
        self.best_pipeline = strategies[best_strategy]
        
        logger.info(f"Best preprocessing strategy: {best_strategy}")
        return comparison_results
    
    def statistical_validation(self, X_original: pd.DataFrame, X_transformed: np.ndarray, 
                             feature_names: List[str]) -> Dict[str, Any]:
        """
        Perform statistical validation of transformations.
        
        Args:
            X_original: Original feature DataFrame
            X_transformed: Transformed feature array
            feature_names: Names of transformed features
            
        Returns:
            Dictionary with statistical test results
        """
        results = {
            'normality_tests': {},
            'distribution_changes': {},
            'correlation_preservation': {}
        }
        
        # Convert transformed array to DataFrame for easier handling
        X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
        
        # Normality tests for numerical features
        for i, feature in enumerate(feature_names):
            if i < len(X_original.columns):
                original_col = X_original.iloc[:, i]
                transformed_col = X_transformed_df.iloc[:, i]
                
                # Skip non-numeric columns
                if not pd.api.types.is_numeric_dtype(original_col):
                    continue
                
                # Shapiro-Wilk test (for small samples)
                if len(transformed_col) <= 5000:
                    try:
                        shapiro_stat, shapiro_p = shapiro(transformed_col.dropna())
                        results['normality_tests'][feature] = {
                            'shapiro_stat': shapiro_stat,
                            'shapiro_p': shapiro_p,
                            'is_normal': shapiro_p > 0.05
                        }
                    except Exception as e:
                        logger.warning(f"Shapiro test failed for {feature}: {str(e)}")
                
                # Distribution comparison
                try:
                    ks_stat, ks_p = stats.ks_2samp(original_col.dropna(), transformed_col.dropna())
                    results['distribution_changes'][feature] = {
                        'ks_statistic': ks_stat,
                        'ks_p_value': ks_p,
                        'distributions_different': ks_p < 0.05
                    }
                except Exception as e:
                    logger.warning(f"KS test failed for {feature}: {str(e)}")
        
        # Correlation preservation (for numerical features only)
        try:
            numeric_original = X_original.select_dtypes(include=[np.number])
            numeric_transformed = X_transformed_df.iloc[:, :len(numeric_original.columns)]
            
            if len(numeric_original.columns) > 1:
                orig_corr = numeric_original.corr().values
                trans_corr = numeric_transformed.corr().values
                
                # Calculate correlation between correlation matrices
                corr_preservation = np.corrcoef(orig_corr.flatten(), trans_corr.flatten())[0, 1]
                results['correlation_preservation'] = {
                    'correlation_coefficient': corr_preservation,
                    'well_preserved': corr_preservation > 0.8
                }
        except Exception as e:
            logger.warning(f"Correlation preservation test failed: {str(e)}")
        
        return results
    
    def automated_hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series, 
                                       base_pipeline: Pipeline, 
                                       param_grid: Dict[str, Any], 
                                       cv: int = 5, n_iter: int = 50) -> Pipeline:
        """
        Automated hyperparameter tuning for preprocessing pipeline.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            base_pipeline: Base pipeline to tune
            param_grid: Parameter grid for tuning
            cv: Number of CV folds
            n_iter: Number of iterations for random search
            
        Returns:
            Best pipeline after tuning
        """
        # Use RandomizedSearchCV for efficiency
        search = RandomizedSearchCV(
            base_pipeline, 
            param_grid, 
            n_iter=n_iter, 
            cv=cv, 
            scoring='r2', 
            n_jobs=-1, 
            random_state=self.random_state
        )
        
        search.fit(X, y)
        
        self.best_pipeline = search.best_estimator_
        
        tuning_results = {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_
        }
        
        self.validation_results['hyperparameter_tuning'] = tuning_results
        
        logger.info(f"Hyperparameter tuning completed. Best score: {search.best_score_:.4f}")
        return search.best_estimator_


class AdvancedPreprocessingOrchestrator:
    """
    Orchestrates advanced preprocessing operations.
    
    Combines feature selection, transformation, and validation
    into a comprehensive preprocessing workflow.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_selector = AdvancedFeatureSelector(random_state)
        self.validation_framework = AdvancedValidationFramework(random_state)
        self.final_pipeline = None
        self.preprocessing_report = {}
        
    def run_comprehensive_preprocessing(self, X: pd.DataFrame, y: pd.Series, 
                                      target_col: str = 'final_test',
                                      feature_selection_methods: List[str] = None,
                                      transformation_methods: List[str] = None,
                                      validation_cv: int = 5) -> Dict[str, Any]:
        """
        Run comprehensive advanced preprocessing.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            target_col: Name of target column
            feature_selection_methods: Methods for feature selection
            transformation_methods: Methods for transformation
            validation_cv: CV folds for validation
            
        Returns:
            Dictionary with comprehensive results
        """
        logger.info("Starting comprehensive advanced preprocessing")
        
        if feature_selection_methods is None:
            feature_selection_methods = ['ensemble_majority']
        
        if transformation_methods is None:
            transformation_methods = ['power', 'quantile']
        
        results = {
            'feature_selection': {},
            'transformations': {},
            'pipeline_comparison': {},
            'final_pipeline': None,
            'recommendations': []
        }
        
        # Feature Selection
        logger.info("Performing feature selection")
        for method in feature_selection_methods:
            if method == 'ensemble_majority':
                selected_features = self.feature_selector.ensemble_feature_selection(
                    X, y, voting='majority'
                )
            elif method == 'univariate':
                selected_features = self.feature_selector.univariate_selection(
                    X, y, k=min(15, X.shape[1]//2)
                )
            elif method == 'model_based':
                selected_features = self.feature_selector.model_based_selection(
                    X, y, n_features=min(15, X.shape[1]//2)
                )
            else:
                logger.warning(f"Unknown feature selection method: {method}")
                continue
            
            results['feature_selection'][method] = selected_features
        
        # Create preprocessing strategies
        strategies = {}
        
        # Base strategy (no advanced preprocessing)
        from sklearn.preprocessing import StandardScaler
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        
        # Identify numerical and categorical features
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        base_preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numerical_features),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
                ]), categorical_features)
            ]
        )
        
        strategies['base'] = Pipeline([('preprocessor', base_preprocessor)])
        
        # Advanced strategies with transformations
        for transform_method in transformation_methods:
            if transform_method in ['power', 'quantile']:
                # Create advanced preprocessor with transformation
                advanced_preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', Pipeline([
                            ('imputer', SimpleImputer(strategy='median')),
                            ('transformer', AdvancedTransformer(transformation_type=transform_method)),
                            ('scaler', StandardScaler())
                        ]), numerical_features),
                        ('cat', Pipeline([
                            ('imputer', SimpleImputer(strategy='most_frequent')),
                            ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
                        ]), categorical_features)
                    ]
                )
                
                strategies[f'advanced_{transform_method}'] = Pipeline([
                    ('preprocessor', advanced_preprocessor)
                ])
        
        # Compare strategies
        logger.info("Comparing preprocessing strategies")
        comparison_results = self.validation_framework.compare_preprocessing_strategies(
            X, y, strategies, cv=validation_cv
        )
        
        results['pipeline_comparison'] = comparison_results
        results['final_pipeline'] = self.validation_framework.best_pipeline
        self.final_pipeline = self.validation_framework.best_pipeline
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results)
        results['recommendations'] = recommendations
        
        self.preprocessing_report = results
        
        logger.info("Comprehensive advanced preprocessing completed")
        return results
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on preprocessing results.
        
        Args:
            results: Preprocessing results
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Feature selection recommendations
        if 'feature_selection' in results:
            for method, features in results['feature_selection'].items():
                if len(features) < 5:
                    recommendations.append(
                        f"Feature selection with {method} resulted in very few features ({len(features)}). "
                        "Consider using less aggressive selection or combining with other methods."
                    )
                elif len(features) > 50:
                    recommendations.append(
                        f"Feature selection with {method} kept many features ({len(features)}). "
                        "Consider more aggressive selection for better model performance."
                    )
        
        # Pipeline comparison recommendations
        if 'pipeline_comparison' in results:
            best_strategy = max(results['pipeline_comparison'].keys(), 
                              key=lambda x: results['pipeline_comparison'][x]['r2']['mean'])
            best_score = results['pipeline_comparison'][best_strategy]['r2']['mean']
            
            if best_score > 0.8:
                recommendations.append(
                    f"Excellent preprocessing performance achieved with {best_strategy} strategy (R² = {best_score:.3f})"
                )
            elif best_score > 0.6:
                recommendations.append(
                    f"Good preprocessing performance with {best_strategy} strategy (R² = {best_score:.3f}). "
                    "Consider additional feature engineering."
                )
            else:
                recommendations.append(
                    f"Moderate preprocessing performance with {best_strategy} strategy (R² = {best_score:.3f}). "
                    "Consider more advanced techniques or additional data collection."
                )
        
        return recommendations
    
    def save_preprocessing_artifacts(self, save_dir: str) -> None:
        """
        Save all preprocessing artifacts.
        
        Args:
            save_dir: Directory to save artifacts
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save final pipeline
        if self.final_pipeline is not None:
            joblib.dump(self.final_pipeline, save_path / 'advanced_preprocessing_pipeline.pkl')
        
        # Save feature selection results
        with open(save_path / 'feature_selection_results.json', 'w') as f:
            json.dump(self.feature_selector.selected_features, f, indent=2)
        
        # Save preprocessing report
        with open(save_path / 'advanced_preprocessing_report.json', 'w') as f:
            json.dump(self.preprocessing_report, f, indent=2, default=str)
        
        logger.info(f"Advanced preprocessing artifacts saved to {save_dir}")


def main():
    """
    Main function for testing advanced preprocessing.
    """
    # Example usage
    db_path = "data/raw/score.db"
    
    # Load data
    conn = sqlite3.connect(db_path)
    data = pd.read_sql_query("SELECT * FROM student_scores", conn)
    conn.close()
    
    # Prepare features and target
    target_col = 'final_test'
    X = data.drop(columns=[target_col])
    y = data[target_col].dropna()
    X = X.loc[y.index]  # Align with non-null target values
    
    # Run advanced preprocessing
    orchestrator = AdvancedPreprocessingOrchestrator()
    results = orchestrator.run_comprehensive_preprocessing(
        X, y, target_col=target_col
    )
    
    # Save results
    orchestrator.save_preprocessing_artifacts('data/processed/advanced_preprocessing')
    
    print("Advanced Preprocessing completed successfully!")
    print(f"Best strategy: {max(results['pipeline_comparison'].keys(), key=lambda x: results['pipeline_comparison'][x]['r2']['mean'])}")
    print(f"Recommendations: {len(results['recommendations'])}")


if __name__ == "__main__":
    main()