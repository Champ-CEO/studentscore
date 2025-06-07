#!/usr/bin/env python3
"""
Phase 4 Task 4.1: Feature Selection (Medium Priority)

This module implements task 4.4.1 from TASKS.md:
- Correlation analysis and multicollinearity detection
- Feature importance analysis
- Statistical feature selection methods
- Dimensionality reduction if needed

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, RFE, RFECV,
    f_regression, mutual_info_regression,
    VarianceThreshold
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Phase4FeatureSelection:
    """
    Applies feature selection for Phase 4 feature engineering.
    """
    
    def __init__(self, df: pd.DataFrame, target_column: str = 'final_test'):
        """
        Initialize with the dataset.
        
        Args:
            df: DataFrame with preprocessed features from previous tasks
            target_column: Name of the target variable column
        """
        self.df = df.copy()
        self.target_column = target_column
        self.selection_results = {}
        self.selected_features = {}
        self.feature_importance_scores = {}
        self.correlation_analysis = {}
        self.audit_log = []
        
        # Validate target column
        if target_column not in self.df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
            
    def prepare_features_and_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare feature matrix and target vector.
        
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Exclude target and ID columns
        exclude_columns = [self.target_column, 'student_id']
        exclude_columns.extend([col for col in self.df.columns if 'id' in col.lower()])
        
        # Get feature columns
        feature_columns = [col for col in self.df.columns if col not in exclude_columns]
        
        # Remove non-numeric columns that weren't properly encoded
        numeric_features = []
        for col in feature_columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                numeric_features.append(col)
            else:
                logger.warning(f"Skipping non-numeric feature: {col}")
                
        X = self.df[numeric_features].copy()
        y = self.df[self.target_column].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        y = y.fillna(y.median())
        
        logger.info(f"Prepared features: {X.shape[1]} features, {X.shape[0]} samples")
        
        return X, y
        
    def analyze_correlations(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Analyze correlations between features and with target.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary containing correlation analysis results
        """
        logger.info("Analyzing feature correlations")
        
        # Feature-target correlations
        target_correlations = {}
        for feature in X.columns:
            try:
                # Pearson correlation
                pearson_corr, pearson_p = pearsonr(X[feature], y)
                # Spearman correlation
                spearman_corr, spearman_p = spearmanr(X[feature], y)
                
                target_correlations[feature] = {
                    'pearson_correlation': pearson_corr,
                    'pearson_p_value': pearson_p,
                    'spearman_correlation': spearman_corr,
                    'spearman_p_value': spearman_p,
                    'abs_pearson': abs(pearson_corr),
                    'significant': pearson_p < 0.05
                }
            except Exception as e:
                logger.warning(f"Failed to calculate correlation for {feature}: {str(e)}")
                
        # Feature-feature correlations (multicollinearity)
        feature_correlations = X.corr()
        
        # Find highly correlated feature pairs
        high_correlations = []
        correlation_threshold = 0.8
        
        for i in range(len(feature_correlations.columns)):
            for j in range(i+1, len(feature_correlations.columns)):
                feature1 = feature_correlations.columns[i]
                feature2 = feature_correlations.columns[j]
                correlation = feature_correlations.iloc[i, j]
                
                if abs(correlation) > correlation_threshold:
                    high_correlations.append({
                        'feature1': feature1,
                        'feature2': feature2,
                        'correlation': correlation,
                        'abs_correlation': abs(correlation)
                    })
                    
        # Sort by absolute correlation
        high_correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
        
        self.correlation_analysis = {
            'target_correlations': target_correlations,
            'feature_correlations': feature_correlations.to_dict(),
            'high_correlations': high_correlations,
            'correlation_threshold': correlation_threshold
        }
        
        logger.info(f"Found {len(high_correlations)} highly correlated feature pairs (|r| > {correlation_threshold})")
        
        return self.correlation_analysis
        
    def calculate_vif(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Variance Inflation Factor for multicollinearity detection.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary mapping feature names to VIF values
        """
        logger.info("Calculating Variance Inflation Factors")
        
        vif_scores = {}
        
        try:
            # Calculate VIF for each feature
            for i, feature in enumerate(X.columns):
                try:
                    vif_value = variance_inflation_factor(X.values, i)
                    vif_scores[feature] = vif_value
                except Exception as e:
                    logger.warning(f"Failed to calculate VIF for {feature}: {str(e)}")
                    vif_scores[feature] = np.nan
                    
            # Log high VIF features
            high_vif_threshold = 10
            high_vif_features = {k: v for k, v in vif_scores.items() if v > high_vif_threshold}
            
            if high_vif_features:
                logger.warning(f"Features with high VIF (>{high_vif_threshold}): {len(high_vif_features)}")
                for feature, vif in sorted(high_vif_features.items(), key=lambda x: x[1], reverse=True)[:5]:
                    logger.warning(f"  {feature}: {vif:.2f}")
                    
        except Exception as e:
            logger.error(f"VIF calculation failed: {str(e)}")
            
        return vif_scores
        
    def univariate_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, List[str]]:
        """
        Apply univariate feature selection methods.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary mapping method names to selected feature lists
        """
        logger.info("Applying univariate feature selection")
        
        univariate_results = {}
        
        # F-regression test
        try:
            f_selector = SelectKBest(score_func=f_regression, k='all')
            f_selector.fit(X, y)
            
            f_scores = dict(zip(X.columns, f_selector.scores_))
            f_pvalues = dict(zip(X.columns, f_selector.pvalues_))
            
            # Select features with p-value < 0.05
            f_selected = [feature for feature, p_val in f_pvalues.items() if p_val < 0.05]
            
            univariate_results['f_regression'] = {
                'selected_features': f_selected,
                'scores': f_scores,
                'p_values': f_pvalues
            }
            
            logger.info(f"F-regression selected {len(f_selected)} features")
            
        except Exception as e:
            logger.warning(f"F-regression selection failed: {str(e)}")
            
        # Mutual information
        try:
            mi_selector = SelectKBest(score_func=mutual_info_regression, k='all')
            mi_selector.fit(X, y)
            
            mi_scores = dict(zip(X.columns, mi_selector.scores_))
            
            # Select top 50% features by mutual information
            mi_threshold = np.percentile(list(mi_scores.values()), 50)
            mi_selected = [feature for feature, score in mi_scores.items() if score >= mi_threshold]
            
            univariate_results['mutual_info'] = {
                'selected_features': mi_selected,
                'scores': mi_scores,
                'threshold': mi_threshold
            }
            
            logger.info(f"Mutual information selected {len(mi_selected)} features")
            
        except Exception as e:
            logger.warning(f"Mutual information selection failed: {str(e)}")
            
        return univariate_results
        
    def model_based_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Apply model-based feature selection methods.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary containing model-based selection results
        """
        logger.info("Applying model-based feature selection")
        
        model_results = {}
        
        # Random Forest feature importance
        try:
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf_model.fit(X, y)
            
            rf_importance = dict(zip(X.columns, rf_model.feature_importances_))
            
            # Select top 50% features by importance
            importance_threshold = np.percentile(list(rf_importance.values()), 50)
            rf_selected = [feature for feature, importance in rf_importance.items() 
                          if importance >= importance_threshold]
            
            model_results['random_forest'] = {
                'selected_features': rf_selected,
                'feature_importance': rf_importance,
                'threshold': importance_threshold,
                'model_score': rf_model.score(X, y)
            }
            
            logger.info(f"Random Forest selected {len(rf_selected)} features")
            
        except Exception as e:
            logger.warning(f"Random Forest selection failed: {str(e)}")
            
        # Recursive Feature Elimination
        try:
            # Use a simpler model for RFE to avoid overfitting
            base_estimator = LinearRegression()
            rfe_selector = RFE(estimator=base_estimator, n_features_to_select=min(20, X.shape[1]//2))
            rfe_selector.fit(X, y)
            
            rfe_selected = X.columns[rfe_selector.support_].tolist()
            rfe_ranking = dict(zip(X.columns, rfe_selector.ranking_))
            
            model_results['rfe'] = {
                'selected_features': rfe_selected,
                'feature_ranking': rfe_ranking,
                'n_features_selected': len(rfe_selected)
            }
            
            logger.info(f"RFE selected {len(rfe_selected)} features")
            
        except Exception as e:
            logger.warning(f"RFE selection failed: {str(e)}")
            
        return model_results
        
    def variance_based_selection(self, X: pd.DataFrame) -> List[str]:
        """
        Remove features with low variance.
        
        Args:
            X: Feature matrix
            
        Returns:
            List of features with sufficient variance
        """
        logger.info("Applying variance-based feature selection")
        
        try:
            # Calculate variance for each feature
            feature_variances = X.var()
            
            # Remove features with very low variance (threshold = 0.01)
            variance_threshold = 0.01
            low_variance_features = feature_variances[feature_variances < variance_threshold].index.tolist()
            high_variance_features = feature_variances[feature_variances >= variance_threshold].index.tolist()
            
            logger.info(f"Removed {len(low_variance_features)} low-variance features")
            if low_variance_features:
                logger.info(f"Low variance features: {low_variance_features[:5]}{'...' if len(low_variance_features) > 5 else ''}")
                
            return high_variance_features
            
        except Exception as e:
            logger.warning(f"Variance-based selection failed: {str(e)}")
            return X.columns.tolist()
            
    def ensemble_feature_selection(self, selection_results: Dict[str, Any]) -> List[str]:
        """
        Combine results from multiple selection methods.
        
        Args:
            selection_results: Dictionary containing results from different selection methods
            
        Returns:
            List of features selected by ensemble method
        """
        logger.info("Applying ensemble feature selection")
        
        # Collect all selected features from different methods
        all_selected_features = set()
        method_selections = {}
        
        # Extract selected features from each method
        for method_name, results in selection_results.items():
            if isinstance(results, dict) and 'selected_features' in results:
                selected = results['selected_features']
                method_selections[method_name] = set(selected)
                all_selected_features.update(selected)
            elif isinstance(results, list):
                method_selections[method_name] = set(results)
                all_selected_features.update(results)
                
        # Count how many methods selected each feature
        feature_votes = {}
        for feature in all_selected_features:
            votes = sum(1 for method_features in method_selections.values() 
                       if feature in method_features)
            feature_votes[feature] = votes
            
        # Select features that were chosen by at least 2 methods (or 1 if only 1-2 methods available)
        min_votes = max(1, min(2, len(method_selections) // 2))
        ensemble_selected = [feature for feature, votes in feature_votes.items() 
                           if votes >= min_votes]
        
        # Sort by number of votes
        ensemble_selected.sort(key=lambda x: feature_votes[x], reverse=True)
        
        logger.info(f"Ensemble method selected {len(ensemble_selected)} features (min votes: {min_votes})")
        
        # Log top features by votes
        top_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info("Top features by votes:")
        for feature, votes in top_features:
            logger.info(f"  {feature}: {votes} votes")
            
        return ensemble_selected
        
    def evaluate_feature_sets(self, X: pd.DataFrame, y: pd.Series, 
                             feature_sets: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate different feature sets using cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_sets: Dictionary mapping set names to feature lists
            
        Returns:
            Dictionary containing evaluation results for each feature set
        """
        logger.info("Evaluating feature sets")
        
        evaluation_results = {}
        
        for set_name, features in feature_sets.items():
            if not features:
                continue
                
            try:
                # Select features
                X_subset = X[features]
                
                # Evaluate with Random Forest
                rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
                rf_scores = cross_val_score(rf_model, X_subset, y, cv=5, scoring='neg_mean_squared_error')
                
                # Evaluate with Linear Regression
                lr_model = LinearRegression()
                lr_scores = cross_val_score(lr_model, X_subset, y, cv=5, scoring='neg_mean_squared_error')
                
                evaluation_results[set_name] = {
                    'n_features': len(features),
                    'rf_mse_mean': -rf_scores.mean(),
                    'rf_mse_std': rf_scores.std(),
                    'lr_mse_mean': -lr_scores.mean(),
                    'lr_mse_std': lr_scores.std(),
                    'features': features
                }
                
                logger.info(f"{set_name}: {len(features)} features, RF MSE: {-rf_scores.mean():.4f}")
                
            except Exception as e:
                logger.warning(f"Evaluation failed for {set_name}: {str(e)}")
                
        return evaluation_results
        
    def select_final_features(self, evaluation_results: Dict[str, Dict[str, float]]) -> List[str]:
        """
        Select the final feature set based on evaluation results.
        
        Args:
            evaluation_results: Dictionary containing evaluation results
            
        Returns:
            List of final selected features
        """
        logger.info("Selecting final feature set")
        
        if not evaluation_results:
            logger.warning("No evaluation results available")
            return []
            
        # Find the best performing feature set (lowest RF MSE)
        best_set = min(evaluation_results.items(), 
                      key=lambda x: x[1]['rf_mse_mean'])
        
        best_set_name, best_results = best_set
        final_features = best_results['features']
        
        logger.info(f"Selected feature set: {best_set_name}")
        logger.info(f"Features: {len(final_features)}")
        logger.info(f"RF MSE: {best_results['rf_mse_mean']:.4f}")
        
        return final_features
        
    def run_feature_selection(self) -> Dict[str, Any]:
        """
        Run the complete feature selection pipeline.
        
        Returns:
            Dictionary containing all selection results
        """
        logger.info("Starting feature selection pipeline")
        
        # Prepare data
        X, y = self.prepare_features_and_target()
        
        # Correlation analysis
        correlation_results = self.analyze_correlations(X, y)
        
        # VIF analysis
        vif_scores = self.calculate_vif(X)
        
        # Variance-based selection
        high_variance_features = self.variance_based_selection(X)
        X_variance_filtered = X[high_variance_features]
        
        # Univariate selection
        univariate_results = self.univariate_feature_selection(X_variance_filtered, y)
        
        # Model-based selection
        model_results = self.model_based_feature_selection(X_variance_filtered, y)
        
        # Combine all selection results
        all_selection_results = {
            'variance_filtered': high_variance_features,
            **univariate_results,
            **model_results
        }
        
        # Ensemble selection
        ensemble_features = self.ensemble_feature_selection(all_selection_results)
        
        # Prepare feature sets for evaluation
        feature_sets = {
            'all_features': X.columns.tolist(),
            'variance_filtered': high_variance_features,
            'ensemble_selected': ensemble_features
        }
        
        # Add individual method results
        for method_name, results in all_selection_results.items():
            if isinstance(results, dict) and 'selected_features' in results:
                feature_sets[method_name] = results['selected_features']
            elif isinstance(results, list):
                feature_sets[method_name] = results
                
        # Evaluate feature sets
        evaluation_results = self.evaluate_feature_sets(X, y, feature_sets)
        
        # Select final features
        final_features = self.select_final_features(evaluation_results)
        
        # Store results
        self.selection_results = {
            'correlation_analysis': correlation_results,
            'vif_scores': vif_scores,
            'univariate_results': univariate_results,
            'model_results': model_results,
            'ensemble_features': ensemble_features,
            'evaluation_results': evaluation_results,
            'final_features': final_features,
            'feature_sets': feature_sets
        }
        
        logger.info(f"Feature selection completed. Final features: {len(final_features)}")
        
        return self.selection_results
        
    def create_selected_dataset(self, features: List[str] = None) -> pd.DataFrame:
        """
        Create dataset with selected features.
        
        Args:
            features: List of features to include. If None, use final selected features.
            
        Returns:
            DataFrame with selected features and target
        """
        if features is None:
            features = self.selection_results.get('final_features', [])
            
        if not features:
            logger.warning("No features selected. Using all numeric features.")
            X, y = self.prepare_features_and_target()
            features = X.columns.tolist()
            
        # Include target column
        selected_columns = features + [self.target_column]
        selected_columns = [col for col in selected_columns if col in self.df.columns]
        
        selected_df = self.df[selected_columns].copy()
        
        logger.info(f"Created selected dataset with {len(features)} features")
        
        return selected_df
        
    def save_selection_results(self, output_path: str = "data/featured/feature_selection_results.json") -> None:
        """
        Save feature selection results.
        
        Args:
            output_path: Path to save the results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.selection_results, f, indent=2, default=str)
            
        logger.info(f"Feature selection results saved to {output_path}")
        
    def save_selected_dataset(self, output_path: str = "data/featured/selected_features_dataset.csv") -> None:
        """
        Save dataset with selected features.
        
        Args:
            output_path: Path to save the dataset
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        selected_df = self.create_selected_dataset()
        selected_df.to_csv(output_path)
        
        logger.info(f"Selected features dataset saved to {output_path}")


def main():
    """
    Main function to run Phase 4 Task 4.1: Feature Selection.
    """
    try:
        # Load data with preprocessed features (assuming previous tasks completed)
        preprocessed_features_path = "data/featured/preprocessed_features.csv"
        
        if not Path(preprocessed_features_path).exists():
            raise FileNotFoundError(
                f"Preprocessed features file not found: {preprocessed_features_path}. "
                "Please run Phase 4 Task 3.1 first."
            )
            
        df = pd.read_csv(preprocessed_features_path, index_col=0)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        
        # Create feature selector
        selector = Phase4FeatureSelection(df)
        
        # Run feature selection
        results = selector.run_feature_selection()
        
        # Save results
        selector.save_selection_results()
        selector.save_selected_dataset()
        
        # Print summary
        final_features = results['final_features']
        print(f"\n=== Phase 4 Task 4.1 Complete ===")
        print(f"Original features: {df.shape[1]}")
        print(f"Selected features: {len(final_features)}")
        print(f"Reduction: {(1 - len(final_features)/df.shape[1])*100:.1f}%")
        
        if 'evaluation_results' in results:
            best_result = min(results['evaluation_results'].items(), 
                            key=lambda x: x[1]['rf_mse_mean'])
            print(f"Best feature set: {best_result[0]}")
            print(f"RF MSE: {best_result[1]['rf_mse_mean']:.4f}")
        
        return selector.create_selected_dataset()
        
    except Exception as e:
        logger.error(f"Phase 4 Task 4.1 failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()