#!/usr/bin/env python3
"""
Phase 6 Task 6.1.1: Comprehensive Perfect Performance Investigation

This module conducts a thorough investigation into the persistent perfect
Linear Regression performance (R² = 1.0, MAE ≈ 0) despite data leakage fixes.

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
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensivePerformanceInvestigator:
    """
    Conducts comprehensive investigation into perfect model performance
    to identify root causes and validate data integrity.
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
        
        # Ensure output directory exists
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"Output path: {self.output_path}")
    
    def load_data(self) -> pd.DataFrame:
        """Load the clean dataset for analysis."""
        data_file = self.data_path / "clean_dataset_no_leakage.csv"
        
        if not data_file.exists():
            raise FileNotFoundError(f"Clean dataset not found at {data_file}")
        
        logger.info(f"Loading data from {data_file}")
        df = pd.read_csv(data_file)
        logger.info(f"Data loaded: {df.shape}")
        
        return df
    
    def investigate_target_variable(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive analysis of target variable properties."""
        logger.info("Investigating target variable properties...")
        
        target = df['final_test'].dropna()
        
        analysis = {
            'basic_stats': {
                'count': len(target),
                'mean': float(target.mean()),
                'std': float(target.std()),
                'min': float(target.min()),
                'max': float(target.max()),
                'median': float(target.median()),
                'unique_values': int(target.nunique()),
                'missing_values': int(df['final_test'].isna().sum())
            },
            'distribution_analysis': {
                'skewness': float(target.skew()),
                'kurtosis': float(target.kurtosis()),
                'is_normal': bool(stats.normaltest(target)[1] > 0.05),
                'normality_pvalue': float(stats.normaltest(target)[1])
            },
            'value_patterns': {
                'value_counts': target.value_counts().head(10).to_dict(),
                'repeated_values_pct': float((target.value_counts() > 1).sum() / len(target.value_counts()) * 100),
                'single_occurrence_pct': float((target.value_counts() == 1).sum() / len(target.value_counts()) * 100)
            },
            'potential_issues': []
        }
        
        # Check for potential issues
        if analysis['basic_stats']['std'] < 0.1:
            analysis['potential_issues'].append("Very low variance in target variable")
        
        if analysis['basic_stats']['unique_values'] < 10:
            analysis['potential_issues'].append("Very few unique target values")
        
        if analysis['value_patterns']['repeated_values_pct'] < 50:
            analysis['potential_issues'].append("Most target values appear only once")
        
        return analysis
    
    def investigate_feature_target_relationships(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze relationships between features and target."""
        logger.info("Investigating feature-target relationships...")
        
        target = df['final_test']
        features = df.drop(['final_test'], axis=1)
        
        # Get numerical features
        numerical_features = features.select_dtypes(include=[np.number]).columns.tolist()
        
        correlations = []
        perfect_predictors = []
        near_perfect_predictors = []
        
        for feature in numerical_features:
            if feature in df.columns:
                feature_data = df[feature].dropna()
                target_aligned = target[feature_data.index].dropna()
                
                if len(feature_data) > 1 and len(target_aligned) > 1:
                    # Align the data
                    common_idx = feature_data.index.intersection(target_aligned.index)
                    if len(common_idx) > 1:
                        feat_vals = feature_data[common_idx]
                        targ_vals = target_aligned[common_idx]
                        
                        # Remove any remaining NaN values
                        mask = ~(pd.isna(feat_vals) | pd.isna(targ_vals))
                        feat_vals = feat_vals[mask]
                        targ_vals = targ_vals[mask]
                        
                        if len(feat_vals) > 1 and feat_vals.std() > 0:
                            try:
                                corr, p_value = pearsonr(feat_vals, targ_vals)
                                
                                correlation_info = {
                                    'feature': feature,
                                    'correlation': float(corr) if not np.isnan(corr) else 0.0,
                                    'p_value': float(p_value) if not np.isnan(p_value) else 1.0,
                                    'abs_correlation': float(abs(corr)) if not np.isnan(corr) else 0.0,
                                    'sample_size': len(feat_vals)
                                }
                                
                                correlations.append(correlation_info)
                                
                                # Check for perfect or near-perfect correlations
                                if abs(corr) > 0.999:
                                    perfect_predictors.append(feature)
                                elif abs(corr) > 0.95:
                                    near_perfect_predictors.append(feature)
                                    
                            except Exception as e:
                                logger.warning(f"Could not calculate correlation for {feature}: {e}")
        
        # Sort correlations by absolute value
        correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
        
        analysis = {
            'total_numerical_features': len(numerical_features),
            'correlations_calculated': len(correlations),
            'perfect_predictors': perfect_predictors,
            'near_perfect_predictors': near_perfect_predictors,
            'top_correlations': correlations[:20],  # Top 20 correlations
            'correlation_stats': {
                'max_correlation': max([c['abs_correlation'] for c in correlations]) if correlations else 0,
                'mean_correlation': np.mean([c['abs_correlation'] for c in correlations]) if correlations else 0,
                'median_correlation': np.median([c['abs_correlation'] for c in correlations]) if correlations else 0
            }
        }
        
        return analysis
    
    def investigate_data_splits(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Investigate train/test split methodology and potential issues."""
        logger.info("Investigating data split methodology...")
        
        # Prepare features and target
        X = df.drop(['final_test'], axis=1)
        y = df['final_test']
        
        # Remove rows with missing target
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        # Get numerical features for analysis
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numerical = X[numerical_features]
        
        # Fill missing values for analysis
        X_filled = X_numerical.fillna(X_numerical.mean())
        
        split_analysis = {
            'data_size': len(X),
            'feature_count': len(X.columns),
            'numerical_feature_count': len(numerical_features),
            'missing_target_count': int(df['final_test'].isna().sum()),
            'split_experiments': []
        }
        
        # Test different random states and split ratios
        random_states = [42, 123, 456, 789, 999]
        test_sizes = [0.2, 0.25, 0.3]
        
        for rs in random_states:
            for test_size in test_sizes:
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_filled, y, test_size=test_size, random_state=rs
                    )
                    
                    # Train simple linear regression
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    
                    # Evaluate
                    y_pred = model.predict(X_test)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    split_analysis['split_experiments'].append({
                        'random_state': rs,
                        'test_size': test_size,
                        'train_size': len(X_train),
                        'test_size_actual': len(X_test),
                        'mae': float(mae),
                        'r2': float(r2),
                        'perfect_performance': bool(r2 > 0.999)
                    })
                    
                except Exception as e:
                    logger.warning(f"Split experiment failed (rs={rs}, test_size={test_size}): {e}")
        
        # Analyze results
        perfect_splits = [exp for exp in split_analysis['split_experiments'] if exp['perfect_performance']]
        split_analysis['perfect_performance_count'] = len(perfect_splits)
        split_analysis['total_experiments'] = len(split_analysis['split_experiments'])
        split_analysis['perfect_performance_rate'] = len(perfect_splits) / len(split_analysis['split_experiments']) if split_analysis['split_experiments'] else 0
        
        return split_analysis
    
    def investigate_cross_validation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Investigate model performance using cross-validation."""
        logger.info("Investigating cross-validation performance...")
        
        # Prepare data
        X = df.drop(['final_test'], axis=1)
        y = df['final_test']
        
        # Remove rows with missing target
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        # Get numerical features
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numerical = X[numerical_features].fillna(X[numerical_features].mean())
        
        cv_analysis = {
            'data_size': len(X),
            'cv_experiments': []
        }
        
        # Test different CV strategies
        cv_strategies = [
            {'name': '5-fold', 'cv': KFold(n_splits=5, shuffle=True, random_state=42)},
            {'name': '10-fold', 'cv': KFold(n_splits=10, shuffle=True, random_state=42)},
            {'name': '5-fold-no-shuffle', 'cv': KFold(n_splits=5, shuffle=False)}
        ]
        
        models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'RandomForest': RandomForestRegressor(n_estimators=10, random_state=42)
        }
        
        for cv_strategy in cv_strategies:
            for model_name, model in models.items():
                try:
                    # Perform cross-validation
                    cv_scores = cross_val_score(
                        model, X_numerical, y, 
                        cv=cv_strategy['cv'], 
                        scoring='r2',
                        n_jobs=1
                    )
                    
                    cv_mae_scores = cross_val_score(
                        model, X_numerical, y, 
                        cv=cv_strategy['cv'], 
                        scoring='neg_mean_absolute_error',
                        n_jobs=1
                    )
                    
                    cv_analysis['cv_experiments'].append({
                        'cv_strategy': cv_strategy['name'],
                        'model': model_name,
                        'r2_scores': cv_scores.tolist(),
                        'r2_mean': float(cv_scores.mean()),
                        'r2_std': float(cv_scores.std()),
                        'mae_scores': (-cv_mae_scores).tolist(),
                        'mae_mean': float((-cv_mae_scores).mean()),
                        'mae_std': float((-cv_mae_scores).std()),
                        'perfect_folds': int((cv_scores > 0.999).sum()),
                        'near_perfect_folds': int((cv_scores > 0.95).sum())
                    })
                    
                except Exception as e:
                    logger.warning(f"CV experiment failed ({cv_strategy['name']}, {model_name}): {e}")
        
        return cv_analysis
    
    def investigate_feature_engineering_pipeline(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Investigate the feature engineering pipeline for potential issues."""
        logger.info("Investigating feature engineering pipeline...")
        
        pipeline_analysis = {
            'total_features': len(df.columns) - 1,  # Exclude target
            'feature_types': {},
            'derived_features': [],
            'interaction_features': [],
            'aggregated_features': [],
            'potential_leakage_features': []
        }
        
        # Analyze feature types
        for col in df.columns:
            if col != 'final_test':
                dtype = str(df[col].dtype)
                if dtype not in pipeline_analysis['feature_types']:
                    pipeline_analysis['feature_types'][dtype] = []
                pipeline_analysis['feature_types'][dtype].append(col)
        
        # Identify different types of engineered features
        for col in df.columns:
            if col != 'final_test':
                col_lower = col.lower()
                
                # Check for interaction features
                if '_x_' in col or '_div_' in col or '_minus_' in col:
                    pipeline_analysis['interaction_features'].append(col)
                
                # Check for aggregated features
                if any(agg in col_lower for agg in ['_mean_', '_median_', '_std_', '_deviation_']):
                    pipeline_analysis['aggregated_features'].append(col)
                
                # Check for derived features
                if any(derived in col_lower for derived in ['_squared', '_ratio', '_score', '_index', '_category']):
                    pipeline_analysis['derived_features'].append(col)
                
                # Check for potential leakage (features that might contain target info)
                if any(leak in col_lower for leak in ['performance', 'test', 'score', 'grade']):
                    pipeline_analysis['potential_leakage_features'].append(col)
        
        return pipeline_analysis
    
    def run_comprehensive_investigation(self) -> Dict[str, Any]:
        """Run the complete investigation and return results."""
        logger.info("Starting comprehensive perfect performance investigation...")
        
        try:
            # Load data
            df = self.load_data()
            
            # Run all investigations
            investigation_results = {
                'investigation_timestamp': datetime.now().isoformat(),
                'dataset_info': {
                    'shape': list(df.shape),
                    'columns': list(df.columns),
                    'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
                },
                'target_investigation': self.investigate_target_variable(df),
                'feature_target_relationships': self.investigate_feature_target_relationships(df),
                'data_split_investigation': self.investigate_data_splits(df),
                'cross_validation_investigation': self.investigate_cross_validation(df),
                'feature_engineering_investigation': self.investigate_feature_engineering_pipeline(df)
            }
            
            # Generate summary and recommendations
            investigation_results['summary'] = self.generate_summary(investigation_results)
            investigation_results['recommendations'] = self.generate_recommendations(investigation_results)
            
            # Save results
            output_file = self.output_path / "comprehensive_performance_investigation.json"
            with open(output_file, 'w') as f:
                json.dump(investigation_results, f, indent=2, default=str)
            
            logger.info(f"Investigation completed. Results saved to {output_file}")
            
            return investigation_results
            
        except Exception as e:
            logger.error(f"Investigation failed: {e}")
            raise
    
    def generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of investigation findings."""
        summary = {
            'critical_findings': [],
            'data_quality_issues': [],
            'model_performance_patterns': [],
            'feature_engineering_issues': []
        }
        
        # Analyze target variable
        target_analysis = results['target_investigation']
        if target_analysis['basic_stats']['std'] < 1.0:
            summary['data_quality_issues'].append("Very low target variable variance")
        
        if len(target_analysis['potential_issues']) > 0:
            summary['data_quality_issues'].extend(target_analysis['potential_issues'])
        
        # Analyze feature relationships
        feature_analysis = results['feature_target_relationships']
        if len(feature_analysis['perfect_predictors']) > 0:
            summary['critical_findings'].append(f"Found {len(feature_analysis['perfect_predictors'])} perfect predictors")
        
        if len(feature_analysis['near_perfect_predictors']) > 0:
            summary['critical_findings'].append(f"Found {len(feature_analysis['near_perfect_predictors'])} near-perfect predictors")
        
        # Analyze cross-validation
        cv_analysis = results['cross_validation_investigation']
        perfect_cv_experiments = [exp for exp in cv_analysis['cv_experiments'] if exp['r2_mean'] > 0.999]
        if len(perfect_cv_experiments) > 0:
            summary['model_performance_patterns'].append(f"Perfect performance in {len(perfect_cv_experiments)} CV experiments")
        
        # Analyze feature engineering
        fe_analysis = results['feature_engineering_investigation']
        if len(fe_analysis['potential_leakage_features']) > 0:
            summary['feature_engineering_issues'].append(f"Found {len(fe_analysis['potential_leakage_features'])} potential leakage features")
        
        return summary
    
    def generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on findings."""
        recommendations = []
        
        summary = results['summary']
        
        # Recommendations based on critical findings
        if any('perfect predictors' in finding for finding in summary['critical_findings']):
            recommendations.append("URGENT: Remove perfect predictor features - they indicate data leakage")
        
        if any('near-perfect predictors' in finding for finding in summary['critical_findings']):
            recommendations.append("Investigate near-perfect predictors for potential data leakage")
        
        # Recommendations based on data quality
        if any('variance' in issue for issue in summary['data_quality_issues']):
            recommendations.append("Investigate target variable - low variance may indicate data quality issues")
        
        # Recommendations based on feature engineering
        if len(results['feature_engineering_investigation']['potential_leakage_features']) > 0:
            recommendations.append("Review and potentially remove features with performance-related names")
        
        # General recommendations
        if len(summary['critical_findings']) == 0 and len(summary['data_quality_issues']) == 0:
            recommendations.append("No obvious data leakage found - investigate model complexity and regularization")
        
        recommendations.append("Consider using more complex validation strategies (time-based, group-based)")
        recommendations.append("Implement feature importance analysis to understand model behavior")
        
        return recommendations

def main():
    """Main execution function."""
    try:
        investigator = ComprehensivePerformanceInvestigator()
        results = investigator.run_comprehensive_investigation()
        
        print("\n=== COMPREHENSIVE PERFORMANCE INVESTIGATION COMPLETED ===")
        print(f"Investigation timestamp: {results['investigation_timestamp']}")
        print(f"Dataset shape: {results['dataset_info']['shape']}")
        
        print("\n=== CRITICAL FINDINGS ===")
        for finding in results['summary']['critical_findings']:
            print(f"- {finding}")
        
        print("\n=== DATA QUALITY ISSUES ===")
        for issue in results['summary']['data_quality_issues']:
            print(f"- {issue}")
        
        print("\n=== RECOMMENDATIONS ===")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"{i}. {rec}")
        
        print("\n=== INVESTIGATION COMPLETE ===")
        
    except Exception as e:
        logger.error(f"Investigation failed: {e}")
        raise

if __name__ == "__main__":
    main()