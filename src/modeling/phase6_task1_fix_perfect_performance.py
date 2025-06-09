#!/usr/bin/env python3
"""
Phase 6 Task 6.1.2: Fix Perfect Performance by Removing Data Leakage

This module removes the identified perfect predictor feature 'performance_vs_age_peers'
and validates that the model performance becomes realistic.

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
from typing import Dict, Any, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Modeling imports
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerfectPerformanceFixer:
    """
    Fixes the perfect performance issue by removing data leakage features
    and validating that model performance becomes realistic.
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
        """Load the current dataset."""
        data_file = self.data_path / "clean_dataset_no_leakage.csv"
        
        if not data_file.exists():
            raise FileNotFoundError(f"Clean dataset not found at {data_file}")
        
        logger.info(f"Loading data from {data_file}")
        df = pd.read_csv(data_file)
        logger.info(f"Data loaded: {df.shape}")
        
        return df
    
    def identify_leakage_features(self, df: pd.DataFrame) -> List[str]:
        """Identify features that cause data leakage based on investigation results."""
        logger.info("Identifying data leakage features...")
        
        # Load investigation results
        investigation_file = self.output_path / "comprehensive_performance_investigation.json"
        
        if investigation_file.exists():
            with open(investigation_file, 'r') as f:
                investigation_results = json.load(f)
            
            perfect_predictors = investigation_results['feature_target_relationships']['perfect_predictors']
            near_perfect_predictors = investigation_results['feature_target_relationships']['near_perfect_predictors']
            
            logger.info(f"Perfect predictors found: {perfect_predictors}")
            logger.info(f"Near-perfect predictors found: {near_perfect_predictors}")
            
            # Combine both types of problematic features
            leakage_features = perfect_predictors + near_perfect_predictors
        else:
            # Fallback: manually identify known leakage feature
            leakage_features = ['performance_vs_age_peers']
            logger.warning("Investigation results not found, using manual identification")
        
        # Additional manual checks for suspicious feature names
        suspicious_patterns = ['performance', 'test_score', 'grade', 'result']
        additional_leakage = []
        
        for col in df.columns:
            if col != 'final_test':  # Don't include target
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in suspicious_patterns):
                    if col not in leakage_features:
                        additional_leakage.append(col)
        
        if additional_leakage:
            logger.info(f"Additional suspicious features found: {additional_leakage}")
            leakage_features.extend(additional_leakage)
        
        # Verify features exist in dataset
        existing_leakage_features = [f for f in leakage_features if f in df.columns]
        missing_features = [f for f in leakage_features if f not in df.columns]
        
        if missing_features:
            logger.warning(f"Some leakage features not found in dataset: {missing_features}")
        
        logger.info(f"Final leakage features to remove: {existing_leakage_features}")
        return existing_leakage_features
    
    def remove_leakage_features(self, df: pd.DataFrame, leakage_features: List[str]) -> pd.DataFrame:
        """Remove identified leakage features from the dataset."""
        logger.info(f"Removing {len(leakage_features)} leakage features...")
        
        # Create a copy of the dataframe
        df_clean = df.copy()
        
        # Remove leakage features
        features_to_remove = [f for f in leakage_features if f in df_clean.columns]
        df_clean = df_clean.drop(columns=features_to_remove)
        
        logger.info(f"Dataset shape before removal: {df.shape}")
        logger.info(f"Dataset shape after removal: {df_clean.shape}")
        logger.info(f"Features removed: {features_to_remove}")
        
        return df_clean, features_to_remove
    
    def validate_performance_fix(self, df_clean: pd.DataFrame) -> Dict[str, Any]:
        """Validate that removing leakage features fixes the perfect performance issue."""
        logger.info("Validating performance fix...")
        
        # Prepare data
        X = df_clean.drop(['final_test'], axis=1)
        y = df_clean['final_test']
        
        # Remove rows with missing target
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        # Get numerical features and handle missing values
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numerical = X[numerical_features]
        
        # Simple imputation for missing values
        imputer = SimpleImputer(strategy='mean')
        X_filled = pd.DataFrame(
            imputer.fit_transform(X_numerical),
            columns=X_numerical.columns,
            index=X_numerical.index
        )
        
        validation_results = {
            'data_info': {
                'total_samples': len(X),
                'total_features': len(X.columns),
                'numerical_features': len(numerical_features),
                'missing_target_count': int(df_clean['final_test'].isna().sum())
            },
            'model_performance': {},
            'cross_validation_results': {},
            'performance_fixed': False
        }
        
        # Test multiple models with different train/test splits
        models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42)
        }
        
        random_states = [42, 123, 456]
        
        for model_name, model in models.items():
            model_results = []
            
            for rs in random_states:
                try:
                    # Train/test split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_filled, y, test_size=0.2, random_state=rs
                    )
                    
                    # Train model
                    model_copy = model.__class__(**model.get_params())
                    model_copy.fit(X_train, y_train)
                    
                    # Evaluate
                    y_pred = model_copy.predict(X_test)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    
                    model_results.append({
                        'random_state': rs,
                        'mae': float(mae),
                        'r2': float(r2),
                        'rmse': float(rmse),
                        'perfect_performance': bool(r2 > 0.999)
                    })
                    
                except Exception as e:
                    logger.warning(f"Model evaluation failed for {model_name} (rs={rs}): {e}")
            
            if model_results:
                validation_results['model_performance'][model_name] = {
                    'individual_results': model_results,
                    'mean_mae': float(np.mean([r['mae'] for r in model_results])),
                    'mean_r2': float(np.mean([r['r2'] for r in model_results])),
                    'mean_rmse': float(np.mean([r['rmse'] for r in model_results])),
                    'perfect_performance_count': sum([r['perfect_performance'] for r in model_results]),
                    'total_experiments': len(model_results)
                }
        
        # Cross-validation testing
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for model_name, model in models.items():
            try:
                cv_r2_scores = cross_val_score(model, X_filled, y, cv=cv, scoring='r2')
                cv_mae_scores = cross_val_score(model, X_filled, y, cv=cv, scoring='neg_mean_absolute_error')
                
                validation_results['cross_validation_results'][model_name] = {
                    'r2_scores': cv_r2_scores.tolist(),
                    'r2_mean': float(cv_r2_scores.mean()),
                    'r2_std': float(cv_r2_scores.std()),
                    'mae_scores': (-cv_mae_scores).tolist(),
                    'mae_mean': float((-cv_mae_scores).mean()),
                    'mae_std': float((-cv_mae_scores).std()),
                    'perfect_folds': int((cv_r2_scores > 0.999).sum())
                }
                
            except Exception as e:
                logger.warning(f"Cross-validation failed for {model_name}: {e}")
        
        # Determine if performance is fixed
        perfect_performance_found = False
        
        for model_name, results in validation_results['model_performance'].items():
            if results['perfect_performance_count'] > 0:
                perfect_performance_found = True
                break
        
        for model_name, results in validation_results['cross_validation_results'].items():
            if results.get('perfect_folds', 0) > 0:
                perfect_performance_found = True
                break
        
        validation_results['performance_fixed'] = not perfect_performance_found
        
        return validation_results
    
    def save_clean_dataset(self, df_clean: pd.DataFrame, removed_features: List[str]) -> str:
        """Save the cleaned dataset without leakage features."""
        output_file = self.output_path / "clean_dataset_final_no_leakage.csv"
        
        logger.info(f"Saving cleaned dataset to {output_file}")
        df_clean.to_csv(output_file, index=False)
        
        # Save metadata about the cleaning process
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'original_shape': list(df_clean.shape),
            'removed_features': removed_features,
            'remaining_features': list(df_clean.columns),
            'cleaning_reason': 'Removed perfect predictor features causing data leakage'
        }
        
        metadata_file = self.output_path / "dataset_cleaning_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Cleaning metadata saved to {metadata_file}")
        
        return str(output_file)
    
    def retrain_models(self, df_clean: pd.DataFrame) -> Dict[str, Any]:
        """Retrain models on the cleaned dataset and save them."""
        logger.info("Retraining models on cleaned dataset...")
        
        # Prepare data
        X = df_clean.drop(['final_test'], axis=1)
        y = df_clean['final_test']
        
        # Remove rows with missing target
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        # Get numerical features and handle missing values
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numerical = X[numerical_features]
        
        # Simple imputation for missing values
        imputer = SimpleImputer(strategy='mean')
        X_filled = pd.DataFrame(
            imputer.fit_transform(X_numerical),
            columns=X_numerical.columns,
            index=X_numerical.index
        )
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_filled, y, test_size=0.2, random_state=42
        )
        
        # Models to train
        models = {
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        retrain_results = {
            'timestamp': datetime.now().isoformat(),
            'data_info': {
                'train_size': len(X_train),
                'test_size': len(X_test),
                'feature_count': len(X_filled.columns)
            },
            'model_results': {},
            'best_model': None
        }
        
        best_mae = float('inf')
        best_model_name = None
        
        for model_name, model in models.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                train_mae = mean_absolute_error(y_train, y_pred_train)
                test_mae = mean_absolute_error(y_test, y_pred_test)
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                
                model_result = {
                    'train_mae': float(train_mae),
                    'test_mae': float(test_mae),
                    'train_r2': float(train_r2),
                    'test_r2': float(test_r2),
                    'overfitting_ratio': float(test_mae / train_mae) if train_mae > 0 else 1.0
                }
                
                retrain_results['model_results'][model_name] = model_result
                
                # Track best model
                if test_mae < best_mae:
                    best_mae = test_mae
                    best_model_name = model_name
                
                # Save model
                model_file = self.output_path / f"models/{model_name}_fixed.joblib"
                model_file.parent.mkdir(exist_ok=True)
                joblib.dump(model, model_file)
                logger.info(f"Model saved: {model_file}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
        
        retrain_results['best_model'] = best_model_name
        
        return retrain_results
    
    def run_fix_process(self) -> Dict[str, Any]:
        """Run the complete fix process."""
        logger.info("Starting perfect performance fix process...")
        
        try:
            # Load data
            df = self.load_data()
            
            # Identify leakage features
            leakage_features = self.identify_leakage_features(df)
            
            if not leakage_features:
                logger.warning("No leakage features identified")
                return {
                    'status': 'no_leakage_found',
                    'message': 'No data leakage features were identified'
                }
            
            # Remove leakage features
            df_clean, removed_features = self.remove_leakage_features(df, leakage_features)
            
            # Validate the fix
            validation_results = self.validate_performance_fix(df_clean)
            
            # Save cleaned dataset
            clean_dataset_path = self.save_clean_dataset(df_clean, removed_features)
            
            # Retrain models
            retrain_results = self.retrain_models(df_clean)
            
            # Compile final results
            fix_results = {
                'timestamp': datetime.now().isoformat(),
                'status': 'completed',
                'leakage_features_removed': removed_features,
                'clean_dataset_path': clean_dataset_path,
                'validation_results': validation_results,
                'retrain_results': retrain_results,
                'performance_fixed': validation_results['performance_fixed']
            }
            
            # Save results
            results_file = self.output_path / "perfect_performance_fix_results.json"
            with open(results_file, 'w') as f:
                json.dump(fix_results, f, indent=2, default=str)
            
            logger.info(f"Fix process completed. Results saved to {results_file}")
            
            return fix_results
            
        except Exception as e:
            logger.error(f"Fix process failed: {e}")
            raise

def main():
    """Main execution function."""
    try:
        fixer = PerfectPerformanceFixer()
        results = fixer.run_fix_process()
        
        print("\n=== PERFECT PERFORMANCE FIX COMPLETED ===")
        print(f"Status: {results['status']}")
        print(f"Timestamp: {results['timestamp']}")
        
        if results['status'] == 'completed':
            print(f"\nFeatures removed: {results['leakage_features_removed']}")
            print(f"Performance fixed: {results['performance_fixed']}")
            print(f"Clean dataset saved to: {results['clean_dataset_path']}")
            
            print("\n=== MODEL PERFORMANCE AFTER FIX ===")
            for model_name, model_results in results['retrain_results']['model_results'].items():
                print(f"{model_name}:")
                print(f"  Test MAE: {model_results['test_mae']:.6f}")
                print(f"  Test R²: {model_results['test_r2']:.6f}")
                print(f"  Overfitting ratio: {model_results['overfitting_ratio']:.3f}")
            
            print(f"\nBest model: {results['retrain_results']['best_model']}")
        
        print("\n=== FIX PROCESS COMPLETE ===")
        
    except Exception as e:
        logger.error(f"Fix process failed: {e}")
        raise

if __name__ == "__main__":
    main()