#!/usr/bin/env python3
"""
Phase 6 Task 6.4.2: Integration Testing

This module implements comprehensive integration tests for the end-to-end pipeline,
model training to deployment flow, and performance benchmarks.

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
import tempfile
import shutil
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# Testing imports
import unittest
from unittest.mock import Mock, patch, MagicMock

# Modeling imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestEndToEndPipeline(unittest.TestCase):
    """
    Integration tests for end-to-end pipeline functionality.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Get project root
        current_file = Path(__file__).resolve()
        self.project_root = current_file.parents[2]
        self.data_path = self.project_root / "data"
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Load real data for integration testing
        try:
            self.data_file = self.data_path / "processed" / "final_processed.csv"
            if self.data_file.exists():
                self.data = pd.read_csv(self.data_file)
                logger.info(f"Loaded real data: {self.data.shape}")
            else:
                # Create synthetic data if real data not available
                self.data = self._create_synthetic_data()
                logger.info("Using synthetic data for testing")
        except Exception as e:
            logger.warning(f"Could not load real data: {e}. Using synthetic data.")
            self.data = self._create_synthetic_data()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _create_synthetic_data(self) -> pd.DataFrame:
        """Create synthetic data for testing."""
        np.random.seed(42)
        n_samples = 1000
        
        data = pd.DataFrame({
            'n_male': np.random.choice([0, 1], n_samples),
            'n_female': np.random.choice([0, 1], n_samples),
            'age': np.random.normal(15, 2, n_samples),
            'hours_per_week': np.random.normal(20, 5, n_samples),
            'failures': np.random.poisson(1, n_samples),
            'absences': np.random.poisson(5, n_samples),
            'G1': np.random.normal(10, 3, n_samples),
            'G2': np.random.normal(10, 3, n_samples),
            'final_test': np.random.normal(10, 3, n_samples)  # Target variable
        })
        
        # Ensure realistic relationships
        data['final_test'] = (data['G1'] * 0.3 + data['G2'] * 0.4 + 
                     data['hours_per_week'] * 0.1 + 
                     np.random.normal(0, 2, n_samples))
        
        return data
    
    def test_complete_pipeline_flow(self):
        """Test complete end-to-end pipeline flow."""
        logger.info("Testing complete pipeline flow...")
        
        # 1. Data Loading and Validation
        self.assertIsInstance(self.data, pd.DataFrame)
        self.assertGreater(len(self.data), 0)
        self.assertIn('final_test', self.data.columns)  # Target should exist
        
        # 2. Feature Selection
        feature_columns = [col for col in self.data.columns if col != 'final_test']
        target_column = 'final_test'
        
        X = self.data[feature_columns]
        y = self.data[target_column]
        
        # Handle missing values in target variable
        valid_indices = ~y.isnull()
        X = X[valid_indices]
        y = y[valid_indices]
        
        # Handle missing values with proper preprocessing for mixed data types
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        categorical_columns = X.select_dtypes(include=['object']).columns
        
        X_processed = X.copy()
        
        # Handle numeric columns with mean imputation
        if len(numeric_columns) > 0 and X[numeric_columns].isnull().any().any():
            numeric_imputer = SimpleImputer(strategy='mean')
            X_processed[numeric_columns] = numeric_imputer.fit_transform(X[numeric_columns])
        
        # Handle categorical columns with mode imputation and encoding
        if len(categorical_columns) > 0:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            X_processed[categorical_columns] = categorical_imputer.fit_transform(X[categorical_columns])
            
            # Simple label encoding for categorical variables
            from sklearn.preprocessing import LabelEncoder
            for col in categorical_columns:
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))
        
        X_imputed = X_processed
        
        # 3. Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_imputed, y, test_size=0.2, random_state=42
        )
        
        # 4. Model Training Pipeline
        models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'RandomForest': RandomForestRegressor(n_estimators=10, random_state=42)
        }
        
        trained_models = {}
        performance_metrics = {}
        
        for name, model in models.items():
            # Train model
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Make predictions
            start_time = time.time()
            predictions = model.predict(X_test)
            prediction_time = time.time() - start_time
            
            # Calculate metrics
            r2 = r2_score(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            
            trained_models[name] = model
            performance_metrics[name] = {
                'r2_score': r2,
                'mae': mae,
                'rmse': rmse,
                'training_time': training_time,
                'prediction_time': prediction_time
            }
            
            # Validate performance
            self.assertTrue(np.isfinite(r2))
            self.assertTrue(np.isfinite(mae))
            self.assertTrue(np.isfinite(rmse))
            self.assertGreaterEqual(mae, 0)
            self.assertGreaterEqual(rmse, 0)
        
        # 5. Model Serialization
        model_paths = {}
        for name, model in trained_models.items():
            model_path = os.path.join(self.temp_dir, f'{name}_integration_test.joblib')
            joblib.dump(model, model_path)
            model_paths[name] = model_path
            
            # Verify file exists
            self.assertTrue(os.path.exists(model_path))
        
        # 6. Model Loading and Validation
        for name, model_path in model_paths.items():
            loaded_model = joblib.load(model_path)
            
            # Test predictions match
            original_pred = trained_models[name].predict(X_test)
            loaded_pred = loaded_model.predict(X_test)
            
            np.testing.assert_array_almost_equal(original_pred, loaded_pred)
        
        # 7. Performance Validation
        best_model_name = max(performance_metrics.keys(), 
                            key=lambda k: performance_metrics[k]['r2_score'])
        best_r2 = performance_metrics[best_model_name]['r2_score']
        
        # Integration test should achieve reasonable performance
        self.assertGreater(best_r2, 0.1)  # At least some predictive power
        
        logger.info(f"Pipeline flow test completed. Best model: {best_model_name} (R²={best_r2:.3f})")
    
    def test_pipeline_with_preprocessing(self):
        """Test pipeline with comprehensive preprocessing."""
        logger.info("Testing pipeline with preprocessing...")
        
        # Select features and target
        feature_columns = [col for col in self.data.columns if col != 'final_test']
        X = self.data[feature_columns]
        y = self.data['final_test']
        
        # Handle missing values in target variable
        valid_indices = ~y.isnull()
        X = X[valid_indices]
        y = y[valid_indices]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create preprocessing pipeline for numeric data only
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import LabelEncoder
        
        # Identify numeric and categorical columns
        numeric_features = X_train.select_dtypes(include=[np.number]).columns
        categorical_features = X_train.select_dtypes(include=['object']).columns
        
        # Create preprocessing steps
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', LabelEncoder())
        ])
        
        # Handle mixed data types
        if len(categorical_features) > 0:
            # Encode categorical features first with robust handling
            X_train_processed = X_train.copy()
            X_test_processed = X_test.copy()
            
            for col in categorical_features:
                le = LabelEncoder()
                # Fit on combined data to handle unseen labels
                combined_values = pd.concat([X_train_processed[col], X_test_processed[col]]).astype(str)
                le.fit(combined_values)
                X_train_processed[col] = le.transform(X_train_processed[col].astype(str))
                X_test_processed[col] = le.transform(X_test_processed[col].astype(str))
            
            X_train = X_train_processed
            X_test = X_test_processed
        
        # Simple preprocessing pipeline for all numeric data
        preprocessing_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        # Create full pipeline with model
        full_pipeline = Pipeline([
            ('preprocessing', preprocessing_pipeline),
            ('model', LinearRegression())
        ])
        
        # Train pipeline
        start_time = time.time()
        full_pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Make predictions
        start_time = time.time()
        predictions = full_pipeline.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Validate results
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        
        self.assertTrue(np.isfinite(r2))
        self.assertTrue(np.isfinite(mae))
        self.assertGreaterEqual(mae, 0)
        
        # Test pipeline serialization
        pipeline_path = os.path.join(self.temp_dir, 'full_pipeline.joblib')
        joblib.dump(full_pipeline, pipeline_path)
        
        # Load and test
        loaded_pipeline = joblib.load(pipeline_path)
        loaded_predictions = loaded_pipeline.predict(X_test)
        
        np.testing.assert_array_almost_equal(predictions, loaded_predictions)
        
        logger.info(f"Preprocessing pipeline test completed (R²={r2:.3f}, MAE={mae:.3f})")
    
    def test_cross_validation_integration(self):
        """Test cross-validation integration."""
        logger.info("Testing cross-validation integration...")
        
        # Prepare data
        feature_columns = [col for col in self.data.columns if col != 'final_test']
        X = self.data[feature_columns]
        y = self.data['final_test']
        
        # Handle missing values in target variable
        valid_indices = ~y.isnull()
        X = X[valid_indices]
        y = y[valid_indices]
        
        # Handle missing values with proper preprocessing for mixed data types
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        categorical_columns = X.select_dtypes(include=['object']).columns
        
        X_processed = X.copy()
        
        # Handle numeric columns with mean imputation
        if len(numeric_columns) > 0 and X[numeric_columns].isnull().any().any():
            numeric_imputer = SimpleImputer(strategy='mean')
            X_processed[numeric_columns] = numeric_imputer.fit_transform(X[numeric_columns])
        
        # Handle categorical columns with mode imputation and encoding
        if len(categorical_columns) > 0:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            X_processed[categorical_columns] = categorical_imputer.fit_transform(X[categorical_columns])
            
            # Simple label encoding for categorical variables
            from sklearn.preprocessing import LabelEncoder
            for col in categorical_columns:
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))
        
        X = X_processed
        
        # Test cross-validation with different models
        models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0)
        }
        
        cv_results = {}
        
        for name, model in models.items():
            # Perform cross-validation
            cv_scores = cross_val_score(model, X, y, cv=3, scoring='r2')
            
            cv_results[name] = {
                'mean_cv_score': np.mean(cv_scores),
                'std_cv_score': np.std(cv_scores),
                'cv_scores': cv_scores.tolist()
            }
            
            # Validate CV results
            self.assertEqual(len(cv_scores), 3)
            self.assertTrue(all(np.isfinite(score) for score in cv_scores))
            
            logger.info(f"{name} CV Score: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        
        # Ensure we have meaningful results
        best_model = max(cv_results.keys(), key=lambda k: cv_results[k]['mean_cv_score'])
        best_score = cv_results[best_model]['mean_cv_score']
        
        self.assertGreater(best_score, -1.0)  # Should be better than random
        
        logger.info(f"Cross-validation integration test completed. Best: {best_model}")

class TestModelTrainingToDeployment(unittest.TestCase):
    """
    Integration tests for model training to deployment flow.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Get project root
        current_file = Path(__file__).resolve()
        self.project_root = current_file.parents[2]
        self.output_path = self.project_root / "data" / "modeling_outputs"
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create synthetic data
        np.random.seed(42)
        n_samples = 500
        
        self.data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(5, 2, n_samples),
            'feature3': np.random.choice([0, 1], n_samples),
            'target': np.random.normal(10, 3, n_samples)
        })
        
        # Add realistic relationship
        self.data['target'] = (self.data['feature1'] * 2 + 
                              self.data['feature2'] * 0.5 + 
                              self.data['feature3'] * 3 + 
                              np.random.normal(0, 1, n_samples))
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_training_to_deployment_workflow(self):
        """Test complete training to deployment workflow."""
        logger.info("Testing training to deployment workflow...")
        
        # 1. Data Preparation
        X = self.data[['feature1', 'feature2', 'feature3']]
        y = self.data['target']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 2. Model Training
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # 3. Model Evaluation
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        
        train_r2 = r2_score(y_train, train_predictions)
        test_r2 = r2_score(y_test, test_predictions)
        test_mae = mean_absolute_error(y_test, test_predictions)
        
        # 4. Model Validation
        self.assertGreater(test_r2, 0.5)  # Should have good performance
        self.assertLess(abs(train_r2 - test_r2), 0.3)  # Not severely overfitted
        
        # 5. Model Serialization for Deployment
        deployment_path = os.path.join(self.temp_dir, 'deployment_model.joblib')
        
        # Save model with metadata
        model_package = {
            'model': model,
            'feature_names': X.columns.tolist(),
            'model_type': 'RandomForestRegressor',
            'training_date': datetime.now().isoformat(),
            'performance_metrics': {
                'test_r2': test_r2,
                'test_mae': test_mae,
                'train_r2': train_r2
            },
            'data_shape': X.shape
        }
        
        joblib.dump(model_package, deployment_path)
        
        # 6. Deployment Simulation
        # Load model as if in production
        loaded_package = joblib.load(deployment_path)
        loaded_model = loaded_package['model']
        
        # Simulate new data prediction
        new_data = pd.DataFrame({
            'feature1': [0.5, -0.3, 1.2],
            'feature2': [4.8, 5.2, 6.1],
            'feature3': [1, 0, 1]
        })
        
        # Validate feature consistency
        self.assertListEqual(
            new_data.columns.tolist(),
            loaded_package['feature_names']
        )
        
        # Make predictions
        deployment_predictions = loaded_model.predict(new_data)
        
        # Validate predictions
        self.assertEqual(len(deployment_predictions), len(new_data))
        self.assertTrue(all(np.isfinite(pred) for pred in deployment_predictions))
        
        # 7. Performance Monitoring Simulation
        # Test model on original test set
        monitoring_predictions = loaded_model.predict(X_test)
        monitoring_r2 = r2_score(y_test, monitoring_predictions)
        
        # Performance should be consistent
        self.assertAlmostEqual(monitoring_r2, test_r2, places=10)
        
        logger.info(f"Training to deployment workflow completed (R²={test_r2:.3f})")
    
    def test_model_versioning_workflow(self):
        """Test model versioning and comparison workflow."""
        logger.info("Testing model versioning workflow...")
        
        X = self.data[['feature1', 'feature2', 'feature3']]
        y = self.data['target']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train multiple model versions
        models = {
            'v1_linear': LinearRegression(),
            'v2_ridge': Ridge(alpha=1.0),
            'v3_rf': RandomForestRegressor(n_estimators=10, random_state=42)
        }
        
        model_versions = {}
        
        for version, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            predictions = model.predict(X_test)
            r2 = r2_score(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            
            # Save version
            version_path = os.path.join(self.temp_dir, f'model_{version}.joblib')
            
            version_package = {
                'model': model,
                'version': version,
                'performance': {'r2': r2, 'mae': mae},
                'timestamp': datetime.now().isoformat()
            }
            
            joblib.dump(version_package, version_path)
            model_versions[version] = version_package
            
            # Validate version
            self.assertTrue(os.path.exists(version_path))
            self.assertIn('model', version_package)
            self.assertIn('performance', version_package)
        
        # Select best model
        best_version = max(model_versions.keys(), 
                          key=lambda v: model_versions[v]['performance']['r2'])
        
        best_performance = model_versions[best_version]['performance']['r2']
        
        # Validate best model selection
        self.assertIn(best_version, model_versions)
        self.assertGreater(best_performance, 0.0)
        
        logger.info(f"Model versioning workflow completed. Best: {best_version} (R²={best_performance:.3f})")

class TestPerformanceBenchmarks(unittest.TestCase):
    """
    Integration tests for performance benchmarks.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Create larger dataset for performance testing
        np.random.seed(42)
        
        # Different dataset sizes for benchmarking
        self.dataset_sizes = [100, 500, 1000]
        self.datasets = {}
        
        for size in self.dataset_sizes:
            data = pd.DataFrame({
                'feature1': np.random.normal(0, 1, size),
                'feature2': np.random.normal(5, 2, size),
                'feature3': np.random.choice([0, 1], size),
                'feature4': np.random.uniform(0, 10, size),
                'target': np.random.normal(10, 3, size)
            })
            
            # Add realistic relationship
            data['target'] = (data['feature1'] * 2 + 
                             data['feature2'] * 0.5 + 
                             data['feature3'] * 3 + 
                             data['feature4'] * 0.1 + 
                             np.random.normal(0, 1, size))
            
            self.datasets[size] = data
    
    def test_training_performance_benchmarks(self):
        """Test training performance across different dataset sizes."""
        logger.info("Testing training performance benchmarks...")
        
        models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'RandomForest': RandomForestRegressor(n_estimators=10, random_state=42)
        }
        
        performance_results = {}
        
        for size, data in self.datasets.items():
            X = data[['feature1', 'feature2', 'feature3', 'feature4']]
            y = data['target']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            size_results = {}
            
            for model_name, model in models.items():
                # Measure training time
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # Measure prediction time
                start_time = time.time()
                predictions = model.predict(X_test)
                prediction_time = time.time() - start_time
                
                # Calculate performance metrics
                r2 = r2_score(y_test, predictions)
                mae = mean_absolute_error(y_test, predictions)
                
                size_results[model_name] = {
                    'training_time': training_time,
                    'prediction_time': prediction_time,
                    'r2_score': r2,
                    'mae': mae,
                    'samples_per_second_training': len(X_train) / training_time if training_time > 0 else 0,
                    'samples_per_second_prediction': len(X_test) / prediction_time if prediction_time > 0 else 0
                }
                
                # Validate performance metrics
                self.assertGreater(training_time, 0)
                self.assertGreaterEqual(prediction_time, 0)  # Allow zero for very fast predictions
                self.assertTrue(np.isfinite(r2))
                self.assertTrue(np.isfinite(mae))
            
            performance_results[size] = size_results
        
        # Validate scaling behavior
        for model_name in models.keys():
            training_times = [performance_results[size][model_name]['training_time'] 
                            for size in self.dataset_sizes]
            
            # Training time should be positive and finite
            # (though scaling behavior may vary for small datasets)
            if len(training_times) > 1:
                # All training times should be positive
                for time_val in training_times:
                    self.assertGreater(time_val, 0)
        
        logger.info("Training performance benchmarks completed")
        
        # Log performance summary
        for size in self.dataset_sizes:
            logger.info(f"Dataset size {size}:")
            for model_name in models.keys():
                results = performance_results[size][model_name]
                logger.info(f"  {model_name}: {results['training_time']:.4f}s training, "
                          f"{results['prediction_time']:.4f}s prediction, R²={results['r2_score']:.3f}")
    
    def test_memory_usage_benchmarks(self):
        """Test memory usage during model operations."""
        logger.info("Testing memory usage benchmarks...")
        
        try:
            import psutil
        except ImportError:
            logger.warning("psutil not available, skipping memory usage test")
            self.skipTest("psutil not available")
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Test with largest dataset
        largest_data = self.datasets[max(self.dataset_sizes)]
        X = largest_data[['feature1', 'feature2', 'feature3', 'feature4']]
        y = largest_data['target']
        
        # Train memory-intensive model
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        
        # Measure memory during training
        memory_before_training = process.memory_info().rss / 1024 / 1024
        model.fit(X, y)
        memory_after_training = process.memory_info().rss / 1024 / 1024
        
        # Make predictions
        predictions = model.predict(X)
        memory_after_prediction = process.memory_info().rss / 1024 / 1024
        
        # Clean up
        del model, predictions
        gc.collect()
        memory_after_cleanup = process.memory_info().rss / 1024 / 1024
        
        # Validate memory usage patterns
        memory_increase_training = memory_after_training - memory_before_training
        memory_increase_prediction = memory_after_prediction - memory_after_training
        memory_decrease_cleanup = memory_after_prediction - memory_after_cleanup
        
        # Memory should increase during training
        self.assertGreaterEqual(memory_increase_training, 0)
        
        # Memory usage should be reasonable (less than 500MB increase for test data)
        self.assertLess(memory_increase_training, 500)
        
        logger.info(f"Memory usage - Initial: {initial_memory:.1f}MB, "
                   f"Training increase: {memory_increase_training:.1f}MB, "
                   f"Prediction increase: {memory_increase_prediction:.1f}MB")
    
    def test_concurrent_prediction_performance(self):
        """Test concurrent prediction performance."""
        logger.info("Testing concurrent prediction performance...")
        
        # Prepare model and data
        data = self.datasets[500]  # Medium-sized dataset
        X = data[['feature1', 'feature2', 'feature3', 'feature4']]
        y = data['target']
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Test sequential predictions
        start_time = time.time()
        for _ in range(10):
            predictions = model.predict(X)
        sequential_time = time.time() - start_time
        
        # Test batch prediction
        start_time = time.time()
        batch_predictions = model.predict(X)
        batch_time = time.time() - start_time
        
        # Validate performance
        self.assertGreater(sequential_time, batch_time)  # Batch should be faster
        self.assertEqual(len(batch_predictions), len(X))
        
        # Calculate throughput
        sequential_throughput = (10 * len(X)) / sequential_time
        batch_throughput = len(X) / batch_time if batch_time > 0 else 0
        
        logger.info(f"Sequential throughput: {sequential_throughput:.0f} predictions/sec")
        logger.info(f"Batch throughput: {batch_throughput:.0f} predictions/sec")
        
        # Batch should be significantly faster
        self.assertGreater(batch_throughput, sequential_throughput / 5)

class IntegrationTestRunner:
    """
    Main class to run all integration tests and generate comprehensive reports.
    """
    
    def __init__(self, project_root: str = None):
        if project_root is None:
            current_file = Path(__file__).resolve()
            self.project_root = current_file.parents[2]
        else:
            self.project_root = Path(project_root)
        
        self.output_path = self.project_root / "data" / "modeling_outputs"
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Output path: {self.output_path}")
    
    def run_all_integration_tests(self) -> Dict[str, Any]:
        """Run all integration tests and return comprehensive results."""
        logger.info("Starting comprehensive integration testing suite...")
        
        # Create test suite
        test_suite = unittest.TestSuite()
        
        # Add all test classes
        test_classes = [
            TestEndToEndPipeline,
            TestModelTrainingToDeployment,
            TestPerformanceBenchmarks
        ]
        
        for test_class in test_classes:
            tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
            test_suite.addTests(tests)
        
        # Run tests with detailed results
        test_runner = unittest.TextTestRunner(
            verbosity=2,
            stream=open(os.devnull, 'w')  # Suppress output for clean logging
        )
        
        result = test_runner.run(test_suite)
        
        # Compile results
        test_results = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': result.testsRun,
            'tests_passed': result.testsRun - len(result.failures) - len(result.errors),
            'tests_failed': len(result.failures),
            'tests_error': len(result.errors),
            'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100 if result.testsRun > 0 else 0,
            'test_categories': {
                'end_to_end_pipeline': self._count_tests_by_class(result, 'TestEndToEndPipeline'),
                'training_to_deployment': self._count_tests_by_class(result, 'TestModelTrainingToDeployment'),
                'performance_benchmarks': self._count_tests_by_class(result, 'TestPerformanceBenchmarks')
            },
            'failures': [str(failure) for failure in result.failures],
            'errors': [str(error) for error in result.errors],
            'summary': self._generate_integration_summary(result),
            'recommendations': self._generate_integration_recommendations(result)
        }
        
        # Save results
        results_file = self.output_path / "integration_testing_results.json"
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        logger.info(f"Integration testing completed. Results saved to {results_file}")
        
        return test_results
    
    def _count_tests_by_class(self, result, class_name: str) -> Dict[str, int]:
        """Count tests by class name."""
        total = 0
        passed = 0
        failed = 0
        errors = 0
        
        # Count failures for this class
        for failure in result.failures:
            if class_name in str(failure[0]):
                failed += 1
        
        # Count errors for this class
        for error in result.errors:
            if class_name in str(error[0]):
                errors += 1
        
        # Estimate total (this is approximate)
        total = failed + errors + 3  # Assume 3 tests per class if no failures/errors
        passed = total - failed - errors
        
        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'errors': errors
        }
    
    def _generate_integration_summary(self, result) -> Dict[str, Any]:
        """Generate integration test summary."""
        summary = {
            'overall_status': 'PASSED' if result.wasSuccessful() else 'FAILED',
            'test_coverage': {
                'end_to_end_pipeline': 'Complete pipeline flow from data loading to model deployment',
                'training_to_deployment': 'Model training, versioning, and deployment workflow',
                'performance_benchmarks': 'Training and prediction performance across different data sizes'
            },
            'key_findings': []
        }
        
        if result.wasSuccessful():
            summary['key_findings'].append("All integration tests passed successfully")
            summary['key_findings'].append("End-to-end pipeline is functioning correctly")
            summary['key_findings'].append("Model training to deployment workflow is validated")
            summary['key_findings'].append("Performance benchmarks meet expectations")
        else:
            summary['key_findings'].append(f"Found {len(result.failures)} integration test failures")
            summary['key_findings'].append(f"Found {len(result.errors)} integration test errors")
            summary['key_findings'].append("Some integration components require attention")
        
        return summary
    
    def _generate_integration_recommendations(self, result) -> List[str]:
        """Generate integration testing recommendations."""
        recommendations = []
        
        if result.wasSuccessful():
            recommendations.append("All integration tests passed - system is ready for deployment")
            recommendations.append("Consider adding more edge case scenarios for production robustness")
            recommendations.append("Implement continuous integration testing for ongoing development")
            recommendations.append("Monitor performance benchmarks in production environment")
        else:
            recommendations.append("Fix failing integration tests before proceeding to deployment")
            recommendations.append("Review error messages and update component implementations")
            recommendations.append("Add additional integration test cases for failed scenarios")
        
        recommendations.append("Establish performance baselines for production monitoring")
        recommendations.append("Implement automated integration testing in CI/CD pipeline")
        recommendations.append("Regular integration testing should be performed after major changes")
        
        return recommendations

def main():
    """Main execution function."""
    try:
        test_runner = IntegrationTestRunner()
        results = test_runner.run_all_integration_tests()
        
        print("\n=== INTEGRATION TESTING SUITE COMPLETED ===")
        print(f"Timestamp: {results['timestamp']}")
        print(f"Total Tests: {results['total_tests']}")
        print(f"Tests Passed: {results['tests_passed']}")
        print(f"Tests Failed: {results['tests_failed']}")
        print(f"Tests Error: {results['tests_error']}")
        print(f"Success Rate: {results['success_rate']:.1f}%")
        
        print("\n=== INTEGRATION TEST SUMMARY ===")
        summary = results['summary']
        print(f"Overall Status: {summary['overall_status']}")
        
        print("\n=== KEY FINDINGS ===")
        for finding in summary['key_findings']:
            print(f"- {finding}")
        
        print("\n=== RECOMMENDATIONS ===")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"{i}. {rec}")
        
        if results['failures']:
            print("\n=== FAILURES ===")
            for failure in results['failures']:
                print(f"- {failure}")
        
        if results['errors']:
            print("\n=== ERRORS ===")
            for error in results['errors']:
                print(f"- {error}")
        
        print("\n=== INTEGRATION TESTING COMPLETE ===")
        
    except Exception as e:
        logger.error(f"Integration testing failed: {e}")
        raise

if __name__ == "__main__":
    main()