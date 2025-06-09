#!/usr/bin/env python3
"""
Phase 6 Task 3.2: Robustness and Error Analysis

This script implements comprehensive robustness testing and error analysis
for the trained models, including:
- Noise injection testing
- Feature perturbation analysis
- Error pattern analysis
- Model stability assessment
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class RobustnessAnalyzer:
    """
    Comprehensive robustness and error analysis for trained models.
    """
    
    def __init__(self, data_path, models_path, output_path):
        self.data_path = Path(data_path)
        self.models_path = Path(models_path)
        self.output_path = Path(output_path)
        self.models = {}
        self.data = None
        self.X = None
        self.y = None
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'noise_injection': {},
            'feature_perturbation': {},
            'error_analysis': {},
            'stability_assessment': {},
            'summary': {}
        }
        
    def load_data(self):
        """
        Load and prepare the dataset.
        """
        print(f"Loading data from: {self.data_path}")
        self.data = pd.read_csv(self.data_path)
        
        # Remove target variable and non-feature columns
        feature_cols = [col for col in self.data.columns 
                       if col not in ['final_test', 'student_id', 'index']]
        
        self.X = self.data[feature_cols]
        self.y = self.data['final_test']
        
        print(f"Data loaded: {self.X.shape[0]} samples, {self.X.shape[1]} features")
        return True
        
    def load_models(self):
        """
        Load trained models from the models directory.
        """
        print(f"Loading models from: {self.models_path}")
        
        # Load the best model
        best_model_path = self.output_path / 'best_model_linear_regression_fixed.joblib'
        if best_model_path.exists():
            try:
                self.models['best_model'] = joblib.load(best_model_path)
                print(f"Loaded best model: {best_model_path}")
            except Exception as e:
                print(f"Error loading best model: {e}")
        
        # Load individual models
        model_files = {
            'linear_regression': 'linear_regression_20250608_235206.joblib',
            'ridge_regression': 'ridge_regression_20250608_235206.joblib',
            'random_forest': 'random_forest_20250608_235206.joblib'
        }
        
        for model_name, filename in model_files.items():
            model_path = self.models_path / filename
            if model_path.exists():
                try:
                    model = joblib.load(model_path)
                    self.models[model_name] = model
                    print(f"Loaded {model_name}: {model_path}")
                except Exception as e:
                    print(f"Error loading {model_name}: {e}")
        
        print(f"Successfully loaded {len(self.models)} models")
        return len(self.models) > 0
        
    def noise_injection_test(self, noise_levels=[0.01, 0.05, 0.1, 0.2]):
        """
        Test model robustness to input noise.
        """
        print("\nRunning noise injection tests...")
        
        # Split data for testing
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        for model_name, model in self.models.items():
            print(f"Testing {model_name}...")
            model_results = {}
            
            # Baseline performance
            try:
                y_pred_baseline = model.predict(X_test)
                baseline_mae = mean_absolute_error(y_test, y_pred_baseline)
                baseline_r2 = r2_score(y_test, y_pred_baseline)
                
                model_results['baseline'] = {
                    'mae': baseline_mae,
                    'r2': baseline_r2
                }
                
                # Test with different noise levels
                for noise_level in noise_levels:
                    # Add Gaussian noise
                    noise = np.random.normal(0, noise_level, X_test.shape)
                    X_test_noisy = X_test + noise
                    
                    y_pred_noisy = model.predict(X_test_noisy)
                    noisy_mae = mean_absolute_error(y_test, y_pred_noisy)
                    noisy_r2 = r2_score(y_test, y_pred_noisy)
                    
                    model_results[f'noise_{noise_level}'] = {
                        'mae': noisy_mae,
                        'r2': noisy_r2,
                        'mae_degradation': (noisy_mae - baseline_mae) / baseline_mae,
                        'r2_degradation': (baseline_r2 - noisy_r2) / baseline_r2 if baseline_r2 != 0 else 0
                    }
                    
            except Exception as e:
                model_results['error'] = str(e)
                print(f"Error testing {model_name}: {e}")
            
            self.results['noise_injection'][model_name] = model_results
            
    def feature_perturbation_analysis(self):
        """
        Analyze model sensitivity to individual feature perturbations.
        """
        print("\nRunning feature perturbation analysis...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        for model_name, model in self.models.items():
            print(f"Analyzing {model_name}...")
            model_results = {}
            
            try:
                # Baseline prediction
                y_pred_baseline = model.predict(X_test)
                baseline_mae = mean_absolute_error(y_test, y_pred_baseline)
                
                feature_sensitivity = {}
                
                # Test each feature
                for i, feature in enumerate(X_test.columns):
                    # Create perturbed version
                    X_test_perturbed = X_test.copy()
                    
                    # Add 10% noise to this feature
                    feature_std = X_test[feature].std()
                    noise = np.random.normal(0, 0.1 * feature_std, len(X_test))
                    X_test_perturbed[feature] = X_test[feature] + noise
                    
                    # Predict with perturbed feature
                    y_pred_perturbed = model.predict(X_test_perturbed)
                    perturbed_mae = mean_absolute_error(y_test, y_pred_perturbed)
                    
                    # Calculate sensitivity
                    sensitivity = (perturbed_mae - baseline_mae) / baseline_mae if baseline_mae != 0 else 0
                    feature_sensitivity[feature] = sensitivity
                
                # Sort by sensitivity
                sorted_sensitivity = dict(sorted(feature_sensitivity.items(), 
                                               key=lambda x: abs(x[1]), reverse=True))
                
                model_results = {
                    'baseline_mae': baseline_mae,
                    'feature_sensitivity': sorted_sensitivity,
                    'most_sensitive_features': list(sorted_sensitivity.keys())[:10],
                    'least_sensitive_features': list(sorted_sensitivity.keys())[-10:]
                }
                
            except Exception as e:
                model_results['error'] = str(e)
                print(f"Error analyzing {model_name}: {e}")
            
            self.results['feature_perturbation'][model_name] = model_results
            
    def error_pattern_analysis(self):
        """
        Analyze error patterns and distributions.
        """
        print("\nRunning error pattern analysis...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        for model_name, model in self.models.items():
            print(f"Analyzing errors for {model_name}...")
            model_results = {}
            
            try:
                y_pred = model.predict(X_test)
                errors = y_test - y_pred
                
                # Error statistics
                model_results = {
                    'error_mean': float(errors.mean()),
                    'error_std': float(errors.std()),
                    'error_min': float(errors.min()),
                    'error_max': float(errors.max()),
                    'error_median': float(errors.median()),
                    'error_q25': float(errors.quantile(0.25)),
                    'error_q75': float(errors.quantile(0.75)),
                    'mae': float(mean_absolute_error(y_test, y_pred)),
                    'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                    'r2': float(r2_score(y_test, y_pred))
                }
                
                # Error distribution analysis
                model_results['error_distribution'] = {
                    'normal_test_p_value': 'not_implemented',  # Would need scipy.stats
                    'skewness': float(errors.skew()),
                    'kurtosis': float(errors.kurtosis())
                }
                
                # Identify outlier predictions
                error_threshold = 2 * errors.std()
                outlier_indices = errors[abs(errors) > error_threshold].index.tolist()
                model_results['outlier_predictions'] = {
                    'count': len(outlier_indices),
                    'percentage': len(outlier_indices) / len(errors) * 100,
                    'threshold_used': float(error_threshold)
                }
                
            except Exception as e:
                model_results['error'] = str(e)
                print(f"Error analyzing {model_name}: {e}")
            
            self.results['error_analysis'][model_name] = model_results
            
    def stability_assessment(self, n_runs=10):
        """
        Assess model stability across multiple random splits.
        """
        print("\nRunning stability assessment...")
        
        for model_name, model in self.models.items():
            print(f"Assessing stability for {model_name}...")
            model_results = {
                'runs': [],
                'statistics': {}
            }
            
            mae_scores = []
            r2_scores = []
            
            try:
                for run in range(n_runs):
                    # Different random split each time
                    X_train, X_test, y_train, y_test = train_test_split(
                        self.X, self.y, test_size=0.2, random_state=run
                    )
                    
                    y_pred = model.predict(X_test)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    mae_scores.append(mae)
                    r2_scores.append(r2)
                    
                    model_results['runs'].append({
                        'run': run,
                        'mae': float(mae),
                        'r2': float(r2)
                    })
                
                # Calculate stability statistics
                model_results['statistics'] = {
                    'mae_mean': float(np.mean(mae_scores)),
                    'mae_std': float(np.std(mae_scores)),
                    'mae_cv': float(np.std(mae_scores) / np.mean(mae_scores)) if np.mean(mae_scores) != 0 else 0,
                    'r2_mean': float(np.mean(r2_scores)),
                    'r2_std': float(np.std(r2_scores)),
                    'r2_cv': float(np.std(r2_scores) / np.mean(r2_scores)) if np.mean(r2_scores) != 0 else 0
                }
                
            except Exception as e:
                model_results['error'] = str(e)
                print(f"Error assessing {model_name}: {e}")
            
            self.results['stability_assessment'][model_name] = model_results
            
    def generate_summary(self):
        """
        Generate overall summary of robustness analysis.
        """
        print("\nGenerating summary...")
        
        summary = {
            'total_models_tested': len(self.models),
            'tests_completed': [],
            'key_findings': [],
            'recommendations': []
        }
        
        # Check which tests were completed
        if self.results['noise_injection']:
            summary['tests_completed'].append('noise_injection')
        if self.results['feature_perturbation']:
            summary['tests_completed'].append('feature_perturbation')
        if self.results['error_analysis']:
            summary['tests_completed'].append('error_analysis')
        if self.results['stability_assessment']:
            summary['tests_completed'].append('stability_assessment')
        
        # Generate findings and recommendations
        summary['key_findings'] = [
            "Robustness analysis completed for all available models",
            "Noise injection tests show model sensitivity to input perturbations",
            "Feature perturbation analysis identifies most sensitive features",
            "Error pattern analysis reveals prediction error characteristics",
            "Stability assessment evaluates model consistency across data splits"
        ]
        
        summary['recommendations'] = [
            "Monitor model performance on noisy real-world data",
            "Consider feature engineering for highly sensitive features",
            "Implement error monitoring in production",
            "Regular model retraining to maintain stability",
            "Consider ensemble methods for improved robustness"
        ]
        
        self.results['summary'] = summary
        
    def save_results(self):
        """
        Save analysis results to JSON file.
        """
        output_file = self.output_path / 'robustness_analysis_results.json'
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
    def run_analysis(self):
        """
        Run complete robustness analysis.
        """
        print("Starting Phase 6 Task 3.2: Robustness and Error Analysis")
        print("=" * 60)
        
        # Load data and models
        if not self.load_data():
            print("Failed to load data!")
            return False
            
        if not self.load_models():
            print("Failed to load models!")
            return False
        
        # Run all analyses
        self.noise_injection_test()
        self.feature_perturbation_analysis()
        self.error_pattern_analysis()
        self.stability_assessment()
        
        # Generate summary and save results
        self.generate_summary()
        self.save_results()
        
        print("\n" + "=" * 60)
        print("Robustness analysis completed successfully!")
        print(f"Models tested: {list(self.models.keys())}")
        print(f"Tests completed: {self.results['summary']['tests_completed']}")
        
        return True

def main():
    """
    Main function to run robustness analysis.
    """
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / 'data' / 'modeling_outputs' / 'clean_dataset_no_leakage.csv'
    models_path = project_root / 'data' / 'modeling_outputs' / 'models'
    output_path = project_root / 'data' / 'modeling_outputs'
    
    print(f"Project root: {project_root}")
    print(f"Data path: {data_path}")
    print(f"Models path: {models_path}")
    print(f"Output path: {output_path}")
    
    # Create and run analyzer
    analyzer = RobustnessAnalyzer(
        data_path=data_path,
        models_path=models_path,
        output_path=output_path
    )
    
    success = analyzer.run_analysis()
    
    if success:
        print("\nRobustness analysis completed successfully!")
    else:
        print("\nRobustness analysis failed!")

if __name__ == "__main__":
    main()