#!/usr/bin/env python3
"""
Phase 3.2.2: Final Test Missing Values Handling

Implements Priority 2 of Phase 3 data preprocessing:
- Handles 495 missing final_test values (3.11%)
- Evaluates exclusion vs. sophisticated imputation strategies
- Implements advanced imputation methods (KNN, iterative)
- Ensures proper handling in train/validation/test splits
- Documents decision rationale and impact assessment

Follows TASKS.md Phase 3.2.2 specifications exactly.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import json
from sklearn.experimental import enable_iterative_imputer  # Enable experimental feature
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinalTestMissingHandler:
    """
    Handles missing final_test values for Phase 3.2.2.
    
    Implements the requirements for task 3.2.2:
    - Identify all 495 missing final_test values (3.11%)
    - Evaluate exclusion vs. advanced imputation strategies
    - Implement KNN and iterative imputation methods
    - Ensure proper train/validation/test split handling
    - Document decision rationale and impact assessment
    """
    
    def __init__(self, input_path: str = "data/processed/imputed.csv"):
        """
        Initialize the FinalTestMissingHandler.
        
        Args:
            input_path: Path to the imputed CSV file from Phase 3.2.1
        """
        self.input_path = input_path
        self.data = None
        self.processed_data = None
        self.imputation_results = {}
        self.split_results = {}
        self.audit_trail = []
        
    def load_data(self) -> pd.DataFrame:
        """
        Load imputed data from Phase 3.2.1.
        
        Returns:
            DataFrame containing the imputed data
        """
        try:
            self.data = pd.read_csv(self.input_path)
            logger.info(f"Loaded {len(self.data)} records from {self.input_path}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def analyze_missing_final_test(self) -> Dict[str, Any]:
        """
        Analyze missing final_test patterns and impact.
        
        Returns:
            Dictionary with missing data analysis
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        missing_mask = self.data['final_test'].isnull()
        missing_count = missing_mask.sum()
        total_count = len(self.data)
        missing_percentage = (missing_count / total_count) * 100
        
        analysis = {
            'total_records': total_count,
            'missing_count': int(missing_count),
            'missing_percentage': round(missing_percentage, 2),
            'non_missing_count': int(total_count - missing_count),
            'missing_by_groups': {},
            'impact_assessment': {}
        }
        
        logger.info(f"Missing final_test analysis: {missing_count} missing ({missing_percentage:.2f}%)")
        
        # Analyze missing patterns by categorical groups
        categorical_cols = ['gender', 'CCA', 'learning_style', 'tuition']
        
        for col in categorical_cols:
            if col in self.data.columns:
                group_analysis = self.data.groupby(col)['final_test'].agg([
                    'count', 
                    lambda x: x.isnull().sum(),
                    lambda x: (x.isnull().sum() / len(x)) * 100
                ]).round(2)
                group_analysis.columns = ['total', 'missing', 'missing_pct']
                analysis['missing_by_groups'][col] = group_analysis.to_dict('index')
                
                logger.info(f"Missing final_test by {col}:")
                for group, stats in group_analysis.to_dict('index').items():
                    logger.info(f"  {group}: {stats['missing']}/{stats['total']} ({stats['missing_pct']:.1f}%)")
        
        # Impact assessment for modeling
        non_missing_data = self.data[~missing_mask]
        if len(non_missing_data) > 0:
            analysis['impact_assessment'] = {
                'available_for_training': len(non_missing_data),
                'percentage_available': (len(non_missing_data) / total_count) * 100,
                'final_test_stats_available': {
                    'mean': non_missing_data['final_test'].mean(),
                    'median': non_missing_data['final_test'].median(),
                    'std': non_missing_data['final_test'].std(),
                    'min': non_missing_data['final_test'].min(),
                    'max': non_missing_data['final_test'].max()
                }
            }
        
        return analysis
    
    def evaluate_exclusion_strategy(self) -> Dict[str, Any]:
        """
        Evaluate the impact of excluding missing final_test records.
        
        Returns:
            Dictionary with exclusion strategy evaluation
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        missing_mask = self.data['final_test'].isnull()
        training_data = self.data[~missing_mask].copy()
        excluded_data = self.data[missing_mask].copy()
        
        evaluation = {
            'strategy': 'exclusion',
            'training_set_size': len(training_data),
            'excluded_set_size': len(excluded_data),
            'data_loss_percentage': (len(excluded_data) / len(self.data)) * 100,
            'training_data_characteristics': {},
            'excluded_data_characteristics': {},
            'bias_assessment': {}
        }
        
        # Analyze training data characteristics
        numeric_cols = ['age', 'attendance_rate']
        for col in numeric_cols:
            if col in training_data.columns:
                evaluation['training_data_characteristics'][col] = {
                    'mean': training_data[col].mean(),
                    'median': training_data[col].median(),
                    'std': training_data[col].std()
                }
        
        # Analyze excluded data characteristics
        for col in numeric_cols:
            if col in excluded_data.columns:
                evaluation['excluded_data_characteristics'][col] = {
                    'mean': excluded_data[col].mean(),
                    'median': excluded_data[col].median(),
                    'std': excluded_data[col].std()
                }
        
        # Bias assessment - compare distributions
        categorical_cols = ['gender', 'CCA', 'learning_style', 'tuition']
        for col in categorical_cols:
            if col in self.data.columns:
                training_dist = training_data[col].value_counts(normalize=True).to_dict()
                excluded_dist = excluded_data[col].value_counts(normalize=True).to_dict()
                
                # Calculate distribution differences
                all_categories = set(training_dist.keys()) | set(excluded_dist.keys())
                max_diff = 0
                for cat in all_categories:
                    train_pct = training_dist.get(cat, 0)
                    excl_pct = excluded_dist.get(cat, 0)
                    diff = abs(train_pct - excl_pct)
                    max_diff = max(max_diff, diff)
                
                evaluation['bias_assessment'][col] = {
                    'max_distribution_difference': max_diff,
                    'training_distribution': training_dist,
                    'excluded_distribution': excluded_dist
                }
        
        logger.info(f"Exclusion strategy: {len(training_data)} training, {len(excluded_data)} excluded")
        logger.info(f"Data loss: {evaluation['data_loss_percentage']:.2f}%")
        
        self.audit_trail.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'evaluated_exclusion_strategy',
            'training_size': len(training_data),
            'excluded_size': len(excluded_data),
            'data_loss_pct': evaluation['data_loss_percentage'],
            'details': 'Evaluated impact of excluding missing final_test records'
        })
        
        return evaluation
    
    def prepare_features_for_imputation(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for advanced imputation methods.
        
        Returns:
            Tuple of (prepared_data, feature_columns)
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Start with numeric features
        numeric_features = ['age', 'attendance_rate']
        
        # Prepare data for imputation
        imputation_data = self.data.copy()
        
        # Encode categorical variables
        categorical_cols = ['gender', 'CCA', 'learning_style', 'tuition']
        label_encoders = {}
        
        for col in categorical_cols:
            if col in imputation_data.columns:
                le = LabelEncoder()
                # Handle missing values in categorical columns
                non_null_mask = imputation_data[col].notna()
                if non_null_mask.sum() > 0:
                    # Fit encoder on non-null values
                    le.fit(imputation_data.loc[non_null_mask, col])
                    
                    # Transform non-null values
                    imputation_data.loc[non_null_mask, f'{col}_encoded'] = le.transform(
                        imputation_data.loc[non_null_mask, col]
                    )
                    
                    # For null values, use a special code (e.g., -1)
                    imputation_data.loc[~non_null_mask, f'{col}_encoded'] = -1
                    
                    label_encoders[col] = le
                    numeric_features.append(f'{col}_encoded')
        
        # Store encoders for later use
        self.label_encoders = label_encoders
        
        logger.info(f"Prepared {len(numeric_features)} features for imputation: {numeric_features}")
        return imputation_data, numeric_features
    
    def knn_imputation(self, n_neighbors: int = 5) -> pd.Series:
        """
        Implement KNN imputation for missing final_test values.
        
        Args:
            n_neighbors: Number of neighbors for KNN imputation
            
        Returns:
            Series with KNN-imputed final_test values
        """
        logger.info(f"Performing KNN imputation with {n_neighbors} neighbors")
        
        # Prepare features
        imputation_data, feature_cols = self.prepare_features_for_imputation()
        
        # Create feature matrix including final_test
        all_features = feature_cols + ['final_test']
        feature_matrix = imputation_data[all_features].copy()
        
        # Apply KNN imputation
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_matrix = imputer.fit_transform(feature_matrix)
        
        # Extract imputed final_test values
        final_test_idx = all_features.index('final_test')
        imputed_final_test = pd.Series(
            imputed_matrix[:, final_test_idx], 
            index=self.data.index
        )
        
        # Calculate imputation statistics
        missing_mask = self.data['final_test'].isnull()
        original_values = self.data.loc[~missing_mask, 'final_test']
        imputed_values = imputed_final_test.loc[missing_mask]
        
        imputation_stats = {
            'method': 'KNN',
            'n_neighbors': n_neighbors,
            'total_missing': int(missing_mask.sum()),
            'imputed_count': int(missing_mask.sum()),
            'features_used': feature_cols,
            'imputed_stats': {
                'mean': imputed_values.mean(),
                'median': imputed_values.median(),
                'std': imputed_values.std(),
                'min': imputed_values.min(),
                'max': imputed_values.max()
            },
            'original_stats': {
                'mean': original_values.mean(),
                'median': original_values.median(),
                'std': original_values.std(),
                'min': original_values.min(),
                'max': original_values.max()
            }
        }
        
        logger.info(f"KNN imputation completed: {missing_mask.sum()} values imputed")
        logger.info(f"Imputed values - mean: {imputed_values.mean():.2f}, std: {imputed_values.std():.2f}")
        
        # Store results
        self.imputation_results['knn'] = {
            'imputed_series': imputed_final_test,
            'statistics': imputation_stats,
            'imputer': imputer
        }
        
        self.audit_trail.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'knn_imputation',
            'n_neighbors': n_neighbors,
            'features_used': feature_cols,
            'statistics': imputation_stats,
            'details': f'KNN imputation with {n_neighbors} neighbors'
        })
        
        return imputed_final_test
    
    def iterative_imputation(self, max_iter: int = 10, random_state: int = 42) -> pd.Series:
        """
        Implement iterative imputation for missing final_test values.
        
        Args:
            max_iter: Maximum number of imputation rounds
            random_state: Random state for reproducibility
            
        Returns:
            Series with iteratively-imputed final_test values
        """
        logger.info(f"Performing iterative imputation with max_iter={max_iter}")
        
        # Prepare features
        imputation_data, feature_cols = self.prepare_features_for_imputation()
        
        # Create feature matrix including final_test
        all_features = feature_cols + ['final_test']
        feature_matrix = imputation_data[all_features].copy()
        
        # Apply iterative imputation
        imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
        imputed_matrix = imputer.fit_transform(feature_matrix)
        
        # Extract imputed final_test values
        final_test_idx = all_features.index('final_test')
        imputed_final_test = pd.Series(
            imputed_matrix[:, final_test_idx], 
            index=self.data.index
        )
        
        # Calculate imputation statistics
        missing_mask = self.data['final_test'].isnull()
        original_values = self.data.loc[~missing_mask, 'final_test']
        imputed_values = imputed_final_test.loc[missing_mask]
        
        imputation_stats = {
            'method': 'Iterative',
            'max_iter': max_iter,
            'random_state': random_state,
            'total_missing': int(missing_mask.sum()),
            'imputed_count': int(missing_mask.sum()),
            'features_used': feature_cols,
            'imputed_stats': {
                'mean': imputed_values.mean(),
                'median': imputed_values.median(),
                'std': imputed_values.std(),
                'min': imputed_values.min(),
                'max': imputed_values.max()
            },
            'original_stats': {
                'mean': original_values.mean(),
                'median': original_values.median(),
                'std': original_values.std(),
                'min': original_values.min(),
                'max': original_values.max()
            }
        }
        
        logger.info(f"Iterative imputation completed: {missing_mask.sum()} values imputed")
        logger.info(f"Imputed values - mean: {imputed_values.mean():.2f}, std: {imputed_values.std():.2f}")
        
        # Store results
        self.imputation_results['iterative'] = {
            'imputed_series': imputed_final_test,
            'statistics': imputation_stats,
            'imputer': imputer
        }
        
        self.audit_trail.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'iterative_imputation',
            'max_iter': max_iter,
            'features_used': feature_cols,
            'statistics': imputation_stats,
            'details': f'Iterative imputation with max_iter={max_iter}'
        })
        
        return imputed_final_test
    
    def validate_imputation_quality(self) -> Dict[str, Any]:
        """
        Validate imputation quality using cross-validation on known values.
        
        Returns:
            Dictionary with validation results
        """
        if not self.imputation_results:
            raise ValueError("No imputation results available. Run imputation methods first.")
        
        logger.info("Validating imputation quality using cross-validation")
        
        # Use records with known final_test values for validation
        complete_mask = self.data['final_test'].notna()
        complete_data = self.data[complete_mask].copy()
        
        if len(complete_data) < 100:
            logger.warning("Insufficient complete records for robust validation")
            return {'validation_possible': False, 'reason': 'Insufficient complete records'}
        
        # Prepare features for validation
        imputation_data, feature_cols = self.prepare_features_for_imputation()
        complete_imputation_data = imputation_data[complete_mask].copy()
        
        validation_results = {
            'validation_possible': True,
            'validation_set_size': len(complete_data),
            'method_performance': {}
        }
        
        # Test each imputation method
        for method_name in self.imputation_results.keys():
            logger.info(f"Validating {method_name} imputation")
            
            # Create artificial missing values for validation
            np.random.seed(42)
            validation_indices = np.random.choice(
                complete_data.index, 
                size=min(50, len(complete_data) // 4), 
                replace=False
            )
            
            # Create validation dataset with artificial missing values
            validation_data = complete_imputation_data.copy()
            true_values = validation_data.loc[validation_indices, 'final_test'].copy()
            validation_data.loc[validation_indices, 'final_test'] = np.nan
            
            # Apply imputation method
            all_features = feature_cols + ['final_test']
            feature_matrix = validation_data[all_features].copy()
            
            if method_name == 'knn':
                imputer = self.imputation_results['knn']['imputer']
                imputed_matrix = imputer.transform(feature_matrix)
            elif method_name == 'iterative':
                imputer = self.imputation_results['iterative']['imputer']
                imputed_matrix = imputer.transform(feature_matrix)
            
            # Extract predicted values
            final_test_idx = all_features.index('final_test')
            # Map validation_indices to positions in the validation_data
            validation_positions = [list(validation_data.index).index(idx) for idx in validation_indices]
            predicted_values = imputed_matrix[validation_positions, final_test_idx]
            
            # Calculate validation metrics
            mse = mean_squared_error(true_values, predicted_values)
            rmse = np.sqrt(mse)
            r2 = r2_score(true_values, predicted_values)
            mae = np.mean(np.abs(true_values - predicted_values))
            
            validation_results['method_performance'][method_name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'validation_samples': len(validation_indices)
            }
            
            logger.info(f"{method_name} validation: RMSE={rmse:.2f}, R²={r2:.3f}, MAE={mae:.2f}")
        
        return validation_results
    
    def compare_strategies(self) -> Dict[str, Any]:
        """
        Compare exclusion vs. imputation strategies.
        
        Returns:
            Dictionary with strategy comparison
        """
        comparison = {
            'strategies_evaluated': ['exclusion', 'knn_imputation', 'iterative_imputation'],
            'exclusion_analysis': self.evaluate_exclusion_strategy(),
            'imputation_methods': {},
            'recommendations': []
        }
        
        # Add imputation method results
        for method_name, method_results in self.imputation_results.items():
            comparison['imputation_methods'][method_name] = method_results['statistics']
        
        # Generate recommendations based on analysis
        exclusion_data_loss = comparison['exclusion_analysis']['data_loss_percentage']
        
        if exclusion_data_loss < 5:  # Less than 5% data loss
            comparison['recommendations'].append(
                f"Low data loss ({exclusion_data_loss:.1f}%) - exclusion strategy is viable"
            )
        else:
            comparison['recommendations'].append(
                f"High data loss ({exclusion_data_loss:.1f}%) - consider imputation strategies"
            )
        
        # Check for bias in exclusion
        max_bias = 0
        for col, bias_info in comparison['exclusion_analysis']['bias_assessment'].items():
            max_bias = max(max_bias, bias_info['max_distribution_difference'])
        
        if max_bias > 0.1:  # More than 10% distribution difference
            comparison['recommendations'].append(
                f"Potential bias detected in exclusion (max diff: {max_bias:.1%}) - imputation may be preferable"
            )
        
        # Validate imputation quality if available
        if self.imputation_results:
            validation = self.validate_imputation_quality()
            if validation.get('validation_possible', False):
                comparison['validation_results'] = validation
                
                # Find best performing method
                best_method = None
                best_r2 = -1
                for method, perf in validation['method_performance'].items():
                    if perf['r2'] > best_r2:
                        best_r2 = perf['r2']
                        best_method = method
                
                if best_r2 > 0.3:  # Reasonable predictive power
                    comparison['recommendations'].append(
                        f"Best imputation method: {best_method} (R² = {best_r2:.3f})"
                    )
                else:
                    comparison['recommendations'].append(
                        f"Low imputation quality (best R² = {best_r2:.3f}) - consider exclusion"
                    )
        
        return comparison
    
    def create_train_validation_test_splits(self, strategy: str = 'exclusion', 
                                          test_size: float = 0.2, 
                                          val_size: float = 0.2,
                                          random_state: int = 42) -> Dict[str, pd.DataFrame]:
        """
        Create proper train/validation/test splits based on chosen strategy.
        
        Args:
            strategy: 'exclusion', 'knn', or 'iterative'
            test_size: Proportion for test set
            val_size: Proportion for validation set (from remaining after test)
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary with train, validation, test, and prediction datasets
        """
        logger.info(f"Creating train/validation/test splits using {strategy} strategy")
        
        if strategy == 'exclusion':
            # Use only records with non-missing final_test for training
            training_data = self.data[self.data['final_test'].notna()].copy()
            prediction_data = self.data[self.data['final_test'].isnull()].copy()
        else:
            # Use imputed data
            if strategy not in self.imputation_results:
                raise ValueError(f"Imputation method '{strategy}' not available")
            
            training_data = self.data.copy()
            training_data['final_test'] = self.imputation_results[strategy]['imputed_series']
            prediction_data = pd.DataFrame()  # No separate prediction set when using imputation
        
        # Split training data into train/val/test
        if len(training_data) == 0:
            raise ValueError("No training data available")
        
        # First split: separate test set
        train_val, test = train_test_split(
            training_data, 
            test_size=test_size, 
            random_state=random_state,
            stratify=training_data['gender'] if 'gender' in training_data.columns else None
        )
        
        # Second split: separate validation from training
        train, validation = train_test_split(
            train_val, 
            test_size=val_size, 
            random_state=random_state,
            stratify=train_val['gender'] if 'gender' in train_val.columns else None
        )
        
        splits = {
            'train': train,
            'validation': validation,
            'test': test
        }
        
        if len(prediction_data) > 0:
            splits['prediction'] = prediction_data
        
        # Log split statistics
        logger.info(f"Data splits created:")
        logger.info(f"  Training: {len(train)} records")
        logger.info(f"  Validation: {len(validation)} records")
        logger.info(f"  Test: {len(test)} records")
        if len(prediction_data) > 0:
            logger.info(f"  Prediction: {len(prediction_data)} records")
        
        self.split_results[strategy] = {
            'strategy': strategy,
            'splits': splits,
            'split_sizes': {name: len(df) for name, df in splits.items()},
            'total_training_data': len(training_data)
        }
        
        self.audit_trail.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'created_data_splits',
            'strategy': strategy,
            'split_sizes': {name: len(df) for name, df in splits.items()},
            'details': f'Created train/val/test splits using {strategy} strategy'
        })
        
        return splits
    
    def make_final_recommendation(self) -> Dict[str, Any]:
        """
        Make final recommendation based on all analyses.
        
        Returns:
            Dictionary with final recommendation and rationale
        """
        if not hasattr(self, 'comparison_results'):
            self.comparison_results = self.compare_strategies()
        
        recommendation = {
            'recommended_strategy': None,
            'rationale': [],
            'implementation_details': {},
            'expected_impact': {}
        }
        
        # Decision logic based on analysis
        exclusion_data_loss = self.comparison_results['exclusion_analysis']['data_loss_percentage']
        
        # Check validation results if available
        best_imputation_r2 = 0
        best_imputation_method = None
        
        if 'validation_results' in self.comparison_results:
            for method, perf in self.comparison_results['validation_results']['method_performance'].items():
                if perf['r2'] > best_imputation_r2:
                    best_imputation_r2 = perf['r2']
                    best_imputation_method = method
        
        # Decision criteria
        if exclusion_data_loss <= 3.5 and best_imputation_r2 < 0.3:
            # Low data loss and poor imputation quality -> recommend exclusion
            recommendation['recommended_strategy'] = 'exclusion'
            recommendation['rationale'].append(f"Low data loss ({exclusion_data_loss:.1f}%)")
            recommendation['rationale'].append(f"Poor imputation quality (best R² = {best_imputation_r2:.3f})")
            
        elif best_imputation_r2 > 0.5:
            # High imputation quality -> recommend best imputation method
            recommendation['recommended_strategy'] = best_imputation_method
            recommendation['rationale'].append(f"High imputation quality (R² = {best_imputation_r2:.3f})")
            recommendation['rationale'].append(f"Preserves all data for training")
            
        elif exclusion_data_loss > 5:
            # High data loss -> recommend best available imputation
            recommendation['recommended_strategy'] = best_imputation_method or 'knn'
            recommendation['rationale'].append(f"High data loss with exclusion ({exclusion_data_loss:.1f}%)")
            recommendation['rationale'].append("Imputation preserves training data")
            
        else:
            # Moderate case -> slight preference for exclusion for simplicity
            recommendation['recommended_strategy'] = 'exclusion'
            recommendation['rationale'].append("Moderate data loss, simple and robust approach")
            recommendation['rationale'].append("Avoids potential imputation bias")
        
        # Add implementation details
        if recommendation['recommended_strategy'] == 'exclusion':
            recommendation['implementation_details'] = {
                'training_records': self.comparison_results['exclusion_analysis']['training_set_size'],
                'excluded_records': self.comparison_results['exclusion_analysis']['excluded_set_size'],
                'handling': 'Exclude missing final_test from training, use for prediction only'
            }
        else:
            method = recommendation['recommended_strategy']
            if method in self.imputation_results:
                recommendation['implementation_details'] = {
                    'method': method,
                    'imputed_records': self.imputation_results[method]['statistics']['imputed_count'],
                    'handling': f'Use {method} imputation for missing final_test values'
                }
        
        logger.info(f"Final recommendation: {recommendation['recommended_strategy']}")
        logger.info(f"Rationale: {'; '.join(recommendation['rationale'])}")
        
        return recommendation
    
    def save_processed_data(self, strategy: str, output_path: str = "data/processed/missing_handled.csv") -> None:
        """
        Save processed data based on chosen strategy.
        
        Args:
            strategy: Strategy to use ('exclusion', 'knn', 'iterative')
            output_path: Path to save the processed data
        """
        if strategy == 'exclusion':
            # Save original data with missing values intact
            processed_data = self.data.copy()
        else:
            # Save data with imputed values
            if strategy not in self.imputation_results:
                raise ValueError(f"Strategy '{strategy}' not available")
            
            processed_data = self.data.copy()
            processed_data['final_test'] = self.imputation_results[strategy]['imputed_series']
            
            # Add indicator for imputed values
            processed_data['final_test_imputed'] = self.data['final_test'].isnull().astype(int)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        processed_data.to_csv(output_file, index=False)
        logger.info(f"Saved processed data using {strategy} strategy to {output_file}")
    
    def save_analysis_results(self, output_path: str = "data/processed/final_test_analysis.json") -> None:
        """
        Save complete analysis results to JSON file.
        
        Args:
            output_path: Path to save the analysis results
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare results for JSON serialization
        results_to_save = {
            'missing_analysis': getattr(self, 'missing_analysis', {}),
            'comparison_results': getattr(self, 'comparison_results', {}),
            'final_recommendation': getattr(self, 'final_recommendation', {}),
            'imputation_methods': {}
        }
        
        # Add imputation method statistics
        for method_name, method_results in self.imputation_results.items():
            results_to_save['imputation_methods'][method_name] = method_results['statistics']
        
        with open(output_file, 'w') as f:
            json.dump(results_to_save, f, indent=2, default=str)
        
        logger.info(f"Saved analysis results to {output_file}")
    
    def save_audit_trail(self, output_path: str = "data/processed/final_test_handling_audit.json") -> None:
        """
        Save audit trail to JSON file.
        
        Args:
            output_path: Path to save the audit trail
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.audit_trail, f, indent=2, default=str)
        
        logger.info(f"Saved audit trail to {output_file}")
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Run the complete final_test missing values analysis.
        
        Returns:
            Dictionary with complete analysis summary
        """
        logger.info("Starting Phase 3.2.2: Final Test Missing Values Handling")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Analyze missing patterns
        self.missing_analysis = self.analyze_missing_final_test()
        
        # Step 3: Test KNN imputation
        self.knn_imputation(n_neighbors=5)
        
        # Step 4: Test iterative imputation
        self.iterative_imputation(max_iter=10)
        
        # Step 5: Compare strategies
        self.comparison_results = self.compare_strategies()
        
        # Step 6: Make final recommendation
        self.final_recommendation = self.make_final_recommendation()
        
        # Step 7: Create data splits for recommended strategy
        recommended_strategy = self.final_recommendation['recommended_strategy']
        self.create_train_validation_test_splits(strategy=recommended_strategy)
        
        # Step 8: Save results
        self.save_processed_data(recommended_strategy)
        self.save_analysis_results()
        self.save_audit_trail()
        
        # Step 9: Generate summary
        summary = {
            'input_file': self.input_path,
            'total_records': len(self.data),
            'missing_analysis': self.missing_analysis,
            'strategies_tested': ['exclusion', 'knn', 'iterative'],
            'final_recommendation': self.final_recommendation,
            'audit_trail_entries': len(self.audit_trail)
        }
        
        logger.info("Phase 3.2.2: Final Test Missing Values Handling completed successfully")
        return summary


def main():
    """
    Main execution function for Phase 3.2.2: Final Test Missing Values Handling.
    """
    # Initialize handler
    handler = FinalTestMissingHandler()
    
    # Run complete analysis
    summary = handler.run_complete_analysis()
    
    # Print summary
    print("\n=== Phase 3.2.2: Final Test Missing Values Handling Summary ===")
    print(f"Total records: {summary['total_records']}")
    print(f"Missing final_test: {summary['missing_analysis']['missing_count']} ({summary['missing_analysis']['missing_percentage']}%)")
    print(f"Strategies tested: {summary['strategies_tested']}")
    print(f"Recommended strategy: {summary['final_recommendation']['recommended_strategy']}")
    
    print("\nRecommendation rationale:")
    for reason in summary['final_recommendation']['rationale']:
        print(f"  - {reason}")
    
    print(f"\nAudit trail entries: {summary['audit_trail_entries']}")
    
    print("\nFinal test missing values handling completed successfully!")
    print("Output files:")
    print("- data/processed/missing_handled.csv")
    print("- data/processed/final_test_analysis.json")
    print("- data/processed/final_test_handling_audit.json")


if __name__ == "__main__":
    main()