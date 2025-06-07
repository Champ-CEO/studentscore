#!/usr/bin/env python3
"""
Phase 3.2.1: Missing Data Imputation for Attendance Rate

Implements Priority 2 of Phase 3 data preprocessing:
- Handles 778 missing attendance_rate values (4.89%)
- Implements median imputation by relevant subgroups
- Tests regression-based imputation for comparison
- Creates missing indicator variables
- Validates imputation strategy effectiveness

Follows TASKS.md Phase 3.2.1 specifications exactly.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttendanceRateImputer:
    """
    Handles missing attendance_rate imputation for Phase 3.2.1.
    
    Implements the requirements for task 3.2.1:
    - Identify all 778 missing attendance_rate values (4.89%)
    - Implement median imputation by subgroups
    - Test regression-based imputation
    - Create missing indicator variables
    - Validate imputation effectiveness
    """
    
    def __init__(self, input_path: str = "data/processed/standardized.csv"):
        """
        Initialize the AttendanceRateImputer.
        
        Args:
            input_path: Path to the standardized CSV file from Phase 3.1.2
        """
        self.input_path = input_path
        self.data = None
        self.imputed_data = None
        self.imputation_results = {}
        self.audit_trail = []
        
    def load_data(self) -> pd.DataFrame:
        """
        Load standardized data from Phase 3.1.2.
        
        Returns:
            DataFrame containing the standardized data
        """
        try:
            self.data = pd.read_csv(self.input_path)
            logger.info(f"Loaded {len(self.data)} records from {self.input_path}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def analyze_missing_attendance(self) -> Dict[str, Any]:
        """
        Analyze missing attendance_rate patterns.
        
        Returns:
            Dictionary with missing data analysis
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        missing_mask = self.data['attendance_rate'].isnull()
        missing_count = missing_mask.sum()
        total_count = len(self.data)
        missing_percentage = (missing_count / total_count) * 100
        
        analysis = {
            'total_records': total_count,
            'missing_count': int(missing_count),
            'missing_percentage': round(missing_percentage, 2),
            'non_missing_count': int(total_count - missing_count),
            'missing_by_groups': {}
        }
        
        logger.info(f"Missing attendance_rate analysis: {missing_count} missing ({missing_percentage:.2f}%)")
        
        # Analyze missing patterns by categorical groups
        categorical_cols = ['gender', 'CCA', 'learning_style', 'tuition']
        
        for col in categorical_cols:
            if col in self.data.columns:
                group_analysis = self.data.groupby(col)['attendance_rate'].agg([
                    'count', 
                    lambda x: x.isnull().sum(),
                    lambda x: (x.isnull().sum() / len(x)) * 100
                ]).round(2)
                group_analysis.columns = ['total', 'missing', 'missing_pct']
                analysis['missing_by_groups'][col] = group_analysis.to_dict('index')
                
                logger.info(f"Missing attendance_rate by {col}:")
                for group, stats in group_analysis.to_dict('index').items():
                    logger.info(f"  {group}: {stats['missing']}/{stats['total']} ({stats['missing_pct']:.1f}%)")
        
        return analysis
    
    def create_missing_indicator(self) -> pd.Series:
        """
        Create missing indicator variable for attendance_rate.
        
        Returns:
            Series with missing indicator (1 = missing, 0 = not missing)
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        missing_indicator = self.data['attendance_rate'].isnull().astype(int)
        
        logger.info(f"Created missing indicator: {missing_indicator.sum()} missing values marked")
        
        self.audit_trail.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'created_missing_indicator',
            'column': 'attendance_rate',
            'missing_count': int(missing_indicator.sum()),
            'details': 'Created binary indicator for missing attendance_rate values'
        })
        
        return missing_indicator
    
    def median_imputation_by_groups(self, groupby_columns: List[str] = None) -> pd.Series:
        """
        Implement median imputation by relevant subgroups.
        
        Args:
            groupby_columns: Columns to group by for imputation
            
        Returns:
            Series with imputed attendance_rate values
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if groupby_columns is None:
            groupby_columns = ['gender', 'CCA', 'learning_style']
        
        # Verify groupby columns exist
        existing_cols = [col for col in groupby_columns if col in self.data.columns]
        if len(existing_cols) != len(groupby_columns):
            missing_cols = [col for col in groupby_columns if col not in self.data.columns]
            logger.warning(f"Some groupby columns not found: {missing_cols}")
            groupby_columns = existing_cols
        
        logger.info(f"Performing median imputation by groups: {groupby_columns}")
        
        # Create copy for imputation
        attendance_imputed = self.data['attendance_rate'].copy()
        missing_mask = attendance_imputed.isnull()
        
        if missing_mask.sum() == 0:
            logger.info("No missing values in attendance_rate")
            return attendance_imputed
        
        # Calculate group medians
        group_medians = self.data.groupby(groupby_columns)['attendance_rate'].median()
        
        # Track imputation statistics
        imputation_stats = {
            'total_missing': int(missing_mask.sum()),
            'imputed_by_group': 0,
            'imputed_by_overall_median': 0,
            'group_medians_used': {},
            'fallback_median': None
        }
        
        # Calculate overall median as fallback
        overall_median = self.data['attendance_rate'].median()
        imputation_stats['fallback_median'] = overall_median
        
        # Impute missing values
        for idx in self.data[missing_mask].index:
            row = self.data.loc[idx]
            group_key = tuple(row[col] for col in groupby_columns)
            
            if group_key in group_medians and not pd.isna(group_medians[group_key]):
                attendance_imputed.loc[idx] = group_medians[group_key]
                imputation_stats['imputed_by_group'] += 1
                
                # Track which group medians were used
                group_str = str(group_key)
                if group_str not in imputation_stats['group_medians_used']:
                    imputation_stats['group_medians_used'][group_str] = {
                        'median_value': group_medians[group_key],
                        'count_used': 0
                    }
                imputation_stats['group_medians_used'][group_str]['count_used'] += 1
            else:
                # Fallback to overall median
                attendance_imputed.loc[idx] = overall_median
                imputation_stats['imputed_by_overall_median'] += 1
                logger.warning(f"Used overall median for group {group_key}")
        
        logger.info(f"Median imputation completed:")
        logger.info(f"  - Imputed by group median: {imputation_stats['imputed_by_group']}")
        logger.info(f"  - Imputed by overall median: {imputation_stats['imputed_by_overall_median']}")
        
        # Store results
        self.imputation_results['median_by_groups'] = {
            'method': 'median_by_groups',
            'groupby_columns': groupby_columns,
            'statistics': imputation_stats,
            'imputed_series': attendance_imputed
        }
        
        self.audit_trail.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'median_imputation_by_groups',
            'groupby_columns': groupby_columns,
            'statistics': imputation_stats,
            'details': f'Median imputation by groups: {groupby_columns}'
        })
        
        return attendance_imputed
    
    def regression_based_imputation(self) -> pd.Series:
        """
        Implement regression-based imputation using correlated features.
        
        Returns:
            Series with regression-imputed attendance_rate values
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        logger.info("Performing regression-based imputation")
        
        # Prepare features for regression
        feature_cols = ['age']
        
        # Add encoded categorical features
        categorical_cols = ['gender', 'CCA', 'learning_style', 'tuition']
        encoded_data = self.data.copy()
        
        # Label encode categorical variables
        label_encoders = {}
        for col in categorical_cols:
            if col in encoded_data.columns:
                le = LabelEncoder()
                # Handle missing values in categorical columns
                non_null_mask = encoded_data[col].notna()
                if non_null_mask.sum() > 0:
                    encoded_data.loc[non_null_mask, f'{col}_encoded'] = le.fit_transform(
                        encoded_data.loc[non_null_mask, col]
                    )
                    label_encoders[col] = le
                    feature_cols.append(f'{col}_encoded')
        
        # Add final_test if available (but handle missing values)
        if 'final_test' in encoded_data.columns:
            feature_cols.append('final_test')
        
        # Prepare training data (records with non-missing attendance_rate)
        train_mask = encoded_data['attendance_rate'].notna()
        
        # Ensure we have complete cases for training
        complete_mask = train_mask.copy()
        for col in feature_cols:
            if col in encoded_data.columns:
                complete_mask &= encoded_data[col].notna()
        
        if complete_mask.sum() < 10:
            logger.warning("Insufficient complete cases for regression. Using median imputation as fallback.")
            return self.median_imputation_by_groups()
        
        # Prepare training data
        X_train = encoded_data.loc[complete_mask, feature_cols]
        y_train = encoded_data.loc[complete_mask, 'attendance_rate']
        
        # Train regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Evaluate model on training data
        y_pred_train = model.predict(X_train)
        train_r2 = r2_score(y_train, y_pred_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        
        logger.info(f"Regression model performance: R² = {train_r2:.3f}, RMSE = {train_rmse:.3f}")
        
        # Prepare prediction data (records with missing attendance_rate)
        missing_mask = encoded_data['attendance_rate'].isnull()
        
        # Check which missing records have complete features
        predict_mask = missing_mask.copy()
        for col in feature_cols:
            if col in encoded_data.columns:
                predict_mask &= encoded_data[col].notna()
        
        attendance_imputed = encoded_data['attendance_rate'].copy()
        
        imputation_stats = {
            'total_missing': int(missing_mask.sum()),
            'imputed_by_regression': 0,
            'imputed_by_median_fallback': 0,
            'model_r2': train_r2,
            'model_rmse': train_rmse,
            'features_used': feature_cols
        }
        
        # Predict for records with complete features
        if predict_mask.sum() > 0:
            X_predict = encoded_data.loc[predict_mask, feature_cols]
            predictions = model.predict(X_predict)
            
            # Ensure predictions are within reasonable bounds (0-100)
            predictions = np.clip(predictions, 0, 100)
            
            attendance_imputed.loc[predict_mask] = predictions
            imputation_stats['imputed_by_regression'] = int(predict_mask.sum())
            
            logger.info(f"Regression imputation applied to {predict_mask.sum()} records")
        
        # Use median for remaining missing values (incomplete features)
        remaining_missing = missing_mask & ~predict_mask
        if remaining_missing.sum() > 0:
            median_value = encoded_data['attendance_rate'].median()
            attendance_imputed.loc[remaining_missing] = median_value
            imputation_stats['imputed_by_median_fallback'] = int(remaining_missing.sum())
            
            logger.info(f"Median fallback applied to {remaining_missing.sum()} records")
        
        # Store results
        self.imputation_results['regression_based'] = {
            'method': 'regression_based',
            'features_used': feature_cols,
            'statistics': imputation_stats,
            'model': model,
            'label_encoders': label_encoders,
            'imputed_series': attendance_imputed
        }
        
        self.audit_trail.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'regression_based_imputation',
            'features_used': feature_cols,
            'statistics': imputation_stats,
            'details': f'Regression-based imputation with R² = {train_r2:.3f}'
        })
        
        return attendance_imputed
    
    def compare_imputation_methods(self) -> Dict[str, Any]:
        """
        Compare different imputation methods.
        
        Returns:
            Dictionary with comparison results
        """
        if not self.imputation_results:
            raise ValueError("No imputation results available. Run imputation methods first.")
        
        comparison = {
            'methods_compared': list(self.imputation_results.keys()),
            'comparison_metrics': {},
            'recommendations': []
        }
        
        # Compare basic statistics
        for method_name, method_results in self.imputation_results.items():
            imputed_series = method_results['imputed_series']
            
            comparison['comparison_metrics'][method_name] = {
                'mean': imputed_series.mean(),
                'median': imputed_series.median(),
                'std': imputed_series.std(),
                'min': imputed_series.min(),
                'max': imputed_series.max(),
                'missing_after_imputation': int(imputed_series.isnull().sum())
            }
        
        # Generate recommendations
        if 'regression_based' in self.imputation_results:
            reg_r2 = self.imputation_results['regression_based']['statistics']['model_r2']
            if reg_r2 > 0.3:  # Reasonable predictive power
                comparison['recommendations'].append(
                    f"Regression-based imputation shows good predictive power (R² = {reg_r2:.3f})"
                )
            else:
                comparison['recommendations'].append(
                    f"Regression-based imputation has low predictive power (R² = {reg_r2:.3f}), consider median imputation"
                )
        
        if 'median_by_groups' in self.imputation_results:
            group_stats = self.imputation_results['median_by_groups']['statistics']
            group_pct = (group_stats['imputed_by_group'] / group_stats['total_missing']) * 100
            comparison['recommendations'].append(
                f"Group-based median imputation successfully used group medians for {group_pct:.1f}% of missing values"
            )
        
        logger.info(f"Imputation method comparison completed")
        return comparison
    
    def select_best_imputation(self, preferred_method: str = 'auto') -> pd.Series:
        """
        Select the best imputation method based on analysis.
        
        Args:
            preferred_method: 'auto', 'median_by_groups', or 'regression_based'
            
        Returns:
            Series with best imputed attendance_rate values
        """
        if not self.imputation_results:
            raise ValueError("No imputation results available. Run imputation methods first.")
        
        if preferred_method == 'auto':
            # Auto-select based on regression performance
            if 'regression_based' in self.imputation_results:
                reg_r2 = self.imputation_results['regression_based']['statistics']['model_r2']
                if reg_r2 > 0.2:  # Threshold for acceptable predictive power
                    selected_method = 'regression_based'
                    logger.info(f"Auto-selected regression-based imputation (R² = {reg_r2:.3f})")
                else:
                    selected_method = 'median_by_groups'
                    logger.info(f"Auto-selected median by groups (regression R² = {reg_r2:.3f} too low)")
            else:
                selected_method = 'median_by_groups'
                logger.info("Auto-selected median by groups (regression not available)")
        else:
            selected_method = preferred_method
            logger.info(f"Using preferred method: {selected_method}")
        
        if selected_method not in self.imputation_results:
            raise ValueError(f"Method '{selected_method}' not available. Available: {list(self.imputation_results.keys())}")
        
        best_imputed = self.imputation_results[selected_method]['imputed_series']
        
        self.audit_trail.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'selected_best_imputation',
            'method_selected': selected_method,
            'selection_criteria': preferred_method,
            'details': f'Selected {selected_method} as final imputation method'
        })
        
        return best_imputed
    
    def create_final_dataset(self, imputation_method: str = 'auto') -> pd.DataFrame:
        """
        Create final dataset with imputed attendance_rate and missing indicator.
        
        Args:
            imputation_method: Method to use for final imputation
            
        Returns:
            DataFrame with imputed data and missing indicator
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Create missing indicator
        missing_indicator = self.create_missing_indicator()
        
        # Get best imputed values
        imputed_attendance = self.select_best_imputation(imputation_method)
        
        # Create final dataset
        self.imputed_data = self.data.copy()
        self.imputed_data['attendance_rate'] = imputed_attendance
        self.imputed_data['attendance_rate_missing'] = missing_indicator
        
        logger.info(f"Created final dataset with imputed attendance_rate and missing indicator")
        return self.imputed_data
    
    def validate_imputation(self) -> Dict[str, Any]:
        """
        Validate the imputation results.
        
        Returns:
            Dictionary with validation results
        """
        if self.imputed_data is None:
            raise ValueError("No imputed data available. Run create_final_dataset() first.")
        
        validation = {
            'validation_passed': True,
            'issues_found': [],
            'statistics': {}
        }
        
        # Check for remaining missing values
        remaining_missing = self.imputed_data['attendance_rate'].isnull().sum()
        if remaining_missing > 0:
            validation['validation_passed'] = False
            validation['issues_found'].append(f"Still {remaining_missing} missing attendance_rate values")
        
        # Check value ranges
        min_val = self.imputed_data['attendance_rate'].min()
        max_val = self.imputed_data['attendance_rate'].max()
        
        if min_val < 0 or max_val > 100:
            validation['validation_passed'] = False
            validation['issues_found'].append(f"Attendance rate values outside valid range: {min_val} to {max_val}")
        
        # Statistical validation
        validation['statistics'] = {
            'total_records': len(self.imputed_data),
            'missing_after_imputation': int(remaining_missing),
            'attendance_rate_stats': {
                'mean': self.imputed_data['attendance_rate'].mean(),
                'median': self.imputed_data['attendance_rate'].median(),
                'std': self.imputed_data['attendance_rate'].std(),
                'min': min_val,
                'max': max_val
            },
            'missing_indicator_stats': {
                'missing_marked': int(self.imputed_data['attendance_rate_missing'].sum()),
                'percentage_missing': (self.imputed_data['attendance_rate_missing'].sum() / len(self.imputed_data)) * 100
            }
        }
        
        if validation['validation_passed']:
            logger.info("Attendance rate imputation validation PASSED")
        else:
            logger.warning(f"Validation issues: {validation['issues_found']}")
        
        return validation
    
    def get_imputation_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of imputation process.
        
        Returns:
            Dictionary with complete imputation summary
        """
        summary = {
            'input_file': self.input_path,
            'total_records': len(self.data) if self.data is not None else 0,
            'methods_tested': list(self.imputation_results.keys()),
            'audit_trail_entries': len(self.audit_trail)
        }
        
        if self.imputed_data is not None:
            summary['final_statistics'] = {
                'records_after_imputation': len(self.imputed_data),
                'missing_values_imputed': int(self.imputed_data['attendance_rate_missing'].sum()),
                'imputation_percentage': (self.imputed_data['attendance_rate_missing'].sum() / len(self.imputed_data)) * 100
            }
        
        return summary
    
    def save_imputed_data(self, output_path: str = "data/processed/imputed.csv") -> None:
        """
        Save imputed data to CSV file.
        
        Args:
            output_path: Path to save the imputed data
        """
        if self.imputed_data is None:
            raise ValueError("No imputed data available. Run imputation process first.")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.imputed_data.to_csv(output_file, index=False)
        logger.info(f"Saved imputed data to {output_file}")
    
    def save_imputation_results(self, output_path: str = "data/processed/attendance_imputation_results.json") -> None:
        """
        Save imputation results and comparison to JSON file.
        
        Args:
            output_path: Path to save the results
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare results for JSON serialization
        results_to_save = {}
        for method_name, method_results in self.imputation_results.items():
            results_to_save[method_name] = {
                'method': method_results['method'],
                'statistics': method_results['statistics']
            }
            
            # Add method-specific details
            if 'groupby_columns' in method_results:
                results_to_save[method_name]['groupby_columns'] = method_results['groupby_columns']
            if 'features_used' in method_results:
                results_to_save[method_name]['features_used'] = method_results['features_used']
        
        with open(output_file, 'w') as f:
            json.dump(results_to_save, f, indent=2, default=str)
        
        logger.info(f"Saved imputation results to {output_file}")
    
    def save_audit_trail(self, output_path: str = "data/processed/attendance_imputation_audit.json") -> None:
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
    
    def run_complete_imputation(self, final_method: str = 'auto') -> Dict[str, Any]:
        """
        Run the complete attendance rate imputation process.
        
        Args:
            final_method: Final imputation method to use
            
        Returns:
            Dictionary with process summary
        """
        logger.info("Starting Phase 3.2.1: Missing Data Imputation for Attendance Rate")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Analyze missing patterns
        missing_analysis = self.analyze_missing_attendance()
        
        # Step 3: Test median imputation by groups
        self.median_imputation_by_groups()
        
        # Step 4: Test regression-based imputation
        self.regression_based_imputation()
        
        # Step 5: Compare methods
        comparison = self.compare_imputation_methods()
        
        # Step 6: Create final dataset
        self.create_final_dataset(final_method)
        
        # Step 7: Validate results
        validation = self.validate_imputation()
        
        # Step 8: Save results
        self.save_imputed_data()
        self.save_imputation_results()
        self.save_audit_trail()
        
        # Step 9: Generate summary
        summary = self.get_imputation_summary()
        summary.update({
            'missing_analysis': missing_analysis,
            'method_comparison': comparison,
            'validation_results': validation
        })
        
        logger.info("Phase 3.2.1: Missing Data Imputation for Attendance Rate completed successfully")
        return summary


def main():
    """
    Main execution function for Phase 3.2.1: Missing Data Imputation for Attendance Rate.
    """
    # Initialize imputer
    imputer = AttendanceRateImputer()
    
    # Run complete imputation process
    summary = imputer.run_complete_imputation()
    
    # Print summary
    print("\n=== Phase 3.2.1: Missing Data Imputation for Attendance Rate Summary ===")
    print(f"Total records: {summary['total_records']}")
    print(f"Missing values found: {summary['missing_analysis']['missing_count']} ({summary['missing_analysis']['missing_percentage']}%)")
    print(f"Methods tested: {summary['methods_tested']}")
    print(f"Values imputed: {summary['final_statistics']['missing_values_imputed']}")
    print(f"Validation passed: {summary['validation_results']['validation_passed']}")
    
    if summary['validation_results']['issues_found']:
        print(f"Issues found: {summary['validation_results']['issues_found']}")
    
    print(f"Audit trail entries: {summary['audit_trail_entries']}")
    
    # Show method comparison
    print("\nMethod comparison:")
    for method, metrics in summary['method_comparison']['comparison_metrics'].items():
        print(f"  {method}: mean={metrics['mean']:.2f}, std={metrics['std']:.2f}")
    
    print("\nAttendance rate imputation completed successfully!")
    print("Output files:")
    print("- data/processed/imputed.csv")
    print("- data/processed/attendance_imputation_results.json")
    print("- data/processed/attendance_imputation_audit.json")


if __name__ == "__main__":
    main()