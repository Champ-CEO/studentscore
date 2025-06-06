import pandas as pd
import numpy as np
import sqlite3
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
from scipy import stats
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidationQuality:
    """
    Comprehensive data validation and quality checking framework.
    
    Provides validation rules, quality scoring, outlier detection and handling
    for the student score dataset.
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None, db_path: Optional[str] = None):
        """
        Initialize the DataValidationQuality checker.
        
        Args:
            data: Pre-loaded DataFrame (alternative to db_path)
            db_path: Path to SQLite database file
        """
        self.data = data
        self.db_path = db_path
        self.validation_results = {}
        self.quality_metrics = {}
        self.outlier_info = {}
        
        # Define expected data schema
        self.expected_schema = {
            'student_id': 'object',
            'age': 'float64',
            'gender': 'category',
            'final_test': 'float64',
            'attendance_rate': 'float64',
            'tuition': 'category',
            'CCA': 'category',
            'direct_admission': 'category',
            'learning_style': 'category',
            'number_of_siblings': 'int64',
            'hours_per_week': 'float64',
            'n_male': 'float64',
            'n_female': 'float64',
            'sleep_time': 'object',
            'wake_time': 'object',
            'mode_of_transport': 'object',
            'bag_color': 'object'
        }
        
        # Define validation rules
        self.validation_rules = {
            'age': {'min': 10, 'max': 25, 'type': 'numeric'},
            'final_test': {'min': 0, 'max': 100, 'type': 'numeric'},
            'attendance_rate': {'min': 0, 'max': 100, 'type': 'numeric'},
            'number_of_siblings': {'min': 0, 'max': 20, 'type': 'integer'},
            'hours_per_week': {'min': 0, 'max': 168, 'type': 'numeric'},
            'n_male': {'min': 0, 'type': 'numeric'},
            'n_female': {'min': 0, 'type': 'numeric'},
            'gender': {'allowed_values': ['Male', 'Female'], 'type': 'categorical'},
            'tuition': {'allowed_values': ['Yes', 'No'], 'type': 'categorical'},
            'CCA': {'allowed_values': ['Sports', 'Arts', 'Clubs', 'None'], 'type': 'categorical'},
            'direct_admission': {'allowed_values': ['Yes', 'No'], 'type': 'categorical'},
            'learning_style': {'allowed_values': ['Visual', 'Auditory', 'Kinesthetic'], 'type': 'categorical'}
        }
        
        # Define quality thresholds
        self.quality_thresholds = {
            'missing_rate_threshold': 0.05,  # 5% missing data threshold
            'outlier_rate_threshold': 0.02,  # 2% outlier threshold
            'duplicate_rate_threshold': 0.01,  # 1% duplicate threshold
            'consistency_threshold': 0.95,  # 95% consistency threshold
            'completeness_threshold': 0.95  # 95% completeness threshold
        }
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from database or return provided DataFrame.
        
        Returns:
            DataFrame with the loaded data
        """
        if self.data is not None:
            logger.info(f"Using provided DataFrame with {len(self.data)} records")
            return self.data.copy()
        
        if self.db_path is None:
            raise ValueError("Either data or db_path must be provided")
        
        try:
            conn = sqlite3.connect(self.db_path)
            query = "SELECT * FROM score"
            data = pd.read_sql_query(query, conn)
            conn.close()
            
            logger.info(f"Loaded {len(data)} records from database")
            self.data = data
            return data.copy()
            
        except Exception as e:
            logger.error(f"Error loading data from database: {e}")
            raise
    
    def validate_schema(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data schema against expected structure.
        
        Args:
            data: DataFrame to validate
        
        Returns:
            Dictionary with schema validation results
        """
        schema_results = {
            'missing_columns': [],
            'extra_columns': [],
            'type_mismatches': [],
            'schema_valid': True
        }
        
        # Check for missing columns
        expected_columns = set(self.expected_schema.keys())
        actual_columns = set(data.columns)
        
        missing_columns = expected_columns - actual_columns
        extra_columns = actual_columns - expected_columns
        
        schema_results['missing_columns'] = list(missing_columns)
        schema_results['extra_columns'] = list(extra_columns)
        
        # Check data types for existing columns
        for column in expected_columns.intersection(actual_columns):
            expected_type = self.expected_schema[column]
            actual_type = str(data[column].dtype)
            
            # Allow some flexibility in type checking
            if not self._types_compatible(actual_type, expected_type):
                schema_results['type_mismatches'].append({
                    'column': column,
                    'expected': expected_type,
                    'actual': actual_type
                })
        
        # Overall schema validity
        schema_results['schema_valid'] = (
            len(missing_columns) == 0 and 
            len(schema_results['type_mismatches']) == 0
        )
        
        return schema_results
    
    def _types_compatible(self, actual_type: str, expected_type: str) -> bool:
        """
        Check if actual and expected data types are compatible.
        
        Args:
            actual_type: Actual data type as string
            expected_type: Expected data type as string
        
        Returns:
            True if types are compatible
        """
        # Define compatible type mappings
        compatible_types = {
            'object': ['object', 'string'],
            'category': ['category', 'object'],
            'float64': ['float64', 'float32', 'int64', 'int32'],
            'int64': ['int64', 'int32', 'float64'],
            'bool': ['bool', 'object']
        }
        
        if expected_type in compatible_types:
            return actual_type in compatible_types[expected_type]
        
        return actual_type == expected_type
    
    def validate_data_rules(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data against business rules and constraints.
        
        Args:
            data: DataFrame to validate
        
        Returns:
            Dictionary with rule validation results
        """
        rule_results = {
            'rule_violations': {},
            'total_violations': 0,
            'rules_passed': 0,
            'rules_failed': 0
        }
        
        for column, rules in self.validation_rules.items():
            if column not in data.columns:
                continue
            
            column_violations = []
            
            # Check numeric constraints
            if rules['type'] in ['numeric', 'integer']:
                if 'min' in rules:
                    min_violations = (data[column] < rules['min']).sum()
                    if min_violations > 0:
                        column_violations.append(f"Values below minimum ({rules['min']}): {min_violations}")
                
                if 'max' in rules:
                    max_violations = (data[column] > rules['max']).sum()
                    if max_violations > 0:
                        column_violations.append(f"Values above maximum ({rules['max']}): {max_violations}")
            
            # Check categorical constraints
            if rules['type'] == 'categorical' and 'allowed_values' in rules:
                invalid_values = ~data[column].isin(rules['allowed_values'] + [np.nan])
                invalid_count = invalid_values.sum()
                if invalid_count > 0:
                    unique_invalid = data[column][invalid_values].unique()
                    column_violations.append(f"Invalid categorical values: {list(unique_invalid)} (count: {invalid_count})")
            
            # Store violations for this column
            if column_violations:
                rule_results['rule_violations'][column] = column_violations
                rule_results['rules_failed'] += len(column_violations)
            else:
                rule_results['rules_passed'] += 1
        
        rule_results['total_violations'] = sum(
            len(violations) for violations in rule_results['rule_violations'].values()
        )
        
        return rule_results
    
    def detect_outliers(self, data: pd.DataFrame, methods: List[str] = ['iqr', 'zscore']) -> Dict[str, Any]:
        """
        Detect outliers using multiple statistical methods.
        
        Args:
            data: DataFrame to analyze
            methods: List of outlier detection methods to use
        
        Returns:
            Dictionary with outlier detection results
        """
        outlier_results = {
            'outliers_by_method': {},
            'outliers_by_column': {},
            'total_outliers': 0,
            'outlier_rate': 0.0
        }
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for method in methods:
            method_outliers = {}
            
            for column in numeric_columns:
                if column == 'index':  # Skip index column
                    continue
                
                column_data = data[column].dropna()
                if len(column_data) == 0:
                    continue
                
                if method == 'iqr':
                    outliers = self._detect_outliers_iqr(column_data)
                elif method == 'zscore':
                    outliers = self._detect_outliers_zscore(column_data)
                elif method == 'modified_zscore':
                    outliers = self._detect_outliers_modified_zscore(column_data)
                else:
                    continue
                
                if outliers.any():
                    method_outliers[column] = {
                        'count': outliers.sum(),
                        'percentage': (outliers.sum() / len(column_data)) * 100,
                        'indices': column_data[outliers].index.tolist(),
                        'values': column_data[outliers].tolist()
                    }
            
            outlier_results['outliers_by_method'][method] = method_outliers
        
        # Combine results across methods
        all_outlier_indices = set()
        for method_results in outlier_results['outliers_by_method'].values():
            for column_results in method_results.values():
                all_outlier_indices.update(column_results['indices'])
        
        outlier_results['total_outliers'] = len(all_outlier_indices)
        outlier_results['outlier_rate'] = (len(all_outlier_indices) / len(data)) * 100
        
        # Summarize by column
        for column in numeric_columns:
            if column == 'index':
                continue
            
            column_outliers = set()
            for method_results in outlier_results['outliers_by_method'].values():
                if column in method_results:
                    column_outliers.update(method_results[column]['indices'])
            
            if column_outliers:
                outlier_results['outliers_by_column'][column] = {
                    'count': len(column_outliers),
                    'percentage': (len(column_outliers) / len(data)) * 100,
                    'indices': list(column_outliers)
                }
        
        self.outlier_info = outlier_results
        return outlier_results
    
    def _detect_outliers_iqr(self, series: pd.Series, factor: float = 1.5) -> pd.Series:
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Args:
            series: Data series to analyze
            factor: IQR factor (typically 1.5)
        
        Returns:
            Boolean series indicating outliers
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        return (series < lower_bound) | (series > upper_bound)
    
    def _detect_outliers_zscore(self, series: pd.Series, threshold: float = 3.0) -> pd.Series:
        """
        Detect outliers using Z-score method.
        
        Args:
            series: Data series to analyze
            threshold: Z-score threshold (typically 3.0)
        
        Returns:
            Boolean series indicating outliers
        """
        z_scores = np.abs(stats.zscore(series))
        return z_scores > threshold
    
    def _detect_outliers_modified_zscore(self, series: pd.Series, threshold: float = 3.5) -> pd.Series:
        """
        Detect outliers using Modified Z-score method (using median).
        
        Args:
            series: Data series to analyze
            threshold: Modified Z-score threshold (typically 3.5)
        
        Returns:
            Boolean series indicating outliers
        """
        median = series.median()
        mad = np.median(np.abs(series - median))
        
        if mad == 0:
            return pd.Series(False, index=series.index)
        
        modified_z_scores = 0.6745 * (series - median) / mad
        return np.abs(modified_z_scores) > threshold
    
    def handle_outliers(self, data: pd.DataFrame, strategy: str = 'cap', 
                       outlier_info: Optional[Dict] = None) -> pd.DataFrame:
        """
        Handle outliers using specified strategy.
        
        Args:
            data: DataFrame to process
            strategy: Outlier handling strategy ('cap', 'remove', 'transform', 'flag')
            outlier_info: Pre-computed outlier information
        
        Returns:
            DataFrame with outliers handled
        """
        if outlier_info is None:
            outlier_info = self.detect_outliers(data)
        
        processed_data = data.copy()
        
        if strategy == 'cap':
            processed_data = self._cap_outliers(processed_data, outlier_info)
        elif strategy == 'remove':
            processed_data = self._remove_outliers(processed_data, outlier_info)
        elif strategy == 'transform':
            processed_data = self._transform_outliers(processed_data, outlier_info)
        elif strategy == 'flag':
            processed_data = self._flag_outliers(processed_data, outlier_info)
        else:
            raise ValueError(f"Unknown outlier handling strategy: {strategy}")
        
        logger.info(f"Outlier handling complete using '{strategy}' strategy")
        return processed_data
    
    def _cap_outliers(self, data: pd.DataFrame, outlier_info: Dict) -> pd.DataFrame:
        """
        Cap outliers at reasonable percentiles.
        
        Args:
            data: DataFrame to process
            outlier_info: Outlier information
        
        Returns:
            DataFrame with capped outliers
        """
        processed_data = data.copy()
        
        for column in outlier_info['outliers_by_column']:
            if column in processed_data.columns:
                # Cap at 1st and 99th percentiles
                lower_cap = processed_data[column].quantile(0.01)
                upper_cap = processed_data[column].quantile(0.99)
                
                processed_data[column] = processed_data[column].clip(lower=lower_cap, upper=upper_cap)
                
                logger.info(f"Capped outliers in {column}: [{lower_cap:.2f}, {upper_cap:.2f}]")
        
        return processed_data
    
    def _remove_outliers(self, data: pd.DataFrame, outlier_info: Dict) -> pd.DataFrame:
        """
        Remove outlier records from the dataset.
        
        Args:
            data: DataFrame to process
            outlier_info: Outlier information
        
        Returns:
            DataFrame with outliers removed
        """
        # Collect all outlier indices
        outlier_indices = set()
        for column_info in outlier_info['outliers_by_column'].values():
            outlier_indices.update(column_info['indices'])
        
        # Remove outlier records
        processed_data = data.drop(index=list(outlier_indices)).reset_index(drop=True)
        
        logger.info(f"Removed {len(outlier_indices)} outlier records")
        return processed_data
    
    def _transform_outliers(self, data: pd.DataFrame, outlier_info: Dict) -> pd.DataFrame:
        """
        Transform outliers using log transformation or other methods.
        
        Args:
            data: DataFrame to process
            outlier_info: Outlier information
        
        Returns:
            DataFrame with transformed outliers
        """
        processed_data = data.copy()
        
        for column in outlier_info['outliers_by_column']:
            if column in processed_data.columns and processed_data[column].min() > 0:
                # Apply log transformation for positive values
                processed_data[f'{column}_log'] = np.log1p(processed_data[column])
                logger.info(f"Applied log transformation to {column}")
        
        return processed_data
    
    def _flag_outliers(self, data: pd.DataFrame, outlier_info: Dict) -> pd.DataFrame:
        """
        Flag outliers with indicator variables.
        
        Args:
            data: DataFrame to process
            outlier_info: Outlier information
        
        Returns:
            DataFrame with outlier flags
        """
        processed_data = data.copy()
        
        for column in outlier_info['outliers_by_column']:
            flag_column = f'{column}_outlier_flag'
            processed_data[flag_column] = 0
            
            outlier_indices = outlier_info['outliers_by_column'][column]['indices']
            processed_data.loc[outlier_indices, flag_column] = 1
            
            logger.info(f"Added outlier flag for {column}: {len(outlier_indices)} outliers flagged")
        
        return processed_data
    
    def calculate_quality_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive data quality metrics.
        
        Args:
            data: DataFrame to analyze
        
        Returns:
            Dictionary with quality metrics
        """
        metrics = {
            'completeness': {},
            'consistency': {},
            'validity': {},
            'uniqueness': {},
            'overall_score': 0.0
        }
        
        # Completeness metrics
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isnull().sum().sum()
        metrics['completeness']['overall'] = (total_cells - missing_cells) / total_cells
        
        for column in data.columns:
            missing_count = data[column].isnull().sum()
            metrics['completeness'][column] = (len(data) - missing_count) / len(data)
        
        # Validity metrics (based on rule validation)
        rule_results = self.validate_data_rules(data)
        total_rules = rule_results['rules_passed'] + rule_results['rules_failed']
        metrics['validity']['rule_compliance'] = rule_results['rules_passed'] / total_rules if total_rules > 0 else 1.0
        
        # Uniqueness metrics
        if 'student_id' in data.columns:
            unique_ids = data['student_id'].nunique()
            total_records = len(data)
            metrics['uniqueness']['student_id'] = unique_ids / total_records
        
        # Consistency metrics (categorical value consistency)
        for column in data.select_dtypes(include=['object', 'category']).columns:
            if column in self.validation_rules and 'allowed_values' in self.validation_rules[column]:
                valid_values = data[column].isin(self.validation_rules[column]['allowed_values'] + [np.nan])
                metrics['consistency'][column] = valid_values.sum() / len(data)
        
        # Calculate overall quality score
        scores = []
        scores.append(metrics['completeness']['overall'])
        scores.append(metrics['validity']['rule_compliance'])
        
        if 'student_id' in metrics['uniqueness']:
            scores.append(metrics['uniqueness']['student_id'])
        
        consistency_scores = [score for score in metrics['consistency'].values() if isinstance(score, (int, float))]
        if consistency_scores:
            scores.append(np.mean(consistency_scores))
        
        metrics['overall_score'] = np.mean(scores)
        
        self.quality_metrics = metrics
        return metrics
    
    def check_quality_thresholds(self, quality_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if quality metrics meet defined thresholds.
        
        Args:
            quality_metrics: Quality metrics to check
        
        Returns:
            Dictionary with threshold check results
        """
        threshold_results = {
            'passed_checks': [],
            'failed_checks': [],
            'overall_pass': True
        }
        
        # Check completeness threshold
        if quality_metrics['completeness']['overall'] >= self.quality_thresholds['completeness_threshold']:
            threshold_results['passed_checks'].append('completeness')
        else:
            threshold_results['failed_checks'].append('completeness')
            threshold_results['overall_pass'] = False
        
        # Check consistency threshold
        consistency_scores = [score for score in quality_metrics['consistency'].values() if isinstance(score, (int, float))]
        if consistency_scores:
            avg_consistency = np.mean(consistency_scores)
            if avg_consistency >= self.quality_thresholds['consistency_threshold']:
                threshold_results['passed_checks'].append('consistency')
            else:
                threshold_results['failed_checks'].append('consistency')
                threshold_results['overall_pass'] = False
        
        # Check validity threshold
        if quality_metrics['validity']['rule_compliance'] >= 0.95:  # 95% rule compliance
            threshold_results['passed_checks'].append('validity')
        else:
            threshold_results['failed_checks'].append('validity')
            threshold_results['overall_pass'] = False
        
        return threshold_results
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive quality report.
        
        Returns:
            Dictionary with complete quality report
        """
        if not self.quality_metrics:
            raise ValueError("Quality metrics not calculated. Run calculate_quality_metrics first.")
        
        report = {
            'data_summary': {
                'total_records': len(self.data) if self.data is not None else 0,
                'total_columns': len(self.data.columns) if self.data is not None else 0
            },
            'schema_validation': self.validation_results.get('schema', {}),
            'rule_validation': self.validation_results.get('rules', {}),
            'quality_metrics': self.quality_metrics,
            'outlier_analysis': self.outlier_info,
            'threshold_checks': self.check_quality_thresholds(self.quality_metrics),
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """
        Generate recommendations based on quality analysis.
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check completeness
        if self.quality_metrics['completeness']['overall'] < 0.95:
            recommendations.append("Consider imputation strategies for missing values")
        
        # Check outliers
        if self.outlier_info and self.outlier_info['outlier_rate'] > 2.0:
            recommendations.append("High outlier rate detected - review outlier handling strategy")
        
        # Check rule violations
        if 'rules' in self.validation_results and self.validation_results['rules']['total_violations'] > 0:
            recommendations.append("Address data rule violations before modeling")
        
        return recommendations
    
    def run_full_validation_pipeline(self, outlier_strategy: str = 'cap') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run the complete validation and quality checking pipeline.
        
        Args:
            outlier_strategy: Strategy for handling outliers
        
        Returns:
            Tuple of (processed_data, quality_report)
        """
        logger.info("Starting full validation and quality pipeline")
        
        # Load data
        data = self.load_data()
        
        # Schema validation
        schema_results = self.validate_schema(data)
        self.validation_results['schema'] = schema_results
        
        # Rule validation
        rule_results = self.validate_data_rules(data)
        self.validation_results['rules'] = rule_results
        
        # Outlier detection
        outlier_results = self.detect_outliers(data)
        
        # Handle outliers
        processed_data = self.handle_outliers(data, strategy=outlier_strategy, outlier_info=outlier_results)
        
        # Calculate quality metrics
        quality_metrics = self.calculate_quality_metrics(processed_data)
        
        # Generate report
        quality_report = self.generate_quality_report()
        
        logger.info("Validation and quality pipeline complete")
        return processed_data, quality_report
    
    def save_quality_report(self, report: Dict[str, Any], output_path: str) -> None:
        """
        Save quality report to file.
        
        Args:
            report: Quality report to save
            output_path: Path to save the report
        """
        try:
            import json
            
            # Ensure output directory exists
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            serializable_report = convert_numpy_types(report)
            
            with open(output_path, 'w') as f:
                json.dump(serializable_report, f, indent=2)
            
            logger.info(f"Quality report saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving quality report: {e}")
            raise


# Example usage
if __name__ == "__main__":
    # Example with database
    db_path = "data/raw/score.db"
    validator = DataValidationQuality(db_path=db_path)
    
    # Run full pipeline
    processed_data, quality_report = validator.run_full_validation_pipeline(outlier_strategy='cap')
    
    # Save results
    processed_data.to_csv("data/processed/validated.csv", index=False)
    validator.save_quality_report(quality_report, "data/processed/quality_report.json")
    
    print(f"Validation complete. Quality score: {quality_report['quality_metrics']['overall_score']:.3f}")