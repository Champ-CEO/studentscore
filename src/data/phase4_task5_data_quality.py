#!/usr/bin/env python3
"""
Phase 4 Task 5.1: Data Quality Targets (Medium Priority)

This module implements task 4.5.1 from TASKS.md:
- Validate data quality targets and thresholds
- Ensure feature engineering maintains data integrity
- Generate comprehensive quality reports
- Implement quality gates for model readiness

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from datetime import datetime
import warnings
from scipy import stats

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Phase4DataQuality:
    """
    Validates and ensures data quality for Phase 4 feature engineering.
    """
    
    def __init__(self, df: pd.DataFrame, target_column: str = 'final_test'):
        """
        Initialize with the dataset.
        
        Args:
            df: DataFrame with selected features from previous tasks
            target_column: Name of the target variable column
        """
        self.df = df.copy()
        self.target_column = target_column
        self.quality_targets = self._define_quality_targets()
        self.quality_results = {}
        self.quality_issues = []
        self.quality_metrics = {}
        self.audit_log = []
        
    def _define_quality_targets(self) -> Dict[str, Any]:
        """
        Define data quality targets and thresholds.
        
        Returns:
            Dictionary containing quality targets
        """
        targets = {
            'completeness': {
                'target_missing_rate': 0.05,  # Max 5% missing values per feature
                'critical_missing_rate': 0.20,  # Critical threshold
                'target_overall_completeness': 0.95  # Min 95% overall completeness
            },
            'consistency': {
                'target_duplicate_rate': 0.01,  # Max 1% duplicates
                'target_outlier_rate': 0.05,  # Max 5% outliers per feature
                'target_data_type_consistency': 1.0  # 100% correct data types
            },
            'validity': {
                'target_range_compliance': 0.95,  # Min 95% values in expected ranges
                'target_format_compliance': 1.0,  # 100% correct formats
                'target_business_rule_compliance': 0.95  # Min 95% business rule compliance
            },
            'accuracy': {
                'target_correlation_preservation': 0.8,  # Min correlation with original features
                'target_distribution_similarity': 0.7,  # Min distribution similarity
                'target_statistical_significance': 0.05  # Max p-value for statistical tests
            },
            'uniqueness': {
                'target_unique_id_rate': 1.0,  # 100% unique IDs
                'target_feature_uniqueness': 0.01  # Max 1% identical feature vectors
            },
            'timeliness': {
                'target_processing_time': 300,  # Max 5 minutes processing time
                'target_data_freshness': 24  # Max 24 hours data age
            }
        }
        
        return targets
        
    def assess_completeness(self) -> Dict[str, Any]:
        """
        Assess data completeness quality.
        
        Returns:
            Dictionary containing completeness assessment results
        """
        logger.info("Assessing data completeness")
        
        completeness_results = {
            'feature_missing_rates': {},
            'overall_completeness': 0.0,
            'critical_missing_features': [],
            'target_compliance': {
                'missing_rate_compliance': True,
                'overall_completeness_compliance': True
            }
        }
        
        # Calculate missing rates per feature
        total_rows = len(self.df)
        target_missing_rate = self.quality_targets['completeness']['target_missing_rate']
        critical_missing_rate = self.quality_targets['completeness']['critical_missing_rate']
        
        for column in self.df.columns:
            missing_count = self.df[column].isnull().sum()
            missing_rate = missing_count / total_rows
            
            completeness_results['feature_missing_rates'][column] = {
                'missing_count': missing_count,
                'missing_rate': missing_rate,
                'compliant': missing_rate <= target_missing_rate,
                'critical': missing_rate > critical_missing_rate
            }
            
            # Track critical missing features
            if missing_rate > critical_missing_rate:
                completeness_results['critical_missing_features'].append({
                    'feature': column,
                    'missing_rate': missing_rate,
                    'missing_count': missing_count
                })
                
        # Calculate overall completeness
        total_cells = self.df.size
        total_missing = self.df.isnull().sum().sum()
        overall_completeness = 1 - (total_missing / total_cells)
        completeness_results['overall_completeness'] = overall_completeness
        
        # Check compliance
        target_overall = self.quality_targets['completeness']['target_overall_completeness']
        completeness_results['target_compliance']['overall_completeness_compliance'] = \
            overall_completeness >= target_overall
            
        # Check feature-level compliance
        non_compliant_features = sum(1 for result in completeness_results['feature_missing_rates'].values() 
                                   if not result['compliant'])
        completeness_results['target_compliance']['missing_rate_compliance'] = \
            non_compliant_features == 0
            
        # Log results
        logger.info(f"Overall completeness: {overall_completeness:.3f}")
        logger.info(f"Critical missing features: {len(completeness_results['critical_missing_features'])}")
        
        if completeness_results['critical_missing_features']:
            for feature_info in completeness_results['critical_missing_features'][:3]:
                logger.warning(f"Critical missing: {feature_info['feature']} ({feature_info['missing_rate']:.3f})")
                
        return completeness_results
        
    def assess_consistency(self) -> Dict[str, Any]:
        """
        Assess data consistency quality.
        
        Returns:
            Dictionary containing consistency assessment results
        """
        logger.info("Assessing data consistency")
        
        consistency_results = {
            'duplicate_analysis': {},
            'outlier_analysis': {},
            'data_type_consistency': {},
            'target_compliance': {
                'duplicate_compliance': True,
                'outlier_compliance': True,
                'data_type_compliance': True
            }
        }
        
        # Duplicate analysis
        total_rows = len(self.df)
        duplicate_rows = self.df.duplicated().sum()
        duplicate_rate = duplicate_rows / total_rows
        
        consistency_results['duplicate_analysis'] = {
            'total_rows': total_rows,
            'duplicate_rows': duplicate_rows,
            'duplicate_rate': duplicate_rate,
            'compliant': duplicate_rate <= self.quality_targets['consistency']['target_duplicate_rate']
        }
        
        # Outlier analysis per numerical feature
        target_outlier_rate = self.quality_targets['consistency']['target_outlier_rate']
        numerical_columns = self.df.select_dtypes(include=[np.number]).columns
        
        for column in numerical_columns:
            if column == self.target_column:
                continue
                
            feature_data = self.df[column].dropna()
            if len(feature_data) == 0:
                continue
                
            # Calculate outliers using IQR method
            q1 = feature_data.quantile(0.25)
            q3 = feature_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = ((feature_data < lower_bound) | (feature_data > upper_bound)).sum()
            outlier_rate = outliers / len(feature_data)
            
            consistency_results['outlier_analysis'][column] = {
                'outlier_count': outliers,
                'outlier_rate': outlier_rate,
                'compliant': outlier_rate <= target_outlier_rate,
                'bounds': {'lower': lower_bound, 'upper': upper_bound}
            }
            
        # Data type consistency
        expected_types = {
            'numerical': [np.number],
            'categorical': ['object', 'category'],
            'boolean': ['bool']
        }
        
        type_issues = []
        for column in self.df.columns:
            dtype = self.df[column].dtype
            
            # Check for mixed types or unexpected types
            if dtype == 'object':
                # Check if it should be numerical
                try:
                    pd.to_numeric(self.df[column].dropna())
                    type_issues.append({
                        'column': column,
                        'issue': 'object_should_be_numeric',
                        'current_type': str(dtype)
                    })
                except (ValueError, TypeError):
                    pass
                    
        consistency_results['data_type_consistency'] = {
            'type_issues': type_issues,
            'compliant': len(type_issues) == 0
        }
        
        # Update compliance flags
        consistency_results['target_compliance']['duplicate_compliance'] = \
            consistency_results['duplicate_analysis']['compliant']
            
        outlier_non_compliant = sum(1 for result in consistency_results['outlier_analysis'].values() 
                                  if not result['compliant'])
        consistency_results['target_compliance']['outlier_compliance'] = \
            outlier_non_compliant == 0
            
        consistency_results['target_compliance']['data_type_compliance'] = \
            consistency_results['data_type_consistency']['compliant']
            
        # Log results
        logger.info(f"Duplicate rate: {duplicate_rate:.4f}")
        logger.info(f"Outlier non-compliant features: {outlier_non_compliant}")
        logger.info(f"Data type issues: {len(type_issues)}")
        
        return consistency_results
        
    def assess_validity(self) -> Dict[str, Any]:
        """
        Assess data validity quality.
        
        Returns:
            Dictionary containing validity assessment results
        """
        logger.info("Assessing data validity")
        
        validity_results = {
            'range_compliance': {},
            'format_compliance': {},
            'business_rule_compliance': {},
            'target_compliance': {
                'range_compliance': True,
                'format_compliance': True,
                'business_rule_compliance': True
            }
        }
        
        # Range compliance for known features
        expected_ranges = {
            'final_test': (0, 100),
            'previous_score': (0, 100),
            'study_hours': (0, 24),
            'attendance': (0, 100),
            'age': (10, 25)
        }
        
        for column, (min_val, max_val) in expected_ranges.items():
            if column in self.df.columns:
                feature_data = self.df[column].dropna()
                if len(feature_data) == 0:
                    continue
                    
                in_range = ((feature_data >= min_val) & (feature_data <= max_val)).sum()
                compliance_rate = in_range / len(feature_data)
                
                validity_results['range_compliance'][column] = {
                    'expected_range': (min_val, max_val),
                    'actual_range': (feature_data.min(), feature_data.max()),
                    'in_range_count': in_range,
                    'compliance_rate': compliance_rate,
                    'compliant': compliance_rate >= self.quality_targets['validity']['target_range_compliance']
                }
                
        # Format compliance (for categorical features)
        format_patterns = {
            'parental_education_level': ['No Education', 'Primary', 'Secondary', 'Higher Secondary', 'Bachelor', 'Master', 'PhD'],
            'distance_from_home': ['Near', 'Moderate', 'Far']
        }
        
        for column, valid_values in format_patterns.items():
            if column in self.df.columns:
                feature_data = self.df[column].dropna()
                if len(feature_data) == 0:
                    continue
                    
                valid_count = feature_data.isin(valid_values).sum()
                compliance_rate = valid_count / len(feature_data)
                
                validity_results['format_compliance'][column] = {
                    'valid_values': valid_values,
                    'unique_values': feature_data.unique().tolist(),
                    'valid_count': valid_count,
                    'compliance_rate': compliance_rate,
                    'compliant': compliance_rate >= self.quality_targets['validity']['target_format_compliance']
                }
                
        # Business rule compliance
        business_rules = []
        
        # Rule 1: Study hours should be reasonable for attendance
        if 'study_hours' in self.df.columns and 'attendance' in self.df.columns:
            study_attendance_data = self.df[['study_hours', 'attendance']].dropna()
            if len(study_attendance_data) > 0:
                # Students with high attendance should generally have reasonable study hours
                high_attendance = study_attendance_data['attendance'] > 80
                reasonable_study = (study_attendance_data['study_hours'] >= 1) & (study_attendance_data['study_hours'] <= 12)
                
                rule_compliance = (high_attendance & reasonable_study).sum() / high_attendance.sum() if high_attendance.sum() > 0 else 1.0
                
                business_rules.append({
                    'rule': 'high_attendance_reasonable_study',
                    'description': 'Students with >80% attendance should have 1-12 study hours',
                    'compliance_rate': rule_compliance,
                    'compliant': rule_compliance >= self.quality_targets['validity']['target_business_rule_compliance']
                })
                
        # Rule 2: Previous score and final test correlation should be positive
        if 'previous_score' in self.df.columns and self.target_column in self.df.columns:
            score_data = self.df[['previous_score', self.target_column]].dropna()
            if len(score_data) > 10:
                correlation = score_data['previous_score'].corr(score_data[self.target_column])
                
                business_rules.append({
                    'rule': 'positive_score_correlation',
                    'description': 'Previous score and final test should be positively correlated',
                    'correlation': correlation,
                    'compliant': correlation > 0.3  # Reasonable positive correlation
                })
                
        validity_results['business_rule_compliance'] = business_rules
        
        # Update compliance flags
        range_non_compliant = sum(1 for result in validity_results['range_compliance'].values() 
                                if not result['compliant'])
        validity_results['target_compliance']['range_compliance'] = range_non_compliant == 0
        
        format_non_compliant = sum(1 for result in validity_results['format_compliance'].values() 
                                 if not result['compliant'])
        validity_results['target_compliance']['format_compliance'] = format_non_compliant == 0
        
        business_non_compliant = sum(1 for rule in business_rules if not rule['compliant'])
        validity_results['target_compliance']['business_rule_compliance'] = business_non_compliant == 0
        
        # Log results
        logger.info(f"Range non-compliant features: {range_non_compliant}")
        logger.info(f"Format non-compliant features: {format_non_compliant}")
        logger.info(f"Business rule violations: {business_non_compliant}")
        
        return validity_results
        
    def assess_accuracy(self) -> Dict[str, Any]:
        """
        Assess data accuracy quality.
        
        Returns:
            Dictionary containing accuracy assessment results
        """
        logger.info("Assessing data accuracy")
        
        accuracy_results = {
            'statistical_tests': {},
            'distribution_analysis': {},
            'target_compliance': {
                'statistical_significance': True,
                'distribution_similarity': True
            }
        }
        
        # Statistical tests for feature engineering validation
        numerical_features = self.df.select_dtypes(include=[np.number]).columns
        
        for feature in numerical_features:
            if feature == self.target_column:
                continue
                
            feature_data = self.df[feature].dropna()
            if len(feature_data) < 10:
                continue
                
            # Normality test
            try:
                shapiro_stat, shapiro_p = stats.shapiro(feature_data.sample(min(5000, len(feature_data))))
                
                accuracy_results['statistical_tests'][feature] = {
                    'normality_test': {
                        'statistic': shapiro_stat,
                        'p_value': shapiro_p,
                        'is_normal': shapiro_p > 0.05
                    }
                }
            except Exception as e:
                logger.warning(f"Statistical test failed for {feature}: {str(e)}")
                
        # Distribution analysis
        for feature in numerical_features:
            if feature == self.target_column:
                continue
                
            feature_data = self.df[feature].dropna()
            if len(feature_data) < 10:
                continue
                
            # Calculate distribution metrics
            skewness = stats.skew(feature_data)
            kurtosis = stats.kurtosis(feature_data)
            
            accuracy_results['distribution_analysis'][feature] = {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'mean': feature_data.mean(),
                'std': feature_data.std(),
                'median': feature_data.median(),
                'reasonable_distribution': abs(skewness) < 3 and abs(kurtosis) < 10
            }
            
        # Update compliance flags
        significant_tests = sum(1 for test_results in accuracy_results['statistical_tests'].values() 
                              if test_results.get('normality_test', {}).get('p_value', 1) < 0.001)
        total_tests = len(accuracy_results['statistical_tests'])
        
        if total_tests > 0:
            significance_rate = significant_tests / total_tests
            accuracy_results['target_compliance']['statistical_significance'] = \
                significance_rate <= self.quality_targets['accuracy']['target_statistical_significance']
                
        reasonable_distributions = sum(1 for dist_results in accuracy_results['distribution_analysis'].values() 
                                     if dist_results['reasonable_distribution'])
        total_distributions = len(accuracy_results['distribution_analysis'])
        
        if total_distributions > 0:
            distribution_rate = reasonable_distributions / total_distributions
            accuracy_results['target_compliance']['distribution_similarity'] = \
                distribution_rate >= self.quality_targets['accuracy']['target_distribution_similarity']
                
        # Log results
        logger.info(f"Statistical tests performed: {total_tests}")
        logger.info(f"Reasonable distributions: {reasonable_distributions}/{total_distributions}")
        
        return accuracy_results
        
    def assess_uniqueness(self) -> Dict[str, Any]:
        """
        Assess data uniqueness quality.
        
        Returns:
            Dictionary containing uniqueness assessment results
        """
        logger.info("Assessing data uniqueness")
        
        uniqueness_results = {
            'id_uniqueness': {},
            'feature_vector_uniqueness': {},
            'target_compliance': {
                'id_uniqueness': True,
                'feature_uniqueness': True
            }
        }
        
        # ID uniqueness
        id_columns = [col for col in self.df.columns if 'id' in col.lower()]
        
        for id_col in id_columns:
            total_values = len(self.df[id_col].dropna())
            unique_values = self.df[id_col].nunique()
            uniqueness_rate = unique_values / total_values if total_values > 0 else 0
            
            uniqueness_results['id_uniqueness'][id_col] = {
                'total_values': total_values,
                'unique_values': unique_values,
                'uniqueness_rate': uniqueness_rate,
                'compliant': uniqueness_rate >= self.quality_targets['uniqueness']['target_unique_id_rate']
            }
            
        # Feature vector uniqueness
        feature_columns = [col for col in self.df.columns 
                          if col not in id_columns and col != self.target_column]
        
        if feature_columns:
            feature_df = self.df[feature_columns].fillna(-999)  # Fill NaN for comparison
            total_rows = len(feature_df)
            unique_rows = len(feature_df.drop_duplicates())
            uniqueness_rate = unique_rows / total_rows if total_rows > 0 else 0
            
            uniqueness_results['feature_vector_uniqueness'] = {
                'total_rows': total_rows,
                'unique_rows': unique_rows,
                'uniqueness_rate': uniqueness_rate,
                'compliant': (1 - uniqueness_rate) <= self.quality_targets['uniqueness']['target_feature_uniqueness']
            }
            
        # Update compliance flags
        id_non_compliant = sum(1 for result in uniqueness_results['id_uniqueness'].values() 
                             if not result['compliant'])
        uniqueness_results['target_compliance']['id_uniqueness'] = id_non_compliant == 0
        
        uniqueness_results['target_compliance']['feature_uniqueness'] = \
            uniqueness_results['feature_vector_uniqueness'].get('compliant', True)
            
        # Log results
        logger.info(f"ID columns checked: {len(id_columns)}")
        logger.info(f"Feature vector uniqueness: {uniqueness_results['feature_vector_uniqueness'].get('uniqueness_rate', 0):.3f}")
        
        return uniqueness_results
        
    def generate_quality_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report.
        
        Returns:
            Dictionary containing complete quality assessment
        """
        logger.info("Generating comprehensive quality report")
        
        # Run all quality assessments
        completeness_results = self.assess_completeness()
        consistency_results = self.assess_consistency()
        validity_results = self.assess_validity()
        accuracy_results = self.assess_accuracy()
        uniqueness_results = self.assess_uniqueness()
        
        # Compile overall results
        quality_report = {
            'assessment_timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'shape': self.df.shape,
                'columns': self.df.columns.tolist(),
                'target_column': self.target_column
            },
            'quality_targets': self.quality_targets,
            'assessments': {
                'completeness': completeness_results,
                'consistency': consistency_results,
                'validity': validity_results,
                'accuracy': accuracy_results,
                'uniqueness': uniqueness_results
            }
        }
        
        # Calculate overall compliance
        all_compliance_flags = []
        for assessment in quality_report['assessments'].values():
            if 'target_compliance' in assessment:
                all_compliance_flags.extend(assessment['target_compliance'].values())
                
        overall_compliance = all(all_compliance_flags) if all_compliance_flags else False
        
        quality_report['overall_compliance'] = overall_compliance
        quality_report['compliance_rate'] = sum(all_compliance_flags) / len(all_compliance_flags) if all_compliance_flags else 0
        
        # Identify critical issues
        critical_issues = []
        
        # Critical completeness issues
        if completeness_results['critical_missing_features']:
            critical_issues.extend([
                f"Critical missing data in {feature['feature']}: {feature['missing_rate']:.1%}"
                for feature in completeness_results['critical_missing_features']
            ])
            
        # Critical consistency issues
        if not consistency_results['target_compliance']['duplicate_compliance']:
            duplicate_rate = consistency_results['duplicate_analysis']['duplicate_rate']
            critical_issues.append(f"High duplicate rate: {duplicate_rate:.1%}")
            
        quality_report['critical_issues'] = critical_issues
        
        # Generate recommendations
        recommendations = self._generate_recommendations(quality_report)
        quality_report['recommendations'] = recommendations
        
        # Store results
        self.quality_results = quality_report
        
        # Log summary
        logger.info(f"Quality assessment completed")
        logger.info(f"Overall compliance: {overall_compliance}")
        logger.info(f"Compliance rate: {quality_report['compliance_rate']:.1%}")
        logger.info(f"Critical issues: {len(critical_issues)}")
        
        return quality_report
        
    def _generate_recommendations(self, quality_report: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on quality assessment.
        
        Args:
            quality_report: Complete quality assessment report
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        assessments = quality_report['assessments']
        
        # Completeness recommendations
        if not assessments['completeness']['target_compliance']['overall_completeness_compliance']:
            recommendations.append("Improve data collection processes to reduce missing values")
            
        if assessments['completeness']['critical_missing_features']:
            recommendations.append("Implement imputation strategies for critical missing features")
            
        # Consistency recommendations
        if not assessments['consistency']['target_compliance']['duplicate_compliance']:
            recommendations.append("Implement deduplication procedures")
            
        outlier_issues = sum(1 for result in assessments['consistency']['outlier_analysis'].values() 
                           if not result['compliant'])
        if outlier_issues > 0:
            recommendations.append(f"Review and handle outliers in {outlier_issues} features")
            
        # Validity recommendations
        if not assessments['validity']['target_compliance']['range_compliance']:
            recommendations.append("Implement data validation rules for range compliance")
            
        if not assessments['validity']['target_compliance']['business_rule_compliance']:
            recommendations.append("Review and enforce business rule compliance")
            
        # Accuracy recommendations
        if not assessments['accuracy']['target_compliance']['distribution_similarity']:
            recommendations.append("Consider additional data transformations for distribution normalization")
            
        # Uniqueness recommendations
        if not assessments['uniqueness']['target_compliance']['feature_uniqueness']:
            recommendations.append("Review feature engineering to reduce identical feature vectors")
            
        return recommendations
        
    def save_quality_report(self, output_path: str = "data/featured/data_quality_report.json") -> None:
        """
        Save the quality report.
        
        Args:
            output_path: Path to save the report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.quality_results, f, indent=2, default=str)
            
        logger.info(f"Quality report saved to {output_path}")
        
    def is_model_ready(self) -> Tuple[bool, List[str]]:
        """
        Determine if the dataset is ready for modeling based on quality gates.
        
        Returns:
            Tuple of (is_ready, blocking_issues)
        """
        if not self.quality_results:
            self.generate_quality_report()
            
        blocking_issues = []
        
        # Check critical quality gates
        assessments = self.quality_results['assessments']
        
        # Gate 1: Overall completeness
        if not assessments['completeness']['target_compliance']['overall_completeness_compliance']:
            blocking_issues.append("Overall data completeness below target")
            
        # Gate 2: Critical missing features
        if assessments['completeness']['critical_missing_features']:
            blocking_issues.append(f"Critical missing data in {len(assessments['completeness']['critical_missing_features'])} features")
            
        # Gate 3: Data type consistency
        if not assessments['consistency']['target_compliance']['data_type_compliance']:
            blocking_issues.append("Data type inconsistencies detected")
            
        # Gate 4: Business rule compliance
        if not assessments['validity']['target_compliance']['business_rule_compliance']:
            blocking_issues.append("Business rule violations detected")
            
        # Gate 5: ID uniqueness
        if not assessments['uniqueness']['target_compliance']['id_uniqueness']:
            blocking_issues.append("ID uniqueness violations detected")
            
        is_ready = len(blocking_issues) == 0
        
        logger.info(f"Model readiness: {'READY' if is_ready else 'NOT READY'}")
        if blocking_issues:
            logger.warning(f"Blocking issues: {blocking_issues}")
            
        return is_ready, blocking_issues


def main():
    """
    Main function to run Phase 4 Task 5.1: Data Quality Targets.
    """
    try:
        # Load data with selected features (assuming previous tasks completed)
        selected_features_path = "data/featured/selected_features_dataset.csv"
        
        if not Path(selected_features_path).exists():
            raise FileNotFoundError(
                f"Selected features file not found: {selected_features_path}. "
                "Please run Phase 4 Task 4.1 first."
            )
            
        df = pd.read_csv(selected_features_path, index_col=0)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        
        # Create quality assessor
        quality_assessor = Phase4DataQuality(df)
        
        # Generate quality report
        quality_report = quality_assessor.generate_quality_report()
        
        # Check model readiness
        is_ready, blocking_issues = quality_assessor.is_model_ready()
        
        # Save report
        quality_assessor.save_quality_report()
        
        # Print summary
        print(f"\n=== Phase 4 Task 5.1 Complete ===")
        print(f"Overall compliance: {quality_report['overall_compliance']}")
        print(f"Compliance rate: {quality_report['compliance_rate']:.1%}")
        print(f"Critical issues: {len(quality_report['critical_issues'])}")
        print(f"Model ready: {'YES' if is_ready else 'NO'}")
        
        if blocking_issues:
            print(f"\nBlocking issues:")
            for issue in blocking_issues:
                print(f"  - {issue}")
                
        if quality_report['recommendations']:
            print(f"\nRecommendations:")
            for rec in quality_report['recommendations'][:3]:
                print(f"  - {rec}")
        
        return quality_report
        
    except Exception as e:
        logger.error(f"Phase 4 Task 5.1 failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()