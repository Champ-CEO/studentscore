#!/usr/bin/env python3
"""
Robust Outlier Handling Module

Implements Phase 3.1.7: Implement robust outlier handling based on EDA findings

This module provides comprehensive outlier detection, analysis, and treatment
strategies based on the findings from exploratory data analysis.
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustOutlierHandler:
    """
    Comprehensive outlier detection and handling system.
    
    Provides multiple outlier detection methods:
    - Statistical methods (Z-score, IQR, Modified Z-score)
    - Machine learning methods (Isolation Forest, One-Class SVM)
    - Multivariate methods (Mahalanobis distance, Elliptic Envelope)
    - Domain-specific methods based on EDA findings
    
    Offers various treatment strategies:
    - Removal, capping, transformation, imputation
    """
    
    def __init__(self, db_path: Optional[str] = None, data: Optional[pd.DataFrame] = None):
        """
        Initialize the Robust Outlier Handler.
        
        Args:
            db_path: Path to SQLite database file
            data: Pre-loaded DataFrame (alternative to db_path)
        """
        self.db_path = db_path
        self.data = data
        self.outlier_results = {}
        self.treatment_history = []
        
        # Define field-specific outlier detection parameters based on EDA
        self.field_parameters = self._define_field_parameters()
        
    def _define_field_parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Define field-specific outlier detection parameters based on EDA findings.
        
        Returns:
            Dictionary with field-specific parameters
        """
        parameters = {
            'age': {
                'methods': ['iqr', 'z_score', 'domain_specific'],
                'domain_rules': {
                    'min_reasonable': 10,
                    'max_reasonable': 25,
                    'suspicious_threshold': 30
                },
                'treatment_strategy': 'cap_and_flag'
            },
            'hours_per_week': {
                'methods': ['iqr', 'isolation_forest', 'domain_specific'],
                'domain_rules': {
                    'max_possible': 168,  # Hours in a week
                    'max_reasonable': 80,  # Reasonable study hours
                    'suspicious_threshold': 100
                },
                'treatment_strategy': 'cap_and_flag'
            },
            'attendance_rate': {
                'methods': ['iqr', 'z_score'],
                'domain_rules': {
                    'min_possible': 0,
                    'max_possible': 100,
                    'suspicious_low': 10,
                    'suspicious_high': 100
                },
                'treatment_strategy': 'investigate_and_cap'
            },
            'final_test': {
                'methods': ['iqr', 'z_score', 'isolation_forest'],
                'domain_rules': {
                    'min_possible': 0,
                    'max_possible': 100
                },
                'treatment_strategy': 'flag_only'  # Don't modify test scores
            },
            'sleep_time': {
                'methods': ['domain_specific'],
                'domain_rules': {
                    'reasonable_range': ('20:00', '02:00'),  # 8 PM to 2 AM
                    'suspicious_early': '18:00',
                    'suspicious_late': '06:00'
                },
                'treatment_strategy': 'flag_and_investigate'
            },
            'wake_time': {
                'methods': ['domain_specific'],
                'domain_rules': {
                    'reasonable_range': ('05:00', '10:00'),  # 5 AM to 10 AM
                    'suspicious_early': '03:00',
                    'suspicious_late': '12:00'
                },
                'treatment_strategy': 'flag_and_investigate'
            }
        }
        
        return parameters
    
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
            raise ValueError("Either db_path or data must be provided")
        
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
    
    def detect_all_outliers(self) -> Dict[str, Any]:
        """
        Detect outliers using all available methods.
        
        Returns:
            Dictionary with comprehensive outlier detection results
        """
        if self.data is None:
            self.data = self.load_data()
        
        logger.info("Starting comprehensive outlier detection")
        
        outlier_results = {
            'univariate_outliers': {},
            'multivariate_outliers': {},
            'domain_specific_outliers': {},
            'outlier_summary': {},
            'recommended_actions': {}
        }
        
        # Detect univariate outliers
        outlier_results['univariate_outliers'] = self._detect_univariate_outliers()
        
        # Detect multivariate outliers
        outlier_results['multivariate_outliers'] = self._detect_multivariate_outliers()
        
        # Detect domain-specific outliers
        outlier_results['domain_specific_outliers'] = self._detect_domain_specific_outliers()
        
        # Generate summary and recommendations
        outlier_results['outlier_summary'] = self._generate_outlier_summary(outlier_results)
        outlier_results['recommended_actions'] = self._generate_recommendations(outlier_results)
        
        self.outlier_results = outlier_results
        logger.info("Outlier detection completed")
        
        return outlier_results
    
    def _detect_univariate_outliers(self) -> Dict[str, Any]:
        """
        Detect univariate outliers using statistical methods.
        
        Returns:
            Dictionary with univariate outlier results
        """
        results = {}
        
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column in self.field_parameters:
                methods = self.field_parameters[column]['methods']
            else:
                methods = ['iqr', 'z_score']  # Default methods
            
            column_results = {
                'total_values': len(self.data[column].dropna()),
                'methods': {}
            }
            
            column_data = self.data[column].dropna()
            
            if len(column_data) == 0:
                continue
            
            # IQR Method
            if 'iqr' in methods:
                iqr_outliers = self._detect_iqr_outliers(column_data)
                column_results['methods']['iqr'] = iqr_outliers
            
            # Z-Score Method
            if 'z_score' in methods:
                z_score_outliers = self._detect_z_score_outliers(column_data)
                column_results['methods']['z_score'] = z_score_outliers
            
            # Modified Z-Score Method
            if 'modified_z_score' in methods:
                mod_z_outliers = self._detect_modified_z_score_outliers(column_data)
                column_results['methods']['modified_z_score'] = mod_z_outliers
            
            # Isolation Forest Method
            if 'isolation_forest' in methods:
                isolation_outliers = self._detect_isolation_forest_outliers(column_data)
                column_results['methods']['isolation_forest'] = isolation_outliers
            
            # Combine results from all methods
            column_results['consensus_outliers'] = self._combine_outlier_methods(column_results['methods'])
            
            results[column] = column_results
        
        return results
    
    def _detect_iqr_outliers(self, data: pd.Series) -> Dict[str, Any]:
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Args:
            data: Series of numeric data
            
        Returns:
            Dictionary with IQR outlier results
        """
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        outlier_indices = data[outlier_mask].index.tolist()
        outlier_values = data[outlier_mask].tolist()
        
        return {
            'method': 'IQR',
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_count': len(outlier_indices),
            'outlier_percentage': (len(outlier_indices) / len(data)) * 100,
            'outlier_indices': outlier_indices,
            'outlier_values': outlier_values,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR
        }
    
    def _detect_z_score_outliers(self, data: pd.Series, threshold: float = 3.0) -> Dict[str, Any]:
        """
        Detect outliers using Z-score method.
        
        Args:
            data: Series of numeric data
            threshold: Z-score threshold for outlier detection
            
        Returns:
            Dictionary with Z-score outlier results
        """
        z_scores = np.abs(stats.zscore(data))
        outlier_mask = z_scores > threshold
        
        outlier_indices = data[outlier_mask].index.tolist()
        outlier_values = data[outlier_mask].tolist()
        outlier_z_scores = z_scores[outlier_mask].tolist()
        
        return {
            'method': 'Z-Score',
            'threshold': threshold,
            'outlier_count': len(outlier_indices),
            'outlier_percentage': (len(outlier_indices) / len(data)) * 100,
            'outlier_indices': outlier_indices,
            'outlier_values': outlier_values,
            'outlier_z_scores': outlier_z_scores,
            'mean': data.mean(),
            'std': data.std()
        }
    
    def _detect_modified_z_score_outliers(self, data: pd.Series, threshold: float = 3.5) -> Dict[str, Any]:
        """
        Detect outliers using Modified Z-score method (using median).
        
        Args:
            data: Series of numeric data
            threshold: Modified Z-score threshold
            
        Returns:
            Dictionary with Modified Z-score outlier results
        """
        median = data.median()
        mad = np.median(np.abs(data - median))
        
        # Avoid division by zero
        if mad == 0:
            mad = np.mean(np.abs(data - median))
        
        modified_z_scores = 0.6745 * (data - median) / mad
        outlier_mask = np.abs(modified_z_scores) > threshold
        
        outlier_indices = data[outlier_mask].index.tolist()
        outlier_values = data[outlier_mask].tolist()
        outlier_scores = modified_z_scores[outlier_mask].tolist()
        
        return {
            'method': 'Modified Z-Score',
            'threshold': threshold,
            'outlier_count': len(outlier_indices),
            'outlier_percentage': (len(outlier_indices) / len(data)) * 100,
            'outlier_indices': outlier_indices,
            'outlier_values': outlier_values,
            'outlier_scores': outlier_scores,
            'median': median,
            'mad': mad
        }
    
    def _detect_isolation_forest_outliers(self, data: pd.Series, contamination: float = 0.1) -> Dict[str, Any]:
        """
        Detect outliers using Isolation Forest method.
        
        Args:
            data: Series of numeric data
            contamination: Expected proportion of outliers
            
        Returns:
            Dictionary with Isolation Forest outlier results
        """
        # Reshape data for sklearn
        data_reshaped = data.values.reshape(-1, 1)
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(data_reshaped)
        
        # Get outlier scores
        outlier_scores = iso_forest.decision_function(data_reshaped)
        
        # Identify outliers (labeled as -1)
        outlier_mask = outlier_labels == -1
        outlier_indices = data[outlier_mask].index.tolist()
        outlier_values = data[outlier_mask].tolist()
        outlier_score_values = outlier_scores[outlier_mask].tolist()
        
        return {
            'method': 'Isolation Forest',
            'contamination': contamination,
            'outlier_count': len(outlier_indices),
            'outlier_percentage': (len(outlier_indices) / len(data)) * 100,
            'outlier_indices': outlier_indices,
            'outlier_values': outlier_values,
            'outlier_scores': outlier_score_values
        }
    
    def _detect_multivariate_outliers(self) -> Dict[str, Any]:
        """
        Detect multivariate outliers using multiple methods.
        
        Returns:
            Dictionary with multivariate outlier results
        """
        results = {}
        
        # Select numeric columns for multivariate analysis
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) < 2:
            logger.warning("Not enough numeric columns for multivariate outlier detection")
            return results
        
        # Prepare data (remove missing values)
        clean_data = self.data[numeric_columns].dropna()
        
        if len(clean_data) < 10:
            logger.warning("Not enough clean data for multivariate outlier detection")
            return results
        
        # Mahalanobis Distance
        try:
            mahalanobis_results = self._detect_mahalanobis_outliers(clean_data)
            results['mahalanobis'] = mahalanobis_results
        except Exception as e:
            logger.warning(f"Mahalanobis distance calculation failed: {e}")
        
        # Elliptic Envelope
        try:
            elliptic_results = self._detect_elliptic_envelope_outliers(clean_data)
            results['elliptic_envelope'] = elliptic_results
        except Exception as e:
            logger.warning(f"Elliptic Envelope calculation failed: {e}")
        
        # Multivariate Isolation Forest
        try:
            multivar_isolation_results = self._detect_multivariate_isolation_outliers(clean_data)
            results['multivariate_isolation_forest'] = multivar_isolation_results
        except Exception as e:
            logger.warning(f"Multivariate Isolation Forest failed: {e}")
        
        return results
    
    def _detect_mahalanobis_outliers(self, data: pd.DataFrame, threshold: float = 3.0) -> Dict[str, Any]:
        """
        Detect outliers using Mahalanobis distance.
        
        Args:
            data: DataFrame with numeric data
            threshold: Threshold for outlier detection
            
        Returns:
            Dictionary with Mahalanobis outlier results
        """
        # Calculate covariance matrix
        cov_matrix = np.cov(data.T)
        
        # Calculate inverse covariance matrix
        try:
            inv_cov_matrix = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if matrix is singular
            inv_cov_matrix = np.linalg.pinv(cov_matrix)
        
        # Calculate mean
        mean = data.mean()
        
        # Calculate Mahalanobis distances
        mahal_distances = []
        for idx, row in data.iterrows():
            diff = row - mean
            mahal_dist = np.sqrt(diff.T @ inv_cov_matrix @ diff)
            mahal_distances.append(mahal_dist)
        
        mahal_distances = np.array(mahal_distances)
        
        # Identify outliers
        outlier_mask = mahal_distances > threshold
        outlier_indices = data[outlier_mask].index.tolist()
        outlier_distances = mahal_distances[outlier_mask].tolist()
        
        return {
            'method': 'Mahalanobis Distance',
            'threshold': threshold,
            'outlier_count': len(outlier_indices),
            'outlier_percentage': (len(outlier_indices) / len(data)) * 100,
            'outlier_indices': outlier_indices,
            'outlier_distances': outlier_distances,
            'mean_distance': np.mean(mahal_distances),
            'std_distance': np.std(mahal_distances)
        }
    
    def _detect_elliptic_envelope_outliers(self, data: pd.DataFrame, contamination: float = 0.1) -> Dict[str, Any]:
        """
        Detect outliers using Elliptic Envelope method.
        
        Args:
            data: DataFrame with numeric data
            contamination: Expected proportion of outliers
            
        Returns:
            Dictionary with Elliptic Envelope outlier results
        """
        # Standardize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Fit Elliptic Envelope
        elliptic_env = EllipticEnvelope(contamination=contamination, random_state=42)
        outlier_labels = elliptic_env.fit_predict(data_scaled)
        
        # Get outlier scores
        outlier_scores = elliptic_env.decision_function(data_scaled)
        
        # Identify outliers (labeled as -1)
        outlier_mask = outlier_labels == -1
        outlier_indices = data[outlier_mask].index.tolist()
        outlier_score_values = outlier_scores[outlier_mask].tolist()
        
        return {
            'method': 'Elliptic Envelope',
            'contamination': contamination,
            'outlier_count': len(outlier_indices),
            'outlier_percentage': (len(outlier_indices) / len(data)) * 100,
            'outlier_indices': outlier_indices,
            'outlier_scores': outlier_score_values
        }
    
    def _detect_multivariate_isolation_outliers(self, data: pd.DataFrame, contamination: float = 0.1) -> Dict[str, Any]:
        """
        Detect multivariate outliers using Isolation Forest.
        
        Args:
            data: DataFrame with numeric data
            contamination: Expected proportion of outliers
            
        Returns:
            Dictionary with multivariate Isolation Forest results
        """
        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(data)
        
        # Get outlier scores
        outlier_scores = iso_forest.decision_function(data)
        
        # Identify outliers (labeled as -1)
        outlier_mask = outlier_labels == -1
        outlier_indices = data[outlier_mask].index.tolist()
        outlier_score_values = outlier_scores[outlier_mask].tolist()
        
        return {
            'method': 'Multivariate Isolation Forest',
            'contamination': contamination,
            'outlier_count': len(outlier_indices),
            'outlier_percentage': (len(outlier_indices) / len(data)) * 100,
            'outlier_indices': outlier_indices,
            'outlier_scores': outlier_score_values
        }
    
    def _detect_domain_specific_outliers(self) -> Dict[str, Any]:
        """
        Detect domain-specific outliers based on business rules.
        
        Returns:
            Dictionary with domain-specific outlier results
        """
        results = {}
        
        for field, params in self.field_parameters.items():
            if field not in self.data.columns:
                continue
            
            if 'domain_specific' not in params['methods']:
                continue
            
            domain_rules = params['domain_rules']
            field_data = self.data[field].dropna()
            
            if len(field_data) == 0:
                continue
            
            outliers = []
            
            if field in ['age', 'hours_per_week', 'attendance_rate', 'final_test']:
                # Numeric domain rules
                for idx, value in field_data.items():
                    numeric_value = pd.to_numeric(value, errors='coerce')
                    
                    if pd.isna(numeric_value):
                        continue
                    
                    # Check against domain rules
                    if 'min_reasonable' in domain_rules and numeric_value < domain_rules['min_reasonable']:
                        outliers.append({
                            'index': idx,
                            'value': numeric_value,
                            'rule_violated': 'below_minimum_reasonable',
                            'threshold': domain_rules['min_reasonable']
                        })
                    
                    if 'max_reasonable' in domain_rules and numeric_value > domain_rules['max_reasonable']:
                        outliers.append({
                            'index': idx,
                            'value': numeric_value,
                            'rule_violated': 'above_maximum_reasonable',
                            'threshold': domain_rules['max_reasonable']
                        })
                    
                    if 'max_possible' in domain_rules and numeric_value > domain_rules['max_possible']:
                        outliers.append({
                            'index': idx,
                            'value': numeric_value,
                            'rule_violated': 'above_maximum_possible',
                            'threshold': domain_rules['max_possible']
                        })
                    
                    if 'min_possible' in domain_rules and numeric_value < domain_rules['min_possible']:
                        outliers.append({
                            'index': idx,
                            'value': numeric_value,
                            'rule_violated': 'below_minimum_possible',
                            'threshold': domain_rules['min_possible']
                        })
            
            elif field in ['sleep_time', 'wake_time']:
                # Time domain rules
                for idx, value in field_data.items():
                    time_str = str(value)
                    
                    try:
                        # Parse time
                        from datetime import datetime
                        time_obj = datetime.strptime(time_str, '%H:%M').time()
                        
                        # Check against domain rules
                        if field == 'sleep_time':
                            if 'suspicious_early' in domain_rules:
                                early_time = datetime.strptime(domain_rules['suspicious_early'], '%H:%M').time()
                                if time_obj < early_time:
                                    outliers.append({
                                        'index': idx,
                                        'value': time_str,
                                        'rule_violated': 'suspiciously_early_sleep',
                                        'threshold': domain_rules['suspicious_early']
                                    })
                            
                            if 'suspicious_late' in domain_rules:
                                late_time = datetime.strptime(domain_rules['suspicious_late'], '%H:%M').time()
                                if time_obj > late_time:
                                    outliers.append({
                                        'index': idx,
                                        'value': time_str,
                                        'rule_violated': 'suspiciously_late_sleep',
                                        'threshold': domain_rules['suspicious_late']
                                    })
                        
                        elif field == 'wake_time':
                            if 'suspicious_early' in domain_rules:
                                early_time = datetime.strptime(domain_rules['suspicious_early'], '%H:%M').time()
                                if time_obj < early_time:
                                    outliers.append({
                                        'index': idx,
                                        'value': time_str,
                                        'rule_violated': 'suspiciously_early_wake',
                                        'threshold': domain_rules['suspicious_early']
                                    })
                            
                            if 'suspicious_late' in domain_rules:
                                late_time = datetime.strptime(domain_rules['suspicious_late'], '%H:%M').time()
                                if time_obj > late_time:
                                    outliers.append({
                                        'index': idx,
                                        'value': time_str,
                                        'rule_violated': 'suspiciously_late_wake',
                                        'threshold': domain_rules['suspicious_late']
                                    })
                    
                    except ValueError:
                        # Invalid time format
                        outliers.append({
                            'index': idx,
                            'value': time_str,
                            'rule_violated': 'invalid_time_format',
                            'threshold': 'HH:MM format expected'
                        })
            
            if outliers:
                results[field] = {
                    'method': 'Domain-Specific Rules',
                    'outlier_count': len(outliers),
                    'outlier_percentage': (len(outliers) / len(field_data)) * 100,
                    'outliers': outliers
                }
        
        return results
    
    def _combine_outlier_methods(self, method_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine results from multiple outlier detection methods.
        
        Args:
            method_results: Dictionary with results from different methods
            
        Returns:
            Dictionary with consensus outlier results
        """
        all_outlier_indices = set()
        method_votes = {}
        
        # Collect all outlier indices and count votes
        for method, results in method_results.items():
            outlier_indices = results.get('outlier_indices', [])
            all_outlier_indices.update(outlier_indices)
            
            for idx in outlier_indices:
                if idx not in method_votes:
                    method_votes[idx] = []
                method_votes[idx].append(method)
        
        # Determine consensus based on number of methods agreeing
        total_methods = len(method_results)
        consensus_outliers = []
        
        for idx in all_outlier_indices:
            vote_count = len(method_votes[idx])
            consensus_level = vote_count / total_methods
            
            consensus_outliers.append({
                'index': idx,
                'vote_count': vote_count,
                'total_methods': total_methods,
                'consensus_level': consensus_level,
                'agreeing_methods': method_votes[idx]
            })
        
        # Sort by consensus level
        consensus_outliers.sort(key=lambda x: x['consensus_level'], reverse=True)
        
        # Categorize by consensus strength
        strong_consensus = [o for o in consensus_outliers if o['consensus_level'] >= 0.75]
        moderate_consensus = [o for o in consensus_outliers if 0.5 <= o['consensus_level'] < 0.75]
        weak_consensus = [o for o in consensus_outliers if o['consensus_level'] < 0.5]
        
        return {
            'total_outliers': len(consensus_outliers),
            'strong_consensus': strong_consensus,
            'moderate_consensus': moderate_consensus,
            'weak_consensus': weak_consensus,
            'consensus_summary': {
                'strong_count': len(strong_consensus),
                'moderate_count': len(moderate_consensus),
                'weak_count': len(weak_consensus)
            }
        }
    
    def _generate_outlier_summary(self, outlier_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary of outlier detection results.
        
        Args:
            outlier_results: Complete outlier detection results
            
        Returns:
            Dictionary with outlier summary
        """
        summary = {
            'total_records': len(self.data),
            'fields_analyzed': [],
            'outlier_counts_by_field': {},
            'outlier_counts_by_method': {},
            'most_problematic_records': [],
            'field_outlier_rates': {}
        }
        
        # Summarize univariate outliers
        for field, field_results in outlier_results['univariate_outliers'].items():
            summary['fields_analyzed'].append(field)
            
            consensus_outliers = field_results['consensus_outliers']
            total_outliers = consensus_outliers['total_outliers']
            
            summary['outlier_counts_by_field'][field] = total_outliers
            summary['field_outlier_rates'][field] = (total_outliers / field_results['total_values']) * 100
            
            # Count by method
            for method, method_results in field_results['methods'].items():
                if method not in summary['outlier_counts_by_method']:
                    summary['outlier_counts_by_method'][method] = 0
                summary['outlier_counts_by_method'][method] += method_results['outlier_count']
        
        # Find most problematic records (appearing as outliers in multiple fields)
        record_outlier_counts = {}
        
        for field, field_results in outlier_results['univariate_outliers'].items():
            consensus_outliers = field_results['consensus_outliers']
            for outlier in consensus_outliers['strong_consensus'] + consensus_outliers['moderate_consensus']:
                idx = outlier['index']
                if idx not in record_outlier_counts:
                    record_outlier_counts[idx] = {'count': 0, 'fields': []}
                record_outlier_counts[idx]['count'] += 1
                record_outlier_counts[idx]['fields'].append(field)
        
        # Sort by outlier count
        most_problematic = sorted(
            [(idx, info) for idx, info in record_outlier_counts.items()],
            key=lambda x: x[1]['count'],
            reverse=True
        )[:10]  # Top 10
        
        summary['most_problematic_records'] = [
            {
                'index': idx,
                'outlier_field_count': info['count'],
                'outlier_fields': info['fields']
            }
            for idx, info in most_problematic
        ]
        
        return summary
    
    def _generate_recommendations(self, outlier_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate treatment recommendations based on outlier detection results.
        
        Args:
            outlier_results: Complete outlier detection results
            
        Returns:
            Dictionary with treatment recommendations
        """
        recommendations = {
            'immediate_actions': [],
            'field_specific_actions': {},
            'investigation_priorities': [],
            'treatment_strategies': {}
        }
        
        # Generate field-specific recommendations
        for field, field_results in outlier_results['univariate_outliers'].items():
            consensus_outliers = field_results['consensus_outliers']
            strong_outliers = len(consensus_outliers['strong_consensus'])
            
            if strong_outliers > 0:
                treatment_strategy = self.field_parameters.get(field, {}).get('treatment_strategy', 'investigate')
                
                field_recommendations = {
                    'outlier_count': strong_outliers,
                    'treatment_strategy': treatment_strategy,
                    'specific_actions': []
                }
                
                if treatment_strategy == 'cap_and_flag':
                    field_recommendations['specific_actions'] = [
                        'Cap extreme values to reasonable limits',
                        'Flag capped records for manual review',
                        'Document transformation for reproducibility'
                    ]
                elif treatment_strategy == 'flag_only':
                    field_recommendations['specific_actions'] = [
                        'Flag outliers for investigation',
                        'Do not modify values automatically',
                        'Investigate data collection process'
                    ]
                elif treatment_strategy == 'investigate_and_cap':
                    field_recommendations['specific_actions'] = [
                        'Investigate outliers for data entry errors',
                        'Cap values only after investigation',
                        'Consider separate analysis for outliers'
                    ]
                elif treatment_strategy == 'flag_and_investigate':
                    field_recommendations['specific_actions'] = [
                        'Flag for manual investigation',
                        'Check for data entry errors',
                        'Validate against source documents'
                    ]
                
                recommendations['field_specific_actions'][field] = field_recommendations
        
        # Generate immediate actions
        summary = outlier_results['outlier_summary']
        
        if summary['most_problematic_records']:
            recommendations['immediate_actions'].append(
                f"Investigate {len(summary['most_problematic_records'])} records with multiple outlier fields"
            )
        
        # High outlier rate fields
        high_rate_fields = [
            field for field, rate in summary['field_outlier_rates'].items()
            if rate > 10  # More than 10% outliers
        ]
        
        if high_rate_fields:
            recommendations['immediate_actions'].append(
                f"Review data collection process for fields: {', '.join(high_rate_fields)}"
            )
        
        # Investigation priorities
        priorities = sorted(
            summary['field_outlier_rates'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]  # Top 5 fields by outlier rate
        
        recommendations['investigation_priorities'] = [
            {
                'field': field,
                'outlier_rate': rate,
                'priority': 'High' if rate > 15 else 'Medium' if rate > 5 else 'Low'
            }
            for field, rate in priorities
        ]
        
        return recommendations
    
    def apply_outlier_treatment(self, treatment_plan: Dict[str, str]) -> pd.DataFrame:
        """
        Apply outlier treatment based on specified plan.
        
        Args:
            treatment_plan: Dictionary mapping field names to treatment strategies
            
        Returns:
            DataFrame with outlier treatment applied
        """
        if self.data is None:
            raise ValueError("No data loaded")
        
        if not self.outlier_results:
            raise ValueError("No outlier detection results available. Run detect_all_outliers() first.")
        
        treated_data = self.data.copy()
        treatment_log = []
        
        for field, strategy in treatment_plan.items():
            if field not in self.outlier_results['univariate_outliers']:
                logger.warning(f"No outlier results available for field: {field}")
                continue
            
            field_results = self.outlier_results['univariate_outliers'][field]
            consensus_outliers = field_results['consensus_outliers']
            
            # Get strong consensus outliers for treatment
            outlier_indices = [o['index'] for o in consensus_outliers['strong_consensus']]
            
            if not outlier_indices:
                continue
            
            original_values = treated_data.loc[outlier_indices, field].copy()
            
            if strategy == 'remove':
                treated_data = treated_data.drop(outlier_indices)
                treatment_log.append({
                    'field': field,
                    'strategy': strategy,
                    'affected_records': len(outlier_indices),
                    'indices': outlier_indices
                })
            
            elif strategy == 'cap':
                # Cap to IQR bounds
                iqr_results = field_results['methods'].get('iqr')
                if iqr_results:
                    lower_bound = iqr_results['lower_bound']
                    upper_bound = iqr_results['upper_bound']
                    
                    treated_data.loc[outlier_indices, field] = treated_data.loc[outlier_indices, field].clip(
                        lower=lower_bound, upper=upper_bound
                    )
                    
                    treatment_log.append({
                        'field': field,
                        'strategy': strategy,
                        'affected_records': len(outlier_indices),
                        'bounds': {'lower': lower_bound, 'upper': upper_bound},
                        'indices': outlier_indices,
                        'original_values': original_values.to_dict()
                    })
            
            elif strategy == 'transform':
                # Apply log transformation to reduce outlier impact
                if treated_data[field].min() > 0:
                    treated_data[field] = np.log1p(treated_data[field])
                    treatment_log.append({
                        'field': field,
                        'strategy': 'log_transform',
                        'affected_records': len(treated_data),
                        'note': 'Applied log1p transformation to entire field'
                    })
            
            elif strategy == 'flag':
                # Add outlier flag column
                flag_column = f'{field}_outlier_flag'
                treated_data[flag_column] = False
                treated_data.loc[outlier_indices, flag_column] = True
                
                treatment_log.append({
                    'field': field,
                    'strategy': strategy,
                    'affected_records': len(outlier_indices),
                    'flag_column': flag_column,
                    'indices': outlier_indices
                })
        
        # Store treatment history
        self.treatment_history.append({
            'timestamp': pd.Timestamp.now(),
            'treatment_plan': treatment_plan,
            'treatment_log': treatment_log,
            'original_shape': self.data.shape,
            'treated_shape': treated_data.shape
        })
        
        logger.info(f"Outlier treatment completed. Shape changed from {self.data.shape} to {treated_data.shape}")
        
        return treated_data
    
    def visualize_outliers(self, field: str, output_dir: str) -> None:
        """
        Create visualizations for outlier analysis.
        
        Args:
            field: Field name to visualize
            output_dir: Directory to save plots
        """
        if field not in self.outlier_results['univariate_outliers']:
            logger.error(f"No outlier results available for field: {field}")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        field_data = self.data[field].dropna()
        field_results = self.outlier_results['univariate_outliers'][field]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Outlier Analysis for {field}', fontsize=16)
        
        # Box plot
        axes[0, 0].boxplot(field_data)
        axes[0, 0].set_title('Box Plot')
        axes[0, 0].set_ylabel(field)
        
        # Histogram
        axes[0, 1].hist(field_data, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Distribution')
        axes[0, 1].set_xlabel(field)
        axes[0, 1].set_ylabel('Frequency')
        
        # Scatter plot with outliers highlighted
        outlier_indices = []
        for method_results in field_results['methods'].values():
            outlier_indices.extend(method_results['outlier_indices'])
        outlier_indices = list(set(outlier_indices))
        
        normal_indices = [idx for idx in field_data.index if idx not in outlier_indices]
        
        axes[1, 0].scatter(normal_indices, field_data[normal_indices], alpha=0.6, label='Normal')
        if outlier_indices:
            axes[1, 0].scatter(outlier_indices, field_data[outlier_indices], 
                             color='red', alpha=0.8, label='Outliers')
        axes[1, 0].set_title('Data Points with Outliers Highlighted')
        axes[1, 0].set_xlabel('Index')
        axes[1, 0].set_ylabel(field)
        axes[1, 0].legend()
        
        # Method comparison
        method_names = list(field_results['methods'].keys())
        outlier_counts = [field_results['methods'][method]['outlier_count'] for method in method_names]
        
        axes[1, 1].bar(method_names, outlier_counts)
        axes[1, 1].set_title('Outliers Detected by Method')
        axes[1, 1].set_xlabel('Detection Method')
        axes[1, 1].set_ylabel('Number of Outliers')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = output_path / f'{field}_outlier_analysis.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Outlier visualization saved to {plot_file}")
    
    def save_outlier_results(self, output_dir: str) -> None:
        """
        Save outlier detection results to files.
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save outlier results
        results_file = output_path / 'outlier_detection_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.outlier_results, f, indent=2, default=str)
        
        # Save treatment history
        if self.treatment_history:
            treatment_file = output_path / 'outlier_treatment_history.json'
            with open(treatment_file, 'w') as f:
                json.dump(self.treatment_history, f, indent=2, default=str)
        
        logger.info(f"Outlier results saved to {output_path}")
    
    def generate_outlier_report(self) -> str:
        """
        Generate a comprehensive outlier analysis report.
        
        Returns:
            Outlier analysis report string
        """
        if not self.outlier_results:
            return "No outlier results available. Run detect_all_outliers() first."
        
        report = []
        report.append("=== Comprehensive Outlier Analysis Report ===")
        report.append("")
        
        summary = self.outlier_results['outlier_summary']
        
        # Overall summary
        report.append(f"Total Records Analyzed: {summary['total_records']}")
        report.append(f"Fields Analyzed: {len(summary['fields_analyzed'])}")
        report.append("")
        
        # Field-wise outlier rates
        report.append("## Outlier Rates by Field")
        for field, rate in sorted(summary['field_outlier_rates'].items(), key=lambda x: x[1], reverse=True):
            count = summary['outlier_counts_by_field'][field]
            report.append(f"- {field}: {count} outliers ({rate:.1f}%)")
        report.append("")
        
        # Method comparison
        report.append("## Detection Method Performance")
        for method, count in summary['outlier_counts_by_method'].items():
            report.append(f"- {method}: {count} outliers detected")
        report.append("")
        
        # Most problematic records
        if summary['most_problematic_records']:
            report.append("## Most Problematic Records")
            for record in summary['most_problematic_records'][:5]:  # Top 5
                report.append(f"- Record {record['index']}: outlier in {record['outlier_field_count']} fields ({', '.join(record['outlier_fields'])})")
            report.append("")
        
        # Recommendations
        recommendations = self.outlier_results['recommended_actions']
        
        if recommendations['immediate_actions']:
            report.append("## Immediate Actions Required")
            for action in recommendations['immediate_actions']:
                report.append(f"- {action}")
            report.append("")
        
        if recommendations['investigation_priorities']:
            report.append("## Investigation Priorities")
            for priority in recommendations['investigation_priorities']:
                report.append(f"- {priority['field']}: {priority['outlier_rate']:.1f}% outliers ({priority['priority']} priority)")
            report.append("")
        
        return "\n".join(report)


def main():
    """
    Main function for testing the Robust Outlier Handler.
    """
    # Example usage
    db_path = "data/raw/score.db"
    
    handler = RobustOutlierHandler(db_path=db_path)
    
    # Detect outliers
    results = handler.detect_all_outliers()
    print("Outlier detection completed")
    
    # Generate report
    report = handler.generate_outlier_report()
    print(report)
    
    # Save results
    handler.save_outlier_results("data/processed")
    print("Results saved")
    
    # Example treatment
    treatment_plan = {
        'age': 'cap',
        'hours_per_week': 'cap',
        'final_test': 'flag'
    }
    
    treated_data = handler.apply_outlier_treatment(treatment_plan)
    print(f"Treatment applied. New shape: {treated_data.shape}")


if __name__ == "__main__":
    main()