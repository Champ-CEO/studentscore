#!/usr/bin/env python3
"""
Imbalanced Data Analysis Module

Implements Phase 3.1.8: Imbalanced Data Analysis

This module analyzes class imbalance in categorical features and target variables,
provides strategies for handling imbalanced data, and implements various
resampling techniques to improve model performance.
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
from collections import Counter
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.preprocessing import LabelEncoder
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImbalancedDataAnalyzer:
    """
    Comprehensive imbalanced data analysis and handling.
    
    Provides analysis of class imbalance in categorical features,
    implements various resampling strategies, and evaluates the
    impact of imbalance on model performance.
    """
    
    def __init__(self, db_path: Optional[str] = None, data: Optional[pd.DataFrame] = None):
        """
        Initialize the Imbalanced Data Analyzer.
        
        Args:
            db_path: Path to SQLite database file
            data: Pre-loaded DataFrame (alternative to db_path)
        """
        self.db_path = db_path
        self.data = data
        self.imbalance_analysis = {}
        self.resampling_results = {}
        self.categorical_features = []
        self.numerical_features = []
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from database or use provided DataFrame.
        
        Returns:
            DataFrame with loaded data
        """
        if self.data is not None:
            return self.data.copy()
        
        if self.db_path is None:
            raise ValueError("Either db_path or data must be provided")
        
        try:
            conn = sqlite3.connect(self.db_path)
            query = "SELECT * FROM student_scores"
            data = pd.read_sql_query(query, conn)
            conn.close()
            
            logger.info(f"Loaded {len(data)} records from database")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def identify_feature_types(self, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Identify categorical and numerical features.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of (categorical_features, numerical_features)
        """
        categorical_features = []
        numerical_features = []
        
        for col in data.columns:
            if col == 'final_test':  # Skip target variable
                continue
                
            if data[col].dtype == 'object' or data[col].nunique() < 10:
                categorical_features.append(col)
            else:
                numerical_features.append(col)
        
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        
        logger.info(f"Identified {len(categorical_features)} categorical and {len(numerical_features)} numerical features")
        return categorical_features, numerical_features
    
    def analyze_categorical_imbalance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze class imbalance in categorical features.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary with imbalance analysis results
        """
        results = {}
        
        for feature in self.categorical_features:
            if feature in data.columns:
                value_counts = data[feature].value_counts()
                total_count = len(data[feature].dropna())
                
                # Calculate imbalance metrics
                proportions = value_counts / total_count
                imbalance_ratio = proportions.max() / proportions.min() if proportions.min() > 0 else float('inf')
                
                # Identify minority and majority classes
                minority_class = proportions.idxmin()
                majority_class = proportions.idxmax()
                
                # Calculate Gini coefficient for imbalance
                sorted_props = np.sort(proportions.values)
                n = len(sorted_props)
                gini = (2 * np.sum((np.arange(1, n + 1) * sorted_props))) / (n * np.sum(sorted_props)) - (n + 1) / n
                
                results[feature] = {
                    'value_counts': value_counts.to_dict(),
                    'proportions': proportions.to_dict(),
                    'imbalance_ratio': imbalance_ratio,
                    'minority_class': minority_class,
                    'majority_class': majority_class,
                    'minority_proportion': proportions.min(),
                    'majority_proportion': proportions.max(),
                    'gini_coefficient': gini,
                    'num_classes': len(value_counts),
                    'is_severely_imbalanced': imbalance_ratio > 10,
                    'is_moderately_imbalanced': 3 < imbalance_ratio <= 10
                }
        
        self.imbalance_analysis['categorical'] = results
        logger.info(f"Analyzed imbalance for {len(results)} categorical features")
        return results
    
    def analyze_target_distribution(self, data: pd.DataFrame, target_col: str = 'final_test') -> Dict[str, Any]:
        """
        Analyze target variable distribution for regression or classification.
        
        Args:
            data: Input DataFrame
            target_col: Name of target column
            
        Returns:
            Dictionary with target distribution analysis
        """
        if target_col not in data.columns:
            logger.warning(f"Target column '{target_col}' not found")
            return {}
        
        target_data = data[target_col].dropna()
        
        # Basic statistics
        stats = {
            'count': len(target_data),
            'mean': target_data.mean(),
            'median': target_data.median(),
            'std': target_data.std(),
            'min': target_data.min(),
            'max': target_data.max(),
            'skewness': target_data.skew(),
            'kurtosis': target_data.kurtosis()
        }
        
        # Create bins for distribution analysis
        n_bins = min(20, int(np.sqrt(len(target_data))))
        hist, bin_edges = np.histogram(target_data, bins=n_bins)
        
        # Calculate distribution imbalance
        bin_proportions = hist / len(target_data)
        max_bin_prop = bin_proportions.max()
        min_bin_prop = bin_proportions[bin_proportions > 0].min() if np.any(bin_proportions > 0) else 0
        
        distribution_imbalance = max_bin_prop / min_bin_prop if min_bin_prop > 0 else float('inf')
        
        results = {
            'statistics': stats,
            'histogram': {
                'counts': hist.tolist(),
                'bin_edges': bin_edges.tolist()
            },
            'distribution_imbalance': distribution_imbalance,
            'is_highly_skewed': abs(stats['skewness']) > 2,
            'is_moderately_skewed': 1 < abs(stats['skewness']) <= 2
        }
        
        self.imbalance_analysis['target'] = results
        logger.info(f"Analyzed target variable '{target_col}' distribution")
        return results
    
    def create_balanced_target_classes(self, data: pd.DataFrame, target_col: str = 'final_test', 
                                     n_classes: int = 3) -> pd.DataFrame:
        """
        Create balanced classes from continuous target for classification analysis.
        
        Args:
            data: Input DataFrame
            target_col: Name of target column
            n_classes: Number of classes to create
            
        Returns:
            DataFrame with added target class column
        """
        data_copy = data.copy()
        target_data = data_copy[target_col].dropna()
        
        # Create quantile-based classes
        quantiles = np.linspace(0, 1, n_classes + 1)
        class_boundaries = target_data.quantile(quantiles).values
        
        # Assign classes
        target_classes = pd.cut(data_copy[target_col], bins=class_boundaries, 
                               labels=[f'Class_{i}' for i in range(n_classes)], 
                               include_lowest=True)
        
        data_copy[f'{target_col}_class'] = target_classes
        
        # Analyze class distribution
        class_counts = target_classes.value_counts()
        class_proportions = class_counts / len(target_classes.dropna())
        
        class_analysis = {
            'class_boundaries': class_boundaries.tolist(),
            'class_counts': class_counts.to_dict(),
            'class_proportions': class_proportions.to_dict(),
            'imbalance_ratio': class_proportions.max() / class_proportions.min() if class_proportions.min() > 0 else float('inf')
        }
        
        self.imbalance_analysis['target_classes'] = class_analysis
        logger.info(f"Created {n_classes} target classes with imbalance ratio: {class_analysis['imbalance_ratio']:.2f}")
        
        return data_copy
    
    def apply_oversampling(self, X: pd.DataFrame, y: pd.Series, method: str = 'SMOTE') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply oversampling techniques to balance the dataset.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            method: Oversampling method ('SMOTE', 'ADASYN', 'BorderlineSMOTE')
            
        Returns:
            Tuple of (resampled_X, resampled_y)
        """
        # Encode categorical features for SMOTE
        X_encoded = X.copy()
        encoders = {}
        
        for col in X_encoded.columns:
            if X_encoded[col].dtype == 'object':
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                encoders[col] = le
        
        # Apply oversampling
        if method == 'SMOTE':
            sampler = SMOTE(random_state=42)
        elif method == 'ADASYN':
            sampler = ADASYN(random_state=42)
        elif method == 'BorderlineSMOTE':
            sampler = BorderlineSMOTE(random_state=42)
        else:
            raise ValueError(f"Unknown oversampling method: {method}")
        
        try:
            X_resampled, y_resampled = sampler.fit_resample(X_encoded, y)
            
            # Convert back to DataFrame
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            
            # Decode categorical features
            for col, encoder in encoders.items():
                X_resampled[col] = encoder.inverse_transform(X_resampled[col].astype(int))
            
            logger.info(f"Applied {method} oversampling: {len(X)} -> {len(X_resampled)} samples")
            return X_resampled, pd.Series(y_resampled, name=y.name)
            
        except Exception as e:
            logger.error(f"Error in {method} oversampling: {str(e)}")
            return X, y
    
    def apply_undersampling(self, X: pd.DataFrame, y: pd.Series, method: str = 'RandomUnderSampler') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply undersampling techniques to balance the dataset.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            method: Undersampling method ('RandomUnderSampler', 'TomekLinks', 'EditedNearestNeighbours')
            
        Returns:
            Tuple of (resampled_X, resampled_y)
        """
        # Encode categorical features
        X_encoded = X.copy()
        encoders = {}
        
        for col in X_encoded.columns:
            if X_encoded[col].dtype == 'object':
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                encoders[col] = le
        
        # Apply undersampling
        if method == 'RandomUnderSampler':
            sampler = RandomUnderSampler(random_state=42)
        elif method == 'TomekLinks':
            sampler = TomekLinks()
        elif method == 'EditedNearestNeighbours':
            sampler = EditedNearestNeighbours()
        else:
            raise ValueError(f"Unknown undersampling method: {method}")
        
        try:
            X_resampled, y_resampled = sampler.fit_resample(X_encoded, y)
            
            # Convert back to DataFrame
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            
            # Decode categorical features
            for col, encoder in encoders.items():
                X_resampled[col] = encoder.inverse_transform(X_resampled[col].astype(int))
            
            logger.info(f"Applied {method} undersampling: {len(X)} -> {len(X_resampled)} samples")
            return X_resampled, pd.Series(y_resampled, name=y.name)
            
        except Exception as e:
            logger.error(f"Error in {method} undersampling: {str(e)}")
            return X, y
    
    def apply_combined_sampling(self, X: pd.DataFrame, y: pd.Series, method: str = 'SMOTETomek') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply combined over/under sampling techniques.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            method: Combined sampling method ('SMOTETomek', 'SMOTEENN')
            
        Returns:
            Tuple of (resampled_X, resampled_y)
        """
        # Encode categorical features
        X_encoded = X.copy()
        encoders = {}
        
        for col in X_encoded.columns:
            if X_encoded[col].dtype == 'object':
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                encoders[col] = le
        
        # Apply combined sampling
        if method == 'SMOTETomek':
            sampler = SMOTETomek(random_state=42)
        elif method == 'SMOTEENN':
            sampler = SMOTEENN(random_state=42)
        else:
            raise ValueError(f"Unknown combined sampling method: {method}")
        
        try:
            X_resampled, y_resampled = sampler.fit_resample(X_encoded, y)
            
            # Convert back to DataFrame
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            
            # Decode categorical features
            for col, encoder in encoders.items():
                X_resampled[col] = encoder.inverse_transform(X_resampled[col].astype(int))
            
            logger.info(f"Applied {method} combined sampling: {len(X)} -> {len(X_resampled)} samples")
            return X_resampled, pd.Series(y_resampled, name=y.name)
            
        except Exception as e:
            logger.error(f"Error in {method} combined sampling: {str(e)}")
            return X, y
    
    def evaluate_resampling_strategies(self, data: pd.DataFrame, target_col: str = 'final_test') -> Dict[str, Any]:
        """
        Evaluate multiple resampling strategies.
        
        Args:
            data: Input DataFrame
            target_col: Name of target column
            
        Returns:
            Dictionary with evaluation results for each strategy
        """
        # Create target classes for evaluation
        data_with_classes = self.create_balanced_target_classes(data, target_col)
        target_class_col = f'{target_col}_class'
        
        # Prepare features and target
        feature_cols = [col for col in data.columns if col not in [target_col, target_class_col]]
        X = data_with_classes[feature_cols].copy()
        y = data_with_classes[target_class_col].dropna()
        X = X.loc[y.index]  # Align with non-null target values
        
        # Remove rows with missing values for resampling
        mask = ~X.isnull().any(axis=1)
        X_clean = X[mask]
        y_clean = y[mask]
        
        strategies = {
            'original': (X_clean, y_clean),
            'SMOTE': self.apply_oversampling(X_clean, y_clean, 'SMOTE'),
            'ADASYN': self.apply_oversampling(X_clean, y_clean, 'ADASYN'),
            'RandomUnderSampler': self.apply_undersampling(X_clean, y_clean, 'RandomUnderSampler'),
            'SMOTETomek': self.apply_combined_sampling(X_clean, y_clean, 'SMOTETomek')
        }
        
        results = {}
        
        for strategy_name, (X_res, y_res) in strategies.items():
            class_counts = y_res.value_counts()
            class_proportions = class_counts / len(y_res)
            
            results[strategy_name] = {
                'sample_count': len(y_res),
                'class_counts': class_counts.to_dict(),
                'class_proportions': class_proportions.to_dict(),
                'imbalance_ratio': class_proportions.max() / class_proportions.min() if class_proportions.min() > 0 else float('inf'),
                'balance_score': 1 - (class_proportions.max() - class_proportions.min())  # Higher is more balanced
            }
        
        self.resampling_results = results
        logger.info(f"Evaluated {len(strategies)} resampling strategies")
        return results
    
    def visualize_imbalance_analysis(self, save_dir: Optional[str] = None) -> None:
        """
        Create visualizations for imbalance analysis.
        
        Args:
            save_dir: Directory to save plots (optional)
        """
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        
        # Plot categorical feature imbalance
        if 'categorical' in self.imbalance_analysis:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Categorical Feature Imbalance Analysis', fontsize=16)
            
            # Imbalance ratios
            features = list(self.imbalance_analysis['categorical'].keys())
            ratios = [self.imbalance_analysis['categorical'][f]['imbalance_ratio'] for f in features]
            
            axes[0, 0].bar(features, ratios)
            axes[0, 0].set_title('Imbalance Ratios by Feature')
            axes[0, 0].set_ylabel('Imbalance Ratio')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Gini coefficients
            gini_coeffs = [self.imbalance_analysis['categorical'][f]['gini_coefficient'] for f in features]
            axes[0, 1].bar(features, gini_coeffs)
            axes[0, 1].set_title('Gini Coefficients by Feature')
            axes[0, 1].set_ylabel('Gini Coefficient')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Number of classes
            num_classes = [self.imbalance_analysis['categorical'][f]['num_classes'] for f in features]
            axes[1, 0].bar(features, num_classes)
            axes[1, 0].set_title('Number of Classes by Feature')
            axes[1, 0].set_ylabel('Number of Classes')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Minority proportions
            min_props = [self.imbalance_analysis['categorical'][f]['minority_proportion'] for f in features]
            axes[1, 1].bar(features, min_props)
            axes[1, 1].set_title('Minority Class Proportions')
            axes[1, 1].set_ylabel('Proportion')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(save_path / 'categorical_imbalance_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Plot resampling strategy comparison
        if self.resampling_results:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            strategies = list(self.resampling_results.keys())
            sample_counts = [self.resampling_results[s]['sample_count'] for s in strategies]
            balance_scores = [self.resampling_results[s]['balance_score'] for s in strategies]
            
            axes[0].bar(strategies, sample_counts)
            axes[0].set_title('Sample Counts by Resampling Strategy')
            axes[0].set_ylabel('Sample Count')
            axes[0].tick_params(axis='x', rotation=45)
            
            axes[1].bar(strategies, balance_scores)
            axes[1].set_title('Balance Scores by Resampling Strategy')
            axes[1].set_ylabel('Balance Score (Higher = More Balanced)')
            axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(save_path / 'resampling_strategy_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def generate_imbalance_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive imbalance analysis report.
        
        Args:
            output_path: Path to save the report (optional)
            
        Returns:
            Dictionary with complete analysis results
        """
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'imbalance_analysis': self.imbalance_analysis,
            'resampling_results': self.resampling_results,
            'recommendations': self._generate_recommendations()
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Imbalance analysis report saved to {output_path}")
        
        return report
    
    def _generate_recommendations(self) -> Dict[str, Any]:
        """
        Generate recommendations based on imbalance analysis.
        
        Returns:
            Dictionary with recommendations
        """
        recommendations = {
            'categorical_features': [],
            'target_variable': [],
            'resampling_strategy': [],
            'modeling_considerations': []
        }
        
        # Categorical feature recommendations
        if 'categorical' in self.imbalance_analysis:
            for feature, analysis in self.imbalance_analysis['categorical'].items():
                if analysis['is_severely_imbalanced']:
                    recommendations['categorical_features'].append({
                        'feature': feature,
                        'issue': 'Severe imbalance',
                        'recommendation': 'Consider grouping rare categories or using stratified sampling'
                    })
                elif analysis['is_moderately_imbalanced']:
                    recommendations['categorical_features'].append({
                        'feature': feature,
                        'issue': 'Moderate imbalance',
                        'recommendation': 'Monitor during model training, consider class weights'
                    })
        
        # Target variable recommendations
        if 'target' in self.imbalance_analysis:
            target_analysis = self.imbalance_analysis['target']
            if target_analysis['is_highly_skewed']:
                recommendations['target_variable'].append({
                    'issue': 'Highly skewed target distribution',
                    'recommendation': 'Consider log transformation or robust regression methods'
                })
        
        # Resampling strategy recommendations
        if self.resampling_results:
            best_strategy = max(self.resampling_results.keys(), 
                              key=lambda x: self.resampling_results[x]['balance_score'])
            recommendations['resampling_strategy'].append({
                'recommended_strategy': best_strategy,
                'reason': f"Highest balance score: {self.resampling_results[best_strategy]['balance_score']:.3f}"
            })
        
        # General modeling considerations
        recommendations['modeling_considerations'] = [
            'Use stratified sampling for train/validation splits',
            'Consider class weights in model training',
            'Use appropriate evaluation metrics (F1, AUC-ROC, precision-recall)',
            'Monitor for overfitting on minority classes',
            'Consider ensemble methods that handle imbalance well'
        ]
        
        return recommendations
    
    def run_complete_analysis(self, target_col: str = 'final_test', 
                            save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Run complete imbalanced data analysis pipeline.
        
        Args:
            target_col: Name of target column
            save_dir: Directory to save results and plots
            
        Returns:
            Dictionary with complete analysis results
        """
        logger.info("Starting complete imbalanced data analysis")
        
        # Load data
        data = self.load_data()
        
        # Identify feature types
        self.identify_feature_types(data)
        
        # Analyze categorical imbalance
        self.analyze_categorical_imbalance(data)
        
        # Analyze target distribution
        self.analyze_target_distribution(data, target_col)
        
        # Evaluate resampling strategies
        self.evaluate_resampling_strategies(data, target_col)
        
        # Generate visualizations
        self.visualize_imbalance_analysis(save_dir)
        
        # Generate report
        report_path = None
        if save_dir:
            report_path = Path(save_dir) / 'imbalanced_data_analysis_report.json'
        
        report = self.generate_imbalance_report(report_path)
        
        logger.info("Completed imbalanced data analysis")
        return report


def main():
    """
    Main function for testing the imbalanced data analysis.
    """
    # Example usage
    db_path = "data/raw/score.db"
    analyzer = ImbalancedDataAnalyzer(db_path=db_path)
    
    # Run complete analysis
    results = analyzer.run_complete_analysis(
        target_col='final_test',
        save_dir='data/processed/imbalance_analysis'
    )
    
    print("Imbalanced Data Analysis completed successfully!")
    print(f"Analysis results: {len(results)} components analyzed")


if __name__ == "__main__":
    main()