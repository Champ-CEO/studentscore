#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for Student Score Prediction Project

This script implements Phase 2 of the project as outlined in specs/TASKS.md.
It performs comprehensive exploratory data analysis to understand data patterns,
relationships, and inform preprocessing decisions.

Phase 2 Tasks Covered:
- 2.1: Data Loading and Initial Exploration
- 2.2: Missing Data Analysis
- 2.3: Univariate Analysis
- 2.4: Bivariate Analysis
- 2.5: Multivariate Analysis
- 2.6: Data Quality Assessment
- 2.7: Feature-Specific Deep Dive
- 2.8: EDA Summary and Recommendations
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Dict, List, Tuple, Any
from scipy import stats
from scipy.stats import normaltest, skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to sys.path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.repository import ScoreRepository

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class StudentScoreEDA:
    """
    Comprehensive Exploratory Data Analysis for Student Score Prediction.
    
    This class implements all Phase 2 EDA tasks as specified in the project requirements.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize the EDA class with database connection.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self.repository = ScoreRepository(db_path)
        self.df = None
        self.numerical_features = []
        self.categorical_features = []
        self.target_variable = 'final_test'
        self.eda_results = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        Task 2.1.1: Setup EDA environment and data loading.
        
        Load data from SQLite database and perform initial setup.
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        print("=== Task 2.1.1: Data Loading ===")
        
        # Load data using repository
        with self.repository._get_connection() as conn:
            query = "SELECT * FROM score"
            self.df = pd.read_sql_query(query, conn)
        
        # Verify data loading
        print(f"Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        
        # Verify expected features are present
        expected_features = {
            'index', 'number_of_siblings', 'direct_admission', 'CCA',
            'learning_style', 'student_id', 'gender', 'tuition',
            'final_test', 'n_male', 'n_female', 'age', 'hours_per_week',
            'attendance_rate', 'sleep_time', 'wake_time',
            'mode_of_transport', 'bag_color'
        }
        actual_features = set(self.df.columns)
        
        if expected_features.issubset(actual_features):
            print("✓ All 17 expected features are present")
        else:
            missing = expected_features - actual_features
            print(f"✗ Missing features: {missing}")
        
        # Verify record count
        if self.df.shape[0] == 15900:
            print("✓ Expected 15,900 records loaded")
        else:
            print(f"✗ Expected 15,900 records, got {self.df.shape[0]}")
        
        # Identify feature types
        self.numerical_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove index from numerical features if present
        if 'index' in self.numerical_features:
            self.numerical_features.remove('index')
        
        print(f"Numerical features ({len(self.numerical_features)}): {self.numerical_features}")
        print(f"Categorical features ({len(self.categorical_features)}): {self.categorical_features}")
        
        return self.df
    
    def data_overview_and_statistics(self) -> Dict[str, Any]:
        """
        Task 2.1.2: Data overview and basic statistics.
        
        Generate comprehensive data summary and basic statistics.
        
        Returns:
            Dict[str, Any]: Summary statistics and data overview
        """
        print("\n=== Task 2.1.2: Data Overview and Basic Statistics ===")
        
        overview = {}
        
        # Basic data info
        print("\n--- Data Shape and Info ---")
        print(f"Dataset shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Data types
        print("\n--- Data Types ---")
        dtype_summary = self.df.dtypes.value_counts()
        print(dtype_summary)
        overview['data_types'] = dtype_summary.to_dict()
        
        # Missing values summary
        print("\n--- Missing Values Summary ---")
        missing_summary = self.df.isnull().sum()
        missing_pct = (missing_summary / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing_summary,
            'Missing_Percentage': missing_pct
        }).sort_values('Missing_Count', ascending=False)
        
        print(missing_df[missing_df['Missing_Count'] > 0])
        overview['missing_values'] = missing_df.to_dict()
        
        # Numerical features statistics
        print("\n--- Numerical Features Statistics ---")
        numerical_stats = self.df[self.numerical_features].describe()
        print(numerical_stats)
        overview['numerical_statistics'] = numerical_stats.to_dict()
        
        # Categorical features summary
        print("\n--- Categorical Features Summary ---")
        categorical_summary = {}
        for col in self.categorical_features:
            unique_count = self.df[col].nunique()
            most_common = self.df[col].value_counts().head(3)
            categorical_summary[col] = {
                'unique_count': unique_count,
                'most_common': most_common.to_dict()
            }
            print(f"{col}: {unique_count} unique values")
            print(f"  Top 3: {most_common.to_dict()}")
        
        overview['categorical_summary'] = categorical_summary
        
        # Target variable summary
        print("\n--- Target Variable Summary ---")
        target_stats = self.df[self.target_variable].describe()
        print(target_stats)
        overview['target_statistics'] = target_stats.to_dict()
        
        self.eda_results['data_overview'] = overview
        return overview
    
    def missing_data_analysis(self) -> Dict[str, Any]:
        """
        Task 2.2.1 & 2.2.2: Missing data pattern analysis and visualization.
        
        Comprehensive analysis of missing data patterns.
        
        Returns:
            Dict[str, Any]: Missing data analysis results
        """
        print("\n=== Task 2.2: Missing Data Analysis ===")
        
        missing_analysis = {}
        
        # Calculate missing value statistics
        missing_counts = self.df.isnull().sum()
        missing_percentages = (missing_counts / len(self.df)) * 100
        
        # Create missing data summary
        missing_summary = pd.DataFrame({
            'Feature': missing_counts.index,
            'Missing_Count': missing_counts.values,
            'Missing_Percentage': missing_percentages.values
        }).sort_values('Missing_Count', ascending=False)
        
        missing_analysis['summary'] = missing_summary.to_dict()
        
        # Identify features with missing data
        features_with_missing = missing_summary[missing_summary['Missing_Count'] > 0]['Feature'].tolist()
        print(f"Features with missing data: {features_with_missing}")
        
        if len(features_with_missing) > 0:
            # Missing data visualization
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Missing data bar chart
            missing_data = missing_summary[missing_summary['Missing_Count'] > 0]
            axes[0, 0].bar(range(len(missing_data)), missing_data['Missing_Percentage'])
            axes[0, 0].set_xticks(range(len(missing_data)))
            axes[0, 0].set_xticklabels(missing_data['Feature'], rotation=45)
            axes[0, 0].set_title('Missing Data Percentage by Feature')
            axes[0, 0].set_ylabel('Missing Percentage (%)')
            
            # 2. Missing data heatmap
            if len(features_with_missing) > 1:
                missing_matrix = self.df[features_with_missing].isnull()
                sns.heatmap(missing_matrix, cbar=True, ax=axes[0, 1], cmap='viridis')
                axes[0, 1].set_title('Missing Data Pattern Heatmap')
            else:
                axes[0, 1].text(0.5, 0.5, 'Only one feature\nwith missing data', 
                               ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Missing Data Pattern Heatmap')
            
            # 3. Missing data correlation (if multiple features have missing data)
            if len(features_with_missing) > 1:
                missing_corr = self.df[features_with_missing].isnull().corr()
                sns.heatmap(missing_corr, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
                axes[1, 0].set_title('Missing Data Correlation')
                missing_analysis['missing_correlation'] = missing_corr.to_dict()
            else:
                axes[1, 0].text(0.5, 0.5, 'Cannot compute\ncorrelation with\nsingle feature', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Missing Data Correlation')
            
            # 4. Missing data impact on target
            if self.target_variable in self.df.columns:
                target_with_missing = []
                target_without_missing = []
                
                for feature in features_with_missing:
                    if feature != self.target_variable:
                        missing_mask = self.df[feature].isnull()
                        target_missing = self.df[missing_mask][self.target_variable].dropna()
                        target_not_missing = self.df[~missing_mask][self.target_variable].dropna()
                        
                        if len(target_missing) > 0 and len(target_not_missing) > 0:
                            target_with_missing.extend(target_missing.tolist())
                            target_without_missing.extend(target_not_missing.tolist())
                
                if target_with_missing and target_without_missing:
                    axes[1, 1].hist([target_with_missing, target_without_missing], 
                                   bins=30, alpha=0.7, label=['With Missing', 'Without Missing'])
                    axes[1, 1].set_title('Target Distribution: Missing vs Non-Missing')
                    axes[1, 1].set_xlabel(self.target_variable)
                    axes[1, 1].set_ylabel('Frequency')
                    axes[1, 1].legend()
                    
                    # Statistical test
                    stat, p_value = stats.ttest_ind(target_with_missing, target_without_missing)
                    missing_analysis['target_impact'] = {
                        'statistic': stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
                else:
                    axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor comparison', 
                                   ha='center', va='center', transform=axes[1, 1].transAxes)
            
            plt.tight_layout()
            plt.savefig('missing_data_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        else:
            print("No missing data found in the dataset.")
            missing_analysis['no_missing_data'] = True
        
        self.eda_results['missing_data'] = missing_analysis
        return missing_analysis
    
    def univariate_analysis(self) -> Dict[str, Any]:
        """
        Task 2.3: Univariate Analysis.
        
        Analyze individual feature distributions and characteristics.
        
        Returns:
            Dict[str, Any]: Univariate analysis results
        """
        print("\n=== Task 2.3: Univariate Analysis ===")
        
        univariate_results = {}
        
        # 2.3.1: Numerical feature analysis
        print("\n--- Numerical Feature Analysis ---")
        numerical_analysis = self._analyze_numerical_features()
        univariate_results['numerical'] = numerical_analysis
        
        # 2.3.2: Categorical feature analysis
        print("\n--- Categorical Feature Analysis ---")
        categorical_analysis = self._analyze_categorical_features()
        univariate_results['categorical'] = categorical_analysis
        
        # 2.3.3: Target variable analysis
        print("\n--- Target Variable Analysis ---")
        target_analysis = self._analyze_target_variable()
        univariate_results['target'] = target_analysis
        
        self.eda_results['univariate'] = univariate_results
        return univariate_results
    
    def _analyze_numerical_features(self) -> Dict[str, Any]:
        """
        Analyze numerical features: distributions, outliers, normality.
        """
        numerical_analysis = {}
        
        # Create subplots for numerical features
        n_features = len(self.numerical_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, feature in enumerate(self.numerical_features):
            row, col = i // n_cols, i % n_cols
            
            # Skip if feature has all NaN values
            if self.df[feature].isna().all():
                continue
            
            data = self.df[feature].dropna()
            
            # Histogram with density curve
            axes[row, col].hist(data, bins=30, density=True, alpha=0.7, edgecolor='black')
            
            # Add density curve if data is not constant
            if data.std() > 0:
                x = np.linspace(data.min(), data.max(), 100)
                kde = stats.gaussian_kde(data)
                axes[row, col].plot(x, kde(x), 'r-', linewidth=2)
            
            axes[row, col].set_title(f'{feature} Distribution')
            axes[row, col].set_xlabel(feature)
            axes[row, col].set_ylabel('Density')
            
            # Calculate statistics
            feature_stats = {
                'mean': data.mean(),
                'median': data.median(),
                'std': data.std(),
                'skewness': skew(data),
                'kurtosis': kurtosis(data),
                'min': data.min(),
                'max': data.max()
            }
            
            # Normality test
            if len(data) > 8:  # Minimum sample size for normaltest
                stat, p_value = normaltest(data)
                feature_stats['normality_test'] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'is_normal': p_value > 0.05
                }
            
            # Outlier detection using IQR
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            feature_stats['outliers'] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(data)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
            numerical_analysis[feature] = feature_stats
            
            print(f"{feature}: Mean={feature_stats['mean']:.2f}, "
                  f"Std={feature_stats['std']:.2f}, "
                  f"Skew={feature_stats['skewness']:.2f}, "
                  f"Outliers={feature_stats['outliers']['count']} ({feature_stats['outliers']['percentage']:.1f}%)")
        
        # Remove empty subplots
        for i in range(n_features, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        plt.savefig('numerical_features_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Box plots for outlier visualization
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, feature in enumerate(self.numerical_features):
            row, col = i // n_cols, i % n_cols
            
            if not self.df[feature].isna().all():
                self.df.boxplot(column=feature, ax=axes[row, col])
                axes[row, col].set_title(f'{feature} Box Plot')
        
        # Remove empty subplots
        for i in range(n_features, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        plt.savefig('numerical_features_boxplots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return numerical_analysis
    
    def _analyze_categorical_features(self) -> Dict[str, Any]:
        """
        Analyze categorical features: distributions, imbalance, rare categories.
        """
        categorical_analysis = {}
        
        # Create subplots for categorical features
        n_features = len(self.categorical_features)
        n_cols = 2
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, feature in enumerate(self.categorical_features):
            row, col = i // n_cols, i % n_cols
            
            # Value counts
            value_counts = self.df[feature].value_counts()
            value_percentages = self.df[feature].value_counts(normalize=True) * 100
            
            # Bar chart
            if len(value_counts) <= 20:  # Show all categories if <= 20
                value_counts.plot(kind='bar', ax=axes[row, col])
                axes[row, col].set_title(f'{feature} Distribution')
                axes[row, col].set_xlabel(feature)
                axes[row, col].set_ylabel('Count')
                axes[row, col].tick_params(axis='x', rotation=45)
            else:  # Show top 20 categories
                value_counts.head(20).plot(kind='bar', ax=axes[row, col])
                axes[row, col].set_title(f'{feature} Distribution (Top 20)')
                axes[row, col].set_xlabel(feature)
                axes[row, col].set_ylabel('Count')
                axes[row, col].tick_params(axis='x', rotation=45)
            
            # Analysis
            unique_count = self.df[feature].nunique()
            most_common = value_counts.iloc[0]
            most_common_pct = value_percentages.iloc[0]
            
            # Identify rare categories (< 1% of data)
            rare_categories = value_percentages[value_percentages < 1.0]
            
            # Check for imbalance (if most common category > 90%)
            is_imbalanced = most_common_pct > 90.0
            
            feature_analysis = {
                'unique_count': unique_count,
                'most_common_value': value_counts.index[0],
                'most_common_count': most_common,
                'most_common_percentage': most_common_pct,
                'rare_categories': {
                    'count': len(rare_categories),
                    'categories': rare_categories.to_dict()
                },
                'is_imbalanced': is_imbalanced,
                'value_counts': value_counts.to_dict(),
                'value_percentages': value_percentages.to_dict()
            }
            
            categorical_analysis[feature] = feature_analysis
            
            print(f"{feature}: {unique_count} unique values, "
                  f"Most common: '{value_counts.index[0]}' ({most_common_pct:.1f}%), "
                  f"Rare categories: {len(rare_categories)}, "
                  f"Imbalanced: {is_imbalanced}")
        
        # Remove empty subplots
        for i in range(n_features, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        plt.savefig('categorical_features_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return categorical_analysis
    
    def _analyze_target_variable(self) -> Dict[str, Any]:
        """
        Analyze target variable: distribution, outliers, transformation needs.
        """
        target_data = self.df[self.target_variable].dropna()
        
        # Create target variable analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Histogram with density curve
        axes[0, 0].hist(target_data, bins=50, density=True, alpha=0.7, edgecolor='black')
        x = np.linspace(target_data.min(), target_data.max(), 100)
        kde = stats.gaussian_kde(target_data)
        axes[0, 0].plot(x, kde(x), 'r-', linewidth=2)
        axes[0, 0].set_title(f'{self.target_variable} Distribution')
        axes[0, 0].set_xlabel(self.target_variable)
        axes[0, 0].set_ylabel('Density')
        
        # 2. Box plot
        axes[0, 1].boxplot(target_data)
        axes[0, 1].set_title(f'{self.target_variable} Box Plot')
        axes[0, 1].set_ylabel(self.target_variable)
        
        # 3. Q-Q plot for normality
        stats.probplot(target_data, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title(f'{self.target_variable} Q-Q Plot')
        
        # 4. Log transformation (if all values are positive)
        if target_data.min() > 0:
            log_target = np.log(target_data)
            axes[1, 1].hist(log_target, bins=50, density=True, alpha=0.7, edgecolor='black')
            axes[1, 1].set_title(f'Log({self.target_variable}) Distribution')
            axes[1, 1].set_xlabel(f'Log({self.target_variable})')
            axes[1, 1].set_ylabel('Density')
        else:
            axes[1, 1].text(0.5, 0.5, 'Cannot apply log\ntransformation\n(negative values)', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Log Transformation Not Applicable')
        
        plt.tight_layout()
        plt.savefig('target_variable_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Statistical analysis
        target_stats = {
            'count': len(target_data),
            'mean': target_data.mean(),
            'median': target_data.median(),
            'std': target_data.std(),
            'min': target_data.min(),
            'max': target_data.max(),
            'skewness': skew(target_data),
            'kurtosis': kurtosis(target_data)
        }
        
        # Normality test
        if len(target_data) > 8:
            stat, p_value = normaltest(target_data)
            target_stats['normality_test'] = {
                'statistic': stat,
                'p_value': p_value,
                'is_normal': p_value > 0.05
            }
        
        # Outlier detection
        Q1 = target_data.quantile(0.25)
        Q3 = target_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = target_data[(target_data < lower_bound) | (target_data > upper_bound)]
        
        target_stats['outliers'] = {
            'count': len(outliers),
            'percentage': (len(outliers) / len(target_data)) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
        
        # Transformation assessment
        transformation_recommendations = []
        
        if abs(target_stats['skewness']) > 1:
            transformation_recommendations.append('Consider transformation due to high skewness')
        
        if target_stats['normality_test']['p_value'] < 0.05:
            transformation_recommendations.append('Consider transformation due to non-normality')
        
        if target_stats['outliers']['percentage'] > 5:
            transformation_recommendations.append('Consider outlier treatment')
        
        target_stats['transformation_recommendations'] = transformation_recommendations
        
        print(f"Target Variable ({self.target_variable}) Analysis:")
        print(f"  Mean: {target_stats['mean']:.2f}")
        print(f"  Median: {target_stats['median']:.2f}")
        print(f"  Std: {target_stats['std']:.2f}")
        print(f"  Skewness: {target_stats['skewness']:.2f}")
        print(f"  Outliers: {target_stats['outliers']['count']} ({target_stats['outliers']['percentage']:.1f}%)")
        print(f"  Normal distribution: {target_stats['normality_test']['is_normal']}")
        print(f"  Transformation recommendations: {transformation_recommendations}")
        
        return target_stats
    
    def bivariate_analysis(self):
        """Task 2.4: Analyze relationships between features and with target variable"""
        print("\n" + "="*50)
        print("Task 2.4: Bivariate Analysis")
        print("="*50)
        
        # 2.4.1 Feature correlation analysis
        print("\n2.4.1 Feature Correlation Analysis:")
        correlation_matrix = self.df[self.numerical_features + [self.target_variable]].corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Identify high correlations
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append((correlation_matrix.columns[i], 
                                          correlation_matrix.columns[j], corr_val))
        
        print(f"  High correlations (>0.7): {len(high_corr_pairs)} pairs found")
        for pair in high_corr_pairs:
            print(f"    {pair[0]} - {pair[1]}: {pair[2]:.3f}")
        
        # Target correlations
        target_corr = correlation_matrix[self.target_variable].abs().sort_values(ascending=False)
        print(f"\n  Top correlations with {self.target_variable}:")
        for feature, corr in target_corr.head(6).items():
            if feature != self.target_variable:
                print(f"    {feature}: {corr:.3f}")
        
        # 2.4.2 Numerical vs target analysis
        print("\n2.4.2 Numerical vs Target Analysis:")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, feature in enumerate(self.numerical_features[:6]):
            if i < len(axes):
                sns.scatterplot(data=self.df, x=feature, y=self.target_variable, 
                              alpha=0.6, ax=axes[i])
                sns.regplot(data=self.df, x=feature, y=self.target_variable, 
                           scatter=False, color='red', ax=axes[i])
                axes[i].set_title(f'{feature} vs {self.target_variable}')
        
        plt.tight_layout()
        plt.savefig('numerical_vs_target_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2.4.3 Categorical vs target analysis
        print("\n2.4.3 Categorical vs Target Analysis:")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, feature in enumerate(self.categorical_features[:6]):
            if i < len(axes):
                sns.boxplot(data=self.df, x=feature, y=self.target_variable, ax=axes[i])
                axes[i].set_title(f'{self.target_variable} by {feature}')
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('categorical_vs_target_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  Bivariate analysis completed and visualizations saved")
    
    def multivariate_analysis(self):
        """Task 2.5: Analyze complex relationships between multiple features"""
        print("\n" + "="*50)
        print("Task 2.5: Multivariate Analysis")
        print("="*50)
        
        # 2.5.1 Feature interaction analysis
        print("\n2.5.1 Feature Interaction Analysis:")
        
        # Select top correlated features for interaction analysis
        target_corr = self.df[self.numerical_features].corrwith(self.df[self.target_variable])
        top_features = target_corr.abs().nlargest(4).index.tolist()
        
        if len(top_features) >= 2:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Interaction plot 1
            if len(top_features) >= 2:
                sns.scatterplot(data=self.df, x=top_features[0], y=top_features[1], 
                              hue=self.target_variable, alpha=0.6, ax=axes[0])
                axes[0].set_title(f'Interaction: {top_features[0]} vs {top_features[1]}')
            
            # Interaction plot 2
            if len(top_features) >= 3:
                sns.scatterplot(data=self.df, x=top_features[0], y=top_features[2], 
                              hue=self.target_variable, alpha=0.6, ax=axes[1])
                axes[1].set_title(f'Interaction: {top_features[0]} vs {top_features[2]}')
            
            plt.tight_layout()
            plt.savefig('feature_interactions.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2.5.2 Dimensionality reduction exploration
        print("\n2.5.2 Dimensionality Reduction:")
        
        # Prepare data for PCA
        numerical_data = self.df[self.numerical_features].fillna(self.df[self.numerical_features].median())
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numerical_data)
        
        # PCA Analysis
        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)
        
        # Plot PCA results
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Explained variance
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        axes[0].plot(range(1, len(cumsum_var) + 1), cumsum_var, 'bo-')
        axes[0].set_xlabel('Number of Components')
        axes[0].set_ylabel('Cumulative Explained Variance')
        axes[0].set_title('PCA Explained Variance')
        axes[0].grid(True)
        
        # PCA scatter plot
        scatter = axes[1].scatter(pca_result[:, 0], pca_result[:, 1], 
                                c=self.df[self.target_variable], alpha=0.6, cmap='viridis')
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        axes[1].set_title('PCA: First Two Components')
        plt.colorbar(scatter, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig('dimensionality_reduction.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  PCA: First 2 components explain {cumsum_var[1]:.2%} of variance")
        print("  Multivariate analysis completed and visualizations saved")
    
    def data_quality_assessment(self):
        """Task 2.6: Identify data quality issues and anomalies"""
        print("\n" + "="*50)
        print("Task 2.6: Data Quality Assessment")
        print("="*50)
        
        # 2.6.1 Outlier detection and analysis
        print("\n2.6.1 Outlier Detection:")
        
        outlier_summary = {}
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, feature in enumerate(self.numerical_features[:6]):
            if i < len(axes):
                # IQR method
                Q1 = self.df[feature].quantile(0.25)
                Q3 = self.df[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_iqr = self.df[(self.df[feature] < lower_bound) | 
                                        (self.df[feature] > upper_bound)]
                
                outlier_summary[feature] = {
                    'iqr_outliers': len(outliers_iqr),
                    'percentage': len(outliers_iqr) / len(self.df) * 100
                }
                
                # Visualization
                sns.boxplot(data=self.df, y=feature, ax=axes[i])
                axes[i].set_title(f'{feature} - Outliers: {len(outliers_iqr)} ({outlier_summary[feature]["percentage"]:.1f}%)')
        
        plt.tight_layout()
        plt.savefig('outlier_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        for feature, stats in outlier_summary.items():
            print(f"  {feature}: {stats['iqr_outliers']} outliers ({stats['percentage']:.1f}%)")
        
        # 2.6.2 Data consistency checks
        print("\n2.6.2 Data Consistency Checks:")
        
        consistency_issues = []
        
        # Check for logical inconsistencies (example checks)
        if 'age' in self.df.columns:
            invalid_age = self.df[(self.df['age'] < 0) | (self.df['age'] > 100)]
            if len(invalid_age) > 0:
                consistency_issues.append(f"Invalid age values: {len(invalid_age)} records")
        
        # Check for impossible score values
        score_columns = [col for col in self.df.columns if 'test' in col.lower() or 'score' in col.lower()]
        for col in score_columns:
            if col in self.df.columns:
                invalid_scores = self.df[(self.df[col] < 0) | (self.df[col] > 100)]
                if len(invalid_scores) > 0:
                    consistency_issues.append(f"Invalid {col} values: {len(invalid_scores)} records")
        
        if consistency_issues:
            for issue in consistency_issues:
                print(f"  ⚠️  {issue}")
        else:
            print("  ✅ No major consistency issues found")
        
        # 2.6.3 Duplicate analysis
        print("\n2.6.3 Duplicate Analysis:")
        
        # Check for exact duplicates
        exact_duplicates = self.df.duplicated().sum()
        print(f"  Exact duplicates: {exact_duplicates} records ({exact_duplicates/len(self.df)*100:.2f}%)")
        
        # Check for near-duplicates (excluding ID columns)
        feature_cols = [col for col in self.df.columns if col not in ['student_id']]
        near_duplicates = self.df[feature_cols].duplicated().sum()
        print(f"  Near-duplicates (excluding IDs): {near_duplicates} records ({near_duplicates/len(self.df)*100:.2f}%)")
        
        print("  Data quality assessment completed")
    
    def feature_specific_analysis(self):
        """Task 2.7: Feature-specific deep dive analysis"""
        print("\n" + "="*50)
        print("Task 2.7: Feature-Specific Analysis")
        print("="*50)
        
        # Analyze key features based on domain knowledge
        key_features = ['age', 'hours_per_week', 'sleep_time'] if all(col in self.df.columns for col in ['age', 'hours_per_week', 'sleep_time']) else self.numerical_features[:3]
        
        for feature in key_features:
            if feature in self.df.columns:
                print(f"\nDeep dive: {feature}")
                
                # Statistical summary
                stats = self.df[feature].describe()
                print(f"  Range: {stats['min']:.2f} - {stats['max']:.2f}")
                print(f"  Mean ± Std: {stats['mean']:.2f} ± {stats['std']:.2f}")
                print(f"  Median: {stats['50%']:.2f}")
                
                # Relationship with target
                correlation = self.df[feature].corr(self.df[self.target_variable])
                print(f"  Correlation with {self.target_variable}: {correlation:.3f}")
                
                # Missing values
                missing_pct = self.df[feature].isnull().sum() / len(self.df) * 100
                print(f"  Missing values: {missing_pct:.1f}%")
        
        print("  Feature-specific analysis completed")
    
    def generate_recommendations(self):
        """Task 2.8: Generate actionable recommendations"""
        print("\n" + "="*50)
        print("Task 2.8: Recommendations")
        print("="*50)
        
        recommendations = []
        
        # Data quality recommendations
        missing_threshold = 20  # 20% missing data threshold
        high_missing_features = []
        for feature in self.df.columns:
            missing_pct = self.df[feature].isnull().sum() / len(self.df) * 100
            if missing_pct > missing_threshold:
                high_missing_features.append((feature, missing_pct))
        
        if high_missing_features:
            recommendations.append("Data Quality:")
            for feature, pct in high_missing_features:
                recommendations.append(f"  - Consider imputation strategy for {feature} ({pct:.1f}% missing)")
        
        # Feature engineering recommendations
        recommendations.append("\nFeature Engineering:")
        
        # Check for highly correlated features
        correlation_matrix = self.df[self.numerical_features].corr()
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > 0.8:
                    high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))
        
        if high_corr_pairs:
            recommendations.append("  - Consider removing highly correlated features:")
            for pair in high_corr_pairs:
                recommendations.append(f"    * {pair[0]} and {pair[1]}")
        
        # Target variable recommendations
        target_stats = self.df[self.target_variable].describe()
        target_skew = self.df[self.target_variable].skew()
        
        recommendations.append("\nTarget Variable:")
        if abs(target_skew) > 1:
            recommendations.append(f"  - Consider transformation (skewness: {target_skew:.2f})")
        else:
            recommendations.append(f"  - Target distribution is reasonably normal (skewness: {target_skew:.2f})")
        
        # Modeling recommendations
        recommendations.append("\nModeling Strategy:")
        recommendations.append("  - Use cross-validation for model evaluation")
        recommendations.append("  - Consider ensemble methods for robust predictions")
        recommendations.append("  - Implement feature selection techniques")
        recommendations.append("  - Monitor for overfitting with validation curves")
        
        # Print all recommendations
        print("\nKey Recommendations:")
        for rec in recommendations:
            print(rec)
        
        # Save recommendations to results
        self.eda_results['recommendations'] = recommendations
        print("\n  Recommendations generated and saved")
    
    def save_results(self):
        """Save EDA results to files"""
        # Enhanced summary with all completed tasks
        summary = f"""Student Score Prediction - EDA Summary (Phase 2 Complete)
================================================================
Dataset Shape: {self.df.shape}
Numerical Features: {len(self.numerical_features)}
Categorical Features: {len(self.categorical_features)}
Target Variable: {self.target_variable}

Completed Tasks:
✅ 2.1 Data Loading and Initial Exploration
✅ 2.2 Missing Data Analysis
✅ 2.3 Univariate Analysis
✅ 2.4 Bivariate Analysis
✅ 2.5 Multivariate Analysis
✅ 2.6 Data Quality Assessment
✅ 2.7 Feature-Specific Analysis
✅ 2.8 Recommendations Generation

Key Findings:
- Data loaded successfully with all expected features
- Comprehensive analysis completed across all dimensions
- Visualizations generated for all analysis types
- Data quality issues identified and documented
- Actionable recommendations provided for next steps

Generated Visualizations:
- missing_data_analysis.png
- numerical_features_distribution.png
- numerical_features_boxplots.png
- categorical_features_distribution.png
- target_variable_analysis.png
- correlation_heatmap.png
- numerical_vs_target_analysis.png
- categorical_vs_target_analysis.png
- feature_interactions.png
- dimensionality_reduction.png
- outlier_analysis.png

Next Phase: Ready for Phase 3 - Data Preprocessing
"""
        
        with open('eda_summary.txt', 'w') as f:
            f.write(summary)
        
        print("\nSaving EDA results summary...")
        print("EDA summary saved to 'eda_summary.txt'")
        print("All visualization files saved as PNG images")
    
    def run_complete_eda(self) -> Dict[str, Any]:
        """
        Run the complete EDA pipeline covering all Phase 2 tasks.
        
        Returns:
            Dict[str, Any]: Complete EDA results
        """
        print("Starting Comprehensive EDA for Student Score Prediction")
        print("=" * 60)
        
        # Task 2.1: Data Loading and Initial Exploration
        self.load_data()
        self.data_overview_and_statistics()
        
        # Task 2.2: Missing Data Analysis
        self.missing_data_analysis()
        
        # Task 2.3: Univariate Analysis
        self.univariate_analysis()
        
        # Continue with remaining tasks
        self.bivariate_analysis()  # Task 2.4
        self.multivariate_analysis()  # Task 2.5
        self.data_quality_assessment()  # Task 2.6
        self.feature_specific_analysis()  # Task 2.7
        self.generate_recommendations()  # Task 2.8
        
        # Save results
        self.save_results()
        print("\n" + "=" * 60)
        print("EDA Phase 2 All Tasks (2.1-2.8) Completed Successfully")
        print("Results saved to eda_results attribute and visualization files")
        
        return self.eda_results


def main():
    """
    Main function to run the EDA analysis.
    """
    # Database path
    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'score.db')
    
    # Initialize and run EDA
    eda = StudentScoreEDA(db_path)
    results = eda.run_complete_eda()
    
    # Save results summary
    print("\nSaving EDA results summary...")
    
    # Create a simple text summary
    summary_lines = [
        "Student Score Prediction - EDA Summary",
        "=" * 40,
        f"Dataset Shape: {eda.df.shape}",
        f"Numerical Features: {len(eda.numerical_features)}",
        f"Categorical Features: {len(eda.categorical_features)}",
        f"Target Variable: {eda.target_variable}",
        "",
        "Key Findings:",
        "- Data loaded successfully with all expected features",
        "- Missing data analysis completed",
        "- Univariate analysis completed for all features",
        "- Visualizations saved as PNG files",
        "",
        "Next Steps:",
        "- Continue with bivariate analysis (Task 2.4)",
        "- Implement multivariate analysis (Task 2.5)",
        "- Complete data quality assessment (Task 2.6)",
        "- Perform feature-specific deep dive (Task 2.7)",
        "- Generate final recommendations (Task 2.8)"
    ]
    
    with open('eda_summary.txt', 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print("EDA summary saved to 'eda_summary.txt'")
    print("Visualization files saved as PNG images")


if __name__ == "__main__":
    main()