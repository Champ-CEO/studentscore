#!/usr/bin/env python3
"""
Test suite for Phase 6 Task 6.1.1: Perfect Performance Analysis

Tests the PerfectPerformanceAnalyzer class to ensure it correctly identifies
data leakage and other issues causing unrealistic model performance.

Author: AI Assistant
Date: 2025-01-08
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Import the class to test
from src.modeling.phase6_task1_perfect_performance_analysis import PerfectPerformanceAnalyzer

class TestPerfectPerformanceAnalyzer:
    """
    Test suite for PerfectPerformanceAnalyzer class.
    """
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_data_normal(self):
        """Create sample data with normal variation."""
        np.random.seed(42)
        n_samples = 1000
        
        # Create features with normal variation
        data = {
            'feature1': np.random.normal(50, 10, n_samples),
            'feature2': np.random.normal(100, 20, n_samples),
            'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
            'feature4': np.random.uniform(0, 1, n_samples),
            'final_test': np.random.normal(75, 15, n_samples)  # Normal target variation
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_data_constant_target(self):
        """Create sample data with constant target (explains perfect performance)."""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'feature1': np.random.normal(50, 10, n_samples),
            'feature2': np.random.normal(100, 20, n_samples),
            'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
            'feature4': np.random.uniform(0, 1, n_samples),
            'final_test': np.full(n_samples, 75.0)  # Constant target
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_data_with_leakage(self):
        """Create sample data with obvious data leakage."""
        np.random.seed(42)
        n_samples = 1000
        
        # Create target first
        target = np.random.normal(75, 15, n_samples)
        
        data = {
            'feature1': np.random.normal(50, 10, n_samples),
            'feature2': np.random.normal(100, 20, n_samples),
            'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
            'leaked_feature': target + np.random.normal(0, 0.001, n_samples),  # Nearly identical to target
            'derived_feature': target * 2 + 10,  # Linear transformation of target
            'final_test': target
        }
        
        return pd.DataFrame(data)
    
    def create_temp_csv(self, data, temp_dir, filename='test_data.csv'):
        """Helper to create temporary CSV file."""
        file_path = Path(temp_dir) / filename
        data.to_csv(file_path, index=False)
        return str(file_path)
    
    def test_initialization(self, temp_dir):
        """Test PerfectPerformanceAnalyzer initialization."""
        data_path = Path(temp_dir) / 'test_data.csv'
        
        analyzer = PerfectPerformanceAnalyzer(
            clean_data_path=str(data_path),
            output_path=temp_dir
        )
        
        assert analyzer.clean_data_path == data_path
        assert analyzer.output_path == Path(temp_dir)
        assert analyzer.random_state == 42
        assert analyzer.data is None
        assert analyzer.X is None
        assert analyzer.y is None
    
    def test_load_data_success(self, temp_dir, sample_data_normal):
        """Test successful data loading."""
        data_path = self.create_temp_csv(sample_data_normal, temp_dir)
        
        analyzer = PerfectPerformanceAnalyzer(
            clean_data_path=data_path,
            output_path=temp_dir
        )
        
        result = analyzer.load_data()
        
        assert result is True
        assert analyzer.data is not None
        assert analyzer.X is not None
        assert analyzer.y is not None
        assert analyzer.data.shape == sample_data_normal.shape
        assert len(analyzer.X.columns) == len(sample_data_normal.columns) - 1
        assert len(analyzer.y) == len(sample_data_normal)
    
    def test_load_data_file_not_found(self, temp_dir):
        """Test data loading with non-existent file."""
        analyzer = PerfectPerformanceAnalyzer(
            clean_data_path=str(Path(temp_dir) / 'nonexistent.csv'),
            output_path=temp_dir
        )
        
        result = analyzer.load_data()
        
        assert result is False
        assert analyzer.data is None
    
    def test_load_data_missing_target(self, temp_dir, sample_data_normal):
        """Test data loading with missing target column."""
        data_path = self.create_temp_csv(sample_data_normal, temp_dir)
        
        analyzer = PerfectPerformanceAnalyzer(
            clean_data_path=data_path,
            output_path=temp_dir
        )
        
        result = analyzer.load_data(target_column='nonexistent_target')
        
        assert result is False
    
    def test_analyze_target_variable_normal(self, temp_dir, sample_data_normal):
        """Test target variable analysis with normal data."""
        data_path = self.create_temp_csv(sample_data_normal, temp_dir)
        
        analyzer = PerfectPerformanceAnalyzer(
            clean_data_path=data_path,
            output_path=temp_dir
        )
        analyzer.load_data()
        
        result = analyzer.analyze_target_variable()
        
        assert 'basic_stats' in result
        assert 'constant_target' in result
        assert 'very_low_variance' in result
        assert 'unrealistic_range' in result
        assert 'distribution' in result
        assert 'value_frequency' in result
        
        # Normal data should not trigger warnings
        assert result['constant_target'] is False
        assert result['very_low_variance'] is False
        assert result['unrealistic_range'] is False
        assert result['mostly_same_values'] is False
        
        # Check basic stats
        assert result['basic_stats']['count'] == len(sample_data_normal)
        assert result['basic_stats']['std'] > 1.0  # Should have reasonable variance
    
    def test_analyze_target_variable_constant(self, temp_dir, sample_data_constant_target):
        """Test target variable analysis with constant target."""
        data_path = self.create_temp_csv(sample_data_constant_target, temp_dir)
        
        analyzer = PerfectPerformanceAnalyzer(
            clean_data_path=data_path,
            output_path=temp_dir
        )
        analyzer.load_data()
        
        result = analyzer.analyze_target_variable()
        
        # Constant target should trigger warnings
        assert result['constant_target'] is True
        assert result['very_low_variance'] is True
        assert result['unrealistic_range'] is True
        assert result['mostly_same_values'] is True
        
        # Check that constant value is detected
        assert 'constant_value' in result
        assert result['constant_value'] == 75.0
        assert result['basic_stats']['std'] < 1e-10
    
    def test_analyze_feature_correlations_normal(self, temp_dir, sample_data_normal):
        """Test feature correlation analysis with normal data."""
        data_path = self.create_temp_csv(sample_data_normal, temp_dir)
        
        analyzer = PerfectPerformanceAnalyzer(
            clean_data_path=data_path,
            output_path=temp_dir
        )
        analyzer.load_data()
        
        result = analyzer.analyze_feature_correlations()
        
        assert 'numerical_features_count' in result
        assert 'perfect_correlations' in result
        assert 'near_perfect_correlations' in result
        assert 'high_correlations' in result
        assert 'correlation_stats' in result
        assert 'all_correlations' in result
        
        # Normal data should not have perfect correlations
        assert len(result['perfect_correlations']) == 0
        assert len(result['near_perfect_correlations']) == 0
        
        # Should have correlation stats
        assert 'max_correlation' in result['correlation_stats']
        assert result['correlation_stats']['max_correlation'] < 0.5  # Random data
    
    def test_analyze_feature_correlations_with_leakage(self, temp_dir, sample_data_with_leakage):
        """Test feature correlation analysis with data leakage."""
        data_path = self.create_temp_csv(sample_data_with_leakage, temp_dir)
        
        analyzer = PerfectPerformanceAnalyzer(
            clean_data_path=data_path,
            output_path=temp_dir
        )
        analyzer.load_data()
        
        result = analyzer.analyze_feature_correlations()
        
        # Should detect perfect/near-perfect correlations
        assert len(result['perfect_correlations']) > 0 or len(result['near_perfect_correlations']) > 0
        
        # Max correlation should be very high
        assert result['correlation_stats']['max_correlation'] > 0.95
    
    def test_analyze_feature_target_relationships_normal(self, temp_dir, sample_data_normal):
        """Test feature-target relationship analysis with normal data."""
        data_path = self.create_temp_csv(sample_data_normal, temp_dir)
        
        analyzer = PerfectPerformanceAnalyzer(
            clean_data_path=data_path,
            output_path=temp_dir
        )
        analyzer.load_data()
        
        result = analyzer.analyze_feature_target_relationships()
        
        assert 'identical_features' in result
        assert 'derived_features' in result
        assert 'suspicious_features' in result
        
        # Normal data should not have leakage
        assert len(result['identical_features']) == 0
        assert len(result['derived_features']) == 0
        assert len(result['suspicious_features']) == 0
    
    def test_analyze_feature_target_relationships_with_leakage(self, temp_dir, sample_data_with_leakage):
        """Test feature-target relationship analysis with data leakage."""
        data_path = self.create_temp_csv(sample_data_with_leakage, temp_dir)
        
        analyzer = PerfectPerformanceAnalyzer(
            clean_data_path=data_path,
            output_path=temp_dir
        )
        analyzer.load_data()
        
        result = analyzer.analyze_feature_target_relationships()
        
        # Should detect derived features (leaked_feature and derived_feature)
        assert len(result['derived_features']) > 0
    
    def test_analyze_data_leakage_patterns(self, temp_dir, sample_data_normal):
        """Test data leakage pattern analysis."""
        data_path = self.create_temp_csv(sample_data_normal, temp_dir)
        
        analyzer = PerfectPerformanceAnalyzer(
            clean_data_path=data_path,
            output_path=temp_dir
        )
        analyzer.load_data()
        
        result = analyzer.analyze_data_leakage_patterns()
        
        assert 'temporal_leakage' in result
        assert 'aggregation_leakage' in result
        assert 'future_information' in result
        assert 'target_encoding_leakage' in result
        
        # Normal data should not have leakage patterns
        assert len(result['aggregation_leakage']) == 0
        assert len(result['future_information']) == 0
    
    def test_test_model_with_random_splits(self, temp_dir, sample_data_normal):
        """Test model performance with random splits."""
        data_path = self.create_temp_csv(sample_data_normal, temp_dir)
        
        analyzer = PerfectPerformanceAnalyzer(
            clean_data_path=data_path,
            output_path=temp_dir
        )
        analyzer.load_data()
        
        result = analyzer.test_model_with_random_splits(n_splits=3)
        
        assert 'n_splits' in result
        assert 'results' in result
        assert 'performance_stats' in result
        assert 'consistent_perfect_performance' in result
        
        assert result['n_splits'] == 3
        assert len(result['results']) == 3
        
        # Normal data should not have perfect performance
        assert result['consistent_perfect_performance'] is False
        
        # Check that all splits have reasonable performance
        for split_result in result['results']:
            assert 'mae' in split_result
            assert 'rmse' in split_result
            assert 'r2' in split_result
            assert split_result['r2'] < 0.99  # Should not be perfect
    
    def test_test_model_with_random_splits_constant_target(self, temp_dir, sample_data_constant_target):
        """Test model performance with constant target (should be perfect)."""
        data_path = self.create_temp_csv(sample_data_constant_target, temp_dir)
        
        analyzer = PerfectPerformanceAnalyzer(
            clean_data_path=data_path,
            output_path=temp_dir
        )
        analyzer.load_data()
        
        result = analyzer.test_model_with_random_splits(n_splits=3)
        
        # Constant target should result in perfect performance
        assert result['consistent_perfect_performance'] is True
        
        # All splits should have very high R²
        for split_result in result['results']:
            assert split_result['r2'] > 0.99
    
    def test_generate_analysis_report(self, temp_dir, sample_data_constant_target):
        """Test analysis report generation."""
        data_path = self.create_temp_csv(sample_data_constant_target, temp_dir)
        
        analyzer = PerfectPerformanceAnalyzer(
            clean_data_path=data_path,
            output_path=temp_dir
        )
        analyzer.load_data()
        
        # Run some analyses to populate results
        analyzer.analyze_target_variable()
        analyzer.analyze_feature_correlations()
        analyzer.test_model_with_random_splits(n_splits=2)
        
        result = analyzer.generate_analysis_report()
        
        assert 'analysis_timestamp' in result
        assert 'dataset_info' in result
        assert 'critical_findings' in result
        assert 'recommendations' in result
        assert 'analysis_results' in result
        assert 'overall_assessment' in result
        
        # Constant target should generate critical findings
        assert len(result['critical_findings']) > 0
        assert len(result['recommendations']) > 0
        
        # Should detect the constant target issue
        findings_text = ' '.join(result['critical_findings'])
        assert 'CONSTANT' in findings_text.upper() or 'IDENTICAL' in findings_text.upper()
    
    def test_run_complete_analysis(self, temp_dir, sample_data_normal):
        """Test complete analysis workflow."""
        data_path = self.create_temp_csv(sample_data_normal, temp_dir)
        
        analyzer = PerfectPerformanceAnalyzer(
            clean_data_path=data_path,
            output_path=temp_dir
        )
        
        result = analyzer.run_complete_analysis()
        
        # Check that all analysis components are present
        assert 'analysis_timestamp' in result
        assert 'dataset_info' in result
        assert 'critical_findings' in result
        assert 'recommendations' in result
        assert 'analysis_results' in result
        assert 'overall_assessment' in result
        
        # Check that all sub-analyses were run
        analysis_results = result['analysis_results']
        assert 'target_analysis' in analysis_results
        assert 'correlation_analysis' in analysis_results
        assert 'relationship_analysis' in analysis_results
        assert 'leakage_analysis' in analysis_results
        assert 'split_test_results' in analysis_results
        
        # Check that output file was created
        output_file = Path(temp_dir) / 'perfect_performance_analysis.json'
        assert output_file.exists()
        
        # Verify output file content
        with open(output_file, 'r') as f:
            saved_result = json.load(f)
        
        assert saved_result['overall_assessment'] == result['overall_assessment']
    
    def test_run_complete_analysis_with_leakage(self, temp_dir, sample_data_with_leakage):
        """Test complete analysis with data containing leakage."""
        data_path = self.create_temp_csv(sample_data_with_leakage, temp_dir)
        
        analyzer = PerfectPerformanceAnalyzer(
            clean_data_path=data_path,
            output_path=temp_dir
        )
        
        result = analyzer.run_complete_analysis()
        
        # Should detect critical issues
        assert len(result['critical_findings']) > 0
        assert 'CRITICAL ISSUES DETECTED' in result['overall_assessment']
        
        # Should provide recommendations
        assert len(result['recommendations']) > 0
    
    def test_error_handling_invalid_data(self, temp_dir):
        """Test error handling with invalid data."""
        # Create invalid CSV
        invalid_data = pd.DataFrame({
            'col1': [1, 2, 'invalid'],
            'col2': [None, None, None]
        })
        data_path = self.create_temp_csv(invalid_data, temp_dir)
        
        analyzer = PerfectPerformanceAnalyzer(
            clean_data_path=data_path,
            output_path=temp_dir
        )
        
        # Should handle errors gracefully
        with pytest.raises(Exception):
            analyzer.run_complete_analysis()

if __name__ == "__main__":
    pytest.main([__file__])