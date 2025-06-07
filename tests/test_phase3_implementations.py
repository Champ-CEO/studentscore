#!/usr/bin/env python3
"""
Test Suite for Phase 3 Implementations

Comprehensive tests for all Phase 3 modules:
- ID Structure Analysis
- Comprehensive Data Validation
- Enhanced Age Processing
- Consistency Checker
- Outlier Handler
- Imbalanced Data Analysis
- Preprocessing Pipeline
- Advanced Preprocessing
- Storage and Backup
"""

import pytest
import pandas as pd
import numpy as np
import sqlite3
import tempfile
import os
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.id_structure_analysis import IDStructureAnalyzer
from data.comprehensive_validation import ComprehensiveDataValidator
from data.enhanced_age_processing import EnhancedAgeProcessor
from data.consistency_checker import ComprehensiveConsistencyChecker
from data.outlier_handler import RobustOutlierHandler
from data.imbalanced_data_analysis import ImbalancedDataAnalyzer
from data.preprocessing_pipeline import DataPreprocessingPipeline
from data.advanced_preprocessing import AdvancedPreprocessingOrchestrator
from data.storage_backup import DataStorageManager, DataBackupManager, DataVersionControl

class TestPhase3Implementations:
    """Test suite for Phase 3 implementations."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample student data for testing."""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'student_id': [f'S{i:06d}' for i in range(1, n_samples + 1)],
            'age': np.random.normal(16, 2, n_samples).astype(int),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'ethnicity': np.random.choice(['Chinese', 'Malay', 'Indian', 'Others'], n_samples),
            'tuition': np.random.choice(['Yes', 'No'], n_samples),
            'cca': np.random.choice(['Sports', 'Arts', 'Academic', 'None'], n_samples),
            'attendance_rate': np.random.uniform(0.7, 1.0, n_samples),
            'final_test': np.random.normal(75, 15, n_samples)
        }
        
        # Add some data quality issues for testing
        # Outliers
        data['age'][0:5] = [5, 25, 30, -1, 100]
        data['final_test'][0:5] = [-50, 150, 200, -100, 300]
        
        # Missing values
        missing_indices = np.random.choice(n_samples, 50, replace=False)
        for idx in missing_indices[:25]:
            data['attendance_rate'][idx] = np.nan
        for idx in missing_indices[25:]:
            data['final_test'][idx] = np.nan
        
        # Inconsistent data
        data['tuition'][10:15] = ['YES', 'NO', 'yes', 'no', 'Y']
        data['cca'][20:25] = ['sports', 'ARTS', 'academic', 'NONE', 'Sport']
        
        # Duplicates
        data['student_id'][50] = data['student_id'][49]
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def temp_db(self, sample_data):
        """Create temporary database with sample data."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file.close()
        
        conn = sqlite3.connect(temp_file.name)
        sample_data.to_sql('student_scores', conn, if_exists='replace', index=False)
        conn.close()
        
        yield temp_file.name
        
        # Cleanup
        os.unlink(temp_file.name)
    
    def test_id_structure_analysis(self, sample_data):
        """Test ID structure analysis functionality."""
        analyzer = IDStructureAnalyzer()
        
        # Test structure analysis
        structure_results = analyzer.analyze_id_structure(sample_data, 'student_id')
        
        assert 'pattern_analysis' in structure_results
        assert 'length_analysis' in structure_results
        assert 'character_analysis' in structure_results
        assert structure_results['total_ids'] == len(sample_data)
        
        # Test feature extraction
        features_df = analyzer.extract_features_from_id(sample_data, 'student_id')
        
        assert 'id_length' in features_df.columns
        assert 'id_numeric_part' in features_df.columns
        assert len(features_df) == len(sample_data)
        
        # Test retention decision
        decision = analyzer.make_id_retention_decision(sample_data, 'student_id')
        
        assert 'retain_id' in decision
        assert 'retain_features' in decision
        assert 'reasoning' in decision
        assert isinstance(decision['retain_id'], bool)
    
    def test_comprehensive_validation(self, sample_data):
        """Test comprehensive data validation."""
        validator = ComprehensiveDataValidator()
        
        # Test comprehensive validation
        validation_results = validator.validate_comprehensive(sample_data)
        
        assert 'data_types' in validation_results
        assert 'missing_values' in validation_results
        assert 'duplicates' in validation_results
        assert 'outliers' in validation_results
        assert 'consistency' in validation_results
        
        # Test specific validations
        missing_report = validator.validate_missing_values(sample_data)
        assert 'missing_summary' in missing_report
        assert 'missing_patterns' in missing_report
        
        duplicate_report = validator.validate_duplicates(sample_data, ['student_id'])
        assert 'duplicate_count' in duplicate_report
        assert duplicate_report['duplicate_count'] > 0  # We added duplicates
        
        outlier_report = validator.validate_outliers(sample_data)
        assert 'outlier_summary' in outlier_report
    
    def test_enhanced_age_processing(self, sample_data):
        """Test enhanced age processing functionality."""
        processor = EnhancedAgeProcessor()
        
        # Test age validation
        validation_results = processor.validate_ages(sample_data, 'age')
        
        assert 'invalid_ages' in validation_results
        assert 'age_statistics' in validation_results
        assert validation_results['total_records'] == len(sample_data)
        
        # Test age correction
        corrected_data = processor.correct_invalid_ages(sample_data, 'age')
        
        # Should have fewer invalid ages after correction
        corrected_validation = processor.validate_ages(corrected_data, 'age')
        assert corrected_validation['invalid_count'] <= validation_results['invalid_count']
        
        # Test feature engineering
        features_df = processor.engineer_age_features(sample_data, 'age')
        
        assert 'age_group' in features_df.columns
        assert 'age_zscore' in features_df.columns
        assert 'age_percentile' in features_df.columns
    
    def test_consistency_checker(self, sample_data):
        """Test consistency checker functionality."""
        checker = ComprehensiveConsistencyChecker()
        
        # Test comprehensive consistency check
        consistency_results = checker.check_comprehensive_consistency(sample_data)
        
        assert 'categorical_consistency' in consistency_results
        assert 'cross_field_consistency' in consistency_results
        assert 'format_consistency' in consistency_results
        
        # Test categorical consistency
        cat_results = checker.check_categorical_consistency(sample_data)
        
        # Should detect inconsistencies in tuition and cca columns
        assert len(cat_results) > 0
        
        # Test standardization
        standardized_data = checker.standardize_categorical_values(sample_data)
        
        # Check that standardization worked
        tuition_values = set(standardized_data['tuition'].dropna().unique())
        assert tuition_values.issubset({'Yes', 'No'})
    
    def test_outlier_handler(self, sample_data):
        """Test outlier handler functionality."""
        handler = RobustOutlierHandler()
        
        # Test outlier detection
        outlier_results = handler.detect_outliers_comprehensive(sample_data)
        
        assert 'statistical_outliers' in outlier_results
        assert 'ml_outliers' in outlier_results
        assert 'multivariate_outliers' in outlier_results
        
        # Should detect outliers in age and final_test
        assert len(outlier_results['statistical_outliers']) > 0
        
        # Test outlier treatment
        treated_data = handler.treat_outliers(sample_data, outlier_results, method='cap')
        
        # Treated data should have fewer extreme values
        assert treated_data['age'].min() >= sample_data['age'].min()
        assert treated_data['final_test'].max() <= sample_data['final_test'].max() or pd.isna(treated_data['final_test'].max())
    
    def test_imbalanced_data_analysis(self, sample_data):
        """Test imbalanced data analysis functionality."""
        analyzer = ImbalancedDataAnalyzer()
        
        # Create imbalanced target for testing
        sample_data['performance_category'] = pd.cut(
            sample_data['final_test'].fillna(sample_data['final_test'].median()), 
            bins=[0, 60, 80, 100], 
            labels=['Low', 'Medium', 'High']
        )
        
        # Test imbalance detection
        imbalance_results = analyzer.analyze_class_imbalance(
            sample_data, 'performance_category'
        )
        
        assert 'class_distribution' in imbalance_results
        assert 'imbalance_metrics' in imbalance_results
        assert 'is_imbalanced' in imbalance_results
        
        # Test resampling strategies
        if imbalance_results['is_imbalanced']:
            resampled_data = analyzer.apply_resampling_strategy(
                sample_data, 'performance_category', strategy='smote'
            )
            
            # Resampled data should be more balanced
            original_dist = sample_data['performance_category'].value_counts()
            resampled_dist = resampled_data['performance_category'].value_counts()
            
            # Check that minority classes have more samples
            min_original = original_dist.min()
            min_resampled = resampled_dist.min()
            assert min_resampled >= min_original
    
    def test_preprocessing_pipeline(self, sample_data):
        """Test preprocessing pipeline functionality."""
        pipeline = DataPreprocessingPipeline()
        
        # Test pipeline creation
        X = sample_data.drop(columns=['final_test'])
        y = sample_data['final_test'].dropna()
        X = X.loc[y.index]
        
        # Test data splitting
        split_data = pipeline.split_data(X, y, test_size=0.2)
        
        assert 'X_train' in split_data
        assert 'X_test' in split_data
        assert 'y_train' in split_data
        assert 'y_test' in split_data
        
        # Test pipeline fitting
        fitted_pipeline = pipeline.fit_pipeline(split_data['X_train'], split_data['y_train'])
        
        assert fitted_pipeline is not None
        
        # Test transformation
        X_transformed = pipeline.transform_data(split_data['X_test'])
        
        assert X_transformed is not None
        assert len(X_transformed) == len(split_data['X_test'])
    
    def test_advanced_preprocessing(self, sample_data):
        """Test advanced preprocessing functionality."""
        orchestrator = AdvancedPreprocessingOrchestrator()
        
        # Prepare data
        X = sample_data.drop(columns=['final_test'])
        y = sample_data['final_test'].dropna()
        X = X.loc[y.index]
        
        # Test comprehensive preprocessing
        results = orchestrator.run_comprehensive_preprocessing(
            X, y, target_col='final_test'
        )
        
        assert 'feature_selection' in results
        assert 'pipeline_comparison' in results
        assert 'final_pipeline' in results
        assert 'recommendations' in results
        
        # Test that final pipeline exists
        assert results['final_pipeline'] is not None
        
        # Test that recommendations are generated
        assert len(results['recommendations']) > 0
    
    def test_storage_and_backup(self, sample_data):
        """Test storage and backup functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test storage manager
            storage_manager = DataStorageManager(temp_dir + "/storage")
            
            # Test data saving
            from data.storage_backup import StorageFormat, CompressionType
            
            saved_path = storage_manager.save_data(
                sample_data, "test_data", 
                StorageFormat.PARQUET, CompressionType.GZIP,
                "Test data for storage", ["test"]
            )
            
            assert os.path.exists(saved_path)
            
            # Test data loading
            loaded_data = storage_manager.load_data(saved_path)
            
            assert len(loaded_data) == len(sample_data)
            assert list(loaded_data.columns) == list(sample_data.columns)
            
            # Test backup manager
            backup_manager = DataBackupManager(temp_dir + "/backups", storage_manager)
            
            backup_path = backup_manager.create_backup(
                sample_data, "test_backup", "Test backup"
            )
            
            assert os.path.exists(backup_path)
            
            # Test backup restoration
            restored_data = backup_manager.restore_backup("test_backup")
            
            assert len(restored_data) == len(sample_data)
            
            # Test version control
            version_control = DataVersionControl(temp_dir + "/versions")
            
            version_id = version_control.commit_version(
                sample_data, "test_dataset", "Initial version"
            )
            
            assert version_id == "v001"
            
            # Test version retrieval
            versioned_data = version_control.get_version("test_dataset", version_id)
            
            assert len(versioned_data) == len(sample_data)
    
    def test_integration_workflow(self, sample_data, temp_db):
        """Test integrated workflow using multiple modules."""
        # Step 1: ID Structure Analysis
        id_analyzer = IDStructureAnalyzer()
        id_results = id_analyzer.analyze_id_structure(sample_data, 'student_id')
        
        # Step 2: Data Validation
        validator = ComprehensiveDataValidator()
        validation_results = validator.validate_comprehensive(sample_data)
        
        # Step 3: Age Processing
        age_processor = EnhancedAgeProcessor()
        corrected_data = age_processor.correct_invalid_ages(sample_data, 'age')
        
        # Step 4: Consistency Checking
        consistency_checker = ComprehensiveConsistencyChecker()
        standardized_data = consistency_checker.standardize_categorical_values(corrected_data)
        
        # Step 5: Outlier Handling
        outlier_handler = RobustOutlierHandler()
        outlier_results = outlier_handler.detect_outliers_comprehensive(standardized_data)
        treated_data = outlier_handler.treat_outliers(
            standardized_data, outlier_results, method='cap'
        )
        
        # Step 6: Preprocessing Pipeline
        pipeline = DataPreprocessingPipeline()
        X = treated_data.drop(columns=['final_test'])
        y = treated_data['final_test'].dropna()
        X = X.loc[y.index]
        
        split_data = pipeline.split_data(X, y, test_size=0.2)
        fitted_pipeline = pipeline.fit_pipeline(split_data['X_train'], split_data['y_train'])
        
        # Verify that the integrated workflow produces valid results
        assert len(treated_data) <= len(sample_data)  # Some outliers may be removed
        assert fitted_pipeline is not None
        assert len(split_data['X_train']) > 0
        assert len(split_data['X_test']) > 0
        
        # Verify data quality improvements
        original_age_issues = len(sample_data[(sample_data['age'] < 10) | (sample_data['age'] > 20)])
        treated_age_issues = len(treated_data[(treated_data['age'] < 10) | (treated_data['age'] > 20)])
        
        assert treated_age_issues <= original_age_issues


def run_comprehensive_tests():
    """Run all Phase 3 implementation tests."""
    print("Running comprehensive Phase 3 implementation tests...")
    
    # Run pytest
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure
    ])


if __name__ == "__main__":
    run_comprehensive_tests()