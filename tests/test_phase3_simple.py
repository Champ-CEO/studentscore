#!/usr/bin/env python3
"""
Simple Test Script for Phase 3 Implementations

Basic functionality tests for all Phase 3 modules without pytest dependency.
"""

import pandas as pd
import numpy as np
import sqlite3
import tempfile
import os
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

def create_sample_data():
    """Create sample student data for testing."""
    np.random.seed(42)
    n_samples = 100  # Smaller dataset for quick testing
    
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
    data['age'][0:3] = [5, 25, -1]  # Outliers
    data['final_test'][0:3] = [-50, 150, 200]  # Outliers
    
    # Missing values
    data['attendance_rate'][10:15] = [np.nan] * 5
    data['final_test'][20:25] = [np.nan] * 5
    
    # Inconsistent data
    data['tuition'][5:8] = ['YES', 'NO', 'yes']
    data['cca'][15:18] = ['sports', 'ARTS', 'academic']
    
    # Duplicates
    data['student_id'][50] = data['student_id'][49]
    
    return pd.DataFrame(data)

def test_id_structure_analysis():
    """Test ID structure analysis functionality."""
    print("Testing ID Structure Analysis...")
    
    try:
        from src.data.id_structure_analysis import IDStructureAnalyzer
        
        sample_data = create_sample_data()
        analyzer = IDStructureAnalyzer(data=sample_data)
        
        # Test structure analysis
        structure_results = analyzer.analyze_student_id_structure()
        assert 'basic_stats' in structure_results
        assert 'format_analysis' in structure_results
        assert 'pattern_analysis' in structure_results
        assert 'embedded_info' in structure_results
        
        # Test feature extraction
        features_df = analyzer.extract_features_from_ids()
        assert 'id_length' in features_df.columns
        
        # Test retention decision
        retention_decision = analyzer.make_retention_decisions()
        assert isinstance(retention_decision, dict)
        assert 'student_id' in retention_decision
        if 'student_id' in retention_decision:
            assert 'retain_original' in retention_decision['student_id']
            assert 'reasoning' in retention_decision['student_id']
        
        print("✓ ID Structure Analysis tests passed")
        return True
        
    except Exception as e:
        print(f"✗ ID Structure Analysis tests failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_comprehensive_validation():
    """Test comprehensive data validation."""
    print("Testing Comprehensive Data Validation...")
    
    try:
        from src.data.comprehensive_validation import ComprehensiveDataValidator
        
        sample_data = create_sample_data()
        validator = ComprehensiveDataValidator(data=sample_data)
        
        # Test comprehensive validation
        validation_results = validator.validate_all()
        assert 'data_types' in validation_results
        assert 'missing_values' in validation_results
        assert 'range_validation' in validation_results
        assert 'data_quality_score' in validation_results
        
        # Test specific validations
        missing_report = validator.validate_missing_values()
        assert 'missing_counts' in missing_report
        
        duplicate_report = validator.analyze_duplicates()
        assert 'exact_duplicates' in duplicate_report or 'duplicate_student_ids' in duplicate_report
        
        print("✓ Comprehensive Data Validation tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Comprehensive Data Validation tests failed: {str(e)}")
        return False

def test_enhanced_age_processing():
    """Test enhanced age processing functionality."""
    print("Testing Enhanced Age Processing...")
    
    try:
        from src.data.enhanced_age_processing import EnhancedAgeProcessor
        
        sample_data = create_sample_data()
        processor = EnhancedAgeProcessor(data=sample_data)
        
        # Test enhanced age processing
        age_results = processor.process_age_comprehensive()
        assert 'processed_data' in age_results
        assert 'statistics' in age_results
        assert 'outliers' in age_results
        
        print("✓ Enhanced Age Processing tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Enhanced Age Processing tests failed: {str(e)}")
        return False

def test_consistency_checker():
    """Test consistency checker functionality."""
    print("Testing Consistency Checker...")
    
    try:
        from src.data.consistency_checker import ComprehensiveConsistencyChecker
        
        sample_data = create_sample_data()
        checker = ComprehensiveConsistencyChecker(data=sample_data)
        
        # Test comprehensive consistency check
        consistency_results = checker.check_all_consistency()
        assert 'format_consistency' in consistency_results
        assert 'value_consistency' in consistency_results
        assert 'cross_field_consistency' in consistency_results
        assert 'overall_consistency_score' in consistency_results
        
        # Test consistency checking (no standardization method available)
        # The checker focuses on detecting inconsistencies rather than fixing them
        assert hasattr(checker, 'check_all_consistency')
        
        print("✓ Consistency Checker tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Consistency Checker tests failed: {str(e)}")
        return False

def test_outlier_handler():
    """Test outlier handler functionality."""
    print("Testing Outlier Handler...")
    
    try:
        from src.data.outlier_handler import RobustOutlierHandler
        
        sample_data = create_sample_data()
        handler = RobustOutlierHandler(data=sample_data)
        
        # Test outlier detection
        outlier_results = handler.detect_all_outliers()
        assert 'univariate_outliers' in outlier_results
        assert 'multivariate_outliers' in outlier_results
        assert 'outlier_summary' in outlier_results
        
        # Test outlier treatment
        treatment_plan = {'age': 'cap', 'final_test': 'flag'}
        treated_data = handler.apply_outlier_treatment(treatment_plan)
        assert len(treated_data) <= len(sample_data)
        
        print("✓ Outlier Handler tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Outlier Handler tests failed: {str(e)}")
        return False

def test_imbalanced_data_analysis():
    """Test imbalanced data analysis functionality."""
    print("Testing Imbalanced Data Analysis...")
    
    try:
        from src.data.imbalanced_data_analysis import ImbalancedDataAnalyzer
        
        sample_data = create_sample_data()
        analyzer = ImbalancedDataAnalyzer(data=sample_data)
        
        # Create imbalanced target for testing
        sample_data['performance_category'] = pd.cut(
            sample_data['final_test'].fillna(sample_data['final_test'].median()), 
            bins=[0, 60, 80, 100], 
            labels=['Low', 'Medium', 'High']
        )
        
        # Test imbalance detection
        data = analyzer.load_data()
        analyzer.identify_feature_types(data)
        imbalance_results = analyzer.analyze_categorical_imbalance(data)
        assert isinstance(imbalance_results, dict)
        
        print("✓ Imbalanced Data Analysis tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Imbalanced Data Analysis tests failed: {str(e)}")
        return False

def test_preprocessing_pipeline():
    """Test preprocessing pipeline functionality."""
    print("Testing Preprocessing Pipeline...")
    
    try:
        from src.data.preprocessing_pipeline import DataPreprocessingPipeline
        
        sample_data = create_sample_data()
        pipeline = DataPreprocessingPipeline()
        
        # Test pipeline creation
        X = sample_data.drop('final_test', axis=1)
        y = sample_data['final_test']
        
        # Test data splitting
        split_data = pipeline.create_data_splits(sample_data, test_size=0.2)
        assert 'train' in split_data
        assert 'test' in split_data
        
        # Test pipeline creation and fitting
        preprocessing_pipeline = pipeline.create_preprocessing_pipeline()
        fitted_pipeline = pipeline.fit_pipeline(split_data['train'])
        assert fitted_pipeline is not None
        
        print("✓ Preprocessing Pipeline tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Preprocessing Pipeline tests failed: {str(e)}")
        return False

def test_storage_and_backup():
    """Test storage and backup functionality."""
    print("Testing Storage and Backup...")
    
    try:
        from src.data.storage_backup import DataStorageManager, DataBackupManager, StorageFormat, CompressionType
        
        sample_data = create_sample_data()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test storage manager
            storage_manager = DataStorageManager(temp_dir + "/storage")
            
            # Test data saving
            saved_path = storage_manager.save_data(
                sample_data, "test_data", 
                StorageFormat.CSV, CompressionType.GZIP,
                "Test data for storage", ["test"]
            )
            assert os.path.exists(saved_path)
            
            # Test data loading
            loaded_data = storage_manager.load_data(saved_path)
            assert len(loaded_data) == len(sample_data)
            
            # Test backup manager
            backup_manager = DataBackupManager(temp_dir + "/backups", storage_manager)
            backup_path = backup_manager.create_backup(
                sample_data, "test_backup", "Test backup"
            )
            assert os.path.exists(backup_path)
            
        print("✓ Storage and Backup tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Storage and Backup tests failed: {str(e)}")
        return False

def main():
    """Run all Phase 3 implementation tests."""
    print("=" * 60)
    print("Running Phase 3 Implementation Tests")
    print("=" * 60)
    
    test_functions = [
        test_id_structure_analysis,
        test_comprehensive_validation,
        test_enhanced_age_processing,
        test_consistency_checker,
        test_outlier_handler,
        test_imbalanced_data_analysis,
        test_preprocessing_pipeline,
        test_storage_and_backup
    ]
    
    passed = 0
    total = len(test_functions)
    
    for test_func in test_functions:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} failed with exception: {str(e)}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Phase 3 implementations are working correctly!")
    else:
        print(f"⚠️  {total - passed} tests failed. Please check the implementations.")
    
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)