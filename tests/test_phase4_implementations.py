#!/usr/bin/env python3
"""
Phase 4 Feature Engineering - Comprehensive Test Suite

This script tests all Phase 4 implementations to ensure they work correctly.
Tests are designed to be run before executing the actual Phase 4 pipeline.

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
import sys
import traceback
from typing import Dict, Any, List, Tuple
import tempfile
import shutil
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src/data to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src" / "data"))

try:
    from src.data.phase4_task1_load_validate import Phase4DataLoader
    from src.data.phase4_task2_derived_features import Phase4DerivedFeatures
    from src.data.phase4_task2_interaction_features import Phase4InteractionFeatures
    from src.data.phase4_task2_transformations import Phase4Transformations
    from src.data.phase4_task3_advanced_preprocessing import Phase4AdvancedPreprocessing
    from src.data.phase4_task4_feature_selection import Phase4FeatureSelection
    from src.data.phase4_task5_data_quality import Phase4DataQuality
    from src.data.phase4_task6_documentation import Phase4Documentation
except ImportError as e:
    logger.error(f"Failed to import Phase 4 modules: {str(e)}")
    logger.error("Please ensure all Phase 4 task files are in src/data directory")
    sys.exit(1)

class Phase4TestSuite:
    """
    Comprehensive test suite for Phase 4 Feature Engineering.
    """
    
    def __init__(self):
        """
        Initialize the test suite.
        """
        self.test_results = {
            'start_time': datetime.now().isoformat(),
            'tests_passed': 0,
            'tests_failed': 0,
            'test_details': [],
            'overall_status': 'pending'
        }
        
        self.sample_data = None
        self.temp_dir = None
        
    def create_sample_data(self) -> pd.DataFrame:
        """
        Create sample data for testing.
        
        Returns:
            Sample DataFrame
        """
        np.random.seed(42)
        n_samples = 1000
        
        # Create realistic student performance data
        data = {
            'student_id': range(1, n_samples + 1),
            'final_test': np.random.normal(75, 15, n_samples).clip(0, 100),
            'previous_score': np.random.normal(70, 12, n_samples).clip(0, 100),
            'study_hours': np.random.exponential(3, n_samples).clip(0, 20),
            'attendance': np.random.normal(85, 10, n_samples).clip(0, 100),
            'age': np.random.randint(16, 25, n_samples),
            'parental_education_level': np.random.choice(
                ['No Education', 'Primary', 'Secondary', 'Higher Secondary', 'Bachelor', 'Master', 'PhD'],
                n_samples, p=[0.05, 0.1, 0.2, 0.25, 0.25, 0.1, 0.05]
            ),
            'distance_from_home': np.random.choice(['Near', 'Moderate', 'Far'], n_samples, p=[0.4, 0.4, 0.2]),
            'internet_access': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
            'family_income': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.3, 0.5, 0.2]),
            'school_type': np.random.choice(['Public', 'Private'], n_samples, p=[0.7, 0.3]),
            'peer_influence': np.random.choice(['Negative', 'Neutral', 'Positive'], n_samples, p=[0.2, 0.3, 0.5]),
            'physical_activity': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.3, 0.4, 0.3]),
            'learning_disabilities': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            'parental_support': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.2, 0.5, 0.3]),
            'extracurricular_activities': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
            'motivation_level': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.2, 0.5, 0.3]),
            'teacher_quality': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.1, 0.6, 0.3]),
            'tutoring_sessions': np.random.poisson(2, n_samples).clip(0, 10),
            'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.5, 0.5])
        }
        
        df = pd.DataFrame(data)
        
        # Add some missing values to test handling
        missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
        missing_columns = np.random.choice(df.columns[1:], size=len(missing_indices))
        
        for idx, col in zip(missing_indices, missing_columns):
            df.loc[idx, col] = np.nan
            
        # Add some correlations to make it realistic
        # Higher study hours should correlate with better scores
        correlation_boost = (df['study_hours'] - df['study_hours'].mean()) * 2
        df['final_test'] = (df['final_test'] + correlation_boost).clip(0, 100)
        
        # Higher attendance should correlate with better scores
        attendance_boost = (df['attendance'] - df['attendance'].mean()) * 0.3
        df['final_test'] = (df['final_test'] + attendance_boost).clip(0, 100)
        
        self.sample_data = df
        logger.info(f"Created sample data with shape: {df.shape}")
        
        return df
        
    def setup_test_environment(self) -> None:
        """
        Set up temporary test environment.
        """
        self.temp_dir = Path(tempfile.mkdtemp(prefix="phase4_test_"))
        logger.info(f"Created temporary test directory: {self.temp_dir}")
        
        # Create necessary subdirectories
        (self.temp_dir / "data" / "processed").mkdir(parents=True, exist_ok=True)
        (self.temp_dir / "data" / "featured").mkdir(parents=True, exist_ok=True)
        
        # Save sample data
        if self.sample_data is None:
            self.create_sample_data()
            
        sample_data_path = self.temp_dir / "data" / "processed" / "final_processed.csv"
        self.sample_data.to_csv(sample_data_path)
        
        logger.info(f"Sample data saved to: {sample_data_path}")
        
    def cleanup_test_environment(self) -> None:
        """
        Clean up temporary test environment.
        """
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            
    def run_test(self, test_name: str, test_function) -> bool:
        """
        Run a single test and record results.
        
        Args:
            test_name: Name of the test
            test_function: Function to execute
            
        Returns:
            True if test passed, False otherwise
        """
        logger.info(f"Running test: {test_name}")
        
        try:
            start_time = datetime.now()
            result = test_function()
            end_time = datetime.now()
            
            duration = (end_time - start_time).total_seconds()
            
            if result:
                logger.info(f"✅ {test_name} PASSED ({duration:.2f}s)")
                self.test_results['tests_passed'] += 1
                status = 'PASSED'
            else:
                logger.error(f"❌ {test_name} FAILED ({duration:.2f}s)")
                self.test_results['tests_failed'] += 1
                status = 'FAILED'
                
            self.test_results['test_details'].append({
                'test_name': test_name,
                'status': status,
                'duration_seconds': duration,
                'timestamp': start_time.isoformat()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {str(e)}")
            logger.error(traceback.format_exc())
            
            self.test_results['tests_failed'] += 1
            self.test_results['test_details'].append({
                'test_name': test_name,
                'status': 'FAILED',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            
            return False
            
    def test_task1_load_validate(self) -> bool:
        """
        Test Task 1: Load and Validate Data.
        
        Returns:
            True if test passed
        """
        try:
            data_path = self.temp_dir / "data" / "processed" / "final_processed.csv"
            
            # Test data loader
            loader = Phase4DataLoader(str(data_path))
            df = loader.load_data()
            
            # Get validation results from the loader
            summary = loader.get_data_summary()
            validation_results = summary['validation_results']
            
            # Verify results
            assert df is not None, "DataFrame should not be None"
            assert len(df) > 0, "DataFrame should not be empty"
            assert validation_results is not None, "Validation results should not be None"
            
            logger.info(f"Loaded data shape: {df.shape}")
            logger.info(f"Validation results: {len(validation_results)} checks")
            
            return True
            
        except Exception as e:
            logger.error(f"Task 1 test failed: {str(e)}")
            return False
            
    def test_task2_1_derived_features(self) -> bool:
        """
        Test Task 2.1: Derived Features.
        
        Returns:
            True if test passed
        """
        try:
            # Test derived features
            processor = Phase4DerivedFeatures(self.sample_data)
            
            # Test individual feature creation
            study_efficiency = processor.create_study_efficiency_score()
            academic_support = processor.create_academic_support_index()
            
            # Verify results
            assert study_efficiency is not None, "Study efficiency score should not be None"
            assert academic_support is not None, "Academic support index should not be None"
            assert len(study_efficiency) == len(self.sample_data), "Row count should be preserved"
            assert len(academic_support) == len(self.sample_data), "Row count should be preserved"
            
            logger.info(f"Study efficiency score created: {len(study_efficiency)} values")
            logger.info(f"Academic support index created: {len(academic_support)} values")
            
            return True
            
        except Exception as e:
            logger.error(f"Task 2.1 test failed: {str(e)}")
            return False
            
    def test_task2_2_interaction_features(self) -> bool:
        """
        Test Task 2.2: Interaction Features.
        
        Returns:
            True if test passed
        """
        try:
            # Test interaction features
            processor = Phase4InteractionFeatures(self.sample_data)
            
            # Test primary interaction
            study_attendance_interaction = processor.create_study_attendance_interaction()
            
            # Test additional interactions
            processor.create_additional_primary_interactions()
            processor.create_efficiency_interactions()
            
            # Verify results
            assert study_attendance_interaction is not None, "Study-attendance interaction should not be None"
            assert len(study_attendance_interaction) == len(self.sample_data), "Row count should be preserved"
            
            logger.info(f"Study-attendance interaction created: {len(study_attendance_interaction)} values")
            logger.info(f"Additional interactions created successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Task 2.2 test failed: {str(e)}")
            return False
            
    def test_task2_3_transformations(self) -> bool:
        """
        Test Task 2.3: Distribution-based Transformations.
        
        Returns:
            True if test passed
        """
        try:
            # Test transformations
            processor = Phase4Transformations(self.sample_data)
            
            # Test skewness analysis
            skewness_analysis = processor.analyze_skewness()
            
            # Test applying transformations
            processor.apply_additional_transformations()
            
            # Verify results
            assert skewness_analysis is not None, "Skewness analysis should not be None"
            assert isinstance(skewness_analysis, dict), "Should return a dictionary of skewness values"
            assert len(skewness_analysis) > 0, "Should have skewed features analysis"
            
            logger.info(f"Skewness analysis completed for {len(skewness_analysis)} features")
            logger.info(f"Transformations applied successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Task 2.3 test failed: {str(e)}")
            return False
            
    def test_task3_1_advanced_preprocessing(self) -> bool:
        """
        Test Task 3.1: Advanced Preprocessing.
        
        Returns:
            True if test passed
        """
        try:
            # Test advanced preprocessing
            processor = Phase4AdvancedPreprocessing(self.sample_data)
            
            # Test feature type identification
            feature_types = processor.identify_feature_types()
            
            # Test preprocessing steps
            processor.apply_numerical_scaling(feature_types)
            processor.apply_categorical_encoding(feature_types)
            
            # Test model-ready dataset creation
            model_ready_df = processor.create_model_ready_dataset()
            
            # Verify results
            assert feature_types is not None, "Feature types should not be None"
            assert model_ready_df is not None, "Model-ready DataFrame should not be None"
            assert len(model_ready_df) == len(self.sample_data), "Row count should be preserved"
            
            logger.info(f"Feature types identified: {len(feature_types)} categories")
            logger.info(f"Model-ready dataset shape: {model_ready_df.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"Task 3.1 test failed: {str(e)}")
            return False
            
    def test_task4_1_feature_selection(self) -> bool:
        """
        Test Task 4.1: Feature Selection.
        
        Returns:
            True if test passed
        """
        try:
            # Create a preprocessed dataset for feature selection
            processor = Phase4AdvancedPreprocessing(self.sample_data)
            feature_types = processor.identify_feature_types()
            processor.apply_numerical_scaling(feature_types)
            processor.apply_categorical_encoding(feature_types)
            preprocessed_df = processor.create_model_ready_dataset()
            
            # Test feature selection
            selector = Phase4FeatureSelection(preprocessed_df)
            selection_results = selector.run_feature_selection()
            selected_df = selector.create_selected_dataset()
            
            # Verify results
            assert selection_results is not None, "Selection results should not be None"
            assert selected_df is not None, "Selected DataFrame should not be None"
            assert len(selected_df) == len(preprocessed_df), "Row count should be preserved"
            assert selected_df.shape[1] <= preprocessed_df.shape[1], "Should have same or fewer features"
            
            logger.info(f"Feature selection completed")
            logger.info(f"Selected dataset shape: {selected_df.shape}")
            logger.info(f"Original dataset shape: {preprocessed_df.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"Task 4.1 test failed: {str(e)}")
            return False
            
    def test_task5_1_data_quality(self) -> bool:
        """
        Test Task 5.1: Data Quality Targets.
        
        Returns:
            True if test passed
        """
        try:
            # Test data quality assessment
            assessor = Phase4DataQuality(self.sample_data)
            quality_report = assessor.generate_quality_report()
            is_ready, blocking_issues = assessor.is_model_ready()
            
            # Verify results
            assert quality_report is not None, "Quality report should not be None"
            assert 'overall_compliance' in quality_report, "Should have overall compliance"
            assert 'assessments' in quality_report, "Should have assessments"
            assert isinstance(is_ready, bool), "Model readiness should be boolean"
            assert isinstance(blocking_issues, list), "Blocking issues should be list"
            
            logger.info(f"Overall compliance: {quality_report['overall_compliance']}")
            logger.info(f"Model ready: {is_ready}")
            logger.info(f"Blocking issues: {len(blocking_issues)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Task 5.1 test failed: {str(e)}")
            return False
            
    def test_task6_1_documentation(self) -> bool:
        """
        Test Task 6.1: Documentation and Validation.
        
        Returns:
            True if test passed
        """
        try:
            # Create a simple processed dataset
            processed_df = self.sample_data.copy()
            processed_df['test_feature'] = processed_df['study_hours'] * 2
            
            # Test documentation
            doc_generator = Phase4Documentation(self.sample_data, processed_df)
            feature_dict = doc_generator.generate_feature_dictionary()
            validation_report = doc_generator.generate_validation_report()
            
            # Verify results
            assert feature_dict is not None, "Feature dictionary should not be None"
            assert validation_report is not None, "Validation report should not be None"
            assert 'features' in feature_dict, "Should have features section"
            assert 'overall_validation_status' in validation_report, "Should have validation status"
            
            logger.info(f"Features documented: {len(feature_dict['features'])}")
            logger.info(f"Validation status: {validation_report['overall_validation_status']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Task 6.1 test failed: {str(e)}")
            return False
            
    def run_all_tests(self) -> bool:
        """
        Run all Phase 4 tests.
        
        Returns:
            True if all tests passed
        """
        logger.info("🧪 Starting Phase 4 Feature Engineering Test Suite")
        
        try:
            # Setup test environment
            self.setup_test_environment()
            
            # Define tests
            tests = [
                ("Task 1: Load and Validate", self.test_task1_load_validate),
                ("Task 2.1: Derived Features", self.test_task2_1_derived_features),
                ("Task 2.2: Interaction Features", self.test_task2_2_interaction_features),
                ("Task 2.3: Transformations", self.test_task2_3_transformations),
                ("Task 3.1: Advanced Preprocessing", self.test_task3_1_advanced_preprocessing),
                ("Task 4.1: Feature Selection", self.test_task4_1_feature_selection),
                ("Task 5.1: Data Quality", self.test_task5_1_data_quality),
                ("Task 6.1: Documentation", self.test_task6_1_documentation)
            ]
            
            # Run tests
            all_passed = True
            for test_name, test_function in tests:
                logger.info(f"\n{'='*60}")
                passed = self.run_test(test_name, test_function)
                if not passed:
                    all_passed = False
                    
            # Finalize results
            self.test_results['end_time'] = datetime.now().isoformat()
            self.test_results['overall_status'] = 'passed' if all_passed else 'failed'
            
            # Print summary
            self.print_test_summary()
            
            return all_passed
            
        finally:
            # Cleanup
            self.cleanup_test_environment()
            
    def print_test_summary(self) -> None:
        """
        Print test summary.
        """
        print(f"\n{'='*80}")
        print("🧪 PHASE 4 FEATURE ENGINEERING - TEST SUMMARY")
        print(f"{'='*80}")
        
        print(f"\n📊 OVERALL STATUS: {self.test_results['overall_status'].upper()}")
        print(f"✅ Tests Passed: {self.test_results['tests_passed']}")
        print(f"❌ Tests Failed: {self.test_results['tests_failed']}")
        print(f"📈 Success Rate: {self.test_results['tests_passed'] / (self.test_results['tests_passed'] + self.test_results['tests_failed']) * 100:.1f}%")
        
        print(f"\n📋 TEST DETAILS:")
        for test in self.test_results['test_details']:
            status_icon = "✅" if test['status'] == 'PASSED' else "❌"
            duration = test.get('duration_seconds', 0)
            print(f"   {status_icon} {test['test_name']} ({duration:.2f}s)")
            
            if test['status'] == 'FAILED' and 'error' in test:
                print(f"      Error: {test['error'][:100]}...")
                
        if self.test_results['overall_status'] == 'passed':
            print(f"\n🎉 All tests passed! Phase 4 implementations are ready for execution.")
        else:
            print(f"\n⚠️ Some tests failed. Please review and fix issues before running Phase 4.")
            
        print(f"\n{'='*80}")


def main():
    """
    Main function to run Phase 4 tests.
    """
    try:
        # Create test suite
        test_suite = Phase4TestSuite()
        
        # Run all tests
        success = test_suite.run_all_tests()
        
        if success:
            logger.info("🎉 All Phase 4 tests passed!")
            return 0
        else:
            logger.error("❌ Some Phase 4 tests failed.")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\n⏹️ Testing interrupted by user.")
        return 130
    except Exception as e:
        logger.error(f"💥 Testing failed with unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)