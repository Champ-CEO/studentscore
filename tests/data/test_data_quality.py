import pytest
import pandas as pd
import numpy as np
import sqlite3
import os
from pathlib import Path
from src.data.data_quality import DataQualityFixer

# Define the path to the test database
TEST_DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'score.db')

@pytest.fixture(scope='module')
def db_path():
    """Provides the path to the test database."""
    if not os.path.exists(TEST_DB_PATH):
        pytest.skip(f"Test database not found at {TEST_DB_PATH}. Run download_db.py first.")
    return TEST_DB_PATH

@pytest.fixture
def quality_fixer(db_path):
    """Provides a DataQualityFixer instance for testing."""
    return DataQualityFixer(db_path)

@pytest.fixture
def sample_data_with_issues():
    """Creates sample data with known quality issues for testing."""
    data = pd.DataFrame({
        'index': [0, 1, 2, 3, 4],
        'number_of_siblings': [1, 0, 2, 1, 0],
        'direct_admission': ['Yes', 'no', 'YES', 'No', 'yes'],
        'CCA': ['Sports', 'CLUBS', 'Arts', 'SPORTS', 'None'],
        'learning_style': ['Visual', 'auditory', 'VISUAL', 'Auditory', 'visual'],
        'student_id': ['ABC123', 'DEF456', 'GHI789', 'ABC123', 'JKL012'],  # Duplicate ABC123
        'gender': ['Male', 'female', 'MALE', 'Female', 'male'],
        'tuition': ['Yes', 'Y', 'No', 'N', 'yes'],
        'final_test': [85.0, 78.0, 92.0, 67.0, 88.0],
        'n_male': [15.0, 12.0, 18.0, 14.0, 16.0],
        'n_female': [10.0, 13.0, 7.0, 11.0, 9.0],
        'age': [16.0, -5.0, 15.0, 17.0, 25.5],  # Negative age and unreasonable age
        'hours_per_week': [10.0, 8.0, 12.0, 9.0, 11.0],
        'attendance_rate': [95.0, 88.0, 92.0, 90.0, 94.0],
        'sleep_time': ['22:00', '23:00', '22:30', '23:30', '22:00'],
        'wake_time': ['6:00', '7:00', '6:30', '5:30', '6:00'],
        'mode_of_transport': ['private', 'public', 'walk', 'private', 'public'],
        'bag_color': ['blue', 'red', 'green', 'yellow', 'black']
    })
    return data

@pytest.fixture
def temp_quality_fixer(sample_data_with_issues):
    """Provides a DataQualityFixer instance with sample data."""
    return DataQualityFixer(data=sample_data_with_issues)

class TestDataQualityFixer:
    """Test suite for DataQualityFixer class."""
    
    def test_load_data_from_database(self, quality_fixer):
        """Test data loading from SQLite database."""
        data = quality_fixer.load_data()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 15900  # Expected number of records
        assert 'student_id' in data.columns
        assert 'age' in data.columns
        assert 'tuition' in data.columns
        assert 'CCA' in data.columns
    
    def test_load_data_from_dataframe(self, sample_data_with_issues):
        """Test data loading from provided DataFrame."""
        fixer = DataQualityFixer(data=sample_data_with_issues)
        data = fixer.load_data()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 5
        pd.testing.assert_frame_equal(data, sample_data_with_issues)
    
    def test_analyze_student_id_uniqueness(self, temp_quality_fixer):
        """Test student_id uniqueness check."""
        temp_quality_fixer.load_data()
        analysis = temp_quality_fixer.analyze_student_id()
        
        assert 'total_records' in analysis
        assert 'unique_ids' in analysis
        assert 'duplicate_count' in analysis
        assert analysis['total_records'] == 5
        assert analysis['unique_ids'] == 4  # ABC123 is duplicated
        assert analysis['duplicate_count'] == 1
        assert 'duplicate_ids' in analysis
        assert 'ABC123' in analysis['duplicate_ids']
    
    def test_analyze_student_id_format_validation(self, temp_quality_fixer):
        """Test student_id format validation and pattern recognition."""
        temp_quality_fixer.load_data()
        analysis = temp_quality_fixer.analyze_student_id()
        
        assert 'id_lengths' in analysis
        assert 'id_patterns' in analysis
        assert 'character_analysis' in analysis
        assert 'most_common_length' in analysis
        
        # All test IDs should be length 6
        assert analysis['most_common_length'] == 6
        assert analysis['character_analysis']['has_letters'] > 0
        assert analysis['character_analysis']['has_numbers'] > 0
    
    def test_extract_id_features(self, temp_quality_fixer):
        """Test extraction of embedded information from ID structure."""
        temp_quality_fixer.load_data()
        id_features = temp_quality_fixer.extract_id_features()
        
        expected_features = [
            'id_length', 'id_numeric_count', 'id_alpha_count', 'id_special_count',
            'id_first_char', 'id_last_char', 'id_first_is_letter', 'id_last_is_number',
            'id_prefix_2', 'id_suffix_2', 'id_complexity'
        ]
        
        for feature in expected_features:
            assert feature in id_features.columns
        
        # Test specific values
        assert id_features.loc[0, 'id_length'] == 6  # ABC123
        assert id_features.loc[0, 'id_alpha_count'] == 3
        assert id_features.loc[0, 'id_numeric_count'] == 3
        assert id_features.loc[0, 'id_first_char'] == 'A'
        assert id_features.loc[0, 'id_last_char'] == '3'
    
    def test_fix_negative_age_values(self, temp_quality_fixer):
        """Test negative age values are corrected/removed."""
        temp_quality_fixer.load_data()
        corrected_age = temp_quality_fixer.fix_age_issues()
        
        # No negative ages should remain
        assert (corrected_age >= 0).all()
        
        # Check that reasonable negative age (-5) was converted to positive (5)
        # Index 1 had age -5, should become 5
        assert corrected_age.iloc[1] == 5.0
        
        # Check that unreasonable age (25.5) was replaced with median
        median_reasonable_age = temp_quality_fixer.data['age'][(temp_quality_fixer.data['age'] >= 10) & (temp_quality_fixer.data['age'] <= 20)].median()
        assert corrected_age.iloc[4] == median_reasonable_age
    
    def test_create_age_groups(self, temp_quality_fixer):
        """Test age-based feature creation (e.g., age groups) is correct."""
        temp_quality_fixer.load_data()
        corrected_age = temp_quality_fixer.fix_age_issues()
        age_features = temp_quality_fixer.create_age_features(corrected_age)
        
        expected_age_features = [
            'age_group', 'age_above_median', 'age_deviation_from_median',
            'is_youngest', 'is_oldest'
        ]
        
        for feature in expected_age_features:
            assert feature in age_features.columns
        
        # Test age group categorization
        assert age_features['age_group'].dtype.name == 'category'
        
        # Test binary features
        assert age_features['age_above_median'].dtype == int
        assert age_features['is_youngest'].dtype == int
        assert age_features['is_oldest'].dtype == int
    
    def test_categorical_standardization_tuition(self, temp_quality_fixer):
        """Test categorical standardization (Y/N → Yes/No)."""
        temp_quality_fixer.load_data()
        standardized_data = temp_quality_fixer.standardize_categorical_values()
        
        # Check that Y/N were converted to Yes/No
        tuition_values = standardized_data['tuition'].unique()
        assert 'Y' not in tuition_values
        assert 'N' not in tuition_values
        assert 'Yes' in tuition_values
        assert 'No' in tuition_values
        
        # Check specific conversions
        assert standardized_data.loc[1, 'tuition'] == 'Yes'  # Y → Yes
        assert standardized_data.loc[3, 'tuition'] == 'No'   # N → No
    
    def test_categorical_standardization_cca_case(self, temp_quality_fixer):
        """Test case standardization (CLUBS → Clubs)."""
        temp_quality_fixer.load_data()
        standardized_data = temp_quality_fixer.standardize_categorical_values()
        
        # Check that uppercase CCA values were converted to proper case
        cca_values = standardized_data['CCA'].unique()
        assert 'CLUBS' not in cca_values
        assert 'SPORTS' not in cca_values
        assert 'Clubs' in cca_values
        assert 'Sports' in cca_values
        
        # Check specific conversions
        assert standardized_data.loc[1, 'CCA'] == 'Clubs'   # CLUBS → Clubs
        assert standardized_data.loc[3, 'CCA'] == 'Sports'  # SPORTS → Sports
    
    def test_data_type_consistency(self, temp_quality_fixer):
        """Test data type consistency enforcement."""
        temp_quality_fixer.load_data()
        standardized_data = temp_quality_fixer.standardize_categorical_values()
        typed_data = temp_quality_fixer.enforce_data_types(standardized_data)
        
        # Check specific data types
        assert typed_data['index'].dtype == 'int64'
        assert typed_data['number_of_siblings'].dtype == 'int64'
        assert typed_data['direct_admission'].dtype.name == 'category'
        assert typed_data['CCA'].dtype.name == 'category'
        assert typed_data['gender'].dtype.name == 'category'
        assert typed_data['tuition'].dtype.name == 'category'
        assert typed_data['final_test'].dtype == 'float64'
        assert typed_data['age'].dtype == 'float64'
    
    def test_complete_quality_fixing_pipeline(self, temp_quality_fixer):
        """Test the complete data quality fixing pipeline."""
        cleaned_data = temp_quality_fixer.fix_all_quality_issues()
        
        # Check that all quality issues were addressed
        assert isinstance(cleaned_data, pd.DataFrame)
        assert len(cleaned_data) == 5  # Same number of records
        
        # Check age fixes
        assert (cleaned_data['age'] >= 0).all()
        
        # Check categorical standardization
        assert 'Y' not in cleaned_data['tuition'].values
        assert 'N' not in cleaned_data['tuition'].values
        assert 'CLUBS' not in cleaned_data['CCA'].values
        
        # Check new features were added
        assert 'age_group' in cleaned_data.columns
        assert 'id_length' in cleaned_data.columns
        
        # Check data types
        assert cleaned_data['CCA'].dtype.name == 'category'
        assert cleaned_data['age'].dtype == 'float64'
    
    def test_quality_report_generation(self, temp_quality_fixer):
        """Test quality report generation."""
        temp_quality_fixer.fix_all_quality_issues()
        report = temp_quality_fixer.get_quality_report()
        
        assert 'student_id_analysis' in report
        assert 'age_fixes' in report
        assert 'categorical_standardization' in report
        
        # Check student_id analysis details
        id_analysis = report['student_id_analysis']
        assert 'duplicate_count' in id_analysis
        assert 'unique_ids' in id_analysis
        
        # Check age fixes details
        age_fixes = report['age_fixes']
        assert 'original_negative_count' in age_fixes
        assert 'final_age_range' in age_fixes
    
    def test_save_cleaned_data(self, temp_quality_fixer, tmp_path):
        """Test saving cleaned data to file."""
        cleaned_data = temp_quality_fixer.fix_all_quality_issues()
        
        output_file = tmp_path / "cleaned_test_data.csv"
        temp_quality_fixer.save_cleaned_data(str(output_file))
        
        # Check that file was created
        assert output_file.exists()
        
        # Check that file contains correct data
        loaded_data = pd.read_csv(output_file)
        assert len(loaded_data) == len(cleaned_data)
        assert list(loaded_data.columns) == list(cleaned_data.columns)
    
    def test_real_data_quality_fixes(self, quality_fixer):
        """Test quality fixes with real database data."""
        quality_fixer.load_data()
        
        # Test student_id analysis
        id_analysis = quality_fixer.analyze_student_id()
        assert id_analysis['total_records'] == 15900
        assert id_analysis['unique_ids'] == 15000  # Based on db-structure.md
        assert id_analysis['duplicate_count'] == 900
        
        # Test age fixes
        corrected_age = quality_fixer.fix_age_issues()
        assert (corrected_age >= 0).all()  # No negative ages
        assert corrected_age.min() >= 10   # Reasonable minimum age
        assert corrected_age.max() <= 25   # Reasonable maximum age
        
        # Test categorical standardization
        standardized_data = quality_fixer.standardize_categorical_values()
        
        # Check tuition standardization
        tuition_values = set(standardized_data['tuition'].unique())
        assert 'Y' not in tuition_values
        assert 'N' not in tuition_values
        assert tuition_values.issubset({'Yes', 'No'})
        
        # Check CCA case standardization
        cca_values = set(standardized_data['CCA'].unique())
        assert 'CLUBS' not in cca_values
        assert 'SPORTS' not in cca_values
        assert 'ARTS' not in cca_values
    
    def test_feature_engineering_from_id_patterns(self, quality_fixer):
        """Test feature engineering from ID-derived patterns."""
        quality_fixer.load_data()
        id_features = quality_fixer.extract_id_features()
        
        # Check that meaningful features were extracted
        assert 'id_length' in id_features.columns
        assert 'id_numeric_count' in id_features.columns
        assert 'id_alpha_count' in id_features.columns
        assert 'id_complexity' in id_features.columns
        
        # Check feature value ranges
        assert id_features['id_length'].min() >= 0
        assert id_features['id_numeric_count'].min() >= 0
        assert id_features['id_alpha_count'].min() >= 0
        assert id_features['id_complexity'].min() >= 0
        assert id_features['id_complexity'].max() <= 3  # Max complexity score
    
    def test_id_retention_decision(self, temp_quality_fixer):
        """Test decision on ID retention vs. removal after feature extraction."""
        cleaned_data = temp_quality_fixer.fix_all_quality_issues()
        
        # Original student_id should still be present for traceability
        assert 'student_id' in cleaned_data.columns
        
        # But we should also have extracted features
        id_feature_columns = [col for col in cleaned_data.columns if col.startswith('id_')]
        assert len(id_feature_columns) > 0
        
        # The extracted features should provide value beyond the original ID
        assert 'id_length' in id_feature_columns
        assert 'id_complexity' in id_feature_columns


# Integration test
def test_complete_data_quality_pipeline_integration(db_path, tmp_path):
    """Integration test for the complete data quality pipeline."""
    fixer = DataQualityFixer(db_path)
    
    # Load data
    original_data = fixer.load_data()
    assert len(original_data) == 15900
    
    # Fix all quality issues
    cleaned_data = fixer.fix_all_quality_issues()
    
    # Verify comprehensive fixes
    assert len(cleaned_data) == 15900  # Same number of records
    assert cleaned_data.shape[1] > original_data.shape[1]  # More columns due to feature engineering
    
    # Check age fixes
    assert (cleaned_data['age'] >= 0).all()
    assert cleaned_data['age'].min() >= 10
    assert cleaned_data['age'].max() <= 25
    
    # Check categorical standardization
    assert 'Y' not in cleaned_data['tuition'].values
    assert 'N' not in cleaned_data['tuition'].values
    assert 'CLUBS' not in cleaned_data['CCA'].values
    
    # Check new features
    assert 'age_group' in cleaned_data.columns
    assert 'id_length' in cleaned_data.columns
    assert 'id_complexity' in cleaned_data.columns
    
    # Save and verify
    output_file = tmp_path / "integration_cleaned_data.csv"
    fixer.save_cleaned_data(str(output_file))
    assert output_file.exists()
    
    # Get quality report
    report = fixer.get_quality_report()
    assert 'student_id_analysis' in report
    assert 'age_fixes' in report
    assert 'categorical_standardization' in report


# To run these tests, navigate to the project root and use:
# pytest tests/data/test_data_quality.py -v
# Ensure score.db is present in data/raw/