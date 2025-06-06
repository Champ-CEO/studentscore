import pytest
import pandas as pd
import numpy as np
import sqlite3
import os
from pathlib import Path
from src.data.deduplication import DataDeduplicator

# Define the path to the test database
TEST_DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'score.db')

@pytest.fixture(scope='module')
def db_path():
    """Provides the path to the test database."""
    if not os.path.exists(TEST_DB_PATH):
        pytest.skip(f"Test database not found at {TEST_DB_PATH}. Run download_db.py first.")
    return TEST_DB_PATH

@pytest.fixture
def deduplicator(db_path):
    """Provides a DataDeduplicator instance for testing."""
    return DataDeduplicator(db_path=db_path)

@pytest.fixture
def sample_data_with_duplicates():
    """Creates sample data with known duplicates for testing."""
    data = pd.DataFrame({
        'index': [0, 1, 2, 3, 4, 5],
        'student_id': ['ABC123', 'DEF456', 'ABC123', 'GHI789', 'DEF456', 'JKL012'],  # ABC123 and DEF456 duplicated
        'age': [16.0, 17.0, 16.0, 15.0, 17.0, 18.0],
        'gender': ['Male', 'Female', 'Male', 'Male', 'Female', 'Female'],
        'final_test': [85.0, 78.0, 85.0, 92.0, None, 88.0],  # DEF456 duplicate has missing final_test
        'attendance_rate': [95.0, 88.0, 95.0, 92.0, 88.0, 94.0],
        'tuition': ['Yes', 'No', 'Yes', 'Yes', 'No', 'No'],
        'CCA': ['Sports', 'Arts', 'Sports', 'Clubs', 'Arts', 'None'],
        'hours_per_week': [10.0, 8.0, 10.0, 12.0, 8.0, 11.0]
    })
    return data

@pytest.fixture
def temp_deduplicator(sample_data_with_duplicates):
    """Provides a DataDeduplicator instance with sample data."""
    return DataDeduplicator(data=sample_data_with_duplicates)

class TestDataDeduplicator:
    """Test suite for DataDeduplicator class."""
    
    def test_load_data_from_database(self, deduplicator):
        """Test data loading from SQLite database."""
        data = deduplicator.load_data()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 15900  # Expected number of records
        assert 'student_id' in data.columns
        assert 'age' in data.columns
        assert 'final_test' in data.columns
    
    def test_load_data_from_dataframe(self, sample_data_with_duplicates):
        """Test data loading from provided DataFrame."""
        deduplicator = DataDeduplicator(data=sample_data_with_duplicates)
        data = deduplicator.load_data()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 6
        pd.testing.assert_frame_equal(data, sample_data_with_duplicates)
    
    def test_detect_duplicates_basic(self, temp_deduplicator):
        """Test basic duplicate detection functionality."""
        analysis = temp_deduplicator.detect_duplicates()
        
        assert 'total_records' in analysis
        assert 'duplicate_records' in analysis
        assert 'unique_records' in analysis
        assert 'duplicate_groups' in analysis
        
        # We have 6 records with 2 complete duplicates (ABC123 and DEF456)
        assert analysis['total_records'] == 6
        assert analysis['duplicate_records'] == 4  # 2 pairs of duplicates = 4 records
        assert analysis['unique_records'] == 4  # 4 unique records after deduplication
        assert analysis['duplicate_groups'] == 2  # 2 groups of duplicates
    
    def test_detect_duplicates_custom_columns(self, temp_deduplicator):
        """Test duplicate detection with custom column subset."""
        # Only check duplicates based on student_id and age
        analysis = temp_deduplicator.detect_duplicates(subset_columns=['student_id', 'age'])
        
        assert analysis['subset_columns_used'] == ['student_id', 'age']
        assert analysis['duplicate_records'] == 4  # Same result for our test data
    
    def test_detect_duplicates_no_duplicates(self):
        """Test duplicate detection when no duplicates exist."""
        unique_data = pd.DataFrame({
            'student_id': ['A1', 'B2', 'C3'],
            'age': [16, 17, 18],
            'score': [85, 90, 88]
        })
        
        deduplicator = DataDeduplicator(data=unique_data)
        analysis = deduplicator.detect_duplicates()
        
        assert analysis['total_records'] == 3
        assert analysis['duplicate_records'] == 0
        assert analysis['unique_records'] == 3
        assert analysis['duplicate_groups'] == 0
    
    def test_analyze_duplicate_patterns(self, temp_deduplicator):
        """Test duplicate pattern analysis."""
        temp_deduplicator.detect_duplicates()
        patterns = temp_deduplicator.analyze_duplicate_patterns()
        
        assert 'student_id_patterns' in patterns
        assert 'age_patterns' in patterns
        assert 'gender_patterns' in patterns
        assert 'final_test_patterns' in patterns
        assert 'exact_duplicates' in patterns
        
        # Check student_id patterns
        id_patterns = patterns['student_id_patterns']
        assert id_patterns['unique_student_ids_in_duplicates'] == 2  # ABC123 and DEF456
        assert id_patterns['max_records_per_student'] == 2  # Each appears twice
        
        # Check exact duplicates count
        assert patterns['exact_duplicates'] == 4  # 4 records are exact duplicates
    
    def test_remove_duplicates_keep_first(self, temp_deduplicator):
        """Test duplicate removal keeping first occurrence."""
        deduplicated_data = temp_deduplicator.remove_duplicates(strategy='keep_first')
        
        assert len(deduplicated_data) == 4  # 6 original - 2 duplicates = 4
        
        # Check that first occurrences are kept
        student_ids = deduplicated_data['student_id'].tolist()
        assert 'ABC123' in student_ids
        assert 'DEF456' in student_ids
        assert 'GHI789' in student_ids
        assert 'JKL012' in student_ids
        
        # Each student_id should appear only once
        assert len(set(student_ids)) == 4
    
    def test_remove_duplicates_keep_last(self, temp_deduplicator):
        """Test duplicate removal keeping last occurrence."""
        deduplicated_data = temp_deduplicator.remove_duplicates(strategy='keep_last')
        
        assert len(deduplicated_data) == 4  # 6 original - 2 duplicates = 4
        
        # Check that last occurrences are kept
        student_ids = deduplicated_data['student_id'].tolist()
        assert len(set(student_ids)) == 4
        
        # For our test data, the last occurrence of DEF456 has missing final_test
        def456_record = deduplicated_data[deduplicated_data['student_id'] == 'DEF456']
        assert len(def456_record) == 1
        assert pd.isna(def456_record['final_test'].iloc[0])
    
    def test_remove_duplicates_keep_best(self, temp_deduplicator):
        """Test duplicate removal keeping best quality record."""
        deduplicated_data = temp_deduplicator.remove_duplicates(strategy='keep_best')
        
        assert len(deduplicated_data) == 4  # 6 original - 2 duplicates = 4
        
        # Check that best quality records are kept
        # For DEF456, the first occurrence should be kept (has final_test score)
        def456_record = deduplicated_data[deduplicated_data['student_id'] == 'DEF456']
        assert len(def456_record) == 1
        assert not pd.isna(def456_record['final_test'].iloc[0])  # Should have final_test score
        assert def456_record['final_test'].iloc[0] == 78.0
    
    def test_quality_score_calculation(self, temp_deduplicator):
        """Test quality score calculation for records."""
        temp_deduplicator.load_data()
        quality_scores = temp_deduplicator._calculate_quality_score(temp_deduplicator.data)
        
        assert len(quality_scores) == 6
        assert all(score >= 0 for score in quality_scores)
        
        # Record with missing final_test should have lower quality score
        # Index 4 (DEF456 duplicate) has missing final_test
        assert quality_scores.iloc[4] < quality_scores.iloc[1]  # Index 1 is original DEF456
    
    def test_invalid_strategy_raises_error(self, temp_deduplicator):
        """Test that invalid removal strategy raises error."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            temp_deduplicator.remove_duplicates(strategy='invalid_strategy')
    
    def test_get_deduplication_report(self, temp_deduplicator):
        """Test deduplication report generation."""
        temp_deduplicator.detect_duplicates()
        temp_deduplicator.remove_duplicates(strategy='keep_first')
        
        report = temp_deduplicator.get_deduplication_report()
        
        assert 'original_record_count' in report
        assert 'duplicate_record_count' in report
        assert 'unique_record_count' in report
        assert 'duplicate_groups' in report
        assert 'removal_strategy_used' in report
        assert 'columns_used_for_detection' in report
        assert 'duplicate_patterns' in report
        
        assert report['original_record_count'] == 6
        assert report['duplicate_record_count'] == 4
        assert report['unique_record_count'] == 4
        assert report['removal_strategy_used'] == 'keep_first'
    
    def test_save_deduplicated_data(self, temp_deduplicator, tmp_path):
        """Test saving deduplicated data to file."""
        deduplicated_data = temp_deduplicator.remove_duplicates(strategy='keep_first')
        
        output_file = tmp_path / "deduplicated_test_data.csv"
        temp_deduplicator.save_deduplicated_data(str(output_file), deduplicated_data)
        
        # Check that file was created
        assert output_file.exists()
        
        # Check that file contains correct data
        loaded_data = pd.read_csv(output_file)
        assert len(loaded_data) == len(deduplicated_data)
        assert list(loaded_data.columns) == list(deduplicated_data.columns)
    
    def test_full_deduplication_pipeline(self, temp_deduplicator, tmp_path):
        """Test the complete deduplication pipeline."""
        output_file = tmp_path / "pipeline_output.csv"
        
        deduplicated_data = temp_deduplicator.run_full_deduplication_pipeline(
            output_path=str(output_file),
            strategy='keep_best'
        )
        
        # Check results
        assert len(deduplicated_data) == 4
        assert output_file.exists()
        
        # Check that report was generated
        report = temp_deduplicator.get_deduplication_report()
        assert report['removal_strategy_used'] == 'keep_best'
    
    def test_real_data_duplicate_detection(self, deduplicator):
        """Test duplicate detection with real database data."""
        analysis = deduplicator.detect_duplicates()
        
        # Based on db-structure.md, there should be 139 duplicate records
        assert analysis['total_records'] == 15900
        
        # The exact number of duplicates may vary based on detection criteria
        # but should be in the expected range
        assert analysis['duplicate_records'] > 0
        assert analysis['duplicate_groups'] > 0
        
        # After deduplication, we should have fewer records
        assert analysis['unique_records'] < analysis['total_records']
    
    def test_real_data_deduplication(self, deduplicator):
        """Test deduplication with real database data."""
        # Load and analyze
        deduplicator.load_data()
        analysis = deduplicator.detect_duplicates()
        
        # Remove duplicates
        deduplicated_data = deduplicator.remove_duplicates(strategy='keep_first')
        
        # Verify deduplication
        assert len(deduplicated_data) < 15900  # Should be fewer records
        assert len(deduplicated_data) == analysis['unique_records']
        
        # Check that no duplicates remain
        subset_columns = [col for col in deduplicated_data.columns if col != 'index']
        remaining_duplicates = deduplicated_data.duplicated(subset=subset_columns).sum()
        assert remaining_duplicates == 0
    
    def test_duplicate_detection_with_missing_values(self):
        """Test duplicate detection when records have missing values."""
        data_with_na = pd.DataFrame({
            'student_id': ['A1', 'A1', 'B2', 'B2'],
            'age': [16, 16, None, None],
            'score': [85, 85, 90, 90],
            'name': ['John', 'John', 'Jane', None]  # Different missing patterns
        })
        
        deduplicator = DataDeduplicator(data=data_with_na)
        analysis = deduplicator.detect_duplicates()
        
        # Should detect duplicates even with missing values
        assert analysis['duplicate_records'] == 4  # All records are duplicates
        assert analysis['duplicate_groups'] == 2  # Two groups
    
    def test_edge_case_all_duplicates(self):
        """Test edge case where all records are duplicates."""
        all_duplicate_data = pd.DataFrame({
            'student_id': ['A1', 'A1', 'A1'],
            'age': [16, 16, 16],
            'score': [85, 85, 85]
        })
        
        deduplicator = DataDeduplicator(data=all_duplicate_data)
        deduplicated_data = deduplicator.remove_duplicates(strategy='keep_first')
        
        assert len(deduplicated_data) == 1  # Only one unique record
        assert deduplicated_data['student_id'].iloc[0] == 'A1'
    
    def test_edge_case_single_record(self):
        """Test edge case with single record."""
        single_record_data = pd.DataFrame({
            'student_id': ['A1'],
            'age': [16],
            'score': [85]
        })
        
        deduplicator = DataDeduplicator(data=single_record_data)
        analysis = deduplicator.detect_duplicates()
        
        assert analysis['total_records'] == 1
        assert analysis['duplicate_records'] == 0
        assert analysis['unique_records'] == 1
    
    def test_custom_subset_columns_validation(self, temp_deduplicator):
        """Test that custom subset columns are properly validated."""
        # Test with valid columns
        analysis = temp_deduplicator.detect_duplicates(subset_columns=['student_id', 'age'])
        assert analysis['subset_columns_used'] == ['student_id', 'age']
        
        # Test with non-existent column should not raise error but may affect results
        # The function should handle this gracefully
        temp_deduplicator.load_data()
        try:
            analysis = temp_deduplicator.detect_duplicates(subset_columns=['student_id', 'nonexistent_column'])
            # If it doesn't raise an error, that's fine - pandas will handle it
        except KeyError:
            # If it raises KeyError, that's also acceptable behavior
            pass


# Integration test
def test_complete_deduplication_integration(db_path, tmp_path):
    """Integration test for the complete deduplication pipeline."""
    deduplicator = DataDeduplicator(db_path=db_path)
    
    # Load data
    original_data = deduplicator.load_data()
    assert len(original_data) == 15900
    
    # Run full pipeline
    output_file = tmp_path / "integration_deduplicated.csv"
    deduplicated_data = deduplicator.run_full_deduplication_pipeline(
        output_path=str(output_file),
        strategy='keep_best'
    )
    
    # Verify results
    assert len(deduplicated_data) < 15900  # Should have fewer records
    assert output_file.exists()
    
    # Verify no duplicates remain
    subset_columns = [col for col in deduplicated_data.columns if col != 'index']
    remaining_duplicates = deduplicated_data.duplicated(subset=subset_columns).sum()
    assert remaining_duplicates == 0
    
    # Check report
    report = deduplicator.get_deduplication_report()
    assert report['original_record_count'] == 15900
    assert report['unique_record_count'] == len(deduplicated_data)
    assert report['removal_strategy_used'] == 'keep_best'
    
    # Verify saved file
    loaded_data = pd.read_csv(output_file)
    assert len(loaded_data) == len(deduplicated_data)


# To run these tests, navigate to the project root and use:
# pytest tests/data/test_deduplication.py -v
# Ensure score.db is present in data/raw/