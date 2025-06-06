import pytest
import pandas as pd
import numpy as np
import sqlite3
import os
from pathlib import Path
from src.data.preprocessing import DataPreprocessor

# Define the path to the test database
TEST_DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'score.db')

@pytest.fixture(scope='module')
def db_path():
    """Provides the path to the test database."""
    if not os.path.exists(TEST_DB_PATH):
        pytest.skip(f"Test database not found at {TEST_DB_PATH}. Run download_db.py first.")
    return TEST_DB_PATH

@pytest.fixture
def preprocessor(db_path):
    """Provides a DataPreprocessor instance for testing."""
    return DataPreprocessor(db_path)

@pytest.fixture
def temp_db_with_missing(tmp_path):
    """Creates a temporary SQLite database with known missing values for testing."""
    db_file = tmp_path / "temp_score.db"
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Create table with same structure as original
    cursor.execute("""
        CREATE TABLE score (
            `index` INTEGER,
            number_of_siblings INTEGER,
            direct_admission TEXT,
            CCA TEXT,
            learning_style TEXT,
            student_id TEXT,
            gender TEXT,
            tuition TEXT,
            final_test REAL,
            n_male REAL,
            n_female REAL,
            age REAL,
            hours_per_week REAL,
            attendance_rate REAL,
            sleep_time TEXT,
            wake_time TEXT,
            mode_of_transport TEXT,
            bag_color TEXT
        )
    """)
    
    # Insert test data with known missing values
    test_data = [
        (0, 1, 'Yes', 'Sports', 'Visual', 'A1', 'Male', 'Yes', 85.0, 15.0, 10.0, 16.0, 10.0, 95.0, '22:00', '6:00', 'private', 'blue'),
        (1, 0, 'No', 'Clubs', 'Auditory', 'A2', 'Female', 'No', None, 12.0, 13.0, 15.0, 8.0, None, '23:00', '7:00', 'public', 'red'),  # Missing final_test and attendance_rate
        (2, 2, 'Yes', 'Sports', 'Visual', 'A3', 'Male', 'Yes', 78.0, 15.0, 10.0, 16.0, 12.0, 88.0, '22:30', '6:30', 'walk', 'green'),
        (3, 1, 'No', 'Clubs', 'Auditory', 'A4', 'Female', 'No', 92.0, 12.0, 13.0, 15.0, 9.0, None, '22:00', '6:00', 'public', 'yellow'),  # Missing attendance_rate
        (4, 0, 'Yes', 'Arts', 'Visual', 'A5', 'Male', 'Yes', None, 20.0, 5.0, 16.0, 15.0, 90.0, '23:30', '5:30', 'private', 'black'),  # Missing final_test
    ]
    
    cursor.executemany("""
        INSERT INTO score VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, test_data)
    
    conn.commit()
    yield conn, str(db_file)
    conn.close()

@pytest.fixture
def temp_preprocessor(temp_db_with_missing):
    """Provides a DataPreprocessor instance with temporary test database."""
    conn, db_file = temp_db_with_missing
    return DataPreprocessor(db_file)

class TestDataPreprocessor:
    """Test suite for DataPreprocessor class."""
    
    def test_load_data(self, preprocessor):
        """Test data loading from SQLite database."""
        data = preprocessor.load_data()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 15900  # Expected number of records
        assert 'final_test' in data.columns
        assert 'attendance_rate' in data.columns
        assert preprocessor.data is not None
    
    def test_detect_missing_values(self, preprocessor):
        """Test missing value detection accuracy."""
        preprocessor.load_data()
        missing_counts = preprocessor.detect_missing_values()
        
        # Based on db-structure.md, expect missing values in final_test and attendance_rate
        assert 'final_test' in missing_counts
        assert 'attendance_rate' in missing_counts
        assert missing_counts['final_test'] == 495  # 3.1% of 15900
        assert missing_counts['attendance_rate'] == 778  # 4.9% of 15900
    
    def test_create_missing_indicators(self, temp_preprocessor):
        """Test missing indicator creation for attendance_rate."""
        temp_preprocessor.load_data()
        
        indicators = temp_preprocessor.create_missing_indicators(['final_test', 'attendance_rate'])
        
        assert 'final_test_missing' in indicators.columns
        assert 'attendance_rate_missing' in indicators.columns
        assert indicators['final_test_missing'].sum() == 2  # 2 missing final_test in test data
        assert indicators['attendance_rate_missing'].sum() == 2  # 2 missing attendance_rate in test data
        assert indicators['final_test_missing'].dtype == int
        assert indicators['attendance_rate_missing'].dtype == int
    
    def test_impute_attendance_rate_by_subgroups(self, temp_preprocessor):
        """Test imputation strategies preserve data distribution."""
        temp_preprocessor.load_data()
        
        # Get original attendance_rate values
        original_attendance = temp_preprocessor.data['attendance_rate'].copy()
        missing_mask = original_attendance.isnull()
        
        # Impute missing values
        imputed_attendance = temp_preprocessor.impute_attendance_rate()
        
        # Check that no missing values remain
        assert imputed_attendance.isnull().sum() == 0
        
        # Check that non-missing values are unchanged
        non_missing_mask = ~missing_mask
        pd.testing.assert_series_equal(
            original_attendance[non_missing_mask], 
            imputed_attendance[non_missing_mask]
        )
        
        # Check that missing values were filled
        assert (imputed_attendance[missing_mask] > 0).all()
    
    def test_exclude_missing_final_test(self, temp_preprocessor):
        """Test final_test missing values are properly excluded from training."""
        temp_preprocessor.load_data()
        
        training_data, prediction_data = temp_preprocessor.exclude_missing_final_test(temp_preprocessor.data)
        
        # Check that training data has no missing final_test
        assert training_data['final_test'].isnull().sum() == 0
        
        # Check that prediction data has only missing final_test
        assert prediction_data['final_test'].isnull().sum() == len(prediction_data)
        
        # Check that total records are preserved
        assert len(training_data) + len(prediction_data) == len(temp_preprocessor.data)
        
        # In test data: 2 records have missing final_test
        assert len(training_data) == 3
        assert len(prediction_data) == 2
    
    def test_process_missing_data_complete_pipeline(self, temp_preprocessor):
        """Test complete missing data processing pipeline."""
        temp_preprocessor.load_data()
        
        # Process missing data
        result = temp_preprocessor.process_missing_data(
            create_indicators=True,
            impute_attendance=True,
            exclude_missing_target=True
        )
        
        # Check that all expected datasets are created
        assert 'full_processed' in result
        assert 'training' in result
        assert 'prediction' in result
        
        full_data = result['full_processed']
        training_data = result['training']
        prediction_data = result['prediction']
        
        # Check missing indicators were created
        assert 'final_test_missing' in full_data.columns
        assert 'attendance_rate_missing' in full_data.columns
        
        # Check attendance_rate was imputed
        assert full_data['attendance_rate'].isnull().sum() == 0
        
        # Check training/prediction split
        assert training_data['final_test'].isnull().sum() == 0
        assert prediction_data['final_test'].isnull().sum() == len(prediction_data)
    
    def test_save_processed_data(self, temp_preprocessor, tmp_path):
        """Test saving processed data to CSV files."""
        temp_preprocessor.load_data()
        temp_preprocessor.process_missing_data()
        
        output_dir = tmp_path / "processed"
        temp_preprocessor.save_processed_data(str(output_dir))
        
        # Check that files were created
        assert (output_dir / "full_processed.csv").exists()
        assert (output_dir / "training.csv").exists()
        assert (output_dir / "prediction.csv").exists()
        
        # Check that files contain data
        full_data = pd.read_csv(output_dir / "full_processed.csv")
        training_data = pd.read_csv(output_dir / "training.csv")
        prediction_data = pd.read_csv(output_dir / "prediction.csv")
        
        assert len(full_data) == 5  # Test data has 5 records
        assert len(training_data) == 3  # 3 records with final_test
        assert len(prediction_data) == 2  # 2 records without final_test
    
    def test_get_missing_summary(self, temp_preprocessor):
        """Test missing data summary generation."""
        temp_preprocessor.load_data()
        
        # Get initial summary
        summary = temp_preprocessor.get_missing_summary()
        
        assert 'original_record_count' in summary
        assert 'original_missing_values' in summary
        assert 'missing_percentages' in summary
        assert summary['original_record_count'] == 5
        
        # Process data and get updated summary
        temp_preprocessor.process_missing_data()
        final_summary = temp_preprocessor.get_missing_summary()
        
        assert 'training_record_count' in final_summary
        assert 'prediction_record_count' in final_summary
        assert final_summary['training_record_count'] == 3
        assert final_summary['prediction_record_count'] == 2
    
    def test_imputation_with_real_data(self, preprocessor):
        """Test imputation with real database to ensure it works with actual data."""
        preprocessor.load_data()
        
        # Get original missing count
        original_missing = preprocessor.data['attendance_rate'].isnull().sum()
        assert original_missing == 778  # Expected from db-structure.md
        
        # Impute missing values
        imputed_attendance = preprocessor.impute_attendance_rate()
        
        # Check that all missing values were imputed
        assert imputed_attendance.isnull().sum() == 0
        
        # Check that imputed values are reasonable (between 40 and 100 based on data range)
        assert (imputed_attendance >= 40).all()
        assert (imputed_attendance <= 100).all()
    
    def test_data_distribution_preservation(self, preprocessor):
        """Test that imputation strategies preserve data distribution."""
        preprocessor.load_data()
        
        # Get original statistics for non-missing values
        original_attendance = preprocessor.data['attendance_rate'].dropna()
        original_mean = original_attendance.mean()
        original_std = original_attendance.std()
        
        # Impute missing values
        imputed_attendance = preprocessor.impute_attendance_rate()
        
        # Check that the distribution hasn't changed dramatically
        # Allow for some variation due to imputation
        new_mean = imputed_attendance.mean()
        new_std = imputed_attendance.std()
        
        # Mean should be within 5% of original
        assert abs(new_mean - original_mean) / original_mean < 0.05
        
        # Standard deviation should be within 10% of original
        assert abs(new_std - original_std) / original_std < 0.10


# Integration test
def test_full_preprocessing_pipeline_integration(db_path, tmp_path):
    """Integration test for the complete preprocessing pipeline."""
    preprocessor = DataPreprocessor(db_path)
    
    # Load data
    data = preprocessor.load_data()
    assert len(data) == 15900
    
    # Process missing data
    result = preprocessor.process_missing_data()
    
    # Verify results
    training_data = result['training']
    prediction_data = result['prediction']
    
    # Check that training data has no missing final_test
    assert training_data['final_test'].isnull().sum() == 0
    assert len(training_data) == 15900 - 495  # Total - missing final_test
    
    # Check that prediction data has only missing final_test
    assert prediction_data['final_test'].isnull().sum() == len(prediction_data)
    assert len(prediction_data) == 495
    
    # Check that attendance_rate was imputed in both datasets
    assert training_data['attendance_rate'].isnull().sum() == 0
    assert prediction_data['attendance_rate'].isnull().sum() == 0
    
    # Save and verify files
    output_dir = tmp_path / "integration_test"
    preprocessor.save_processed_data(str(output_dir))
    
    assert (output_dir / "training.csv").exists()
    assert (output_dir / "prediction.csv").exists()
    assert (output_dir / "full_processed.csv").exists()


# To run these tests, navigate to the project root and use:
# pytest tests/data/test_preprocessing.py -v
# Ensure score.db is present in data/raw/