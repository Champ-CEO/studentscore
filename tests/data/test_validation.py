import pytest
import sqlite3
import os
from src.data.validation import DataValidator
from src.data.repository import ScoreRepository

# Define the path to the test database relative to this test file
TEST_DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'score.db')

@pytest.fixture(scope='module')
def db_path():
    """Provides the path to the test database."""
    if not os.path.exists(TEST_DB_PATH):
        pytest.skip(f"Test database not found at {TEST_DB_PATH}. Run download_db.py first.")
    return TEST_DB_PATH

@pytest.fixture(scope='module')
def validator(db_path):
    """Provides a DataValidator instance for testing."""
    return DataValidator(db_path)

@pytest.fixture
def temp_db(tmp_path):
    """Creates a temporary SQLite database for testing specific scenarios."
    db_file = tmp_path / "temp_score.db"
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
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
    conn.commit()
    yield conn, str(db_file)
    conn.close()

def test_validate_record_count(validator):
    """Test validation of record count."""
    # Assuming score.db has 15900 records as per db-structure.md
    assert validator.validate_record_count(expected_count=15900) is True
    assert validator.validate_record_count(expected_count=100) is False

def test_validate_features_present(validator):
    """Test validation of all expected features being present."""
    expected_features = {
        'index', 'number_of_siblings', 'direct_admission', 'CCA',
        'learning_style', 'student_id', 'gender', 'tuition',
        'final_test', 'n_male', 'n_female', 'age', 'hours_per_week',
        'attendance_rate', 'sleep_time', 'wake_time',
        'mode_of_transport', 'bag_color'
    }
    assert validator.validate_features_present(expected_features) is True
    assert validator.validate_features_present(expected_features | {'non_existent_feature'}) is False

def test_validate_data_types(validator):
    """Test validation of data types for a sample of records."""
    expected_types = {
        'index': 'INTEGER',
        'number_of_siblings': 'INTEGER',
        'direct_admission': 'TEXT',
        'CCA': 'TEXT',
        'learning_style': 'TEXT',
        'student_id': 'TEXT',
        'gender': 'TEXT',
        'tuition': 'TEXT',
        'final_test': 'REAL',
        'n_male': 'REAL',
        'n_female': 'REAL',
        'age': 'REAL',
        'hours_per_week': 'REAL',
        'attendance_rate': 'REAL',
        'sleep_time': 'TEXT',
        'wake_time': 'TEXT',
        'mode_of_transport': 'TEXT',
        'bag_color': 'TEXT'
    }
    results = validator.validate_data_types(expected_types)
    for col, is_valid in results.items():
        assert is_valid, f"Data type validation failed for column: {col}"

def test_validate_primary_key_constraints(validator):
    """Test uniqueness for student_id (de-facto primary key)."""
    assert validator.validate_primary_key_constraints(pk_column='student_id') is False # Should be False due to duplicates

def test_check_missing_values(validator):
    """Test checking for missing values."""
    missing_columns = ['final_test', 'attendance_rate']
    missing_counts = validator.check_missing_values(missing_columns)
    assert missing_counts['final_test'] > 0
    assert missing_counts['attendance_rate'] > 0

def test_check_data_inconsistencies_tuition(temp_db):
    """Test detection of tuition format inconsistency."""
    conn, db_file = temp_db
    cursor = conn.cursor()
    cursor.execute("INSERT INTO score (tuition) VALUES ('Yes'), ('Y'), ('No'), ('N')")
    conn.commit()
    validator = DataValidator(db_file)
    inconsistencies = validator.check_data_inconsistencies()
    assert inconsistencies['tuition_format'] is True

def test_check_data_inconsistencies_cca_case(temp_db):
    """Test detection of CCA case inconsistency."""
    conn, db_file = temp_db
    cursor = conn.cursor()
    cursor.execute("INSERT INTO score (CCA) VALUES ('Clubs'), ('CLUBS')")
    conn.commit()
    validator = DataValidator(db_file)
    inconsistencies = validator.check_data_inconsistencies()
    assert inconsistencies['cca_case'] is True

def test_check_data_inconsistencies_negative_age(temp_db):
    """Test detection of negative age values."""
    conn, db_file = temp_db
    cursor = conn.cursor()
    cursor.execute("INSERT INTO score (age) VALUES (15), (-5)")
    conn.commit()
    validator = DataValidator(db_file)
    inconsistencies = validator.check_data_inconsistencies()
    assert inconsistencies['negative_age'] is True

def test_check_data_inconsistencies_duplicates(temp_db):
    """Test detection of duplicate records."""
    conn, db_file = temp_db
    cursor = conn.cursor()
    cursor.execute("INSERT INTO score (student_id, age) VALUES ('A1', 10), ('A1', 10), ('B1', 12)")
    conn.commit()
    validator = DataValidator(db_file)
    inconsistencies = validator.check_data_inconsistencies()
    assert inconsistencies['duplicate_records'] == 1 # One duplicate of 'A1'

def test_run_all_validations(validator):
    """Test running all validations and getting a summary."""
    summary = validator.run_all_validations()
    assert 'record_count_valid' in summary
    assert 'features_present_valid' in summary
    assert 'data_types_valid' in summary
    assert 'student_id_unique' in summary
    assert 'missing_values' in summary
    assert 'data_inconsistencies' in summary
    
    # Based on db-structure.md, some will be False/non-zero
    assert summary['record_count_valid'] is True
    assert summary['features_present_valid'] is True
    assert summary['student_id_unique'] is False # student_id is not unique
    assert summary['missing_values']['final_test'] > 0
    assert summary['data_inconsistencies']['tuition_format'] is True
    assert summary['data_inconsistencies']['cca_case'] is True
    assert summary['data_inconsistencies']['negative_age'] is True
    assert summary['data_inconsistencies']['duplicate_records'] > 0


# To run these tests, navigate to the project root and use:
# pytest tests/data/test_validation.py
# Ensure score.db is present in data/raw/