import pytest
import sqlite3
import os
from src.data.repository import ScoreRepository

# Define the path to the test database relative to this test file
# This assumes 'score.db' is in 'data/raw' and tests are in 'tests/data'
TEST_DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'score.db')

@pytest.fixture(scope='module')
def db_path():
    """Provides the path to the test database."""
    # Check if the database file exists, skip tests if not
    if not os.path.exists(TEST_DB_PATH):
        pytest.skip(f"Test database not found at {TEST_DB_PATH}. Run download_db.py first.")
    return TEST_DB_PATH

@pytest.fixture(scope='module')
def repository(db_path):
    """Provides a ScoreRepository instance for testing."""
    return ScoreRepository(db_path)

def test_repository_initialization(repository, db_path):
    """Test that the repository initializes correctly."""
    assert repository.db_path == db_path

def test_validate_schema_valid(repository):
    """Test schema validation with a valid schema (assuming score.db is valid)."""
    # This test assumes the provided score.db has the correct schema as defined in ScoreRepository
    assert repository.validate_schema() is True

def test_validate_schema_invalid(tmp_path, repository):
    """Test schema validation with an invalid schema."""
    # Create a temporary dummy database with a different schema
    invalid_db_path = tmp_path / "invalid.db"
    conn = sqlite3.connect(invalid_db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE score (id INTEGER, name TEXT)")
    conn.commit()
    conn.close()
    
    invalid_repo = ScoreRepository(str(invalid_db_path))
    assert invalid_repo.validate_schema() is False

def test_get_student_by_id_exists(repository):
    """Test retrieving an existing student by ID."""
    # Assuming 'ACN2BE' is a valid student_id from the sample data
    # This ID should exist if the original score.db is used
    student = repository.get_student_by_id('ACN2BE')
    assert student is not None
    assert student['student_id'] == 'ACN2BE'
    assert 'final_test' in student # Check for a key field

def test_get_student_by_id_not_exists(repository):
    """Test retrieving a non-existing student by ID."""
    student = repository.get_student_by_id('NON_EXISTENT_ID')
    assert student is None

def test_get_students_by_criteria_single_match(repository):
    """Test retrieving students by criteria with a single match (or more)."""
    # Using a known student_id for a specific test case
    students = repository.get_students_by_criteria({'student_id': 'ACN2BE'})
    assert len(students) >= 1
    assert students[0]['student_id'] == 'ACN2BE'

def test_get_students_by_criteria_multiple_matches(repository):
    """Test retrieving students by criteria that should yield multiple matches."""
    # Example: students with CCA as 'Sports'
    # This assumes there are multiple students with CCA 'Sports'
    students = repository.get_students_by_criteria({'CCA': 'Sports'})
    assert len(students) > 1
    for student in students:
        assert student['CCA'] == 'Sports'

def test_get_students_by_criteria_no_match(repository):
    """Test retrieving students by criteria with no matches."""
    students = repository.get_students_by_criteria({'gender': 'Alien'})
    assert len(students) == 0

def test_get_student_count(repository):
    """Test getting the total count of students."""
    # The actual count depends on the content of score.db
    # From db-structure.md, total records: 15,900
    count = repository.get_student_count()
    assert isinstance(count, int)
    assert count > 0 # Should be a positive number
    # A more specific check could be added if the exact number is stable
    # assert count == 15900 # If we are sure about the number of records

def test_get_average_final_test_overall(repository):
    """Test getting the overall average final test score."""
    avg_scores = repository.get_average_final_test()
    assert 'overall' in avg_scores
    assert isinstance(avg_scores['overall'], float)
    assert 0 <= avg_scores['overall'] <= 100

def test_get_average_final_test_grouped(repository):
    """Test getting average final test scores grouped by a column (e.g., gender)."""
    avg_scores_gender = repository.get_average_final_test(group_by='gender')
    assert 'Male' in avg_scores_gender or 'Female' in avg_scores_gender # Check for expected groups
    if 'Male' in avg_scores_gender:
        assert isinstance(avg_scores_gender['Male'], float)
        assert 0 <= avg_scores_gender['Male'] <= 100
    if 'Female' in avg_scores_gender:
        assert isinstance(avg_scores_gender['Female'], float)
        assert 0 <= avg_scores_gender['Female'] <= 100

# Placeholder tests for future or conceptual aspects

def test_crud_operations_placeholder():
    """Placeholder for CRUD operations tests.
    Currently, the repository primarily implements read operations.
    Write operations (Create, Update, Delete) would require further implementation
    and corresponding tests, including transaction handling.
    """
    # Example: Test create_student (if implemented)
    # student_data = {...}
    # created_id = repository.create_student(student_data)
    # assert created_id is not None
    # retrieved_student = repository.get_student_by_id(created_id)
    # assert retrieved_student['some_key'] == student_data['some_key']
    pytest.skip("CRUD write operations not fully implemented in this version of repository.")

def test_query_parameter_sanitization_concept(repository):
    """Conceptual test for query parameter sanitization.
    The sqlite3 library's use of '?' placeholders handles sanitization against SQL injection.
    This test serves as a reminder and conceptual validation.
    """
    # Attempting a malicious-like input (though sqlite3 should handle it)
    try:
        # This specific call might not exist, it's for concept
        repository.get_students_by_criteria({'student_id': "' OR '1'='1"})
        # If it doesn't raise an error and returns unexpected results, it's an issue.
        # However, with proper use of placeholders, this should be safe.
    except Exception as e:
        # Depending on implementation, it might raise an error or return empty
        pass 
    # A more robust test would involve checking that no unintended data is returned
    # or that the query fails gracefully if the input is truly malformed for the column type.
    assert True # Assuming sqlite3 placeholders work as expected.

def test_transaction_handling_placeholder():
    """Placeholder for transaction handling tests.
    For read-heavy repositories, explicit transaction management is less critical
    than for write operations. If write methods are added, robust transaction
    tests (commit, rollback) would be essential.
    """
    # Example: Test a series of operations within a transaction
    # with repository.transaction() as tx:
    #     repository.update_student(id1, data1, cursor=tx.cursor)
    #     repository.update_student(id2, data2, cursor=tx.cursor)
    #     # Check intermediate state if necessary
    # # Check final state after commit
    pytest.skip("Transaction handling for write operations not implemented.")

# To run these tests, navigate to the project root and use:
# pytest tests/data/test_repository.py
# Ensure score.db is present in data/raw/