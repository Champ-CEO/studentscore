import sqlite3
import os
import sys
from typing import Dict, Any, List, Set

# Add project root to sys.path to allow for relative imports when run directly
# This needs to be done before attempting to import from 'src'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.repository import ScoreRepository

class DataValidator:
    """A class to perform data integrity and structure validation on the score database."""

    def __init__(self, db_path: str):
        """Initialize the DataValidator with the database path."""
        self.repository = ScoreRepository(db_path)
        self.db_path = db_path

    def validate_record_count(self, expected_count: int = 15900) -> bool:
        """Verify the total number of records in the 'score' table."""
        actual_count = self.repository.get_student_count()
        return actual_count == expected_count

    def validate_features_present(self, expected_features: Set[str]) -> bool:
        """Verify all expected features (columns) are present in the 'score' table."""
        with self.repository._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(score)")
            columns = {row[1] for row in cursor.fetchall()}
            return expected_features.issubset(columns)

    def validate_data_types(self, expected_types: Dict[str, str]) -> Dict[str, bool]:
        """Validate data types for a sample of records."""
        # This is a simplified type check. For full validation, one might need to fetch all data
        # and check each value, or use a more robust data profiling library.
        validation_results = {}
        with self.repository._get_connection() as conn:
            conn.row_factory = sqlite3.Row # Allows access by column name
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM score LIMIT 100") # Sample 100 rows
            sample_rows = cursor.fetchall()

            for col_name, expected_type_str in expected_types.items():
                is_valid = True
                for row in sample_rows:
                    value = row[col_name]
                    # Handle NULLs, which can be of any type conceptually
                    if value is None:
                        continue

                    # Basic type checking based on expected string representation
                    if expected_type_str == 'INTEGER':
                        if not isinstance(value, int):
                            is_valid = False
                            break
                    elif expected_type_str == 'REAL':
                        if not isinstance(value, (int, float)):
                            is_valid = False
                            break
                    elif expected_type_str == 'TEXT':
                        if not isinstance(value, str):
                            is_valid = False
                            break
                    # Add more type checks as needed (e.g., for BLOB, NUMERIC)
                validation_results[col_name] = is_valid
        return validation_results

    def validate_primary_key_constraints(self, pk_column: str = 'student_id') -> bool:
        """Verify uniqueness for a de-facto primary key column like 'student_id'.
        
        Note: The db-structure.md states no primary keys are defined. This checks for uniqueness.
        """
        with self.repository._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT({pk_column}), COUNT(DISTINCT {pk_column}) FROM score")
            total_count, distinct_count = cursor.fetchone()
            return total_count == distinct_count

    def check_missing_values(self, columns_to_check: List[str]) -> Dict[str, int]:
        """Check for missing values (NULLs) in specified columns."""
        missing_counts = {}
        with self.repository._get_connection() as conn:
            cursor = conn.cursor()
            for col in columns_to_check:
                cursor.execute(f"SELECT COUNT(*) FROM score WHERE {col} IS NULL")
                missing_counts[col] = cursor.fetchone()[0]
        return missing_counts

    def check_data_inconsistencies(self) -> Dict[str, Any]:
        """Check for specific data inconsistencies as per db-structure.md."""
        inconsistencies = {
            'tuition_format': False,
            'cca_case': False,
            'negative_age': False,
            'duplicate_records': 0
        }
        with self.repository._get_connection() as conn:
            cursor = conn.cursor()

            # Tuition field: Mixed formats ('Yes'/'No' vs 'Y'/'N')
            cursor.execute("SELECT DISTINCT tuition FROM score")
            tuition_values = {row[0] for row in cursor.fetchall() if row[0] is not None}
            if not tuition_values.issubset({'Yes', 'No', 'Y', 'N'}): # Check for unexpected values
                inconsistencies['tuition_format'] = True
            if ('Y' in tuition_values or 'N' in tuition_values) and \
               ('Yes' in tuition_values or 'No' in tuition_values):
                inconsistencies['tuition_format'] = True

            # CCA field: Case inconsistency ('Clubs' vs 'CLUBS')
            cursor.execute("SELECT DISTINCT CCA FROM score")
            cca_values = {row[0] for row in cursor.fetchall() if row[0] is not None}
            # Check if there are values that are the same ignoring case but different in case
            lower_cca_values = {val.lower() for val in cca_values}
            if len(lower_cca_values) < len(cca_values):
                inconsistencies['cca_case'] = True

            # Age field: Contains negative values
            cursor.execute("SELECT MIN(age) FROM score")
            min_age = cursor.fetchone()[0]
            if min_age is not None and min_age < 0:
                inconsistencies['negative_age'] = True

            # Duplicate records (excluding 'index' column)
            # This is a more complex check, assuming 'index' is not part of the uniqueness criteria
            # A simpler approach for checking duplicates based on all other columns:
            # Note: This query might be slow on very large datasets.
            columns_for_duplicate_check = [
                'number_of_siblings', 'direct_admission', 'CCA', 'learning_style',
                'student_id', 'gender', 'tuition', 'final_test', 'n_male', 'n_female',
                'age', 'hours_per_week', 'attendance_rate', 'sleep_time',
                'wake_time', 'mode_of_transport', 'bag_color'
            ]
            cols_str = ", ".join(columns_for_duplicate_check)
            cursor.execute(f"""
                SELECT {cols_str}, COUNT(*) as count
                FROM score
                GROUP BY {cols_str}
                HAVING count > 1
            """)
            duplicate_rows = cursor.fetchall()
            inconsistencies['duplicate_records'] = sum(row[-1] - 1 for row in duplicate_rows)

        return inconsistencies

    def run_all_validations(self) -> Dict[str, Any]:
        """Run all defined data validation checks and return a summary."""
        results = {}

        # 1. Record Count
        results['record_count_valid'] = self.validate_record_count(expected_count=15900)

        # 2. Feature Presence
        expected_features = {
            'index', 'number_of_siblings', 'direct_admission', 'CCA',
            'learning_style', 'student_id', 'gender', 'tuition',
            'final_test', 'n_male', 'n_female', 'age', 'hours_per_week',
            'attendance_rate', 'sleep_time', 'wake_time',
            'mode_of_transport', 'bag_color'
        }
        results['features_present_valid'] = self.validate_features_present(expected_features)

        # 3. Data Types (simplified)
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
        results['data_types_valid'] = self.validate_data_types(expected_types)

        # 4. Primary Key Constraints (uniqueness of student_id)
        results['student_id_unique'] = self.validate_primary_key_constraints(pk_column='student_id')

        # 5. Missing Values
        missing_check_columns = ['final_test', 'attendance_rate']
        results['missing_values'] = self.check_missing_values(missing_check_columns)

        # 6. Data Inconsistencies
        results['data_inconsistencies'] = self.check_data_inconsistencies()

        return results

# Example Usage (for testing/demonstration)
if __name__ == "__main__":
    # Assuming score.db is in data/raw relative to project root
    current_dir = os.path.dirname(__file__)
    db_path = os.path.join(current_dir, '..', '..', 'data', 'raw', 'score.db')

    validator = DataValidator(db_path)
    validation_summary = validator.run_all_validations()

    print("--- Data Validation Summary ---")
    for key, value in validation_summary.items():
        print(f"{key}: {value}")

    # Detailed breakdown of inconsistencies if any
    if validation_summary.get('data_inconsistencies'):
        inconsistencies = validation_summary['data_inconsistencies']
        print("\n--- Detailed Inconsistencies ---")
        if inconsistencies['tuition_format']:
            print("- Tuition format inconsistency detected (e.g., 'Y'/'N' mixed with 'Yes'/'No').")
        if inconsistencies['cca_case']:
            print("- CCA case inconsistency detected (e.g., 'Clubs' vs 'CLUBS').")
        if inconsistencies['negative_age']:
            print("- Negative age values detected.")
        if inconsistencies['duplicate_records'] > 0:
            print(f"- {inconsistencies['duplicate_records']} duplicate records found (excluding index).")

    if all(validation_summary.get(k, True) for k in ['record_count_valid', 'features_present_valid', 'student_id_unique']) and \
       all(validation_summary['data_types_valid'].values()) and \
       not any(validation_summary['data_inconsistencies'].values()):
        print("\nOverall: Data structure and integrity checks passed (ignoring missing values for now).")
    else:
        print("\nOverall: Data structure and integrity checks FAILED.")