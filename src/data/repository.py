import sqlite3
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager

class ScoreRepository:
    """Repository pattern implementation for score database access."""
    
    def __init__(self, db_path: str):
        """Initialize repository with database path.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
    
    @contextmanager
    def _get_connection(self) -> sqlite3.Connection:
        """Create and manage database connection context.
        
        Yields:
            sqlite3.Connection: Database connection object
        """
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def validate_schema(self) -> bool:
        """Validate that the database schema matches expected structure.
        
        Returns:
            bool: True if schema is valid, False otherwise
        """
        expected_columns = {
            'index', 'number_of_siblings', 'direct_admission', 'CCA',
            'learning_style', 'student_id', 'gender', 'tuition',
            'final_test', 'n_male', 'n_female', 'age', 'hours_per_week',
            'attendance_rate', 'sleep_time', 'wake_time',
            'mode_of_transport', 'bag_color'
        }
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM score LIMIT 1")
            columns = {description[0] for description in cursor.description}
            return columns == expected_columns
    
    def get_student_by_id(self, student_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve student record by student ID.
        
        Args:
            student_id: Student's unique identifier
            
        Returns:
            Optional[Dict[str, Any]]: Student record as dictionary or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM score WHERE student_id = ?",
                (student_id,)
            )
            row = cursor.fetchone()
            if row:
                return dict(zip([col[0] for col in cursor.description], row))
            return None
    
    def get_students_by_criteria(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve student records matching given criteria.
        
        Args:
            criteria: Dictionary of column names and values to filter by
            
        Returns:
            List[Dict[str, Any]]: List of matching student records
        """
        where_clause = " AND ".join(f"{k} = ?" for k in criteria.keys())
        query = f"SELECT * FROM score WHERE {where_clause}"
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(criteria.values()))
            columns = [col[0] for col in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_student_count(self) -> int:
        """Get total number of student records.
        
        Returns:
            int: Total number of records
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM score")
            return cursor.fetchone()[0]
    
    def get_average_final_test(self, group_by: Optional[str] = None) -> Dict[str, float]:
        """Calculate average final test scores, optionally grouped by a column.
        
        Args:
            group_by: Optional column name to group results by
            
        Returns:
            Dict[str, float]: Average scores by group or overall average
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if group_by:
                cursor.execute(
                    f"SELECT {group_by}, AVG(final_test) FROM score "
                    f"WHERE final_test IS NOT NULL GROUP BY {group_by}"
                )
                return dict(cursor.fetchall())
            else:
                cursor.execute(
                    "SELECT AVG(final_test) FROM score WHERE final_test IS NOT NULL"
                )
                return {"overall": cursor.fetchone()[0]}