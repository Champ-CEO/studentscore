import sqlite3
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataQualityFixer:
    """
    Handles data quality issues for the student score dataset.
    
    This class implements the requirements for task 2.1.2:
    - Analyze student_id for uniqueness and format consistency
    - Age validation and correction logic
    - Categorical value standardization
    - Data type enforcement
    - Feature creation from cleaned data
    """
    
    def __init__(self, db_path: str = None, data: pd.DataFrame = None):
        """
        Initialize the DataQualityFixer.
        
        Args:
            db_path: Path to the SQLite database file
            data: DataFrame to work with (alternative to db_path)
        """
        self.db_path = db_path
        self.data = data
        self.cleaned_data = None
        self.quality_report = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from SQLite database or use provided DataFrame.
        
        Returns:
            DataFrame containing the raw data
        """
        if self.data is not None:
            logger.info(f"Using provided DataFrame with {len(self.data)} records")
            return self.data
        
        if self.db_path is None:
            raise ValueError("Either db_path or data must be provided")
        
        try:
            conn = sqlite3.connect(self.db_path)
            self.data = pd.read_sql_query("SELECT * FROM score", conn)
            conn.close()
            logger.info(f"Loaded {len(self.data)} records from database")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def analyze_student_id(self) -> Dict[str, Any]:
        """
        Comprehensive student_id analysis and anomaly detection.
        
        Returns:
            Dictionary with analysis results
        """
        if self.data is None:
            self.load_data()
        
        student_ids = self.data['student_id']
        
        analysis = {
            'total_records': len(student_ids),
            'unique_ids': student_ids.nunique(),
            'duplicate_count': len(student_ids) - student_ids.nunique(),
            'null_count': student_ids.isnull().sum(),
            'id_lengths': student_ids.str.len().value_counts().to_dict(),
            'id_patterns': {},
            'character_analysis': {},
            'potential_features': []
        }
        
        # Analyze ID patterns
        non_null_ids = student_ids.dropna()
        
        # Length analysis
        most_common_length = student_ids.str.len().mode()[0] if len(student_ids.str.len().mode()) > 0 else None
        analysis['most_common_length'] = most_common_length
        
        # Character pattern analysis
        if len(non_null_ids) > 0:
            # Check for consistent patterns
            patterns = []
            for id_val in non_null_ids.head(100):  # Sample first 100 for pattern analysis
                pattern = re.sub(r'[A-Z]', 'L', str(id_val))  # Letters to L
                pattern = re.sub(r'[0-9]', 'N', pattern)      # Numbers to N
                patterns.append(pattern)
            
            pattern_counts = Counter(patterns)
            analysis['id_patterns'] = dict(pattern_counts.most_common(10))
            
            # Character type analysis
            has_letters = non_null_ids.str.contains(r'[A-Za-z]').sum()
            has_numbers = non_null_ids.str.contains(r'[0-9]').sum()
            has_special = non_null_ids.str.contains(r'[^A-Za-z0-9]').sum()
            
            analysis['character_analysis'] = {
                'has_letters': has_letters,
                'has_numbers': has_numbers,
                'has_special_chars': has_special,
                'all_uppercase': non_null_ids.str.isupper().sum(),
                'all_lowercase': non_null_ids.str.islower().sum()
            }
        
        # Identify potential features that could be extracted
        if most_common_length and most_common_length >= 4:
            analysis['potential_features'].extend([
                'id_length',
                'id_first_char',
                'id_last_char',
                'id_numeric_count',
                'id_alpha_count'
            ])
        
        # Check for duplicates
        if analysis['duplicate_count'] > 0:
            duplicates = student_ids[student_ids.duplicated(keep=False)]
            analysis['duplicate_ids'] = duplicates.value_counts().to_dict()
        
        logger.info(f"Student ID analysis completed: {analysis['unique_ids']}/{analysis['total_records']} unique IDs")
        self.quality_report['student_id_analysis'] = analysis
        
        return analysis
    
    def extract_id_features(self) -> pd.DataFrame:
        """
        Extract meaningful features from student_id structure.
        
        Returns:
            DataFrame with new ID-derived features
        """
        if self.data is None:
            self.load_data()
        
        id_features = pd.DataFrame(index=self.data.index)
        student_ids = self.data['student_id'].fillna('')
        
        # Basic features
        id_features['id_length'] = student_ids.str.len()
        id_features['id_numeric_count'] = student_ids.str.count(r'[0-9]')
        id_features['id_alpha_count'] = student_ids.str.count(r'[A-Za-z]')
        id_features['id_special_count'] = student_ids.str.count(r'[^A-Za-z0-9]')
        
        # Position-based features (if IDs have consistent length)
        most_common_length = student_ids.str.len().mode()[0] if len(student_ids.str.len().mode()) > 0 else 0
        
        if most_common_length >= 2:
            id_features['id_first_char'] = student_ids.str[0]
            id_features['id_last_char'] = student_ids.str[-1]
            
            # Check if first/last characters follow patterns
            id_features['id_first_is_letter'] = student_ids.str[0].str.match(r'[A-Za-z]').astype(int)
            id_features['id_last_is_number'] = student_ids.str[-1].str.match(r'[0-9]').astype(int)
        
        if most_common_length >= 4:
            # Extract potential year/code from different positions
            id_features['id_prefix_2'] = student_ids.str[:2]
            id_features['id_suffix_2'] = student_ids.str[-2:]
            
            # Check for numeric patterns that might indicate years
            numeric_parts = student_ids.str.extract(r'([0-9]{2,4})')
            if not numeric_parts.empty:
                id_features['id_numeric_part'] = numeric_parts[0]
        
        # Complexity features
        id_features['id_complexity'] = (
            id_features['id_alpha_count'] > 0
        ).astype(int) + (
            id_features['id_numeric_count'] > 0
        ).astype(int) + (
            id_features['id_special_count'] > 0
        ).astype(int)
        
        logger.info(f"Extracted {len(id_features.columns)} ID-derived features")
        return id_features
    
    def fix_age_issues(self) -> pd.Series:
        """
        Fix negative age values and create age-based features.
        
        Returns:
            Series with corrected age values
        """
        if self.data is None:
            self.load_data()
        
        age_series = self.data['age'].copy()
        original_negative_count = (age_series < 0).sum()
        
        if original_negative_count > 0:
            logger.warning(f"Found {original_negative_count} negative age values")
            
            # Strategy: Replace negative ages with absolute value if reasonable, otherwise use median
            median_age = age_series[age_series >= 0].median()
            
            # Fix negative ages
            negative_mask = age_series < 0
            for idx in age_series[negative_mask].index:
                negative_age = age_series.loc[idx]
                abs_age = abs(negative_age)
                
                # If absolute value is reasonable (between 10 and 20), use it
                if 10 <= abs_age <= 20:
                    age_series.loc[idx] = abs_age
                    logger.info(f"Fixed negative age {negative_age} to {abs_age} at index {idx}")
                else:
                    # Use median for unreasonable values
                    age_series.loc[idx] = median_age
                    logger.info(f"Replaced unreasonable age {negative_age} with median {median_age} at index {idx}")
        
        # Validate age range (should be between 10 and 25 for students)
        unreasonable_mask = (age_series < 10) | (age_series > 25)
        unreasonable_count = unreasonable_mask.sum()
        
        if unreasonable_count > 0:
            logger.warning(f"Found {unreasonable_count} unreasonable age values")
            median_age = age_series[(age_series >= 10) & (age_series <= 25)].median()
            age_series[unreasonable_mask] = median_age
        
        self.quality_report['age_fixes'] = {
            'original_negative_count': original_negative_count,
            'unreasonable_count': unreasonable_count,
            'final_age_range': (age_series.min(), age_series.max()),
            'final_age_mean': age_series.mean()
        }
        
        logger.info(f"Age correction completed. Range: {age_series.min():.1f} - {age_series.max():.1f}")
        return age_series
    
    def create_age_features(self, age_series: pd.Series) -> pd.DataFrame:
        """
        Create age-based features from corrected age values.
        
        Args:
            age_series: Corrected age values
            
        Returns:
            DataFrame with age-based features
        """
        age_features = pd.DataFrame(index=age_series.index)
        
        # Age groups
        age_features['age_group'] = pd.cut(
            age_series, 
            bins=[0, 14, 15, 16, 17, 100], 
            labels=['under_15', '15', '16', '17', 'over_17'],
            include_lowest=True
        )
        
        # Age relative to median
        median_age = age_series.median()
        age_features['age_above_median'] = (age_series > median_age).astype(int)
        age_features['age_deviation_from_median'] = age_series - median_age
        
        # Age categories
        age_features['is_youngest'] = (age_series == age_series.min()).astype(int)
        age_features['is_oldest'] = (age_series == age_series.max()).astype(int)
        
        logger.info(f"Created {len(age_features.columns)} age-based features")
        return age_features
    
    def standardize_categorical_values(self) -> pd.DataFrame:
        """
        Standardize categorical values across the dataset.
        
        Returns:
            DataFrame with standardized categorical values
        """
        if self.data is None:
            self.load_data()
        
        standardized_data = self.data.copy()
        
        # Fix tuition field (Y/N → Yes/No)
        tuition_mapping = {'Y': 'Yes', 'N': 'No', 'yes': 'Yes', 'no': 'No'}
        original_tuition_values = standardized_data['tuition'].value_counts().to_dict()
        standardized_data['tuition'] = standardized_data['tuition'].map(tuition_mapping).fillna(standardized_data['tuition'])
        
        # Fix CCA case inconsistency (CLUBS → Clubs)
        cca_mapping = {
            'CLUBS': 'Clubs',
            'SPORTS': 'Sports', 
            'ARTS': 'Arts',
            'NONE': 'None'
        }
        original_cca_values = standardized_data['CCA'].value_counts().to_dict()
        standardized_data['CCA'] = standardized_data['CCA'].map(cca_mapping).fillna(standardized_data['CCA'])
        
        # Standardize direct_admission
        admission_mapping = {'yes': 'Yes', 'no': 'No', 'YES': 'Yes', 'NO': 'No'}
        standardized_data['direct_admission'] = standardized_data['direct_admission'].map(admission_mapping).fillna(standardized_data['direct_admission'])
        
        # Standardize gender
        gender_mapping = {'male': 'Male', 'female': 'Female', 'MALE': 'Male', 'FEMALE': 'Female'}
        standardized_data['gender'] = standardized_data['gender'].map(gender_mapping).fillna(standardized_data['gender'])
        
        # Standardize learning_style
        style_mapping = {'visual': 'Visual', 'auditory': 'Auditory', 'VISUAL': 'Visual', 'AUDITORY': 'Auditory'}
        standardized_data['learning_style'] = standardized_data['learning_style'].map(style_mapping).fillna(standardized_data['learning_style'])
        
        self.quality_report['categorical_standardization'] = {
            'tuition_before': original_tuition_values,
            'tuition_after': standardized_data['tuition'].value_counts().to_dict(),
            'cca_before': original_cca_values,
            'cca_after': standardized_data['CCA'].value_counts().to_dict()
        }
        
        logger.info("Categorical value standardization completed")
        return standardized_data
    
    def enforce_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce consistent data types across the dataset.
        
        Args:
            data: DataFrame to enforce types on
            
        Returns:
            DataFrame with enforced data types
        """
        typed_data = data.copy()
        
        # Define expected data types
        type_mapping = {
            'index': 'int64',
            'number_of_siblings': 'int64',
            'direct_admission': 'category',
            'CCA': 'category',
            'learning_style': 'category',
            'student_id': 'string',
            'gender': 'category',
            'tuition': 'category',
            'final_test': 'float64',
            'n_male': 'float64',
            'n_female': 'float64',
            'age': 'float64',
            'hours_per_week': 'float64',
            'attendance_rate': 'float64',
            'sleep_time': 'string',
            'wake_time': 'string',
            'mode_of_transport': 'category',
            'bag_color': 'category'
        }
        
        # Apply type conversions
        for column, dtype in type_mapping.items():
            if column in typed_data.columns:
                try:
                    if dtype == 'category':
                        typed_data[column] = typed_data[column].astype('category')
                    elif dtype == 'string':
                        typed_data[column] = typed_data[column].astype('string')
                    else:
                        typed_data[column] = pd.to_numeric(typed_data[column], errors='coerce')
                        if dtype == 'int64':
                            # Handle NaN values before converting to int
                            typed_data[column] = typed_data[column].fillna(0).astype('int64')
                        else:
                            typed_data[column] = typed_data[column].astype(dtype)
                except Exception as e:
                    logger.warning(f"Could not convert {column} to {dtype}: {e}")
        
        logger.info("Data type enforcement completed")
        return typed_data
    
    def fix_all_quality_issues(self) -> pd.DataFrame:
        """
        Complete data quality fixing pipeline.
        
        Returns:
            DataFrame with all quality issues fixed
        """
        if self.data is None:
            self.load_data()
        
        logger.info("Starting comprehensive data quality fixing...")
        
        # 1. Analyze student_id
        id_analysis = self.analyze_student_id()
        
        # 2. Standardize categorical values
        standardized_data = self.standardize_categorical_values()
        
        # 3. Fix age issues
        corrected_age = self.fix_age_issues()
        standardized_data['age'] = corrected_age
        
        # 4. Create age-based features
        age_features = self.create_age_features(corrected_age)
        
        # 5. Extract ID features
        id_features = self.extract_id_features()
        
        # 6. Combine all data
        cleaned_data = pd.concat([standardized_data, age_features, id_features], axis=1)
        
        # 7. Enforce data types
        cleaned_data = self.enforce_data_types(cleaned_data)
        
        self.cleaned_data = cleaned_data
        
        logger.info(f"Data quality fixing completed. Final dataset shape: {cleaned_data.shape}")
        return cleaned_data
    
    def get_quality_report(self) -> Dict[str, Any]:
        """
        Get comprehensive quality report.
        
        Returns:
            Dictionary with quality analysis and fixes applied
        """
        return self.quality_report
    
    def save_cleaned_data(self, output_path: str) -> None:
        """
        Save cleaned data to CSV file.
        
        Args:
            output_path: Path to save the cleaned data
        """
        if self.cleaned_data is None:
            raise ValueError("No cleaned data available. Run fix_all_quality_issues() first.")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.cleaned_data.to_csv(output_file, index=False)
        logger.info(f"Cleaned data saved to {output_file}")


def main():
    """
    Example usage of the DataQualityFixer.
    """
    # Initialize fixer
    db_path = "data/raw/score.db"
    fixer = DataQualityFixer(db_path)
    
    # Load and analyze data
    fixer.load_data()
    
    # Fix all quality issues
    cleaned_data = fixer.fix_all_quality_issues()
    
    # Get quality report
    report = fixer.get_quality_report()
    print("Data Quality Report:")
    for section, details in report.items():
        print(f"\n{section}:")
        if isinstance(details, dict):
            for key, value in details.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {details}")
    
    # Save cleaned data
    fixer.save_cleaned_data("data/processed/cleaned_data.csv")
    
    print(f"\nCleaned data shape: {cleaned_data.shape}")
    print(f"Original columns: {len(fixer.data.columns)}")
    print(f"Final columns: {len(cleaned_data.columns)}")


if __name__ == "__main__":
    main()