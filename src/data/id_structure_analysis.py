#!/usr/bin/env python3
"""
ID Structure Analysis Module

Implements Phase 3.1.2a tasks:
- ID Structure Analysis (3.1.2a)
- Feature Extraction from ID (3.1.2a.1)
- ID Retention Decision (3.1.2a.2)

This module analyzes student_id and other ID fields for patterns,
extracts useful features, and makes retention decisions.
"""

import pandas as pd
import numpy as np
import sqlite3
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from collections import Counter, defaultdict
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IDStructureAnalyzer:
    """
    Analyzes ID structure and extracts features from ID fields.
    
    Handles comprehensive analysis of student_id and other ID fields
    to identify patterns, extract embedded information, and make
    retention decisions based on predictive value.
    """
    
    def __init__(self, db_path: Optional[str] = None, data: Optional[pd.DataFrame] = None):
        """
        Initialize the ID Structure Analyzer.
        
        Args:
            db_path: Path to SQLite database file
            data: Pre-loaded DataFrame (alternative to db_path)
        """
        self.db_path = db_path
        self.data = data
        self.analysis_results = {}
        self.extracted_features = {}
        self.retention_decisions = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from database or return provided DataFrame.
        
        Returns:
            DataFrame with the loaded data
        """
        if self.data is not None:
            logger.info(f"Using provided DataFrame with {len(self.data)} records")
            return self.data.copy()
        
        if self.db_path is None:
            raise ValueError("Either db_path or data must be provided")
        
        try:
            conn = sqlite3.connect(self.db_path)
            query = "SELECT * FROM score"
            data = pd.read_sql_query(query, conn)
            conn.close()
            
            logger.info(f"Loaded {len(data)} records from database")
            self.data = data
            return data.copy()
            
        except Exception as e:
            logger.error(f"Error loading data from database: {e}")
            raise
    
    def analyze_student_id_structure(self) -> Dict[str, Any]:
        """
        Comprehensive analysis of student_id structure and patterns.
        
        Returns:
            Dictionary containing detailed analysis results
        """
        if self.data is None:
            self.data = self.load_data()
        
        student_ids = self.data['student_id'].dropna()
        
        analysis = {
            'basic_stats': {
                'total_records': len(self.data),
                'non_null_ids': len(student_ids),
                'null_count': self.data['student_id'].isnull().sum(),
                'unique_ids': student_ids.nunique(),
                'duplicate_count': len(student_ids) - student_ids.nunique()
            },
            'format_analysis': self._analyze_id_format(student_ids),
            'pattern_analysis': self._analyze_id_patterns(student_ids),
            'embedded_info': self._extract_embedded_information(student_ids),
            'quality_issues': self._identify_quality_issues(student_ids)
        }
        
        self.analysis_results['student_id'] = analysis
        logger.info(f"Student ID analysis completed: {analysis['basic_stats']['unique_ids']} unique IDs")
        
        return analysis
    
    def _analyze_id_format(self, ids: pd.Series) -> Dict[str, Any]:
        """
        Analyze the format characteristics of ID fields.
        
        Args:
            ids: Series of ID values to analyze
            
        Returns:
            Dictionary with format analysis results
        """
        format_analysis = {
            'length_distribution': ids.str.len().value_counts().to_dict(),
            'character_types': {
                'numeric_only': sum(ids.str.isnumeric()),
                'alpha_only': sum(ids.str.isalpha()),
                'alphanumeric': sum(ids.str.isalnum() & ~ids.str.isnumeric() & ~ids.str.isalpha()),
                'contains_special': sum(~ids.str.isalnum())
            },
            'case_analysis': {
                'uppercase': sum(ids.str.isupper()),
                'lowercase': sum(ids.str.islower()),
                'mixed_case': sum(~ids.str.isupper() & ~ids.str.islower() & ids.str.isalpha())
            }
        }
        
        # Most common length
        if format_analysis['length_distribution']:
            most_common_length = max(format_analysis['length_distribution'].items(), key=lambda x: x[1])[0]
            format_analysis['most_common_length'] = most_common_length
        
        return format_analysis
    
    def _analyze_id_patterns(self, ids: pd.Series) -> Dict[str, Any]:
        """
        Identify common patterns in ID structure.
        
        Args:
            ids: Series of ID values to analyze
            
        Returns:
            Dictionary with pattern analysis results
        """
        patterns = []
        pattern_counts = Counter()
        
        # Convert IDs to pattern representations
        for id_val in ids.head(1000):  # Sample for performance
            pattern = self._id_to_pattern(str(id_val))
            patterns.append(pattern)
            pattern_counts[pattern] += 1
        
        # Analyze segments if consistent pattern exists
        segment_analysis = {}
        most_common_pattern = pattern_counts.most_common(1)[0] if pattern_counts else None
        
        if most_common_pattern and most_common_pattern[1] > len(ids) * 0.8:  # 80% consistency
            segment_analysis = self._analyze_pattern_segments(ids, most_common_pattern[0])
        
        return {
            'pattern_distribution': dict(pattern_counts.most_common(10)),
            'most_common_pattern': most_common_pattern,
            'pattern_consistency': most_common_pattern[1] / len(ids) if most_common_pattern else 0,
            'segment_analysis': segment_analysis
        }
    
    def _id_to_pattern(self, id_str: str) -> str:
        """
        Convert an ID string to a pattern representation.
        
        Args:
            id_str: ID string to convert
            
        Returns:
            Pattern string (e.g., "NNNNLLL" for "2023ABC")
        """
        pattern = ""
        for char in id_str:
            if char.isdigit():
                pattern += "N"
            elif char.isalpha():
                pattern += "L"
            else:
                pattern += "S"  # Special character
        return pattern
    
    def _analyze_pattern_segments(self, ids: pd.Series, pattern: str) -> Dict[str, Any]:
        """
        Analyze segments of IDs that follow a consistent pattern.
        
        Args:
            ids: Series of ID values
            pattern: Pattern string to analyze
            
        Returns:
            Dictionary with segment analysis
        """
        segment_analysis = {}
        
        # Filter IDs that match the pattern
        matching_ids = ids[ids.apply(lambda x: self._id_to_pattern(str(x)) == pattern)]
        
        if len(matching_ids) == 0:
            return segment_analysis
        
        # Analyze each position in the pattern
        for i, char_type in enumerate(pattern):
            position_values = matching_ids.str[i].value_counts()
            
            segment_info = {
                'position': i,
                'type': char_type,
                'unique_values': len(position_values),
                'most_common': position_values.head(5).to_dict(),
                'entropy': self._calculate_entropy(position_values)
            }
            
            # Special analysis for numeric segments
            if char_type == 'N':
                numeric_values = matching_ids.str[i].astype(int)
                segment_info.update({
                    'min_value': numeric_values.min(),
                    'max_value': numeric_values.max(),
                    'range': numeric_values.max() - numeric_values.min(),
                    'sequential': self._check_sequential(numeric_values)
                })
            
            segment_analysis[f'position_{i}'] = segment_info
        
        return segment_analysis
    
    def _calculate_entropy(self, value_counts: pd.Series) -> float:
        """
        Calculate entropy of value distribution.
        
        Args:
            value_counts: Series with value counts
            
        Returns:
            Entropy value
        """
        probabilities = value_counts / value_counts.sum()
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        return entropy
    
    def _check_sequential(self, values: pd.Series) -> bool:
        """
        Check if numeric values are sequential.
        
        Args:
            values: Series of numeric values
            
        Returns:
            True if values are mostly sequential
        """
        sorted_values = sorted(values.unique())
        if len(sorted_values) < 2:
            return False
        
        differences = [sorted_values[i+1] - sorted_values[i] for i in range(len(sorted_values)-1)]
        most_common_diff = Counter(differences).most_common(1)[0][1]
        
        return most_common_diff / len(differences) > 0.8
    
    def _extract_embedded_information(self, ids: pd.Series) -> Dict[str, Any]:
        """
        Extract embedded information from ID patterns.
        
        Args:
            ids: Series of ID values
            
        Returns:
            Dictionary with extracted information
        """
        embedded_info = {
            'potential_year': self._extract_year_info(ids),
            'potential_sequence': self._extract_sequence_info(ids),
            'potential_category': self._extract_category_info(ids),
            'checksum_analysis': self._analyze_checksum(ids)
        }
        
        return embedded_info
    
    def _extract_year_info(self, ids: pd.Series) -> Dict[str, Any]:
        """
        Extract potential year information from IDs.
        
        Args:
            ids: Series of ID values
            
        Returns:
            Dictionary with year extraction results
        """
        year_patterns = []
        current_year = datetime.now().year
        
        for id_val in ids.head(100):  # Sample for analysis
            # Look for 4-digit years (2000-2030)
            year_matches = re.findall(r'(20[0-3][0-9])', str(id_val))
            if year_matches:
                year_patterns.extend([int(y) for y in year_matches])
            
            # Look for 2-digit years (00-30)
            two_digit_matches = re.findall(r'([0-3][0-9])', str(id_val))
            for match in two_digit_matches:
                year_val = int(match)
                if year_val <= 30:
                    year_patterns.append(2000 + year_val)
                elif year_val >= 70:
                    year_patterns.append(1900 + year_val)
        
        year_counter = Counter(year_patterns)
        
        return {
            'years_found': len(year_counter),
            'year_distribution': dict(year_counter.most_common(10)),
            'likely_year_field': len(year_patterns) > len(ids) * 0.5,
            'year_range': (min(year_patterns), max(year_patterns)) if year_patterns else None
        }
    
    def _extract_sequence_info(self, ids: pd.Series) -> Dict[str, Any]:
        """
        Extract potential sequence number information.
        
        Args:
            ids: Series of ID values
            
        Returns:
            Dictionary with sequence analysis
        """
        # Extract all numeric sequences of different lengths
        sequences = {
            'length_2': [],
            'length_3': [],
            'length_4': [],
            'length_5+': []
        }
        
        for id_val in ids.head(100):
            numeric_parts = re.findall(r'\d+', str(id_val))
            for part in numeric_parts:
                length = len(part)
                if length == 2:
                    sequences['length_2'].append(int(part))
                elif length == 3:
                    sequences['length_3'].append(int(part))
                elif length == 4:
                    sequences['length_4'].append(int(part))
                elif length >= 5:
                    sequences['length_5+'].append(int(part))
        
        sequence_analysis = {}
        for length, values in sequences.items():
            if values:
                sequence_analysis[length] = {
                    'count': len(values),
                    'range': (min(values), max(values)),
                    'unique_count': len(set(values)),
                    'sequential_likelihood': self._assess_sequential_likelihood(values)
                }
        
        return sequence_analysis
    
    def _extract_category_info(self, ids: pd.Series) -> Dict[str, Any]:
        """
        Extract potential category information from alphabetic parts.
        
        Args:
            ids: Series of ID values
            
        Returns:
            Dictionary with category analysis
        """
        alpha_parts = []
        
        for id_val in ids.head(100):
            alpha_matches = re.findall(r'[A-Za-z]+', str(id_val))
            alpha_parts.extend(alpha_matches)
        
        alpha_counter = Counter(alpha_parts)
        
        return {
            'alpha_parts_found': len(alpha_counter),
            'most_common_alpha': dict(alpha_counter.most_common(10)),
            'likely_category_field': len(set(alpha_parts)) < len(alpha_parts) * 0.5,
            'average_alpha_length': np.mean([len(part) for part in alpha_parts]) if alpha_parts else 0
        }
    
    def _analyze_checksum(self, ids: pd.Series) -> Dict[str, Any]:
        """
        Analyze potential checksum digits in IDs.
        
        Args:
            ids: Series of ID values
            
        Returns:
            Dictionary with checksum analysis
        """
        # Simple checksum analysis - look for patterns in last digits
        last_digits = []
        
        for id_val in ids.head(100):
            id_str = str(id_val)
            if id_str and id_str[-1].isdigit():
                last_digits.append(int(id_str[-1]))
        
        if not last_digits:
            return {'checksum_likely': False}
        
        digit_distribution = Counter(last_digits)
        
        # Even distribution suggests possible checksum
        expected_freq = len(last_digits) / 10
        variance = np.var(list(digit_distribution.values()))
        
        return {
            'checksum_likely': variance < expected_freq * 0.5,
            'last_digit_distribution': dict(digit_distribution),
            'distribution_variance': variance
        }
    
    def _assess_sequential_likelihood(self, values: List[int]) -> float:
        """
        Assess likelihood that values are sequential.
        
        Args:
            values: List of numeric values
            
        Returns:
            Likelihood score (0-1)
        """
        if len(values) < 2:
            return 0.0
        
        sorted_values = sorted(set(values))
        if len(sorted_values) < 2:
            return 0.0
        
        differences = [sorted_values[i+1] - sorted_values[i] for i in range(len(sorted_values)-1)]
        most_common_diff = Counter(differences).most_common(1)[0]
        
        return most_common_diff[1] / len(differences)
    
    def _identify_quality_issues(self, ids: pd.Series) -> Dict[str, Any]:
        """
        Identify quality issues in ID fields.
        
        Args:
            ids: Series of ID values
            
        Returns:
            Dictionary with quality issues
        """
        issues = {
            'malformed_ids': [],
            'inconsistent_length': False,
            'special_characters': [],
            'case_inconsistency': False,
            'leading_zeros': 0
        }
        
        # Check for length consistency
        lengths = ids.str.len()
        if lengths.nunique() > 1:
            issues['inconsistent_length'] = True
            issues['length_distribution'] = lengths.value_counts().to_dict()
        
        # Check for special characters
        for id_val in ids.head(100):
            if not str(id_val).isalnum():
                special_chars = re.findall(r'[^A-Za-z0-9]', str(id_val))
                issues['special_characters'].extend(special_chars)
        
        issues['special_characters'] = list(set(issues['special_characters']))
        
        # Check for case inconsistency
        alpha_ids = ids[ids.str.contains(r'[A-Za-z]', na=False)]
        if len(alpha_ids) > 0:
            upper_count = alpha_ids.str.isupper().sum()
            lower_count = alpha_ids.str.islower().sum()
            if upper_count > 0 and lower_count > 0:
                issues['case_inconsistency'] = True
        
        # Check for leading zeros
        issues['leading_zeros'] = ids.str.startswith('0').sum()
        
        return issues
    
    def extract_features_from_ids(self) -> pd.DataFrame:
        """
        Extract features from ID analysis results.
        
        Returns:
            DataFrame with extracted features
        """
        if self.data is None:
            self.data = self.load_data()
        
        if 'student_id' not in self.analysis_results:
            self.analyze_student_id_structure()
        
        features_df = pd.DataFrame(index=self.data.index)
        
        # Extract features based on analysis results
        student_ids = self.data['student_id'].fillna('')
        
        # Basic features
        features_df['id_length'] = student_ids.str.len()
        features_df['id_numeric_count'] = student_ids.str.count(r'\d')
        features_df['id_alpha_count'] = student_ids.str.count(r'[A-Za-z]')
        features_df['id_special_count'] = student_ids.str.count(r'[^A-Za-z0-9]')
        
        # Pattern-based features
        analysis = self.analysis_results['student_id']
        if analysis['pattern_analysis']['most_common_pattern']:
            pattern = analysis['pattern_analysis']['most_common_pattern'][0]
            features_df['id_follows_common_pattern'] = student_ids.apply(
                lambda x: self._id_to_pattern(str(x)) == pattern
            ).astype(int)
        
        # Year extraction if detected
        if analysis['embedded_info']['potential_year']['likely_year_field']:
            features_df['id_extracted_year'] = student_ids.apply(self._extract_year_from_id)
        
        # Sequence extraction
        features_df['id_max_numeric_sequence'] = student_ids.apply(self._extract_max_numeric_sequence)
        
        # Quality indicators
        features_df['id_has_special_chars'] = (features_df['id_special_count'] > 0).astype(int)
        features_df['id_is_malformed'] = student_ids.apply(self._is_malformed_id).astype(int)
        
        self.extracted_features['student_id'] = features_df
        logger.info(f"Extracted {len(features_df.columns)} features from student_id")
        
        return features_df
    
    def _extract_year_from_id(self, id_val: str) -> Optional[int]:
        """
        Extract year from ID value.
        
        Args:
            id_val: ID string
            
        Returns:
            Extracted year or None
        """
        # Look for 4-digit years first
        year_matches = re.findall(r'(20[0-3][0-9])', str(id_val))
        if year_matches:
            return int(year_matches[0])
        
        # Look for 2-digit years
        two_digit_matches = re.findall(r'([0-3][0-9])', str(id_val))
        for match in two_digit_matches:
            year_val = int(match)
            if year_val <= 30:
                return 2000 + year_val
        
        return None
    
    def _extract_max_numeric_sequence(self, id_val: str) -> int:
        """
        Extract the maximum numeric sequence from ID.
        
        Args:
            id_val: ID string
            
        Returns:
            Maximum numeric value found
        """
        numeric_parts = re.findall(r'\d+', str(id_val))
        if numeric_parts:
            return max(int(part) for part in numeric_parts)
        return 0
    
    def _is_malformed_id(self, id_val: str) -> bool:
        """
        Check if ID is malformed based on common patterns.
        
        Args:
            id_val: ID string
            
        Returns:
            True if ID appears malformed
        """
        id_str = str(id_val)
        
        # Check for common malformation patterns
        if len(id_str) == 0:
            return True
        
        # Check for excessive special characters
        special_count = len(re.findall(r'[^A-Za-z0-9]', id_str))
        if special_count > len(id_str) * 0.3:
            return True
        
        # Check for repeated characters (more than 4 in a row)
        if re.search(r'(.)\1{4,}', id_str):
            return True
        
        return False
    
    def make_retention_decisions(self, target_column: str = 'final_test') -> Dict[str, Dict[str, Any]]:
        """
        Make decisions about which ID fields and features to retain.
        
        Args:
            target_column: Target variable for predictive value assessment
            
        Returns:
            Dictionary with retention decisions
        """
        if self.data is None:
            self.data = self.load_data()
        
        decisions = {}
        
        # Analyze student_id retention
        if 'student_id' not in self.extracted_features:
            self.extract_features_from_ids()
        
        id_features = self.extracted_features['student_id']
        
        # Calculate correlations with target
        target_data = self.data[target_column].dropna()
        correlations = {}
        
        for col in id_features.columns:
            if id_features[col].dtype in ['int64', 'float64']:
                # Align indices
                aligned_feature = id_features[col].loc[target_data.index]
                correlation = aligned_feature.corr(target_data)
                correlations[col] = abs(correlation) if not pd.isna(correlation) else 0
        
        # Make retention decision based on predictive value
        high_correlation_threshold = 0.1
        useful_features = [col for col, corr in correlations.items() if corr > high_correlation_threshold]
        
        decisions['student_id'] = {
            'retain_original': False,  # Usually not useful for ML
            'retain_extracted_features': len(useful_features) > 0,
            'useful_features': useful_features,
            'feature_correlations': correlations,
            'reasoning': self._generate_retention_reasoning('student_id', useful_features, correlations)
        }
        
        self.retention_decisions = decisions
        logger.info(f"Retention decision for student_id: retain_extracted_features={decisions['student_id']['retain_extracted_features']}")
        
        return decisions
    
    def _generate_retention_reasoning(self, id_field: str, useful_features: List[str], correlations: Dict[str, float]) -> str:
        """
        Generate reasoning for retention decisions.
        
        Args:
            id_field: Name of ID field
            useful_features: List of useful extracted features
            correlations: Dictionary of feature correlations
            
        Returns:
            Reasoning string
        """
        if len(useful_features) == 0:
            return f"No extracted features from {id_field} show significant correlation with target variable. Original ID field provides no predictive value for ML models."
        
        max_corr = max(correlations.values()) if correlations else 0
        return f"Extracted {len(useful_features)} useful features from {id_field} with maximum correlation of {max_corr:.3f}. These features may provide predictive value."
    
    def save_analysis_results(self, output_dir: str) -> None:
        """
        Save analysis results to files.
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save analysis results
        analysis_file = output_path / 'id_structure_analysis.json'
        with open(analysis_file, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        # Save extracted features
        if self.extracted_features:
            for id_field, features_df in self.extracted_features.items():
                features_file = output_path / f'{id_field}_extracted_features.csv'
                features_df.to_csv(features_file, index=True)
        
        # Save retention decisions
        if self.retention_decisions:
            decisions_file = output_path / 'id_retention_decisions.json'
            with open(decisions_file, 'w') as f:
                json.dump(self.retention_decisions, f, indent=2, default=str)
        
        logger.info(f"Analysis results saved to {output_path}")
    
    def generate_summary_report(self) -> str:
        """
        Generate a summary report of the ID structure analysis.
        
        Returns:
            Summary report string
        """
        if not self.analysis_results:
            return "No analysis results available. Run analyze_student_id_structure() first."
        
        report = []
        report.append("=== ID Structure Analysis Summary ===")
        report.append("")
        
        for id_field, analysis in self.analysis_results.items():
            report.append(f"## {id_field.upper()} Analysis")
            report.append("")
            
            # Basic stats
            stats = analysis['basic_stats']
            report.append(f"- Total records: {stats['total_records']}")
            report.append(f"- Non-null IDs: {stats['non_null_ids']}")
            report.append(f"- Unique IDs: {stats['unique_ids']}")
            report.append(f"- Duplicates: {stats['duplicate_count']}")
            report.append("")
            
            # Format analysis
            format_info = analysis['format_analysis']
            if 'most_common_length' in format_info:
                report.append(f"- Most common length: {format_info['most_common_length']}")
            
            char_types = format_info['character_types']
            report.append(f"- Numeric only: {char_types['numeric_only']}")
            report.append(f"- Contains letters: {char_types['alpha_only'] + char_types['alphanumeric']}")
            report.append("")
            
            # Pattern analysis
            pattern_info = analysis['pattern_analysis']
            if pattern_info['most_common_pattern']:
                pattern, count = pattern_info['most_common_pattern']
                consistency = pattern_info['pattern_consistency']
                report.append(f"- Most common pattern: {pattern} ({consistency:.1%} consistency)")
            report.append("")
            
            # Embedded information
            embedded = analysis['embedded_info']
            if embedded['potential_year']['likely_year_field']:
                year_range = embedded['potential_year']['year_range']
                report.append(f"- Contains year information: {year_range}")
            
            if embedded['potential_sequence']:
                report.append("- Contains sequence numbers")
            report.append("")
        
        # Retention decisions
        if self.retention_decisions:
            report.append("## Retention Decisions")
            report.append("")
            
            for id_field, decision in self.retention_decisions.items():
                report.append(f"### {id_field}")
                report.append(f"- Retain original: {decision['retain_original']}")
                report.append(f"- Retain extracted features: {decision['retain_extracted_features']}")
                if decision['useful_features']:
                    report.append(f"- Useful features: {', '.join(decision['useful_features'])}")
                report.append(f"- Reasoning: {decision['reasoning']}")
                report.append("")
        
        return "\n".join(report)


def main():
    """
    Main function for testing the ID Structure Analyzer.
    """
    # Example usage
    db_path = "data/raw/score.db"
    
    analyzer = IDStructureAnalyzer(db_path=db_path)
    
    # Run analysis
    analysis_results = analyzer.analyze_student_id_structure()
    print("Analysis completed")
    
    # Extract features
    features = analyzer.extract_features_from_ids()
    print(f"Extracted {len(features.columns)} features")
    
    # Make retention decisions
    decisions = analyzer.make_retention_decisions()
    print("Retention decisions made")
    
    # Generate report
    report = analyzer.generate_summary_report()
    print(report)
    
    # Save results
    analyzer.save_analysis_results("data/processed")
    print("Results saved")


if __name__ == "__main__":
    main()