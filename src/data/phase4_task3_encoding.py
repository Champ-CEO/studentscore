#!/usr/bin/env python3
"""
Phase 4 Task 3.3: Advanced Categorical Encoding Strategy

This module implements task 4.3.3 from TASKS.md:
- Implement One-Hot Encoding
- Implement Target Encoding (in a subsequent step)

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging
from sklearn.preprocessing import OneHotEncoder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Phase4Encoding:
    """
    Applies advanced categorical encoding strategies for Phase 4 feature engineering.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with the dataset.
        
        Args:
            df: DataFrame with features from previous tasks
        """
        self.df = df.copy()
        self.encoded_feature_definitions = {}
        self.audit_log = []
        self.encoder = None # For OneHotEncoder instance

    def one_hot_encode(self, columns_to_encode: List[str]) -> None:
        """
        Apply One-Hot Encoding to specified columns.

        Args:
            columns_to_encode: List of column names to be one-hot encoded.
        """
        logger.info(f"Starting One-Hot Encoding for columns: {columns_to_encode}")
        
        missing_cols = [col for col in columns_to_encode if col not in self.df.columns]
        if missing_cols:
            logger.error(f"Missing columns for One-Hot Encoding: {missing_cols}. Aborting OHE.")
            raise ValueError(f"Missing columns for One-Hot Encoding: {missing_cols}")

        # Separate data for encoding
        data_to_encode = self.df[columns_to_encode].copy()
        
        # Handle NaN values before encoding - OHE can handle NaNs if specified, 
        # or we can fill them (e.g., with a placeholder like 'Missing')
        # For simplicity, let's fill with 'Missing' to create an explicit category for NaNs
        for col in columns_to_encode:
            if data_to_encode[col].isnull().any():
                logger.info(f"Filling NaNs in column '{col}' with 'Missing' before OHE.")
                data_to_encode[col] = data_to_encode[col].fillna('Missing')

        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore') # handle_unknown='ignore' is safer for test/new data
        
        try:
            encoded_data = self.encoder.fit_transform(data_to_encode)
            encoded_feature_names = self.encoder.get_feature_names_out(columns_to_encode)
            
            encoded_df = pd.DataFrame(encoded_data, columns=encoded_feature_names, index=self.df.index)
            
            # Drop original columns and concatenate new encoded columns
            self.df = self.df.drop(columns=columns_to_encode)
            self.df = pd.concat([self.df, encoded_df], axis=1)
            
            logger.info(f"One-Hot Encoding completed. Added {len(encoded_feature_names)} new features.")
            logger.info(f"New feature names: {list(encoded_feature_names)}")

            for i, original_col in enumerate(columns_to_encode):
                self.encoded_feature_definitions[original_col] = {
                    'original_column': original_col,
                    'encoding_type': 'One-Hot Encoding',
                    'new_columns': [name for name in encoded_feature_names if name.startswith(original_col + '_')],
                    'rationale': 'Convert categorical variable into a format that can be provided to ML algorithms.',
                    'source_columns': [original_col],
                    'created_by': 'Phase4Encoding.one_hot_encode'
                }
                self.audit_log.append({
                    'feature_group': original_col,
                    'action': 'one-hot encoded',
                    'new_features_count': len(self.encoded_feature_definitions[original_col]['new_columns']),
                    'new_feature_names': self.encoded_feature_definitions[original_col]['new_columns']
                })

        except Exception as e:
            logger.error(f"Error during One-Hot Encoding: {e}")
            raise

    def save_encoded_features(self, output_path: Path, definitions_path: Path) -> None:
        """
        Save the DataFrame with encoded features and their definitions.
        
        Args:
            output_path: Path to save the processed DataFrame (CSV).
            definitions_path: Path to save the feature definitions (JSON).
        """
        logger.info(f"Saving encoded DataFrame to {output_path}")
        self.df.to_csv(output_path, index=False)
        
        logger.info(f"Saving encoded feature definitions to {definitions_path}")
        with open(definitions_path, 'w') as f:
            json.dump(self.encoded_feature_definitions, f, indent=4)
            
        logger.info("Audit log for encoding process:")
        for entry in self.audit_log:
            logger.info(entry)

def main():
    """
    Main function to run the encoding process.
    """
    logger.info("Starting Phase 4 Encoding Process")
    
    # Define paths - these should be configured as needed
    # Assuming input data comes from a previous phase (e.g., interaction features)
    # This path needs to be updated to the correct input file from the previous step
    input_data_path = Path("data/featured/interaction_features.csv") 
    output_data_path = Path("data/featured/encoded_features.csv")
    feature_definitions_path = Path("data/featured/encoding_definitions.json")
    
    # Create output directories if they don't exist
    output_data_path.parent.mkdir(parents=True, exist_ok=True)
    feature_definitions_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not input_data_path.exists():
        logger.error(f"Input data file not found: {input_data_path}")
        print(f"Error: Input data file not found at {input_data_path}. Please ensure the output from the previous feature engineering step is available.")
        return

    logger.info(f"Loading data from {input_data_path}")
    try:
        df = pd.read_csv(input_data_path)
    except Exception as e:
        logger.error(f"Failed to load data from {input_data_path}: {e}")
        return

    # Initialize encoder class
    encoder_processor = Phase4Encoding(df)
    
    # Define columns for One-Hot Encoding based on TASKS.md (low cardinality)
    # Example: ['gender', 'transport_mode', 'learning_style']
    # These should be verified against the actual columns present in the dataset
    # and their cardinality.
    # For now, let's assume these are the correct columns and exist.
    # A more robust approach would be to dynamically identify low cardinality categorical columns.
    one_hot_encode_cols = ['gender', 'transport_mode', 'learning_style'] # Example list
    
    # Filter out columns that might not exist or are not suitable
    actual_ohe_cols = [col for col in one_hot_encode_cols if col in df.columns and df[col].nunique() < 10] # Arbitrary threshold for low cardinality
    if not actual_ohe_cols:
        logger.warning("No suitable columns found for One-Hot Encoding from the predefined list.")
    else:
        logger.info(f"Proceeding with One-Hot Encoding for: {actual_ohe_cols}")
        encoder_processor.one_hot_encode(columns_to_encode=actual_ohe_cols)
    
    # Placeholder for Target Encoding (Task 4.3.3.2) - to be implemented next
    # target_encode_cols = ['extracurricular_activities'] # Example
    # encoder_processor.target_encode(columns_to_encode=target_encode_cols, target_column='final_test')
    
    # Save processed data and definitions
    encoder_processor.save_encoded_features(output_data_path, feature_definitions_path)
    
    logger.info("Phase 4 Encoding Process Completed.")
    logger.info(f"Encoded data saved to: {output_data_path}")
    logger.info(f"Feature definitions saved to: {feature_definitions_path}")

if __name__ == "__main__":
    main()