#!/usr/bin/env python3
"""
Fix Data Leakage in Phase 5 Modeling

This script identifies and removes features that cause data leakage by using the target variable
(final_test) in their calculation, which leads to unrealistically perfect model performance.

Data Leakage Issues Found:
1. Features derived from final_test (target variable) are included in training
2. This causes R² ≈ 1.0 and MAE ≈ 0, which is unrealistic
3. Models can't generalize because they're essentially predicting from the answer

Author: AI Assistant
Date: 2025-01-08
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLeakageDetector:
    """Detect and fix data leakage in the modeling dataset."""
    
    def __init__(self, 
                 data_path: str = 'data/processed/final_processed.csv',
                 feature_docs_path: str = 'data/featured',
                 output_path: str = 'data/modeling_outputs',
                 target_column: str = 'final_test'):
        self.data_path = Path(data_path)
        self.feature_docs_path = Path(feature_docs_path)
        self.output_path = Path(output_path)
        self.target_column = target_column
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.leaky_features = []
        self.clean_features = []
        
    def identify_leaky_features(self) -> List[str]:
        """
        Identify features that use the target variable in their calculation.
        
        Returns:
            List of feature names that cause data leakage
        """
        logger.info("Identifying features that cause data leakage...")
        
        leaky_features = []
        
        # Load interaction definitions to check for target usage
        interaction_def_path = self.feature_docs_path / 'interaction_definitions.json'
        if interaction_def_path.exists():
            with open(interaction_def_path, 'r') as f:
                interaction_defs = json.load(f)
            
            for feature_name, definition in interaction_defs.items():
                if isinstance(definition, dict):
                    # Check if target column is in the source columns
                    source_columns = definition.get('source_columns', [])
                    formula = definition.get('formula', '')
                    
                    if (self.target_column in source_columns or 
                        self.target_column in formula):
                        leaky_features.append(feature_name)
                        logger.warning(f"Leaky feature found: {feature_name} - uses {self.target_column}")
        
        # Also check for obvious patterns in feature names
        df = pd.read_csv(self.data_path)
        for col in df.columns:
            if (self.target_column in col and col != self.target_column):
                if col not in leaky_features:
                    leaky_features.append(col)
                    logger.warning(f"Leaky feature found by name pattern: {col}")
        
        # Additional patterns that might indicate leakage
        performance_related = ['performance_level']  # This is based on final_test quartiles
        for feature in performance_related:
            if feature in df.columns and feature not in leaky_features:
                leaky_features.append(feature)
                logger.warning(f"Leaky feature found (performance-based): {feature}")
        
        self.leaky_features = leaky_features
        logger.info(f"Total leaky features identified: {len(leaky_features)}")
        
        return leaky_features
    
    def create_clean_dataset(self) -> bool:
        """
        Create a clean dataset without data leakage.
        
        Returns:
            bool: Success status
        """
        try:
            logger.info("Creating clean dataset without data leakage...")
            
            # Load original data
            df = pd.read_csv(self.data_path)
            logger.info(f"Original dataset shape: {df.shape}")
            
            # Identify leaky features
            leaky_features = self.identify_leaky_features()
            
            # Remove leaky features
            clean_df = df.drop(columns=leaky_features, errors='ignore')
            logger.info(f"Clean dataset shape: {clean_df.shape}")
            logger.info(f"Removed {len(leaky_features)} leaky features")
            
            # Save clean dataset
            clean_data_path = self.output_path / 'clean_dataset_no_leakage.csv'
            clean_df.to_csv(clean_data_path, index=False)
            logger.info(f"Clean dataset saved to: {clean_data_path}")
            
            # Save leakage analysis report
            leakage_report = {
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'original_features': len(df.columns),
                'clean_features': len(clean_df.columns),
                'leaky_features_removed': len(leaky_features),
                'leaky_features_list': leaky_features,
                'clean_features_list': clean_df.columns.tolist(),
                'target_column': self.target_column,
                'data_leakage_explanation': {
                    'issue': 'Features derived from target variable included in training',
                    'impact': 'Unrealistically perfect model performance (R² ≈ 1.0)',
                    'solution': 'Remove all features that use target variable in calculation'
                }
            }
            
            report_path = self.output_path / 'data_leakage_analysis.json'
            with open(report_path, 'w') as f:
                json.dump(leakage_report, f, indent=2)
            logger.info(f"Leakage analysis report saved to: {report_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating clean dataset: {e}")
            return False
    
    def validate_clean_dataset(self) -> Dict[str, Any]:
        """
        Validate that the clean dataset doesn't have obvious leakage.
        
        Returns:
            Dict with validation results
        """
        try:
            logger.info("Validating clean dataset...")
            
            clean_data_path = self.output_path / 'clean_dataset_no_leakage.csv'
            if not clean_data_path.exists():
                return {'status': 'error', 'message': 'Clean dataset not found'}
            
            df = pd.read_csv(clean_data_path)
            
            # Check for remaining leaky features
            remaining_leaky = []
            for col in df.columns:
                if (self.target_column in col and col != self.target_column):
                    remaining_leaky.append(col)
            
            # Basic statistics
            if self.target_column in df.columns:
                target_stats = {
                    'count': len(df[self.target_column].dropna()),
                    'mean': df[self.target_column].mean(),
                    'std': df[self.target_column].std(),
                    'min': df[self.target_column].min(),
                    'max': df[self.target_column].max()
                }
            else:
                target_stats = {'error': 'Target column not found'}
            
            validation_results = {
                'validation_timestamp': pd.Timestamp.now().isoformat(),
                'dataset_shape': df.shape,
                'remaining_leaky_features': remaining_leaky,
                'leakage_free': len(remaining_leaky) == 0,
                'target_statistics': target_stats,
                'feature_count': len(df.columns) - 1,  # Excluding target
                'missing_values': df.isnull().sum().sum()
            }
            
            # Save validation report
            validation_path = self.output_path / 'clean_dataset_validation.json'
            with open(validation_path, 'w') as f:
                json.dump(validation_results, f, indent=2)
            logger.info(f"Validation report saved to: {validation_path}")
            
            if validation_results['leakage_free']:
                logger.info("✅ Clean dataset validation passed - no data leakage detected")
            else:
                logger.warning(f"⚠️ Potential leakage still detected: {remaining_leaky}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating clean dataset: {e}")
            return {'status': 'error', 'message': str(e)}

def main():
    """Main function to fix data leakage."""
    logger.info("Starting data leakage detection and fixing...")
    
    detector = DataLeakageDetector()
    
    # Create clean dataset
    if detector.create_clean_dataset():
        logger.info("✅ Clean dataset created successfully")
        
        # Validate clean dataset
        validation_results = detector.validate_clean_dataset()
        
        if validation_results.get('leakage_free', False):
            logger.info("🎉 Data leakage successfully fixed!")
            logger.info("Next steps:")
            logger.info("1. Update modeling pipeline to use clean_dataset_no_leakage.csv")
            logger.info("2. Re-run Phase 5 models with realistic performance expectations")
            logger.info("3. Consider implementing XGBoost and Neural Networks with proper data")
        else:
            logger.warning("⚠️ Additional validation may be needed")
    else:
        logger.error("❌ Failed to create clean dataset")

if __name__ == "__main__":
    main()