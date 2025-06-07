#!/usr/bin/env python3
"""
Phase 4 Feature Engineering - Complete Execution Script

This script executes all Phase 4 tasks in the correct order as specified in TASKS.md:
1. Task 1: Load and Validate Data
2. Task 2.1: Derived Features (Study Efficiency Score, Academic Support Index)
3. Task 2.2: Interaction Features (Study × Attendance)
4. Task 2.3: Distribution-based Transformations
5. Task 3.1: Advanced Preprocessing (Scaling, Encoding)
6. Task 4.1: Feature Selection
7. Task 5.1: Data Quality Targets
8. Task 6.1: Documentation and Validation

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
from datetime import datetime
import sys
import traceback
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phase4_execution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import Phase 4 task modules
try:
    from phase4_task1_load_validate import Phase4DataLoader
    from phase4_task2_derived_features import Phase4DerivedFeatures
    from phase4_task2_interaction_features import Phase4InteractionFeatures
    from phase4_task2_transformations import Phase4Transformations
    from phase4_task3_advanced_preprocessing import Phase4AdvancedPreprocessing
    from phase4_task4_feature_selection import Phase4FeatureSelection
    from phase4_task5_data_quality import Phase4DataQuality
    from phase4_task6_documentation import Phase4Documentation
except ImportError as e:
    logger.error(f"Failed to import Phase 4 modules: {str(e)}")
    logger.error("Please ensure all Phase 4 task files are in the same directory")
    sys.exit(1)

class Phase4Executor:
    """
    Executes all Phase 4 Feature Engineering tasks in sequence.
    """
    
    def __init__(self, input_data_path: str = "data/processed/final_processed.csv",
                 output_dir: str = "data/featured"):
        """
        Initialize the Phase 4 executor.
        
        Args:
            input_data_path: Path to the input processed data
            output_dir: Directory for output files
        """
        self.input_data_path = Path(input_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.execution_log = {
            'start_time': datetime.now().isoformat(),
            'tasks_completed': [],
            'tasks_failed': [],
            'current_data_path': None,
            'final_status': 'pending'
        }
        
        self.current_df = None
        self.original_df = None
        
    def load_initial_data(self) -> bool:
        """
        Load the initial processed data.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Loading initial data from {self.input_data_path}")
            
            if not self.input_data_path.exists():
                raise FileNotFoundError(f"Input data file not found: {self.input_data_path}")
                
            self.original_df = pd.read_csv(self.input_data_path, index_col=0)
            self.current_df = self.original_df.copy()
            
            logger.info(f"Successfully loaded data with shape: {self.current_df.shape}")
            logger.info(f"Columns: {list(self.current_df.columns)}")
            
            self.execution_log['current_data_path'] = str(self.input_data_path)
            return True
            
        except Exception as e:
            logger.error(f"Failed to load initial data: {str(e)}")
            self.execution_log['tasks_failed'].append({
                'task': 'load_initial_data',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False
            
    def execute_task1_load_validate(self) -> bool:
        """
        Execute Task 1: Load and Validate Data.
        
        Returns:
            True if successful, False otherwise
        """
        task_name = "Task 1: Load and Validate Data"
        logger.info(f"Starting {task_name}")
        
        try:
            # Create data loader
            loader = Phase4DataLoader(str(self.input_data_path))
            
            # Load and validate data
            df, validation_results = loader.load_and_validate()
            
            # Update current dataframe
            self.current_df = df
            
            # Save validation report
            validation_path = self.output_dir / "phase4_task1_validation_report.json"
            with open(validation_path, 'w') as f:
                json.dump(validation_results, f, indent=2, default=str)
                
            logger.info(f"✅ {task_name} completed successfully")
            self.execution_log['tasks_completed'].append({
                'task': task_name,
                'timestamp': datetime.now().isoformat(),
                'output_shape': self.current_df.shape,
                'validation_passed': validation_results.get('overall_validation_passed', False)
            })
            
            return True
            
        except Exception as e:
            logger.error(f"❌ {task_name} failed: {str(e)}")
            logger.error(traceback.format_exc())
            self.execution_log['tasks_failed'].append({
                'task': task_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False
            
    def execute_task2_1_derived_features(self) -> bool:
        """
        Execute Task 2.1: Derived Features.
        
        Returns:
            True if successful, False otherwise
        """
        task_name = "Task 2.1: Derived Features"
        logger.info(f"Starting {task_name}")
        
        try:
            # Create derived features processor
            derived_processor = Phase4DerivedFeatures(self.current_df)
            
            # Create derived features
            enhanced_df = derived_processor.create_derived_features()
            
            # Update current dataframe
            self.current_df = enhanced_df
            
            # Save intermediate result
            output_path = self.output_dir / "phase4_task2_1_derived_features.csv"
            self.current_df.to_csv(output_path)
            
            # Save documentation
            derived_processor.save_documentation(str(self.output_dir / "phase4_task2_1_documentation.json"))
            
            logger.info(f"✅ {task_name} completed successfully")
            self.execution_log['tasks_completed'].append({
                'task': task_name,
                'timestamp': datetime.now().isoformat(),
                'output_shape': self.current_df.shape,
                'features_added': derived_processor.get_summary()['features_created']
            })
            
            return True
            
        except Exception as e:
            logger.error(f"❌ {task_name} failed: {str(e)}")
            logger.error(traceback.format_exc())
            self.execution_log['tasks_failed'].append({
                'task': task_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False
            
    def execute_task2_2_interaction_features(self) -> bool:
        """
        Execute Task 2.2: Interaction Features.
        
        Returns:
            True if successful, False otherwise
        """
        task_name = "Task 2.2: Interaction Features"
        logger.info(f"Starting {task_name}")
        
        try:
            # Create interaction features processor
            interaction_processor = Phase4InteractionFeatures(self.current_df)
            
            # Create interaction features
            enhanced_df = interaction_processor.create_interaction_features()
            
            # Update current dataframe
            self.current_df = enhanced_df
            
            # Save intermediate result
            output_path = self.output_dir / "phase4_task2_2_interaction_features.csv"
            self.current_df.to_csv(output_path)
            
            # Save documentation
            interaction_processor.save_documentation(str(self.output_dir / "phase4_task2_2_documentation.json"))
            
            logger.info(f"✅ {task_name} completed successfully")
            self.execution_log['tasks_completed'].append({
                'task': task_name,
                'timestamp': datetime.now().isoformat(),
                'output_shape': self.current_df.shape,
                'features_added': interaction_processor.get_summary()['features_created']
            })
            
            return True
            
        except Exception as e:
            logger.error(f"❌ {task_name} failed: {str(e)}")
            logger.error(traceback.format_exc())
            self.execution_log['tasks_failed'].append({
                'task': task_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False
            
    def execute_task2_3_transformations(self) -> bool:
        """
        Execute Task 2.3: Distribution-based Transformations.
        
        Returns:
            True if successful, False otherwise
        """
        task_name = "Task 2.3: Distribution-based Transformations"
        logger.info(f"Starting {task_name}")
        
        try:
            # Create transformations processor
            transform_processor = Phase4Transformations(self.current_df)
            
            # Apply transformations
            transformed_df = transform_processor.apply_transformations()
            
            # Update current dataframe
            self.current_df = transformed_df
            
            # Save intermediate result
            output_path = self.output_dir / "phase4_task2_3_transformations.csv"
            self.current_df.to_csv(output_path)
            
            # Save documentation
            transform_processor.save_documentation(str(self.output_dir / "phase4_task2_3_documentation.json"))
            
            logger.info(f"✅ {task_name} completed successfully")
            self.execution_log['tasks_completed'].append({
                'task': task_name,
                'timestamp': datetime.now().isoformat(),
                'output_shape': self.current_df.shape,
                'transformations_applied': transform_processor.get_summary()['transformations_applied']
            })
            
            return True
            
        except Exception as e:
            logger.error(f"❌ {task_name} failed: {str(e)}")
            logger.error(traceback.format_exc())
            self.execution_log['tasks_failed'].append({
                'task': task_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False
            
    def execute_task3_1_advanced_preprocessing(self) -> bool:
        """
        Execute Task 3.1: Advanced Preprocessing.
        
        Returns:
            True if successful, False otherwise
        """
        task_name = "Task 3.1: Advanced Preprocessing"
        logger.info(f"Starting {task_name}")
        
        try:
            # Create advanced preprocessing processor
            preprocessing_processor = Phase4AdvancedPreprocessing(self.current_df)
            
            # Apply preprocessing
            preprocessed_df = preprocessing_processor.apply_preprocessing()
            
            # Update current dataframe
            self.current_df = preprocessed_df
            
            # Save intermediate result
            output_path = self.output_dir / "phase4_task3_1_preprocessed.csv"
            self.current_df.to_csv(output_path)
            
            # Save documentation
            preprocessing_processor.save_documentation(str(self.output_dir / "phase4_task3_1_documentation.json"))
            
            logger.info(f"✅ {task_name} completed successfully")
            self.execution_log['tasks_completed'].append({
                'task': task_name,
                'timestamp': datetime.now().isoformat(),
                'output_shape': self.current_df.shape,
                'preprocessing_applied': preprocessing_processor.get_summary()['preprocessing_steps']
            })
            
            return True
            
        except Exception as e:
            logger.error(f"❌ {task_name} failed: {str(e)}")
            logger.error(traceback.format_exc())
            self.execution_log['tasks_failed'].append({
                'task': task_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False
            
    def execute_task4_1_feature_selection(self) -> bool:
        """
        Execute Task 4.1: Feature Selection.
        
        Returns:
            True if successful, False otherwise
        """
        task_name = "Task 4.1: Feature Selection"
        logger.info(f"Starting {task_name}")
        
        try:
            # Create feature selection processor
            selection_processor = Phase4FeatureSelection(self.current_df)
            
            # Run feature selection pipeline
            selection_results = selection_processor.run_feature_selection_pipeline()
            
            # Create selected dataset
            selected_df = selection_processor.create_selected_dataset()
            
            # Update current dataframe
            self.current_df = selected_df
            
            # Save final selected features dataset
            output_path = self.output_dir / "selected_features_dataset.csv"
            self.current_df.to_csv(output_path)
            
            # Save selection results
            selection_processor.save_results(str(self.output_dir / "phase4_task4_1_selection_results.json"))
            
            logger.info(f"✅ {task_name} completed successfully")
            self.execution_log['tasks_completed'].append({
                'task': task_name,
                'timestamp': datetime.now().isoformat(),
                'output_shape': self.current_df.shape,
                'features_selected': len(selection_results['selected_features']),
                'selection_methods': list(selection_results['method_results'].keys())
            })
            
            return True
            
        except Exception as e:
            logger.error(f"❌ {task_name} failed: {str(e)}")
            logger.error(traceback.format_exc())
            self.execution_log['tasks_failed'].append({
                'task': task_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False
            
    def execute_task5_1_data_quality(self) -> bool:
        """
        Execute Task 5.1: Data Quality Targets.
        
        Returns:
            True if successful, False otherwise
        """
        task_name = "Task 5.1: Data Quality Targets"
        logger.info(f"Starting {task_name}")
        
        try:
            # Create data quality assessor
            quality_assessor = Phase4DataQuality(self.current_df)
            
            # Generate quality report
            quality_report = quality_assessor.generate_quality_report()
            
            # Check model readiness
            is_ready, blocking_issues = quality_assessor.is_model_ready()
            
            # Save quality report
            quality_assessor.save_quality_report(str(self.output_dir / "data_quality_report.json"))
            
            logger.info(f"✅ {task_name} completed successfully")
            self.execution_log['tasks_completed'].append({
                'task': task_name,
                'timestamp': datetime.now().isoformat(),
                'overall_compliance': quality_report['overall_compliance'],
                'compliance_rate': quality_report['compliance_rate'],
                'model_ready': is_ready,
                'blocking_issues': blocking_issues
            })
            
            return True
            
        except Exception as e:
            logger.error(f"❌ {task_name} failed: {str(e)}")
            logger.error(traceback.format_exc())
            self.execution_log['tasks_failed'].append({
                'task': task_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False
            
    def execute_task6_1_documentation(self) -> bool:
        """
        Execute Task 6.1: Documentation and Validation.
        
        Returns:
            True if successful, False otherwise
        """
        task_name = "Task 6.1: Documentation and Validation"
        logger.info(f"Starting {task_name}")
        
        try:
            # Create documentation generator
            doc_generator = Phase4Documentation(self.original_df, self.current_df)
            
            # Generate feature dictionary
            feature_dict = doc_generator.generate_feature_dictionary()
            
            # Generate validation report
            validation_report = doc_generator.generate_validation_report()
            
            # Save documentation
            doc_generator.save_documentation(str(self.output_dir))
            
            logger.info(f"✅ {task_name} completed successfully")
            self.execution_log['tasks_completed'].append({
                'task': task_name,
                'timestamp': datetime.now().isoformat(),
                'features_documented': len(feature_dict['features']),
                'validation_status': validation_report['overall_validation_status'],
                'transformations_documented': len(feature_dict['transformations_applied']['summary']['transformation_types_applied'])
            })
            
            return True
            
        except Exception as e:
            logger.error(f"❌ {task_name} failed: {str(e)}")
            logger.error(traceback.format_exc())
            self.execution_log['tasks_failed'].append({
                'task': task_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False
            
    def execute_all_tasks(self) -> bool:
        """
        Execute all Phase 4 tasks in sequence.
        
        Returns:
            True if all tasks completed successfully, False otherwise
        """
        logger.info("🚀 Starting Phase 4 Feature Engineering - Complete Execution")
        logger.info(f"Input: {self.input_data_path}")
        logger.info(f"Output Directory: {self.output_dir}")
        
        # Load initial data
        if not self.load_initial_data():
            logger.error("Failed to load initial data. Aborting execution.")
            return False
            
        # Define task execution sequence
        tasks = [
            ('Task 1: Load and Validate', self.execute_task1_load_validate),
            ('Task 2.1: Derived Features', self.execute_task2_1_derived_features),
            ('Task 2.2: Interaction Features', self.execute_task2_2_interaction_features),
            ('Task 2.3: Transformations', self.execute_task2_3_transformations),
            ('Task 3.1: Advanced Preprocessing', self.execute_task3_1_advanced_preprocessing),
            ('Task 4.1: Feature Selection', self.execute_task4_1_feature_selection),
            ('Task 5.1: Data Quality', self.execute_task5_1_data_quality),
            ('Task 6.1: Documentation', self.execute_task6_1_documentation)
        ]
        
        # Execute tasks sequentially
        all_successful = True
        for task_name, task_function in tasks:
            logger.info(f"\n{'='*60}")
            logger.info(f"Executing: {task_name}")
            logger.info(f"Current data shape: {self.current_df.shape if self.current_df is not None else 'N/A'}")
            
            success = task_function()
            
            if not success:
                logger.error(f"Task failed: {task_name}")
                all_successful = False
                
                # Ask user if they want to continue
                try:
                    user_input = input(f"\nTask '{task_name}' failed. Continue with remaining tasks? (y/n): ")
                    if user_input.lower() != 'y':
                        logger.info("User chose to stop execution.")
                        break
                except:
                    # If running in non-interactive mode, continue
                    logger.warning("Non-interactive mode detected. Continuing with remaining tasks.")
                    
            else:
                logger.info(f"✅ Task completed: {task_name}")
                if self.current_df is not None:
                    logger.info(f"Current data shape: {self.current_df.shape}")
                    
        # Finalize execution log
        self.execution_log['end_time'] = datetime.now().isoformat()
        self.execution_log['final_status'] = 'success' if all_successful else 'partial_success'
        self.execution_log['final_data_shape'] = self.current_df.shape if self.current_df is not None else None
        
        # Save execution log
        log_path = self.output_dir / "phase4_execution_log.json"
        with open(log_path, 'w') as f:
            json.dump(self.execution_log, f, indent=2, default=str)
            
        # Print final summary
        self.print_execution_summary()
        
        return all_successful
        
    def print_execution_summary(self) -> None:
        """
        Print execution summary.
        """
        print(f"\n{'='*80}")
        print("🎯 PHASE 4 FEATURE ENGINEERING - EXECUTION SUMMARY")
        print(f"{'='*80}")
        
        print(f"\n📊 OVERALL STATUS: {self.execution_log['final_status'].upper()}")
        print(f"⏱️  Start Time: {self.execution_log['start_time']}")
        print(f"⏱️  End Time: {self.execution_log.get('end_time', 'N/A')}")
        
        if self.original_df is not None and self.current_df is not None:
            print(f"\n📈 DATA TRANSFORMATION:")
            print(f"   Original Shape: {self.original_df.shape}")
            print(f"   Final Shape: {self.current_df.shape}")
            print(f"   Features Added: {self.current_df.shape[1] - self.original_df.shape[1]}")
            
        print(f"\n✅ COMPLETED TASKS ({len(self.execution_log['tasks_completed'])}):")
        for task in self.execution_log['tasks_completed']:
            print(f"   • {task['task']}")
            
        if self.execution_log['tasks_failed']:
            print(f"\n❌ FAILED TASKS ({len(self.execution_log['tasks_failed'])}):")
            for task in self.execution_log['tasks_failed']:
                print(f"   • {task['task']}: {task['error'][:100]}...")
                
        print(f"\n📁 OUTPUT DIRECTORY: {self.output_dir}")
        print(f"📄 Execution Log: {self.output_dir / 'phase4_execution_log.json'}")
        
        if self.current_df is not None:
            print(f"🎯 Final Dataset: {self.output_dir / 'selected_features_dataset.csv'}")
            
        print(f"\n{'='*80}")


def main():
    """
    Main function to execute all Phase 4 tasks.
    """
    try:
        # Create executor
        executor = Phase4Executor()
        
        # Execute all tasks
        success = executor.execute_all_tasks()
        
        if success:
            logger.info("🎉 Phase 4 Feature Engineering completed successfully!")
            return 0
        else:
            logger.warning("⚠️ Phase 4 Feature Engineering completed with some failures.")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\n⏹️ Execution interrupted by user.")
        return 130
    except Exception as e:
        logger.error(f"💥 Phase 4 execution failed with unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)